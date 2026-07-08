from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


from screensift.common.io import ensure_dir, load_yaml, write_json
from screensift.common.logging_utils import setup_logger


def get_target_config(config: dict[str, Any], target: str) -> dict[str, Any]:
    if "targets" in config:
        targets = config["targets"]
        if target not in targets:
            raise KeyError(f"Target {target!r} not found in targets config. Available: {sorted(targets)}")
        return targets[target]
    if target in config:
        return config[target]
    if config.get("target_id") == target:
        return config
    raise KeyError(f"Target {target!r} not found in targets config.")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def archive_url(target_config: dict[str, Any], archive_type: str) -> str | None:
    key = {
        "full": "lit_pcba_full_data_url",
        "ave": "lit_pcba_ave_unbiased_url",
    }[archive_type]
    value = target_config.get(key)
    return str(value) if value else None


def project_dirs(paths: dict[str, Any], target_config: dict[str, Any], target: str) -> tuple[Path, Path, Path, Path]:
    raw_lit_pcba_dir = paths.get("raw_lit_pcba_dir")
    results_report_dir = paths.get("results_report_dir")
    if not raw_lit_pcba_dir:
        raise KeyError("paths.yml must define raw_lit_pcba_dir")
    if not results_report_dir:
        raise KeyError("paths.yml must define results_report_dir")

    target_id = target_config.get("target_id", target)
    raw_root = Path(raw_lit_pcba_dir)
    downloads_dir = raw_root / "downloads"
    extracted_dir = raw_root / "extracted"
    target_raw_dir = raw_root / target_id
    report_dir = Path(results_report_dir)
    return downloads_dir, extracted_dir, target_raw_dir, report_dir


def manual_instruction(target: str, target_raw_dir: Path) -> str:
    return (
        f"Manual fallback: download the official LIT-PCBA archive, extract the {target} files, "
        f"and place active/inactive ligand files under {target_raw_dir}."
    )


def write_manifest(
    report_dir: Path,
    target: str,
    archive_type: str,
    url: str | None,
    archive_path: Path | None,
    extracted_dir: Path,
    target_raw_dir: Path,
    found_candidates: list[Path],
    copied_files: list[Path],
    status: str,
    message: str,
) -> None:
    manifest_path = report_dir / f"{target.lower()}_download_manifest.json"
    write_json(
        {
            "target": target,
            "archive_type": archive_type,
            "archive_url": url,
            "archive_path": str(archive_path) if archive_path else None,
            "extracted_dir": str(extracted_dir),
            "target_raw_dir": str(target_raw_dir),
            "found_candidates": [str(path) for path in found_candidates],
            "copied_files": [str(path) for path in copied_files],
            "status": status,
            "message": message,
        },
        manifest_path,
    )


def local_target_files(target_raw_dir: Path) -> list[Path]:
    if not target_raw_dir.exists():
        return []
    return [path for path in sorted(target_raw_dir.rglob("*")) if path.is_file()]


def archive_filename(url: str) -> str:
    return Path(urlparse(url).path).name or "lit_pcba_archive.tgz"


def download_archive(url: str, output_path: Path, force: bool) -> None:
    if output_path.exists() and not force:
        return

    ensure_dir(output_path.parent)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp_path.replace(output_path)


def safe_extract_tar(archive_path: Path, extracted_dir: Path, force: bool) -> None:
    archive_extract_dir = extracted_dir / archive_path.stem.replace(".tar", "")
    if archive_extract_dir.exists() and not force:
        return
    if archive_extract_dir.exists() and force:
        shutil.rmtree(archive_extract_dir)
    ensure_dir(archive_extract_dir)

    with tarfile.open(archive_path, "r:*") as tar:
        root = archive_extract_dir.resolve()
        for member in tar.getmembers():
            member_path = (archive_extract_dir / member.name).resolve()
            try:
                member_path.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"Unsafe path in archive member: {member.name}")
        tar.extractall(archive_extract_dir)


def extracted_archive_dir(extracted_dir: Path, archive_path: Path) -> Path:
    return extracted_dir / archive_path.stem.replace(".tar", "")


def discover_candidates(search_dir: Path, target: str) -> list[Path]:
    target_lower = target.lower()
    if not search_dir.exists():
        return []
    return [path for path in sorted(search_dir.rglob("*")) if target_lower in path.as_posix().lower()]


def candidate_source_files(candidates: list[Path]) -> list[Path]:
    files: dict[Path, Path] = {}
    for candidate in candidates:
        if candidate.is_file():
            files[candidate.resolve()] = candidate
        elif candidate.is_dir():
            for path in sorted(candidate.rglob("*")):
                if path.is_file():
                    files[path.resolve()] = path
    return list(files.values())


def copy_candidate_files(source_files: list[Path], candidates: list[Path], target_raw_dir: Path, force: bool) -> list[Path]:
    copied: list[Path] = []
    candidate_dirs = [candidate for candidate in candidates if candidate.is_dir()]
    ensure_dir(target_raw_dir)

    for source_file in source_files:
        base_dir = next((directory for directory in candidate_dirs if source_file.is_relative_to(directory)), None)
        if base_dir is not None:
            relative_path = source_file.relative_to(base_dir)
        else:
            relative_path = Path(source_file.name)

        destination = target_raw_dir / relative_path
        if destination.exists() and not force:
            continue
        ensure_dir(destination.parent)
        shutil.copy2(source_file, destination)
        copied.append(destination)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and stage MAPK1 LIT-PCBA raw data.")
    parser.add_argument("--target-config", "--config", default="configs/targets.yml", help="Target configuration YAML.")
    parser.add_argument("--paths", default="configs/paths.yml", help="Project path configuration YAML.")
    parser.add_argument("--target", default="MAPK1", help="Target identifier.")
    parser.add_argument("--archive", choices=["full", "ave"], default="full", help="LIT-PCBA archive type to download.")
    parser.add_argument("--force", type=parse_bool, default=False, help="Overwrite existing downloads/extractions/files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger("download_lit_pcba")

    target_config = get_target_config(load_yaml(args.target_config), args.target)
    paths = load_yaml(args.paths)
    downloads_dir, extracted_dir, target_raw_dir, report_dir = project_dirs(paths, target_config, args.target)
    ensure_dir(downloads_dir)
    ensure_dir(extracted_dir)
    ensure_dir(target_raw_dir)
    ensure_dir(report_dir)

    existing_files = local_target_files(target_raw_dir)
    if existing_files and not args.force:
        message = f"Found existing files under {target_raw_dir}; leaving them untouched."
        logger.info(message)
        write_manifest(
            report_dir,
            args.target,
            args.archive,
            archive_url(target_config, args.archive),
            None,
            extracted_dir,
            target_raw_dir,
            existing_files,
            [],
            "manual_or_existing_data_present",
            message,
        )
        return 0

    url = archive_url(target_config, args.archive)
    if not url:
        message = f"No configured LIT-PCBA {args.archive!r} archive URL for {args.target}. {manual_instruction(args.target, target_raw_dir)}"
        logger.warning(message)
        print(message)
        write_manifest(report_dir, args.target, args.archive, url, None, extracted_dir, target_raw_dir, [], [], "missing_url", message)
        return 0

    archive_path = downloads_dir / archive_filename(url)
    found_candidates: list[Path] = []
    copied_files: list[Path] = []

    try:
        logger.info("Downloading or reusing %s", archive_path)
        download_archive(url, archive_path, args.force)
        logger.info("Extracting or reusing %s", archive_path)
        safe_extract_tar(archive_path, extracted_dir, args.force)
        search_dir = extracted_archive_dir(extracted_dir, archive_path)

        found_candidates = discover_candidates(search_dir, args.target)
        for candidate in found_candidates:
            logger.info("Discovered MAPK1 archive candidate: %s", candidate)

        if not found_candidates:
            message = f"No {args.target} paths were found inside the extracted archive. {manual_instruction(args.target, target_raw_dir)}"
            logger.warning(message)
            print(message)
            write_manifest(
                report_dir,
                args.target,
                args.archive,
                url,
                archive_path,
                search_dir,
                target_raw_dir,
                found_candidates,
                copied_files,
                "target_not_found_in_archive",
                message,
            )
            return 0

        source_files = candidate_source_files(found_candidates)
        copied_files = copy_candidate_files(source_files, found_candidates, target_raw_dir, args.force)
        message = f"Staged {len(copied_files)} files for {args.target} under {target_raw_dir}."
        logger.info(message)
        write_manifest(
            report_dir,
            args.target,
            args.archive,
            url,
            archive_path,
            search_dir,
            target_raw_dir,
            found_candidates,
            copied_files,
            "complete",
            message,
        )
        return 0
    except Exception as exc:
        message = f"LIT-PCBA download/extraction failed: {exc}. {manual_instruction(args.target, target_raw_dir)}"
        logger.warning(message)
        print(message)
        write_manifest(
            report_dir,
            args.target,
            args.archive,
            url,
            archive_path,
            extracted_dir,
            target_raw_dir,
            found_candidates,
            copied_files,
            "download_or_extraction_failed",
            message,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
