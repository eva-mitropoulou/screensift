from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import requests


from screensift.common.io import ensure_dir, load_yaml, write_json
from screensift.common.logging_utils import setup_logger


RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD_URL_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.cif"


def get_receptor_config(config: dict[str, Any], target: str) -> dict[str, Any]:
    if "receptors" in config:
        receptors = config["receptors"]
        if target not in receptors:
            raise KeyError(f"Target {target!r} not found in receptor config. Available: {sorted(receptors)}")
        return receptors[target]
    if "targets" in config:
        targets = config["targets"]
        if target not in targets:
            raise KeyError(f"Target {target!r} not found in receptor config. Available: {sorted(targets)}")
        return targets[target]
    if config.get("target_id") == target:
        return config
    if target in config:
        return config[target]
    raise KeyError(f"Target {target!r} not found in receptor config.")


def target_raw_pdb_dir(paths: dict[str, Any], target: str) -> Path:
    raw_pdb_dir = paths.get("raw_pdb_dir")
    if not raw_pdb_dir:
        raise KeyError("paths.yml must define raw_pdb_dir")
    return Path(raw_pdb_dir) / target


def report_dir(paths: dict[str, Any]) -> Path:
    results_report_dir = paths.get("results_report_dir")
    if not results_report_dir:
        raise KeyError("paths.yml must define results_report_dir")
    return Path(results_report_dir)


def discover_local_structure_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    allowed = {".cif", ".mmcif", ".pdb", ".ent"}
    return [path for path in sorted(raw_dir.rglob("*")) if path.is_file() and path.suffix.lower() in allowed]


def normalize_pdb_id(value: str) -> str:
    cleaned = "".join(ch for ch in str(value).strip().upper() if ch.isalnum())
    if len(cleaned) != 4:
        raise ValueError(f"Invalid PDB ID {value!r}; expected four alphanumeric characters.")
    return cleaned


def rcsb_query_payload(receptor_config: dict[str, Any]) -> dict[str, Any]:
    criteria = receptor_config.get("selection_criteria", {})
    max_resolution = float(criteria.get("max_resolution_angstrom", 3.0))
    max_receptors = int(receptor_config.get("max_receptors", 10))
    uniprot_accession = receptor_config.get("uniprot_accession")
    aliases = receptor_config.get("aliases") or [receptor_config.get("target_id"), receptor_config.get("target_name")]
    organism = criteria.get("organism", "Homo sapiens")

    nodes: list[dict[str, Any]] = []
    if uniprot_accession:
        nodes.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                    "operator": "exact_match",
                    "value": str(uniprot_accession),
                },
            }
        )
    else:
        query_text = " ".join(str(value) for value in aliases if value)
        nodes.append({"type": "terminal", "service": "full_text", "parameters": {"value": query_text}})

    if organism:
        nodes.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entity_source_organism.scientific_name",
                    "operator": "exact_match",
                    "value": str(organism),
                },
            }
        )

    nodes.extend(
        [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "exptl.method",
                    "operator": "in",
                    "value": ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"],
                },
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "less_or_equal",
                    "value": max_resolution,
                },
            },
        ]
    )

    return {
        "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max(max_receptors * 4, 20)},
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
            "scoring_strategy": "combined",
        },
    }


def query_rcsb_for_pdb_ids(receptor_config: dict[str, Any]) -> list[str]:
    response = requests.post(RCSB_SEARCH_URL, json=rcsb_query_payload(receptor_config), timeout=60)
    response.raise_for_status()
    payload = response.json()
    identifiers = [result.get("identifier") for result in payload.get("result_set", [])]
    pdb_ids: list[str] = []
    for identifier in identifiers:
        if not identifier:
            continue
        try:
            pdb_ids.append(normalize_pdb_id(str(identifier)))
        except ValueError:
            continue
    return list(dict.fromkeys(pdb_ids))


def download_mmcif(pdb_id: str, raw_dir: Path, force: bool = False) -> Path:
    ensure_dir(raw_dir)
    out_path = raw_dir / f"{pdb_id.lower()}.cif"
    if out_path.exists() and not force:
        return out_path

    url = RCSB_DOWNLOAD_URL_TEMPLATE.format(pdb_id=pdb_id.upper())
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    out_path.write_bytes(response.content)
    return out_path


def fetch_pdbs(
    receptor_config_path: str | Path,
    paths_path: str | Path,
    target: str,
    out_report: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    logger = setup_logger("fetch_pdbs")
    receptor_config = get_receptor_config(load_yaml(receptor_config_path), target)
    paths = load_yaml(paths_path)
    raw_dir = target_raw_pdb_dir(paths, target)
    report_path = Path(out_report)
    ensure_dir(raw_dir)
    ensure_dir(report_path.parent)

    warnings: list[str] = []
    downloaded_files: list[str] = []
    discovered_before = discover_local_structure_files(raw_dir)
    for local_file in discovered_before:
        logger.info("Discovered local receptor file: %s", local_file)

    manual_ids = [normalize_pdb_id(value) for value in receptor_config.get("manual_pdb_ids", [])]
    query_mode = "manual_pdb_ids" if manual_ids else "rcsb_search"
    max_receptors = int(receptor_config.get("max_receptors", 10))
    candidate_ids: list[str] = manual_ids

    if not candidate_ids:
        try:
            candidate_ids = query_rcsb_for_pdb_ids(receptor_config)
            logger.info("RCSB query discovered %d candidate entries.", len(candidate_ids))
        except Exception as exc:
            warnings.append(f"RCSB search failed: {exc}")
            candidate_ids = []

    for pdb_id in candidate_ids[:max_receptors]:
        try:
            path = download_mmcif(pdb_id, raw_dir, force=force)
            downloaded_files.append(str(path))
            logger.info("Downloaded or reused receptor file: %s", path)
        except Exception as exc:
            warning = f"Failed to download {pdb_id}: {exc}"
            warnings.append(warning)
            logger.warning(warning)

    discovered_after = discover_local_structure_files(raw_dir)
    status = "complete" if discovered_after else "missing_receptor_files"
    if not discovered_after:
        warnings.append(
            f"No receptor structures found under {raw_dir}. Add PDB/mmCIF files there manually or set manual_pdb_ids in configs/receptors.yml."
        )

    manifest = {
        "target": target,
        "query_mode": query_mode,
        "downloaded_files": downloaded_files,
        "discovered_local_files": [str(path) for path in discovered_after],
        "warnings": warnings,
        "status": status,
    }
    write_json(manifest, report_path)
    return manifest


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch MAPK1 receptor structures from RCSB PDB.")
    parser.add_argument("--receptor-config", default="configs/receptors.yml", help="Receptor configuration YAML.")
    parser.add_argument("--paths", default="configs/paths.yml", help="Project path configuration YAML.")
    parser.add_argument("--target", default="MAPK1", help="Target identifier.")
    parser.add_argument("--out-report", default="results/reports/mapk1_pdb_fetch_manifest.json", help="Fetch manifest JSON.")
    parser.add_argument("--force", type=parse_bool, default=False, help="Overwrite existing downloaded receptor files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fetch_pdbs(args.receptor_config, args.paths, args.target, args.out_report, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
