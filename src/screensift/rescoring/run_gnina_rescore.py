from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd


from screensift.common.io import ensure_dir
from screensift.common.external import tool_command


RAW_COLUMNS = [
    "ligand_id",
    "activity_label",
    "pdb_id",
    "best_score_unidock",
    "gnina_receptor_input",
    "gnina_ligand_input",
    "converted_ligand_input",
    "gnina_stdout_log",
    "gnina_stderr_log",
    "status",
    "error_message",
    "raw_stdout_tail",
]


def safe_id(value: Any, default: str = "item") -> str:
    text = str(value if value is not None and not pd.isna(value) else default).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or default


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def resolve_gnina_bin(value: str | None = None) -> str:
    candidate = value or shutil.which("gnina")
    if not candidate:
        raise FileNotFoundError("GNINA binary/wrapper not found. Source env/activate_gnina.sh or set --gnina-bin.")
    return candidate


def element_from_atom_name(atom_name: str) -> str:
    letters = "".join(char for char in atom_name.strip() if char.isalpha())
    if not letters:
        return "C"
    letters = letters.capitalize()
    if len(letters) >= 2 and letters[:2] in {"Cl", "Br"}:
        return letters[:2]
    return letters[0]


def pdbqt_to_pdb_first_model(pdbqt_path: str | Path, out_path: str | Path) -> Path:
    source = Path(pdbqt_path)
    destination = Path(out_path)
    if not source.exists():
        raise FileNotFoundError(f"Ligand PDBQT not found for conversion: {source}")
    ensure_dir(destination.parent)

    lines: list[str] = []
    atom_count = 0
    for raw_line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        record = raw_line[:6].strip()
        if record in {"ATOM", "HETATM"}:
            atom_count += 1
            serial_text = raw_line[6:11].strip()
            try:
                serial = int(serial_text)
            except ValueError:
                serial = atom_count
            atom_name = raw_line[12:16].strip() or f"C{atom_count}"
            try:
                x = float(raw_line[30:38])
                y = float(raw_line[38:46])
                z = float(raw_line[46:54])
            except ValueError as exc:
                raise ValueError(f"Could not parse coordinates in {source}: {raw_line}") from exc
            element = element_from_atom_name(atom_name)
            lines.append(
                f"HETATM{serial:5d} {atom_name:<4s} LIG A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
            )
        elif raw_line.startswith("ENDMDL") and atom_count:
            break

    if atom_count == 0:
        raise ValueError(f"No ATOM/HETATM records found in ligand PDBQT: {source}")
    lines.append("END\n")
    destination.write_text("".join(lines), encoding="utf-8")
    return destination


def prepare_ligand_for_gnina(row: pd.Series, converted_root: Path) -> tuple[str, str]:
    ligand = Path(str(row["gnina_ligand_input"]))
    if not ligand.exists():
        raise FileNotFoundError(f"GNINA ligand input not found: {ligand}")
    if ligand.suffix.lower() == ".pdbqt":
        pdb_id = safe_id(row.get("pdb_id")).lower()
        ligand_id = safe_id(row.get("ligand_id"))
        converted = converted_root / pdb_id / f"{ligand_id}_gnina_input.pdb"
        if converted.exists() and converted.stat().st_size > 0 and converted.stat().st_mtime >= ligand.stat().st_mtime:
            return str(converted), "pdbqt_converted_to_pdb_reused"
        pdbqt_to_pdb_first_model(ligand, converted)
        return str(converted), "pdbqt_converted_to_pdb"
    return str(ligand), "native"


def tail_text(text: str, max_lines: int = 40) -> str:
    return "\n".join(text.splitlines()[-max_lines:])


def run_one_gnina_score(
    row: pd.Series,
    gnina_bin: str,
    log_dir: Path,
    converted_ligand_root: Path,
    score_only: bool,
    timeout_seconds: int,
    gnina_mode: str = "native",
    gnina_image: str | None = None,
    gpu: str | bool | None = None,
    docker_args: list[str] | str | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    ligand_id = safe_id(row.get("ligand_id"))
    pdb_id = safe_id(row.get("pdb_id"))
    run_id = safe_id(f"{pdb_id}_{ligand_id}")
    stdout_log = log_dir / f"{run_id}.stdout.log"
    stderr_log = log_dir / f"{run_id}.stderr.log"
    ensure_dir(stdout_log.parent)

    receptor = Path(str(row["gnina_receptor_input"]))
    source_ligand = str(row["gnina_ligand_input"])
    if not receptor.exists():
        error = f"GNINA receptor input not found: {receptor}"
        stdout_log.write_text("", encoding="utf-8")
        stderr_log.write_text(error, encoding="utf-8")
        return raw_row(row, source_ligand, "", stdout_log, stderr_log, "failed", error, "")

    try:
        ligand, conversion_status = prepare_ligand_for_gnina(row, converted_ligand_root)
        gnina_args: list[str] = []
        if score_only:
            gnina_args.append("--score_only")
        gnina_args.extend(["-r", str(receptor), "-l", ligand])
        gnina_args.extend(extra_args or [])
        command = tool_command(
            gnina_bin,
            gnina_args,
            mode=gnina_mode,
            image=gnina_image,
            gpu=gpu,
            docker_args=docker_args,
        )
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_seconds)
        stdout_log.write_text(completed.stdout, encoding="utf-8")
        stderr_log.write_text(completed.stderr, encoding="utf-8")
        status = "complete" if completed.returncode == 0 else "failed"
        error_message = "" if completed.returncode == 0 else f"returncode={completed.returncode}; stderr={completed.stderr.strip()[:1000]}"
        return raw_row(
            row,
            source_ligand,
            ligand,
            stdout_log,
            stderr_log,
            status,
            error_message,
            tail_text(completed.stdout),
        )
    except Exception as exc:
        stdout_log.write_text("", encoding="utf-8")
        stderr_log.write_text(str(exc), encoding="utf-8")
        return raw_row(row, source_ligand, source_ligand, stdout_log, stderr_log, "failed", str(exc), "")


def raw_row(
    row: pd.Series,
    gnina_ligand_input: str,
    converted_ligand_input: str,
    stdout_log: Path,
    stderr_log: Path,
    status: str,
    error_message: str,
    raw_stdout_tail: str,
) -> dict[str, Any]:
    return {
        "ligand_id": row.get("ligand_id"),
        "activity_label": row.get("activity_label"),
        "pdb_id": row.get("pdb_id"),
        "best_score_unidock": row.get("best_score_unidock", row.get("best_score")),
        "gnina_receptor_input": row.get("gnina_receptor_input"),
        "gnina_ligand_input": gnina_ligand_input,
        "converted_ligand_input": converted_ligand_input,
        "gnina_stdout_log": str(stdout_log),
        "gnina_stderr_log": str(stderr_log),
        "status": status,
        "error_message": error_message,
        "raw_stdout_tail": raw_stdout_tail,
    }


def run_key(row: pd.Series | dict[str, Any]) -> str:
    return f"{safe_id(row.get('pdb_id'))}::{safe_id(row.get('ligand_id'))}"


def success_status(value: Any) -> bool:
    return str(value).strip().lower() in {"complete", "success", "succeeded", "ok"}


def load_completed_rows(raw_path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return [], set()
    existing = pd.read_csv(raw_path, dtype={"ligand_id": str})
    for column in RAW_COLUMNS:
        if column not in existing.columns:
            existing[column] = pd.NA
    completed = existing[existing["status"].map(success_status)].copy()
    completed_rows = completed[RAW_COLUMNS].to_dict(orient="records")
    completed_keys = {run_key(row) for row in completed_rows}
    return completed_rows, completed_keys


def write_raw_checkpoint(rows: list[dict[str, Any]], out_path: str | Path) -> None:
    out = Path(out_path)
    ensure_dir(out.parent)
    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    tmp = out.with_name(f"{out.name}.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(out)


def run_gnina_rescore(
    subset_path: str | Path,
    out_path: str | Path,
    limit: int | None = None,
    score_only: bool = True,
    gnina_bin: str | None = None,
    log_dir: str | Path = "results/reports/gnina_logs",
    converted_ligand_root: str | Path = "results/poses/gnina/MAPK1/phase1_inputs",
    timeout_seconds: int = 1800,
    resume: bool = False,
    checkpoint_every: int = 50,
    max_workers: int = 1,
    gnina_mode: str = "native",
    gnina_image: str | None = None,
    gpu: str | bool | None = None,
    docker_args: list[str] | str | None = None,
    extra_args: list[str] | None = None,
) -> pd.DataFrame:
    max_workers = max(1, int(max_workers))
    subset = pd.read_csv(subset_path, dtype={"ligand_id": str})
    required = ["gnina_receptor_input", "gnina_ligand_input", "valid_for_gnina"]
    missing = [column for column in required if column not in subset.columns]
    if missing:
        raise ValueError(f"GNINA subset table is missing required columns: {missing}")

    valid = subset[subset["valid_for_gnina"].map(parse_bool)].copy()
    if limit is not None:
        valid = valid.head(limit)

    resolved_bin = gnina_bin or "gnina" if gnina_mode == "docker" else resolve_gnina_bin(gnina_bin)
    log_path = ensure_dir(log_dir)
    converted_root = ensure_dir(converted_ligand_root)
    out = Path(out_path)
    rows: list[dict[str, Any]]
    completed_keys: set[str]
    if resume:
        rows, completed_keys = load_completed_rows(out)
    else:
        rows, completed_keys = [], set()

    pending = valid[~valid.apply(lambda row: run_key(row) in completed_keys, axis=1)].copy()
    total = len(valid)
    skipped = total - len(pending)
    if skipped:
        print(f"Resuming GNINA rescoring: skipping {skipped} completed rows; pending={len(pending)}", flush=True)
    elif resume:
        print(f"Resuming GNINA rescoring: no completed rows found; pending={len(pending)}", flush=True)

    def score_row(row: pd.Series) -> dict[str, Any]:
        return run_one_gnina_score(
            row,
            resolved_bin,
            log_path,
            converted_root,
            score_only,
            timeout_seconds,
            gnina_mode=gnina_mode,
            gnina_image=gnina_image,
            gpu=gpu,
            docker_args=docker_args,
            extra_args=extra_args,
        )

    def handle_result(result: dict[str, Any], completed_count: int) -> None:
        nonlocal processed_since_checkpoint
        rows.append(result)
        processed_since_checkpoint += 1
        if processed_since_checkpoint >= checkpoint_every:
            write_raw_checkpoint(rows, out)
            successful = sum(1 for item in rows if success_status(item.get("status")))
            print(
                f"GNINA checkpoint: processed={completed_count}/{total} "
                f"successful={successful} failed={len(rows) - successful}",
                flush=True,
            )
            processed_since_checkpoint = 0

    checkpoint_every = max(1, int(checkpoint_every))
    processed_since_checkpoint = 0
    completed_count = skipped
    if max_workers == 1:
        for _index, row in pending.iterrows():
            completed_count += 1
            handle_result(score_row(row), completed_count)
    else:
        print(f"Running GNINA rescoring with max_workers={max_workers}", flush=True)
        executor = ThreadPoolExecutor(max_workers=max_workers)
        interrupted = False
        futures = []
        try:
            futures = [executor.submit(score_row, row) for _index, row in pending.iterrows()]
            for future in as_completed(futures):
                completed_count += 1
                handle_result(future.result(), completed_count)
        except KeyboardInterrupt:
            interrupted = True
            for future in futures:
                future.cancel()
            write_raw_checkpoint(rows, out)
            raise
        finally:
            executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    write_raw_checkpoint(rows, out)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GNINA score-only rescoring on a selected subset.")
    parser.add_argument("--subset", default="results/tables/mapk1_phase1_gnina_subset.csv")
    parser.add_argument("--out", default="results/tables/mapk1_phase1_gnina_smoke_raw.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--gnina-bin", default=None)
    parser.add_argument("--gnina-mode", choices=["native", "docker"], default="native")
    parser.add_argument("--gnina-image", default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--docker-arg", action="append", default=[])
    parser.add_argument("--gnina-extra-arg", action="append", default=[])
    parser.add_argument("--log-dir", default="results/reports/gnina_logs")
    parser.add_argument("--converted-ligand-root", default="results/poses/gnina/MAPK1/phase1_inputs")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--max-workers", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = run_gnina_rescore(
        args.subset,
        args.out,
        limit=args.limit,
        score_only=args.score_only,
        gnina_bin=args.gnina_bin,
        log_dir=args.log_dir,
        converted_ligand_root=args.converted_ligand_root,
        timeout_seconds=args.timeout_seconds,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        max_workers=args.max_workers,
        gnina_mode=args.gnina_mode,
        gnina_image=args.gnina_image,
        gpu=args.gpu,
        docker_args=args.docker_arg,
        extra_args=args.gnina_extra_arg,
    )
    complete = int((frame["status"] == "complete").sum()) if not frame.empty else 0
    print(f"GNINA rescoring complete: attempts={len(frame)} successful={complete} failed={len(frame) - complete}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
