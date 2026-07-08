from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem


from screensift.common.io import ensure_dir, write_json
from screensift.common.logging_utils import setup_logger


FAILURE_COLUMNS = ["sdf_path", "pdbqt_path", "failure_reason"]


@dataclass
class PdbqtResult:
    sdf_path: str
    pdbqt_path: str
    success: bool
    failure_reason: str | None
    skipped_existing: bool = False


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def discover_sdf_files(sdf_dir: str | Path, limit: int | None = None) -> list[Path]:
    root = Path(sdf_dir)
    if not root.exists():
        return []
    files = sorted(path for path in root.rglob("*.sdf") if path.is_file())
    return files[:limit] if limit is not None else files


def output_path_for_sdf(sdf_path: Path, sdf_dir: Path, out_dir: Path) -> Path:
    relative = sdf_path.relative_to(sdf_dir)
    return (out_dir / relative).with_suffix(".pdbqt")


def ligand_prep_command() -> list[str] | None:
    for name in ("mk_prepare_ligand.py", "mk_prepare_ligand"):
        found = shutil.which(name)
        if found:
            return [found]
    return None


def convert_with_cli(command: list[str], sdf_path: Path, out_path: Path) -> tuple[bool, str]:
    completed = subprocess.run(
        [*command, "-i", str(sdf_path), "-o", str(out_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
        return True, " ".join([*command, "-i", str(sdf_path), "-o", str(out_path)])
    return False, f"rc={completed.returncode}; stderr={completed.stderr.strip()[:500]}"


def convert_with_meeko_api(sdf_path: Path, out_path: Path) -> tuple[bool, str]:
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
    except Exception as exc:
        return False, f"Meeko API import failed: {exc}"

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next((candidate for candidate in supplier if candidate is not None), None)
    if mol is None:
        return False, "RDKit could not read SDF molecule."

    try:
        preparator = MoleculePreparation()
        setups = preparator.prepare(mol)
        if not setups:
            return False, "Meeko did not return a molecule setup."
        written = PDBQTWriterLegacy.write_string(setups[0])
        if isinstance(written, tuple):
            pdbqt_text = written[0]
            is_ok = bool(written[1]) if len(written) > 1 else True
            error_msg = str(written[2]) if len(written) > 2 else ""
            if not is_ok:
                return False, error_msg or "Meeko PDBQT writer returned failure."
        else:
            pdbqt_text = str(written)
        out_path.write_text(pdbqt_text, encoding="utf-8")
        return True, "meeko_api"
    except Exception as exc:
        return False, f"Meeko API conversion failed: {exc}"


def convert_one_sdf(sdf_path: Path, sdf_dir: Path, out_dir: Path, force: bool, command: list[str] | None) -> PdbqtResult:
    out_path = output_path_for_sdf(sdf_path, sdf_dir, out_dir)
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        return PdbqtResult(str(sdf_path), str(out_path), True, None, skipped_existing=True)

    ensure_dir(out_path.parent)
    if command:
        ok, message = convert_with_cli(command, sdf_path, out_path)
        if ok:
            return PdbqtResult(str(sdf_path), str(out_path), True, None)

    ok, message = convert_with_meeko_api(sdf_path, out_path)
    if ok:
        return PdbqtResult(str(sdf_path), str(out_path), True, None)
    return PdbqtResult(str(sdf_path), str(out_path), False, message)


def prepare_pdbqt_ligands(
    sdf_dir: str | Path,
    out_dir: str | Path,
    report_path: str | Path,
    failures_path: str | Path,
    n_jobs: int = 16,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    logger = setup_logger("prepare_pdbqt_ligands")
    sdf_dir = Path(sdf_dir)
    out_dir = ensure_dir(out_dir)
    report_path = Path(report_path)
    failures_path = Path(failures_path)
    ensure_dir(report_path.parent)
    ensure_dir(failures_path.parent)

    sdf_files = discover_sdf_files(sdf_dir, limit=limit)
    command = ligand_prep_command()
    command_used = " ".join(command) if command else "meeko_api"

    if not sdf_files:
        pd.DataFrame(columns=FAILURE_COLUMNS).to_csv(failures_path, index=False)
        manifest = {
            "status": "empty_input",
            "total_input_sdf": 0,
            "successful_pdbqt": 0,
            "failed": 0,
            "n_jobs": n_jobs,
            "command_used": command_used,
            "limit": limit,
        }
        write_json(manifest, report_path)
        return manifest

    worker_jobs = max(1, int(n_jobs))
    if worker_jobs == 1:
        results = [convert_one_sdf(path, sdf_dir, out_dir, force, command) for path in sdf_files]
    else:
        results = Parallel(n_jobs=worker_jobs, backend="loky")(
            delayed(convert_one_sdf)(path, sdf_dir, out_dir, force, command) for path in sdf_files
        )

    failures = [
        {"sdf_path": result.sdf_path, "pdbqt_path": result.pdbqt_path, "failure_reason": result.failure_reason}
        for result in results
        if not result.success
    ]
    pd.DataFrame(failures, columns=FAILURE_COLUMNS).to_csv(failures_path, index=False)

    successful = sum(1 for result in results if result.success)
    skipped_existing = sum(1 for result in results if result.skipped_existing)
    manifest = {
        "status": "complete" if successful else "no_pdbqt_created",
        "total_input_sdf": int(len(sdf_files)),
        "successful_pdbqt": int(successful),
        "failed": int(len(results) - successful),
        "skipped_existing": int(skipped_existing),
        "n_jobs": int(worker_jobs),
        "command_used": command_used,
        "limit": limit,
        "failures_path": str(failures_path),
    }
    write_json(manifest, report_path)
    logger.info("Wrote PDBQT ligand manifest to %s", report_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ligand PDBQT files from SDF conformers.")
    parser.add_argument("--sdf-dir", default="data/processed/ligands/MAPK1/phase1/sdf", help="Input SDF directory.")
    parser.add_argument("--out-dir", default="data/processed/ligands/MAPK1/phase1/pdbqt", help="Output PDBQT directory.")
    parser.add_argument("--report", default="results/reports/mapk1_phase1_pdbqt_manifest.json", help="Manifest JSON.")
    parser.add_argument("--failures", default="results/tables/mapk1_phase1_pdbqt_failures.csv", help="Failure CSV.")
    parser.add_argument("--n-jobs", type=int, default=16, help="Parallel worker count.")
    parser.add_argument("--limit", type=int, default=None, help="Optional SDF row limit for smoke tests.")
    parser.add_argument("--force", type=parse_bool, default=False, help="Regenerate existing PDBQT files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepare_pdbqt_ligands(
        args.sdf_dir,
        args.out_dir,
        args.report,
        args.failures,
        n_jobs=args.n_jobs,
        limit=args.limit,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
