from __future__ import annotations

import argparse
import re
import signal
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem


from screensift.common.io import ensure_dir, write_json
from screensift.common.logging_utils import setup_logger


FAILURE_COLUMNS = ["ligand_id", "raw_smiles", "canonical_smiles", "output_path", "failure_reason"]


@dataclass
class ConformerResult:
    ligand_id: str
    output_path: str | None
    success: bool
    failure_reason: str | None
    skipped_existing: bool = False


class ConformerTimeoutError(TimeoutError):
    pass


def _timeout_handler(signum, frame) -> None:
    raise ConformerTimeoutError("3D conformer generation timed out.")


def safe_filename(value: Any, max_length: int = 90) -> str:
    text = str(value if value is not None else "ligand").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    if not text:
        text = "ligand"
    return text[:max_length]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def ligand_identifier(row: dict[str, Any], index: int) -> str:
    for key in ("ligand_id", "inchikey", "canonical_smiles"):
        value = row.get(key)
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            return str(value)
    return f"ligand_{index:07d}"


def smiles_from_row(row: dict[str, Any]) -> str | None:
    for key in ("canonical_smiles", "raw_smiles", "smiles", "SMILES"):
        value = row.get(key)
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            return str(value)
    return None


def generate_3d_mol(smiles: str, seed: int = 42) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES.")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True
    embed_status = AllChem.EmbedMolecule(mol, params)
    if embed_status != 0:
        raise ValueError(f"RDKit ETKDG embedding failed with status {embed_status}.")

    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception as exc:
        raise ValueError(f"3D force-field optimization failed: {exc}") from exc
    return mol


def sdf_output_path(out_dir: Path, ligand_id: str, index: int, batch_size: int) -> Path:
    batch_dir = out_dir / f"batch_{index // batch_size:03d}"
    return batch_dir / f"{index:07d}_{safe_filename(ligand_id)}.sdf"


def generate_one_conformer(
    row: dict[str, Any],
    index: int,
    out_dir: str | Path,
    force: bool = False,
    seed: int = 42,
    batch_size: int = 1000,
    timeout_seconds: int | None = None,
) -> ConformerResult:
    ligand_id = ligand_identifier(row, index)
    out_path = sdf_output_path(Path(out_dir), ligand_id, index, batch_size)
    if out_path.exists() and not force:
        return ConformerResult(ligand_id=ligand_id, output_path=str(out_path), success=True, failure_reason=None, skipped_existing=True)

    smiles = smiles_from_row(row)
    if smiles is None:
        return ConformerResult(ligand_id=ligand_id, output_path=str(out_path), success=False, failure_reason="missing_smiles")

    try:
        ensure_dir(out_path.parent)
        if timeout_seconds and timeout_seconds > 0:
            previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(timeout_seconds))
        else:
            previous_handler = None
        try:
            mol = generate_3d_mol(smiles, seed=seed + index)
        finally:
            if timeout_seconds and timeout_seconds > 0:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous_handler)
        mol.SetProp("_Name", ligand_id)
        writer = Chem.SDWriter(str(out_path))
        writer.write(mol)
        writer.close()
        return ConformerResult(ligand_id=ligand_id, output_path=str(out_path), success=True, failure_reason=None)
    except Exception as exc:
        return ConformerResult(ligand_id=ligand_id, output_path=str(out_path), success=False, failure_reason=str(exc))


def failure_rows(input_rows: list[dict[str, Any]], results: list[ConformerResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row, result in zip(input_rows, results):
        if result.success:
            continue
        rows.append(
            {
                "ligand_id": result.ligand_id,
                "raw_smiles": row.get("raw_smiles"),
                "canonical_smiles": row.get("canonical_smiles"),
                "output_path": result.output_path,
                "failure_reason": result.failure_reason,
            }
        )
    return rows


def generate_conformers(
    input_path: str | Path,
    out_dir: str | Path,
    report_path: str | Path,
    failures_path: str | Path,
    n_jobs: int = 16,
    limit: int | None = None,
    batch_size: int = 1000,
    force: bool = False,
    seed: int = 42,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    logger = setup_logger("generate_3d_conformers")
    input_path = Path(input_path)
    out_dir = ensure_dir(out_dir)
    report_path = Path(report_path)
    failures_path = Path(failures_path)
    ensure_dir(report_path.parent)
    ensure_dir(failures_path.parent)

    ligands = pd.read_csv(input_path)
    total_input = int(len(ligands))
    if limit is not None:
        ligands = ligands.head(limit).copy()
    records = ligands.to_dict(orient="records")

    if not records:
        pd.DataFrame(columns=FAILURE_COLUMNS).to_csv(failures_path, index=False)
        manifest = {
            "status": "empty_input",
            "total_input": total_input,
            "attempted": 0,
            "successful": 0,
            "failed": 0,
            "n_jobs": n_jobs,
            "batches": 0,
            "limit": limit,
        }
        write_json(manifest, report_path)
        return manifest

    worker_jobs = max(1, int(n_jobs))
    if worker_jobs == 1:
        results = [
            generate_one_conformer(row, index, out_dir, force=force, seed=seed, batch_size=batch_size, timeout_seconds=timeout_seconds)
            for index, row in enumerate(records)
        ]
    else:
        results = Parallel(n_jobs=worker_jobs, backend="loky")(
            delayed(generate_one_conformer)(row, index, out_dir, force=force, seed=seed, batch_size=batch_size, timeout_seconds=timeout_seconds)
            for index, row in enumerate(records)
        )

    failures = failure_rows(records, results)
    pd.DataFrame(failures, columns=FAILURE_COLUMNS).to_csv(failures_path, index=False)

    successful = sum(1 for result in results if result.success)
    skipped_existing = sum(1 for result in results if result.skipped_existing)
    batches = len({Path(str(result.output_path)).parent for result in results if result.output_path})
    manifest = {
        "status": "complete",
        "total_input": total_input,
        "attempted": int(len(records)),
        "successful": int(successful),
        "failed": int(len(results) - successful),
        "skipped_existing": int(skipped_existing),
        "n_jobs": int(worker_jobs),
        "batches": int(batches),
        "limit": limit,
        "batch_size": int(batch_size),
        "timeout_seconds": timeout_seconds,
        "failures_path": str(failures_path),
    }
    write_json(manifest, report_path)
    logger.info("Wrote 3D conformer manifest to %s", report_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 3D ligand conformers with RDKit ETKDG.")
    parser.add_argument("--input", default="data/processed/splits/mapk1_phase1_screening_set.csv", help="Input ligand CSV.")
    parser.add_argument("--out-dir", default="data/processed/ligands/MAPK1/phase1/sdf", help="Output SDF directory.")
    parser.add_argument("--report", default="results/reports/mapk1_phase1_3d_manifest.json", help="Manifest JSON.")
    parser.add_argument("--failures", default="results/tables/mapk1_phase1_3d_failures.csv", help="Failure CSV.")
    parser.add_argument("--n-jobs", type=int, default=16, help="Parallel worker count.")
    parser.add_argument("--limit", type=int, default=None, help="Optional input row limit for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Ligand files per output batch directory.")
    parser.add_argument("--force", type=parse_bool, default=False, help="Regenerate existing SDF files.")
    parser.add_argument("--seed", type=int, default=42, help="RDKit embedding seed base.")
    parser.add_argument("--timeout-seconds", type=int, default=None, help="Optional per-ligand timeout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generate_conformers(
        args.input,
        args.out_dir,
        args.report,
        args.failures,
        n_jobs=args.n_jobs,
        limit=args.limit,
        batch_size=args.batch_size,
        force=args.force,
        seed=args.seed,
        timeout_seconds=args.timeout_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
