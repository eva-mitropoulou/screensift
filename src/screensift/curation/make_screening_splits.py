from __future__ import annotations

import argparse
import zlib
from pathlib import Path
from typing import Any

import pandas as pd


from screensift.common.io import ensure_dir, load_yaml, write_json
from screensift.common.logging_utils import setup_logger


def assign_scaffold_folds(
    frame: pd.DataFrame,
    n_folds: int,
    smiles_col: str = "canonical_smiles",
    fold_column: str = "scaffold_fold",
) -> pd.DataFrame:
    """Deterministically assign each ligand to one of ``n_folds`` by its
    Bemis-Murcko scaffold, so no scaffold spans two folds.

    This lets a downstream evaluation hold out whole scaffolds (a
    scaffold-disjoint split), which controls the analog bias that
    ``metrics.scaffold_leakage`` only measures. Assignment is a stable crc32 of
    the scaffold SMILES, so it is reproducible across runs and processes and
    independent of row order. Ligands without a parseable scaffold fall back to
    a per-ligand hash so they are still assigned rather than dropped.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    def scaffold_key(smiles: Any) -> str:
        if smiles is None or pd.isna(smiles) or not str(smiles).strip():
            return ""
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or ""

    out = frame.copy()
    if smiles_col not in out.columns:
        raise ValueError(f"Scaffold folds require a SMILES column: {smiles_col!r}")
    scaffolds = out[smiles_col].map(scaffold_key)

    def fold_for(scaffold: str, fallback: str) -> int:
        key = scaffold or f"__nosaffold__{fallback}"
        return int(zlib.crc32(key.encode("utf-8")) % n_folds)

    ligand_ids = out["ligand_id"].astype(str) if "ligand_id" in out.columns else scaffolds.index.astype(str)
    out[fold_column] = [fold_for(scaffold, lid) for scaffold, lid in zip(scaffolds, ligand_ids)]
    out["scaffold_smiles"] = scaffolds
    return out


def infer_target_slug(curated_path: Path) -> str:
    name = curated_path.name
    suffix = "_ligands_curated.csv"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return "mapk1"


def output_path(out_dir: Path, target_slug: str, run_id: str = "phase1") -> Path:
    return out_dir / f"{target_slug}_{run_id}_screening_set.csv"


def empty_split_frame(columns: list[str], split_id_column: str) -> pd.DataFrame:
    split_columns = list(columns)
    if split_id_column not in split_columns:
        split_columns.append(split_id_column)
    return pd.DataFrame(columns=split_columns)


def shuffled_frame(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_split(
    actives: pd.DataFrame,
    sampled_inactives: pd.DataFrame,
    split_id: str,
    split_id_column: str,
    shuffle_seed: int,
) -> pd.DataFrame:
    split = pd.concat([actives, sampled_inactives], ignore_index=True)
    split[split_id_column] = split_id
    return shuffled_frame(split, shuffle_seed)


def make_splits(
    curated_path: str | Path,
    screening_config_path: str | Path,
    out_dir: str | Path,
    report_path: str | Path,
    target_slug: str | None = None,
    run_id: str = "phase1",
) -> dict[str, Any]:
    logger = setup_logger("make_screening_splits")
    curated_path = Path(curated_path)
    out_dir = Path(out_dir)
    report_path = Path(report_path)
    target_slug = target_slug or infer_target_slug(curated_path)
    target_id = target_slug.upper()

    config = load_yaml(screening_config_path)
    random_seed = int(config.get("random_seed", 42))
    include_all_inactives = bool(config.get("include_all_inactives", False))
    requested_phase1_inactives = int(config.get("phase1_inactives", 5000))
    activity_label_column = str(config.get("activity_label_column", "activity_label"))
    active_label = str(config.get("active_label", "active"))
    inactive_label = str(config.get("inactive_label", "inactive"))
    split_id_column = str(config.get("split_id_column", "split_id"))

    phase1_path = output_path(out_dir, target_slug, run_id)
    ensure_dir(out_dir)
    ensure_dir(report_path.parent)

    curated = pd.read_csv(curated_path)
    if activity_label_column not in curated.columns:
        raise ValueError(f"Curated table must contain {activity_label_column!r}: {curated_path}")

    if curated.empty:
        empty = empty_split_frame(list(curated.columns), split_id_column)
        empty.to_csv(phase1_path, index=False)
        manifest = {
            "status": "empty_curated_input",
            "total_curated": 0,
            "total_actives": 0,
            "total_inactives": 0,
            "phase1_actives": 0,
            "phase1_inactives": 0,
            "random_seed": random_seed,
            "warnings": ["Curated ligand table is empty; wrote empty phase 1 screening split."],
        }
        write_json(manifest, report_path)
        return manifest

    actives = curated[curated[activity_label_column] == active_label].copy()
    inactives = curated[curated[activity_label_column] == inactive_label].copy()
    shuffled_inactives = shuffled_frame(inactives, random_seed)

    phase1_inactives = len(shuffled_inactives) if include_all_inactives else requested_phase1_inactives
    n_phase1 = min(phase1_inactives, len(shuffled_inactives))
    phase1_inactive_frame = shuffled_inactives.iloc[:n_phase1].copy()

    phase1 = build_split(actives, phase1_inactive_frame, f"{target_id}_{run_id}", split_id_column, random_seed + 1)

    warnings: list[str] = []
    scaffold_folds = config.get("scaffold_folds")
    scaffold_summary: dict[str, Any] = {}
    if scaffold_folds:
        n_folds = int(scaffold_folds)
        if n_folds < 2:
            raise ValueError("scaffold_folds must be >= 2 when set.")
        smiles_col = str(config.get("smiles_column", "canonical_smiles"))
        if smiles_col not in phase1.columns:
            warnings.append(
                f"scaffold_folds requested but SMILES column {smiles_col!r} is absent; skipped scaffold folding."
            )
        else:
            phase1 = assign_scaffold_folds(phase1, n_folds, smiles_col=smiles_col)
            # A scaffold that lands in exactly one fold is holdout-safe.
            per_scaffold_folds = phase1.groupby("scaffold_smiles")["scaffold_fold"].nunique()
            scaffold_summary = {
                "scaffold_folds": n_folds,
                "distinct_scaffolds": int(phase1["scaffold_smiles"].replace("", pd.NA).nunique()),
                "fold_sizes": {str(k): int(v) for k, v in phase1["scaffold_fold"].value_counts().sort_index().items()},
                "scaffolds_spanning_multiple_folds": int((per_scaffold_folds > 1).sum()),
            }

    phase1.to_csv(phase1_path, index=False)

    if not include_all_inactives and len(inactives) < phase1_inactives:
        warnings.append(f"Requested {phase1_inactives} phase1 inactives, but only {len(inactives)} are available.")
    if actives.empty and config.get("require_all_actives", True):
        warnings.append("No active ligands are present in the curated table.")

    manifest = {
        "status": "complete",
        "total_curated": int(len(curated)),
        "total_actives": int(len(actives)),
        "total_inactives": int(len(inactives)),
        "phase1_actives": int((phase1[activity_label_column] == active_label).sum()),
        "phase1_inactives": int((phase1[activity_label_column] == inactive_label).sum()),
        "include_all_inactives": include_all_inactives,
        "random_seed": random_seed,
        "target_slug": target_slug,
        "run_id": run_id,
        "warnings": warnings,
        **({"scaffold_split": scaffold_summary} if scaffold_summary else {}),
    }
    write_json(manifest, report_path)
    logger.info("Wrote screening split to %s", phase1_path)
    logger.info("Wrote split manifest to %s", report_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a screening split CSV from curated ligands.")
    parser.add_argument("--curated", default="results/tables/mapk1_ligands_curated.csv", help="Curated ligand table.")
    parser.add_argument("--screening-config", default="configs/screening.yml", help="Screening configuration YAML.")
    parser.add_argument("--out-dir", default="data/processed/splits", help="Output split directory.")
    parser.add_argument(
        "--report",
        default="results/reports/mapk1_screening_splits_manifest.json",
        help="Screening split manifest JSON.",
    )
    parser.add_argument("--target-slug", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-id", default="phase1", help="Run identifier used in split output names.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    make_splits(args.curated, args.screening_config, args.out_dir, args.report, args.target_slug, args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
