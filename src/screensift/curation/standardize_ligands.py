from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


from screensift.common.io import ensure_dir
from screensift.common.logging_utils import setup_logger


def parse_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standardize audited ligand records into curated and failure tables.")
    parser.add_argument("--input", default="results/tables/mapk1_dataset_audit.csv", help="Input audit CSV.")
    parser.add_argument("--out", default="results/tables/mapk1_ligands_curated.csv", help="Curated ligand CSV.")
    parser.add_argument("--failures", default="results/tables/mapk1_curation_failures.csv", help="Failure CSV.")
    parser.add_argument("--deduplicate-by", default="inchikey", help="Column used to deduplicate valid molecules.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger("standardize_ligands")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Audit CSV not found: {input_path}")

    audit = pd.read_csv(input_path)
    valid_column = "valid" if "valid" in audit.columns else "is_valid"
    if valid_column not in audit.columns:
        raise ValueError(f"Audit CSV must contain a valid or is_valid column: {input_path}")
    if args.deduplicate_by not in audit.columns:
        raise ValueError(f"Deduplication column {args.deduplicate_by!r} not found in {input_path}")

    valid_mask = parse_bool_series(audit[valid_column])
    valid = audit[valid_mask].copy()
    failures = audit[~valid_mask].copy()

    curated = valid.drop_duplicates(args.deduplicate_by, keep="first")

    out_path = Path(args.out)
    failures_path = Path(args.failures)
    ensure_dir(out_path.parent)
    ensure_dir(failures_path.parent)

    curated.to_csv(out_path, index=False)
    failures.to_csv(failures_path, index=False)

    logger.info("Wrote %d curated molecules to %s.", len(curated), out_path)
    logger.info("Wrote %d curation failures to %s.", len(failures), failures_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
