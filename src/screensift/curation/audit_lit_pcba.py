from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


from screensift.common.chemistry import DESCRIPTOR_KEYS, canonicalize_smiles, compute_basic_descriptors, inchikey_from_smiles
from screensift.common.io import ensure_dir, load_yaml, read_text_table_auto, write_json
from screensift.common.logging_utils import setup_logger


CANDIDATE_SMILES_COLUMNS = ["smiles", "SMILES", "canonical_smiles", "molecule", "ligand_smiles", "smiles_str"]
CANDIDATE_ID_COLUMNS = ["ligand_id", "molecule_id", "compound_id", "id", "name", "title"]
SUPPORTED_SUFFIXES = {".smi", ".csv", ".tsv", ".txt"}
BASE_COLUMNS = [
    "ligand_id",
    "source_file",
    "activity_label",
    "raw_smiles",
    "canonical_smiles",
    "inchikey",
    "valid",
    "failure_reason",
]
AUDIT_COLUMNS = BASE_COLUMNS + DESCRIPTOR_KEYS


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


def target_raw_dir(paths: dict[str, Any], target_config: dict[str, Any], target: str) -> Path:
    target_id = target_config.get("target_id", target)
    raw_lit_pcba_dir = paths.get("raw_lit_pcba_dir")
    if not raw_lit_pcba_dir:
        raise KeyError("paths.yml must define raw_lit_pcba_dir")
    return Path(raw_lit_pcba_dir) / target_id


def logical_suffix(path: Path) -> str:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes and suffixes[-1] == ".gz" and len(suffixes) >= 2:
        return suffixes[-2]
    return suffixes[-1] if suffixes else ""


def is_supported_ligand_file(path: Path) -> bool:
    return logical_suffix(path) in SUPPORTED_SUFFIXES


def classify_lit_pcba_file(path: Path) -> str | None:
    text = path.as_posix().lower()
    if "inactive" in text or "inactives" in text:
        return "inactive"
    if "active" in text or "actives" in text:
        return "active"
    return None


def find_ligand_files(raw_dir: Path, logger) -> list[tuple[Path, str]]:
    if not raw_dir.exists():
        return []

    files: list[tuple[Path, str]] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or not is_supported_ligand_file(path):
            continue
        label = classify_lit_pcba_file(path)
        if label is None:
            logger.warning("Skipping ligand-like file with unclear activity label: %s", path)
            continue
        files.append((path, label))
    return files


def infer_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    columns = list(frame.columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate

    lower_lookup = {str(column).lower(): column for column in columns}
    for candidate in candidates:
        match = lower_lookup.get(candidate.lower())
        if match is not None:
            return str(match)
    return None


def smiles_column(frame: pd.DataFrame) -> str:
    candidate = infer_column(frame, CANDIDATE_SMILES_COLUMNS)
    if candidate is not None:
        return candidate
    if frame.shape[1] == 0:
        raise ValueError("Cannot infer SMILES column from a table with zero columns.")
    return str(frame.columns[0])


def read_no_header(path: Path) -> pd.DataFrame:
    suffix = logical_suffix(path)
    if suffix == ".csv":
        sep = ","
    elif suffix == ".tsv":
        sep = "\t"
    else:
        sep = r"\s+"

    frame = pd.read_csv(path, sep=sep, engine="python", header=None, compression="infer")
    if frame.shape[1] == 0:
        raise ValueError(f"No columns found in {path}")
    frame = frame.rename(columns={frame.columns[0]: "smiles"})
    if frame.shape[1] > 1:
        frame = frame.rename(columns={frame.columns[1]: "ligand_id"})
    return frame


def first_header_looks_like_smiles(frame: pd.DataFrame) -> bool:
    if frame.shape[1] == 0:
        return False
    header_value = str(frame.columns[0]).strip()
    if not header_value:
        return False
    return canonicalize_smiles(header_value) is not None


def load_ligand_table(path: Path) -> tuple[pd.DataFrame, str, str | None]:
    header_frame = read_text_table_auto(path)
    explicit_smiles_col = infer_column(header_frame, CANDIDATE_SMILES_COLUMNS)
    if explicit_smiles_col is not None:
        return header_frame, explicit_smiles_col, infer_column(header_frame, CANDIDATE_ID_COLUMNS)

    if logical_suffix(path) == ".smi" or first_header_looks_like_smiles(header_frame):
        no_header_frame = read_no_header(path)
        return no_header_frame, "smiles", infer_column(no_header_frame, CANDIDATE_ID_COLUMNS)

    return header_frame, smiles_column(header_frame), infer_column(header_frame, CANDIDATE_ID_COLUMNS)


def ligand_id_for_row(row: pd.Series, id_column: str | None, source_file: Path, row_number: int) -> str:
    if id_column is not None:
        value = row.get(id_column)
        if not pd.isna(value) and str(value).strip():
            return str(value).strip()
    return f"{source_file.stem}_{row_number:06d}"


def audit_smiles(raw_smiles: Any, ligand_id: str, source_file: Path, activity_label: str) -> dict[str, Any]:
    value = "" if pd.isna(raw_smiles) else str(raw_smiles).strip()
    record: dict[str, Any] = {
        "ligand_id": ligand_id,
        "source_file": str(source_file),
        "activity_label": activity_label,
        "raw_smiles": value,
        "canonical_smiles": None,
        "inchikey": None,
        "valid": False,
        "failure_reason": None,
    }
    record.update({key: None for key in DESCRIPTOR_KEYS})

    if not value:
        record["failure_reason"] = "empty_smiles"
        return record

    canonical = canonicalize_smiles(value)
    if canonical is None:
        record["failure_reason"] = "invalid_smiles"
        return record

    inchikey = inchikey_from_smiles(value)
    if inchikey is None:
        record["canonical_smiles"] = canonical
        record["failure_reason"] = "inchikey_failed"
        return record

    descriptors = compute_basic_descriptors(value)
    if descriptors is None:
        record["canonical_smiles"] = canonical
        record["inchikey"] = inchikey
        record["failure_reason"] = "descriptor_failed"
        return record

    record.update(descriptors)
    record["canonical_smiles"] = canonical
    record["inchikey"] = inchikey
    record["valid"] = True
    return record


def expected_counts(target_config: dict[str, Any]) -> tuple[int | None, int | None]:
    expected = target_config.get("expected_lit_pcba_counts", {})
    active = expected.get("actives")
    inactive = expected.get("inactives")
    return (int(active) if active is not None else None, int(inactive) if inactive is not None else None)


def empty_outputs(out: Path, manifest: Path, raw_dir: Path, target_config: dict[str, Any], target: str) -> None:
    ensure_dir(out.parent)
    ensure_dir(manifest.parent)
    slug = target.lower()
    curated_path = out.parent / f"{slug}_ligands_curated.csv"
    failures_path = out.parent / f"{slug}_curation_failures.csv"

    empty = pd.DataFrame(columns=AUDIT_COLUMNS)
    empty.to_csv(out, index=False)
    empty.to_csv(curated_path, index=False)
    empty.to_csv(failures_path, index=False)

    expected_active, expected_inactive = expected_counts(target_config)
    write_json(
        {
            "status": "missing_raw_files",
            "target_id": target_config.get("target_id", target),
            "expected_active_count": expected_active,
            "expected_inactive_count": expected_inactive,
            "raw_active_records": 0,
            "raw_inactive_records": 0,
            "valid_active_records": 0,
            "valid_inactive_records": 0,
            "duplicate_records_removed": 0,
            "activity_conflicts_removed": 0,
            "final_active_count": 0,
            "final_inactive_count": 0,
            "observed_counts_match_expected": False,
            "warnings": [
                "No active/inactive LIT-PCBA files were found.",
                f"Manual fallback: place MAPK1 LIT-PCBA files under {raw_dir}.",
            ],
        },
        manifest,
    )


def count_by_label(frame: pd.DataFrame, label_column: str = "activity_label") -> dict[str, int]:
    if frame.empty or label_column not in frame.columns:
        return {"active": 0, "inactive": 0}
    counts = frame[label_column].value_counts().to_dict()
    return {"active": int(counts.get("active", 0)), "inactive": int(counts.get("inactive", 0))}


def valid_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    if frame["valid"].dtype == bool:
        return frame["valid"]
    return frame["valid"].astype(str).str.lower().isin({"true", "1", "yes"})


def curate_audit_frame(audit_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    valid = audit_frame[valid_mask(audit_frame)].copy()
    invalid_failures = audit_frame[~valid_mask(audit_frame)].copy()
    if valid.empty:
        return valid, invalid_failures, 0, 0

    label_counts = valid.groupby("inchikey")["activity_label"].nunique()
    conflict_keys = set(label_counts[label_counts > 1].index)
    conflict_failures = valid[valid["inchikey"].isin(conflict_keys)].copy()
    if not conflict_failures.empty:
        conflict_failures["valid"] = False
        conflict_failures["failure_reason"] = "activity_conflict"

    valid_no_conflicts = valid[~valid["inchikey"].isin(conflict_keys)].copy()
    if valid_no_conflicts.empty:
        failures = pd.concat([invalid_failures, conflict_failures], ignore_index=True)
        return valid_no_conflicts, failures[AUDIT_COLUMNS], 0, len(conflict_failures)

    valid_no_conflicts["_label_priority"] = valid_no_conflicts["activity_label"].map({"active": 0, "inactive": 1}).fillna(2)
    valid_no_conflicts = valid_no_conflicts.sort_values(["inchikey", "_label_priority", "source_file", "ligand_id"])
    duplicate_records_removed = int(valid_no_conflicts.duplicated("inchikey", keep="first").sum())
    curated = valid_no_conflicts.drop_duplicates("inchikey", keep="first").drop(columns=["_label_priority"])
    failures = pd.concat([invalid_failures, conflict_failures], ignore_index=True)
    return curated[AUDIT_COLUMNS], failures[AUDIT_COLUMNS], duplicate_records_removed, len(conflict_failures)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit MAPK1 LIT-PCBA ligand files.")
    parser.add_argument("--target-config", default="configs/targets.yml", help="Target configuration YAML.")
    parser.add_argument("--paths", default="configs/paths.yml", help="Project path configuration YAML.")
    parser.add_argument("--target", default="MAPK1", help="Target identifier.")
    parser.add_argument("--out", default="results/tables/mapk1_dataset_audit.csv", help="Audit CSV output.")
    parser.add_argument(
        "--manifest",
        default="results/reports/mapk1_dataset_manifest.json",
        help="Dataset manifest JSON output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger("audit_lit_pcba")

    target_config = get_target_config(load_yaml(args.target_config), args.target)
    paths = load_yaml(args.paths)
    raw_dir = target_raw_dir(paths, target_config, args.target)
    out = Path(args.out)
    manifest = Path(args.manifest)
    ensure_dir(out.parent)
    ensure_dir(manifest.parent)

    ligand_files = find_ligand_files(raw_dir, logger)
    if not ligand_files:
        logger.warning("No active/inactive ligand files found under %s.", raw_dir)
        print(f"No MAPK1 LIT-PCBA files found under {raw_dir}. Place active/inactive .smi, .csv, .tsv, .txt, or .gz files there.")
        empty_outputs(out, manifest, raw_dir, target_config, args.target)
        return 0

    rows: list[dict[str, Any]] = []
    for ligand_file, label in ligand_files:
        logger.info("Auditing %s as %s.", ligand_file, label)
        try:
            frame, smiles_col, id_col = load_ligand_table(ligand_file)
        except Exception as exc:
            logger.warning("Skipping unreadable ligand file %s: %s", ligand_file, exc)
            continue

        if smiles_col not in frame.columns:
            raise ValueError(f"Inferred SMILES column {smiles_col!r} not found in {ligand_file}")

        for row_number, (_, row) in enumerate(frame.iterrows(), start=1):
            rows.append(audit_smiles(row[smiles_col], ligand_id_for_row(row, id_col, ligand_file, row_number), ligand_file, label))

    audit_frame = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    curated, failures, duplicate_records_removed, activity_conflicts_removed = curate_audit_frame(audit_frame)

    slug = args.target.lower()
    curated_path = out.parent / f"{slug}_ligands_curated.csv"
    failures_path = out.parent / f"{slug}_curation_failures.csv"

    audit_frame.to_csv(out, index=False)
    curated.to_csv(curated_path, index=False)
    failures.to_csv(failures_path, index=False)

    raw_counts = count_by_label(audit_frame)
    valid_counts = count_by_label(audit_frame[valid_mask(audit_frame)])
    final_counts = count_by_label(curated)
    expected_active, expected_inactive = expected_counts(target_config)
    observed_counts_match_expected = (
        expected_active is not None
        and expected_inactive is not None
        and raw_counts["active"] == expected_active
        and raw_counts["inactive"] == expected_inactive
    )

    warnings: list[str] = []
    if expected_active is not None and raw_counts["active"] != expected_active:
        warnings.append(f"Observed active raw count {raw_counts['active']} differs from expected {expected_active}.")
    if expected_inactive is not None and raw_counts["inactive"] != expected_inactive:
        warnings.append(f"Observed inactive raw count {raw_counts['inactive']} differs from expected {expected_inactive}.")
    if activity_conflicts_removed:
        warnings.append(f"Removed {activity_conflicts_removed} records with active/inactive InChIKey conflicts.")

    write_json(
        {
            "status": "complete",
            "target_id": target_config.get("target_id", args.target),
            "expected_active_count": expected_active,
            "expected_inactive_count": expected_inactive,
            "raw_active_records": raw_counts["active"],
            "raw_inactive_records": raw_counts["inactive"],
            "valid_active_records": valid_counts["active"],
            "valid_inactive_records": valid_counts["inactive"],
            "duplicate_records_removed": duplicate_records_removed,
            "activity_conflicts_removed": activity_conflicts_removed,
            "final_active_count": final_counts["active"],
            "final_inactive_count": final_counts["inactive"],
            "observed_counts_match_expected": observed_counts_match_expected,
            "warnings": warnings,
        },
        manifest,
    )

    logger.info("Wrote audit table to %s.", out)
    logger.info("Wrote curated table to %s.", curated_path)
    logger.info("Wrote failures table to %s.", failures_path)
    logger.info("Wrote manifest to %s.", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
