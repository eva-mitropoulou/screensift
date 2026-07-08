from __future__ import annotations

import argparse
import operator
from pathlib import Path
from typing import Any

import pandas as pd


from screensift.common.chemistry import DESCRIPTOR_KEYS, canonicalize_smiles, compute_basic_descriptors, inchikey_from_smiles
from screensift.common.io import ensure_dir, load_yaml, read_text_table_auto, write_json
from screensift.common.logging_utils import setup_logger


ACTIVITY_ACTIVE = "active"
ACTIVITY_INACTIVE = "inactive"
ACTIVITY_UNLABELED = "unlabeled"

BASE_COLUMNS = [
    "ligand_id",
    "source_row",
    "activity_label",
    "raw_smiles",
    "canonical_smiles",
    "inchikey",
    "valid",
    "failure_reason",
]

OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "=": operator.eq,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adapt an arbitrary ligand table to the canonical screening schema.")
    parser.add_argument("--schema", required=True, help="Dataset schema YAML.")
    parser.add_argument("--target-slug", required=True, help="Lowercase target/run slug used for output names.")
    parser.add_argument("--out-audit", default=None, help="Audit CSV output.")
    parser.add_argument("--out-curated", default=None, help="Curated ligand CSV output.")
    parser.add_argument("--out-failures", default=None, help="Curation failure CSV output.")
    parser.add_argument("--manifest", default=None, help="Dataset manifest JSON output.")
    return parser.parse_args()


def default_outputs(target_slug: str) -> dict[str, Path]:
    return {
        "audit": Path(f"results/tables/{target_slug}_dataset_audit.csv"),
        "curated": Path(f"results/tables/{target_slug}_ligands_curated.csv"),
        "failures": Path(f"results/tables/{target_slug}_curation_failures.csv"),
        "manifest": Path(f"results/reports/{target_slug}_dataset_manifest.json"),
    }


def read_input_table(schema: dict[str, Any]) -> pd.DataFrame:
    input_cfg = schema.get("input", {})
    path = input_cfg.get("path")
    if not path:
        raise ValueError("Dataset schema must define input.path")
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"Input ligand table not found: {table_path}")

    sep = input_cfg.get("sep")
    if sep is not None:
        return pd.read_csv(table_path, sep=sep, engine="python")
    return read_text_table_auto(table_path)


def text_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalized_values(values: list[Any] | None) -> set[str]:
    return {text_value(value).lower() for value in values or []}


def value_from_column(row: pd.Series, column: str | None) -> Any:
    if not column:
        return None
    return row.get(column)


def label_from_label_mode(row: pd.Series, columns: dict[str, Any], activity_cfg: dict[str, Any]) -> tuple[str, str]:
    column = columns.get("activity_label")
    if not column:
        return ACTIVITY_UNLABELED, ""
    raw = text_value(row.get(column)).lower()
    if raw in normalized_values(activity_cfg.get("active_values", ["active", "1", "true"])):
        return ACTIVITY_ACTIVE, ""
    if raw in normalized_values(activity_cfg.get("inactive_values", ["inactive", "0", "false"])):
        return ACTIVITY_INACTIVE, ""
    if activity_cfg.get("allow_unlabeled", False):
        return ACTIVITY_UNLABELED, ""
    return ACTIVITY_UNLABELED, "unrecognized_activity_label"


def relation_allowed(row: pd.Series, rule: dict[str, Any], columns: dict[str, Any]) -> bool:
    allowed = rule.get("allowed_relations")
    if not allowed:
        return True
    relation_column = rule.get("relation_column") or columns.get("activity_relation")
    if not relation_column:
        # The rule restricts relations but no relation column is mapped, so the
        # restriction cannot be enforced. Fail closed: an unqualified value (e.g.
        # a ">" IC50) must not be silently accepted as an active/inactive label.
        raise ValueError(
            "Threshold rule declares 'allowed_relations' but no relation column is "
            "mapped. Set columns.activity_relation or rule.relation_column, or drop "
            "'allowed_relations'."
        )
    relation = text_value(row.get(relation_column))
    return relation in {str(item) for item in allowed}


def units_allowed(row: pd.Series, rule: dict[str, Any], columns: dict[str, Any]) -> bool:
    allowed = rule.get("allowed_units")
    if not allowed:
        return True
    units_column = rule.get("units_column") or columns.get("activity_units")
    if not units_column:
        # See relation_allowed: enforcing a unit whitelist without a units column
        # is impossible, so fail closed rather than mislabel (e.g. a uM value
        # treated as if it were nM).
        raise ValueError(
            "Threshold rule declares 'allowed_units' but no units column is mapped. "
            "Set columns.activity_units or rule.units_column, or drop 'allowed_units'."
        )
    units = text_value(row.get(units_column)).lower()
    return units in {str(item).lower() for item in allowed}


def rule_matches(row: pd.Series, rule: dict[str, Any], columns: dict[str, Any]) -> bool:
    column = rule.get("column")
    op_text = str(rule.get("operator", "")).strip()
    if not column or op_text not in OPS:
        raise ValueError(f"Threshold rule must define column and a supported operator: {rule}")
    if not relation_allowed(row, rule, columns) or not units_allowed(row, rule, columns):
        return False
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    threshold = pd.to_numeric(pd.Series([rule.get("value")]), errors="coerce").iloc[0]
    if pd.isna(value) or pd.isna(threshold):
        return False
    return bool(OPS[op_text](float(value), float(threshold)))


def label_from_threshold_mode(row: pd.Series, columns: dict[str, Any], activity_cfg: dict[str, Any]) -> tuple[str, str]:
    active_rule = activity_cfg.get("active_if")
    inactive_rule = activity_cfg.get("inactive_if")
    active = bool(active_rule and rule_matches(row, active_rule, columns))
    inactive = bool(inactive_rule and rule_matches(row, inactive_rule, columns))
    if active and inactive:
        return ACTIVITY_UNLABELED, "conflicting_activity_rules"
    if active:
        return ACTIVITY_ACTIVE, ""
    if inactive:
        return ACTIVITY_INACTIVE, ""
    if activity_cfg.get("drop_intermediate", True):
        return ACTIVITY_UNLABELED, "intermediate_or_unassigned_activity"
    return ACTIVITY_UNLABELED, ""


def activity_label(row: pd.Series, columns: dict[str, Any], activity_cfg: dict[str, Any]) -> tuple[str, str]:
    mode = str(activity_cfg.get("mode", "label")).lower()
    if mode == "label":
        return label_from_label_mode(row, columns, activity_cfg)
    if mode == "threshold":
        return label_from_threshold_mode(row, columns, activity_cfg)
    if mode in {"none", "unlabeled"}:
        return ACTIVITY_UNLABELED, ""
    raise ValueError(f"Unsupported activity mode: {mode}")


def ligand_id_for_row(row: pd.Series, columns: dict[str, Any], source_row: int) -> str:
    id_col = columns.get("ligand_id")
    value = text_value(row.get(id_col)) if id_col else ""
    return value or f"ligand_{source_row:07d}"


def audit_row(row: pd.Series, source_row: int, schema: dict[str, Any]) -> dict[str, Any]:
    columns = schema.get("columns", {})
    activity_cfg = schema.get("activity", {})
    smiles_col = columns.get("smiles")
    if not smiles_col:
        raise ValueError("Dataset schema must define columns.smiles")
    raw_smiles = text_value(row.get(smiles_col))
    label, label_failure = activity_label(row, columns, activity_cfg)

    record: dict[str, Any] = {
        "ligand_id": ligand_id_for_row(row, columns, source_row),
        "source_row": source_row,
        "activity_label": label,
        "raw_smiles": raw_smiles,
        "canonical_smiles": None,
        "inchikey": None,
        "valid": False,
        "failure_reason": "",
    }

    for column in schema.get("metadata_columns", []) or []:
        if column in row.index:
            record[column] = row.get(column)

    if label_failure:
        record["failure_reason"] = label_failure
    if not raw_smiles:
        record["failure_reason"] = "empty_smiles"
        return record

    canonical = canonicalize_smiles(raw_smiles)
    if canonical is None:
        record["failure_reason"] = "invalid_smiles"
        return record

    inchikey = inchikey_from_smiles(raw_smiles)
    if inchikey is None:
        record["canonical_smiles"] = canonical
        record["failure_reason"] = "inchikey_failed"
        return record

    descriptors = compute_basic_descriptors(canonical)
    if descriptors is None:
        record["canonical_smiles"] = canonical
        record["inchikey"] = inchikey
        record["failure_reason"] = "descriptor_failed"
        return record

    record["canonical_smiles"] = canonical
    record["inchikey"] = inchikey
    record.update(descriptors)
    if record["failure_reason"]:
        return record
    record["valid"] = True
    return record


def split_valid_failures(audit: pd.DataFrame, deduplicate_by: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if audit.empty:
        return audit.copy(), audit.copy(), {"activity_conflicts": 0, "duplicates_removed": 0}

    valid = audit[audit["valid"].astype(bool)].copy()
    failures = audit[~audit["valid"].astype(bool)].copy()

    conflict_keys: set[Any] = set()
    if deduplicate_by in valid.columns and "activity_label" in valid.columns:
        labeled = valid[valid["activity_label"].isin([ACTIVITY_ACTIVE, ACTIVITY_INACTIVE])]
        label_counts = labeled.groupby(deduplicate_by)["activity_label"].nunique()
        conflict_keys = set(label_counts[label_counts > 1].index)
        if conflict_keys:
            conflict_rows = valid[valid[deduplicate_by].isin(conflict_keys)].copy()
            conflict_rows["valid"] = False
            conflict_rows["failure_reason"] = "activity_conflict"
            failures = pd.concat([failures, conflict_rows], ignore_index=True)
            valid = valid[~valid[deduplicate_by].isin(conflict_keys)].copy()

    if deduplicate_by not in valid.columns:
        raise ValueError(f"Deduplication column {deduplicate_by!r} not found after schema adaptation.")

    priority = {ACTIVITY_ACTIVE: 0, ACTIVITY_INACTIVE: 1, ACTIVITY_UNLABELED: 2}
    valid["_label_priority"] = valid["activity_label"].map(priority).fillna(3)
    valid = valid.sort_values([deduplicate_by, "_label_priority", "ligand_id"]).reset_index(drop=True)
    duplicates_removed = int(valid.duplicated(deduplicate_by, keep="first").sum())
    curated = valid.drop_duplicates(deduplicate_by, keep="first").drop(columns=["_label_priority"])

    summary = {
        "activity_conflicts": int(len(conflict_keys)),
        "duplicates_removed": duplicates_removed,
    }
    return curated, failures, summary


def adapt_dataset(
    schema_path: str | Path,
    target_slug: str,
    out_audit: str | Path | None = None,
    out_curated: str | Path | None = None,
    out_failures: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    logger = setup_logger("adapt_dataset_schema")
    schema = load_yaml(schema_path)
    outputs = default_outputs(target_slug)
    audit_path = Path(out_audit) if out_audit else outputs["audit"]
    curated_path = Path(out_curated) if out_curated else outputs["curated"]
    failures_path = Path(out_failures) if out_failures else outputs["failures"]
    manifest_out = Path(manifest_path) if manifest_path else outputs["manifest"]

    frame = read_input_table(schema)
    rows = [audit_row(row, source_row=index + 1, schema=schema) for index, row in frame.iterrows()]
    audit = pd.DataFrame(rows)
    for column in BASE_COLUMNS + DESCRIPTOR_KEYS:
        if column not in audit.columns:
            audit[column] = pd.NA
    preferred = [col for col in BASE_COLUMNS + DESCRIPTOR_KEYS if col in audit.columns]
    remaining = [col for col in audit.columns if col not in preferred]
    audit = audit[preferred + remaining]

    deduplicate_by = str(schema.get("deduplicate_by", "inchikey"))
    curated, failures, dedup_summary = split_valid_failures(audit, deduplicate_by)

    for path in [audit_path, curated_path, failures_path, manifest_out]:
        ensure_dir(path.parent)
    audit.to_csv(audit_path, index=False)
    curated.to_csv(curated_path, index=False)
    failures.to_csv(failures_path, index=False)

    label_counts = curated["activity_label"].value_counts(dropna=False).to_dict() if "activity_label" in curated.columns else {}
    manifest = {
        "status": "complete",
        "schema": str(schema_path),
        "input_path": str(schema.get("input", {}).get("path", "")),
        "target_slug": target_slug,
        "input_rows": int(len(frame)),
        "audit_rows": int(len(audit)),
        "curated_rows": int(len(curated)),
        "failure_rows": int(len(failures)),
        "activity_label_counts": {str(key): int(value) for key, value in label_counts.items()},
        "deduplicate_by": deduplicate_by,
        **dedup_summary,
        "outputs": {
            "audit": str(audit_path),
            "curated": str(curated_path),
            "failures": str(failures_path),
        },
    }
    write_json(manifest, manifest_out)
    logger.info("Wrote canonical audit table to %s", audit_path)
    logger.info("Wrote curated ligand table to %s", curated_path)
    logger.info("Wrote dataset manifest to %s", manifest_out)
    return manifest


def main() -> int:
    args = parse_args()
    adapt_dataset(args.schema, args.target_slug, args.out_audit, args.out_curated, args.out_failures, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
