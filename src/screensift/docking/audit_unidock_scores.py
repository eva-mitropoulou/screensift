from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


from screensift.common.io import ensure_dir


SUCCESS_STATUSES = {"success", "succeeded", "ok", "complete", "completed"}
REQUIRED_CONTEXT_COLUMNS = [
    "docking_id",
    "ligand_pdbqt",
    "receptor_pdbqt",
    "pdb_id",
    "output_pose_file",
    "status",
]
QC_FLAG_COLUMNS = [
    "score_missing",
    "score_nonfinite",
    "score_positive",
    "score_extreme_low",
    "score_extreme_high",
    "pose_file_missing",
    "status_not_success",
]


def infer_raw_score_path(scores_path: str | Path) -> Path:
    path = Path(scores_path)
    if path.name.endswith("_scores.csv"):
        return path.with_name(path.name.replace("_scores.csv", "_raw.csv"))
    return path.with_name(f"{path.stem}_raw.csv")


def load_scores_with_context(scores_path: str | Path) -> tuple[pd.DataFrame, list[str], Path]:
    path = Path(scores_path)
    if not path.exists():
        raise FileNotFoundError(f"Uni-Dock score table not found: {path}")

    scores = pd.read_csv(path)
    missing = [column for column in REQUIRED_CONTEXT_COLUMNS if column not in scores.columns]
    if not missing:
        return scores, [], path

    raw_path = infer_raw_score_path(path)
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        raw_missing = [column for column in REQUIRED_CONTEXT_COLUMNS if column not in raw.columns]
        if raw_missing:
            raise ValueError(
                f"Score table {path} is missing {missing}; fallback raw table {raw_path} "
                f"is also missing required columns: {raw_missing}"
            )
        return raw, [f"Loaded raw context table {raw_path} because {path} was missing columns: {missing}"], raw_path

    raise ValueError(
        f"Score table {path} is missing required columns: {missing}. "
        f"Expected raw fallback table at {raw_path}, but it does not exist."
    )


def coerce_score_column(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "best_score" in result.columns:
        score_source = "best_score"
    elif "score" in result.columns:
        score_source = "score"
    else:
        raise ValueError("Uni-Dock table must contain either 'best_score' or 'score'.")
    result["score"] = pd.to_numeric(result[score_source], errors="coerce")
    if "best_score" not in result.columns:
        result["best_score"] = result["score"]
    return result


def ligand_stem_from_row(row: pd.Series) -> str:
    ligand_pdbqt = row.get("ligand_pdbqt")
    if ligand_pdbqt is not None and not pd.isna(ligand_pdbqt) and str(ligand_pdbqt).strip():
        return Path(str(ligand_pdbqt)).stem
    ligand_id = row.get("ligand_id")
    if ligand_id is not None and not pd.isna(ligand_id) and str(ligand_id).strip():
        return str(ligand_id)
    docking_id = row.get("docking_id")
    if docking_id is not None and not pd.isna(docking_id):
        text = str(docking_id)
        return text.split("_", 1)[1] if "_" in text else text
    return ""


def split_index_from_ligand_stem(stem: str) -> int | None:
    match = re.match(r"^(\d+)_", stem)
    if not match:
        return None
    return int(match.group(1))


def ligand_id_from_ligand_stem(stem: str) -> str:
    match = re.match(r"^\d+_(.+)$", stem)
    return match.group(1) if match else stem


def build_split_lookup(splits_path: str | Path) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    path = Path(splits_path)
    if not path.exists():
        raise FileNotFoundError(f"Screening split table not found: {path}")

    splits = pd.read_csv(path)
    warnings: list[str] = []
    if "activity_label" not in splits.columns:
        raise ValueError(f"Screening split table must contain activity_label: {path}")
    if "ligand_id" not in splits.columns:
        warnings.append("Screening split table has no ligand_id column; label merge will use row indices only.")

    optional_columns = ["ligand_id", "activity_label", "canonical_smiles", "inchikey", "raw_smiles"]
    by_index: dict[int, dict[str, Any]] = {}
    by_ligand_id: dict[str, dict[str, Any]] = {}
    for split_index, row in splits.reset_index(drop=True).iterrows():
        record = {column: row.get(column) for column in optional_columns if column in splits.columns}
        by_index[int(split_index)] = record
        if "ligand_id" in splits.columns and not pd.isna(row.get("ligand_id")):
            by_ligand_id[str(row.get("ligand_id"))] = record
    return by_index, by_ligand_id, warnings


def merge_split_labels(frame: pd.DataFrame, splits_path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    by_index, by_ligand_id, warnings = build_split_lookup(splits_path)
    merged = frame.copy()

    ligand_stems: list[str] = []
    split_indices: list[int | None] = []
    split_ligand_ids: list[str] = []
    activity_labels: list[Any] = []
    canonical_smiles: list[Any] = []
    inchikeys: list[Any] = []
    raw_smiles: list[Any] = []
    merge_methods: list[str] = []

    for _index, row in merged.iterrows():
        stem = ligand_stem_from_row(row)
        ligand_stems.append(stem)
        split_index = split_index_from_ligand_stem(stem)
        split_indices.append(split_index)
        ligand_id_candidate = ligand_id_from_ligand_stem(stem)

        split_record = None
        merge_method = "unmatched"
        if split_index is not None and split_index in by_index:
            split_record = by_index[split_index]
            merge_method = "split_index"
        elif ligand_id_candidate in by_ligand_id:
            split_record = by_ligand_id[ligand_id_candidate]
            merge_method = "ligand_id"

        if split_record is None:
            split_ligand_ids.append(ligand_id_candidate)
            activity_labels.append(pd.NA)
            canonical_smiles.append(pd.NA)
            inchikeys.append(pd.NA)
            raw_smiles.append(pd.NA)
        else:
            split_ligand_ids.append(str(split_record.get("ligand_id", ligand_id_candidate)))
            activity_labels.append(split_record.get("activity_label", pd.NA))
            canonical_smiles.append(split_record.get("canonical_smiles", pd.NA))
            inchikeys.append(split_record.get("inchikey", pd.NA))
            raw_smiles.append(split_record.get("raw_smiles", pd.NA))
        merge_methods.append(merge_method)

    merged["ligand_file_id"] = ligand_stems
    merged["split_index"] = split_indices
    merged["ligand_id"] = split_ligand_ids
    merged["activity_label"] = activity_labels
    merged["canonical_smiles"] = canonical_smiles
    merged["inchikey"] = inchikeys
    merged["raw_smiles"] = raw_smiles
    merged["activity_merge_method"] = merge_methods

    unmatched_count = int((merged["activity_merge_method"] == "unmatched").sum())
    if unmatched_count:
        warnings.append(f"Could not merge activity labels for {unmatched_count} docking rows.")
    return merged, warnings


def status_is_success(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in SUCCESS_STATUSES


def path_exists(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and Path(text).exists()


def is_finite_score(value: Any) -> bool:
    if pd.isna(value):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def add_qc_flags(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    finite = result["score"].map(is_finite_score)
    result["score_missing"] = result["score"].isna()
    result["score_nonfinite"] = ~finite & ~result["score_missing"]
    result["score_positive"] = finite & (result["score"] > 0)
    result["score_extreme_low"] = finite & (result["score"] < -30)
    result["score_extreme_high"] = finite & (result["score"] > 20)
    result["pose_file_missing"] = ~result["output_pose_file"].map(path_exists)
    result["status_not_success"] = ~result["status"].map(status_is_success)
    result["valid_for_ranking"] = (
        finite
        & ~result["pose_file_missing"]
        & (result["score"] <= 0)
        & (result["score"] >= -30)
        & ~result["status_not_success"]
    )
    result["qc_flag_reasons"] = result.apply(
        lambda row: ";".join(flag for flag in QC_FLAG_COLUMNS if bool(row[flag])),
        axis=1,
    )
    return result


def best_per_ligand(clean: pd.DataFrame) -> pd.DataFrame:
    valid = clean[clean["valid_for_ranking"]].copy()
    columns = [
        "ligand_id",
        "activity_label",
        "best_score",
        "best_pdb_id",
        "best_receptor_pdbqt",
        "best_output_pose_file",
        "n_valid_receptors",
        "canonical_smiles",
        "inchikey",
    ]
    if valid.empty:
        return pd.DataFrame(columns=columns)

    best_indices = valid.groupby("ligand_id", dropna=False)["score"].idxmin()
    best = valid.loc[best_indices].copy()
    receptor_counts = valid.groupby("ligand_id", dropna=False).size().rename("n_valid_receptors")
    best = best.merge(receptor_counts, left_on="ligand_id", right_index=True, how="left")
    best["best_score"] = best["score"]
    best["best_pdb_id"] = best["pdb_id"]
    best["best_receptor_pdbqt"] = best["receptor_pdbqt"]
    best["best_output_pose_file"] = best["output_pose_file"]
    return best[columns].sort_values(["best_score", "ligand_id"]).reset_index(drop=True)


def receptor_summary(clean: pd.DataFrame) -> pd.DataFrame:
    if clean.empty:
        return pd.DataFrame()

    grouped = clean.groupby("pdb_id", dropna=False)
    summary = grouped.agg(
        total=("score", "size"),
        valid=("valid_for_ranking", "sum"),
        flagged=("qc_flag_reasons", lambda values: int((values.astype(str) != "").sum())),
        score_min=("score", "min"),
        score_median=("score", "median"),
        score_max=("score", "max"),
    ).reset_index()

    if "activity_label" in clean.columns:
        labels = (
            clean.pivot_table(
                index="pdb_id",
                columns="activity_label",
                values="docking_id",
                aggfunc="count",
                fill_value=0,
            )
            .add_prefix("label_")
            .reset_index()
        )
        summary = summary.merge(labels, on="pdb_id", how="left")
    return summary


def table_preview(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if frame.empty:
        return "None\n"
    available = [column for column in columns if column in frame.columns]
    return frame[available].head(max_rows).to_string(index=False) + "\n"


def write_report(
    clean: pd.DataFrame,
    flagged: pd.DataFrame,
    best: pd.DataFrame,
    report_path: str | Path,
    warnings: list[str],
    source_path: Path,
) -> None:
    summary = receptor_summary(clean)
    score_series = pd.to_numeric(clean["score"], errors="coerce") if not clean.empty else pd.Series(dtype=float)
    valid_scores = pd.to_numeric(clean.loc[clean["valid_for_ranking"], "score"], errors="coerce")
    flag_counts = {flag: int(clean[flag].sum()) for flag in QC_FLAG_COLUMNS} if not clean.empty else {}

    best_examples = clean[clean["valid_for_ranking"]].sort_values("score", ascending=True)
    positive_examples = clean.sort_values("score", ascending=False, na_position="last")
    missing_examples = clean[clean["score_missing"] | clean["pose_file_missing"]]

    report = [
        "# Uni-Dock Score QC",
        "",
        "This is a score and pose-file sanity audit for Uni-Dock phase 1 outputs. It flags values that are unsuitable for enrichment ranking; it does not rerun docking or delete poses.",
        "",
        "## Inputs",
        "",
        f"- score_source: `{source_path}`",
        f"- total_rows: {len(clean)}",
        f"- flagged_rows: {len(flagged)}",
        f"- valid_for_ranking: {int(clean['valid_for_ranking'].sum()) if not clean.empty else 0}",
        f"- best_per_ligand_rows: {len(best)}",
        "",
        "## Score Ranges",
        "",
        f"- raw_min: {score_series.min() if not score_series.empty else 'NA'}",
        f"- raw_median: {score_series.median() if not score_series.empty else 'NA'}",
        f"- raw_max: {score_series.max() if not score_series.empty else 'NA'}",
        f"- valid_min: {valid_scores.min() if not valid_scores.empty else 'NA'}",
        f"- valid_median: {valid_scores.median() if not valid_scores.empty else 'NA'}",
        f"- valid_max: {valid_scores.max() if not valid_scores.empty else 'NA'}",
        "",
        "## Flag Counts",
        "",
    ]
    report.extend(f"- {flag}: {count}" for flag, count in flag_counts.items())
    report.extend(["", "## Receptor Summary", "", "```text", summary.to_string(index=False), "```"])

    if warnings:
        report.extend(["", "## Warnings", ""])
        report.extend(f"- {warning}" for warning in warnings)

    report.extend(
        [
            "",
            "## Top 20 Best Valid Scores",
            "",
            "```text",
            table_preview(best_examples, ["ligand_id", "activity_label", "pdb_id", "score", "output_pose_file"]),
            "```",
            "",
            "## Top 20 Most Positive Or Extreme Scores",
            "",
            "```text",
            table_preview(positive_examples, ["ligand_id", "activity_label", "pdb_id", "score", "qc_flag_reasons", "output_pose_file"]),
            "```",
            "",
            "## Missing Pose Or Score Examples",
            "",
            "```text",
            table_preview(missing_examples, ["ligand_id", "activity_label", "pdb_id", "score", "qc_flag_reasons", "output_pose_file"]),
            "```",
            "",
        ]
    )

    report_file = Path(report_path)
    ensure_dir(report_file.parent)
    report_file.write_text("\n".join(report), encoding="utf-8")


def audit_unidock_scores(
    scores_path: str | Path,
    splits_path: str | Path,
    out_clean: str | Path,
    out_flagged: str | Path,
    out_best: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    scores, warnings, source_path = load_scores_with_context(scores_path)
    scores = coerce_score_column(scores)
    scores, merge_warnings = merge_split_labels(scores, splits_path)
    warnings.extend(merge_warnings)
    clean = add_qc_flags(scores)
    flagged = clean[clean["qc_flag_reasons"] != ""].copy()
    best = best_per_ligand(clean)

    for output_path, frame in [(out_clean, clean), (out_flagged, flagged), (out_best, best)]:
        path = Path(output_path)
        ensure_dir(path.parent)
        frame.to_csv(path, index=False)
    write_report(clean, flagged, best, report, warnings, source_path)

    valid_scores = pd.to_numeric(clean.loc[clean["valid_for_ranking"], "score"], errors="coerce")
    return {
        "total_rows": int(len(clean)),
        "flagged_rows": int(len(flagged)),
        "valid_for_ranking": int(clean["valid_for_ranking"].sum()) if not clean.empty else 0,
        "best_per_ligand_rows": int(len(best)),
        "valid_score_min": float(valid_scores.min()) if not valid_scores.empty else None,
        "valid_score_median": float(valid_scores.median()) if not valid_scores.empty else None,
        "valid_score_max": float(valid_scores.max()) if not valid_scores.empty else None,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Uni-Dock phase 1 scores and create enrichment-ready tables.")
    parser.add_argument("--scores", default="results/tables/mapk1_phase1_unidock_scores.csv")
    parser.add_argument("--splits", default="data/processed/splits/mapk1_phase1_screening_set.csv")
    parser.add_argument("--out-clean", default="results/tables/mapk1_phase1_unidock_scores_clean.csv")
    parser.add_argument("--out-flagged", default="results/tables/mapk1_phase1_unidock_score_flags.csv")
    parser.add_argument("--out-best", default="results/tables/mapk1_phase1_unidock_best_per_ligand.csv")
    parser.add_argument("--report", default="results/reports/mapk1_phase1_unidock_score_qc.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = audit_unidock_scores(args.scores, args.splits, args.out_clean, args.out_flagged, args.out_best, args.report)
    print(
        "Uni-Dock QC complete: "
        f"rows={summary['total_rows']} "
        f"flagged={summary['flagged_rows']} "
        f"valid_for_ranking={summary['valid_for_ranking']} "
        f"best_per_ligand={summary['best_per_ligand_rows']}"
    )
    for warning in summary["warnings"]:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
