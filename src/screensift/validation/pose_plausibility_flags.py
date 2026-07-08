from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from screensift.common.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flag MAPK1 selected-pose plausibility issues.")
    parser.add_argument("--inspection-summary", default="results/tables/mapk1_phase1_pose_inspection_summary.csv")
    parser.add_argument("--pose-locations", default="results/tables/mapk1_phase1_selected_pose_locations.csv")
    parser.add_argument("--out", default="results/tables/mapk1_phase1_pose_plausibility_flags.csv")
    parser.add_argument("--report", default="results/reports/mapk1_phase1_pose_plausibility_report.md")
    return parser.parse_args()


def read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(table_path)
    except Exception:
        return pd.DataFrame()


def flags_for_row(row: pd.Series) -> list[str]:
    flags: list[str] = []
    categories = str(row.get("inspection_categories", ""))
    anomaly_flags = str(row.get("anomaly_flags", ""))
    action = str(row.get("recommended_action", ""))
    activity = str(row.get("activity_label", "")).lower()
    pose_found = bool(row.get("pose_found", False))
    success = bool(row.get("interaction_analysis_success", False))
    n_total = int(pd.to_numeric(pd.Series([row.get("n_total_interactions", 0)]), errors="coerce").fillna(0).iloc[0])

    if not pose_found:
        flags.append("missing_pose")
    if pose_found and not success:
        flags.append("pose_load_failed")
    if pose_found and success and n_total == 0:
        flags.append("no_detected_interactions")
    if pose_found and success and 0 < n_total < 5:
        flags.append("very_few_interactions")
    if anomaly_flags and anomaly_flags.lower() != "nan":
        flags.append("score_anomaly_requires_review")
    if "extreme_unidock_negative" in anomaly_flags or "extreme_unidock_negative_top_hit" in categories:
        flags.append("extreme_unidock_negative_requires_review")
    if "consensus_inactive_false_positive" in categories and activity == "inactive":
        flags.append("inactive_high_confidence_false_positive")
    if "active_false_negative" in categories and activity == "active":
        flags.append("active_low_confidence_false_negative")
    if action == "credible_structure_added_case":
        flags.append("credible_low_similarity_active")
    return sorted(set(flags))


def build_plausibility_flags(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        flags = flags_for_row(row)
        rows.append(
            {
                "ligand_id": row.get("ligand_id", ""),
                "activity_label": row.get("activity_label", ""),
                "inspection_categories": row.get("inspection_categories", ""),
                "recommended_action": row.get("recommended_action", ""),
                "pose_found": row.get("pose_found", False),
                "interaction_analysis_success": row.get("interaction_analysis_success", False),
                "n_total_interactions": row.get("n_total_interactions", 0),
                "plausibility_flags": ";".join(flags),
                "n_plausibility_flags": len(flags),
            }
        )
    return pd.DataFrame(rows)


def write_report(flags: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    counts: dict[str, int] = {}
    for text in flags.get("plausibility_flags", pd.Series(dtype=str)).fillna("").astype(str):
        for flag in [part for part in text.split(";") if part]:
            counts[flag] = counts.get(flag, 0) + 1
    lines = [
        "# MAPK1 Phase 1 Pose Plausibility Report",
        "",
        "These flags are inspection aids, not final conclusions.",
        "",
        f"- Rows flagged: {len(flags)}",
        f"- Flag counts: {dict(sorted(counts.items()))}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = read_table(args.inspection_summary)
    flags = build_plausibility_flags(summary)
    out = Path(args.out)
    ensure_dir(out.parent)
    flags.to_csv(out, index=False)
    write_report(flags, Path(args.report))
    print(f"Pose plausibility flags written: {out} ({len(flags)} rows)")


if __name__ == "__main__":
    main()
