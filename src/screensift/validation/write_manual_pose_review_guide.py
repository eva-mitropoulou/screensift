from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from screensift.common.io import ensure_dir, markdown_table


REQUIRED_WORDING = (
    "Pose inspection is used to prioritize retrospective cases for analysis.\n"
    "It does not validate binding, potency, or prospective discovery."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write manual pose review guide for MAPK1 Step 9.")
    parser.add_argument("--inspection-summary", default="results/tables/mapk1_phase1_pose_inspection_summary.csv")
    parser.add_argument("--plausibility-flags", default="results/tables/mapk1_phase1_pose_plausibility_flags.csv")
    parser.add_argument("--out", default="results/reports/mapk1_phase1_manual_pose_review_guide.md")
    return parser.parse_args()


def read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(table_path)
    except Exception:
        return pd.DataFrame()


def table_for_category(df: pd.DataFrame, category: str, n: int = 10) -> str:
    if df.empty or "inspection_categories" not in df.columns:
        return "_None available._"
    sub = df[df["inspection_categories"].fillna("").astype(str).str.contains(category, regex=False)]
    if sub.empty:
        return "_None available._"
    present = [
        col
        for col in [
            "ligand_id",
            "activity_label",
            "ecfp4_active_similarity",
            "unidock_best_score",
            "CNNscore",
            "CNNaffinity",
            "gnina_affinity",
            "n_total_interactions",
            "recommended_action",
            "anomaly_flags",
        ]
        if col in sub.columns
    ]
    ordered = sub.sort_values(["manual_priority", "n_total_interactions"], ascending=[True, False])
    return markdown_table(ordered, columns=present, max_rows=n)


def write_guide(summary: pd.DataFrame, flags: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    missing = int((~summary["pose_found"].astype(bool)).sum()) if "pose_found" in summary.columns else len(summary)
    action_counts = summary["recommended_action"].value_counts().to_dict() if "recommended_action" in summary.columns else {}
    top = summary.sort_values(["manual_priority", "n_total_interactions"], ascending=[True, False]).head(20) if not summary.empty else summary
    lines = [
        "# MAPK1 Phase 1 Manual Pose Review Guide",
        "",
        REQUIRED_WORDING,
        "",
        "## What Step 9 Does",
        "",
        "Step 9 prioritizes selected retrospective cases for manual pose and failure analysis using available pose files, receptor files, and lightweight interaction/contact checks.",
        "",
        "## What Step 9 Does Not Prove",
        "",
        "It does not perform new docking, GNINA rescoring, MD, analog generation, binding validation, potency prediction, or prospective discovery.",
        "",
        "## Priority List For Manual Inspection",
        "",
        f"- Recommended action counts: {action_counts}",
        f"- Missing poses: {missing}",
        "",
        markdown_table(
            top,
            columns=[
                "ligand_id",
                "activity_label",
                "inspection_categories",
                "manual_priority",
                "n_total_interactions",
                "recommended_action",
                "anomaly_flags",
            ],
        )
        if not top.empty
        else "_No priority cases available._",
        "",
        "## Top Novel-SAV Cases",
        "",
        table_for_category(summary, "novel_sav"),
        "",
        "## Top Low-Similarity Active Cases",
        "",
        table_for_category(summary, "low_similarity_active"),
        "",
        "## Top Score Anomalies",
        "",
        table_for_category(summary, "score_anomaly_top_hit"),
        "",
        "## Top Consensus Inactive False Positives",
        "",
        table_for_category(summary, "consensus_inactive_false_positive"),
        "",
        "## Top Active False Negatives",
        "",
        table_for_category(summary, "active_false_negative"),
        "",
        "## How To Open PyMOL Scripts",
        "",
        "Generated scripts are under `results/figures/pose_panels/pymol_scripts/`. Open one with PyMOL, for example:",
        "",
        "```bash",
        "pymol results/figures/pose_panels/pymol_scripts/<ligand_id>_view.pml",
        "```",
        "",
        "## What To Look For Manually",
        "",
        "- Ligand inside the kinase pocket.",
        "- Hinge-region contacts or a plausible kinase-like interaction pattern.",
        "- Severe clashes with receptor atoms.",
        "- Ligand outside the docking box or pocket.",
        "- Unrealistic buried hydrophobic bulk.",
        "- Disconnected or broken ligand geometry.",
        "- Whether score-anomaly cases look like scoring artifacts.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = read_table(args.inspection_summary)
    flags = read_table(args.plausibility_flags)
    write_guide(summary, flags, Path(args.out))
    print(f"Manual pose review guide written: {args.out}")


if __name__ == "__main__":
    main()
