from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from screensift.common.io import ensure_dir, markdown_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Step 8 MAPK1 pose/failure inspection report.")
    parser.add_argument("--cleanup-report", default="results/reports/mapk1_pre_step8_baseline_cleanup_report.md")
    parser.add_argument("--inspection-table", default="results/tables/mapk1_phase1_inspection_shortlist.csv")
    parser.add_argument("--pose-locations", default="results/tables/mapk1_phase1_selected_pose_locations.csv")
    parser.add_argument("--pose-qc", default="results/tables/mapk1_phase1_selected_pose_qc.csv")
    parser.add_argument("--out", default="results/reports/mapk1_phase1_pose_failure_inspection_report.md")
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def counts_by_category(inspection: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if inspection.empty or "inspection_categories" not in inspection.columns:
        return counts
    for text in inspection["inspection_categories"].fillna("").astype(str):
        for category in [part for part in text.split(";") if part]:
            counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def top_rows(df: pd.DataFrame, columns: list[str], n: int = 10) -> str:
    if df.empty:
        return "_None available._"
    present = [col for col in columns if col in df.columns]
    if not present:
        return "_No matching columns available._"
    return markdown_table(df, columns=present, max_rows=n)


def write_report(cleanup_text: str, inspection: pd.DataFrame, poses: pd.DataFrame, qc: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    category_counts = counts_by_category(inspection)
    pose_found = int(poses["pose_found"].sum()) if "pose_found" in poses.columns else 0
    pose_missing = int(len(poses) - pose_found)
    qc_attempts = int(qc["qc_attempted"].sum()) if "qc_attempted" in qc.columns else 0
    qc_success = int(qc["qc_success"].sum()) if "qc_success" in qc.columns else 0

    def category_subset(name: str) -> pd.DataFrame:
        if inspection.empty or "inspection_categories" not in inspection.columns:
            return inspection.head(0)
        return inspection[inspection["inspection_categories"].fillna("").astype(str).str.contains(name, regex=False)]

    lines = [
        "# MAPK1 Phase 1 Pose / Failure Inspection Report",
        "",
        "No ligand is claimed as a discovered inhibitor. These are retrospective inspection candidates.",
        "",
        "## Pre-Step-8 Cleanup Summary",
        "",
        "The PDB-native baseline is deprecated and the CCD-native baseline is official for native-ligand-only reference.",
        "",
        cleanup_text.strip() if cleanup_text else "_Cleanup report missing._",
        "",
        "## Step 7b Interpretation Carried Into Step 8",
        "",
        "- ECFP4 dominates globally in the retrospective phase 1 benchmark.",
        "- Native-ligand-only CCD ECFP4 is weak and is used only as a contextual structural-reference baseline.",
        "- Structure methods add value mainly in low-similarity regions and selected failure/discordance cases.",
        "",
        "## Shortlist Summary",
        "",
        f"- Inspection ligands: {len(inspection)}",
        f"- Counts by category: {category_counts}",
        "",
        "## Top Novel-SAV Cases",
        "",
        top_rows(
            category_subset("novel_sav"),
            ["ligand_id", "activity_label", "ecfp4_active_similarity", "best_structure_method", "best_structure_percentile", "SAV", "anomaly_flags"],
        ),
        "",
        "## Top Low-Similarity Actives Recovered By Structure Methods",
        "",
        top_rows(
            category_subset("low_similarity_active"),
            ["ligand_id", "activity_label", "ecfp4_active_similarity", "unidock_best_score", "CNNscore", "CNNaffinity", "gnina_affinity"],
        ),
        "",
        "## Similarity-Driven Analog-Bias Cases",
        "",
        top_rows(
            category_subset("similarity_driven_analog_bias_hit"),
            ["ligand_id", "activity_label", "ecfp4_active_similarity", "ecfp4_similarity_bin", "unidock_best_score", "CNNscore", "CNNaffinity"],
        ),
        "",
        "## Consensus Inactive False Positives",
        "",
        top_rows(
            category_subset("consensus_inactive_false_positive"),
            ["ligand_id", "activity_label", "ecfp4_active_similarity", "unidock_best_score", "CNNscore", "CNNaffinity", "gnina_affinity"],
        ),
        "",
        "## Active False Negatives",
        "",
        top_rows(
            category_subset("active_false_negative"),
            ["ligand_id", "activity_label", "ecfp4_active_similarity", "unidock_best_score", "CNNscore", "CNNaffinity", "gnina_affinity"],
        ),
        "",
        "## Score-Anomaly Top Hits",
        "",
        "The extreme Uni-Dock negative row and related anomalous-score hits are retained for manual inspection, not treated as validated hits.",
        "",
        top_rows(
            category_subset("score_anomaly_top_hit"),
            ["ligand_id", "activity_label", "unidock_best_score", "ligand_efficiency", "CNNscore", "CNNaffinity", "gnina_affinity", "anomaly_flags"],
        ),
        "",
        "## Pose Availability",
        "",
        f"- Pose files found on VM: {pose_found}",
        f"- Pose files missing / transfer needed: {pose_missing}",
        "",
        "## Pose QC",
        "",
        f"- Pose QC attempts: {qc_attempts}",
        f"- File-level QC successes: {qc_success}",
        "",
        "## Recommended Manual Inspection",
        "",
        top_rows(
            inspection.sort_values("manual_priority", ascending=True) if "manual_priority" in inspection.columns else inspection,
            ["ligand_id", "activity_label", "inspection_categories", "manual_priority", "ecfp4_active_similarity", "anomaly_flags", "reason_for_selection"],
            n=20,
        ),
        "",
        "## Later Analog-Prioritization Candidates",
        "",
        "Prioritize low-similarity active and Novel-SAV cases only after pose plausibility and score-anomaly checks. High-similarity hits are useful as analog-bias controls, not discovery claims.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cleanup_path = Path(args.cleanup_report)
    cleanup_text = cleanup_path.read_text(encoding="utf-8") if cleanup_path.exists() else ""
    inspection = read_table(Path(args.inspection_table))
    poses = read_table(Path(args.pose_locations))
    qc = read_table(Path(args.pose_qc))
    write_report(cleanup_text, inspection, poses, qc, Path(args.out))
    print(f"Inspection report written: {args.out}")


if __name__ == "__main__":
    main()
