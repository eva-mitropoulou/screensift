from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd

from screensift.common.io import ensure_dir, markdown_table
from screensift.validation.pose_io import stable_ligand_id


REQUIRED_COLUMNS = [
    "ligand_id",
    "triage_tier",
    "activity_label",
    "inspection_categories",
    "manual_pose_verdict",
    "inside_pocket",
    "plausible_contacts",
    "hinge_or_pocket_contacts",
    "obvious_clash",
    "chemically_intact",
    "surface_only_pose",
    "score_artifact_concern",
    "pose_review_showcase",
    "analog_seed_decision",
    "manual_notes",
]

REQUIRED_WORDING = (
    "Manual visual pose review is used to decide which retrospective candidates are suitable for "
    "analog-prioritization. It does not validate binding or potency."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write manual pose verdict CSV and Markdown templates.")
    parser.add_argument("--triage-table", default="results/tables/mapk1_phase1_candidate_triage.csv")
    parser.add_argument("--pose-panel-report", default="results/reports/mapk1_phase1_pose_review_panel_manifest.md")
    parser.add_argument("--out-csv", default="results/tables/mapk1_phase1_manual_pose_verdict_template.csv")
    parser.add_argument("--out-md", default="results/reports/mapk1_phase1_manual_pose_verdict_template.md")
    return parser.parse_args()


def read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(table_path)
    if "ligand_id" in df.columns:
        df["ligand_id"] = df["ligand_id"].map(stable_ligand_id)
    return df


def selected_ligands_from_report(path: str | Path) -> list[str]:
    report_path = Path(path)
    if not report_path.exists():
        return []
    text = report_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"<!--\s*selected_ligands:\s*([^>]+?)\s*-->", text)
    if not match:
        return []
    return [stable_ligand_id(part.strip()) for part in match.group(1).split(",") if part.strip()]


def build_template(triage: pd.DataFrame, selected_ids: list[str]) -> pd.DataFrame:
    if triage.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    work = triage.copy()
    work["ligand_id"] = work["ligand_id"].map(stable_ligand_id)
    if selected_ids:
        selected_set = set(selected_ids)
        work = work[work["ligand_id"].isin(selected_set)].copy()
        order = {ligand_id: idx for idx, ligand_id in enumerate(selected_ids)}
        work["_order"] = work["ligand_id"].map(order).fillna(9999)
        work = work.sort_values("_order")
    else:
        work = work[work.get("triage_tier", pd.Series(index=work.index, dtype=object)).isin(["A_analog_seed", "B_backup_seed"])].copy()
    for col in REQUIRED_COLUMNS:
        if col not in work.columns:
            work[col] = ""
    work["manual_pose_verdict"] = "unclear"
    work["analog_seed_decision"] = ""
    for col in [
        "inside_pocket",
        "plausible_contacts",
        "hinge_or_pocket_contacts",
        "obvious_clash",
        "chemically_intact",
        "surface_only_pose",
        "score_artifact_concern",
        "pose_review_showcase",
    ]:
        work[col] = ""
    work["manual_notes"] = ""
    return work[REQUIRED_COLUMNS].reset_index(drop=True)


def write_markdown(template: pd.DataFrame, out_md: Path) -> None:
    ensure_dir(out_md.parent)
    lines = [
        "# MAPK1 Phase 1 Manual Pose Verdict Template",
        "",
        REQUIRED_WORDING,
        "",
        "## How To Fill This Table",
        "",
        "Open the corresponding pose-review PyMOL script or PNG for each ligand, then fill the CSV fields.",
        "",
        "Allowed `manual_pose_verdict` values:",
        "",
        "- `keep`: plausible pose suitable for Step 10 analog prioritization",
        "- `backup`: plausible but weaker or unclear",
        "- `reject`: bad pose, clash, outside pocket, broken ligand, or scoring artifact",
        "- `unclear`: needs another look",
        "",
        "Allowed `analog_seed_decision` values:",
        "",
        "- `use_for_step10`",
        "- `backup_only`",
        "- `do_not_optimize`",
        "",
        "Boolean fields can be filled with `yes`, `no`, or `unclear`.",
        "",
        "## Ligands Included",
        "",
    ]
    if template.empty:
        lines.append("_No ligands selected._")
    else:
        lines.append(markdown_table(template, columns=["ligand_id", "triage_tier", "activity_label", "inspection_categories"]))
    lines.extend(
        [
            "",
            "## Reminder",
            "",
            "These are retrospective inspection candidates. Do not describe any ligand here as a validated inhibitor or prospective hit.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def run(triage_table: str | Path, pose_panel_report: str | Path, out_csv: str | Path, out_md: str | Path) -> pd.DataFrame:
    triage = read_table(triage_table)
    selected_ids = selected_ligands_from_report(pose_panel_report)
    template = build_template(triage, selected_ids)
    csv_path = Path(out_csv)
    ensure_dir(csv_path.parent)
    template.to_csv(csv_path, index=False)
    write_markdown(template, Path(out_md))
    return template


def main() -> None:
    args = parse_args()
    template = run(args.triage_table, args.pose_panel_report, args.out_csv, args.out_md)
    print(f"Manual pose verdict template written: rows={len(template)}")


if __name__ == "__main__":
    main()
