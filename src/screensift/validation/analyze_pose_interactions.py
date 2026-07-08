from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from screensift.common.io import ensure_dir, load_yaml, markdown_table
from screensift.validation.pose_io import (
    build_pose_index,
    find_receptor_for_pose,
    heavy_atoms,
    parse_structure_atoms,
    pose_source,
    resolve_pose_path,
    stable_ligand_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze interactions for MAPK1 Step 8 selected poses.")
    parser.add_argument("--inspection-table", default="results/tables/mapk1_phase1_inspection_shortlist.csv")
    parser.add_argument("--pose-locations", default="results/tables/mapk1_phase1_selected_pose_locations.csv")
    parser.add_argument("--prepared-receptors", default="results/tables/mapk1_prepared_receptors.csv")
    parser.add_argument("--pose-config", default="configs/pose_inspection.yml")
    parser.add_argument("--out-interactions", default="results/tables/mapk1_phase1_pose_interactions.csv")
    parser.add_argument("--out-summary", default="results/tables/mapk1_phase1_pose_inspection_summary.csv")
    parser.add_argument("--report", default="results/reports/mapk1_phase1_pose_interaction_report.md")
    return parser.parse_args()


def read_optional_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(table_path)
    except Exception:
        return pd.DataFrame()
    if "ligand_id" in df.columns:
        df["ligand_id"] = df["ligand_id"].map(stable_ligand_id)
    return df


def merge_inspection_and_pose_locations(inspection: pd.DataFrame, locations: pd.DataFrame) -> pd.DataFrame:
    if inspection.empty:
        return pd.DataFrame()
    inspection = inspection.copy()
    inspection["ligand_id"] = inspection["ligand_id"].map(stable_ligand_id)
    if locations.empty or "ligand_id" not in locations.columns:
        for col in ["pose_found", "selected_pose_file", "pose_source", "transfer_needed", "notes"]:
            inspection[col] = False if col in {"pose_found", "transfer_needed"} else ""
        return inspection
    locations = locations.copy()
    locations["ligand_id"] = locations["ligand_id"].map(stable_ligand_id)
    return inspection.merge(locations.drop_duplicates("ligand_id"), on="ligand_id", how="left", suffixes=("", "_pose"))


def distance(atom_a: Any, atom_b: Any) -> float:
    return math.sqrt((atom_a.x - atom_b.x) ** 2 + (atom_a.y - atom_b.y) ** 2 + (atom_a.z - atom_b.z) ** 2)


def is_hbond_element(element: str) -> bool:
    return element.upper() in {"N", "O", "S"}


def is_hydrophobic_element(element: str) -> bool:
    return element.upper() in {"C", "CL", "BR", "F", "I"}


def fallback_interaction_counts(receptor_path: Path, pose_path: Path, cutoffs: dict[str, float]) -> dict[str, Any]:
    receptor_atoms = heavy_atoms(parse_structure_atoms(receptor_path))
    ligand_atoms = heavy_atoms(parse_structure_atoms(pose_path))
    if not receptor_atoms:
        return {"success": False, "failure_reason": "receptor_atoms_not_parsed"}
    if not ligand_atoms:
        return {"success": False, "failure_reason": "ligand_atoms_not_parsed"}

    hbond_cutoff = float(cutoffs.get("hbond_angstrom", 3.5))
    hydrophobic_cutoff = float(cutoffs.get("hydrophobic_angstrom", 4.5))
    pi_cutoff = float(cutoffs.get("pi_stack_angstrom", 5.0))
    hbond_pairs: set[tuple[int, int]] = set()
    hydrophobic_pairs: set[tuple[int, int]] = set()
    pi_pairs: set[tuple[int, int]] = set()

    for lig in ligand_atoms:
        for rec in receptor_atoms:
            dist = distance(lig, rec)
            if dist <= hbond_cutoff and is_hbond_element(lig.element) and is_hbond_element(rec.element):
                hbond_pairs.add((lig.serial, rec.serial))
            if dist <= hydrophobic_cutoff and is_hydrophobic_element(lig.element) and is_hydrophobic_element(rec.element):
                hydrophobic_pairs.add((lig.serial, rec.serial))
            if dist <= pi_cutoff and lig.element == "C" and rec.element == "C":
                # This is only a crude aromatic-contact proxy, not a ring-normal pi-stack classifier.
                pi_pairs.add((lig.serial, rec.serial))

    return {
        "success": True,
        "failure_reason": "",
        "n_hbond_interactions": len(hbond_pairs),
        "n_hydrophobic_interactions": len(hydrophobic_pairs),
        "n_pi_interactions": len(pi_pairs),
        "n_ligand_heavy_atoms": len(ligand_atoms),
        "n_receptor_heavy_atoms": len(receptor_atoms),
    }


def prolif_available() -> bool:
    try:
        import prolif  # noqa: F401

        return True
    except Exception:
        return False


def recommended_action(row: pd.Series, success: bool, n_interactions: int, pose_found: bool) -> tuple[str, str]:
    categories = str(row.get("inspection_categories", ""))
    activity = str(row.get("activity_label", "")).lower()
    anomaly_flags = str(row.get("anomaly_flags", ""))
    if not pose_found:
        return "missing_pose_needs_transfer", "Pose not available on this machine."
    if not success or n_interactions == 0:
        return "likely_scoring_artifact", "Pose could not be analyzed or has no detected fallback contacts."
    if "extreme_unidock_negative" in anomaly_flags or "extreme_unidock_negative_top_hit" in categories:
        return "inspect_manually", "Extreme score anomaly needs manual inspection regardless of contact count."
    if "consensus_inactive_false_positive" in categories and activity == "inactive":
        return "false_positive_failure_case", "Inactive consensus false positive; inspect for scoring artifacts."
    if "active_false_negative" in categories and activity == "active":
        return "false_negative_failure_case", "Known active ranked poorly; inspect for preparation or receptor mismatch."
    if ("novel_sav" in categories or "low_similarity_active" in categories) and activity == "active" and n_interactions > 0:
        return "credible_structure_added_case", "Low-similarity active has detectable contacts and merits manual pose review."
    return "inspect_manually", "Retrospective inspection candidate."


def analyze_poses(
    inspection: pd.DataFrame,
    pose_locations: pd.DataFrame,
    prepared_receptors: pd.DataFrame,
    pose_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = merge_inspection_and_pose_locations(inspection, pose_locations)
    max_ligands = int(pose_config.get("pose_inspection", {}).get("max_ligands", len(merged) or 0))
    merged = merged.head(max_ligands).copy()
    pose_index = build_pose_index(["results/poses"])
    cutoffs = pose_config.get("interaction_checks", {}).get("distance_cutoffs", {})
    prolif_ok = prolif_available() and bool(pose_config.get("pose_inspection", {}).get("run_prolif_if_available", True))

    interaction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for _, row in merged.iterrows():
        ligand_id = stable_ligand_id(row.get("ligand_id"))
        pose_path, pose_note = resolve_pose_path(row, pose_index=pose_index)
        pose_found = pose_path is not None and pose_path.exists()
        row_for_receptor = row.copy()
        if pose_path:
            row_for_receptor["selected_pose_file"] = str(pose_path)
        receptor_path, receptor_note = find_receptor_for_pose(row_for_receptor, prepared_receptors)

        attempted = bool(pose_found and receptor_path and receptor_path.exists())
        success = False
        failure_reason = ""
        counts = {
            "n_hbond_interactions": 0,
            "n_hydrophobic_interactions": 0,
            "n_pi_interactions": 0,
        }
        notes: list[str] = []
        if pose_note:
            notes.append(pose_note)
        if not pose_found:
            failure_reason = "missing_pose"
        elif not receptor_path or not receptor_path.exists():
            failure_reason = receptor_note or "missing_receptor"
        else:
            fallback = fallback_interaction_counts(receptor_path, pose_path, cutoffs)
            success = bool(fallback.get("success"))
            failure_reason = str(fallback.get("failure_reason", ""))
            counts.update(
                {
                    "n_hbond_interactions": int(fallback.get("n_hbond_interactions", 0)),
                    "n_hydrophobic_interactions": int(fallback.get("n_hydrophobic_interactions", 0)),
                    "n_pi_interactions": int(fallback.get("n_pi_interactions", 0)),
                }
            )
            notes.append("Fallback distance-contact analysis used.")
            if prolif_ok:
                notes.append("ProLIF is importable, but fallback contact counts are reported for robust batch operation.")
            else:
                notes.append("ProLIF unavailable or disabled.")

        n_total = counts["n_hbond_interactions"] + counts["n_hydrophobic_interactions"] + counts["n_pi_interactions"]
        action, interpretation = recommended_action(row, success, n_total, pose_found)

        interaction_rows.append(
            {
                "ligand_id": ligand_id,
                "activity_label": row.get("activity_label", ""),
                "inspection_categories": row.get("inspection_categories", ""),
                "pose_found": pose_found,
                "pose_file": str(pose_path) if pose_path else "",
                "receptor_file": str(receptor_path) if receptor_path else "",
                "interaction_analysis_attempted": attempted,
                "interaction_analysis_success": success,
                **counts,
                "n_total_interactions": n_total,
                "prolif_success": False,
                "fallback_used": attempted,
                "interaction_notes": " ".join(notes),
                "failure_reason": failure_reason,
            }
        )
        summary_rows.append(
            {
                "ligand_id": ligand_id,
                "activity_label": row.get("activity_label", ""),
                "canonical_smiles": row.get("canonical_smiles", ""),
                "inspection_categories": row.get("inspection_categories", ""),
                "manual_priority": row.get("manual_priority", ""),
                "ecfp4_active_similarity": row.get("ecfp4_active_similarity", ""),
                "unidock_best_score": row.get("unidock_best_score", ""),
                "CNNscore": row.get("CNNscore", ""),
                "CNNaffinity": row.get("CNNaffinity", ""),
                "gnina_affinity": row.get("gnina_affinity", ""),
                "anomaly_flags": row.get("anomaly_flags", ""),
                "pose_found": pose_found,
                "pose_file": str(pose_path) if pose_path else "",
                "receptor_file": str(receptor_path) if receptor_path else "",
                "interaction_analysis_success": success,
                "n_total_interactions": n_total,
                "pose_interpretation": interpretation,
                "recommended_action": action,
            }
        )

    return pd.DataFrame(interaction_rows), pd.DataFrame(summary_rows)


def write_report(interactions: pd.DataFrame, summary: pd.DataFrame, report_path: Path) -> None:
    ensure_dir(report_path.parent)
    n_total = len(summary)
    n_pose = int(summary["pose_found"].sum()) if "pose_found" in summary.columns else 0
    n_attempted = int(interactions["interaction_analysis_attempted"].sum()) if "interaction_analysis_attempted" in interactions.columns else 0
    n_success = int(interactions["interaction_analysis_success"].sum()) if "interaction_analysis_success" in interactions.columns else 0
    actions = summary["recommended_action"].value_counts().to_dict() if "recommended_action" in summary.columns else {}
    top = summary.sort_values(["manual_priority", "n_total_interactions"], ascending=[True, False]).head(20)
    lines = [
        "# MAPK1 Phase 1 Pose Interaction Report",
        "",
        "Pose inspection is used to prioritize retrospective cases for analysis. It does not validate binding, potency, or prospective discovery.",
        "",
        f"- Shortlist ligands analyzed: {n_total}",
        f"- Poses found: {n_pose}",
        f"- Interaction analyses attempted: {n_attempted}",
        f"- Successful fallback interaction analyses: {n_success}",
        f"- Recommended action counts: {actions}",
        "",
        "## Top Manual Review Cases",
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
        else "_No cases available._",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    inspection = read_optional_table(args.inspection_table)
    locations = read_optional_table(args.pose_locations)
    receptors = read_optional_table(args.prepared_receptors)
    config = load_yaml(args.pose_config)
    interactions, summary = analyze_poses(inspection, locations, receptors, config)

    out_interactions = Path(args.out_interactions)
    out_summary = Path(args.out_summary)
    ensure_dir(out_interactions.parent)
    ensure_dir(out_summary.parent)
    interactions.to_csv(out_interactions, index=False)
    summary.to_csv(out_summary, index=False)
    write_report(interactions, summary, Path(args.report))
    print(
        f"Pose interaction analysis complete: ligands={len(summary)} "
        f"poses={int(summary['pose_found'].sum()) if not summary.empty else 0} "
        f"success={int(interactions['interaction_analysis_success'].sum()) if not interactions.empty else 0}"
    )


if __name__ == "__main__":
    main()
