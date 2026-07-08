from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from screensift.common.io import ensure_dir, markdown_table


REQUIRED_WORDING = "These ligands are retrospective analog-prioritization seeds, not discovered inhibitors."
SEVERE_ANOMALY_FLAGS = {
    "extreme_unidock_negative",
    "suspicious_unidock_extreme_negative",
    "suspicious_ligand_efficiency_extreme",
    "extreme_positive_gnina_affinity",
    "out_of_range_cnnscore",
    "suspicious_cnnaffinity_extreme",
}

OUTPUT_COLUMNS = [
    "ligand_id",
    "activity_label",
    "evidence_bucket",
    "evidence_tags",
    "primary_evidence",
    "candidate_use_case",
    "triage_tier",
    "recommended_next_step",
    "canonical_smiles",
    "inspection_categories",
    "manual_priority",
    "ecfp4_active_similarity",
    "best_structure_method",
    "SAV",
    "unidock_best_score",
    "CNNscore",
    "CNNaffinity",
    "gnina_affinity",
    "anomaly_flags",
    "clean_population_excluded",
    "n_total_interactions",
    "pose_interpretation",
    "recommended_action",
    "triage_reason",
    "pymol_script",
]

EVIDENCE_BUCKET_ORDER = {
    "consensus_supported": 1,
    "analog_supported": 2,
    "novelty_pose_review": 3,
    "structure_supported": 4,
    "anomaly_review": 5,
    "failure_analysis": 6,
    "deprioritized": 7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triage inspected ligands into evidence buckets.")
    parser.add_argument("--inspection-summary", default="results/tables/mapk1_phase1_pose_inspection_summary.csv")
    parser.add_argument("--interactions", default="results/tables/mapk1_phase1_pose_interactions.csv")
    parser.add_argument("--plausibility-flags", default="results/tables/mapk1_phase1_pose_plausibility_flags.csv")
    parser.add_argument("--inspection-shortlist", default="results/tables/mapk1_phase1_inspection_shortlist.csv")
    parser.add_argument("--out-table", default="results/tables/mapk1_phase1_candidate_triage.csv")
    parser.add_argument("--out-report", default="results/reports/mapk1_phase1_candidate_triage_report.md")
    parser.add_argument("--out-seeds", default="results/tables/mapk1_phase1_step10_seed_ligands.csv")
    return parser.parse_args()


def stable_ligand_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(table_path)
    if "ligand_id" in df.columns:
        df["ligand_id"] = df["ligand_id"].map(stable_ligand_id)
    return df


def merge_inputs(summary: pd.DataFrame, interactions: pd.DataFrame, flags: pd.DataFrame, shortlist: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    merged = summary.copy()
    merged["ligand_id"] = merged["ligand_id"].map(stable_ligand_id)

    if not interactions.empty:
        interaction_cols = [
            col
            for col in [
                "ligand_id",
                "n_hbond_interactions",
                "n_hydrophobic_interactions",
                "n_pi_interactions",
                "n_total_interactions",
                "interaction_analysis_attempted",
                "interaction_analysis_success",
                "failure_reason",
            ]
            if col in interactions.columns
        ]
        merged = merged.merge(interactions[interaction_cols].drop_duplicates("ligand_id"), on="ligand_id", how="left", suffixes=("", "_interaction"))
        if "n_total_interactions_interaction" in merged.columns:
            merged["n_total_interactions"] = merged["n_total_interactions"].fillna(merged["n_total_interactions_interaction"])
    if not flags.empty:
        flag_cols = [col for col in ["ligand_id", "plausibility_flags", "n_plausibility_flags"] if col in flags.columns]
        merged = merged.merge(flags[flag_cols].drop_duplicates("ligand_id"), on="ligand_id", how="left")
    if not shortlist.empty:
        shortlist_cols = [
            col
            for col in [
                "ligand_id",
                "best_structure_method",
                "best_structure_percentile",
                "SAV",
                "clean_population_excluded",
                "nearest_active_ligand_id",
            ]
            if col in shortlist.columns
        ]
        merged = merged.merge(shortlist[shortlist_cols].drop_duplicates("ligand_id"), on="ligand_id", how="left", suffixes=("", "_shortlist"))
    return merged


def flags_set(text: Any) -> set[str]:
    if text is None or pd.isna(text):
        return set()
    return {part.strip() for part in str(text).split(";") if part.strip() and part.strip().lower() != "nan"}


def has_category(row: pd.Series, needle: str) -> bool:
    return needle in str(row.get("inspection_categories", ""))


def has_severe_anomaly(row: pd.Series) -> bool:
    return bool(flags_set(row.get("anomaly_flags")) & SEVERE_ANOMALY_FLAGS)


def is_clean_excluded(row: pd.Series) -> bool:
    return parse_bool(row.get("clean_population_excluded", False))


def numeric(row: pd.Series, col: str, default: float = 0.0) -> float:
    value = pd.to_numeric(pd.Series([row.get(col, default)]), errors="coerce").iloc[0]
    return float(default if pd.isna(value) else value)


def triage_score(row: pd.Series) -> float:
    score = 0.0
    activity = str(row.get("activity_label", "")).lower()
    action = str(row.get("recommended_action", ""))
    categories = str(row.get("inspection_categories", ""))
    sim = numeric(row, "ecfp4_active_similarity", 1.0)
    n_interactions = numeric(row, "n_total_interactions", 0.0)

    if activity == "active":
        score += 30
    if action == "credible_structure_added_case":
        score += 35
    elif action == "inspect_manually":
        score += 8
    elif action in {"false_positive_failure_case", "false_negative_failure_case"}:
        score -= 40
    if "novel_sav" in categories:
        score += 22
    if "low_similarity_active" in categories:
        score += 16
    if sim < 0.30:
        score += 20
    elif sim < 0.50:
        score += 12
    elif sim < 0.70:
        score += 4
    if not is_clean_excluded(row):
        score += 10
    else:
        score -= 35
    if has_severe_anomaly(row):
        score -= 55
    elif flags_set(row.get("anomaly_flags")):
        score -= 8
    if parse_bool(row.get("pose_found", False)):
        score += 5
    if parse_bool(row.get("interaction_analysis_success", False)):
        score += 10
    if n_interactions >= 20:
        score += 12
    elif n_interactions >= 5:
        score += 6
    elif n_interactions > 0:
        score += 2
    else:
        score -= 25
    if "similarity_driven_analog_bias_hit" in categories and "novel_sav" not in categories and "low_similarity_active" not in categories:
        score -= 30
    return score


def analog_seed_eligible(row: pd.Series) -> bool:
    categories = str(row.get("inspection_categories", ""))
    return all(
        [
            str(row.get("activity_label", "")).lower() == "active",
            str(row.get("recommended_action", "")) == "credible_structure_added_case",
            "novel_sav" in categories,
            numeric(row, "ecfp4_active_similarity", 1.0) < 0.50,
            not is_clean_excluded(row),
            not has_severe_anomaly(row),
            parse_bool(row.get("pose_found", False)),
            parse_bool(row.get("interaction_analysis_success", False)),
            numeric(row, "n_total_interactions", 0.0) >= 5,
            "similarity_driven_analog_bias_hit" not in categories,
        ]
    )


def failure_analysis_case(row: pd.Series) -> bool:
    categories = str(row.get("inspection_categories", ""))
    action = str(row.get("recommended_action", ""))
    return (
        "consensus_inactive_false_positive" in categories
        or "active_false_negative" in categories
        or action in {"false_positive_failure_case", "false_negative_failure_case"}
        or "score_anomaly_top_hit" in categories
        or "extreme_unidock_negative_top_hit" in categories
    )


def structure_supported(row: pd.Series) -> bool:
    method = str(row.get("best_structure_method", "") or "").strip().lower()
    action = str(row.get("recommended_action", "") or "").strip()
    sav = numeric(row, "SAV", 0.0)
    n_interactions = numeric(row, "n_total_interactions", 0.0)
    return bool(
        method
        and method != "nan"
        and action in {"credible_structure_added_case", "inspect_manually"}
        and sav > 0.0
        and n_interactions > 0
        and not has_severe_anomaly(row)
    )


def analog_supported(row: pd.Series) -> bool:
    categories = str(row.get("inspection_categories", ""))
    sim = numeric(row, "ecfp4_active_similarity", 0.0)
    return bool("similarity_driven_analog_bias_hit" in categories or sim >= 0.70)


def low_similarity(row: pd.Series) -> bool:
    return numeric(row, "ecfp4_active_similarity", 1.0) < 0.50


def evidence_tags_for_row(row: pd.Series) -> list[str]:
    tags: list[str] = []
    if analog_supported(row):
        tags.append("analog_supported")
    if structure_supported(row):
        tags.append("structure_supported")
    if analog_supported(row) and structure_supported(row):
        tags.append("consensus_supported")
    if low_similarity(row) and structure_supported(row):
        tags.append("low_similarity_structure_supported")
    if has_severe_anomaly(row) or "score_anomaly" in str(row.get("inspection_categories", "")):
        tags.append("anomaly_review")
    if failure_analysis_case(row):
        tags.append("failure_analysis")
    if str(row.get("activity_label", "")).lower() == "active":
        tags.append("known_active")
    elif str(row.get("activity_label", "")).lower() == "inactive":
        tags.append("known_inactive")
    return tags


def evidence_bucket_for_row(row: pd.Series) -> str:
    tags = set(evidence_tags_for_row(row))
    if "anomaly_review" in tags:
        return "anomaly_review"
    if "failure_analysis" in tags:
        return "failure_analysis"
    if "consensus_supported" in tags:
        return "consensus_supported"
    if "analog_supported" in tags:
        return "analog_supported"
    if "low_similarity_structure_supported" in tags:
        return "novelty_pose_review"
    if "structure_supported" in tags:
        return "structure_supported"
    return "deprioritized"


def primary_evidence_for_bucket(bucket: str) -> str:
    return {
        "consensus_supported": "ligand_similarity_and_structure",
        "analog_supported": "ligand_similarity",
        "novelty_pose_review": "structure_score_low_similarity",
        "structure_supported": "structure_score",
        "anomaly_review": "score_or_pose_anomaly",
        "failure_analysis": "benchmark_failure_mode",
        "deprioritized": "insufficient_support",
    }.get(bucket, "unclassified")


def use_case_for_bucket(bucket: str) -> str:
    return {
        "consensus_supported": "high_confidence_review",
        "analog_supported": "analog_prioritization",
        "novelty_pose_review": "manual_pose_review",
        "structure_supported": "structure_review",
        "anomaly_review": "inspect_or_reject",
        "failure_analysis": "failure_analysis",
        "deprioritized": "hold",
    }.get(bucket, "review")


def triage_reason(row: pd.Series, tier: str) -> str:
    reasons: list[str] = []
    categories = str(row.get("inspection_categories", ""))
    if str(row.get("activity_label", "")).lower() == "active":
        reasons.append("active ligand")
    else:
        reasons.append("inactive or non-active label")
    if "novel_sav" in categories:
        reasons.append("Novel-SAV/structure-added-value case")
    if numeric(row, "ecfp4_active_similarity", 1.0) < 0.30:
        reasons.append("ECFP4 Tanimoto < 0.30")
    elif numeric(row, "ecfp4_active_similarity", 1.0) < 0.50:
        reasons.append("ECFP4 Tanimoto < 0.50")
    if has_severe_anomaly(row):
        reasons.append("severe score anomaly prevents analog-seed use")
    if is_clean_excluded(row):
        reasons.append("excluded from clean score sensitivity population")
    if failure_analysis_case(row):
        reasons.append("failure-analysis category")
    if numeric(row, "n_total_interactions", 0.0) <= 0:
        reasons.append("no detected interactions")
    elif numeric(row, "n_total_interactions", 0.0) < 5:
        reasons.append("few detected interactions")
    if tier == "A_analog_seed":
        reasons.append("selected as one of the strongest analog-seed candidates")
    elif tier == "B_backup_seed":
        reasons.append("backup analog seed after Tier A")
    return "; ".join(reasons)


def pymol_script_for_ligand(ligand_id: str, script_dir: str | Path = "results/figures/pose_panels/pymol_scripts") -> str:
    path = Path(script_dir) / f"{stable_ligand_id(ligand_id)}_view.pml"
    return str(path) if path.exists() else ""


def assign_tiers(df: pd.DataFrame, tier_a_max: int = 5, tier_b_max: int = 15, tier_b_min: int = 0) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    work = df.copy()
    work["triage_score"] = work.apply(triage_score, axis=1)
    work["analog_seed_eligible"] = work.apply(analog_seed_eligible, axis=1)
    work["failure_analysis_case"] = work.apply(failure_analysis_case, axis=1)
    work["evidence_bucket"] = work.apply(evidence_bucket_for_row, axis=1)
    work["evidence_tags"] = work.apply(lambda row: ";".join(evidence_tags_for_row(row)), axis=1)
    work["primary_evidence"] = work["evidence_bucket"].map(primary_evidence_for_bucket)
    work["candidate_use_case"] = work["evidence_bucket"].map(use_case_for_bucket)
    work["triage_tier"] = "E_reject_or_hold"
    work["recommended_next_step"] = "do_not_optimize"

    eligible = work[work["analog_seed_eligible"]].copy()
    eligible = eligible.sort_values(
        ["triage_score", "SAV", "n_total_interactions", "ecfp4_active_similarity"],
        ascending=[False, False, False, True],
        na_position="last",
    )
    tier_a_ids = set(eligible.head(tier_a_max)["ligand_id"])
    tier_b_ids = set(eligible[~eligible["ligand_id"].isin(tier_a_ids)].head(tier_b_max)["ligand_id"])

    # Optional compatibility path: allow extra backup-only cases, but never pad seeds with
    # rows that have no explicit evidence support.
    if len(tier_b_ids) < tier_b_min:
        relaxed = work[
            (~work["ligand_id"].isin(tier_a_ids | tier_b_ids))
            & work["activity_label"].fillna("").astype(str).str.lower().eq("active")
            & work["recommended_action"].fillna("").astype(str).eq("credible_structure_added_case")
            & (pd.to_numeric(work["ecfp4_active_similarity"], errors="coerce") < 0.70)
            & work["evidence_bucket"].isin(["novelty_pose_review", "structure_supported", "consensus_supported"])
            & (~work.apply(has_severe_anomaly, axis=1))
            & (~work.apply(is_clean_excluded, axis=1))
        ].copy()
        relaxed = relaxed.sort_values(["triage_score", "SAV", "n_total_interactions"], ascending=[False, False, False], na_position="last")
        tier_b_ids.update(relaxed.head(tier_b_max - len(tier_b_ids))["ligand_id"])

    work.loc[work["ligand_id"].isin(tier_a_ids), ["triage_tier", "recommended_next_step"]] = [
        "A_analog_seed",
        "analog_prioritization",
    ]
    work.loc[work["ligand_id"].isin(tier_b_ids), ["triage_tier", "recommended_next_step"]] = [
        "B_backup_seed",
        "analog_prioritization",
    ]

    remaining = ~work["ligand_id"].isin(tier_a_ids | tier_b_ids)
    work.loc[remaining & work["failure_analysis_case"], ["triage_tier", "recommended_next_step"]] = [
        "D_failure_analysis",
        "failure_analysis",
    ]
    manual = (
        remaining
        & ~work["failure_analysis_case"]
        & work["activity_label"].fillna("").astype(str).str.lower().eq("active")
        & work["recommended_action"].fillna("").astype(str).isin(["credible_structure_added_case", "inspect_manually"])
        & (~work.apply(has_severe_anomaly, axis=1))
    )
    work.loc[manual, ["triage_tier", "recommended_next_step"]] = ["C_manual_review_only", "manual_pose_review"]

    work["triage_reason"] = work.apply(lambda row: triage_reason(row, str(row["triage_tier"])), axis=1)
    work["pymol_script"] = work["ligand_id"].map(pymol_script_for_ligand)

    for col in OUTPUT_COLUMNS:
        if col not in work.columns:
            work[col] = pd.NA
    tier_order = {
        "A_analog_seed": 1,
        "B_backup_seed": 2,
        "C_manual_review_only": 3,
        "D_failure_analysis": 4,
        "E_reject_or_hold": 5,
    }
    work["_tier_order"] = work["triage_tier"].map(tier_order).fillna(99)
    work["_bucket_order"] = work["evidence_bucket"].map(EVIDENCE_BUCKET_ORDER).fillna(99)
    work = work.sort_values(["_tier_order", "_bucket_order", "triage_score"], ascending=[True, True, False])
    return work[OUTPUT_COLUMNS].reset_index(drop=True)


def build_seed_table(triage: pd.DataFrame) -> pd.DataFrame:
    seeds = triage[triage["triage_tier"].isin(["A_analog_seed", "B_backup_seed"])].copy()
    seeds = seeds[seeds["recommended_next_step"].eq("analog_prioritization")]
    return seeds.reset_index(drop=True)


def write_report(triage: pd.DataFrame, seeds: pd.DataFrame, out_report: Path) -> None:
    ensure_dir(out_report.parent)

    def table_for(tier: str, n: int = 20) -> str:
        sub = triage[triage["triage_tier"].eq(tier)].copy()
        if sub.empty:
            return "_None._"
        cols = [
            "ligand_id",
            "activity_label",
            "evidence_bucket",
            "primary_evidence",
            "ecfp4_active_similarity",
            "best_structure_method",
            "SAV",
            "n_total_interactions",
            "anomaly_flags",
            "pymol_script",
            "triage_reason",
        ]
        return markdown_table(sub, columns=cols, max_rows=n)

    reject = triage[triage["recommended_next_step"].eq("do_not_optimize")]
    tier_a = triage[triage["triage_tier"].eq("A_analog_seed")]
    tier_b = triage[triage["triage_tier"].eq("B_backup_seed")]
    failure = triage[triage["triage_tier"].eq("D_failure_analysis")]
    tier_a_anomalies = tier_a[tier_a["anomaly_flags"].fillna("").astype(str).str.strip().ne("")]
    bucket_counts = triage["evidence_bucket"].value_counts().rename_axis("evidence_bucket").reset_index(name="count")
    lines = [
        "# Candidate Triage Report",
        "",
        REQUIRED_WORDING,
        "",
        "## Summary",
        "",
        f"- Tier A analog seeds: {len(tier_a)}",
        f"- Tier B backup seeds: {len(tier_b)}",
        f"- Failure-analysis cases: {len(failure)}",
        f"- Do-not-optimize / reject-or-hold cases: {len(reject)}",
        f"- Step 10 seed ligands: {len(seeds)}",
        f"- Tier A ligands with score anomalies: {len(tier_a_anomalies)}",
        "",
        "## Evidence Buckets",
        "",
        markdown_table(bucket_counts),
        "",
        "Candidates are assigned to evidence buckets before manual review so ligand-similarity support, structure-score support, consensus support, novelty/pose-review cases, anomalies, and failure modes remain visible instead of being collapsed into one weighted score.",
        "",
        "## Tier A Candidates",
        "",
        table_for("A_analog_seed"),
        "",
        "## Why Each Tier A Ligand Was Selected",
        "",
        table_for("A_analog_seed", n=10),
        "",
        "## Tier B Candidates",
        "",
        table_for("B_backup_seed"),
        "",
        "## Failure-Analysis Cases",
        "",
        table_for("D_failure_analysis"),
        "",
        "## Ligands That Should NOT Be Optimized",
        "",
        table_for("E_reject_or_hold"),
        "",
        "## Recommended PyMOL Scripts To Open First",
        "",
        table_for("A_analog_seed", n=5),
        "",
        "## Interpretation",
        "",
        "Tier A and Tier B are candidates for later analog-prioritization work only. Inactive false positives, active false negatives, severe score anomalies, and 2D-similarity-driven cases are retained for failure analysis or manual review rather than optimization.",
        "",
    ]
    out_report.write_text("\n".join(lines), encoding="utf-8")


def run_triage(
    inspection_summary: str | Path,
    interactions: str | Path,
    plausibility_flags: str | Path,
    inspection_shortlist: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = read_table(inspection_summary)
    interaction_df = read_table(interactions)
    flags = read_table(plausibility_flags)
    shortlist = read_table(inspection_shortlist)
    merged = merge_inputs(summary, interaction_df, flags, shortlist)
    triage = assign_tiers(merged)
    seeds = build_seed_table(triage)
    return triage, seeds


def main() -> None:
    args = parse_args()
    triage, seeds = run_triage(args.inspection_summary, args.interactions, args.plausibility_flags, args.inspection_shortlist)

    out_table = Path(args.out_table)
    out_report = Path(args.out_report)
    out_seeds = Path(args.out_seeds)
    ensure_dir(out_table.parent)
    ensure_dir(out_report.parent)
    ensure_dir(out_seeds.parent)
    triage.to_csv(out_table, index=False)
    seeds.to_csv(out_seeds, index=False)
    write_report(triage, seeds, out_report)
    tier_counts = triage["triage_tier"].value_counts().to_dict() if "triage_tier" in triage.columns else {}
    print(f"Candidate triage written: {out_table}; tier_counts={tier_counts}; seeds={len(seeds)}")


if __name__ == "__main__":
    main()
