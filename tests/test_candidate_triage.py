from pathlib import Path

import pandas as pd


from screensift.validation.triage_candidates import assign_tiers, build_seed_table, run_triage  # noqa: E402


def _row(
    ligand_id: str,
    activity: str = "active",
    action: str = "credible_structure_added_case",
    categories: str = "novel_sav_tanimoto_lt_0_30;low_similarity_active_tanimoto_lt_0_30",
    sim: float = 0.20,
    anomalies: str = "",
    clean_excluded: bool = False,
    interactions: int = 30,
) -> dict:
    return {
        "ligand_id": ligand_id,
        "activity_label": activity,
        "canonical_smiles": "CCO",
        "inspection_categories": categories,
        "manual_priority": 1,
        "ecfp4_active_similarity": sim,
        "unidock_best_score": -8.0,
        "CNNscore": 0.8,
        "CNNaffinity": 6.5,
        "gnina_affinity": -8.0,
        "anomaly_flags": anomalies,
        "clean_population_excluded": clean_excluded,
        "pose_found": True,
        "interaction_analysis_success": True,
        "n_total_interactions": interactions,
        "pose_interpretation": "test",
        "recommended_action": action,
        "best_structure_method": "gnina_cnnscore",
        "SAV": 0.8,
    }


def test_credible_active_low_similarity_case_becomes_tier_a() -> None:
    df = pd.DataFrame([_row(str(i), sim=0.20 + i * 0.01, interactions=50 - i) for i in range(6)])

    triage = assign_tiers(df)

    assert triage.loc[triage["ligand_id"].eq("0"), "triage_tier"].iloc[0] == "A_analog_seed"
    assert triage.loc[triage["ligand_id"].eq("0"), "evidence_bucket"].iloc[0] == "novelty_pose_review"
    assert triage.loc[triage["ligand_id"].eq("0"), "primary_evidence"].iloc[0] == "structure_score_low_similarity"


def test_score_anomaly_is_not_tier_a() -> None:
    df = pd.DataFrame(
        [
            _row("good"),
            _row("bad", anomalies="extreme_unidock_negative;suspicious_ligand_efficiency_extreme", interactions=100),
        ]
    )

    triage = assign_tiers(df)

    assert triage.loc[triage["ligand_id"].eq("bad"), "triage_tier"].iloc[0] != "A_analog_seed"
    assert triage.loc[triage["ligand_id"].eq("bad"), "evidence_bucket"].iloc[0] == "anomaly_review"


def test_inactive_false_positive_goes_to_failure_analysis() -> None:
    df = pd.DataFrame(
        [
            _row("good"),
            _row(
                "fp",
                activity="inactive",
                action="false_positive_failure_case",
                categories="consensus_inactive_false_positive",
            ),
        ]
    )

    triage = assign_tiers(df)

    assert triage.loc[triage["ligand_id"].eq("fp"), "triage_tier"].iloc[0] == "D_failure_analysis"
    assert triage.loc[triage["ligand_id"].eq("fp"), "recommended_next_step"].iloc[0] == "failure_analysis"
    assert triage.loc[triage["ligand_id"].eq("fp"), "evidence_bucket"].iloc[0] == "failure_analysis"


def test_false_negative_is_not_optimized() -> None:
    df = pd.DataFrame(
        [
            _row("good"),
            _row("fn", action="false_negative_failure_case", categories="active_false_negative"),
        ]
    )

    triage = assign_tiers(df)

    assert triage.loc[triage["ligand_id"].eq("fn"), "recommended_next_step"].iloc[0] == "failure_analysis"


def test_similarity_and_structure_support_are_bucketed_separately() -> None:
    df = pd.DataFrame(
        [
            _row(
                "analog",
                action="inspect_manually",
                categories="similarity_driven_analog_bias_hit",
                sim=0.82,
                interactions=0,
            ),
            _row("consensus", sim=0.82, interactions=30),
            _row("structure", sim=0.42, interactions=30),
        ]
    )

    triage = assign_tiers(df)

    assert triage.loc[triage["ligand_id"].eq("analog"), "evidence_bucket"].iloc[0] == "analog_supported"
    assert triage.loc[triage["ligand_id"].eq("consensus"), "evidence_bucket"].iloc[0] == "consensus_supported"
    assert triage.loc[triage["ligand_id"].eq("structure"), "evidence_bucket"].iloc[0] == "novelty_pose_review"
    assert "structure_supported" in triage.loc[triage["ligand_id"].eq("consensus"), "evidence_tags"].iloc[0]


def test_tier_a_seed_file_is_created(tmp_path: Path) -> None:
    summary = pd.DataFrame([_row(str(i), sim=0.20 + i * 0.01, interactions=40 - i) for i in range(7)])
    interactions = summary[["ligand_id", "n_total_interactions", "interaction_analysis_success"]].copy()
    flags = pd.DataFrame({"ligand_id": summary["ligand_id"], "plausibility_flags": ["credible_low_similarity_active"] * len(summary)})
    shortlist = summary[["ligand_id", "best_structure_method", "SAV", "clean_population_excluded"]].copy()
    summary_path = tmp_path / "summary.csv"
    interactions_path = tmp_path / "interactions.csv"
    flags_path = tmp_path / "flags.csv"
    shortlist_path = tmp_path / "shortlist.csv"
    summary.to_csv(summary_path, index=False)
    interactions.to_csv(interactions_path, index=False)
    flags.to_csv(flags_path, index=False)
    shortlist.to_csv(shortlist_path, index=False)

    triage, seeds = run_triage(summary_path, interactions_path, flags_path, shortlist_path)
    seeds_path = tmp_path / "mapk1_phase1_step10_seed_ligands.csv"
    seeds.to_csv(seeds_path, index=False)

    assert seeds_path.exists()
    assert not seeds.empty
    assert set(seeds["triage_tier"]).issubset({"A_analog_seed", "B_backup_seed"})
    assert "A_analog_seed" in set(triage["triage_tier"])
    assert {"evidence_bucket", "evidence_tags", "primary_evidence", "candidate_use_case"}.issubset(triage.columns)
