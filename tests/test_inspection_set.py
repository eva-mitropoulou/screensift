from pathlib import Path

import pandas as pd


from screensift.validation.build_inspection_set import build_inspection_shortlist  # noqa: E402


def _write_minimal_tables(tables_dir: Path) -> None:
    base = pd.DataFrame(
        [
            {"ligand_id": "A1", "activity_label": "active", "canonical_smiles": "CCO", "ecfp4_active_similarity": 0.20},
            {"ligand_id": "A2", "activity_label": "active", "canonical_smiles": "CCC", "ecfp4_active_similarity": 0.25},
            {"ligand_id": "I1", "activity_label": "inactive", "canonical_smiles": "CCCC", "ecfp4_active_similarity": 0.75},
            {"ligand_id": "I2", "activity_label": "inactive", "canonical_smiles": "CCN", "ecfp4_active_similarity": 0.40},
        ]
    )
    base.to_csv(tables_dir / "mapk1_phase1_full_population_with_ecfp4_and_anomalies.csv", index=False)
    pd.DataFrame(
        [
            {
                "ligand_id": "A1",
                "activity_label": "active",
                "all_active_loo_ecfp4_analog_neighborhood": 0.20,
                "SAV": 0.80,
                "best_structure_percentile": 0.90,
            }
        ]
    ).to_csv(tables_dir / "mapk1_phase1_structure_added_value_cases_full.csv", index=False)
    pd.DataFrame(
        [{"ligand_id": "A2", "activity_label": "active", "ecfp4_active_similarity": 0.25, "best_structure_rank": 10}]
    ).to_csv(tables_dir / "mapk1_phase1_novel_actives_full.csv", index=False)
    pd.DataFrame(
        [{"ligand_id": "I1", "activity_label": "inactive", "ecfp4_active_similarity": 0.75, "n_high_rank_methods": 3}]
    ).to_csv(tables_dir / "mapk1_phase1_false_positive_consensus_full.csv", index=False)
    pd.DataFrame([{"ligand_id": "A2", "activity_label": "active", "ecfp4_active_similarity": 0.25}]).to_csv(
        tables_dir / "mapk1_phase1_active_false_negatives_full.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "ligand_id": "I2",
                "activity_label": "inactive",
                "unidock_best_score": -29.0,
                "anomaly_flags": "extreme_unidock_negative;suspicious_ligand_efficiency_extreme",
                "rank_unidock_best": 1,
            }
        ]
    ).to_csv(tables_dir / "mapk1_phase1_score_anomalies_in_top_hits.csv", index=False)
    pd.DataFrame(
        [{"ligand_id": "I1", "activity_label": "inactive", "ecfp4_active_similarity": 0.75, "best_structure_rank": 5}]
    ).to_csv(tables_dir / "mapk1_phase1_similarity_driven_hits_full.csv", index=False)


def _config(max_total: int = 100, max_per_category: int = 15) -> dict:
    return {
        "inspection_set_limits": {"max_total_ligands": max_total, "max_per_category": max_per_category},
        "categories": {
            "score_anomaly_top_hits": {
                "include_flags": [
                    "extreme_unidock_negative",
                    "suspicious_ligand_efficiency_extreme",
                    "extreme_positive_gnina_affinity",
                ]
            }
        },
        "baseline_policy": {
            "allow_pdb_native_baseline_for_selection": False,
            "allow_ccd_native_baseline_for_reference": True,
        },
    }


def test_inspection_set_selects_required_categories(tmp_path: Path) -> None:
    _write_minimal_tables(tmp_path)

    shortlist, manifest = build_inspection_shortlist(tmp_path, _config())
    categories = ";".join(shortlist["inspection_categories"])

    assert manifest["pdb_native_baseline_used_for_selection"] is False
    assert "novel_sav_tanimoto_lt_0_30" in categories
    assert "low_similarity_active_tanimoto_lt_0_30" in categories
    assert "consensus_inactive_false_positive" in categories
    assert "active_false_negative" in categories
    assert "extreme_unidock_negative_top_hit" in categories
    assert "similarity_driven_analog_bias_hit" in categories


def test_category_caps_are_respected(tmp_path: Path) -> None:
    _write_minimal_tables(tmp_path)
    extra = pd.DataFrame(
        [{"ligand_id": f"A{i}", "activity_label": "active", "ecfp4_active_similarity": 0.1, "best_structure_rank": i} for i in range(10)]
    )
    extra.to_csv(tmp_path / "mapk1_phase1_novel_actives_full.csv", index=False)

    shortlist, manifest = build_inspection_shortlist(tmp_path, _config(max_total=3, max_per_category=1))

    assert len(shortlist) <= 3
    assert all(count <= 3 for count in manifest["counts_by_category"].values())
