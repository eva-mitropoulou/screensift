from pathlib import Path

import pandas as pd


from screensift.metrics.evaluate_phase1_scores import add_rank_and_fusion_columns, build_population


def test_cnn_vs_is_derived_from_cnnscore_and_cnnaffinity(tmp_path: Path) -> None:
    gnina = pd.DataFrame(
        {
            "ligand_id": ["lig1", "lig2"],
            "activity_label": ["active", "inactive"],
            "cnnscore": [0.5, 0.25],
            "cnnaffinity": [6.0, 4.0],
            "affinity": [-8.0, -6.0],
        }
    )
    unidock = pd.DataFrame(
        {
            "ligand_id": ["lig1", "lig2"],
            "best_score": [-9.0, -5.0],
            "canonical_smiles": ["CCO", "CCC"],
        }
    )
    gnina_path = tmp_path / "gnina.csv"
    unidock_path = tmp_path / "unidock.csv"
    gnina.to_csv(gnina_path, index=False)
    unidock.to_csv(unidock_path, index=False)

    population = build_population(gnina_path, unidock_path)
    ranked = add_rank_and_fusion_columns(
        population,
        {
            "score_methods": {
                "unidock_best": {"direction": "lower"},
                "gnina_cnnscore": {"direction": "higher"},
                "gnina_cnnaffinity": {"direction": "higher"},
                "gnina_cnn_vs": {"direction": "higher"},
                "gnina_affinity": {"direction": "lower"},
            },
            "rank_fusion_methods": [],
        },
    )

    assert ranked.loc[ranked["ligand_id"].eq("lig1"), "CNN_VS"].iloc[0] == 3.0
    assert ranked.loc[ranked["ligand_id"].eq("lig2"), "CNN_VS"].iloc[0] == 1.0
    assert "rankscore_gnina_cnn_vs" in ranked.columns
    assert "pct_gnina_cnn_vs" in ranked.columns
    assert ranked.sort_values("rankscore_gnina_cnn_vs", ascending=False)["ligand_id"].iloc[0] == "lig1"
