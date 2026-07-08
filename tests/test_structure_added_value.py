from pathlib import Path

import pandas as pd


from screensift.metrics.structure_added_value import (
    compute_structure_added_value,
    percentile_rank,
    structure_added_value_cases,
)


def test_sav_positive_when_structure_percentile_is_higher() -> None:
    df = pd.DataFrame({"ecfp4_pct": [0.2], "structure_pct": [0.8]})

    result = compute_structure_added_value(df, "ecfp4_pct", "structure_pct")

    assert result.loc[0, "SAV"] > 0


def test_percentile_rank_lower_direction() -> None:
    score = pd.Series([-9.0, -5.0, -1.0])

    pct = percentile_rank(score, "lower")

    assert pct.iloc[0] == 1.0


def test_novel_sav_counts_only_active_ligands_below_threshold() -> None:
    df = pd.DataFrame(
        [
            {"ligand_id": "a1", "activity_label": "active", "is_active": 1, "canonical_smiles": "CCO", "ecfp4": 0.2, "unidock_best_score": -10.0, "CNNscore": 0.1, "CNNaffinity": 3.0, "gnina_affinity": -7.0},
            {"ligand_id": "a2", "activity_label": "active", "is_active": 1, "canonical_smiles": "CCN", "ecfp4": 0.9, "unidock_best_score": -9.0, "CNNscore": 0.1, "CNNaffinity": 3.0, "gnina_affinity": -7.0},
            {"ligand_id": "i1", "activity_label": "inactive", "is_active": 0, "canonical_smiles": "CCC", "ecfp4": 0.1, "unidock_best_score": -1.0, "CNNscore": 0.1, "CNNaffinity": 3.0, "gnina_affinity": -7.0},
        ]
    )
    cases, summary = structure_added_value_cases(
        df,
        "ecfp4",
        {"unidock_best": ("unidock_best_score", "lower")},
        top_k_values=[1, 2],
        novelty_thresholds=[0.70, 0.50, 0.30],
    )

    assert not cases.empty
    novel_1 = summary[(summary["metric"] == "Novel-SAV@1") & (summary["threshold"] == 0.70)].iloc[0]
    assert int(novel_1["active_count"]) == 1
