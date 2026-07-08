from pathlib import Path

import pandas as pd


from screensift.metrics.rank_fusion import mean_percentile_fusion, percentile_rank_score, reciprocal_rank_fusion


def test_percentile_rank_score_handles_lower_direction() -> None:
    df = pd.DataFrame({"score": [-10.0, -5.0, -1.0]})

    pct = percentile_rank_score(df, "score", "lower")

    assert pct.iloc[0] == 1.0
    assert pct.iloc[-1] < pct.iloc[0]


def test_mean_percentile_fusion_ranks_expected_ligand_highest() -> None:
    df = pd.DataFrame(
        {
            "pct_a": [1.0, 0.4, 0.2],
            "pct_b": [0.9, 0.5, 0.1],
        },
        index=["best", "middle", "last"],
    )

    fused = mean_percentile_fusion(df, ["pct_a", "pct_b"])

    assert fused.idxmax() == "best"


def test_reciprocal_rank_fusion_prefers_consistent_top_rank() -> None:
    df = pd.DataFrame({"score_a": [0.9, 0.8, 0.1], "score_b": [0.8, 0.2, 0.1]}, index=["a", "b", "c"])

    fused = reciprocal_rank_fusion(df, ["score_a", "score_b"])

    assert fused.idxmax() == "a"
