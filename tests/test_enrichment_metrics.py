from pathlib import Path

import numpy as np
import pandas as pd


from screensift.metrics.enrichment import (
    bootstrap_metric_ci,
    compute_enrichment_factor,
    compute_pr_auc,
    compute_roc_auc,
    compute_topk_recovery,
    make_rank_score,
)


def test_enrichment_factor_known_example() -> None:
    y = pd.Series([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    score = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    ef20 = compute_enrichment_factor(y, score, 0.20)

    assert ef20 == 2.5


def test_roc_pr_and_lower_direction_are_valid() -> None:
    y = pd.Series([1, 1, 0, 0])
    lower_is_better = pd.Series([-9.0, -8.0, -3.0, -2.0])
    rank_score = make_rank_score(lower_is_better, "lower")

    assert compute_roc_auc(y, rank_score) == 1.0
    assert compute_pr_auc(y, rank_score) == 1.0
    assert rank_score.iloc[0] > rank_score.iloc[-1]


def test_topk_recovery_and_bootstrap_ci() -> None:
    y = pd.Series([1, 0, 1, 0, 0, 1])
    score = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

    recovery = compute_topk_recovery(y, score, 3)
    ci = bootstrap_metric_ci(y, score, compute_roc_auc, n_boot=50, seed=42)

    assert recovery["topk_actives"] == 2
    assert np.isfinite(ci["metric"])
    assert np.isfinite(ci["ci_low"])
    assert np.isfinite(ci["ci_high"])
    assert ci["n_boot_valid"] > 0
