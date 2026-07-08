from __future__ import annotations

import pandas as pd

from screensift.metrics.enrichment import make_rank_score


def percentile_rank_score(df: pd.DataFrame, score_col: str, direction: str) -> pd.Series:
    """Return percentile scores where higher is always better."""
    rank_score = make_rank_score(df[score_col], direction)
    return rank_score.rank(method="average", pct=True)


def reciprocal_rank_fusion(df: pd.DataFrame, rank_cols: list[str], k: int = 60) -> pd.Series:
    """Combine higher-is-better score columns using reciprocal rank fusion."""
    if not rank_cols:
        raise ValueError("At least one rank column is required for reciprocal rank fusion.")
    fused = pd.Series(0.0, index=df.index)
    for col in rank_cols:
        ranks = pd.to_numeric(df[col], errors="coerce").rank(method="average", ascending=False)
        fused = fused.add(1.0 / (k + ranks), fill_value=0.0)
    return fused


def mean_percentile_fusion(df: pd.DataFrame, percentile_cols: list[str]) -> pd.Series:
    """Primary fusion score: mean of higher-is-better percentile columns."""
    if not percentile_cols:
        raise ValueError("At least one percentile column is required for fusion.")
    return df[percentile_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
