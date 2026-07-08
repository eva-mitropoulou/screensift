from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ACTIVE_STRINGS = {"active", "actives", "1", "true", "yes", "y"}
INACTIVE_STRINGS = {"inactive", "inactives", "0", "false", "no", "n"}


def normalize_activity_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with an integer is_active column."""
    out = df.copy()
    label_col = None
    for candidate in ["activity_label", "label", "active", "is_active"]:
        if candidate in out.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError("No activity label column found. Expected activity_label, label, active, or is_active.")

    values = out[label_col]
    if pd.api.types.is_bool_dtype(values):
        out["is_active"] = values.astype(int)
        return out
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        out["is_active"] = numeric.where(numeric.isin([0, 1])).astype("Int64")
        return out

    def convert(value: Any) -> int | pd.NA:
        text = str(value).strip().lower()
        if text in ACTIVE_STRINGS:
            return 1
        if text in INACTIVE_STRINGS:
            return 0
        return pd.NA

    out["is_active"] = values.map(convert).astype("Int64")
    return out


def make_rank_score(series: pd.Series, direction: str) -> pd.Series:
    """Convert a raw score to a higher-is-better rank score."""
    numeric = pd.to_numeric(series, errors="coerce")
    normalized_direction = direction.strip().lower()
    if normalized_direction == "lower":
        return -numeric
    if normalized_direction == "higher":
        return numeric
    raise ValueError(f"Unsupported score direction: {direction!r}")


def _valid_arrays(y_true: Any, y_score: Any) -> tuple[np.ndarray, np.ndarray]:
    y = pd.Series(y_true).astype(float).to_numpy()
    score = pd.to_numeric(pd.Series(y_score), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(score)
    return y[mask].astype(int), score[mask]


def _has_two_classes(y_true: np.ndarray) -> bool:
    return len(np.unique(y_true)) == 2


def compute_roc_auc(y_true: Any, y_score: Any) -> float:
    y, score = _valid_arrays(y_true, y_score)
    if len(y) == 0 or not _has_two_classes(y):
        return float("nan")
    return float(roc_auc_score(y, score))


def compute_pr_auc(y_true: Any, y_score: Any) -> float:
    """Compute average precision, the early-recognition-friendly PR summary."""
    y, score = _valid_arrays(y_true, y_score)
    if len(y) == 0 or not _has_two_classes(y):
        return float("nan")
    return float(average_precision_score(y, score))


def compute_enrichment_factor(y_true: Any, y_score: Any, top_fraction: float) -> float:
    y, score = _valid_arrays(y_true, y_score)
    if len(y) == 0:
        return float("nan")
    total_actives = int(y.sum())
    if total_actives == 0:
        return float("nan")
    top_n = max(1, int(np.ceil(len(y) * top_fraction)))
    order = np.argsort(-score, kind="mergesort")
    top = y[order[:top_n]]
    overall_active_fraction = total_actives / len(y)
    top_active_fraction = float(top.sum()) / top_n
    return float(top_active_fraction / overall_active_fraction)


def compute_topk_recovery(y_true: Any, y_score: Any, k: int) -> dict[str, float | int]:
    y, score = _valid_arrays(y_true, y_score)
    if len(y) == 0:
        return {"topk_actives": 0, "topk_recovery_fraction": float("nan")}
    top_n = min(max(1, int(k)), len(y))
    order = np.argsort(-score, kind="mergesort")
    top_actives = int(y[order[:top_n]].sum())
    total_actives = int(y.sum())
    fraction = float(top_actives / total_actives) if total_actives else float("nan")
    return {"topk_actives": top_actives, "topk_recovery_fraction": fraction}


def compute_bedroc(y_true: Any, y_score: Any, alpha: float = 20.0) -> float:
    """Return NaN until a verified BEDROC implementation is added.

    BEDROC is useful for early recognition, but an incorrect implementation is
    worse than omitting it. The Step 6 report documents this explicitly.
    """
    _ = (y_true, y_score, alpha)
    return float("nan")


def bootstrap_metric_ci(
    y_true: Any,
    y_score: Any,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Compute a stratified bootstrap CI while preserving class balance."""
    y, score = _valid_arrays(y_true, y_score)
    observed = float(metric_fn(y, score)) if len(y) else float("nan")
    if len(y) == 0 or not _has_two_classes(y) or n_boot <= 0:
        return {"metric": observed, "ci_low": float("nan"), "ci_high": float("nan"), "n_boot_valid": 0}

    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return {"metric": observed, "ci_low": float("nan"), "ci_high": float("nan"), "n_boot_valid": 0}

    values: list[float] = []
    for _ in range(int(n_boot)):
        sample_pos = rng.choice(pos, size=len(pos), replace=True)
        sample_neg = rng.choice(neg, size=len(neg), replace=True)
        sample_idx = np.concatenate([sample_pos, sample_neg])
        sample_y = y[sample_idx]
        sample_score = score[sample_idx]
        if not _has_two_classes(sample_y):
            continue
        value = float(metric_fn(sample_y, sample_score))
        if np.isfinite(value):
            values.append(value)

    if not values:
        return {"metric": observed, "ci_low": float("nan"), "ci_high": float("nan"), "n_boot_valid": 0}

    alpha = 1.0 - confidence
    low = float(np.quantile(values, alpha / 2.0))
    high = float(np.quantile(values, 1.0 - alpha / 2.0))
    return {"metric": observed, "ci_low": low, "ci_high": high, "n_boot_valid": len(values)}
