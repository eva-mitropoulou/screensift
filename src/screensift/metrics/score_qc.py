from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS = {
    "cnnscore_min": 0.0,
    "cnnscore_max": 1.0,
    "cnnaffinity_min": -5.0,
    "cnnaffinity_max": 20.0,
    "gnina_affinity_positive_threshold": 0.0,
    "gnina_affinity_extreme_positive_threshold": 20.0,
    "unidock_suspicious_negative_threshold": -15.0,
    "unidock_extreme_negative_threshold": -20.0,
    "unidock_positive_threshold": 0.0,
    "ligand_efficiency_extreme_negative_threshold": -0.8,
}

GNINA_FLAGS = [
    "out_of_range_cnnscore",
    "suspicious_cnnaffinity_extreme",
    "positive_gnina_affinity",
    "extreme_positive_gnina_affinity",
]

UNIDOCK_FLAGS = [
    "suspicious_unidock_extreme_negative",
    "extreme_unidock_negative",
    "suspicious_unidock_positive",
    "suspicious_ligand_efficiency_extreme",
]


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def add_score_anomaly_flags(df: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    """Add GNINA, Uni-Dock, and ligand-efficiency anomaly flags."""
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    out = df.copy()

    cnnscore = _numeric(out, "CNNscore")
    cnnaffinity = _numeric(out, "CNNaffinity")
    gnina_affinity = _numeric(out, "gnina_affinity")
    unidock = _numeric(out, "unidock_best_score")
    heavy_atoms = _numeric(out, "heavy_atoms")

    out["ligand_efficiency"] = unidock / heavy_atoms.where(heavy_atoms > 0)

    out["out_of_range_cnnscore"] = (cnnscore < limits["cnnscore_min"]) | (cnnscore > limits["cnnscore_max"])
    out["suspicious_cnnaffinity_extreme"] = (
        (cnnaffinity < limits["cnnaffinity_min"]) | (cnnaffinity > limits["cnnaffinity_max"])
    )
    out["positive_gnina_affinity"] = gnina_affinity > limits["gnina_affinity_positive_threshold"]
    out["extreme_positive_gnina_affinity"] = (
        gnina_affinity > limits["gnina_affinity_extreme_positive_threshold"]
    )
    out["suspicious_unidock_extreme_negative"] = (
        unidock < limits["unidock_suspicious_negative_threshold"]
    )
    out["extreme_unidock_negative"] = unidock < limits["unidock_extreme_negative_threshold"]
    out["suspicious_unidock_positive"] = unidock > limits["unidock_positive_threshold"]
    out["suspicious_ligand_efficiency_extreme"] = (
        pd.to_numeric(out["ligand_efficiency"], errors="coerce")
        < limits["ligand_efficiency_extreme_negative_threshold"]
    )

    for flag in GNINA_FLAGS + UNIDOCK_FLAGS:
        out[flag] = out[flag].fillna(False).astype(bool)

    out["anomaly_flags"] = out.apply(
        lambda row: ";".join(flag for flag in GNINA_FLAGS + UNIDOCK_FLAGS if bool(row[flag])),
        axis=1,
    )
    out["has_any_score_anomaly"] = out["anomaly_flags"].astype(str).str.len() > 0
    return out


def anomaly_counts(df: pd.DataFrame) -> dict[str, int]:
    """Count anomaly flags that are present in a scored population table."""
    return {flag: int(df[flag].sum()) for flag in GNINA_FLAGS + UNIDOCK_FLAGS if flag in df.columns}
