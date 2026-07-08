from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from screensift.common.io import load_yaml
from screensift.metrics.score_qc import add_score_anomaly_flags


DEFAULT_CLEAN_EXCLUSION_FLAGS = [
    "extreme_unidock_negative",
    "suspicious_unidock_extreme_negative",
    "suspicious_ligand_efficiency_extreme",
    "extreme_positive_gnina_affinity",
    "out_of_range_cnnscore",
    "suspicious_cnnaffinity_extreme",
]


def _split_flags(value: Any) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {part.strip() for part in str(value).split(";") if part.strip()}


def _merge_anomaly_table(score_population: pd.DataFrame, anomaly_path: Path) -> pd.DataFrame:
    anomalies = pd.read_csv(anomaly_path, dtype={"ligand_id": str}, low_memory=False)
    if "ligand_id" not in anomalies.columns or "anomaly_flags" not in anomalies.columns:
        raise ValueError(f"Anomaly table missing ligand_id/anomaly_flags columns: {anomaly_path}")
    merged_flags = (
        anomalies.groupby("ligand_id")["anomaly_flags"]
        .apply(lambda values: ";".join(sorted(set().union(*[_split_flags(value) for value in values]))))
        .reset_index()
    )
    out = score_population.drop(columns=["anomaly_flags", "has_any_score_anomaly"], errors="ignore").merge(
        merged_flags,
        on="ligand_id",
        how="left",
    )
    out["anomaly_flags"] = out["anomaly_flags"].fillna("")
    return out


def load_or_reconstruct_anomaly_flags(
    score_population: pd.DataFrame,
    tables_dir: str | Path,
    metrics_config: str | Path,
    clean_exclusion_flags: list[str] | None = None,
) -> pd.DataFrame:
    """Merge Step 6 anomalies if present, otherwise reconstruct from score thresholds."""
    out = score_population.copy()
    out["ligand_id"] = out["ligand_id"].astype(str)
    anomaly_path = Path(tables_dir) / "mapk1_phase1_all_score_anomalies.csv"
    if anomaly_path.exists():
        out = _merge_anomaly_table(out, anomaly_path)
    else:
        config = load_yaml(metrics_config)
        out = add_score_anomaly_flags(out, config.get("anomaly_thresholds", {}))

    flags = sorted(set().union(*out["anomaly_flags"].map(_split_flags))) if "anomaly_flags" in out.columns else []
    for flag in flags:
        out[flag] = out["anomaly_flags"].map(lambda value, name=flag: name in _split_flags(value))
    for flag in clean_exclusion_flags or DEFAULT_CLEAN_EXCLUSION_FLAGS:
        if flag not in out.columns:
            out[flag] = False

    out["has_any_score_anomaly"] = out["anomaly_flags"].fillna("").astype(str).str.len() > 0
    excluded_flags = clean_exclusion_flags or DEFAULT_CLEAN_EXCLUSION_FLAGS
    out["clean_population_excluded"] = out[excluded_flags].any(axis=1)
    return out
