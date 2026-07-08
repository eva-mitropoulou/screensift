from __future__ import annotations

import math
from typing import Any

import pandas as pd


def percentile_rank(series: pd.Series, direction: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    rank_score = -numeric if direction == "lower" else numeric
    return rank_score.rank(method="average", pct=True)


def compute_best_structure_percentile(
    df: pd.DataFrame,
    structure_method_cols: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    out = df.copy()
    percentile_cols: list[str] = []
    for method, (col, direction) in structure_method_cols.items():
        if col not in out.columns:
            continue
        pct_col = f"pct_{method}"
        out[pct_col] = percentile_rank(out[col], direction)
        percentile_cols.append(pct_col)
    if not percentile_cols:
        out["best_structure_percentile"] = pd.NA
        out["best_structure_method"] = pd.NA
        return out

    out["best_structure_percentile"] = out[percentile_cols].max(axis=1)

    def best_method(row: pd.Series) -> str | None:
        values = {col: row[col] for col in percentile_cols if pd.notna(row[col])}
        if not values:
            return None
        best_col = max(values, key=values.get)
        return best_col.removeprefix("pct_")

    out["best_structure_method"] = out.apply(best_method, axis=1)
    return out


def compute_structure_added_value(
    df: pd.DataFrame,
    ecfp4_percentile_col: str,
    structure_percentile_col: str,
) -> pd.DataFrame:
    out = df.copy()
    out["SAV"] = pd.to_numeric(out[structure_percentile_col], errors="coerce") - pd.to_numeric(
        out[ecfp4_percentile_col], errors="coerce"
    )
    return out


def structure_added_value_cases(
    df: pd.DataFrame,
    ecfp4_similarity_col: str,
    structure_method_cols: dict[str, tuple[str, str]],
    top_k_values: list[int],
    novelty_thresholds: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["ecfp4_percentile"] = percentile_rank(out[ecfp4_similarity_col], "higher")
    out = compute_best_structure_percentile(out, structure_method_cols)
    out = compute_structure_added_value(out, "ecfp4_percentile", "best_structure_percentile")

    rows: list[dict[str, Any]] = []
    active = out[out["is_active"].eq(1)].copy()
    for k in top_k_values:
        structure_top = active["best_structure_percentile"].ge(1.0 - (k / len(out)))
        ecfp4_top = active["ecfp4_percentile"].ge(1.0 - (k / len(out)))
        sav_mask = structure_top & ~ecfp4_top
        rows.append({"metric": f"SAV@{k}", "threshold": pd.NA, "k": k, "active_count": int(sav_mask.sum())})
        for threshold in novelty_thresholds:
            novel_mask = pd.to_numeric(active[ecfp4_similarity_col], errors="coerce") < threshold
            rows.append(
                {
                    "metric": f"Novel-SAV@{k}",
                    "threshold": threshold,
                    "k": k,
                    "active_count": int((sav_mask & novel_mask).sum()),
                }
            )

    cases = active[active["SAV"].gt(0)].copy()
    cases["ecfp4_similarity"] = cases[ecfp4_similarity_col]
    cases["novelty_threshold_passed"] = cases[ecfp4_similarity_col].apply(
        lambda value: ";".join([f"<{threshold}" for threshold in novelty_thresholds if pd.notna(value) and value < threshold])
    )
    cases["reason"] = "active_ligand_ranked_better_by_structure_than_ecfp4"
    keep_cols = [
        "ligand_id",
        "activity_label",
        "canonical_smiles",
        "ecfp4_similarity",
        "ecfp4_percentile",
        "best_structure_method",
        "best_structure_percentile",
        "SAV",
        "unidock_best_score",
        "CNNscore",
        "CNNaffinity",
        "gnina_affinity",
        "anomaly_flags",
        "clean_population_excluded",
        "novelty_threshold_passed",
        "reason",
    ]
    return cases[[col for col in keep_cols if col in cases.columns]].sort_values("SAV", ascending=False), pd.DataFrame(rows)
