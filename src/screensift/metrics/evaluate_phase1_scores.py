from __future__ import annotations

import argparse
import math
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from screensift.common.io import ensure_dir, load_yaml
from screensift.common.logging_utils import setup_logger
from screensift.metrics.enrichment import (
    bootstrap_metric_ci,
    compute_bedroc,
    compute_enrichment_factor,
    compute_pr_auc,
    compute_roc_auc,
    compute_topk_recovery,
    make_rank_score,
    normalize_activity_labels,
)
from screensift.metrics.rank_fusion import mean_percentile_fusion, percentile_rank_score
from screensift.metrics.score_qc import GNINA_FLAGS, UNIDOCK_FLAGS, add_score_anomaly_flags, anomaly_counts
from screensift.viz.screening_plots import make_screening_plots


LOGGER = setup_logger(__name__)

METHOD_SCORE_COLUMNS = {
    "unidock_best": "unidock_best_score",
    "gnina_cnnscore": "CNNscore",
    "gnina_cnnaffinity": "CNNaffinity",
    "gnina_cnn_vs": "CNN_VS",
    "gnina_affinity": "gnina_affinity",
}

METHOD_RANK_COLUMNS = {
    "unidock_best": "rankscore_unidock_best",
    "gnina_cnnscore": "rankscore_gnina_cnnscore",
    "gnina_cnnaffinity": "rankscore_gnina_cnnaffinity",
    "gnina_cnn_vs": "rankscore_gnina_cnn_vs",
    "gnina_affinity": "rankscore_gnina_affinity",
}

METHOD_PERCENTILE_COLUMNS = {
    "unidock_best": "pct_unidock_best",
    "gnina_cnnscore": "pct_gnina_cnnscore",
    "gnina_cnnaffinity": "pct_gnina_cnnaffinity",
    "gnina_cnn_vs": "pct_gnina_cnn_vs",
    "gnina_affinity": "pct_gnina_affinity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MAPK1 phase 1 Uni-Dock/GNINA scores.")
    parser.add_argument("--metrics-config", default="configs/metrics.yml")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--reports-dir", default="results/reports")
    parser.add_argument("--out-prefix", default="mapk1_phase1")
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required score table not found: {path}")
    return pd.read_csv(path, dtype={"ligand_id": str}, low_memory=False)


def discover_unidock_table(tables_dir: Path, prefix: str) -> Path:
    candidates = sorted(tables_dir.glob(f"{prefix}_unidock*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No Uni-Dock phase1 score tables found under {tables_dir}")

    def score(path: Path) -> tuple[int, int]:
        name = path.name.lower()
        priority = 0
        if "best" in name:
            priority = 3
        elif "scores" in name:
            priority = 2
        try:
            rows = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
        except OSError:
            rows = 0
        return priority, rows

    selected = max(candidates, key=score)
    LOGGER.info("Selected Uni-Dock score table: %s", selected)
    return selected


def _coalesce(df: pd.DataFrame, cols: list[str], default: Any = pd.NA) -> pd.Series:
    existing = [col for col in cols if col in df.columns]
    if not existing:
        return pd.Series(default, index=df.index)
    out = df[existing[0]]
    for col in existing[1:]:
        out = out.combine_first(df[col])
    return out


def _standardize_gnina(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    for source, target in [
        ("cnnscore", "CNNscore"),
        ("gnina_cnnscore", "CNNscore"),
        ("cnnaffinity", "CNNaffinity"),
        ("gnina_cnnaffinity", "CNNaffinity"),
        ("cnn_vs", "CNN_VS"),
        ("CNN_VS_score", "CNN_VS"),
        ("gnina_cnn_vs", "CNN_VS"),
        ("affinity", "gnina_affinity"),
        ("GNINA_affinity", "gnina_affinity"),
    ]:
        if source in out.columns and target not in out.columns:
            rename[source] = target
    return out.rename(columns=rename)


def _standardize_unidock(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    for source in ["best_score", "unidock_score", "score"]:
        if source in out.columns and "unidock_best_score" not in out.columns:
            rename[source] = "unidock_best_score"
            break
    return out.rename(columns=rename)


def _find_join_key(left: pd.DataFrame, right: pd.DataFrame) -> str:
    for key in ["ligand_id", "molecule_id", "compound_id", "inchikey", "canonical_smiles", "smiles"]:
        if key in left.columns and key in right.columns:
            return key
    raise ValueError(
        "Could not merge GNINA and Uni-Dock tables. No stable join key found among "
        "ligand_id, molecule_id, compound_id, inchikey, canonical_smiles, smiles."
    )


def _compute_heavy_atoms_from_smiles(smiles: pd.Series) -> pd.Series:
    try:
        from rdkit import Chem
    except Exception as exc:
        LOGGER.warning("RDKit unavailable; heavy_atoms will remain missing: %s", exc)
        return pd.Series(np.nan, index=smiles.index)

    def count_atoms(value: Any) -> float:
        if pd.isna(value) or not str(value).strip():
            return float("nan")
        mol = Chem.MolFromSmiles(str(value))
        if mol is None:
            return float("nan")
        return float(mol.GetNumHeavyAtoms())

    return smiles.map(count_atoms)


def build_population(gnina_path: Path, unidock_path: Path) -> pd.DataFrame:
    gnina = _standardize_gnina(_read_csv(gnina_path))
    unidock = _standardize_unidock(_read_csv(unidock_path))

    if {"unidock_best_score", "CNNscore", "CNNaffinity", "gnina_affinity"}.issubset(gnina.columns):
        merged = gnina.copy()
        if "canonical_smiles" not in merged.columns and "ligand_id" in unidock.columns:
            merged = merged.merge(
                unidock[["ligand_id"] + [c for c in ["canonical_smiles", "inchikey", "heavy_atoms", "mol_wt", "logp"] if c in unidock.columns]],
                on="ligand_id",
                how="left",
            )
    else:
        join_key = _find_join_key(gnina, unidock)
        LOGGER.info("Merging GNINA and Uni-Dock tables on %s", join_key)
        merged = gnina.merge(unidock, on=join_key, how="inner", suffixes=("_gnina", "_unidock"))

    population = pd.DataFrame(index=merged.index)
    population["ligand_id"] = _coalesce(merged, ["ligand_id", "ligand_id_gnina", "ligand_id_unidock"]).astype(str)
    population["activity_label"] = _coalesce(
        merged,
        ["activity_label", "activity_label_gnina", "activity_label_unidock", "label", "label_gnina", "label_unidock"],
    )
    population["canonical_smiles"] = _coalesce(merged, ["canonical_smiles", "canonical_smiles_gnina", "canonical_smiles_unidock"])
    population["inchikey"] = _coalesce(merged, ["inchikey", "inchikey_gnina", "inchikey_unidock"])
    population["pdb_id"] = _coalesce(merged, ["pdb_id", "pdb_id_gnina", "best_pdb_id", "best_pdb_id_unidock"])
    population["unidock_best_score"] = pd.to_numeric(
        _coalesce(merged, ["unidock_best_score", "unidock_best_score_unidock", "best_score_unidock"]),
        errors="coerce",
    )
    population["CNNscore"] = pd.to_numeric(_coalesce(merged, ["CNNscore", "CNNscore_gnina"]), errors="coerce")
    population["CNNaffinity"] = pd.to_numeric(_coalesce(merged, ["CNNaffinity", "CNNaffinity_gnina"]), errors="coerce")
    population["CNN_VS"] = pd.to_numeric(
        _coalesce(merged, ["CNN_VS", "CNN_VS_gnina", "cnn_vs", "gnina_cnn_vs"]),
        errors="coerce",
    )
    population["CNN_VS"] = population["CNN_VS"].combine_first(population["CNNscore"] * population["CNNaffinity"])
    population["gnina_affinity"] = pd.to_numeric(
        _coalesce(merged, ["gnina_affinity", "gnina_affinity_gnina"]),
        errors="coerce",
    )

    for optional in ["heavy_atoms", "mol_wt", "logp"]:
        population[optional] = pd.to_numeric(_coalesce(merged, [optional, f"{optional}_gnina", f"{optional}_unidock"]), errors="coerce")

    if population["heavy_atoms"].isna().any() and population["canonical_smiles"].notna().any():
        computed = _compute_heavy_atoms_from_smiles(population["canonical_smiles"])
        population["heavy_atoms"] = population["heavy_atoms"].combine_first(computed)

    population = normalize_activity_labels(population)
    population = population[population["is_active"].notna()].copy()
    population["is_active"] = population["is_active"].astype(int)
    population = population.drop_duplicates(subset=["ligand_id"], keep="first").reset_index(drop=True)
    return population


def add_rank_and_fusion_columns(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    methods = config.get("score_methods", {})
    for method, score_col in METHOD_SCORE_COLUMNS.items():
        direction = methods.get(method, {}).get("direction", "higher")
        out[METHOD_RANK_COLUMNS[method]] = make_rank_score(out[score_col], direction)
        out[METHOD_PERCENTILE_COLUMNS[method]] = percentile_rank_score(out, score_col, direction)

    for fusion in config.get("rank_fusion_methods", []):
        name = fusion["name"]
        percentile_cols = [METHOD_PERCENTILE_COLUMNS[method] for method in fusion["methods"]]
        out[name] = mean_percentile_fusion(out, percentile_cols)
    return out


def _stable_seed(seed: int, *parts: str) -> int:
    text = "::".join(parts)
    return int(seed + zlib.crc32(text.encode("utf-8")) % 1_000_000)


def _metric_value(y: pd.Series, score: pd.Series, metric: str) -> float:
    if metric == "roc_auc":
        return compute_roc_auc(y, score)
    if metric == "pr_auc":
        return compute_pr_auc(y, score)
    if metric == "ef1":
        return compute_enrichment_factor(y, score, 0.01)
    if metric == "ef5":
        return compute_enrichment_factor(y, score, 0.05)
    raise ValueError(f"Unsupported bootstrap metric: {metric}")


def evaluate_method(
    df: pd.DataFrame,
    population_name: str,
    method: str,
    raw_score_col: str,
    ranking_col: str,
    higher_is_better: bool,
    n_boot: int,
    confidence: float,
    seed: int,
    notes: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    working = df[["is_active", ranking_col]].copy()
    working[ranking_col] = pd.to_numeric(working[ranking_col], errors="coerce")
    working = working.dropna(subset=["is_active", ranking_col])

    y = working["is_active"].astype(int)
    score = working[ranking_col].astype(float)
    n_total = int(len(working))
    n_active = int(y.sum())
    n_inactive = int(n_total - n_active)

    row: dict[str, Any] = {
        "population": population_name,
        "method": method,
        "n_total": n_total,
        "n_active": n_active,
        "n_inactive": n_inactive,
        "score_col": raw_score_col,
        "ranking_col": ranking_col,
        "higher_is_better": higher_is_better,
        "roc_auc": compute_roc_auc(y, score),
        "pr_auc": compute_pr_auc(y, score),
        "ef1": compute_enrichment_factor(y, score, 0.01),
        "ef5": compute_enrichment_factor(y, score, 0.05),
        "ef10": compute_enrichment_factor(y, score, 0.10),
        "bedroc_alpha20": compute_bedroc(y, score, alpha=20.0),
        "notes": notes,
    }
    ci_rows: list[dict[str, Any]] = []
    for metric_name in ["roc_auc", "pr_auc", "ef1", "ef5"]:
        ci = bootstrap_metric_ci(
            y,
            score,
            lambda yy, ss, metric=metric_name: _metric_value(pd.Series(yy), pd.Series(ss), metric),
            n_boot=n_boot,
            seed=_stable_seed(seed, population_name, method, metric_name),
            confidence=confidence,
        )
        row[f"{metric_name}_low"] = ci["ci_low"]
        row[f"{metric_name}_high"] = ci["ci_high"]
        row[f"{metric_name}_n_boot_valid"] = ci["n_boot_valid"]
        ci_rows.append(
            {
                "population": population_name,
                "method": method,
                "metric": metric_name,
                "value": ci["metric"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "n_boot_valid": ci["n_boot_valid"],
            }
        )

    for k in [50, 100, 250]:
        recovery = compute_topk_recovery(y, score, k)
        row[f"top{k}_actives"] = recovery["topk_actives"]
        row[f"top{k}_active_recovery_fraction"] = recovery["topk_recovery_fraction"]

    return row, ci_rows


def method_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    methods = config.get("score_methods", {})
    specs: list[dict[str, Any]] = []
    for method, score_col in METHOD_SCORE_COLUMNS.items():
        direction = methods.get(method, {}).get("direction", "higher")
        specs.append(
            {
                "method": method,
                "raw_score_col": score_col,
                "ranking_col": METHOD_RANK_COLUMNS[method],
                "higher_is_better": direction == "higher",
            }
        )
    for fusion in config.get("rank_fusion_methods", []):
        specs.append(
            {
                "method": fusion["name"],
                "raw_score_col": fusion["name"],
                "ranking_col": fusion["name"],
                "higher_is_better": True,
            }
        )
    return specs


def evaluate_population(
    df: pd.DataFrame,
    population_name: str,
    specs: list[dict[str, Any]],
    config: dict[str, Any],
    notes: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_boot = int(config.get("bootstrap_iterations", 1000))
    confidence = float(config.get("bootstrap_confidence", 0.95))
    seed = int(config.get("random_seed", 42))
    rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec["ranking_col"] not in df.columns:
            continue
        row, ci = evaluate_method(
            df,
            population_name,
            spec["method"],
            spec["raw_score_col"],
            spec["ranking_col"],
            spec["higher_is_better"],
            n_boot,
            confidence,
            seed,
            notes=notes,
        )
        rows.append(row)
        ci_rows.extend(ci)
    return pd.DataFrame(rows), pd.DataFrame(ci_rows)


def build_qc_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"section": "population", "item": "n_total", "value": len(df)},
        {"section": "population", "item": "n_active", "value": int(df["is_active"].sum())},
        {"section": "population", "item": "n_inactive", "value": int((df["is_active"] == 0).sum())},
        {"section": "population", "item": "activity_prevalence", "value": float(df["is_active"].mean())},
        {"section": "population", "item": "duplicate_ligand_ids", "value": int(df["ligand_id"].duplicated().sum())},
    ]
    for col in ["unidock_best_score", "CNNscore", "CNNaffinity", "CNN_VS", "gnina_affinity"]:
        score = pd.to_numeric(df[col], errors="coerce")
        rows.extend(
            [
                {"section": "score", "item": f"{col}_missing", "value": int(score.isna().sum())},
                {"section": "score", "item": f"{col}_nonfinite", "value": int((~np.isfinite(score.dropna())).sum())},
                {"section": "score", "item": f"{col}_min", "value": float(score.min())},
                {"section": "score", "item": f"{col}_median", "value": float(score.median())},
                {"section": "score", "item": f"{col}_mean", "value": float(score.mean())},
                {"section": "score", "item": f"{col}_max", "value": float(score.max())},
            ]
        )
    for flag, count in anomaly_counts(df).items():
        rows.append({"section": "anomaly", "item": flag, "value": count})
    rows.append({"section": "anomaly", "item": "has_any_score_anomaly", "value": int(df["has_any_score_anomaly"].sum())})
    return pd.DataFrame(rows)


def anomaly_table(df: pd.DataFrame, flags: list[str]) -> pd.DataFrame:
    cols = [
        "ligand_id",
        "activity_label",
        "canonical_smiles",
        "heavy_atoms",
        "mol_wt",
        "logp",
        "unidock_best_score",
        "ligand_efficiency",
        "CNNscore",
        "CNNaffinity",
        "CNN_VS",
        "gnina_affinity",
        "anomaly_flags",
    ]
    mask = df[flags].any(axis=1) if flags else pd.Series(False, index=df.index)
    return df.loc[mask, [col for col in cols if col in df.columns]].copy()


def build_top100_by_method(df: pd.DataFrame, specs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    keep_cols = [
        "ligand_id",
        "activity_label",
        "canonical_smiles",
        "unidock_best_score",
        "ligand_efficiency",
        "CNNscore",
        "CNNaffinity",
        "CNN_VS",
        "gnina_affinity",
        "anomaly_flags",
    ]
    for spec in specs:
        col = spec["ranking_col"]
        if col not in df.columns:
            continue
        top = df.sort_values(col, ascending=False, kind="mergesort").head(100).copy()
        top.insert(0, "method_rank", range(1, len(top) + 1))
        top.insert(0, "method", spec["method"])
        rows.append(top[["method", "method_rank"] + [c for c in keep_cols if c in top.columns]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_rank_fusion_scores(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ligand_id",
        "activity_label",
        "is_active",
        "canonical_smiles",
        "unidock_best_score",
        "CNNscore",
        "CNNaffinity",
        "CNN_VS",
        "gnina_affinity",
        "ligand_efficiency",
        "rankscore_unidock_best",
        "rankscore_gnina_cnnscore",
        "rankscore_gnina_cnnaffinity",
        "rankscore_gnina_cnn_vs",
        "rankscore_gnina_affinity",
        "pct_unidock_best",
        "pct_gnina_cnnscore",
        "pct_gnina_cnnaffinity",
        "pct_gnina_cnn_vs",
        "pct_gnina_affinity",
        "fusion_unidock_cnnscore",
        "fusion_unidock_cnnaffinity",
        "fusion_cnnscore_cnnaffinity",
        "fusion_unidock_cnnscore_cnnaffinity",
        "anomaly_flags",
    ]
    return df[[col for col in cols if col in df.columns]].copy()


def build_rank_disagreement_cases(df: pd.DataFrame) -> pd.DataFrame:
    case_specs = [
        (
            "top_unidock_poor_cnnscore",
            df["pct_unidock_best"].ge(0.99) & df["pct_gnina_cnnscore"].le(0.50),
        ),
        (
            "top_cnnscore_poor_unidock",
            df["pct_gnina_cnnscore"].ge(0.99) & df["pct_unidock_best"].le(0.50),
        ),
        (
            "top_cnnaffinity_poor_cnnscore",
            df["pct_gnina_cnnaffinity"].ge(0.99) & df["pct_gnina_cnnscore"].le(0.50),
        ),
        (
            "active_ligands_missed_by_all_methods",
            df["is_active"].eq(1)
            & df["pct_unidock_best"].lt(0.50)
            & df["pct_gnina_cnnscore"].lt(0.50)
            & df["pct_gnina_cnnaffinity"].lt(0.50)
            & df["pct_gnina_cnn_vs"].lt(0.50)
            & df["pct_gnina_affinity"].lt(0.50),
        ),
        (
            "inactive_ligands_ranked_high_by_all_methods",
            df["is_active"].eq(0)
            & df["pct_unidock_best"].ge(0.95)
            & df["pct_gnina_cnnscore"].ge(0.95)
            & df["pct_gnina_cnnaffinity"].ge(0.95)
            & df["pct_gnina_cnn_vs"].ge(0.95),
        ),
        ("extreme_positive_gnina_affinity", df["extreme_positive_gnina_affinity"]),
        ("extreme_negative_unidock", df["extreme_unidock_negative"]),
        (
            "large_unidock_cnnaffinity_rank_shift",
            (df["pct_unidock_best"] - df["pct_gnina_cnnaffinity"]).abs().ge(0.80),
        ),
    ]
    cols = [
        "ligand_id",
        "activity_label",
        "canonical_smiles",
        "unidock_best_score",
        "ligand_efficiency",
        "CNNscore",
        "CNNaffinity",
        "CNN_VS",
        "gnina_affinity",
        "pct_unidock_best",
        "pct_gnina_cnnscore",
        "pct_gnina_cnnaffinity",
        "pct_gnina_cnn_vs",
        "pct_gnina_affinity",
        "fusion_unidock_cnnscore_cnnaffinity",
        "anomaly_flags",
    ]
    rows = []
    for name, mask in case_specs:
        subset = df.loc[mask, [col for col in cols if col in df.columns]].copy()
        if subset.empty:
            continue
        subset["disagreement_type"] = name
        rows.append(subset)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols + ["disagreement_type"])


def _fmt_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "(none)"
    return "```text\n" + df.head(max_rows).to_string(index=False) + "\n```"


def write_reports(
    population: pd.DataFrame,
    qc: pd.DataFrame,
    metrics: pd.DataFrame,
    sensitivity: pd.DataFrame,
    top100: pd.DataFrame,
    reports_dir: Path,
    prefix: str,
) -> None:
    main = metrics[metrics["population"].eq("main")].copy()
    main_sorted_ef = main.sort_values("ef1", ascending=False)
    main_sorted_pr = main.sort_values("pr_auc", ascending=False)
    best_ef = main_sorted_ef.iloc[0]["method"] if not main_sorted_ef.empty else "n/a"
    best_pr = main_sorted_pr.iloc[0]["method"] if not main_sorted_pr.empty else "n/a"
    base = main[~main["method"].str.startswith("fusion_")]
    fusion = main[main["method"].str.startswith("fusion_")]
    fusion_helped_ef = bool((not base.empty) and (not fusion.empty) and fusion["ef1"].max() > base["ef1"].max())
    fusion_helped_pr = bool((not base.empty) and (not fusion.empty) and fusion["pr_auc"].max() > base["pr_auc"].max())

    metric_cols = ["population", "method", "n_total", "n_active", "roc_auc", "pr_auc", "ef1", "ef5", "ef10", "top50_actives", "top100_actives", "top250_actives"]
    top_cols = ["method", "method_rank", "ligand_id", "activity_label", "unidock_best_score", "CNNscore", "CNNaffinity", "CNN_VS", "gnina_affinity", "anomaly_flags"]

    metrics_report = f"""# MAPK1 Phase 1 Metrics Report

## Input Population

- Total ligands: {len(population)}
- Active ligands: {int(population["is_active"].sum())}
- Inactive ligands: {int((population["is_active"] == 0).sum())}
- Activity prevalence: {population["is_active"].mean():.4f}

## Score Directions

- Uni-Dock best score: lower / more negative is better.
- GNINA affinity: lower / more negative is better.
- GNINA CNNscore: higher is better.
- GNINA CNNaffinity: higher is better.
- GNINA CNN_VS: CNNscore multiplied by CNNaffinity; higher is better.
- Rank-fusion scores are exploratory and higher is better.

## Main Metrics

{_fmt_table(main[metric_cols].sort_values("ef1", ascending=False), max_rows=50)}

## Bootstrap CI Summary

Bootstrap confidence intervals are included in `results/tables/{prefix}_method_metrics.csv` and `results/tables/{prefix}_method_metrics_bootstrap_ci.csv`.

## Interpretation

- Best method by EF1%: {best_ef}
- Best method by PR-AUC: {best_pr}
- Rank fusion improved over the best individual method by EF1%: {fusion_helped_ef}
- Rank fusion improved over the best individual method by PR-AUC: {fusion_helped_pr}

These metrics are enrichment summaries for a benchmark screen. They should not be read as binding free energies or experimental confirmation.

## Top Ranked Examples

{_fmt_table(top100[top100["method"].isin(["unidock_best", "gnina_cnn_vs", "gnina_cnnaffinity", "fusion_unidock_cnnscore_cnnaffinity"])][top_cols], max_rows=80)}

## Limitations

The LIT-PCBA benchmark can contain ligand-series and scaffold leakage effects. Score anomalies are flagged but retained in the main analysis, with sensitivity metrics reported separately.
"""

    qc_report = f"""# MAPK1 Phase 1 Score QC Report

Score anomalies were not removed from the main analysis. They are written to anomaly tables for inspection and sensitivity analysis.

## QC Summary

{_fmt_table(qc, max_rows=80)}

## Known Suspicious Values

- GNINA affinity positive values are flagged as `positive_gnina_affinity`.
- GNINA affinity values greater than 20 are flagged as `extreme_positive_gnina_affinity`.
- Uni-Dock scores below -20 are flagged as `extreme_unidock_negative`.
- Ligand efficiencies below -0.8 are flagged as `suspicious_ligand_efficiency_extreme`.
"""

    sens_cols = ["population", "method", "n_total", "n_active", "roc_auc", "pr_auc", "ef1", "ef5", "top50_actives", "notes"]
    sensitivity_report = f"""# MAPK1 Phase 1 Anomaly Sensitivity Report

## Sensitivity Metrics

{_fmt_table(sensitivity[sens_cols], max_rows=80)}

## Manual Inspection Recommendations

- Inspect the extreme Uni-Dock ligand/pose around -29 kcal/mol.
- Inspect rows with extreme positive GNINA affinity.
- Inspect highly ranked inactive false positives.
- Inspect active ligands missed by all methods.

The sensitivity rows show whether EF1%, PR-AUC, or top-k recovery changed after excluding specific anomalous score regions.
"""

    (reports_dir / f"{prefix}_metrics_report.md").write_text(metrics_report, encoding="utf-8")
    (reports_dir / f"{prefix}_score_qc_report.md").write_text(qc_report, encoding="utf-8")
    (reports_dir / f"{prefix}_anomaly_sensitivity_report.md").write_text(sensitivity_report, encoding="utf-8")


def evaluate_sensitivity(population: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants: list[tuple[str, pd.DataFrame, list[str], str]] = [
        (
            "gnina_affinity_no_positive",
            population[population["gnina_affinity"].le(0)].copy(),
            ["gnina_affinity"],
            "Excluded rows where GNINA affinity > 0.",
        ),
        (
            "gnina_affinity_no_extreme_positive",
            population[population["gnina_affinity"].le(20)].copy(),
            ["gnina_affinity"],
            "Excluded rows where GNINA affinity > 20.",
        ),
        (
            "unidock_best_no_extreme_negative",
            population[population["unidock_best_score"].ge(-20)].copy(),
            ["unidock_best"],
            "Excluded rows where Uni-Dock score < -20.",
        ),
        (
            "fusion_unidock_cnnscore_cnnaffinity_no_extreme_unidock",
            population[population["unidock_best_score"].ge(-20)].copy(),
            ["fusion_unidock_cnnscore_cnnaffinity"],
            "Recomputed fusion after excluding rows where Uni-Dock score < -20.",
        ),
        (
            "gnina_all_scores_plausible_only",
            population[
                population["CNNscore"].between(0, 1)
                & population["CNNaffinity"].between(-5, 20)
                & population["gnina_affinity"].le(0)
            ].copy(),
            [
                "gnina_cnnscore",
                "gnina_cnnaffinity",
                "gnina_cnn_vs",
                "gnina_affinity",
                "fusion_unidock_cnnscore",
                "fusion_unidock_cnnaffinity",
                "fusion_cnnscore_cnnaffinity",
                "fusion_unidock_cnnscore_cnnaffinity",
            ],
            "Kept rows with plausible GNINA score ranges only.",
        ),
    ]
    all_metrics: list[pd.DataFrame] = []
    all_ci: list[pd.DataFrame] = []
    all_specs = method_specs(config)
    for population_name, subset, allowed_methods, notes in variants:
        if subset.empty:
            continue
        subset = add_rank_and_fusion_columns(subset, config)
        specs = [spec for spec in all_specs if spec["method"] in allowed_methods]
        metrics_df, ci_df = evaluate_population(subset, population_name, specs, config, notes=notes)
        all_metrics.append(metrics_df)
        all_ci.append(ci_df)
    metrics_out = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    ci_out = pd.concat(all_ci, ignore_index=True) if all_ci else pd.DataFrame()
    return metrics_out, ci_out


def run(metrics_config: Path, tables_dir: Path, figures_dir: Path, reports_dir: Path, prefix: str) -> dict[str, Any]:
    config = load_yaml(metrics_config)
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)
    ensure_dir(reports_dir)

    gnina_path = tables_dir / f"{prefix}_gnina_all_valid_scores.csv"
    unidock_path = discover_unidock_table(tables_dir, prefix)
    population = build_population(gnina_path, unidock_path)
    population = add_score_anomaly_flags(population, config.get("anomaly_thresholds", {}))
    population = add_rank_and_fusion_columns(population, config)

    specs = method_specs(config)
    main_metrics, main_ci = evaluate_population(population, "main", specs, config, notes="Full shared Uni-Dock/GNINA all-valid population.")
    sensitivity_metrics, sensitivity_ci = evaluate_sensitivity(population, config)
    metrics = pd.concat([main_metrics, sensitivity_metrics], ignore_index=True)
    ci = pd.concat([main_ci, sensitivity_ci], ignore_index=True)
    qc = build_qc_table(population)
    rank_fusion_scores = build_rank_fusion_scores(population)
    top100 = build_top100_by_method(population, specs)
    disagreement = build_rank_disagreement_cases(population)

    population.to_csv(tables_dir / f"{prefix}_score_population.csv", index=False)
    qc.to_csv(tables_dir / f"{prefix}_score_qc.csv", index=False)
    metrics.to_csv(tables_dir / f"{prefix}_method_metrics.csv", index=False)
    ci.to_csv(tables_dir / f"{prefix}_method_metrics_bootstrap_ci.csv", index=False)
    rank_fusion_scores.to_csv(tables_dir / f"{prefix}_rank_fusion_scores.csv", index=False)
    top100.to_csv(tables_dir / f"{prefix}_top100_by_method.csv", index=False)
    disagreement.to_csv(tables_dir / f"{prefix}_rank_disagreement_cases.csv", index=False)
    anomaly_table(population, GNINA_FLAGS).to_csv(tables_dir / f"{prefix}_gnina_score_anomalies.csv", index=False)
    anomaly_table(population, UNIDOCK_FLAGS).to_csv(tables_dir / f"{prefix}_unidock_score_anomalies.csv", index=False)
    anomaly_table(population, GNINA_FLAGS + UNIDOCK_FLAGS).to_csv(tables_dir / f"{prefix}_all_score_anomalies.csv", index=False)

    make_screening_plots(population, metrics, figures_dir, prefix)
    write_reports(population, qc, metrics, sensitivity_metrics, top100, reports_dir, prefix)

    main = metrics[metrics["population"].eq("main")]
    best_ef = main.sort_values("ef1", ascending=False).iloc[0]["method"] if not main.empty else None
    best_pr = main.sort_values("pr_auc", ascending=False).iloc[0]["method"] if not main.empty else None
    counts = anomaly_counts(population)
    return {
        "n_total": int(len(population)),
        "n_active": int(population["is_active"].sum()),
        "n_inactive": int((population["is_active"] == 0).sum()),
        "best_method_by_ef1": best_ef,
        "best_method_by_pr_auc": best_pr,
        "anomaly_counts": counts,
        "outputs_prefix": prefix,
    }


def main() -> None:
    args = parse_args()
    summary = run(
        metrics_config=Path(args.metrics_config),
        tables_dir=Path(args.tables_dir),
        figures_dir=Path(args.figures_dir),
        reports_dir=Path(args.reports_dir),
        prefix=args.out_prefix,
    )
    LOGGER.info("Step 6 complete: %s", summary)


if __name__ == "__main__":
    main()
