from __future__ import annotations

import argparse
import glob
import math
import zlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from screensift.common.io import ensure_dir, load_yaml
from screensift.common.logging_utils import setup_logger
from screensift.metrics.enrichment import (
    bootstrap_metric_ci,
    compute_enrichment_factor,
    compute_pr_auc,
    compute_roc_auc,
    compute_topk_recovery,
    make_rank_score,
)
from screensift.metrics.objective_2d_baselines import (
    compute_all_active_leave_one_out,
    compute_few_active_similarity,
    compute_inactive_reference_control,
    compute_label_shuffle_control,
    compute_native_ligand_similarity,
    compute_near_analog_subset,
    compute_scaffold_holdout_similarity,
    prepare_fingerprints,
    scaffold_from_smiles,
    strip_fingerprints,
)
from screensift.metrics.structure_added_value import structure_added_value_cases


LOGGER = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run objective 2D ligand-similarity audit.")
    parser.add_argument("--audit-config", default="configs/objective_2d_audit.yml")
    parser.add_argument("--score-population", default="example/mapk1/expected_outputs/mapk1_2receptor_score_population.csv")
    parser.add_argument("--full-population", default="results/tables/mapk1_phase1_full_population_with_ecfp4_and_anomalies.csv")
    parser.add_argument("--clean-population", default="results/tables/mapk1_phase1_clean_population_with_ecfp4_and_anomalies.csv")
    parser.add_argument("--metrics-config", default="configs/metrics.yml")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--reports-dir", default="results/reports")
    parser.add_argument("--out-prefix", default="mapk1_phase1")
    return parser.parse_args()


def _load_population(path: Path, fallback: Path | None = None) -> pd.DataFrame:
    source = path if path.exists() else fallback
    if source is None or not source.exists():
        raise FileNotFoundError(f"Population file not found: {path}")
    df = pd.read_csv(source, dtype={"ligand_id": str}, low_memory=False)
    if "canonical_smiles" not in df.columns and "smiles" in df.columns:
        df["canonical_smiles"] = df["smiles"]
    required = ["ligand_id", "activity_label", "canonical_smiles", "unidock_best_score", "CNNscore", "CNNaffinity", "gnina_affinity"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Population {source} is missing required columns: {missing}")
    if "is_active" not in df.columns:
        df["is_active"] = df["activity_label"].astype(str).str.lower().eq("active").astype(int)
    df["is_active"] = pd.to_numeric(df["is_active"], errors="coerce").fillna(0).astype(int)
    if "clean_population_excluded" not in df.columns:
        df["clean_population_excluded"] = False
    if "anomaly_flags" not in df.columns:
        df["anomaly_flags"] = ""
    if "scaffold_smiles" not in df.columns:
        df["scaffold_smiles"] = df["canonical_smiles"].map(scaffold_from_smiles)
    return df


def _structure_specs(config: dict[str, Any], df: pd.DataFrame) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for method, method_config in config["structure_methods"].items():
        col = next((candidate for candidate in method_config["score_column_candidates"] if candidate in df.columns), None)
        if col is None:
            continue
        specs.append({"method": method, "score_col": col, "direction": str(method_config["direction"])})
    return specs


def _metric_row(
    df: pd.DataFrame,
    population: str,
    baseline: str,
    method: str,
    score_col: str,
    direction: str,
    metrics_config: dict[str, Any],
    repeat: int | None = None,
    k: int | None = None,
    threshold: float | None = None,
    bootstrap: bool = False,
    notes: str = "",
) -> dict[str, Any]:
    work = df[["is_active", score_col]].copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.dropna(subset=["is_active", score_col])
    y = work["is_active"].astype(int)
    score = make_rank_score(work[score_col], direction)
    top_ks = metrics_config.get("top_ks", [50, 100, 250])
    row: dict[str, Any] = {
        "population": population,
        "baseline": baseline,
        "method": method,
        "repeat": repeat,
        "k": k,
        "threshold": threshold,
        "n_total": int(len(work)),
        "n_active": int(y.sum()),
        "n_inactive": int(len(work) - y.sum()),
        "score_col": score_col,
        "direction": direction,
        "roc_auc": compute_roc_auc(y, score),
        "pr_auc": compute_pr_auc(y, score),
        "ef1": compute_enrichment_factor(y, score, 0.01),
        "ef5": compute_enrichment_factor(y, score, 0.05),
        "ef10": compute_enrichment_factor(y, score, 0.10),
        "notes": notes,
    }
    for top_k in top_ks:
        recovery = compute_topk_recovery(y, score, int(top_k))
        row[f"top{top_k}_actives"] = recovery["topk_actives"]
        row[f"top{top_k}_active_recovery_fraction"] = recovery["topk_recovery_fraction"]

    if bootstrap:
        n_boot = int(metrics_config.get("bootstrap_iterations", 1000))
        confidence = float(metrics_config.get("bootstrap_confidence", 0.95))
        seed = int(metrics_config.get("random_seed", 42))
        metric_fns = {
            "roc_auc": compute_roc_auc,
            "pr_auc": compute_pr_auc,
            "ef1": lambda yy, ss: compute_enrichment_factor(yy, ss, 0.01),
            "ef5": lambda yy, ss: compute_enrichment_factor(yy, ss, 0.05),
        }
        for metric_name, fn in metric_fns.items():
            # Derive the per-cell bootstrap seed with a stable checksum, not the
            # builtin hash(): hash() of a tuple of str is salted per process via
            # PYTHONHASHSEED, so CIs would differ run-to-run. crc32 is
            # deterministic across processes (matches _stable_seed in
            # evaluate_phase1_scores).
            cell_key = "::".join(str(part) for part in (population, baseline, method, metric_name))
            cell_seed = seed + zlib.crc32(cell_key.encode("utf-8")) % 100000
            ci = bootstrap_metric_ci(y, score, fn, n_boot=n_boot, seed=cell_seed, confidence=confidence)
            row[f"{metric_name}_low"] = ci["ci_low"]
            row[f"{metric_name}_high"] = ci["ci_high"]
            row[f"{metric_name}_n_boot_valid"] = ci["n_boot_valid"]
    return row


def _method_rows(
    df: pd.DataFrame,
    population: str,
    baseline: str,
    baseline_method: tuple[str, str, str] | None,
    structure_specs: list[dict[str, str]],
    metrics_config: dict[str, Any],
    repeat: int | None = None,
    k: int | None = None,
    threshold: float | None = None,
    bootstrap: bool = False,
    notes: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if baseline_method is not None:
        method, score_col, direction = baseline_method
        rows.append(
            _metric_row(
                df,
                population,
                baseline,
                method,
                score_col,
                direction,
                metrics_config,
                repeat=repeat,
                k=k,
                threshold=threshold,
                bootstrap=bootstrap,
                notes=notes,
            )
        )
    for spec in structure_specs:
        rows.append(
            _metric_row(
                df,
                population,
                baseline,
                spec["method"],
                spec["score_col"],
                spec["direction"],
                metrics_config,
                repeat=repeat,
                k=k,
                threshold=threshold,
                bootstrap=bootstrap,
                notes=notes,
            )
        )
    return rows


def _summarize_repeats(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metrics = ["roc_auc", "pr_auc", "ef1", "ef5", "ef10", "top50_actives", "top100_actives", "top250_actives"]
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_repeats"] = int(group["repeat"].nunique())
        row["n_total_mean"] = float(group["n_total"].mean())
        row["n_active_mean"] = float(group["n_active"].mean())
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_median"] = np.nan
                row[f"{metric}_q025"] = np.nan
                row[f"{metric}_q975"] = np.nan
            else:
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                row[f"{metric}_median"] = float(values.median())
                row[f"{metric}_q025"] = float(values.quantile(0.025))
                row[f"{metric}_q975"] = float(values.quantile(0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def _native_sdf_files(config: dict[str, Any]) -> list[str]:
    native_config = config.get("native_ligand_baseline", {})
    files = sorted(glob.glob(native_config.get("native_ligand_sdf_glob", "")))
    files.extend(sorted(glob.glob(native_config.get("native_ligand_pdb_glob", ""))))
    table = Path(native_config.get("native_ligand_table", ""))
    if table.exists():
        try:
            native = pd.read_csv(table)
            if "sdf_path" in native.columns:
                files.extend(native["sdf_path"].dropna().astype(str).tolist())
            if "pdb_path" in native.columns:
                files.extend(native["pdb_path"].dropna().astype(str).tolist())
        except Exception as exc:
            LOGGER.warning("Could not read native ligand table %s: %s", table, exc)
    return sorted(set(files))


def _single_baselines(
    pop: pd.DataFrame,
    population_name: str,
    config: dict[str, Any],
    native_files: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    metrics_config = config["metrics"]
    structure_specs = _structure_specs(config, pop)
    pop = compute_all_active_leave_one_out(pop)
    rows = _method_rows(
        pop,
        population_name,
        "all_active_loo_ecfp4_analog_neighborhood",
        ("all_active_loo_ecfp4_analog_neighborhood", "all_active_loo_ecfp4_analog_neighborhood", "higher"),
        structure_specs,
        metrics_config,
        bootstrap=True,
        notes="All-active leave-one-out analog-neighborhood diagnostic, not prospective baseline.",
    )

    native_warning = None
    native_enabled = bool(config.get("native_ligand_baseline", {}).get("enabled", True))
    if native_enabled:
        fp_config = config["fingerprint"]
        pop, native_warning = compute_native_ligand_similarity(
            pop,
            native_files,
            radius=int(fp_config.get("radius", 2)),
            n_bits=int(fp_config.get("n_bits", 2048)),
            use_chirality=bool(fp_config.get("use_chirality", False)),
        )
        if pop["native_ligand_ecfp4"].notna().any():
            rows.extend(
                _method_rows(
                    pop,
                    population_name,
                    "native_ligand_ecfp4",
                    ("native_ligand_ecfp4", "native_ligand_ecfp4", "higher"),
                    structure_specs,
                    metrics_config,
                    bootstrap=True,
                    notes="Native/co-crystal ligand ECFP4 baseline.",
                )
            )
    return pop, pd.DataFrame(rows), native_warning


def _few_active_repeats(pop: pd.DataFrame, population_name: str, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config["few_active_baseline"]
    metrics_config = config["metrics"]
    structure_specs = _structure_specs(config, pop)
    rows: list[dict[str, Any]] = []
    near_rows: list[dict[str, Any]] = []
    thresholds = config["near_analog_removed"]["thresholds"]
    min_actives = int(config["near_analog_removed"].get("min_actives_required", 10))
    for k in cfg["k_values"]:
        for repeat in range(int(cfg["n_repeats"])):
            eval_df, _ = compute_few_active_similarity(pop, int(k), int(cfg["random_seed"]) + repeat + int(k) * 1000)
            if eval_df.empty or int(eval_df["is_active"].sum()) < int(cfg.get("min_test_actives_required", 10)):
                continue
            rows.extend(
                _method_rows(
                    eval_df,
                    population_name,
                    "few_active_ecfp4",
                    ("few_active_ecfp4", "few_active_ecfp4", "higher"),
                    structure_specs,
                    metrics_config,
                    repeat=repeat,
                    k=int(k),
                    notes="Few-active ECFP4; sampled reference actives removed from evaluation.",
                )
            )
            for threshold in thresholds:
                subset = compute_near_analog_subset(eval_df, "few_active_ecfp4", float(threshold))
                if int(subset["is_active"].sum()) < min_actives:
                    continue
                near_rows.extend(
                    _method_rows(
                        subset,
                        population_name,
                        "few_active_ecfp4_near_analog_removed",
                        ("few_active_ecfp4", "few_active_ecfp4", "higher"),
                        structure_specs,
                        metrics_config,
                        repeat=repeat,
                        k=int(k),
                        threshold=float(threshold),
                        notes="Near-analog-removed few-active subset.",
                    )
                )
    return pd.DataFrame(rows), pd.DataFrame(near_rows)


def _scaffold_holdout_repeats(pop: pd.DataFrame, population_name: str, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config["scaffold_holdout_baseline"]
    metrics_config = config["metrics"]
    structure_specs = _structure_specs(config, pop)
    rows: list[dict[str, Any]] = []
    near_rows: list[dict[str, Any]] = []
    thresholds = config["near_analog_removed"]["thresholds"]
    min_actives = int(config["near_analog_removed"].get("min_actives_required", 10))
    if not cfg.get("enabled", True):
        return pd.DataFrame(), pd.DataFrame()
    for repeat in range(int(cfg["n_repeats"])):
        eval_df, _ = compute_scaffold_holdout_similarity(
            pop,
            repeat_seed=int(cfg["random_seed"]) + repeat,
            reference_scaffold_fraction=float(cfg.get("reference_scaffold_fraction", 0.30)),
            min_reference_active_scaffolds=int(cfg.get("min_reference_active_scaffolds", 5)),
            min_test_active_scaffolds=int(cfg.get("min_test_active_scaffolds", 5)),
            min_test_actives_required=int(cfg.get("min_test_actives_required", 10)),
        )
        if eval_df.empty:
            continue
        rows.extend(
            _method_rows(
                eval_df,
                population_name,
                "scaffold_holdout_ecfp4",
                ("scaffold_holdout_ecfp4", "scaffold_holdout_ecfp4", "higher"),
                structure_specs,
                metrics_config,
                repeat=repeat,
                notes="Scaffold-held-out ECFP4; reference active scaffolds excluded from evaluation.",
            )
        )
        for threshold in thresholds:
            subset = compute_near_analog_subset(eval_df, "scaffold_holdout_ecfp4", float(threshold))
            if int(subset["is_active"].sum()) < min_actives:
                continue
            near_rows.extend(
                _method_rows(
                    subset,
                    population_name,
                    "scaffold_holdout_ecfp4_near_analog_removed",
                    ("scaffold_holdout_ecfp4", "scaffold_holdout_ecfp4", "higher"),
                    structure_specs,
                    metrics_config,
                    repeat=repeat,
                    threshold=float(threshold),
                    notes="Near-analog-removed scaffold-held-out subset.",
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(near_rows)


def _single_near_analog_metrics(pop: pd.DataFrame, population_name: str, config: dict[str, Any], native_available: bool) -> pd.DataFrame:
    metrics_config = config["metrics"]
    structure_specs = _structure_specs(config, pop)
    rows: list[dict[str, Any]] = []
    thresholds = config["near_analog_removed"]["thresholds"]
    min_actives = int(config["near_analog_removed"].get("min_actives_required", 10))
    baselines = [("all_active_loo_ecfp4", "all_active_loo_ecfp4_analog_neighborhood")]
    if native_available:
        baselines.append(("native_ligand_ecfp4", "native_ligand_ecfp4"))
    for baseline_name, score_col in baselines:
        for threshold in thresholds:
            subset = compute_near_analog_subset(pop, score_col, float(threshold))
            if subset.empty or int(subset["is_active"].sum()) < min_actives:
                continue
            rows.extend(
                _method_rows(
                    subset,
                    population_name,
                    f"{baseline_name}_near_analog_removed",
                    (baseline_name, score_col, "higher"),
                    structure_specs,
                    metrics_config,
                    threshold=float(threshold),
                    notes=f"Near-analog-removed subset using {score_col} < {threshold}.",
                )
            )
    return pd.DataFrame(rows)


def _negative_controls(pop: pd.DataFrame, population_name: str, config: dict[str, Any]) -> pd.DataFrame:
    controls = config["negative_controls"]
    metrics_config = config["metrics"]
    structure_specs = _structure_specs(config, pop)
    rows: list[dict[str, Any]] = []
    label_cfg = controls.get("label_shuffle", {})
    if label_cfg.get("enabled", True):
        for repeat in range(int(label_cfg["n_repeats"])):
            shuffled = compute_label_shuffle_control(pop, int(label_cfg["random_seed"]) + repeat)
            rows.extend(
                _method_rows(
                    shuffled,
                    population_name,
                    "label_shuffle_control",
                    ("label_shuffle_all_active_loo_ecfp4", "all_active_loo_ecfp4_analog_neighborhood", "higher"),
                    structure_specs,
                    metrics_config,
                    repeat=repeat,
                    notes="Activity labels shuffled; scores unchanged.",
                )
            )
    inactive_cfg = controls.get("inactive_reference", {})
    if inactive_cfg.get("enabled", True):
        for k in inactive_cfg["k_values"]:
            for repeat in range(int(inactive_cfg["n_repeats"])):
                eval_df, _ = compute_inactive_reference_control(pop, int(k), int(inactive_cfg["random_seed"]) + repeat + int(k) * 1000)
                if eval_df.empty:
                    continue
                rows.extend(
                    _method_rows(
                        eval_df,
                        population_name,
                        "inactive_reference_control",
                        ("inactive_reference_ecfp4", "inactive_reference_ecfp4", "higher"),
                        structure_specs,
                        metrics_config,
                        repeat=repeat,
                        k=int(k),
                        notes="Similarity to sampled inactive references.",
                    )
                )
    return pd.DataFrame(rows)


def _crossover_summary(few_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if few_summary.empty:
        return pd.DataFrame()
    structure_methods = ["unidock_best", "gnina_cnnscore", "gnina_cnnaffinity", "gnina_affinity"]
    for population in sorted(few_summary["population"].dropna().unique()):
        sub = few_summary[few_summary["population"].eq(population)]
        for metric in ["ef1", "pr_auc"]:
            ecfp = sub[sub["method"].eq("few_active_ecfp4")]
            for target in structure_methods + ["best_structure_method"]:
                crossover_k = None
                detail = "no crossover up to max k"
                for k in sorted(ecfp["k"].dropna().unique()):
                    ecfp_value = ecfp.loc[ecfp["k"].eq(k), f"{metric}_mean"]
                    if ecfp_value.empty:
                        continue
                    if target == "best_structure_method":
                        target_value = sub[sub["k"].eq(k) & sub["method"].isin(structure_methods)][f"{metric}_mean"].max()
                    else:
                        target_series = sub[sub["k"].eq(k) & sub["method"].eq(target)][f"{metric}_mean"]
                        target_value = target_series.iloc[0] if not target_series.empty else np.nan
                    if pd.notna(target_value) and float(ecfp_value.iloc[0]) > float(target_value):
                        crossover_k = int(k)
                        detail = "few-active ECFP4 exceeds target"
                        break
                rows.append(
                    {
                        "population": population,
                        "metric": metric,
                        "target": target,
                        "crossover_k": crossover_k,
                        "interpretation": detail if crossover_k is not None else detail,
                    }
                )
    return pd.DataFrame(rows)


def _objective_summary(single: pd.DataFrame, few_summary: pd.DataFrame, scaffold_summary: pd.DataFrame, crossover: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for population in sorted(single["population"].dropna().unique()) if not single.empty else []:
        sub = single[single["population"].eq(population)]
        for method in ["all_active_loo_ecfp4_analog_neighborhood", "unidock_best", "gnina_cnnscore", "gnina_cnnaffinity", "gnina_affinity", "native_ligand_ecfp4"]:
            row = sub[sub["method"].eq(method)]
            if row.empty:
                continue
            rows.append({"population": population, "analysis": "single_baseline", **row.iloc[0].to_dict()})
    if not few_summary.empty:
        best_few = few_summary[few_summary["method"].eq("few_active_ecfp4")].copy()
        best_few["analysis"] = "few_active_summary"
        rows.extend(best_few.to_dict("records"))
    if not scaffold_summary.empty:
        scaffold_ecfp = scaffold_summary[scaffold_summary["method"].eq("scaffold_holdout_ecfp4")].copy()
        scaffold_ecfp["analysis"] = "scaffold_holdout_summary"
        rows.extend(scaffold_ecfp.to_dict("records"))
    if not crossover.empty:
        cross = crossover.copy()
        cross["analysis"] = "prior_knowledge_crossover"
        rows.extend(cross.to_dict("records"))
    return pd.DataFrame(rows)


def _plot_prior_knowledge(summary: pd.DataFrame, figures_dir: Path, prefix: str, population: str, metric: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if not summary.empty:
        for method, group in summary[summary["population"].eq(population)].groupby("method"):
            if method not in {"few_active_ecfp4", "unidock_best", "gnina_cnnscore", "gnina_cnnaffinity", "gnina_affinity"}:
                continue
            group = group.sort_values("k")
            ax.plot(group["k"], group[f"{metric}_mean"], marker="o", label=method)
    ax.set_xlabel("Known active references (k)")
    ax.set_ylabel(metric)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}_prior_knowledge_curve_{metric}_{population}.png", dpi=160)
    plt.close(fig)


def _plot_near_analog(metrics: pd.DataFrame, figures_dir: Path, prefix: str, population: str, metric: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if not metrics.empty:
        sub = metrics[metrics["population"].eq(population)]
        for method, group in sub.groupby("method"):
            group = group.dropna(subset=["threshold"]).sort_values("threshold")
            if group.empty:
                continue
            grouped = group.groupby("threshold")[metric].mean().reset_index()
            ax.plot(grouped["threshold"], grouped[metric], marker="o", label=method)
    ax.set_xlabel("Maximum ECFP4 similarity threshold")
    ax.set_ylabel(metric)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}_near_analog_removed_{metric}_{population}.png", dpi=160)
    plt.close(fig)


def _plot_bar(summary: pd.DataFrame, figures_dir: Path, out_name: str, metric: str = "ef1_mean") -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if not summary.empty and metric in summary.columns:
        data = summary.groupby("method")[metric].mean().sort_values(ascending=False)
        ax.bar(range(len(data)), data.to_numpy())
        ax.set_xticks(range(len(data)), data.index, rotation=45, ha="right")
    ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(figures_dir / out_name, dpi=160)
    plt.close(fig)


def _plot_sav(cases: pd.DataFrame, figures_dir: Path, prefix: str, population: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if not cases.empty and "SAV" in cases.columns:
        ax.hist(pd.to_numeric(cases["SAV"], errors="coerce").dropna(), bins=30)
    ax.set_xlabel("Structure added value")
    ax.set_ylabel("Active ligand count")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}_structure_added_value_distribution_{population}.png", dpi=160)
    plt.close(fig)


def _fmt(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "(none)"
    return "```text\n" + df.head(max_rows).to_string(index=False) + "\n```"


def _write_reports(
    single: pd.DataFrame,
    few_summary: pd.DataFrame,
    scaffold_summary: pd.DataFrame,
    near_metrics: pd.DataFrame,
    negative_summary: pd.DataFrame,
    crossover: pd.DataFrame,
    sav_summary: pd.DataFrame,
    native_warning: str | None,
    reports_dir: Path,
    prefix: str,
) -> None:
    all_active = single[single["method"].eq("all_active_loo_ecfp4_analog_neighborhood")]
    native = single[single["method"].eq("native_ligand_ecfp4")]
    report = f"""# MAPK1 Phase 1 Objective 2D Baseline Audit

## Framing

Step 7 all-active leave-one-out ECFP4 is retained here as an analog-neighborhood diagnostic, not as a prospective baseline. It uses almost all known actives as references, so strong global performance mainly indicates ligand-series/analog-neighborhood signal.

## Corrected Baselines

- All-active LOO ECFP4: analog-neighborhood diagnostic.
- Native-ligand-only ECFP4: skipped if no native ligand SDFs exist.
- Few-active ECFP4: k=1, 5, 10, 25 known actives, with sampled actives removed from evaluation.
- Scaffold-held-out ECFP4: active reference scaffolds excluded from evaluation.
- Near-analog-removed analysis: Tanimoto thresholds 0.70, 0.50, 0.30.
- Negative controls: label shuffle and inactive references.

## All-Active LOO Diagnostic

{_fmt(all_active[["population", "method", "n_total", "n_active", "roc_auc", "pr_auc", "ef1", "ef5", "top50_actives"]])}

## Native Ligand Baseline

{native_warning or "Native ligand baseline ran."}

{_fmt(native[["population", "method", "n_total", "n_active", "roc_auc", "pr_auc", "ef1", "ef5", "top50_actives"]])}

## Few-Active Prior Knowledge

{_fmt(few_summary[["population", "k", "method", "n_repeats", "pr_auc_mean", "ef1_mean", "ef5_mean", "top50_actives_mean"]], max_rows=80)}

## Scaffold-Held-Out

{_fmt(scaffold_summary[["population", "method", "n_repeats", "pr_auc_mean", "ef1_mean", "ef5_mean", "top50_actives_mean"]], max_rows=60)}

## Near-Analog-Removed

{_fmt(near_metrics[["population", "baseline", "method", "k", "threshold", "n_total", "n_active", "pr_auc", "ef1", "ef5"]], max_rows=80)}

## Negative Controls

{_fmt(negative_summary[["population", "baseline", "k", "method", "n_repeats", "pr_auc_mean", "ef1_mean"]], max_rows=80)}

## Prior-Knowledge Crossover

{_fmt(crossover, max_rows=80)}
"""

    calibration = f"""# MAPK1 Phase 1 Prior-Knowledge Calibration Report

## k = 1, 5, 10, 25

{_fmt(few_summary[["population", "k", "method", "n_repeats", "roc_auc_mean", "pr_auc_mean", "ef1_mean", "ef1_q025", "ef1_q975"]], max_rows=100)}

## Crossover Points

{_fmt(crossover, max_rows=100)}

If k=1 already crosses a structure method, ligand similarity dominates with minimal prior SAR. If no crossover occurs up to k=25, structure retained value under this prior-knowledge regime.
"""

    sav = f"""# MAPK1 Phase 1 Structure Added Value Report

SAV = best_structure_percentile - ECFP4 percentile. SAV > 0 means a structure method ranked the active ligand better than ECFP4.

## SAV Counts

{_fmt(sav_summary, max_rows=80)}

Top structure-added-value cases are written to `results/tables/{prefix}_structure_added_value_cases_full.csv` and `results/tables/{prefix}_structure_added_value_cases_clean.csv`.
"""

    limitations = """# MAPK1 Phase 1 2D Baseline Limitations

- All-active LOO is an analog-neighborhood diagnostic, not a prospective baseline.
- Native-ligand-only ECFP4 is more prospective but depends on co-crystal ligand representativeness.
- Few-active ECFP4 depends strongly on which known actives are sampled.
- Scaffold-held-out evaluation is stricter but can be noisy due scaffold imbalance.
- Near-analog removal reduces active count and can destabilize EF1.
- ECFP4 is 2D only and cannot provide pose, interaction, or selectivity hypotheses.
- Docking scores are noisy and are not affinity estimates.
"""

    (reports_dir / f"{prefix}_objective_2d_audit_report.md").write_text(report, encoding="utf-8")
    (reports_dir / f"{prefix}_prior_knowledge_calibration_report.md").write_text(calibration, encoding="utf-8")
    (reports_dir / f"{prefix}_structure_added_value_report.md").write_text(sav, encoding="utf-8")
    (reports_dir / f"{prefix}_2d_baseline_limitations.md").write_text(limitations, encoding="utf-8")


def run(
    audit_config: Path,
    score_population: Path,
    full_population: Path,
    clean_population: Path,
    metrics_config: Path,
    tables_dir: Path,
    figures_dir: Path,
    reports_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    _ = metrics_config
    config = load_yaml(audit_config)
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)
    ensure_dir(reports_dir)

    fp_config = config["fingerprint"]
    full = _load_population(full_population, fallback=score_population)
    clean = _load_population(clean_population, fallback=None) if clean_population.exists() else full[~full["clean_population_excluded"].astype(bool)].copy()
    full = prepare_fingerprints(full, radius=int(fp_config["radius"]), n_bits=int(fp_config["n_bits"]), use_chirality=bool(fp_config["use_chirality"]))
    clean = prepare_fingerprints(clean, radius=int(fp_config["radius"]), n_bits=int(fp_config["n_bits"]), use_chirality=bool(fp_config["use_chirality"]))

    native_files = _native_sdf_files(config)
    single_tables: list[pd.DataFrame] = []
    near_tables: list[pd.DataFrame] = []
    few_repeat_tables: list[pd.DataFrame] = []
    few_summary_tables: list[pd.DataFrame] = []
    scaffold_repeat_tables: list[pd.DataFrame] = []
    scaffold_summary_tables: list[pd.DataFrame] = []
    negative_tables: list[pd.DataFrame] = []
    sav_summary_tables: list[pd.DataFrame] = []
    sav_cases: dict[str, pd.DataFrame] = {}
    audit_populations: dict[str, pd.DataFrame] = {}
    native_warnings: list[str] = []

    for population_name, pop in [("full", full), ("clean", clean)]:
        pop, single, native_warning = _single_baselines(pop, population_name, config, native_files)
        if native_warning:
            native_warnings.append(f"{population_name}: {native_warning}")
        audit_populations[population_name] = pop
        single_tables.append(single)
        native_available = "native_ligand_ecfp4" in pop.columns and pop["native_ligand_ecfp4"].notna().any()
        near_tables.append(_single_near_analog_metrics(pop, population_name, config, native_available=native_available))

        few_repeats, few_near = _few_active_repeats(pop, population_name, config)
        few_repeat_tables.append(few_repeats)
        near_tables.append(few_near)
        few_summary_tables.append(_summarize_repeats(few_repeats, ["population", "baseline", "k", "method"]) if not few_repeats.empty else pd.DataFrame())

        scaffold_repeats, scaffold_near = _scaffold_holdout_repeats(pop, population_name, config)
        scaffold_repeat_tables.append(scaffold_repeats)
        near_tables.append(scaffold_near)
        scaffold_summary_tables.append(_summarize_repeats(scaffold_repeats, ["population", "baseline", "method"]) if not scaffold_repeats.empty else pd.DataFrame())

        negative = _negative_controls(pop, population_name, config)
        negative_tables.append(negative)

        structure_cols = {
            "unidock_best": ("unidock_best_score", "lower"),
            "gnina_cnnscore": ("CNNscore", "higher"),
            "gnina_cnnaffinity": ("CNNaffinity", "higher"),
            "gnina_affinity": ("gnina_affinity", "lower"),
        }
        sav_case_table, sav_summary = structure_added_value_cases(
            pop,
            "all_active_loo_ecfp4_analog_neighborhood",
            structure_cols,
            top_k_values=config["structure_added_value"]["top_k_values"],
            novelty_thresholds=config["structure_added_value"]["novelty_thresholds"],
        )
        sav_summary["population"] = population_name
        sav_cases[population_name] = sav_case_table
        sav_summary_tables.append(sav_summary)

    single_metrics = pd.concat(single_tables, ignore_index=True) if single_tables else pd.DataFrame()
    few_repeats = pd.concat(few_repeat_tables, ignore_index=True) if few_repeat_tables else pd.DataFrame()
    few_summary = pd.concat(few_summary_tables, ignore_index=True) if few_summary_tables else pd.DataFrame()
    scaffold_repeats = pd.concat(scaffold_repeat_tables, ignore_index=True) if scaffold_repeat_tables else pd.DataFrame()
    scaffold_summary = pd.concat(scaffold_summary_tables, ignore_index=True) if scaffold_summary_tables else pd.DataFrame()
    near_metrics = pd.concat(near_tables, ignore_index=True) if near_tables else pd.DataFrame()
    negative_metrics = pd.concat(negative_tables, ignore_index=True) if negative_tables else pd.DataFrame()
    negative_summary = _summarize_repeats(negative_metrics, ["population", "baseline", "k", "method"]) if not negative_metrics.empty else pd.DataFrame()
    crossover = _crossover_summary(few_summary)
    objective_summary = _objective_summary(single_metrics, few_summary, scaffold_summary, crossover)
    sav_summary_all = pd.concat(sav_summary_tables, ignore_index=True) if sav_summary_tables else pd.DataFrame()

    for population_name, pop in audit_populations.items():
        strip_fingerprints(pop).to_csv(tables_dir / f"{prefix}_objective_2d_audit_population_{population_name}.csv", index=False)
    single_metrics[single_metrics["population"].eq("full")].to_csv(tables_dir / f"{prefix}_objective_2d_single_baseline_metrics_full.csv", index=False)
    single_metrics[single_metrics["population"].eq("clean")].to_csv(tables_dir / f"{prefix}_objective_2d_single_baseline_metrics_clean.csv", index=False)
    few_repeats[few_repeats["population"].eq("full")].to_csv(tables_dir / f"{prefix}_few_active_ecfp4_repeat_metrics_full.csv", index=False)
    few_repeats[few_repeats["population"].eq("clean")].to_csv(tables_dir / f"{prefix}_few_active_ecfp4_repeat_metrics_clean.csv", index=False)
    few_summary[few_summary["population"].eq("full")].to_csv(tables_dir / f"{prefix}_few_active_ecfp4_summary_full.csv", index=False)
    few_summary[few_summary["population"].eq("clean")].to_csv(tables_dir / f"{prefix}_few_active_ecfp4_summary_clean.csv", index=False)
    scaffold_repeats[scaffold_repeats["population"].eq("full")].to_csv(tables_dir / f"{prefix}_scaffold_holdout_ecfp4_repeat_metrics_full.csv", index=False)
    scaffold_repeats[scaffold_repeats["population"].eq("clean")].to_csv(tables_dir / f"{prefix}_scaffold_holdout_ecfp4_repeat_metrics_clean.csv", index=False)
    scaffold_summary[scaffold_summary["population"].eq("full")].to_csv(tables_dir / f"{prefix}_scaffold_holdout_ecfp4_summary_full.csv", index=False)
    scaffold_summary[scaffold_summary["population"].eq("clean")].to_csv(tables_dir / f"{prefix}_scaffold_holdout_ecfp4_summary_clean.csv", index=False)
    near_metrics[near_metrics["population"].eq("full")].to_csv(tables_dir / f"{prefix}_near_analog_removed_metrics_full.csv", index=False)
    near_metrics[near_metrics["population"].eq("clean")].to_csv(tables_dir / f"{prefix}_near_analog_removed_metrics_clean.csv", index=False)
    negative_metrics[negative_metrics["population"].eq("full")].to_csv(tables_dir / f"{prefix}_negative_control_metrics_full.csv", index=False)
    negative_metrics[negative_metrics["population"].eq("clean")].to_csv(tables_dir / f"{prefix}_negative_control_metrics_clean.csv", index=False)
    crossover.to_csv(tables_dir / f"{prefix}_prior_knowledge_crossover_summary.csv", index=False)
    objective_summary.to_csv(tables_dir / f"{prefix}_objective_2d_vs_structure_summary.csv", index=False)
    for population_name, cases in sav_cases.items():
        cases.to_csv(tables_dir / f"{prefix}_structure_added_value_cases_{population_name}.csv", index=False)

    for population_name in ["full", "clean"]:
        _plot_prior_knowledge(few_summary, figures_dir, prefix, population_name, "ef1")
        _plot_prior_knowledge(few_summary, figures_dir, prefix, population_name, "pr_auc")
        _plot_near_analog(near_metrics, figures_dir, prefix, population_name, "ef1")
        _plot_near_analog(near_metrics, figures_dir, prefix, population_name, "pr_auc")
        _plot_bar(scaffold_summary[scaffold_summary["population"].eq(population_name)], figures_dir, f"{prefix}_scaffold_holdout_summary_{population_name}.png")
        _plot_bar(negative_summary[negative_summary["population"].eq(population_name)], figures_dir, f"{prefix}_negative_controls_{population_name}.png")
        _plot_sav(sav_cases.get(population_name, pd.DataFrame()), figures_dir, prefix, population_name)

    _write_reports(
        single_metrics,
        few_summary,
        scaffold_summary,
        near_metrics,
        negative_summary,
        crossover,
        sav_summary_all,
        "; ".join(native_warnings) if native_warnings else None,
        reports_dir,
        prefix,
    )

    all_active_full = single_metrics[
        single_metrics["population"].eq("full") & single_metrics["method"].eq("all_active_loo_ecfp4_analog_neighborhood")
    ]
    native_full = single_metrics[
        single_metrics["population"].eq("full") & single_metrics["method"].eq("native_ligand_ecfp4")
    ]
    return {
        "full_population_count": int(len(audit_populations["full"])),
        "clean_population_count": int(len(audit_populations["clean"])),
        "all_active_loo_full_ef1": float(all_active_full["ef1"].iloc[0]) if not all_active_full.empty else None,
        "all_active_loo_full_pr_auc": float(all_active_full["pr_auc"].iloc[0]) if not all_active_full.empty else None,
        "native_ligand_baseline": "ran" if not native_full.empty else "skipped",
        "few_active_rows": int(len(few_repeats)),
        "scaffold_holdout_rows": int(len(scaffold_repeats)),
        "negative_control_rows": int(len(negative_metrics)),
        "sav_active_cases_full": int(len(sav_cases.get("full", pd.DataFrame()))),
    }


def main() -> None:
    args = parse_args()
    summary = run(
        audit_config=Path(args.audit_config),
        score_population=Path(args.score_population),
        full_population=Path(args.full_population),
        clean_population=Path(args.clean_population),
        metrics_config=Path(args.metrics_config),
        tables_dir=Path(args.tables_dir),
        figures_dir=Path(args.figures_dir),
        reports_dir=Path(args.reports_dir),
        prefix=args.out_prefix,
    )
    LOGGER.info("Step 7b complete: %s", summary)


if __name__ == "__main__":
    main()
