from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from screensift.common.io import ensure_dir, load_yaml
from screensift.common.logging_utils import setup_logger
from screensift.metrics.anomaly_utils import load_or_reconstruct_anomaly_flags
from screensift.metrics.enrichment import make_rank_score
from screensift.metrics.evaluate_phase1_scores import evaluate_method
from screensift.metrics.ligand_baseline import compute_leave_one_active_out_similarity
from screensift.metrics.rank_fusion import mean_percentile_fusion, percentile_rank_score
from screensift.metrics.scaffold_leakage import write_scaffold_outputs


LOGGER = setup_logger(__name__)

BASE_METHODS = [
    {"method": "ecfp4_active_similarity", "raw_score_col": "ecfp4_active_similarity", "ranking_col": "rankscore_ecfp4_active_similarity", "higher_is_better": True},
    {"method": "unidock_best", "raw_score_col": "unidock_best_score", "ranking_col": "rankscore_unidock_best", "higher_is_better": False},
    {"method": "gnina_cnnscore", "raw_score_col": "CNNscore", "ranking_col": "rankscore_gnina_cnnscore", "higher_is_better": True},
    {"method": "gnina_cnnaffinity", "raw_score_col": "CNNaffinity", "ranking_col": "rankscore_gnina_cnnaffinity", "higher_is_better": True},
    {"method": "gnina_cnn_vs", "raw_score_col": "CNN_VS", "ranking_col": "rankscore_gnina_cnn_vs", "higher_is_better": True},
    {"method": "gnina_affinity", "raw_score_col": "gnina_affinity", "ranking_col": "rankscore_gnina_affinity", "higher_is_better": False},
    {"method": "fusion_unidock_cnnscore", "raw_score_col": "fusion_unidock_cnnscore", "ranking_col": "fusion_unidock_cnnscore", "higher_is_better": True},
    {"method": "fusion_unidock_cnnaffinity", "raw_score_col": "fusion_unidock_cnnaffinity", "ranking_col": "fusion_unidock_cnnaffinity", "higher_is_better": True},
    {"method": "fusion_cnnscore_cnnaffinity", "raw_score_col": "fusion_cnnscore_cnnaffinity", "ranking_col": "fusion_cnnscore_cnnaffinity", "higher_is_better": True},
    {"method": "fusion_unidock_cnnscore_cnnaffinity", "raw_score_col": "fusion_unidock_cnnscore_cnnaffinity", "ranking_col": "fusion_unidock_cnnscore_cnnaffinity", "higher_is_better": True},
]

ECFP4_FUSION_METHODS = [
    {"method": "fusion_unidock_ecfp4", "raw_score_col": "fusion_unidock_ecfp4", "ranking_col": "fusion_unidock_ecfp4", "higher_is_better": True},
    {"method": "fusion_cnnscore_ecfp4", "raw_score_col": "fusion_cnnscore_ecfp4", "ranking_col": "fusion_cnnscore_ecfp4", "higher_is_better": True},
    {"method": "fusion_cnnaffinity_ecfp4", "raw_score_col": "fusion_cnnaffinity_ecfp4", "ranking_col": "fusion_cnnaffinity_ecfp4", "higher_is_better": True},
    {"method": "fusion_unidock_cnnscore_ecfp4", "raw_score_col": "fusion_unidock_cnnscore_ecfp4", "ranking_col": "fusion_unidock_cnnscore_ecfp4", "higher_is_better": True},
    {"method": "fusion_unidock_cnnscore_cnnaffinity_ecfp4", "raw_score_col": "fusion_unidock_cnnscore_cnnaffinity_ecfp4", "ranking_col": "fusion_unidock_cnnscore_cnnaffinity_ecfp4", "higher_is_better": True},
]

STRUCTURE_METHODS_FOR_NOVELTY = [
    "unidock_best",
    "gnina_cnnscore",
    "gnina_cnnaffinity",
    "gnina_cnn_vs",
    "gnina_affinity",
    "fusion_unidock_cnnscore",
    "fusion_unidock_cnnaffinity",
    "fusion_unidock_cnnscore_cnnaffinity",
    "fusion_unidock_cnnscore_cnnaffinity_ecfp4",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ligand-only ECFP4 baseline and leakage analyses.")
    parser.add_argument("--baseline-config", default="configs/ligand_baseline.yml")
    parser.add_argument("--score-population", default="example/mapk1/expected_outputs/mapk1_2receptor_score_population.csv")
    parser.add_argument("--metrics-config", default="configs/metrics.yml")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--reports-dir", default="results/reports")
    parser.add_argument("--out-prefix", default="mapk1_phase1")
    return parser.parse_args()


def _compute_heavy_atoms_from_smiles(smiles: pd.Series) -> pd.Series:
    from rdkit import Chem

    def count_atoms(value: Any) -> float:
        if pd.isna(value) or not str(value).strip():
            return float("nan")
        mol = Chem.MolFromSmiles(str(value))
        return float(mol.GetNumHeavyAtoms()) if mol is not None else float("nan")

    return smiles.map(count_atoms)


def load_score_population(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Step 6 score population not found: {path}")
    df = pd.read_csv(path, dtype={"ligand_id": str}, low_memory=False)
    if "ligand_id" not in df.columns:
        df["ligand_id"] = [f"ligand_{i:07d}" for i in range(len(df))]
    smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else "smiles" if "smiles" in df.columns else None
    if smiles_col is None:
        raise ValueError("Expected canonical_smiles or smiles in Step 6 score population.")
    if "canonical_smiles" not in df.columns:
        df["canonical_smiles"] = df[smiles_col]
    if "CNN_VS" not in df.columns and {"CNNscore", "CNNaffinity"}.issubset(df.columns):
        df["CNN_VS"] = pd.to_numeric(df["CNNscore"], errors="coerce") * pd.to_numeric(df["CNNaffinity"], errors="coerce")
    required = ["activity_label", "unidock_best_score", "CNNscore", "CNNaffinity", "CNN_VS", "gnina_affinity"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Step 6 score population missing required columns: {missing}")
    if "heavy_atoms" not in df.columns:
        df["heavy_atoms"] = np.nan
    if df["heavy_atoms"].isna().any():
        df["heavy_atoms"] = pd.to_numeric(df["heavy_atoms"], errors="coerce").combine_first(
            _compute_heavy_atoms_from_smiles(df["canonical_smiles"])
        )
    if "is_active" not in df.columns:
        df["is_active"] = df["activity_label"].astype(str).str.lower().eq("active").astype(int)
    return df


def add_ecfp4_similarity(population: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    fp_config = config.get("fingerprint", {})
    similarity = compute_leave_one_active_out_similarity(
        population,
        smiles_col="canonical_smiles",
        activity_col="activity_label",
        id_cols=["ligand_id", "canonical_smiles", "inchikey"],
        radius=int(fp_config.get("radius", 2)),
        n_bits=int(fp_config.get("n_bits", 2048)),
        use_chirality=bool(fp_config.get("use_chirality", False)),
        novelty_thresholds=config.get("novelty_thresholds", {}),
    )
    keep = [
        "ligand_id",
        "ecfp4_active_similarity",
        "ecfp4_nearest_active_ligand_id",
        "ecfp4_nearest_active_similarity",
        "ecfp4_similarity_bin",
        "ecfp4_valid",
        "ecfp4_failure_reason",
    ]
    return population.drop(columns=[col for col in keep if col != "ligand_id" and col in population.columns], errors="ignore").merge(
        similarity[keep],
        on="ligand_id",
        how="left",
    )


def add_rank_and_ecfp4_fusions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rankscore_unidock_best"] = make_rank_score(out["unidock_best_score"], "lower")
    out["rankscore_gnina_cnnscore"] = make_rank_score(out["CNNscore"], "higher")
    out["rankscore_gnina_cnnaffinity"] = make_rank_score(out["CNNaffinity"], "higher")
    out["rankscore_gnina_cnn_vs"] = make_rank_score(out["CNN_VS"], "higher")
    out["rankscore_gnina_affinity"] = make_rank_score(out["gnina_affinity"], "lower")
    out["rankscore_ecfp4_active_similarity"] = make_rank_score(out["ecfp4_active_similarity"], "higher")

    pct_specs = [
        ("pct_unidock_best", "unidock_best_score", "lower"),
        ("pct_gnina_cnnscore", "CNNscore", "higher"),
        ("pct_gnina_cnnaffinity", "CNNaffinity", "higher"),
        ("pct_gnina_cnn_vs", "CNN_VS", "higher"),
        ("pct_gnina_affinity", "gnina_affinity", "lower"),
        ("pct_ecfp4_active_similarity", "ecfp4_active_similarity", "higher"),
    ]
    for pct_col, score_col, direction in pct_specs:
        out[pct_col] = percentile_rank_score(out, score_col, direction)

    base_fusions = {
        "fusion_unidock_cnnscore": ["pct_unidock_best", "pct_gnina_cnnscore"],
        "fusion_unidock_cnnaffinity": ["pct_unidock_best", "pct_gnina_cnnaffinity"],
        "fusion_cnnscore_cnnaffinity": ["pct_gnina_cnnscore", "pct_gnina_cnnaffinity"],
        "fusion_unidock_cnnscore_cnnaffinity": [
            "pct_unidock_best",
            "pct_gnina_cnnscore",
            "pct_gnina_cnnaffinity",
        ],
        "fusion_unidock_ecfp4": ["pct_unidock_best", "pct_ecfp4_active_similarity"],
        "fusion_cnnscore_ecfp4": ["pct_gnina_cnnscore", "pct_ecfp4_active_similarity"],
        "fusion_cnnaffinity_ecfp4": ["pct_gnina_cnnaffinity", "pct_ecfp4_active_similarity"],
        "fusion_unidock_cnnscore_ecfp4": [
            "pct_unidock_best",
            "pct_gnina_cnnscore",
            "pct_ecfp4_active_similarity",
        ],
        "fusion_unidock_cnnscore_cnnaffinity_ecfp4": [
            "pct_unidock_best",
            "pct_gnina_cnnscore",
            "pct_gnina_cnnaffinity",
            "pct_ecfp4_active_similarity",
        ],
    }
    for name, cols in base_fusions.items():
        out[name] = mean_percentile_fusion(out, cols)
    return out


def evaluate_specs(df: pd.DataFrame, population_name: str, specs: list[dict[str, Any]], config: dict[str, Any], notes: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_boot = int(config.get("bootstrap_iterations", 1000))
    confidence = 0.95
    seed = int(config.get("random_seed", 42))
    for spec in specs:
        if spec["ranking_col"] not in df.columns:
            continue
        row, _ = evaluate_method(
            df,
            population_name,
            spec["method"],
            spec["raw_score_col"],
            spec["ranking_col"],
            spec["higher_is_better"],
            n_boot=n_boot,
            confidence=confidence,
            seed=seed,
            notes=notes,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_structure_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    method_to_col = {
        "unidock_best": "rankscore_unidock_best",
        "gnina_cnnscore": "rankscore_gnina_cnnscore",
        "gnina_cnnaffinity": "rankscore_gnina_cnnaffinity",
        "gnina_cnn_vs": "rankscore_gnina_cnn_vs",
        "gnina_affinity": "rankscore_gnina_affinity",
        "fusion_unidock_cnnscore": "fusion_unidock_cnnscore",
        "fusion_unidock_cnnaffinity": "fusion_unidock_cnnaffinity",
        "fusion_unidock_cnnscore_cnnaffinity": "fusion_unidock_cnnscore_cnnaffinity",
        "fusion_unidock_cnnscore_cnnaffinity_ecfp4": "fusion_unidock_cnnscore_cnnaffinity_ecfp4",
        "ecfp4_active_similarity": "rankscore_ecfp4_active_similarity",
    }
    for method, col in method_to_col.items():
        if col in out.columns:
            out[f"rank_{method}"] = pd.to_numeric(out[col], errors="coerce").rank(method="first", ascending=False)
    rank_cols = [f"rank_{method}" for method in STRUCTURE_METHODS_FOR_NOVELTY if f"rank_{method}" in out.columns]
    n = len(out)
    top1 = max(1, math.ceil(n * 0.01))
    top5 = max(1, math.ceil(n * 0.05))
    out["best_structure_rank"] = out[rank_cols].min(axis=1) if rank_cols else np.nan
    out["structure_top1pct"] = out["best_structure_rank"].le(top1)
    out["structure_top5pct"] = out["best_structure_rank"].le(top5)
    out["structure_top100"] = out["best_structure_rank"].le(min(100, n))
    out["structure_high_ranked"] = out[["structure_top1pct", "structure_top5pct", "structure_top100"]].any(axis=1)
    return out


def write_case_tables(df: pd.DataFrame, tables_dir: Path, prefix: str, suffix: str) -> dict[str, int]:
    ranked = add_structure_ranks(df)
    common_cols = [
        "ligand_id",
        "activity_label",
        "canonical_smiles",
        "scaffold_smiles",
        "ecfp4_active_similarity",
        "ecfp4_similarity_bin",
        "unidock_best_score",
        "CNNscore",
        "CNNaffinity",
        "CNN_VS",
        "gnina_affinity",
        "best_structure_rank",
        "structure_top1pct",
        "structure_top5pct",
        "structure_top100",
        "anomaly_flags",
        "clean_population_excluded",
    ]
    rank_cols = [col for col in ranked.columns if col.startswith("rank_")]
    cols = [col for col in common_cols + rank_cols if col in ranked.columns]

    novel = ranked[
        ranked["is_active"].eq(1)
        & ranked["ecfp4_active_similarity"].lt(0.50)
        & ranked["structure_high_ranked"]
    ].copy()
    novel["requires_manual_inspection"] = novel["has_any_score_anomaly"].astype(bool)
    novel.sort_values(["ecfp4_active_similarity", "best_structure_rank"], ascending=[True, True]).to_csv(
        tables_dir / f"{prefix}_novel_actives_{suffix}.csv",
        index=False,
        columns=[col for col in cols + ["requires_manual_inspection"] if col in novel.columns],
    )

    similarity_driven = ranked[
        ranked["ecfp4_active_similarity"].ge(0.70) & ranked["structure_high_ranked"]
    ].copy()
    similarity_driven.sort_values(["ecfp4_active_similarity", "best_structure_rank"], ascending=[False, True]).to_csv(
        tables_dir / f"{prefix}_similarity_driven_hits_{suffix}.csv",
        index=False,
        columns=cols,
    )

    high_rank_flags = pd.DataFrame(index=ranked.index)
    for method in STRUCTURE_METHODS_FOR_NOVELTY:
        col = f"rank_{method}"
        if col in ranked.columns:
            high_rank_flags[method] = ranked[col].le(100)
    ranked["n_high_rank_methods"] = high_rank_flags.sum(axis=1) if not high_rank_flags.empty else 0
    false_pos = ranked[ranked["is_active"].eq(0) & ranked["n_high_rank_methods"].ge(2)].copy()
    false_pos["high_ecfp4_explains"] = false_pos["ecfp4_active_similarity"].ge(0.70)
    false_pos.sort_values(["n_high_rank_methods", "ecfp4_active_similarity"], ascending=[False, False]).to_csv(
        tables_dir / f"{prefix}_false_positive_consensus_{suffix}.csv",
        index=False,
        columns=[col for col in cols + ["n_high_rank_methods", "high_ecfp4_explains"] if col in false_pos.columns],
    )

    active_fn = ranked[
        ranked["is_active"].eq(1)
        & ranked["best_structure_rank"].gt(max(100, math.ceil(len(ranked) * 0.05)))
        & ranked["ecfp4_active_similarity"].rank(method="first", ascending=False).gt(max(100, math.ceil(len(ranked) * 0.05)))
    ].copy()
    active_fn.sort_values(["best_structure_rank", "ecfp4_active_similarity"], ascending=[False, True]).to_csv(
        tables_dir / f"{prefix}_active_false_negatives_{suffix}.csv",
        index=False,
        columns=cols,
    )
    return {
        "novel_actives": int(len(novel)),
        "similarity_driven_hits": int(len(similarity_driven)),
        "consensus_inactive_false_positives": int(len(false_pos)),
        "active_false_negatives": int(len(active_fn)),
    }


def write_anomaly_top_hit_intersection(df: pd.DataFrame, tables_dir: Path, prefix: str) -> int:
    ranked = add_structure_ranks(df)
    method_rank_cols = [col for col in ranked.columns if col.startswith("rank_")]
    n = len(ranked)
    top1 = max(1, math.ceil(n * 0.01))
    top5 = max(1, math.ceil(n * 0.05))
    ranked["top1_any_method"] = ranked[method_rank_cols].le(top1).any(axis=1)
    ranked["top5_any_method"] = ranked[method_rank_cols].le(top5).any(axis=1)
    ranked["top100_any_method"] = ranked[method_rank_cols].le(min(100, n)).any(axis=1)
    intersection = ranked[
        ranked["has_any_score_anomaly"] & ranked[["top1_any_method", "top5_any_method", "top100_any_method"]].any(axis=1)
    ].copy()
    cols = [
        "ligand_id",
        "activity_label",
        "canonical_smiles",
        "ecfp4_active_similarity",
        "unidock_best_score",
        "CNNscore",
        "CNNaffinity",
        "CNN_VS",
        "gnina_affinity",
        "anomaly_flags",
        "top1_any_method",
        "top5_any_method",
        "top100_any_method",
    ] + method_rank_cols
    intersection.sort_values(["top1_any_method", "top5_any_method", "unidock_best_score"], ascending=[False, False, True]).to_csv(
        tables_dir / f"{prefix}_score_anomalies_in_top_hits.csv",
        index=False,
        columns=[col for col in cols if col in intersection.columns],
    )
    return int(len(intersection))


def plot_outputs(full: pd.DataFrame, clean: pd.DataFrame, full_metrics: pd.DataFrame, clean_metrics: pd.DataFrame, scaffold: pd.DataFrame, tables_dir: Path, figures_dir: Path, prefix: str) -> None:
    active = full["is_active"].eq(1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(full.loc[~active, "ecfp4_active_similarity"].dropna(), bins=40, alpha=0.55, label="inactive")
    ax.hist(full.loc[active, "ecfp4_active_similarity"].dropna(), bins=40, alpha=0.55, label="active")
    ax.set_xlabel("ECFP4 max active similarity")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}_ecfp4_similarity_distribution_by_activity.png", dpi=160)
    plt.close(fig)

    for score_col, out_name, ylabel in [
        ("unidock_best_score", f"{prefix}_ecfp4_vs_unidock_scatter.png", "Uni-Dock best score"),
        ("CNNscore", f"{prefix}_ecfp4_vs_cnnscore_scatter.png", "CNNscore"),
        ("CNN_VS", f"{prefix}_ecfp4_vs_cnn_vs_scatter.png", "CNN_VS"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(full.loc[~active, "ecfp4_active_similarity"], full.loc[~active, score_col], s=8, alpha=0.35, label="inactive")
        ax.scatter(full.loc[active, "ecfp4_active_similarity"], full.loc[active, score_col], s=14, alpha=0.75, label="active")
        ax.set_xlabel("ECFP4 max active similarity")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / out_name, dpi=160)
        plt.close(fig)

    for metrics, suffix in [(full_metrics, "full"), (clean_metrics, "clean")]:
        for metric_col, label in [("ef1", "EF1%"), ("pr_auc", "PR-AUC")]:
            subset = metrics[metrics["population"].eq(suffix)].sort_values(metric_col, ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(subset))
            ax.bar(x, pd.to_numeric(subset[metric_col], errors="coerce"))
            ax.set_xticks(x, subset["method"], rotation=45, ha="right")
            ax.set_ylabel(label)
            fig.tight_layout()
            fig.savefig(figures_dir / f"{prefix}_all_methods_{metric_col}_comparison_{suffix}.png", dpi=160)
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(scaffold["active_fraction"].dropna(), bins=30)
    ax.set_xlabel("Scaffold active fraction")
    ax.set_ylabel("Scaffold count")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}_scaffold_active_fraction_histogram.png", dpi=160)
    plt.close(fig)

    anomaly_top = pd.read_csv(tables_dir / f"{prefix}_score_anomalies_in_top_hits.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    values = [
        int(anomaly_top["top1_any_method"].sum()) if not anomaly_top.empty else 0,
        int(anomaly_top["top5_any_method"].sum()) if not anomaly_top.empty else 0,
        int(anomaly_top["top100_any_method"].sum()) if not anomaly_top.empty else 0,
    ]
    ax.bar(["top1%", "top5%", "top100"], values)
    ax.set_ylabel("Anomalous rows")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}_score_anomaly_top_hit_overlap.png", dpi=160)
    plt.close(fig)


def _fmt_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "(none)"
    return "```text\n" + df.head(max_rows).to_string(index=False) + "\n```"


def write_reports(
    full: pd.DataFrame,
    clean: pd.DataFrame,
    full_metrics: pd.DataFrame,
    clean_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    scaffold_summary: pd.DataFrame,
    case_counts_full: dict[str, int],
    case_counts_clean: dict[str, int],
    anomaly_top_count: int,
    reports_dir: Path,
    prefix: str,
) -> None:
    full_main = full_metrics[full_metrics["population"].eq("full")]
    clean_main = clean_metrics[clean_metrics["population"].eq("clean")]
    ecfp_full = full_main[full_main["method"].eq("ecfp4_active_similarity")]
    ecfp_clean = clean_main[clean_main["method"].eq("ecfp4_active_similarity")]
    best_full_ef = full_main.sort_values("ef1", ascending=False).iloc[0]["method"]
    best_full_pr = full_main.sort_values("pr_auc", ascending=False).iloc[0]["method"]
    severe_excluded = int(full["clean_population_excluded"].sum())
    anomaly_rows = int(full["has_any_score_anomaly"].sum())

    ligand_report = f"""# MAPK1 Phase 1 Ligand Baseline Report

## Why This Baseline Exists

The ECFP4/Tanimoto baseline tests whether 2D ligand similarity to known actives explains active recovery better than structure-based scoring. This is mandatory for leakage-aware interpretation.

For active ligands, the nearest-active calculation uses leave-one-active-out matching and excludes self matches by ligand ID, canonical SMILES, and InChIKey where available.

## Full-Population ECFP4 Metrics

{_fmt_table(ecfp_full[["method", "n_total", "n_active", "roc_auc", "pr_auc", "ef1", "ef5", "ef10", "top50_actives", "top100_actives"]])}

## Clean-Population ECFP4 Metrics

{_fmt_table(ecfp_clean[["method", "n_total", "n_active", "roc_auc", "pr_auc", "ef1", "ef5", "ef10", "top50_actives", "top100_actives"]])}

## Comparison Against Structure-Based Methods

{_fmt_table(comparison[["method", "population_type", "roc_auc", "pr_auc", "ef1", "ef5", "ef10", "top50_actives", "top100_actives", "notes"]], max_rows=60)}

## Conservative Interpretation

- Best full-population method by EF1%: {best_full_ef}
- Best full-population method by PR-AUC: {best_full_pr}
- ECFP4 and ECFP4 fusions should be interpreted as leakage/analog-bias checks, not as mechanistic docking evidence.
"""

    anomaly_report = f"""# MAPK1 Phase 1 Anomaly-Aware Ligand Baseline Report

Anomalous scores were not deleted from the canonical analysis; they were flagged and handled by sensitivity analysis.

## Inherited Step 6 Anomalies

- Rows with any score anomaly: {anomaly_rows}
- Rows excluded from clean sensitivity analysis: {severe_excluded}
- Clean population size: {len(clean)}
- Full population size: {len(full)}

Clean-population exclusion flags were severe-only flags. Rows with only `positive_gnina_affinity` were retained unless they also had `extreme_positive_gnina_affinity`.

## Sensitivity Comparison

{_fmt_table(pd.concat([full_main.assign(population_type="full"), clean_main.assign(population_type="clean")], ignore_index=True)[["method", "population_type", "n_total", "n_active", "pr_auc", "ef1", "ef5", "top50_actives"]], max_rows=80)}

The extreme Uni-Dock -29-like row and extreme positive GNINA affinity rows are preserved in the full analysis and marked for manual inspection if they appear among top hits.
"""

    novelty_report = f"""# MAPK1 Phase 1 Novelty And Failure Cases

## Case Counts

```text
full_population:
  novel_actives: {case_counts_full["novel_actives"]}
  similarity_driven_hits: {case_counts_full["similarity_driven_hits"]}
  consensus_inactive_false_positives: {case_counts_full["consensus_inactive_false_positives"]}
  active_false_negatives: {case_counts_full["active_false_negatives"]}

clean_score_population:
  novel_actives: {case_counts_clean["novel_actives"]}
  similarity_driven_hits: {case_counts_clean["similarity_driven_hits"]}
  consensus_inactive_false_positives: {case_counts_clean["consensus_inactive_false_positives"]}
  active_false_negatives: {case_counts_clean["active_false_negatives"]}

score_anomalies_in_top_hits: {anomaly_top_count}
```

## Recommendations

- Manually inspect low-similarity actives recovered by structure-based methods.
- Treat high-similarity hits as possible analog-bias-driven cases.
- Inspect consensus inactive false positives before any hit prioritization.
- Inspect active false negatives for preparation, tautomer/protonation, and receptor-state issues.
- Inspect score-anomalous top hits before interpreting them as real hits.
"""

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{prefix}_ligand_baseline_report.md").write_text(ligand_report, encoding="utf-8")
    (reports_dir / f"{prefix}_anomaly_aware_ligand_baseline_report.md").write_text(anomaly_report, encoding="utf-8")
    (reports_dir / f"{prefix}_novelty_and_failure_cases.md").write_text(novelty_report, encoding="utf-8")


def run(
    baseline_config: Path,
    score_population: Path,
    metrics_config: Path,
    tables_dir: Path,
    figures_dir: Path,
    reports_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    config = load_yaml(baseline_config)
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)
    ensure_dir(reports_dir)

    population = load_score_population(score_population)
    population = load_or_reconstruct_anomaly_flags(
        population,
        tables_dir=tables_dir,
        metrics_config=metrics_config,
        clean_exclusion_flags=config.get("clean_population_exclusion_flags"),
    )

    full = add_ecfp4_similarity(population, config)
    full = add_rank_and_ecfp4_fusions(full)
    clean = full[~full["clean_population_excluded"].astype(bool)].copy()
    clean = add_ecfp4_similarity(clean.drop(columns=[col for col in full.columns if col.startswith("ecfp4_")], errors="ignore"), config)
    clean = add_rank_and_ecfp4_fusions(clean)

    full_annotated, scaffold_summary = write_scaffold_outputs(
        full,
        tables_dir / f"{prefix}_scaffold_leakage.csv",
        reports_dir / f"{prefix}_scaffold_leakage_report.md",
    )
    full["scaffold_smiles"] = full_annotated["scaffold_smiles"]
    clean_scaffold = full[["ligand_id", "scaffold_smiles"]]
    clean = clean.merge(clean_scaffold, on="ligand_id", how="left", suffixes=("", "_full"))

    full.to_csv(tables_dir / f"{prefix}_full_population_with_ecfp4_and_anomalies.csv", index=False)
    clean.to_csv(tables_dir / f"{prefix}_clean_population_with_ecfp4_and_anomalies.csv", index=False)

    ecfp_cols = [
        "ligand_id",
        "activity_label",
        "canonical_smiles",
        "inchikey",
        "ecfp4_active_similarity",
        "ecfp4_nearest_active_ligand_id",
        "ecfp4_nearest_active_similarity",
        "ecfp4_similarity_bin",
        "ecfp4_valid",
        "ecfp4_failure_reason",
    ]
    full[[col for col in ecfp_cols if col in full.columns]].to_csv(tables_dir / f"{prefix}_ecfp4_scores_full.csv", index=False)
    clean[[col for col in ecfp_cols if col in clean.columns]].to_csv(tables_dir / f"{prefix}_ecfp4_scores_clean.csv", index=False)

    fusion_cols = [
        "ligand_id",
        "activity_label",
        "ecfp4_active_similarity",
        "pct_ecfp4_active_similarity",
        "fusion_unidock_ecfp4",
        "fusion_cnnscore_ecfp4",
        "fusion_cnnaffinity_ecfp4",
        "fusion_unidock_cnnscore_ecfp4",
        "fusion_unidock_cnnscore_cnnaffinity_ecfp4",
        "anomaly_flags",
        "clean_population_excluded",
    ]
    full[[col for col in fusion_cols if col in full.columns]].to_csv(tables_dir / f"{prefix}_ecfp4_fusion_scores_full.csv", index=False)
    clean[[col for col in fusion_cols if col in clean.columns]].to_csv(tables_dir / f"{prefix}_ecfp4_fusion_scores_clean.csv", index=False)

    all_specs = BASE_METHODS + ECFP4_FUSION_METHODS
    fusion_specs = ECFP4_FUSION_METHODS
    full_metrics = evaluate_specs(full, "full", all_specs, config, "Full canonical population; score anomalies retained.")
    clean_metrics = evaluate_specs(clean, "clean", all_specs, config, "Clean score population; severe anomalies excluded for sensitivity only.")
    full_fusion_metrics = evaluate_specs(full, "full", fusion_specs, config, "ECFP4 fusion methods on full population.")
    clean_fusion_metrics = evaluate_specs(clean, "clean", fusion_specs, config, "ECFP4 fusion methods on clean sensitivity population.")

    full_metrics.to_csv(tables_dir / f"{prefix}_all_method_metrics_with_ecfp4_full.csv", index=False)
    clean_metrics.to_csv(tables_dir / f"{prefix}_all_method_metrics_with_ecfp4_clean.csv", index=False)
    full_fusion_metrics.to_csv(tables_dir / f"{prefix}_ecfp4_fusion_metrics_full.csv", index=False)
    clean_fusion_metrics.to_csv(tables_dir / f"{prefix}_ecfp4_fusion_metrics_clean.csv", index=False)

    comparison = pd.concat(
        [
            full_metrics.assign(population_type="full"),
            clean_metrics.assign(population_type="clean"),
        ],
        ignore_index=True,
    )
    if (tables_dir / f"{prefix}_method_metrics.csv").exists():
        comparison["notes"] = comparison["notes"].fillna("") + " Step 6 metrics table was present for comparison."
    comparison[[
        "method",
        "population_type",
        "roc_auc",
        "pr_auc",
        "ef1",
        "ef5",
        "ef10",
        "top50_actives",
        "top100_actives",
        "notes",
    ]].to_csv(tables_dir / f"{prefix}_step6_vs_ecfp4_comparison.csv", index=False)

    case_counts_full = write_case_tables(full, tables_dir, prefix, "full")
    case_counts_clean = write_case_tables(clean, tables_dir, prefix, "clean")
    anomaly_top_count = write_anomaly_top_hit_intersection(full, tables_dir, prefix)
    plot_outputs(full, clean, full_metrics, clean_metrics, scaffold_summary, tables_dir, figures_dir, prefix)
    write_reports(
        full,
        clean,
        full_metrics,
        clean_metrics,
        comparison,
        scaffold_summary,
        case_counts_full,
        case_counts_clean,
        anomaly_top_count,
        reports_dir,
        prefix,
    )

    return {
        "full_n_total": int(len(full)),
        "full_n_active": int(full["is_active"].sum()),
        "full_n_inactive": int((full["is_active"] == 0).sum()),
        "clean_n_total": int(len(clean)),
        "clean_n_active": int(clean["is_active"].sum()),
        "clean_n_inactive": int((clean["is_active"] == 0).sum()),
        "score_anomaly_rows": int(full["has_any_score_anomaly"].sum()),
        "clean_excluded_rows": int(full["clean_population_excluded"].sum()),
        "full_best_ef1": full_metrics.sort_values("ef1", ascending=False).iloc[0]["method"],
        "full_best_pr_auc": full_metrics.sort_values("pr_auc", ascending=False).iloc[0]["method"],
        "novel_actives_full": case_counts_full["novel_actives"],
        "consensus_false_positives_full": case_counts_full["consensus_inactive_false_positives"],
        "anomaly_top_hit_rows": anomaly_top_count,
    }


def main() -> None:
    args = parse_args()
    summary = run(
        baseline_config=Path(args.baseline_config),
        score_population=Path(args.score_population),
        metrics_config=Path(args.metrics_config),
        tables_dir=Path(args.tables_dir),
        figures_dir=Path(args.figures_dir),
        reports_dir=Path(args.reports_dir),
        prefix=args.out_prefix,
    )
    LOGGER.info("Step 7 complete: %s", summary)


if __name__ == "__main__":
    main()
