from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from screensift.common.io import ensure_dir


BASE_METHODS = {
    "unidock_best": "rankscore_unidock_best",
    "gnina_cnnscore": "rankscore_gnina_cnnscore",
    "gnina_cnnaffinity": "rankscore_gnina_cnnaffinity",
    "gnina_cnn_vs": "rankscore_gnina_cnn_vs",
    "gnina_affinity": "rankscore_gnina_affinity",
    "fusion_unidock_cnnscore_cnnaffinity": "fusion_unidock_cnnscore_cnnaffinity",
}

RANKSCORE_LABELS = {
    "rankscore_unidock_best": "Uni-Dock\nhigher=better",
    "rankscore_gnina_cnnscore": "CNNscore\nhigher=better",
    "rankscore_gnina_cnnaffinity": "CNNaffinity\nhigher=better",
    "rankscore_gnina_affinity": "GNINA affinity\nhigher=better",
}

RAW_SCORE_LABELS = {
    "unidock_best_score": "Uni-Dock",
    "CNNscore": "CNNscore",
    "CNNaffinity": "CNNaffinity",
    "CNN_VS": "CNN_VS",
    "gnina_affinity": "GNINA affinity",
}

SEVERE_ANOMALY_FLAGS = [
    "extreme_unidock_negative",
    "suspicious_unidock_extreme_negative",
    "suspicious_ligand_efficiency_extreme",
    "extreme_positive_gnina_affinity",
    "out_of_range_cnnscore",
    "suspicious_cnnaffinity_extreme",
]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _valid_xy(df: pd.DataFrame, score_col: str) -> tuple[np.ndarray, np.ndarray]:
    y = _numeric(df["is_active"])
    score = _numeric(df[score_col])
    mask = y.notna() & score.notna()
    return y[mask].astype(int).to_numpy(), score[mask].to_numpy(dtype=float)


def plot_score_correlation_heatmap(population: pd.DataFrame, out: Path) -> None:
    cols = [
        "rankscore_unidock_best",
        "rankscore_gnina_cnnscore",
        "rankscore_gnina_cnnaffinity",
        "rankscore_gnina_affinity",
    ]
    work = population.copy()
    if "rankscore_unidock_best" not in work.columns and "unidock_best_score" in work.columns:
        work["rankscore_unidock_best"] = -_numeric(work["unidock_best_score"])
    if "rankscore_gnina_cnnscore" not in work.columns and "CNNscore" in work.columns:
        work["rankscore_gnina_cnnscore"] = _numeric(work["CNNscore"])
    if "rankscore_gnina_cnnaffinity" not in work.columns and "CNNaffinity" in work.columns:
        work["rankscore_gnina_cnnaffinity"] = _numeric(work["CNNaffinity"])
    if "rankscore_gnina_affinity" not in work.columns and "gnina_affinity" in work.columns:
        work["rankscore_gnina_affinity"] = -_numeric(work["gnina_affinity"])

    available = [col for col in cols if col in work.columns]
    corr = work[available].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1)
    labels = [RANKSCORE_LABELS.get(col, col) for col in available]
    ax.set_xticks(range(len(available)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(available)), labels)
    for i in range(len(available)):
        for j in range(len(available)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center")
    ax.set_title("Spearman correlation of direction-standardized rankscores")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_roc_curves(population: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for method, score_col in BASE_METHODS.items():
        if score_col not in population.columns:
            continue
        y, score = _valid_xy(population, score_col)
        if len(set(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(fpr, tpr, label=method)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_pr_curves(population: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for method, score_col in BASE_METHODS.items():
        if score_col not in population.columns:
            continue
        y, score = _valid_xy(population, score_col)
        if len(set(y)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y, score)
        ax.plot(recall, precision, label=method)
    prevalence = float(pd.to_numeric(population["is_active"], errors="coerce").mean())
    ax.axhline(prevalence, linestyle="--", linewidth=1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_ef_barplot(metrics: pd.DataFrame, out: Path) -> None:
    main = metrics[metrics["population"].eq("main")].copy()
    main = main.sort_values("ef1", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(main))
    ax.bar(x, pd.to_numeric(main["ef1"], errors="coerce"))
    ax.set_xticks(x, main["method"], rotation=45, ha="right")
    ax.set_ylabel("EF1%")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _without_severe_score_anomalies(population: pd.DataFrame) -> pd.DataFrame:
    available = [flag for flag in SEVERE_ANOMALY_FLAGS if flag in population.columns]
    if not available:
        return population.copy()
    mask = population[available].fillna(False).astype(bool).any(axis=1)
    return population.loc[~mask].copy()


def plot_score_distributions(population: pd.DataFrame, out: Path, title_suffix: str = "") -> None:
    score_cols = ["unidock_best_score", "CNNscore", "CNNaffinity", "CNN_VS", "gnina_affinity"]
    n_cols = 3
    n_rows = int(np.ceil(len(score_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 7))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes_flat, score_cols):
        if col not in population.columns:
            ax.set_visible(False)
            continue
        active = _numeric(population.loc[population["is_active"].eq(1), col]).dropna()
        inactive = _numeric(population.loc[population["is_active"].eq(0), col]).dropna()
        ax.hist(inactive, bins=40, alpha=0.55, label="inactive")
        ax.hist(active, bins=40, alpha=0.55, label="active")
        ax.set_title(f"{RAW_SCORE_LABELS.get(col, col)}{title_suffix}")
        ax.legend(fontsize=8)
    for ax in axes_flat[len(score_cols):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_rank_disagreement(population: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    x = _numeric(population["pct_unidock_best"])
    y = _numeric(population["pct_gnina_cnnaffinity"])
    active = _numeric(population["is_active"]).fillna(0).astype(int)
    ax.scatter(x[active.eq(0)], y[active.eq(0)], s=8, alpha=0.35, label="inactive")
    ax.scatter(x[active.eq(1)], y[active.eq(1)], s=14, alpha=0.75, label="active")
    ax.set_xlabel("Uni-Dock percentile")
    ax.set_ylabel("CNNaffinity percentile")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def make_screening_plots(population: pd.DataFrame, metrics: pd.DataFrame, out_dir: str | Path, prefix: str) -> list[Path]:
    out_path = ensure_dir(out_dir)
    outputs = [
        out_path / f"{prefix}_score_correlation_heatmap.png",
        out_path / f"{prefix}_roc_curves.png",
        out_path / f"{prefix}_pr_curves.png",
        out_path / f"{prefix}_ef_barplot.png",
        out_path / f"{prefix}_score_distributions_by_activity.png",
        out_path / f"{prefix}_clean_score_distributions_by_activity.png",
        out_path / f"{prefix}_rank_disagreement_scatter.png",
    ]
    plot_score_correlation_heatmap(population, outputs[0])
    plot_roc_curves(population, outputs[1])
    plot_pr_curves(population, outputs[2])
    plot_ef_barplot(metrics, outputs[3])
    plot_score_distributions(population, outputs[4], title_suffix="\nfull incl. flags")
    plot_score_distributions(_without_severe_score_anomalies(population), outputs[5], title_suffix="\nsevere flags excluded")
    plot_rank_disagreement(population, outputs[6])
    return outputs
