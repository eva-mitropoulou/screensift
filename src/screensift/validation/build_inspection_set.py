from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from screensift.common.io import ensure_dir, load_yaml, write_json


OUTPUT_COLUMNS = [
    "ligand_id",
    "activity_label",
    "canonical_smiles",
    "inspection_categories",
    "manual_priority",
    "ecfp4_active_similarity",
    "ecfp4_similarity_bin",
    "unidock_best_score",
    "ligand_efficiency",
    "CNNscore",
    "CNNaffinity",
    "gnina_affinity",
    "anomaly_flags",
    "clean_population_excluded",
    "scaffold_smiles",
    "nearest_active_ligand_id",
    "best_structure_method",
    "best_structure_percentile",
    "SAV",
    "requires_manual_inspection",
    "reason_for_selection",
]

CATEGORY_PRIORITY = {
    "novel_sav_tanimoto_lt_0_30": 1,
    "novel_sav_tanimoto_lt_0_50": 2,
    "novel_sav_tanimoto_lt_0_70": 2,
    "low_similarity_active_tanimoto_lt_0_30": 3,
    "low_similarity_active_tanimoto_lt_0_50": 4,
    "extreme_unidock_negative_top_hit": 5,
    "score_anomaly_top_hit": 5,
    "consensus_inactive_false_positive": 6,
    "active_false_negative": 7,
    "similarity_driven_analog_bias_hit": 8,
}


@dataclass
class SelectedLigand:
    ligand_id: str
    values: dict[str, Any] = field(default_factory=dict)
    categories: set[str] = field(default_factory=set)
    reasons: set[str] = field(default_factory=set)
    priority: int = 999

    def add(self, category: str, reason: str, row: pd.Series | dict[str, Any], context: dict[str, Any]) -> None:
        self.categories.add(category)
        self.reasons.add(reason)
        self.priority = min(self.priority, CATEGORY_PRIORITY.get(category, 999))
        merged = {**context, **dict(row)}
        for key, value in merged.items():
            if key not in self.values or _is_missing(self.values[key]):
                self.values[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an inspection/failure shortlist.")
    parser.add_argument("--inspection-config", default="configs/inspection.yml")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument("--poses-dir", default="results/poses")
    parser.add_argument("--reports-dir", default="results/reports")
    parser.add_argument("--out-prefix", default="mapk1_phase1")
    return parser.parse_args()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def stable_ligand_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def read_optional_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "ligand_id" in df.columns:
        df["ligand_id"] = df["ligand_id"].map(stable_ligand_id)
    return df


def similarity_column(df: pd.DataFrame) -> str | None:
    for col in [
        "ecfp4_active_similarity",
        "ecfp4_similarity",
        "all_active_loo_ecfp4_analog_neighborhood",
        "native_ligand_ecfp4_ccd",
    ]:
        if col in df.columns:
            return col
    return None


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([float("nan")] * len(df), index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def string_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df[column].fillna("").astype(str)


def context_by_ligand(base: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if base.empty or "ligand_id" not in base.columns:
        return {}
    base = base.copy()
    base["ligand_id"] = base["ligand_id"].map(stable_ligand_id)
    return {
        stable_ligand_id(row["ligand_id"]): row.to_dict()
        for _, row in base.drop_duplicates("ligand_id").iterrows()
        if stable_ligand_id(row["ligand_id"])
    }


def sort_candidates(df: pd.DataFrame, preferred_cols: list[tuple[str, bool]]) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    sort_cols: list[str] = []
    ascending: list[bool] = []
    for col, asc in preferred_cols:
        if col in work.columns:
            sort_cols.append(col)
            ascending.append(asc)
    if not sort_cols:
        return work
    for col in sort_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    return work.sort_values(sort_cols, ascending=ascending, na_position="last")


def add_candidates(
    selected: dict[str, SelectedLigand],
    candidates: pd.DataFrame,
    category: str,
    reason: str,
    context: dict[str, dict[str, Any]],
    max_per_category: int,
) -> int:
    added = 0
    if candidates.empty or "ligand_id" not in candidates.columns:
        return 0
    for _, row in candidates.head(max_per_category).iterrows():
        ligand_id = stable_ligand_id(row.get("ligand_id"))
        if not ligand_id:
            continue
        item = selected.setdefault(ligand_id, SelectedLigand(ligand_id=ligand_id))
        item.add(category, reason, row, context.get(ligand_id, {}))
        added += 1
    return added


def category_counts(shortlist: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if shortlist.empty or "inspection_categories" not in shortlist.columns:
        return counts
    for categories in shortlist["inspection_categories"].fillna("").astype(str):
        for category in [part for part in categories.split(";") if part]:
            counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def _category_threshold_label(prefix: str, threshold: float) -> str:
    return f"{prefix}_tanimoto_lt_{threshold:.2f}".replace(".", "_")


def build_inspection_shortlist(tables_dir: Path, config: dict[str, Any], out_prefix: str = "mapk1_phase1") -> tuple[pd.DataFrame, dict[str, Any]]:
    limits = config.get("inspection_set_limits", {})
    max_total = int(limits.get("max_total_ligands", 100))
    max_per_category = int(limits.get("max_per_category", 15))
    selected: dict[str, SelectedLigand] = {}
    sources: dict[str, int] = {}

    base = read_optional_table(tables_dir / f"{out_prefix}_full_population_with_ecfp4_and_anomalies.csv")
    if base.empty:
        base = read_optional_table(tables_dir / f"{out_prefix}_score_population.csv")
    context = context_by_ligand(base)

    sav = read_optional_table(tables_dir / f"{out_prefix}_structure_added_value_cases_full.csv")
    sources["structure_added_value_cases_full"] = len(sav)
    if not sav.empty:
        sim_col = similarity_column(sav)
        sav_work = sav.copy()
        if sim_col:
            sav_work["ecfp4_active_similarity"] = numeric_series(sav_work, sim_col)
        sav_work = sav_work[string_series(sav_work, "activity_label").str.lower().eq("active")]
        sav_work = sort_candidates(sav_work, [("SAV", False), ("best_structure_percentile", False)])
        for threshold in [0.30, 0.50, 0.70]:
            candidates = sav_work[numeric_series(sav_work, "ecfp4_active_similarity").lt(threshold)]
            add_candidates(
                selected,
                candidates,
                _category_threshold_label("novel_sav", threshold),
                f"Active ligand ranked better by structure than ECFP4 at Tanimoto < {threshold:.2f}.",
                context,
                max_per_category,
            )

    novel = read_optional_table(tables_dir / f"{out_prefix}_novel_actives_full.csv")
    sources["novel_actives_full"] = len(novel)
    if not novel.empty:
        novel_work = novel[string_series(novel, "activity_label").str.lower().eq("active")].copy()
        novel_work = sort_candidates(novel_work, [("best_structure_rank", True), ("ecfp4_active_similarity", True)])
        for threshold in [0.30, 0.50]:
            candidates = novel_work[numeric_series(novel_work, "ecfp4_active_similarity").lt(threshold)]
            add_candidates(
                selected,
                candidates,
                _category_threshold_label("low_similarity_active", threshold),
                f"Low-similarity active recovered by at least one structure-based rank at Tanimoto < {threshold:.2f}.",
                context,
                max_per_category,
            )

    anomalies = read_optional_table(tables_dir / f"{out_prefix}_score_anomalies_in_top_hits.csv")
    sources["score_anomalies_in_top_hits"] = len(anomalies)
    if not anomalies.empty:
        include_flags = set(config.get("categories", {}).get("score_anomaly_top_hits", {}).get("include_flags", []))
        flags = string_series(anomalies, "anomaly_flags")
        mask = flags.apply(lambda text: any(flag in set(text.split(";")) for flag in include_flags))
        anomaly_work = anomalies[mask].copy()
        anomaly_work = sort_candidates(anomaly_work, [("rank_unidock_best", True), ("unidock_best_score", True)])
        extreme = anomaly_work[string_series(anomaly_work, "anomaly_flags").str.contains("extreme_unidock_negative", regex=False)]
        add_candidates(
            selected,
            extreme,
            "extreme_unidock_negative_top_hit",
            "Extreme Uni-Dock negative score appears in top-ranked anomaly cases and requires manual pose inspection.",
            context,
            max_per_category,
        )
        add_candidates(
            selected,
            anomaly_work,
            "score_anomaly_top_hit",
            "Score-anomalous row appears among top-ranked method outputs.",
            context,
            max_per_category,
        )

    false_pos = read_optional_table(tables_dir / f"{out_prefix}_false_positive_consensus_full.csv")
    sources["false_positive_consensus_full"] = len(false_pos)
    if not false_pos.empty:
        fp_work = false_pos[string_series(false_pos, "activity_label").str.lower().eq("inactive")].copy()
        fp_work = sort_candidates(fp_work, [("n_high_rank_methods", False), ("best_structure_rank", True), ("ecfp4_active_similarity", False)])
        add_candidates(
            selected,
            fp_work,
            "consensus_inactive_false_positive",
            "Inactive ligand ranked highly by multiple methods.",
            context,
            max_per_category,
        )

    false_neg = read_optional_table(tables_dir / f"{out_prefix}_active_false_negatives_full.csv")
    sources["active_false_negatives_full"] = len(false_neg)
    if not false_neg.empty:
        fn_work = false_neg[string_series(false_neg, "activity_label").str.lower().eq("active")].copy()
        fn_work = sort_candidates(fn_work, [("ecfp4_active_similarity", True), ("unidock_best_score", False)])
        add_candidates(
            selected,
            fn_work,
            "active_false_negative",
            "Active ligand low-ranked by all main structure/ligand-similarity methods.",
            context,
            max_per_category,
        )

    sim_hits = read_optional_table(tables_dir / f"{out_prefix}_similarity_driven_hits_full.csv")
    sources["similarity_driven_hits_full"] = len(sim_hits)
    if not sim_hits.empty:
        sim_work = sim_hits[numeric_series(sim_hits, "ecfp4_active_similarity").ge(0.70)].copy()
        sim_work = sort_candidates(sim_work, [("ecfp4_active_similarity", False), ("best_structure_rank", True)])
        add_candidates(
            selected,
            sim_work,
            "similarity_driven_analog_bias_hit",
            "High ECFP4 similarity hit likely reflects analog-neighborhood signal.",
            context,
            max_per_category,
        )

    rows: list[dict[str, Any]] = []
    for item in selected.values():
        values = dict(item.values)
        values["ligand_id"] = item.ligand_id
        values["inspection_categories"] = ";".join(sorted(item.categories, key=lambda c: CATEGORY_PRIORITY.get(c, 999)))
        values["manual_priority"] = item.priority
        values["reason_for_selection"] = " ".join(sorted(item.reasons))
        values["requires_manual_inspection"] = True
        if "nearest_active_ligand_id" not in values and "ecfp4_nearest_active_ligand_id" in values:
            values["nearest_active_ligand_id"] = values.get("ecfp4_nearest_active_ligand_id")
        if "ecfp4_active_similarity" not in values:
            sim = similarity_column(pd.DataFrame([values]))
            if sim:
                values["ecfp4_active_similarity"] = values.get(sim)
        rows.append(values)

    shortlist = pd.DataFrame(rows)
    if shortlist.empty:
        shortlist = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        for col in OUTPUT_COLUMNS:
            if col not in shortlist.columns:
                shortlist[col] = pd.NA
        shortlist = shortlist[OUTPUT_COLUMNS].copy()
        shortlist["_sort_score"] = pd.to_numeric(shortlist.get("SAV"), errors="coerce").fillna(-math.inf)
        shortlist = shortlist.sort_values(["manual_priority", "_sort_score", "ligand_id"], ascending=[True, False, True])
        shortlist = shortlist.drop(columns=["_sort_score"]).head(max_total).reset_index(drop=True)

    manifest = {
        "total_selected": int(len(shortlist)),
        "max_total_ligands": max_total,
        "max_per_category": max_per_category,
        "counts_by_category": category_counts(shortlist),
        "source_rows": sources,
        "pdb_native_baseline_used_for_selection": False,
        "ccd_native_baseline_reference_allowed": bool(
            config.get("baseline_policy", {}).get("allow_ccd_native_baseline_for_reference", True)
        ),
        "selection_note": "No ligand is claimed as a discovered inhibitor. These are retrospective inspection candidates.",
    }
    return shortlist, manifest


def main() -> None:
    args = parse_args()
    config = load_yaml(args.inspection_config)
    tables_dir = Path(args.tables_dir)
    reports_dir = Path(args.reports_dir)
    ensure_dir(tables_dir)
    ensure_dir(reports_dir)

    shortlist, manifest = build_inspection_shortlist(tables_dir, config, out_prefix=args.out_prefix)
    configured_outputs = config.get("outputs", {}) if args.out_prefix == "mapk1_phase1" else {}
    out_table = Path(configured_outputs.get("inspection_table", f"results/tables/{args.out_prefix}_inspection_shortlist.csv"))
    out_manifest = Path(configured_outputs.get("inspection_manifest", f"results/reports/{args.out_prefix}_inspection_manifest.json"))
    ensure_dir(out_table.parent)
    ensure_dir(out_manifest.parent)
    shortlist.to_csv(out_table, index=False)
    write_json(manifest, out_manifest)
    print(f"Inspection shortlist written: {out_table} ({len(shortlist)} ligands)")


if __name__ == "__main__":
    main()
