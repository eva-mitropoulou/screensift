from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


from screensift.common.io import ensure_dir


OUTPUT_COLUMNS = [
    "ligand_id",
    "activity_label",
    "best_score",
    "pdb_id",
    "receptor_pdbqt",
    "receptor_clean_pdb",
    "output_pose_file",
    "selection_reasons",
    "ecfp_max_tanimoto",
    "gnina_receptor_input",
    "gnina_ligand_input",
    "gnina_output_file",
    "valid_for_gnina",
    "invalid_reason",
]
SELECTION_REASON_ORDER = [
    "top_unidock",
    "all_valid_actives",
    "random_inactive_controls",
    "diversity_controls",
    "top_ecfp_similarity",
]


def normalize_ligand_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def load_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"Required input table not found: {table_path}")
    return pd.read_csv(table_path, dtype={"ligand_id": str})


def normalize_best_unidock(best: pd.DataFrame) -> pd.DataFrame:
    frame = best.copy()
    rename_map = {
        "best_pdb_id": "pdb_id",
        "best_receptor_pdbqt": "receptor_pdbqt",
        "best_output_pose_file": "output_pose_file",
    }
    frame = frame.rename(columns={source: target for source, target in rename_map.items() if source in frame.columns})
    required = ["ligand_id", "best_score", "pdb_id", "receptor_pdbqt", "output_pose_file"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Best Uni-Dock table is missing required columns: {missing}")

    frame["ligand_id"] = frame["ligand_id"].map(normalize_ligand_id)
    frame["best_score"] = pd.to_numeric(frame["best_score"], errors="coerce")
    return frame


def merge_phase1_split(best: pd.DataFrame, phase1_split: pd.DataFrame) -> pd.DataFrame:
    frame = best.copy()
    split = phase1_split.copy()
    if "ligand_id" not in split.columns:
        return frame
    split["ligand_id"] = split["ligand_id"].map(normalize_ligand_id)

    merge_columns = ["ligand_id"]
    for column in ["activity_label", "canonical_smiles", "inchikey", "raw_smiles"]:
        if column in split.columns:
            merge_columns.append(column)
    split = split[merge_columns].drop_duplicates("ligand_id")

    merged = frame.merge(split, on="ligand_id", how="left", suffixes=("", "_split"))
    for column in ["activity_label", "canonical_smiles", "inchikey", "raw_smiles"]:
        split_column = f"{column}_split"
        if split_column not in merged.columns:
            continue
        if column in merged.columns:
            merged[column] = merged[column].where(merged[column].notna(), merged[split_column])
        else:
            merged[column] = merged[split_column]
        merged = merged.drop(columns=[split_column])
    return merged


def score_valid_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    finite = result["best_score"].map(lambda value: pd.notna(value) and math.isfinite(float(value)))
    result["score_valid_for_selection"] = finite & (result["best_score"] <= 0) & (result["best_score"] >= -30)
    return result[result["score_valid_for_selection"]].copy()


def add_selection_reason(reasons: dict[str, set[str]], ligand_ids: pd.Series, reason: str) -> None:
    for ligand_id in ligand_ids.map(normalize_ligand_id).tolist():
        if ligand_id:
            reasons[ligand_id].add(reason)


def select_random_inactives(candidates: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count <= 0 or "activity_label" not in candidates.columns:
        return candidates.head(0)
    inactive = candidates[candidates["activity_label"].astype(str).str.lower() == "inactive"].copy()
    if inactive.empty:
        return inactive
    return inactive.sample(n=min(count, len(inactive)), random_state=seed)


def select_diversity_controls(candidates: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count <= 0 or candidates.empty:
        return candidates.head(0)
    frame = candidates.copy()
    try:
        frame["score_bin"] = pd.qcut(frame["best_score"], q=min(5, len(frame)), duplicates="drop")
    except ValueError:
        return frame.sample(n=min(count, len(frame)), random_state=seed)

    bins = [group for _bin, group in frame.groupby("score_bin", observed=True)]
    if not bins:
        return frame.head(0)
    per_bin = max(1, count // len(bins))
    selected = []
    for index, group in enumerate(bins):
        selected.append(group.sample(n=min(per_bin, len(group)), random_state=seed + index))
    result = pd.concat(selected, ignore_index=True)
    if len(result) < count:
        remaining = frame[~frame["ligand_id"].isin(result["ligand_id"])]
        if not remaining.empty:
            extra = remaining.sample(n=min(count - len(result), len(remaining)), random_state=seed + 1000)
            result = pd.concat([result, extra], ignore_index=True)
    return result.drop(columns=["score_bin"], errors="ignore").head(count)


def compute_ecfp_similarity_group(
    candidates: pd.DataFrame,
    top_n: int,
    active_reference_count: int = 25,
) -> tuple[pd.Series, pd.DataFrame, str | None]:
    scores = pd.Series(pd.NA, index=candidates.index, dtype="Float64")
    if top_n <= 0:
        return scores, candidates.head(0), "ECFP similarity subset disabled by top_ecfp_similarity=0."
    if "canonical_smiles" not in candidates.columns:
        return scores, candidates.head(0), "ECFP similarity subset skipped; canonical_smiles column is unavailable."

    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except Exception as exc:
        return scores, candidates.head(0), f"ECFP similarity subset skipped; RDKit import failed: {exc}"

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints: dict[int, Any] = {}
    for index, smiles in candidates["canonical_smiles"].items():
        if pd.isna(smiles) or not str(smiles).strip():
            continue
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            continue
        fingerprints[index] = generator.GetFingerprint(mol)

    if not fingerprints:
        return scores, candidates.head(0), "ECFP similarity subset skipped; no valid fingerprints could be generated."

    if "activity_label" not in candidates.columns:
        return scores, candidates.head(0), "ECFP similarity subset skipped; activity labels are unavailable."
    active_candidates = candidates[candidates["activity_label"].astype(str).str.lower() == "active"].sort_values("best_score")
    reference_indices = [index for index in active_candidates.index if index in fingerprints][:active_reference_count]
    if not reference_indices:
        return scores, candidates.head(0), "ECFP similarity subset skipped; no active reference fingerprints were available."
    reference_fps = [fingerprints[index] for index in reference_indices]

    for index, fingerprint in fingerprints.items():
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, reference_fps)
        scores.loc[index] = max(similarities) if similarities else pd.NA

    ranked = candidates.assign(ecfp_max_tanimoto=scores).dropna(subset=["ecfp_max_tanimoto"])
    selected = ranked.sort_values(["ecfp_max_tanimoto", "best_score"], ascending=[False, True]).head(top_n)
    return scores, selected, None


def infer_receptor_clean_pdb(row: pd.Series, receptor_root: str | Path) -> str:
    pdb_id = str(row["pdb_id"]).lower()
    return str(Path(receptor_root) / pdb_id / "receptor_clean.pdb")


def infer_gnina_output_file(row: pd.Series, output_root: str | Path) -> str:
    pdb_id = str(row["pdb_id"]).lower()
    ligand_id = normalize_ligand_id(row["ligand_id"])
    safe_ligand = "".join(char if char.isalnum() or char in "._-" else "_" for char in ligand_id).strip("._-") or "ligand"
    return str(Path(output_root) / pdb_id / f"{safe_ligand}_gnina_score.sdf")


def annotate_gnina_paths(frame: pd.DataFrame, receptor_root: str | Path, output_root: str | Path) -> pd.DataFrame:
    result = frame.copy()
    result["receptor_clean_pdb"] = result.apply(lambda row: infer_receptor_clean_pdb(row, receptor_root), axis=1)
    result["gnina_receptor_input"] = result["receptor_clean_pdb"]
    result["gnina_ligand_input"] = result["output_pose_file"]
    result["gnina_output_file"] = result.apply(lambda row: infer_gnina_output_file(row, output_root), axis=1)

    valid_values: list[bool] = []
    reasons: list[str] = []
    for row in result.to_dict(orient="records"):
        invalid: list[str] = []
        if not Path(str(row["receptor_clean_pdb"])).exists():
            invalid.append("missing_receptor_clean_pdb")
        if not Path(str(row["output_pose_file"])).exists():
            invalid.append("missing_ligand_pose")
        valid_values.append(not invalid)
        reasons.append(";".join(invalid))
    result["valid_for_gnina"] = valid_values
    result["invalid_reason"] = reasons
    return result


def build_subset(
    best: pd.DataFrame,
    top_unidock: int,
    random_inactives: int,
    include_all_actives: bool,
    seed: int,
    top_ecfp_similarity: int = 300,
    diversity_controls: int = 0,
    receptor_root: str | Path = "data/processed/receptors/MAPK1",
    gnina_output_root: str | Path = "results/poses/gnina/MAPK1/phase1",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = score_valid_candidates(best)
    warnings: list[str] = []
    if "activity_label" not in candidates.columns:
        warnings.append("Activity labels are unavailable; active and inactive control groups may be incomplete.")
        candidates["activity_label"] = pd.NA

    reasons: dict[str, set[str]] = defaultdict(set)
    group_counts: dict[str, int] = {}

    top = candidates.sort_values(["best_score", "ligand_id"], ascending=[True, True]).head(max(0, top_unidock))
    add_selection_reason(reasons, top["ligand_id"], "top_unidock")
    group_counts["top_unidock"] = len(top)

    if include_all_actives:
        actives = candidates[candidates["activity_label"].astype(str).str.lower() == "active"].copy()
    else:
        actives = candidates.head(0)
    add_selection_reason(reasons, actives["ligand_id"], "all_valid_actives")
    group_counts["all_valid_actives"] = len(actives)

    random_controls = select_random_inactives(candidates, random_inactives, seed)
    add_selection_reason(reasons, random_controls["ligand_id"], "random_inactive_controls")
    group_counts["random_inactive_controls"] = len(random_controls)

    diversity = select_diversity_controls(candidates, diversity_controls, seed)
    add_selection_reason(reasons, diversity["ligand_id"], "diversity_controls")
    group_counts["diversity_controls"] = len(diversity)

    ecfp_scores, ecfp_selected, ecfp_warning = compute_ecfp_similarity_group(candidates, top_ecfp_similarity)
    candidates["ecfp_max_tanimoto"] = ecfp_scores
    if ecfp_warning:
        warnings.append(ecfp_warning)
        group_counts["top_ecfp_similarity"] = 0
        ecfp_included = False
    else:
        add_selection_reason(reasons, ecfp_selected["ligand_id"], "top_ecfp_similarity")
        group_counts["top_ecfp_similarity"] = len(ecfp_selected)
        ecfp_included = True

    selected_ids = sorted(reasons)
    selected = candidates[candidates["ligand_id"].isin(selected_ids)].copy()
    selected["selection_reasons"] = selected["ligand_id"].map(
        lambda ligand_id: ";".join(reason for reason in SELECTION_REASON_ORDER if reason in reasons[ligand_id])
    )
    selected = annotate_gnina_paths(selected, receptor_root, gnina_output_root)
    selected = selected.sort_values(["best_score", "ligand_id"], ascending=[True, True]).reset_index(drop=True)

    for column in OUTPUT_COLUMNS:
        if column not in selected.columns:
            selected[column] = pd.NA
    selected = selected[OUTPUT_COLUMNS]

    metadata = {
        "input_rows": int(len(best)),
        "valid_candidate_rows": int(len(candidates)),
        "selected_total_before_deduplication": int(sum(group_counts.values())),
        "selected_total_after_deduplication": int(len(selected)),
        "group_counts": group_counts,
        "ecfp_included": ecfp_included,
        "warnings": warnings,
    }
    return selected, metadata


def table_preview(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "None\n"
    columns = ["ligand_id", "activity_label", "best_score", "pdb_id", "selection_reasons", "valid_for_gnina"]
    available = [column for column in columns if column in frame.columns]
    return frame[available].head(max_rows).to_string(index=False) + "\n"


def write_report(subset: pd.DataFrame, metadata: dict[str, Any], report_path: str | Path) -> None:
    score_series = pd.to_numeric(subset["best_score"], errors="coerce") if not subset.empty else pd.Series(dtype=float)
    activity_counts = subset["activity_label"].value_counts(dropna=False).to_dict() if "activity_label" in subset else {}
    reason_counts: dict[str, int] = {}
    for reason in SELECTION_REASON_ORDER:
        reason_counts[reason] = int(subset["selection_reasons"].astype(str).str.contains(reason, regex=False).sum()) if not subset.empty else 0

    missing_receptor_count = int(subset["invalid_reason"].astype(str).str.contains("missing_receptor_clean_pdb", regex=False).sum()) if not subset.empty else 0
    missing_ligand_count = int(subset["invalid_reason"].astype(str).str.contains("missing_ligand_pose", regex=False).sum()) if not subset.empty else 0

    lines = [
        "# MAPK1 Phase 1 GNINA Subset Selection",
        "",
        "GNINA has not been run yet. This table only selects and validates the inputs for later GNINA score-only rescoring.",
        "",
        "## Summary",
        "",
        f"- input_rows: {metadata['input_rows']}",
        f"- valid_candidate_rows: {metadata['valid_candidate_rows']}",
        f"- selected_total_before_deduplication: {metadata['selected_total_before_deduplication']}",
        f"- selected_total_after_deduplication: {metadata['selected_total_after_deduplication']}",
        f"- valid_for_gnina: {int(subset['valid_for_gnina'].sum()) if not subset.empty else 0}",
        f"- missing_receptor_count: {missing_receptor_count}",
        f"- missing_ligand_count: {missing_ligand_count}",
        f"- score_min: {score_series.min() if not score_series.empty else 'NA'}",
        f"- score_median: {score_series.median() if not score_series.empty else 'NA'}",
        f"- score_max: {score_series.max() if not score_series.empty else 'NA'}",
        f"- ecfp_subset_included: {metadata['ecfp_included']}",
        "",
        "## Activity Counts",
        "",
    ]
    lines.extend(f"- {label}: {count}" for label, count in activity_counts.items())
    lines.extend(["", "## Selection Group Counts", ""])
    lines.extend(f"- {reason}: {count}" for reason, count in reason_counts.items())

    if metadata["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in metadata["warnings"])

    lines.extend(
        [
            "",
            "## Top 20 Selected Ligands By Uni-Dock Score",
            "",
            "```text",
            table_preview(subset.sort_values("best_score")),
            "```",
            "",
        ]
    )

    path = Path(report_path)
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")


def select_gnina_subset(
    best_unidock_path: str | Path,
    phase1_split_path: str | Path,
    out_path: str | Path,
    report_path: str | Path,
    top_unidock: int = 500,
    random_inactives: int = 500,
    include_all_actives: bool = False,
    seed: int = 42,
    top_ecfp_similarity: int = 300,
    diversity_controls: int = 0,
    receptor_root: str | Path = "data/processed/receptors/MAPK1",
    gnina_output_root: str | Path = "results/poses/gnina/MAPK1/phase1",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    best = normalize_best_unidock(load_table(best_unidock_path))
    split = load_table(phase1_split_path)
    merged = merge_phase1_split(best, split)
    subset, metadata = build_subset(
        merged,
        top_unidock=top_unidock,
        random_inactives=random_inactives,
        include_all_actives=include_all_actives,
        seed=seed,
        top_ecfp_similarity=top_ecfp_similarity,
        diversity_controls=diversity_controls,
        receptor_root=receptor_root,
        gnina_output_root=gnina_output_root,
    )

    out = Path(out_path)
    ensure_dir(out.parent)
    subset.to_csv(out, index=False)
    write_report(subset, metadata, report_path)
    return subset, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a MAPK1 phase 1 subset for GNINA score-only rescoring.")
    parser.add_argument("--best-unidock", default="results/tables/mapk1_phase1_unidock_best_per_ligand.csv")
    parser.add_argument("--phase1-split", default="data/processed/splits/mapk1_phase1_screening_set.csv")
    parser.add_argument("--out", default="results/tables/mapk1_phase1_gnina_subset.csv")
    parser.add_argument("--report", default="results/reports/mapk1_phase1_gnina_subset_report.md")
    parser.add_argument("--top-unidock", type=int, default=500)
    parser.add_argument("--random-inactives", type=int, default=500)
    parser.add_argument("--include-all-actives", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-ecfp-similarity", type=int, default=300)
    parser.add_argument("--diversity-controls", type=int, default=0)
    parser.add_argument("--receptor-root", default="data/processed/receptors/MAPK1")
    parser.add_argument("--gnina-output-root", default="results/poses/gnina/MAPK1/phase1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subset, metadata = select_gnina_subset(
        args.best_unidock,
        args.phase1_split,
        args.out,
        args.report,
        top_unidock=args.top_unidock,
        random_inactives=args.random_inactives,
        include_all_actives=args.include_all_actives,
        seed=args.seed,
        top_ecfp_similarity=args.top_ecfp_similarity,
        diversity_controls=args.diversity_controls,
        receptor_root=args.receptor_root,
        gnina_output_root=args.gnina_output_root,
    )
    print(
        "GNINA subset selected: "
        f"rows={len(subset)} "
        f"valid_for_gnina={int(subset['valid_for_gnina'].sum()) if not subset.empty else 0} "
        f"ecfp_included={metadata['ecfp_included']}"
    )
    for warning in metadata["warnings"]:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
