from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from screensift.common.io import load_yaml, read_text_table_auto
from screensift.curation.adapt_dataset_schema import audit_row, split_valid_failures


_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _load_schema(schema: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(schema, dict):
        return schema
    return load_yaml(schema)


def _load_data(data: str | Path | pd.DataFrame | None, schema: dict[str, Any]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if data is not None:
        return read_text_table_auto(data)
    input_path = schema.get("input", {}).get("path")
    if not input_path:
        raise ValueError("Provide data or define input.path in the schema.")
    sep = schema.get("input", {}).get("sep")
    if sep is not None:
        return pd.read_csv(input_path, sep=sep, engine="python")
    return read_text_table_auto(input_path)


def _normalize(series: pd.Series, direction: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    min_value = values.min()
    max_value = values.max()
    if pd.isna(min_value) or pd.isna(max_value) or min_value == max_value:
        return pd.Series(np.where(values.notna(), 1.0, np.nan), index=series.index)
    if direction == "lower":
        return (max_value - values) / (max_value - min_value)
    if direction == "higher":
        return (values - min_value) / (max_value - min_value)
    raise ValueError(f"Unsupported score direction: {direction!r}. Use 'higher' or 'lower'.")


def _best_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _structure_columns(
    frame: pd.DataFrame,
    structure_score_columns: dict[str, str] | list[str] | None,
) -> dict[str, str]:
    if isinstance(structure_score_columns, dict):
        return structure_score_columns
    if isinstance(structure_score_columns, list):
        return {column: "higher" for column in structure_score_columns}

    defaults: dict[str, tuple[list[str], str]] = {
        "unidock_best_score": (["unidock_best_score", "best_score", "unidock_score", "score"], "lower"),
        "gnina_cnnscore": (["CNNscore", "cnnscore", "gnina_cnnscore"], "higher"),
        "gnina_cnnaffinity": (["CNNaffinity", "cnnaffinity", "gnina_cnnaffinity"], "higher"),
        "gnina_affinity": (["gnina_affinity", "affinity", "GNINA_affinity"], "lower"),
    }
    found: dict[str, str] = {}
    for _, (candidates, direction) in defaults.items():
        column = _best_existing_column(frame, candidates)
        if column:
            found[column] = direction
    return found


def _similarity_column(frame: pd.DataFrame, similarity_score_column: str | None) -> str | None:
    if similarity_score_column:
        if similarity_score_column not in frame.columns:
            raise ValueError(f"Similarity score column not found: {similarity_score_column}")
        return similarity_score_column
    return _best_existing_column(
        frame,
        [
            "ecfp4_active_similarity",
            "active_similarity",
            "similarity_score",
            "tanimoto",
            "tanimoto_similarity",
        ],
    )


def _fingerprint(smiles: Any):
    if smiles is None or pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return _MORGAN_GENERATOR.GetFingerprint(mol)


def _active_similarity(frame: pd.DataFrame) -> pd.Series:
    if "activity_label" not in frame.columns or "canonical_smiles" not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    fps = frame["canonical_smiles"].map(_fingerprint)
    active_indices = frame.index[frame["activity_label"].astype(str).str.lower().eq("active") & fps.notna()].tolist()
    if not active_indices:
        return pd.Series(np.nan, index=frame.index)

    active_fps = [(idx, fps.loc[idx]) for idx in active_indices]
    similarities: list[float] = []
    for idx, fp in fps.items():
        if fp is None:
            similarities.append(np.nan)
            continue
        best = 0.0
        for active_idx, active_fp in active_fps:
            if idx == active_idx:
                continue
            best = max(best, float(DataStructs.TanimotoSimilarity(fp, active_fp)))
        similarities.append(best)
    return pd.Series(similarities, index=frame.index)


def _evidence_bucket(row: pd.Series, structure_cutoff: float, similarity_cutoff: float) -> str:
    structure_hit = bool(pd.notna(row.get("structure_score_norm")) and row["structure_score_norm"] >= structure_cutoff)
    similarity_hit = bool(pd.notna(row.get("similarity_score_norm")) and row["similarity_score_norm"] >= similarity_cutoff)
    if structure_hit and similarity_hit:
        return "consensus_supported"
    if similarity_hit:
        return "analog_supported"
    if structure_hit:
        return "structure_supported"
    return "deprioritized"


def _validate_aggregation_mode(mode: str, name: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in {"max", "weighted_mean"}:
        raise ValueError(f"Unsupported {name}: {mode!r}. Use 'max' or 'weighted_mean'.")
    return normalized


def _validate_evidence_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in {"similarity", "structure", "combined"}:
        raise ValueError(f"Unsupported evidence_mode: {mode!r}. Use 'similarity', 'structure', or 'combined'.")
    return normalized


def _validate_weights(weights: dict[str, float] | None, required_keys: list[str], name: str) -> dict[str, float]:
    if not weights:
        raise ValueError(f"{name}='weighted_mean' requires explicit weights.")
    missing = [key for key in required_keys if key not in weights]
    if missing:
        raise ValueError(f"{name} weights are missing required keys: {missing}")
    parsed = {key: float(weights[key]) for key in required_keys}
    if any(value < 0 for value in parsed.values()):
        raise ValueError(f"{name} weights must be non-negative.")
    if sum(parsed.values()) <= 0:
        raise ValueError(f"{name} weights must sum to a positive value.")
    return parsed


def _weighted_mean(frame: pd.DataFrame, columns: list[str], weights: dict[str, float]) -> pd.Series:
    numerator = pd.Series(0.0, index=frame.index)
    denominator = pd.Series(0.0, index=frame.index)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        weight = float(weights[column])
        present = values.notna()
        numerator = numerator.add(values.fillna(0.0) * weight, fill_value=0.0)
        denominator = denominator.add(present.astype(float) * weight, fill_value=0.0)
    return numerator.divide(denominator).where(denominator > 0, np.nan)


def find_candidates(
    schema: str | Path | dict[str, Any],
    data: str | Path | pd.DataFrame | None = None,
    target: str | None = None,
    evidence_mode: str = "combined",
    structure_score_columns: dict[str, str] | list[str] | None = None,
    similarity_score_column: str | None = None,
    compute_similarity_if_missing: bool = True,
    structure_aggregation: str = "max",
    structure_weights: dict[str, float] | None = None,
    candidate_aggregation: str = "max",
    candidate_weights: dict[str, float] | None = None,
    structure_score_cutoff: float = 0.70,
    similarity_score_cutoff: float = 0.70,
    n_candidates: int = 100,
) -> pd.DataFrame:
    """Adapt a ligand dataset and return candidate rows with normalized evidence scores.

    `evidence_mode` controls which evidence channels can affect ranking:
    `similarity`, `structure`, or `combined`.
    `structure_score_columns` can be a mapping of `{column_name: "higher"|"lower"}`.
    If omitted, common Uni-Dock/GNINA column names are detected.
    If no similarity column is provided, ECFP4/Tanimoto similarity to known
    actives is computed when active labels are available.
    Aggregation modes are `max` or `weighted_mean`. Weighted aggregation requires
    explicit weights for every contributing evidence channel.
    """
    evidence_mode = _validate_evidence_mode(evidence_mode)
    structure_aggregation = _validate_aggregation_mode(structure_aggregation, "structure_aggregation")
    candidate_aggregation = _validate_aggregation_mode(candidate_aggregation, "candidate_aggregation")
    schema_data = _load_schema(schema)
    source = _load_data(data, schema_data).reset_index(drop=True)
    source["_source_row"] = np.arange(1, len(source) + 1)

    audit_rows = [audit_row(row, int(row["_source_row"]), schema_data) for _, row in source.iterrows()]
    audit = pd.DataFrame(audit_rows)
    curated, _, _ = split_valid_failures(audit, str(schema_data.get("deduplicate_by", "inchikey")))
    if curated.empty:
        return pd.DataFrame(
            columns=[
                "target",
                "ligand_id",
                "activity_label",
                "canonical_smiles",
                "similarity_score_norm",
                "structure_score_norm",
                "candidate_score",
                "evidence_bucket",
            ]
        )

    merged = curated.merge(source, left_on="source_row", right_on="_source_row", how="left", suffixes=("", "_input"))
    use_similarity = evidence_mode in {"similarity", "combined"}
    use_structure = evidence_mode in {"structure", "combined"}

    sim_col = _similarity_column(merged, similarity_score_column) if use_similarity else None
    if sim_col:
        raw_similarity = pd.to_numeric(merged[sim_col], errors="coerce")
        merged["similarity_score_raw"] = raw_similarity
        # A supplied column that is already a bounded [0, 1] similarity (e.g. a
        # precomputed Tanimoto) is passed through unchanged so the shared
        # similarity_score_cutoff keeps its absolute "resembles a known active"
        # meaning -- identical to the computed-Tanimoto path below. Only columns
        # on an unknown/unbounded scale are population min-max normalized.
        finite = raw_similarity.dropna()
        already_bounded = not finite.empty and finite.min() >= 0.0 and finite.max() <= 1.0
        if already_bounded:
            merged["similarity_score_norm"] = raw_similarity
        else:
            merged["similarity_score_norm"] = _normalize(merged[sim_col], "higher")
    elif use_similarity and compute_similarity_if_missing:
        merged["similarity_score_raw"] = _active_similarity(merged)
        merged["similarity_score_norm"] = merged["similarity_score_raw"]
    else:
        merged["similarity_score_raw"] = np.nan
        merged["similarity_score_norm"] = np.nan

    structure_cols = _structure_columns(merged, structure_score_columns) if use_structure else {}
    normalized_structure_cols: list[str] = []
    for column, direction in structure_cols.items():
        norm_col = f"{column}_norm"
        merged[norm_col] = _normalize(merged[column], direction)
        normalized_structure_cols.append(norm_col)

    if normalized_structure_cols:
        if structure_aggregation == "max":
            merged["structure_score_norm"] = merged[normalized_structure_cols].max(axis=1, skipna=True)
        else:
            normalized_weights = _validate_weights(
                {f"{column}_norm": value for column, value in (structure_weights or {}).items()},
                normalized_structure_cols,
                "structure_aggregation",
            )
            merged["structure_score_norm"] = _weighted_mean(merged, normalized_structure_cols, normalized_weights)
        # idxmax(axis=1) raises "Encountered all NA values" on pandas >= 2.1 for
        # rows whose structure scores are all NaN (a ligand that failed docking
        # in every receptor -- common with partial coverage). Only label rows
        # that actually have a structure score; the rest use the "" sentinel,
        # matching the no-structure branch below.
        has_structure = merged[normalized_structure_cols].notna().any(axis=1)
        primary = pd.Series("", index=merged.index, dtype=object)
        if has_structure.any():
            primary.loc[has_structure] = (
                merged.loc[has_structure, normalized_structure_cols]
                .idxmax(axis=1)
                .str.replace("_norm", "", regex=False)
            )
        merged["primary_structure_score"] = primary
    else:
        merged["structure_score_norm"] = np.nan
        merged["primary_structure_score"] = ""

    if evidence_mode == "similarity":
        merged["candidate_score"] = merged["similarity_score_norm"]
    elif evidence_mode == "structure":
        merged["candidate_score"] = merged["structure_score_norm"]
    elif candidate_aggregation == "max":
        candidate_cols = ["structure_score_norm", "similarity_score_norm"]
        merged["candidate_score"] = merged[candidate_cols].max(axis=1, skipna=True)
    else:
        candidate_cols = ["structure_score_norm", "similarity_score_norm"]
        normalized_candidate_weights = _validate_weights(candidate_weights, candidate_cols, "candidate_aggregation")
        merged["candidate_score"] = _weighted_mean(merged, candidate_cols, normalized_candidate_weights)
    merged["candidate_score"] = merged["candidate_score"].fillna(0.0)
    merged["evidence_bucket"] = merged.apply(
        lambda row: _evidence_bucket(row, structure_score_cutoff, similarity_score_cutoff),
        axis=1,
    )
    merged["target"] = target or schema_data.get("target", {}).get("target_id", "")

    output_cols = [
        "target",
        "ligand_id",
        "activity_label",
        "canonical_smiles",
        "inchikey",
        "similarity_score_raw",
        "similarity_score_norm",
        "structure_score_norm",
        "primary_structure_score",
        "candidate_score",
        "evidence_bucket",
    ]
    output_cols.extend(column for column in structure_cols if column not in output_cols)
    ranked = merged.sort_values(["candidate_score", "structure_score_norm", "similarity_score_norm"], ascending=False, na_position="last")
    return ranked[[column for column in output_cols if column in ranked.columns]].head(n_candidates).reset_index(drop=True)
