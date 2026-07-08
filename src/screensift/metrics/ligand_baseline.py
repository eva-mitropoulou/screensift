from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.warning")


@dataclass(frozen=True)
class FingerprintRecord:
    index: int
    ligand_id: str
    canonical_smiles: str | None
    inchikey: str | None
    is_active: bool
    fingerprint: Any
    failure_reason: str


def mol_from_smiles_safe(smiles: Any):
    if pd.isna(smiles) or not str(smiles).strip():
        return None
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def compute_ecfp4(mol, radius: int = 2, n_bits: int = 2048, use_chirality: bool = False):
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius,
            nBits=n_bits,
            useChirality=use_chirality,
        )
    except Exception:
        return None


def _is_active(value: Any) -> bool:
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        return int(value) == 1
    return str(value).strip().lower() in {"active", "actives", "true", "1", "yes"}


def _similarity_bin(value: float | None, thresholds: dict[str, float] | None = None) -> str:
    if value is None or pd.isna(value):
        return "missing"
    limits = thresholds or {"low_similarity": 0.30, "medium_similarity": 0.50, "high_similarity": 0.70}
    if value < limits["low_similarity"]:
        return "low"
    if value < limits["medium_similarity"]:
        return "medium"
    if value < limits["high_similarity"]:
        return "high"
    return "very_high"


def _text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _build_records(
    df: pd.DataFrame,
    smiles_col: str,
    activity_col: str,
    id_cols: list[str],
    radius: int,
    n_bits: int,
    use_chirality: bool,
) -> list[FingerprintRecord]:
    records: list[FingerprintRecord] = []
    for idx, row in df.iterrows():
        smiles = _text(row.get(smiles_col))
        mol = mol_from_smiles_safe(smiles)
        fp = compute_ecfp4(mol, radius=radius, n_bits=n_bits, use_chirality=use_chirality)
        reason = "" if fp is not None else "invalid_or_missing_smiles"
        ligand_id = _text(row.get("ligand_id")) or str(idx)
        records.append(
            FingerprintRecord(
                index=idx,
                ligand_id=ligand_id,
                canonical_smiles=_text(row.get("canonical_smiles")) if "canonical_smiles" in id_cols else smiles,
                inchikey=_text(row.get("inchikey")) if "inchikey" in id_cols else None,
                is_active=_is_active(row.get(activity_col)),
                fingerprint=fp,
                failure_reason=reason,
            )
        )
    return records


def _is_nonself_active(query: FingerprintRecord, reference: FingerprintRecord) -> bool:
    if not reference.is_active:
        return False
    if query.index == reference.index:
        return False
    if query.ligand_id and reference.ligand_id and query.ligand_id == reference.ligand_id:
        return False
    if query.canonical_smiles and reference.canonical_smiles and query.canonical_smiles == reference.canonical_smiles:
        return False
    if query.inchikey and reference.inchikey and query.inchikey == reference.inchikey:
        return False
    return True


def compute_leave_one_active_out_similarity(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    activity_col: str = "activity_label",
    id_cols: list[str] | None = None,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = False,
    novelty_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute max ECFP4 similarity to active ligands with active self-matches excluded."""
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column not found: {smiles_col}")
    if activity_col not in df.columns:
        raise ValueError(f"Activity column not found: {activity_col}")

    identifiers = id_cols or ["ligand_id", "canonical_smiles", "inchikey"]
    records = _build_records(df, smiles_col, activity_col, identifiers, radius, n_bits, use_chirality)
    active_refs = [record for record in records if record.is_active and record.fingerprint is not None]

    rows: list[dict[str, Any]] = []
    for record in records:
        nearest_id: str | None = None
        nearest_similarity = float("nan")
        valid = record.fingerprint is not None
        reason = record.failure_reason

        if valid:
            refs = (
                [ref for ref in active_refs if _is_nonself_active(record, ref)]
                if record.is_active
                else active_refs
            )
            if not refs:
                valid = False
                reason = "no_nonself_active_reference" if record.is_active else "no_active_reference"
            else:
                similarities = DataStructs.BulkTanimotoSimilarity(record.fingerprint, [ref.fingerprint for ref in refs])
                best_idx = int(np.argmax(similarities))
                nearest_similarity = float(similarities[best_idx])
                nearest_id = refs[best_idx].ligand_id

        rows.append(
            {
                "ligand_id": record.ligand_id,
                "activity_label": df.loc[record.index, "activity_label"] if "activity_label" in df.columns else df.loc[record.index, activity_col],
                "canonical_smiles": df.loc[record.index, smiles_col],
                "inchikey": df.loc[record.index, "inchikey"] if "inchikey" in df.columns else pd.NA,
                "ecfp4_active_similarity": nearest_similarity,
                "ecfp4_nearest_active_ligand_id": nearest_id,
                "ecfp4_nearest_active_similarity": nearest_similarity,
                "ecfp4_similarity_bin": _similarity_bin(nearest_similarity, novelty_thresholds),
                "ecfp4_valid": bool(valid),
                "ecfp4_failure_reason": reason,
            }
        )
    return pd.DataFrame(rows)
