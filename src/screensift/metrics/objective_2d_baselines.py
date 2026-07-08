from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.warning")


def mol_from_smiles_safe(smiles: Any):
    if pd.isna(smiles) or not str(smiles).strip():
        return None
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def compute_ecfp4_fingerprint(smiles: Any, radius: int = 2, n_bits: int = 2048, use_chirality: bool = False):
    mol = mol_from_smiles_safe(smiles)
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useChirality=use_chirality)
    except Exception:
        return None


def prepare_fingerprints(
    df: pd.DataFrame,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = False,
    smiles_col: str = "canonical_smiles",
) -> pd.DataFrame:
    out = df.copy()
    if "ligand_id" not in out.columns:
        out["ligand_id"] = [f"ligand_{i:07d}" for i in range(len(out))]
    out["ligand_id"] = out["ligand_id"].astype(str)
    out["_ecfp4_fp"] = out[smiles_col].map(
        lambda smiles: compute_ecfp4_fingerprint(smiles, radius=radius, n_bits=n_bits, use_chirality=use_chirality)
    )
    out["_ecfp4_valid"] = out["_ecfp4_fp"].notna()
    if "is_active" not in out.columns:
        out["is_active"] = out["activity_label"].astype(str).str.lower().eq("active").astype(int)
    out["is_active"] = pd.to_numeric(out["is_active"], errors="coerce").fillna(0).astype(int)
    return out


def compute_max_tanimoto_to_references(query_fps: list[Any], reference_fps: list[Any]) -> list[float]:
    if not reference_fps:
        return [float("nan")] * len(query_fps)
    values: list[float] = []
    for fp in query_fps:
        if fp is None:
            values.append(float("nan"))
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, reference_fps)
        values.append(float(max(sims)) if sims else float("nan"))
    return values


def _is_same_ligand(query: pd.Series, ref: pd.Series) -> bool:
    for col in ["ligand_id", "canonical_smiles", "inchikey"]:
        if col in query.index and col in ref.index and pd.notna(query[col]) and pd.notna(ref[col]):
            if str(query[col]) == str(ref[col]):
                return True
    return False


def compute_all_active_leave_one_out(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    active_refs = out[out["is_active"].eq(1) & out["_ecfp4_valid"]].copy()
    similarities: list[float] = []
    nearest_ids: list[str | None] = []
    valid: list[bool] = []
    reasons: list[str] = []
    for _, row in out.iterrows():
        if row["_ecfp4_fp"] is None:
            similarities.append(float("nan"))
            nearest_ids.append(None)
            valid.append(False)
            reasons.append("invalid_or_missing_smiles")
            continue
        refs = active_refs
        if int(row["is_active"]) == 1:
            refs = active_refs[~active_refs.apply(lambda ref: _is_same_ligand(row, ref), axis=1)]
        if refs.empty:
            similarities.append(float("nan"))
            nearest_ids.append(None)
            valid.append(False)
            reasons.append("no_nonself_active_reference" if int(row["is_active"]) == 1 else "no_active_reference")
            continue
        sims = DataStructs.BulkTanimotoSimilarity(row["_ecfp4_fp"], refs["_ecfp4_fp"].tolist())
        best_i = int(np.argmax(sims))
        similarities.append(float(sims[best_i]))
        nearest_ids.append(str(refs.iloc[best_i]["ligand_id"]))
        valid.append(True)
        reasons.append("")
    out["all_active_loo_ecfp4_analog_neighborhood"] = similarities
    out["all_active_loo_nearest_ligand_id"] = nearest_ids
    out["all_active_loo_ecfp4_valid"] = valid
    out["all_active_loo_ecfp4_failure_reason"] = reasons
    return out


def _mol_from_native_reference(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        return [Chem.MolFromPDBFile(str(path), sanitize=True, removeHs=False)]
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
        return [mol for mol in supplier]
    return []


def _native_reference_fps(native_ligand_files: list[str | Path], radius: int, n_bits: int, use_chirality: bool) -> tuple[list[Any], list[str]]:
    fps: list[Any] = []
    names: list[str] = []
    for path_like in native_ligand_files:
        path = Path(path_like)
        if not path.exists():
            continue
        for idx, mol in enumerate(_mol_from_native_reference(path)):
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useChirality=use_chirality)
            fps.append(fp)
            names.append(f"{path}:{idx}")
    return fps, names


def compute_native_ligand_similarity(
    df: pd.DataFrame,
    native_ligand_sdf_files: list[str | Path],
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = False,
) -> tuple[pd.DataFrame, str | None]:
    fps, names = _native_reference_fps(native_ligand_sdf_files, radius, n_bits, use_chirality)
    out = df.copy()
    if not fps:
        out["native_ligand_ecfp4"] = np.nan
        out["native_ligand_reference"] = pd.NA
        return out, "No native ligand SDF or PDB references were found; native-ligand-only baseline skipped."
    warning = None
    suffixes = {Path(path).suffix.lower() for path in native_ligand_sdf_files}
    if ".pdb" in suffixes:
        warning = "Native-ligand-only baseline used PDB ligand references; SDF is preferred because PDB bond orders may be inferred."
    similarities: list[float] = []
    nearest: list[str | None] = []
    for fp in out["_ecfp4_fp"]:
        if fp is None:
            similarities.append(float("nan"))
            nearest.append(None)
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
        best_i = int(np.argmax(sims))
        similarities.append(float(sims[best_i]))
        nearest.append(names[best_i])
    out["native_ligand_ecfp4"] = similarities
    out["native_ligand_reference"] = nearest
    return out, warning


def compute_few_active_similarity(df: pd.DataFrame, k: int, repeat_seed: int) -> tuple[pd.DataFrame, list[str]]:
    active = df[df["is_active"].eq(1) & df["_ecfp4_valid"]].copy()
    if len(active) < k:
        return pd.DataFrame(), []
    rng = np.random.default_rng(repeat_seed)
    ref_idx = rng.choice(active.index.to_numpy(), size=k, replace=False)
    refs = active.loc[ref_idx]
    eval_df = df.drop(index=ref_idx).copy()
    ref_fps = refs["_ecfp4_fp"].tolist()
    eval_df["few_active_ecfp4"] = compute_max_tanimoto_to_references(eval_df["_ecfp4_fp"].tolist(), ref_fps)
    eval_df["few_active_k"] = k
    eval_df["few_active_reference_ligand_ids"] = ";".join(refs["ligand_id"].astype(str).tolist())
    return eval_df, refs["ligand_id"].astype(str).tolist()


def scaffold_from_smiles(smiles: Any) -> str | None:
    mol = mol_from_smiles_safe(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or None


def compute_scaffold_holdout_similarity(
    df: pd.DataFrame,
    repeat_seed: int,
    reference_scaffold_fraction: float = 0.30,
    min_reference_active_scaffolds: int = 5,
    min_test_active_scaffolds: int = 5,
    min_test_actives_required: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    work = df.copy()
    if "scaffold_smiles" not in work.columns:
        work["scaffold_smiles"] = work["canonical_smiles"].map(scaffold_from_smiles)
    active_scaffolds = sorted(work.loc[work["is_active"].eq(1) & work["scaffold_smiles"].notna(), "scaffold_smiles"].unique())
    if len(active_scaffolds) < min_reference_active_scaffolds + min_test_active_scaffolds:
        return pd.DataFrame(), []
    rng = np.random.default_rng(repeat_seed)
    n_ref = max(min_reference_active_scaffolds, int(round(len(active_scaffolds) * reference_scaffold_fraction)))
    n_ref = min(n_ref, len(active_scaffolds) - min_test_active_scaffolds)
    ref_scaffolds = set(rng.choice(active_scaffolds, size=n_ref, replace=False).tolist())
    refs = work[work["is_active"].eq(1) & work["scaffold_smiles"].isin(ref_scaffolds) & work["_ecfp4_valid"]].copy()
    eval_df = work[~work["scaffold_smiles"].isin(ref_scaffolds)].copy()
    if refs.empty or int(eval_df["is_active"].sum()) < min_test_actives_required:
        return pd.DataFrame(), sorted(ref_scaffolds)
    eval_df["scaffold_holdout_ecfp4"] = compute_max_tanimoto_to_references(eval_df["_ecfp4_fp"].tolist(), refs["_ecfp4_fp"].tolist())
    eval_df["reference_active_scaffolds"] = ";".join(sorted(ref_scaffolds))
    return eval_df, sorted(ref_scaffolds)


def compute_near_analog_subset(df: pd.DataFrame, similarity_col: str, threshold: float) -> pd.DataFrame:
    if similarity_col not in df.columns:
        return pd.DataFrame(columns=df.columns)
    return df[pd.to_numeric(df[similarity_col], errors="coerce") < threshold].copy()


def compute_label_shuffle_control(df: pd.DataFrame, repeat_seed: int) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(repeat_seed)
    shuffled = out["is_active"].to_numpy().copy()
    rng.shuffle(shuffled)
    out["is_active"] = shuffled.astype(int)
    out["activity_label"] = np.where(out["is_active"].eq(1), "active", "inactive")
    return out


def compute_inactive_reference_control(df: pd.DataFrame, k: int, repeat_seed: int) -> tuple[pd.DataFrame, list[str]]:
    inactive = df[df["is_active"].eq(0) & df["_ecfp4_valid"]].copy()
    if len(inactive) < k:
        return pd.DataFrame(), []
    rng = np.random.default_rng(repeat_seed)
    ref_idx = rng.choice(inactive.index.to_numpy(), size=k, replace=False)
    refs = inactive.loc[ref_idx]
    eval_df = df.drop(index=ref_idx).copy()
    eval_df["inactive_reference_ecfp4"] = compute_max_tanimoto_to_references(eval_df["_ecfp4_fp"].tolist(), refs["_ecfp4_fp"].tolist())
    eval_df["inactive_reference_k"] = k
    eval_df["inactive_reference_ligand_ids"] = ";".join(refs["ligand_id"].astype(str).tolist())
    return eval_df, refs["ligand_id"].astype(str).tolist()


def strip_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["_ecfp4_fp"], errors="ignore")
