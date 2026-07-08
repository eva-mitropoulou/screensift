from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors

from screensift.common.io import ensure_dir
from screensift.metrics.enrichment import (
    compute_enrichment_factor,
    compute_pr_auc,
    compute_roc_auc,
    compute_topk_recovery,
)


RDLogger.DisableLog("rdApp.warning")
CCD_DOWNLOAD_URL = "https://files.rcsb.org/ligands/download/{resname}_ideal.sdf"
SOLVENT_OR_ION_RESNAMES = {"HOH", "WAT", "DOD", "NA", "K", "CL", "MG", "CA", "ZN", "MN", "NI", "CO", "FE"}
SUSPICIOUS_ELEMENTS = {"U", "PU", "HG", "PB", "CD", "AS"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check native ligand reference chemistry before Step 8.")
    parser.add_argument("--native-qc", default="results/tables/mapk1_phase1_native_ligand_reference_qc.csv")
    parser.add_argument("--native-table", default="results/tables/mapk1_native_ligands.csv")
    parser.add_argument("--receptor-selection", default="results/tables/mapk1_receptor_selection_table.csv")
    parser.add_argument("--native-glob", default="data/processed/receptors/MAPK1/*/native_ligand.pdb")
    parser.add_argument("--out", default="results/tables/mapk1_native_ligand_reference_sanity.csv")
    parser.add_argument("--report", default="results/reports/mapk1_native_ligand_reference_sanity_report.md")
    return parser.parse_args()


def native_ligand_resnames_from_pdb(path: Path) -> list[str]:
    counts: dict[str, int] = {}
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(("HETATM", "ATOM")):
            resname = line[17:20].strip()
            if resname:
                counts[resname] = counts.get(resname, 0) + 1
    return [name for name, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def load_ligand_mol(path: Path):
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdb":
            return Chem.MolFromPDBFile(str(path), sanitize=True, removeHs=False)
        if suffix == ".sdf":
            supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
            for mol in supplier:
                if mol is not None:
                    return mol
    except Exception:
        return None
    return None


def canonical_smiles(mol) -> str:
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)
    except Exception:
        return ""


def formal_charge(mol) -> int | None:
    if mol is None:
        return None
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


def aromatic_ring_count(mol) -> int:
    if mol is None:
        return 0
    return int(
        sum(
            1
            for ring in mol.GetRingInfo().AtomRings()
            if ring and all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring)
        )
    )


def molecule_elements(mol) -> list[str]:
    if mol is None:
        return []
    return sorted({atom.GetSymbol().upper() for atom in mol.GetAtoms()})


def heavy_atom_count(mol) -> int:
    if mol is None:
        return 0
    return int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1))


def sanity_flags_for_mol(
    mol,
    file_path: str,
    resname: str | None,
    duplicate: bool = False,
) -> list[str]:
    flags: list[str] = []
    path = Path(file_path)
    if mol is None and not path.exists():
        return ["missing_file"]
    if mol is None:
        return ["load_failed"]

    smiles = canonical_smiles(mol)
    heavy_atoms = heavy_atom_count(mol)
    n_rings = int(mol.GetRingInfo().NumRings())
    n_fragments = len(Chem.GetMolFrags(mol))
    elements = set(molecule_elements(mol))
    bond_types = {str(bond.GetBondType()) for bond in mol.GetBonds()}
    resname_upper = (resname or "").upper()

    if not resname_upper:
        flags.append("missing_resname")
    if not smiles:
        flags.append("empty_smiles")
    if heavy_atoms < 6:
        flags.append("too_few_heavy_atoms")
    if n_fragments > 1:
        flags.append("disconnected_fragments")
    if bond_types and bond_types == {"SINGLE"} and heavy_atoms >= 10 and path.suffix.lower() == ".pdb":
        flags.append("only_single_bonds_possible_pdb_artifact")
    if n_rings == 0 and heavy_atoms >= 10:
        flags.append("no_rings_for_expected_kinase_ligand")
    if elements & SUSPICIOUS_ELEMENTS:
        flags.append("suspicious_elements")
    if resname_upper in SOLVENT_OR_ION_RESNAMES or heavy_atoms <= 3:
        flags.append("likely_solvent_or_ion")
    if duplicate:
        flags.append("duplicate_reference")
    if any(flag in flags for flag in ["load_failed", "empty_smiles", "only_single_bonds_possible_pdb_artifact", "disconnected_fragments", "likely_solvent_or_ion"]):
        flags.append("chemically_suspicious")
    return sorted(set(flags))


def analyze_reference(path: Path, duplicate: bool = False, source: str = "pdb_native") -> dict[str, Any]:
    resnames = native_ligand_resnames_from_pdb(path) if path.suffix.lower() == ".pdb" else []
    resname = resnames[0] if resnames else ""
    mol = load_ligand_mol(path)
    smiles = canonical_smiles(mol)
    flags = sanity_flags_for_mol(mol, str(path), resname, duplicate=duplicate)
    recommendation = "use_with_caution"
    if not path.exists() or mol is None or "likely_solvent_or_ion" in flags:
        recommendation = "exclude"
    elif "only_single_bonds_possible_pdb_artifact" in flags:
        recommendation = "prefer_ccd_reference"
    elif not flags or flags == ["duplicate_reference"]:
        recommendation = "chemically_sane"
    return {
        "pdb_id": path.parent.name.upper() if path.parent else "",
        "file_path": str(path),
        "source": source,
        "native_ligand_resname": resname,
        "heavy_atoms": heavy_atom_count(mol),
        "canonical_smiles": smiles,
        "formal_charge": formal_charge(mol),
        "n_rings": int(mol.GetRingInfo().NumRings()) if mol is not None else 0,
        "aromatic_rings": aromatic_ring_count(mol),
        "n_fragments": len(Chem.GetMolFrags(mol)) if mol is not None else 0,
        "elements": ";".join(molecule_elements(mol)),
        "load_success": mol is not None,
        "sanity_flags": ";".join(flags),
        "recommendation": recommendation,
    }


def discover_native_paths(native_glob: str) -> list[Path]:
    return sorted(Path(".").glob(native_glob))


def load_optional_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def fetch_ccd_reference(resname: str, cache_dir: Path, session: requests.Session | None = None, timeout: int = 20) -> tuple[Path | None, str | None]:
    if not resname:
        return None, "missing_resname"
    ensure_dir(cache_dir)
    out = cache_dir / f"{resname.upper()}_ideal.sdf"
    if out.exists() and out.stat().st_size > 0:
        return out, None
    client = session or requests.Session()
    url = CCD_DOWNLOAD_URL.format(resname=resname.upper())
    try:
        response = client.get(url, timeout=timeout)
        if response.status_code != 200 or not response.content:
            return None, f"download_failed_status_{response.status_code}"
        out.write_bytes(response.content)
        return out, None
    except Exception as exc:
        return None, f"download_failed: {exc}"


def fingerprint(mol):
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def tanimoto(mol_a, mol_b) -> float:
    fp_a = fingerprint(mol_a)
    fp_b = fingerprint(mol_b)
    if fp_a is None or fp_b is None:
        return float("nan")
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def build_ccd_qc(sanity: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for resname in sorted(set(sanity["native_ligand_resname"].dropna().astype(str))):
        ccd_file, failure = fetch_ccd_reference(resname, cache_dir)
        ccd_mol = load_ligand_mol(ccd_file) if ccd_file else None
        pdb_rows = sanity[sanity["native_ligand_resname"].eq(resname)]
        pdb_path = Path(str(pdb_rows.iloc[0]["file_path"])) if not pdb_rows.empty else None
        pdb_mol = load_ligand_mol(pdb_path) if pdb_path else None
        ccd_smiles = canonical_smiles(ccd_mol)
        pdb_smiles = canonical_smiles(pdb_mol)
        ccd_heavy = heavy_atom_count(ccd_mol)
        pdb_heavy = heavy_atom_count(pdb_mol)
        sim = tanimoto(pdb_mol, ccd_mol)
        mismatch_flags: list[str] = []
        if ccd_mol is None:
            mismatch_flags.append("ccd_load_failed")
        if pdb_mol is not None and ccd_mol is not None and abs(ccd_heavy - pdb_heavy) > 3:
            mismatch_flags.append("heavy_atom_count_mismatch")
        if np.isfinite(sim) and sim < 0.50:
            mismatch_flags.append("large_pdb_ccd_tanimoto_mismatch")
        rows.append(
            {
                "resname": resname,
                "source": "RCSB_CCD_ideal_sdf",
                "ccd_file": str(ccd_file) if ccd_file else "",
                "load_success": ccd_mol is not None,
                "canonical_smiles": ccd_smiles,
                "heavy_atoms": ccd_heavy,
                "failure_reason": failure or "",
                "pdb_canonical_smiles": pdb_smiles,
                "pdb_heavy_atoms": pdb_heavy,
                "pdb_ccd_tanimoto": sim,
                "mismatch_flags": ";".join(mismatch_flags),
            }
        )
    return pd.DataFrame(rows)


def score_population_to_references(population: pd.DataFrame, reference_mols: list, score_col: str) -> pd.DataFrame:
    refs = [fingerprint(mol) for mol in reference_mols if mol is not None]
    out = population.copy()
    values: list[float] = []
    for smiles in out["canonical_smiles"]:
        mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
        fp = fingerprint(mol)
        if fp is None or not refs:
            values.append(float("nan"))
        else:
            values.append(float(max(DataStructs.BulkTanimotoSimilarity(fp, refs))))
    out[score_col] = values
    return out


def metric_row(population: pd.DataFrame, method: str, score_col: str, population_name: str) -> dict[str, Any]:
    working = population[["activity_label", score_col]].copy()
    working["is_active"] = working["activity_label"].astype(str).str.lower().eq("active").astype(int)
    working[score_col] = pd.to_numeric(working[score_col], errors="coerce")
    working = working.dropna(subset=[score_col])
    y = working["is_active"]
    score = working[score_col]
    row = {
        "population": population_name,
        "method": method,
        "score_col": score_col,
        "n_total": int(len(working)),
        "n_active": int(y.sum()),
        "roc_auc": compute_roc_auc(y, score),
        "pr_auc": compute_pr_auc(y, score),
        "ef1": compute_enrichment_factor(y, score, 0.01),
        "ef5": compute_enrichment_factor(y, score, 0.05),
        "ef10": compute_enrichment_factor(y, score, 0.10),
    }
    for k in [50, 100]:
        recovery = compute_topk_recovery(y, score, k)
        row[f"top{k}_actives"] = recovery["topk_actives"]
    return row


def recompute_ccd_baseline(ccd_qc: pd.DataFrame, tables_dir: Path, reports_dir: Path) -> tuple[bool, str]:
    good = ccd_qc[ccd_qc["load_success"].astype(bool)]
    if good.empty:
        report = (
            "# MAPK1 Phase 1 Native Ligand CCD Baseline Report\n\n"
            "No CCD references loaded successfully, so CCD-native ECFP4 was not computed.\n"
        )
        (reports_dir / "mapk1_phase1_native_ligand_ccd_baseline_report.md").write_text(report, encoding="utf-8")
        return False, "No CCD references loaded successfully."
    reference_mols = [load_ligand_mol(Path(path)) for path in good["ccd_file"].dropna().astype(str)]
    rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for pop_name, path in [
        ("full", tables_dir / "mapk1_phase1_full_population_with_ecfp4_and_anomalies.csv"),
        ("clean", tables_dir / "mapk1_phase1_clean_population_with_ecfp4_and_anomalies.csv"),
    ]:
        if not path.exists():
            continue
        population = pd.read_csv(path, dtype={"ligand_id": str}, low_memory=False)
        scored = score_population_to_references(population, reference_mols, "native_ligand_ecfp4_ccd")
        scored["population"] = pop_name
        rows.append(scored[["population", "ligand_id", "activity_label", "canonical_smiles", "native_ligand_ecfp4_ccd"]])
        metric_rows.append(metric_row(scored, "native_ligand_ecfp4_ccd", "native_ligand_ecfp4_ccd", pop_name))
    scores = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    scores.to_csv(tables_dir / "mapk1_phase1_native_ligand_ecfp4_ccd_scores.csv", index=False)
    metrics.to_csv(tables_dir / "mapk1_phase1_native_ligand_ecfp4_ccd_metrics.csv", index=False)

    comparison_lines = []
    step7b_metrics = tables_dir / "mapk1_phase1_objective_2d_single_baseline_metrics_full.csv"
    if step7b_metrics.exists():
        existing = pd.read_csv(step7b_metrics)
        comparison = existing[existing["method"].isin(["native_ligand_ecfp4", "unidock_best", "gnina_cnnscore", "gnina_cnnaffinity", "gnina_affinity"])]
        comparison_lines.append("## Step 7b PDB-native and structure rows\n\n```text\n" + comparison[["method", "pr_auc", "ef1", "ef5", "top50_actives"]].to_string(index=False) + "\n```")
    report = (
        "# MAPK1 Phase 1 Native Ligand CCD Baseline Report\n\n"
        "CCD-derived native ligand references are preferred over PDB-derived references for ECFP4 fingerprinting.\n\n"
        "## CCD-native ECFP4 metrics\n\n"
        "```text\n"
        + metrics.to_string(index=False)
        + "\n```\n\n"
        + "\n\n".join(comparison_lines)
    )
    (reports_dir / "mapk1_phase1_native_ligand_ccd_baseline_report.md").write_text(report, encoding="utf-8")
    return True, "CCD-native ECFP4 baseline computed."


def write_reports(
    sanity: pd.DataFrame,
    ccd_qc: pd.DataFrame,
    ccd_computed: bool,
    ccd_message: str,
    report_path: Path,
    final_report_path: Path,
) -> None:
    n_files = len(sanity)
    n_loaded = int(sanity["load_success"].sum()) if not sanity.empty else 0
    n_suspicious = int(sanity["sanity_flags"].fillna("").str.contains("chemically_suspicious").sum()) if not sanity.empty else 0
    n_sane = int((sanity["recommendation"] == "chemically_sane").sum()) if not sanity.empty else 0
    resnames = ", ".join(sorted(sanity["native_ligand_resname"].dropna().astype(str).unique())) if not sanity.empty else "none"
    ccd_loaded = int(ccd_qc["load_success"].sum()) if not ccd_qc.empty else 0
    pdb_artifacts = int(sanity["sanity_flags"].fillna("").str.contains("only_single_bonds_possible_pdb_artifact").sum()) if not sanity.empty else 0
    recommendation = "treat as approximate/cautious"
    if ccd_computed and ccd_loaded == n_files and ccd_loaded > 0:
        recommendation = "trust CCD-derived native ligand baseline; treat PDB-derived baseline cautiously"
    elif n_suspicious == n_files:
        recommendation = "exclude PDB-derived native ligand baseline from main conclusions"

    sanity_table = sanity.to_string(index=False) if not sanity.empty else "(none)"
    ccd_table = ccd_qc.to_string(index=False) if not ccd_qc.empty else "(none)"
    report = f"""# MAPK1 Native Ligand Reference Sanity Report

## Summary

- Native PDB/SDF ligand files found: {n_files}
- Loaded by RDKit: {n_loaded}
- Chemically sane references: {n_sane}
- Chemically suspicious references: {n_suspicious}
- PDB single-bond artifact flags: {pdb_artifacts}
- Native ligand resnames: {resnames}
- CCD references loaded: {ccd_loaded}

## Native Reference Rows

```text
{sanity_table}
```

## CCD Reference Rows

```text
{ccd_table}
```
"""
    report_path.write_text(report, encoding="utf-8")

    required_pdb_wording = (
        "Native-ligand-only ECFP4 is included as an approximate structural-reference baseline, "
        "but PDB bond-order limitations mean it should not be overinterpreted."
    )
    required_ccd_wording = "CCD-derived native ligand references are preferred over PDB-derived references for ECFP4 fingerprinting."
    final = f"""# MAPK1 Native Ligand Baseline QC Before Step 8

## Summary

- Native PDB ligand files found: {n_files}
- Loaded by RDKit: {n_loaded}
- Chemically sane references: {n_sane}
- Chemically suspicious references: {n_suspicious}
- Native ligand resnames found: {resnames}
- CCD references found/loaded: {ccd_loaded}
- CCD-native ECFP4 computed: {ccd_computed}
- CCD status: {ccd_message}

## Interpretation

{required_pdb_wording}

{required_ccd_wording if ccd_loaded else ""}

Final recommendation: {recommendation}.
"""
    final_report_path.write_text(final, encoding="utf-8")


def run(
    native_qc: Path,
    native_table: Path,
    receptor_selection: Path,
    native_glob: str,
    out: Path,
    report: Path,
) -> dict[str, Any]:
    _ = (load_optional_table(native_qc), load_optional_table(native_table), load_optional_table(receptor_selection))
    ensure_dir(out.parent)
    ensure_dir(report.parent)
    paths = discover_native_paths(native_glob)
    initial_rows = [analyze_reference(path, duplicate=False, source="pdb_native") for path in paths]
    seen_smiles: set[str] = set()
    rows: list[dict[str, Any]] = []
    for row in initial_rows:
        smiles = row["canonical_smiles"]
        duplicate = bool(smiles and smiles in seen_smiles)
        if smiles:
            seen_smiles.add(smiles)
        rows.append(analyze_reference(Path(row["file_path"]), duplicate=duplicate, source=row["source"]))
    sanity = pd.DataFrame(rows)
    sanity.to_csv(out, index=False)

    cache_dir = ensure_dir("data/raw/pdb/chemical_components")
    ccd_qc = build_ccd_qc(sanity, cache_dir) if not sanity.empty else pd.DataFrame()
    ccd_qc_path = out.parent / "mapk1_native_ligand_ccd_reference_qc.csv"
    ccd_qc.to_csv(ccd_qc_path, index=False)
    ccd_computed, ccd_message = recompute_ccd_baseline(ccd_qc, out.parent, report.parent)
    write_reports(
        sanity,
        ccd_qc,
        ccd_computed,
        ccd_message,
        report,
        report.parent / "mapk1_native_ligand_baseline_qc_before_step8.md",
    )
    return {
        "native_files_found": len(paths),
        "loaded_successfully": int(sanity["load_success"].sum()) if not sanity.empty else 0,
        "chemically_suspicious": int(sanity["sanity_flags"].fillna("").str.contains("chemically_suspicious").sum()) if not sanity.empty else 0,
        "resnames": sorted(sanity["native_ligand_resname"].dropna().astype(str).unique()) if not sanity.empty else [],
        "ccd_loaded": int(ccd_qc["load_success"].sum()) if not ccd_qc.empty else 0,
        "ccd_computed": ccd_computed,
    }


def main() -> None:
    args = parse_args()
    summary = run(
        native_qc=Path(args.native_qc),
        native_table=Path(args.native_table),
        receptor_selection=Path(args.receptor_selection),
        native_glob=args.native_glob,
        out=Path(args.out),
        report=Path(args.report),
    )
    print(summary)


if __name__ == "__main__":
    main()
