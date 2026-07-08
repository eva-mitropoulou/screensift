from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


from screensift.common.io import ensure_dir
from screensift.common.logging_utils import setup_logger


OUTPUT_COLUMNS = ["scaffold", "n_total", "n_active", "n_inactive", "active_fraction"]
WARNING_TEXT = (
    "This is a basic leakage/analog-bias screen based on Bemis-Murcko scaffolds and ECFP4 nearest neighbors. "
    "It is not the final full leakage analysis."
)


def mol_from_row(row: pd.Series) -> Chem.Mol | None:
    smiles = row.get("canonical_smiles") or row.get("raw_smiles")
    if pd.isna(smiles) or not str(smiles).strip():
        return None
    return Chem.MolFromSmiles(str(smiles))


def scaffold_from_mol(mol: Chem.Mol | None) -> str | None:
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold, canonical=True)


def fingerprint_from_mol(mol: Chem.Mol | None):
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def nearest_neighbor_max(fps_a: list[Any], fps_b: list[Any], self_compare: bool = False) -> float | None:
    if not fps_a or not fps_b:
        return None
    maxima: list[float] = []
    for index, fp in enumerate(fps_a):
        similarities = DataStructs.BulkTanimotoSimilarity(fp, fps_b)
        if self_compare and len(similarities) > 1:
            similarities[index] = -1.0
        if similarities:
            maxima.append(max(similarities))
    return float(max(maxima)) if maxima else None


def run_leakage_audit(curated_path: str | Path, out_path: str | Path, report_path: str | Path) -> dict[str, Any]:
    logger = setup_logger("leakage_audit")
    curated_path = Path(curated_path)
    out_path = Path(out_path)
    report_path = Path(report_path)
    ensure_dir(out_path.parent)
    ensure_dir(report_path.parent)

    curated = pd.read_csv(curated_path)
    if curated.empty:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(out_path, index=False)
        report = {
            "status": "empty_curated_input",
            "n_scaffolds": 0,
            "duplicate_scaffold_count": 0,
            "active_active_nn_tanimoto_max": None,
            "active_inactive_nn_tanimoto_max": None,
        }
        write_markdown_report(report_path, report)
        return report

    required = {"activity_label", "canonical_smiles"}
    missing = sorted(required - set(curated.columns))
    if missing:
        raise ValueError(f"Curated table is missing required columns: {missing}")

    valid = curated.copy()
    if "valid" in valid.columns:
        valid = valid[valid["valid"].astype(str).str.lower().isin({"true", "1", "yes"})]

    valid["mol"] = valid.apply(mol_from_row, axis=1)
    valid = valid[valid["mol"].notna()].copy()
    valid["scaffold"] = valid["mol"].map(scaffold_from_mol)

    grouped = valid.groupby("scaffold", dropna=False)["activity_label"]
    scaffold_table = grouped.agg(
        n_total="size",
        n_active=lambda labels: int((labels == "active").sum()),
        n_inactive=lambda labels: int((labels == "inactive").sum()),
    ).reset_index()
    if scaffold_table.empty:
        scaffold_table = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        scaffold_table["active_fraction"] = scaffold_table["n_active"] / scaffold_table["n_total"]
        scaffold_table = scaffold_table[OUTPUT_COLUMNS].sort_values(["n_active", "n_total", "scaffold"], ascending=[False, False, True])

    scaffold_table.to_csv(out_path, index=False)

    active_mols = valid.loc[valid["activity_label"] == "active", "mol"].tolist()
    inactive_mols = valid.loc[valid["activity_label"] == "inactive", "mol"].tolist()
    active_fps = [fp for fp in (fingerprint_from_mol(mol) for mol in active_mols) if fp is not None]
    inactive_fps = [fp for fp in (fingerprint_from_mol(mol) for mol in inactive_mols) if fp is not None]

    warnings = [WARNING_TEXT]
    if len(inactive_fps) > 50000:
        inactive_fps = inactive_fps[:50000]
        warnings.append("Active-inactive nearest-neighbor calculation was capped at 50,000 inactive fingerprints.")

    duplicate_scaffold_count = int((scaffold_table["n_total"] > 1).sum()) if not scaffold_table.empty else 0
    report = {
        "status": "complete",
        "n_curated_valid_molecules": int(len(valid)),
        "n_scaffolds": int(len(scaffold_table)),
        "duplicate_scaffold_count": duplicate_scaffold_count,
        "active_active_nn_tanimoto_max": nearest_neighbor_max(active_fps, active_fps, self_compare=True),
        "active_inactive_nn_tanimoto_max": nearest_neighbor_max(active_fps, inactive_fps, self_compare=False),
        "warnings": warnings,
    }
    write_markdown_report(report_path, report)
    logger.info("Wrote leakage scaffold table to %s", out_path)
    logger.info("Wrote leakage report to %s", report_path)
    return report


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# MAPK1 Basic Leakage Audit",
        "",
        WARNING_TEXT,
        "",
        f"- Status: `{report.get('status')}`",
        f"- Scaffolds: `{report.get('n_scaffolds')}`",
        f"- Duplicate scaffold count: `{report.get('duplicate_scaffold_count')}`",
        f"- Active-active nearest-neighbor ECFP4 Tanimoto max: `{report.get('active_active_nn_tanimoto_max')}`",
        f"- Active-inactive nearest-neighbor ECFP4 Tanimoto max: `{report.get('active_inactive_nn_tanimoto_max')}`",
        "",
        "## Warnings",
    ]
    for warning in report.get("warnings", [WARNING_TEXT]):
        lines.append(f"- {warning}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a basic MAPK1 leakage and scaffold audit.")
    parser.add_argument("--curated", default="results/tables/mapk1_ligands_curated.csv", help="Curated ligand table.")
    parser.add_argument("--out", default="results/tables/mapk1_leakage_audit_basic.csv", help="Scaffold audit CSV.")
    parser.add_argument("--report", default="results/reports/mapk1_leakage_audit_basic.md", help="Markdown report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_leakage_audit(args.curated, args.out, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
