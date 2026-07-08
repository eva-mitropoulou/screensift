from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def scaffold_from_smiles(smiles: str | None) -> str | None:
    if smiles is None or pd.isna(smiles) or not str(smiles).strip():
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or None


def compute_scaffold_leakage(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    activity_col: str = "activity_label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column not found: {smiles_col}")
    working = df.copy()
    working["scaffold_smiles"] = working[smiles_col].map(scaffold_from_smiles)
    working["is_active_scaffold"] = working[activity_col].astype(str).str.lower().eq("active")
    summary = (
        working.dropna(subset=["scaffold_smiles"])
        .groupby("scaffold_smiles", dropna=False)
        .agg(
            n_total=("ligand_id", "count"),
            n_active=("is_active_scaffold", "sum"),
        )
        .reset_index()
    )
    summary["n_active"] = summary["n_active"].astype(int)
    summary["n_inactive"] = summary["n_total"] - summary["n_active"]
    summary["active_fraction"] = summary["n_active"] / summary["n_total"]
    summary["scaffold_class"] = "mixed"
    summary.loc[summary["n_active"].eq(0), "scaffold_class"] = "inactive_only"
    summary.loc[summary["n_inactive"].eq(0), "scaffold_class"] = "active_only"
    return working, summary.sort_values(["n_active", "n_total"], ascending=False).reset_index(drop=True)


def scaffold_report_text(summary: pd.DataFrame, target: str | None = None) -> str:
    total_scaffolds = len(summary)
    active_containing = int(summary["n_active"].gt(0).sum()) if not summary.empty else 0
    inactive_containing = int(summary["n_inactive"].gt(0).sum()) if not summary.empty else 0
    mixed = int(summary["scaffold_class"].eq("mixed").sum()) if not summary.empty else 0
    active_only = int(summary["scaffold_class"].eq("active_only").sum()) if not summary.empty else 0
    inactive_only = int(summary["scaffold_class"].eq("inactive_only").sum()) if not summary.empty else 0
    active_singletons = int(((summary["n_active"] == 1) & (summary["n_total"] == 1)).sum()) if not summary.empty else 0
    actives_on_mixed = int(summary.loc[summary["scaffold_class"].eq("mixed"), "n_active"].sum()) if not summary.empty else 0
    actives_on_active_only = int(summary.loc[summary["scaffold_class"].eq("active_only"), "n_active"].sum()) if not summary.empty else 0
    inactives_on_active_containing = int(summary.loc[summary["n_active"].gt(0), "n_inactive"].sum()) if not summary.empty else 0

    top_active = summary[summary["n_active"].gt(0)].sort_values(["active_fraction", "n_active"], ascending=False).head(15)
    top_mixed = summary[summary["scaffold_class"].eq("mixed")].sort_values("n_total", ascending=False).head(15)

    title = f"{target} Scaffold Leakage Report" if target else "Scaffold Leakage Report"
    return f"""# {title}

## Summary

- Total scaffolds: {total_scaffolds}
- Active-containing scaffolds: {active_containing}
- Inactive-containing scaffolds: {inactive_containing}
- Mixed scaffolds: {mixed}
- Active-only scaffolds: {active_only}
- Inactive-only scaffolds: {inactive_only}
- Active singleton scaffolds: {active_singletons}
- Actives on mixed scaffolds: {actives_on_mixed}
- Actives on active-only scaffolds: {actives_on_active_only}
- Inactives on active-containing scaffolds: {inactives_on_active_containing}

## Top Active-Enriched Scaffolds

```text
{top_active.to_string(index=False)}
```

## Top Mixed Scaffolds By Count

```text
{top_mixed.to_string(index=False)}
```
"""


def write_scaffold_outputs(
    df: pd.DataFrame,
    table_path: str | Path,
    report_path: str | Path,
    smiles_col: str = "canonical_smiles",
    activity_col: str = "activity_label",
    target: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotated, summary = compute_scaffold_leakage(df, smiles_col=smiles_col, activity_col=activity_col)
    table = Path(table_path)
    report = Path(report_path)
    table.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table, index=False)
    report.write_text(scaffold_report_text(summary, target=target), encoding="utf-8")
    return annotated, summary
