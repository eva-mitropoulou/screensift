from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from screensift.common.io import ensure_dir, write_json


PDB_NATIVE_STATUS = "deprecated_pdb_bond_order_artifact"
CCD_NATIVE_STATUS = "official_native_ligand_baseline"
REQUIRED_CLEANUP_WORDING = (
    "The PDB-derived native-ligand ECFP4 baseline is retained for provenance but excluded from scientific "
    "conclusions because PDB ligand bond-order perception produced chemically unreliable fingerprints. "
    "CCD-derived native ligand references are used as the official native-ligand-only ECFP4 baseline."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mark deprecated PDB-native and official CCD-native baselines before Step 8.")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument("--reports-dir", default="results/reports")
    parser.add_argument("--out-report", default="results/reports/mapk1_pre_step8_baseline_cleanup_report.md")
    return parser.parse_args()


def _combined_lower_text(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=str)
    present = [col for col in columns if col in df.columns]
    if not present:
        return pd.Series([""] * len(df), index=df.index)
    text = df[present].fillna("").astype(str).agg(" ".join, axis=1)
    return text.str.lower()


def mark_baseline_status(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Add or update baseline_status for native-ligand ECFP4 rows."""
    out = df.copy()
    text = _combined_lower_text(out, ["baseline", "method", "score_col", "notes", "baseline_name"])
    pdb_mask = text.str.contains("native_ligand_ecfp4", regex=False) & ~text.str.contains("ccd", regex=False)
    ccd_mask = text.str.contains("native_ligand_ecfp4_ccd", regex=False) | (
        text.str.contains("ccd", regex=False) & text.str.contains("native", regex=False)
    )

    if pdb_mask.any() or ccd_mask.any():
        if "baseline_status" not in out.columns:
            out["baseline_status"] = ""
        out.loc[pdb_mask, "baseline_status"] = PDB_NATIVE_STATUS
        out.loc[ccd_mask, "baseline_status"] = CCD_NATIVE_STATUS

    return out, {
        "pdb_native_rows_deprecated": int(pdb_mask.sum()),
        "ccd_native_rows_marked_official": int(ccd_mask.sum()),
    }


def candidate_metric_tables(tables_dir: Path) -> list[Path]:
    patterns = [
        "*objective_2d_single_baseline_metrics*.csv",
        "*objective_2d_vs_structure_summary*.csv",
        "*native_ligand*ecfp4*metrics*.csv",
        "*step6_vs_ecfp4_comparison*.csv",
    ]
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(tables_dir.glob(pattern))
    return sorted(paths)


def cleanup_baseline_tables(tables_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "tables_processed": [],
        "tables_modified": [],
        "pdb_native_rows_deprecated": 0,
        "ccd_native_rows_marked_official": 0,
        "pdb_native_deprecated": False,
        "ccd_native_official": False,
    }

    for table_path in candidate_metric_tables(tables_dir):
        try:
            original = pd.read_csv(table_path)
        except Exception as exc:
            summary["tables_processed"].append({"path": str(table_path), "status": "read_failed", "error": str(exc)})
            continue

        marked, counts = mark_baseline_status(original)
        summary["tables_processed"].append({"path": str(table_path), "status": "processed", **counts})
        summary["pdb_native_rows_deprecated"] += counts["pdb_native_rows_deprecated"]
        summary["ccd_native_rows_marked_official"] += counts["ccd_native_rows_marked_official"]

        if counts["pdb_native_rows_deprecated"] or counts["ccd_native_rows_marked_official"]:
            marked.to_csv(table_path, index=False)
            summary["tables_modified"].append(str(table_path))

    summary["pdb_native_deprecated"] = summary["pdb_native_rows_deprecated"] > 0
    summary["ccd_native_official"] = summary["ccd_native_rows_marked_official"] > 0
    return summary


def write_cleanup_report(summary: dict[str, Any], out_report: Path) -> None:
    ensure_dir(out_report.parent)
    report = [
        "# Pre-Step-8 Baseline Cleanup",
        "",
        REQUIRED_CLEANUP_WORDING,
        "",
        "## What Was Deprecated",
        "",
        f"- PDB-native ECFP4 rows marked `{PDB_NATIVE_STATUS}`: {summary['pdb_native_rows_deprecated']}",
        f"- Tables modified: {len(summary['tables_modified'])}",
        "",
        "## Why It Was Deprecated",
        "",
        (
            "Pre-Step-8 native ligand QC showed that all PDB-derived native ligand references were chemically "
            "suspicious because PDB bond-order perception produced all/single-bond fingerprints that do not match "
            "CCD-derived ligand chemistry."
        ),
        "",
        "## CCD-Native Replacement",
        "",
        f"- CCD-native rows marked `{CCD_NATIVE_STATUS}`: {summary['ccd_native_rows_marked_official']}",
        "- CCD-derived native-ligand ECFP4 is the official native-ligand-only baseline for downstream reference.",
        "- Native-ligand-only ECFP4 remains weak after CCD repair and is not used as a primary ranking method.",
        "",
        "## Step 8 Input Policy",
        "",
        "- Step 8 selection uses Uni-Dock/GNINA scores, active/few-active/scaffold/near-analog ECFP4 analyses, and CCD-native results only as contextual reference.",
        "- Step 8 does not use deprecated PDB-native ECFP4 scores for ranking or selection.",
        "- Deprecated rows are retained for provenance and reproducibility.",
        "",
        "## Final Recommendation",
        "",
        (
            "Exclude PDB-native ECFP4 from scientific conclusions. Use CCD-native ECFP4 as the official "
            "native-ligand-only baseline, while interpreting it as a weak structural-reference ligand baseline."
        ),
        "",
    ]
    out_report.write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    args = parse_args()
    tables_dir = Path(args.tables_dir)
    reports_dir = Path(args.reports_dir)
    ensure_dir(reports_dir)

    summary = cleanup_baseline_tables(tables_dir)
    summary["required_wording"] = REQUIRED_CLEANUP_WORDING
    write_cleanup_report(summary, Path(args.out_report))
    write_json(summary, reports_dir / "mapk1_pre_step8_baseline_cleanup_summary.json")
    print(
        "Baseline cleanup complete: "
        f"pdb_deprecated={summary['pdb_native_rows_deprecated']} "
        f"ccd_official={summary['ccd_native_rows_marked_official']}"
    )


if __name__ == "__main__":
    main()
