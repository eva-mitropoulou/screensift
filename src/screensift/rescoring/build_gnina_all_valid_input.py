from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


from screensift.common.io import ensure_dir


OUTPUT_COLUMNS = [
    "ligand_id",
    "activity_label",
    "best_score_unidock",
    "pdb_id",
    "receptor_pdbqt",
    "receptor_clean_pdb",
    "output_pose_file",
    "gnina_receptor_input",
    "gnina_ligand_input",
    "gnina_output_file",
    "valid_for_gnina",
    "invalid_reason",
]


def safe_id(value: Any, default: str = "item") -> str:
    text = str(value if value is not None and not pd.isna(value) else default).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or default


def is_finite_score(value: Any) -> bool:
    try:
        return pd.notna(value) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def normalize_best_unidock(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rename_map = {
        "best_score": "best_score_unidock",
        "best_pdb_id": "pdb_id",
        "best_receptor_pdbqt": "receptor_pdbqt",
        "best_output_pose_file": "output_pose_file",
    }
    result = result.rename(columns={source: target for source, target in rename_map.items() if source in result.columns})
    required = ["ligand_id", "best_score_unidock", "pdb_id", "receptor_pdbqt", "output_pose_file"]
    missing = [column for column in required if column not in result.columns]
    if missing:
        raise ValueError(f"Uni-Dock best-per-ligand table is missing required columns: {missing}")
    result["ligand_id"] = result["ligand_id"].astype(str)
    result["best_score_unidock"] = pd.to_numeric(result["best_score_unidock"], errors="coerce")
    return result


def infer_receptor_clean_pdb(pdb_id: Any, receptor_root: str | Path) -> str:
    return str(Path(receptor_root) / safe_id(pdb_id).lower() / "receptor_clean.pdb")


def infer_gnina_output_file(ligand_id: Any, pdb_id: Any, output_root: str | Path) -> str:
    return str(Path(output_root) / safe_id(pdb_id).lower() / f"{safe_id(ligand_id)}_gnina_score.sdf")


def validate_row(row: pd.Series) -> tuple[bool, str]:
    reasons: list[str] = []
    if not is_finite_score(row["best_score_unidock"]):
        reasons.append("score_missing_or_nonfinite")
    elif float(row["best_score_unidock"]) > 0 or float(row["best_score_unidock"]) < -30:
        reasons.append("score_outside_valid_range")
    if not Path(str(row["output_pose_file"])).exists():
        reasons.append("missing_ligand_pose")
    if not Path(str(row["receptor_clean_pdb"])).exists():
        reasons.append("missing_receptor_clean_pdb")
    if not str(row.get("pdb_id", "")).strip():
        reasons.append("missing_pdb_id")
    return not reasons, ";".join(reasons)


def build_gnina_all_valid_input(
    unidock_best_path: str | Path,
    out_path: str | Path,
    report_path: str | Path,
    receptor_root: str | Path = "data/processed/receptors/MAPK1",
    gnina_output_root: str | Path = "results/poses/gnina/MAPK1/phase1_all_valid",
) -> pd.DataFrame:
    best = normalize_best_unidock(pd.read_csv(unidock_best_path, dtype={"ligand_id": str}))
    best["receptor_clean_pdb"] = best["pdb_id"].map(lambda pdb_id: infer_receptor_clean_pdb(pdb_id, receptor_root))
    best["gnina_receptor_input"] = best["receptor_clean_pdb"]
    best["gnina_ligand_input"] = best["output_pose_file"]
    best["gnina_output_file"] = best.apply(
        lambda row: infer_gnina_output_file(row["ligand_id"], row["pdb_id"], gnina_output_root),
        axis=1,
    )

    validations = best.apply(validate_row, axis=1)
    best["valid_for_gnina"] = [valid for valid, _reason in validations]
    best["invalid_reason"] = [reason for _valid, reason in validations]

    for column in OUTPUT_COLUMNS:
        if column not in best.columns:
            best[column] = pd.NA
    output = best[OUTPUT_COLUMNS].copy()
    out = Path(out_path)
    ensure_dir(out.parent)
    output.to_csv(out, index=False)
    write_report(output, report_path)
    return output


def write_report(frame: pd.DataFrame, report_path: str | Path) -> None:
    valid = frame[frame["valid_for_gnina"]].copy()
    scores = pd.to_numeric(valid["best_score_unidock"], errors="coerce")
    activity_counts = valid["activity_label"].value_counts(dropna=False).to_dict()
    invalid = frame[~frame["valid_for_gnina"]].copy()
    missing_receptor_count = int(frame["invalid_reason"].astype(str).str.contains("missing_receptor_clean_pdb", regex=False).sum())
    missing_ligand_count = int(frame["invalid_reason"].astype(str).str.contains("missing_ligand_pose", regex=False).sum())

    lines = [
        "# MAPK1 Phase 1 GNINA All-Valid Input",
        "",
        "This table expands GNINA score-only rescoring from the selected subset to every valid Uni-Dock best-per-ligand phase 1 row.",
        "",
        "## Summary",
        "",
        f"- total_input_rows: {len(frame)}",
        f"- valid_for_gnina_rows: {len(valid)}",
        f"- invalid_rows: {len(invalid)}",
        f"- missing_receptor_count: {missing_receptor_count}",
        f"- missing_ligand_pose_count: {missing_ligand_count}",
        f"- score_min: {scores.min() if not scores.empty else 'NA'}",
        f"- score_median: {scores.median() if not scores.empty else 'NA'}",
        f"- score_max: {scores.max() if not scores.empty else 'NA'}",
        "",
        "## Activity Counts For Valid Rows",
        "",
    ]
    lines.extend(f"- {label}: {count}" for label, count in activity_counts.items())
    if not invalid.empty:
        lines.extend(["", "## Invalid Row Examples", "", "```text"])
        lines.append(invalid[["ligand_id", "pdb_id", "best_score_unidock", "invalid_reason"]].head(20).to_string(index=False))
        lines.append("```")
    path = Path(report_path)
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full phase 1 all-valid GNINA rescoring input table.")
    parser.add_argument("--unidock-best", default="results/tables/mapk1_phase1_unidock_best_per_ligand.csv")
    parser.add_argument("--out", default="results/tables/mapk1_phase1_gnina_all_valid_input.csv")
    parser.add_argument("--report", default="results/reports/mapk1_phase1_gnina_all_valid_input_report.md")
    parser.add_argument("--receptor-root", default="data/processed/receptors/MAPK1")
    parser.add_argument("--gnina-output-root", default="results/poses/gnina/MAPK1/phase1_all_valid")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = build_gnina_all_valid_input(
        args.unidock_best,
        args.out,
        args.report,
        receptor_root=args.receptor_root,
        gnina_output_root=args.gnina_output_root,
    )
    valid = int(frame["valid_for_gnina"].sum()) if not frame.empty else 0
    print(f"GNINA all-valid input built: rows={len(frame)} valid_for_gnina={valid} invalid={len(frame) - valid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
