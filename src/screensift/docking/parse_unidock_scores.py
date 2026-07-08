from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


from screensift.common.io import ensure_dir
from screensift.docking.run_unidock import parse_best_score_from_pose


OUTPUT_COLUMNS = ["ligand_id", "pdb_id", "best_score", "output_pose_file", "status"]


def infer_ligand_id(ligand_pdbqt: Any, docking_id: Any) -> str:
    if ligand_pdbqt is not None and str(ligand_pdbqt).strip() and str(ligand_pdbqt).lower() != "nan":
        return Path(str(ligand_pdbqt)).stem
    text = str(docking_id)
    return text.split("_", 1)[1] if "_" in text else text


def parse_unidock_scores(raw_path: str | Path, out_path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)
    rows: list[dict[str, Any]] = []
    for row in raw.to_dict(orient="records"):
        score = row.get("best_score")
        if pd.isna(score) and row.get("output_pose_file"):
            score = parse_best_score_from_pose(row["output_pose_file"])
        rows.append(
            {
                "ligand_id": infer_ligand_id(row.get("ligand_pdbqt"), row.get("docking_id")),
                "pdb_id": row.get("pdb_id"),
                "best_score": score,
                "output_pose_file": row.get("output_pose_file"),
                "status": row.get("status"),
            }
        )

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    frame.to_csv(out_path, index=False)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse raw Uni-Dock smoke outputs into a clean score table.")
    parser.add_argument("--raw", default="results/tables/mapk1_phase1_unidock_smoke_raw.csv")
    parser.add_argument("--out", default="results/tables/mapk1_phase1_unidock_smoke_scores.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parse_unidock_scores(args.raw, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
