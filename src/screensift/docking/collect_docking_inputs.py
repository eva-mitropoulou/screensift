from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


from screensift.common.io import ensure_dir


OUTPUT_COLUMNS = [
    "docking_id",
    "ligand_pdbqt",
    "receptor_pdbqt",
    "pdb_id",
    "center_x",
    "center_y",
    "center_z",
    "size_x",
    "size_y",
    "size_z",
    "output_dir",
]


def safe_id(value: Any, max_length: int = 120) -> str:
    text = str(value if value is not None else "item").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return (text or "item")[:max_length]


def find_ligand_pdbqt_files(ligand_pdbqt_dir: str | Path) -> list[Path]:
    root = Path(ligand_pdbqt_dir)
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.pdbqt") if path.is_file())


def load_complete_boxes(boxes_path: str | Path, single_receptor: bool = False) -> pd.DataFrame:
    boxes = pd.read_csv(boxes_path)
    required = {
        "pdb_id",
        "center_x",
        "center_y",
        "center_z",
        "size_x",
        "size_y",
        "size_z",
        "receptor_pdbqt",
        "status",
    }
    missing = sorted(required - set(boxes.columns))
    if missing:
        raise ValueError(f"Docking boxes table is missing required columns: {missing}")

    complete = boxes[(boxes["status"] == "complete") & boxes["receptor_pdbqt"].notna()].copy()
    complete = complete[complete["receptor_pdbqt"].map(lambda value: Path(str(value)).exists())]
    complete = complete.sort_values("pdb_id").reset_index(drop=True)
    if single_receptor and not complete.empty:
        complete = complete.head(1).copy()
    return complete


def collect_docking_inputs(
    ligand_pdbqt_dir: str | Path,
    boxes_path: str | Path,
    out_path: str | Path,
    limit: int | None = None,
    single_receptor: bool = False,
    output_root: str | Path = "results/poses/unidock/MAPK1/phase1",
) -> pd.DataFrame:
    ligand_files = find_ligand_pdbqt_files(ligand_pdbqt_dir)
    if limit is not None:
        ligand_files = ligand_files[:limit]
    boxes = load_complete_boxes(boxes_path, single_receptor=single_receptor)

    rows: list[dict[str, Any]] = []
    for box in boxes.to_dict(orient="records"):
        pdb_id = safe_id(box["pdb_id"])
        receptor_output_dir = Path(output_root) / pdb_id.lower()
        for ligand_path in ligand_files:
            ligand_id = safe_id(ligand_path.stem)
            rows.append(
                {
                    "docking_id": safe_id(f"{pdb_id}_{ligand_id}"),
                    "ligand_pdbqt": str(ligand_path),
                    "receptor_pdbqt": str(box["receptor_pdbqt"]),
                    "pdb_id": pdb_id,
                    "center_x": float(box["center_x"]),
                    "center_y": float(box["center_y"]),
                    "center_z": float(box["center_z"]),
                    "size_x": float(box["size_x"]),
                    "size_y": float(box["size_y"]),
                    "size_z": float(box["size_z"]),
                    "output_dir": str(receptor_output_dir),
                }
            )

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    frame.to_csv(out_path, index=False)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect receptor-ligand Uni-Dock input combinations.")
    parser.add_argument("--ligand-pdbqt-dir", default="data/processed/ligands/MAPK1/phase1/pdbqt")
    parser.add_argument("--boxes", default="results/tables/mapk1_docking_boxes.csv")
    parser.add_argument("--out", default="results/tables/mapk1_phase1_docking_inputs.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--single-receptor", action="store_true")
    parser.add_argument("--output-root", default="results/poses/unidock/MAPK1/phase1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    collect_docking_inputs(
        args.ligand_pdbqt_dir,
        args.boxes,
        args.out,
        limit=args.limit,
        single_receptor=args.single_receptor,
        output_root=args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
