from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem


from screensift.common.io import ensure_dir, load_yaml
from screensift.common.logging_utils import setup_logger
from screensift.receptors.fetch_pdbs import get_receptor_config


OUTPUT_COLUMNS = [
    "pdb_id",
    "center_x",
    "center_y",
    "center_z",
    "size_x",
    "size_y",
    "size_z",
    "ligand_file",
    "receptor_pdbqt",
    "status",
    "notes",
]


def coordinates_from_pdb(path: str | Path) -> np.ndarray:
    coords: list[tuple[float, float, float]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except ValueError:
            continue
    return np.asarray(coords, dtype=float)


def coordinates_from_sdf(path: str | Path) -> np.ndarray:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    for mol in supplier:
        if mol is None or mol.GetNumConformers() == 0:
            continue
        conf = mol.GetConformer()
        coords = [conf.GetAtomPosition(index) for index in range(mol.GetNumAtoms())]
        return np.asarray([(point.x, point.y, point.z) for point in coords], dtype=float)
    return np.empty((0, 3), dtype=float)


def ligand_coordinates(path: str | Path) -> np.ndarray:
    ligand_path = Path(path)
    if ligand_path.suffix.lower() in {".sdf", ".sd"}:
        return coordinates_from_sdf(ligand_path)
    return coordinates_from_pdb(ligand_path)


def calculate_docking_box(
    coordinates: np.ndarray,
    padding: float = 6.0,
    min_box_size: float = 18.0,
) -> dict[str, float]:
    if coordinates.size == 0:
        raise ValueError("Cannot calculate docking box from zero coordinates.")
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"Expected coordinate array with shape (n, 3), got {coordinates.shape}.")

    # Center on the bounding-box midpoint, not the atom centroid. The box size
    # is derived from the axis-aligned extent, so only the midpoint guarantees
    # the configured `padding` clearance on every face; a centroid is pulled
    # toward denser regions and can leave < padding on one side (or clip the
    # ligand when the box hits the min_box_size floor).
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    center = (lower + upper) / 2.0
    extents = upper - lower
    sizes = np.maximum(extents + (2.0 * padding), min_box_size)
    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "size_x": float(sizes[0]),
        "size_y": float(sizes[1]),
        "size_z": float(sizes[2]),
    }


def prepared_receptor_dirs(receptor_dir: str | Path) -> list[Path]:
    root = Path(receptor_dir)
    if not root.exists():
        return []
    return [path for path in sorted(root.iterdir()) if path.is_dir()]


def box_row_for_receptor(receptor_path: Path, padding: float, min_box_size: float) -> dict[str, Any]:
    ligand_candidates = [receptor_path / "native_ligand.sdf", receptor_path / "native_ligand.pdb"]
    ligand_file = next((path for path in ligand_candidates if path.exists()), None)
    receptor_pdbqt = receptor_path / "receptor.pdbqt"
    row = {
        "pdb_id": receptor_path.name.upper(),
        "center_x": None,
        "center_y": None,
        "center_z": None,
        "size_x": None,
        "size_y": None,
        "size_z": None,
        "ligand_file": str(ligand_file) if ligand_file else None,
        "receptor_pdbqt": str(receptor_pdbqt) if receptor_pdbqt.exists() else None,
        "status": "missing_ligand",
        "notes": "No native ligand file was found for this prepared receptor.",
    }
    if ligand_file is None:
        return row

    try:
        coords = ligand_coordinates(ligand_file)
        box = calculate_docking_box(coords, padding=padding, min_box_size=min_box_size)
        row.update(box)
        row["status"] = "complete"
        row["notes"] = "Docking box calculated from native ligand coordinates."
    except Exception as exc:
        row["status"] = "failed"
        row["notes"] = str(exc)
    return row


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    complete = sum(1 for row in rows if row.get("status") == "complete")
    lines = [
        "# MAPK1 Docking Box Report",
        "",
        f"- Prepared receptors inspected: `{len(rows)}`",
        f"- Boxes defined: `{complete}`",
        "",
        "Docking boxes are derived from native ligand coordinates with configured padding and minimum size.",
        "",
        "## Receptor Status",
    ]
    if not rows:
        lines.append("- No prepared receptor directories were found.")
    for row in rows:
        lines.append(f"- {row['pdb_id']}: `{row['status']}` - {row['notes']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def define_docking_boxes(
    receptor_dir: str | Path,
    receptor_config_path: str | Path,
    out_path: str | Path,
    report_path: str | Path,
    target: str = "MAPK1",
) -> pd.DataFrame:
    logger = setup_logger("define_docking_boxes")
    receptor_config = get_receptor_config(load_yaml(receptor_config_path), target)
    box_config = receptor_config.get("docking_box", {})
    padding = float(box_config.get("box_padding_angstrom", 6.0))
    min_box_size = float(box_config.get("min_box_size_angstrom", 18.0))

    rows = [box_row_for_receptor(path, padding, min_box_size) for path in prepared_receptor_dirs(receptor_dir)]
    out_path = Path(out_path)
    report_path = Path(report_path)
    ensure_dir(out_path.parent)
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    frame.to_csv(out_path, index=False)
    write_report(report_path, rows)
    logger.info("Wrote docking box table to %s", out_path)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Define MAPK1 docking boxes from native ligand coordinates.")
    parser.add_argument("--receptor-dir", default="data/processed/receptors/MAPK1", help="Prepared receptor directory.")
    parser.add_argument("--receptor-config", default="configs/receptors.yml", help="Receptor configuration YAML.")
    parser.add_argument("--out", default="results/tables/mapk1_docking_boxes.csv", help="Docking box CSV.")
    parser.add_argument("--report", default="results/reports/mapk1_docking_box_report.md", help="Docking box Markdown report.")
    parser.add_argument("--target", default="MAPK1", help="Target identifier.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    define_docking_boxes(args.receptor_dir, args.receptor_config, args.out, args.report, target=args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
