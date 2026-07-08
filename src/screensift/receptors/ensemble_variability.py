from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.Data import PDBData
from Bio.PDB import PDBParser, Superimposer


from screensift.common.io import ensure_dir
from screensift.common.logging_utils import setup_logger
from screensift.receptors.define_docking_boxes import ligand_coordinates


OUTPUT_COLUMNS = [
    "residue_key",
    "chain_id",
    "residue_number",
    "residue_name",
    "n_structures",
    "coordinate_dispersion_angstrom",
    "near_pocket",
]


def receptor_clean_files(receptor_dir: str | Path) -> list[Path]:
    root = Path(receptor_dir)
    if not root.exists():
        return []
    return [path for path in sorted(root.glob("*/receptor_clean.pdb")) if path.is_file()]


def parse_pdb(path: Path):
    return PDBParser(QUIET=True).get_structure(path.parent.name.upper(), path)


def residue_key(chain_id: str, residue: Any) -> str:
    hetflag, resseq, icode = residue.id
    return f"{chain_id}:{resseq}:{icode.strip()}:{residue.get_resname().strip()}"


def residue_one_letter(residue: Any) -> str | None:
    resname = residue.get_resname().strip().upper()
    return PDBData.protein_letters_3to1_extended.get(resname) or PDBData.protein_letters_3to1.get(resname)


def primary_chain_ca_residues(structure: Any) -> list[tuple[str, Any, str]]:
    model = next(structure.get_models())
    chain_residues: list[list[tuple[str, Any, str]]] = []
    for chain in model:
        residues: list[tuple[str, Any, str]] = []
        for residue in chain:
            if "CA" not in residue or residue.id[0] != " ":
                continue
            aa = residue_one_letter(residue)
            if not aa:
                continue
            residues.append((residue_key(chain.id, residue), residue, aa))
        if residues:
            chain_residues.append(residues)
    if not chain_residues:
        return []
    return max(chain_residues, key=len)


def sequence_from_residues(residues: list[tuple[str, Any, str]]) -> str:
    return "".join(aa for _key, _residue, aa in residues)


def ca_atom_map(structure: Any) -> dict[str, Any]:
    return {key: residue["CA"] for key, residue, _aa in primary_chain_ca_residues(structure)}


def sequence_aligned_ca_atom_maps(structures: list[Any]) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if not structures:
        return [], warnings

    residue_lists = [primary_chain_ca_residues(structure) for structure in structures]
    if any(not residues for residues in residue_lists):
        warnings.append("At least one receptor structure had no protein C-alpha residues in its primary chain.")
        return [], warnings

    reference_residues = residue_lists[0]
    reference_sequence = sequence_from_residues(reference_residues)
    reference_map = {key: residue["CA"] for key, residue, _aa in reference_residues}
    aligned_maps: list[dict[str, Any]] = [reference_map]

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5

    for structure_index, moving_residues in enumerate(residue_lists[1:], start=2):
        moving_sequence = sequence_from_residues(moving_residues)
        if not moving_sequence:
            aligned_maps.append({})
            warnings.append(f"Structure {structure_index} has an empty primary-chain sequence after filtering.")
            continue
        alignment = aligner.align(reference_sequence, moving_sequence)[0]
        structure_map: dict[str, Any] = {}
        for ref_block, moving_block in zip(*alignment.aligned):
            ref_start, ref_end = int(ref_block[0]), int(ref_block[1])
            moving_start, moving_end = int(moving_block[0]), int(moving_block[1])
            block_len = min(ref_end - ref_start, moving_end - moving_start)
            for offset in range(block_len):
                ref_index = ref_start + offset
                moving_index = moving_start + offset
                ref_key, _ref_residue, ref_aa = reference_residues[ref_index]
                _moving_key, moving_residue, moving_aa = moving_residues[moving_index]
                if ref_aa != moving_aa:
                    continue
                structure_map[ref_key] = moving_residue["CA"]
        aligned_maps.append(structure_map)

    return aligned_maps, warnings


def common_residue_keys(maps: list[dict[str, Any]]) -> list[str]:
    if not maps:
        return []
    common = set(maps[0])
    for atom_map in maps[1:]:
        common &= set(atom_map)
    return sorted(common, key=lambda key: (key.split(":")[0], int(key.split(":")[1]), key))


def align_structures(structures: list[Any], keys: list[str]) -> list[dict[str, Any]]:
    if not structures or not keys:
        return []
    atom_maps, _warnings = sequence_aligned_ca_atom_maps(structures)
    reference_atoms = [atom_maps[0][key] for key in keys]
    aligned_maps: list[dict[str, Any]] = [atom_maps[0]]

    for structure, atom_map in zip(structures[1:], atom_maps[1:]):
        moving_atoms = [atom_map[key] for key in keys]
        superimposer = Superimposer()
        superimposer.set_atoms(reference_atoms, moving_atoms)
        superimposer.apply(structure.get_atoms())
        aligned_maps.append(atom_map)
    return aligned_maps


def key_metadata(key: str) -> tuple[str, int, str]:
    chain_id, residue_number, _icode, residue_name = key.split(":", 3)
    return chain_id, int(residue_number), residue_name


def pocket_keys_from_reference(reference_structure: Any, ligand_coords: np.ndarray | None, distance_cutoff: float = 6.0) -> set[str]:
    if ligand_coords is None or ligand_coords.size == 0:
        return set()
    pocket_keys: set[str] = set()
    model = next(reference_structure.get_models())
    for chain in model:
        for residue in chain:
            key = residue_key(chain.id, residue)
            atom_coords = np.asarray([atom.coord for atom in residue.get_atoms()], dtype=float)
            if atom_coords.size == 0:
                continue
            deltas = atom_coords[:, None, :] - ligand_coords[None, :, :]
            min_distance = float(np.sqrt((deltas * deltas).sum(axis=2)).min())
            if min_distance <= distance_cutoff:
                pocket_keys.add(key)
    return pocket_keys


def first_ligand_coordinates(receptor_files: list[Path], boxes_path: Path) -> np.ndarray | None:
    if boxes_path.exists() and boxes_path.stat().st_size > 0:
        try:
            boxes = pd.read_csv(boxes_path)
            if "ligand_file" in boxes.columns:
                for value in boxes["ligand_file"].dropna():
                    ligand_path = Path(str(value))
                    if ligand_path.exists():
                        coords = ligand_coordinates(ligand_path)
                        if coords.size:
                            return coords
        except Exception:
            pass

    for receptor_file in receptor_files:
        for name in ("native_ligand.sdf", "native_ligand.pdb"):
            ligand_path = receptor_file.parent / name
            if ligand_path.exists():
                coords = ligand_coordinates(ligand_path)
                if coords.size:
                    return coords
    return None


def variability_table(receptor_files: list[Path], boxes_path: Path) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if len(receptor_files) < 2:
        warnings.append("At least two prepared receptors are needed for receptor ensemble variability; wrote empty or zero-dispersion output.")
    if not receptor_files:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), warnings

    structures = [parse_pdb(path) for path in receptor_files]
    atom_maps, sequence_warnings = sequence_aligned_ca_atom_maps(structures)
    warnings.extend(sequence_warnings)
    keys = common_residue_keys(atom_maps)
    if not keys:
        warnings.append("No common sequence-aligned C-alpha residues were found across prepared receptor structures.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS), warnings
    if len(keys) < 100:
        warnings.append(
            f"Only {len(keys)} common sequence-aligned C-alpha residues were found; "
            "treat receptor ensemble variability as a coarse diagnostic."
        )

    aligned_maps = align_structures(structures, keys)
    coords = np.asarray([[aligned_maps[i][key].coord for key in keys] for i in range(len(aligned_maps))], dtype=float)
    mean_coords = coords.mean(axis=0)
    dispersion = np.sqrt(((coords - mean_coords[None, :, :]) ** 2).sum(axis=2)).mean(axis=0)

    ligand_coords = first_ligand_coordinates(receptor_files, boxes_path)
    pocket_keys = pocket_keys_from_reference(structures[0], ligand_coords)

    rows: list[dict[str, Any]] = []
    for key, value in zip(keys, dispersion):
        chain_id, residue_number, residue_name = key_metadata(key)
        rows.append(
            {
                "residue_key": key,
                "chain_id": chain_id,
                "residue_number": residue_number,
                "residue_name": residue_name,
                "n_structures": len(receptor_files),
                "coordinate_dispersion_angstrom": float(value),
                "near_pocket": key in pocket_keys,
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS), warnings


def write_figure(frame: pd.DataFrame, figure_path: Path) -> None:
    ensure_dir(figure_path.parent)
    fig, ax = plt.subplots(figsize=(9, 4))
    if frame.empty:
        ax.text(0.5, 0.5, "No receptor ensemble variability data", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.plot(frame["residue_number"], frame["coordinate_dispersion_angstrom"], color="#2f6f73", linewidth=1.5)
        pocket = frame[frame["near_pocket"].astype(bool)]
        if not pocket.empty:
            ax.scatter(
                pocket["residue_number"],
                pocket["coordinate_dispersion_angstrom"],
                color="#b3432f",
                s=18,
                label="Near native ligand",
            )
            ax.legend(frameon=False)
        ax.set_xlabel("Residue number")
        ax.set_ylabel("Coordinate dispersion (A)")
        ax.set_title("MAPK1 receptor ensemble variability")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def write_report(path: Path, frame: pd.DataFrame, receptor_files: list[Path], warnings: list[str]) -> None:
    ensure_dir(path.parent)
    max_dispersion = None if frame.empty else float(frame["coordinate_dispersion_angstrom"].max())
    pocket_count = 0 if frame.empty else int(frame["near_pocket"].sum())
    lines = [
        "# MAPK1 Receptor Ensemble Variability",
        "",
        "This report uses per-residue coordinate dispersion after sequence-mapped C-alpha alignment. It is not RMSF.",
        "",
        "Residues are matched by aligning each receptor primary-chain sequence to the reference receptor sequence, "
        "then superimposing the corresponding C-alpha atoms. This avoids overly strict PDB residue-number matching.",
        "",
        f"- Prepared receptors: `{len(receptor_files)}`",
        f"- Common sequence-aligned C-alpha residues: `{len(frame)}`",
        f"- Pocket residues marked: `{pocket_count}`",
        f"- Maximum coordinate dispersion: `{max_dispersion}`",
        "",
        "## Warnings",
    ]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No ensemble variability warnings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ensemble_variability(
    receptor_dir: str | Path,
    boxes_path: str | Path,
    out_path: str | Path,
    figure_path: str | Path,
    report_path: str | Path,
) -> pd.DataFrame:
    logger = setup_logger("ensemble_variability")
    receptor_files = receptor_clean_files(receptor_dir)
    frame, warnings = variability_table(receptor_files, Path(boxes_path))
    out_path = Path(out_path)
    figure_path = Path(figure_path)
    report_path = Path(report_path)
    ensure_dir(out_path.parent)
    frame.to_csv(out_path, index=False)
    write_figure(frame, figure_path)
    write_report(report_path, frame, receptor_files, warnings)
    logger.info("Wrote receptor ensemble variability table to %s", out_path)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MAPK1 receptor ensemble variability.")
    parser.add_argument("--receptor-dir", default="data/processed/receptors/MAPK1", help="Prepared receptor directory.")
    parser.add_argument("--boxes", default="results/tables/mapk1_docking_boxes.csv", help="Docking box CSV.")
    parser.add_argument("--out", default="results/tables/mapk1_receptor_ensemble_variability.csv", help="Output CSV.")
    parser.add_argument(
        "--figure",
        default="results/figures/mapk1_receptor_ensemble_variability.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--report",
        default="results/reports/mapk1_receptor_ensemble_variability.md",
        help="Output Markdown report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_ensemble_variability(args.receptor_dir, args.boxes, args.out, args.figure, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
