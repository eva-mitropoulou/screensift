from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from Bio.PDB import PDBIO, Select
from Bio.PDB.Polypeptide import is_aa


from screensift.common.io import ensure_dir, load_yaml, write_json
from screensift.common.logging_utils import setup_logger
from screensift.receptors.select_receptors import parse_structure


WATER_RESNAMES = {"HOH", "WAT", "DOD", "H2O"}


class ProteinSelect(Select):
    def accept_residue(self, residue: Any) -> bool:
        return bool(is_aa(residue, standard=True))


class LigandResidueSelect(Select):
    def __init__(self, chain_id: str, residue_id: tuple[Any, Any, Any]) -> None:
        self.chain_id = chain_id
        self.residue_id = residue_id

    def accept_residue(self, residue: Any) -> bool:
        parent = residue.get_parent()
        return bool(parent.id == self.chain_id and residue.id == self.residue_id)


def parse_boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def likely_ligand_residues(structure: Any) -> list[tuple[str, Any, int]]:
    residues: list[tuple[str, Any, int]] = []
    for residue in structure.get_residues():
        hetero_flag = str(residue.id[0]).strip()
        resname = residue.get_resname().strip()
        if not hetero_flag:
            continue
        if resname in WATER_RESNAMES:
            continue
        if is_aa(residue, standard=True):
            continue
        heavy_atoms = sum(1 for atom in residue.get_atoms() if atom.element != "H")
        if heavy_atoms > 0:
            residues.append((residue.get_parent().id, residue, heavy_atoms))
    residues.sort(key=lambda item: (item[2], item[1].get_resname()), reverse=True)
    return residues


def write_protein_pdb(structure: Any, out_path: Path) -> None:
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path), ProteinSelect())


def write_ligand_pdb(structure: Any, chain_id: str, residue: Any, out_path: Path) -> None:
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path), LigandResidueSelect(chain_id, residue.id))


def meeko_receptor_command() -> str | None:
    for name in ("mk_prepare_receptor.py", "mk_prepare_receptor"):
        found = shutil.which(name)
        if found:
            return found
    return None


def prepare_receptor_pdbqt(receptor_pdb: Path, out_prefix: Path) -> tuple[str, Path | None, str]:
    command = meeko_receptor_command()
    if command is None:
        return "missing_meeko_receptor_command", None, "mk_prepare_receptor.py was not found on PATH."

    pdbqt_path = out_prefix.with_suffix(".pdbqt")
    candidates = [
        [command, "-i", str(receptor_pdb), "-o", str(out_prefix), "-p"],
        [command, "--read_pdb", str(receptor_pdb), "-o", str(out_prefix), "-p"],
        [command, "-i", str(receptor_pdb), "-o", str(out_prefix), "-p", "-a", "--default_altloc", "A"],
        [command, "--read_pdb", str(receptor_pdb), "-o", str(out_prefix), "-p", "-a", "--default_altloc", "A"],
    ]
    errors: list[str] = []
    for candidate in candidates:
        try:
            completed = subprocess.run(candidate, check=False, capture_output=True, text=True, timeout=300)
        except Exception as exc:
            errors.append(f"{' '.join(candidate)} -> {exc}")
            continue
        if completed.returncode == 0 and pdbqt_path.exists():
            return "complete", pdbqt_path, " ".join(candidate)
        errors.append(f"{' '.join(candidate)} -> rc={completed.returncode}; stderr={completed.stderr.strip()[:500]}")
    return "failed", None, " | ".join(errors)


def write_notes(
    notes_path: Path,
    pdb_id: str,
    source_file: Path,
    ligand_rows: list[tuple[str, Any, int]],
    receptor_pdbqt_status: str,
    receptor_pdbqt_note: str,
) -> None:
    lines = [
        f"# Receptor Prep Notes: {pdb_id}",
        "",
        f"- Source file: `{source_file}`",
        f"- Receptor PDBQT status: `{receptor_pdbqt_status}`",
        "",
        "## Non-water Hetero Residues",
    ]
    if ligand_rows:
        for chain_id, residue, heavy_atoms in ligand_rows:
            hetflag, resseq, icode = residue.id
            lines.append(
                f"- {residue.get_resname().strip()} chain {chain_id} residue {resseq}{icode.strip()} "
                f"({heavy_atoms} heavy atoms; hetero flag `{hetflag}`)"
            )
    else:
        lines.append("- None detected.")
    lines.extend(["", "## Preparation Log", receptor_pdbqt_note or "No preparation notes."])
    notes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_one_receptor(row: pd.Series, out_dir: Path) -> dict[str, Any]:
    pdb_id = str(row.get("pdb_id") or Path(str(row["file_path"])).stem).lower()
    source_file = Path(str(row["file_path"]))
    receptor_dir = ensure_dir(out_dir / pdb_id)
    receptor_clean = receptor_dir / "receptor_clean.pdb"
    native_ligand = receptor_dir / "native_ligand.pdb"
    notes_path = receptor_dir / "receptor_prep_notes.md"

    result: dict[str, Any] = {
        "pdb_id": pdb_id.upper(),
        "source_file": str(source_file),
        "receptor_dir": str(receptor_dir),
        "receptor_clean_pdb": str(receptor_clean),
        "native_ligand_file": None,
        "receptor_pdbqt": None,
        "status": "failed",
        "receptor_pdbqt_status": "not_attempted",
        "message": "",
    }

    try:
        structure = parse_structure(source_file)
        ligand_rows = likely_ligand_residues(structure)
        write_protein_pdb(structure, receptor_clean)
        if ligand_rows:
            chain_id, ligand_residue, _heavy_atoms = ligand_rows[0]
            write_ligand_pdb(structure, chain_id, ligand_residue, native_ligand)
            result["native_ligand_file"] = str(native_ligand)

        pdbqt_status, pdbqt_path, pdbqt_note = prepare_receptor_pdbqt(receptor_clean, receptor_dir / "receptor")
        result["receptor_pdbqt_status"] = pdbqt_status
        result["receptor_pdbqt"] = str(pdbqt_path) if pdbqt_path else None
        result["status"] = "complete" if receptor_clean.exists() else "failed"
        result["message"] = pdbqt_note
        write_notes(notes_path, result["pdb_id"], source_file, ligand_rows, pdbqt_status, pdbqt_note)
    except Exception as exc:
        result["status"] = "failed"
        result["message"] = str(exc)
        notes_path.write_text(f"# Receptor Prep Notes: {result['pdb_id']}\n\nPreparation failed: {exc}\n", encoding="utf-8")
    return result


def prepare_receptors(
    selection_path: str | Path,
    paths_path: str | Path,
    out_dir: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    logger = setup_logger("prepare_receptors")
    _paths = load_yaml(paths_path)
    selection_path = Path(selection_path)
    out_dir = ensure_dir(out_dir)
    report_path = Path(report_path)
    ensure_dir(report_path.parent)

    if not selection_path.exists() or selection_path.stat().st_size == 0:
        manifest = {"status": "missing_selection", "receptors": [], "warnings": [f"Selection file not found: {selection_path}"]}
        write_json(manifest, report_path)
        return manifest

    selection = pd.read_csv(selection_path)
    if selection.empty or "selected" not in selection.columns:
        manifest = {
            "status": "no_selected_receptors",
            "receptors": [],
            "warnings": ["No selected receptor rows are available."],
        }
        write_json(manifest, report_path)
        return manifest

    selected = selection[selection["selected"].map(parse_boolish)].copy()
    if selected.empty:
        manifest = {
            "status": "no_selected_receptors",
            "receptors": [],
            "warnings": ["Receptor selection did not select any structures."],
        }
        write_json(manifest, report_path)
        return manifest

    results: list[dict[str, Any]] = []
    for _index, row in selected.iterrows():
        result = prepare_one_receptor(row, out_dir)
        results.append(result)
        logger.info("Prepared receptor %s with status %s", result["pdb_id"], result["status"])

    successful = sum(1 for result in results if result["status"] == "complete")
    manifest = {
        "status": "complete" if successful else "no_receptors_prepared",
        "total_selected": int(len(selected)),
        "successful": int(successful),
        "failed": int(len(results) - successful),
        "receptors": results,
        "warnings": [
            "Meeko receptor PDBQT preparation is best-effort in Step 3; receptor_clean.pdb is retained if it fails.",
            "If strict Meeko preparation fails, the script retries with --allow_bad_res --default_altloc A and records the command used.",
        ],
    }
    write_json(manifest, report_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare selected MAPK1 receptor structures.")
    parser.add_argument("--selection", default="results/tables/mapk1_receptor_selection.csv", help="Receptor selection CSV.")
    parser.add_argument("--paths", default="configs/paths.yml", help="Project path configuration YAML.")
    parser.add_argument("--out-dir", default="data/processed/receptors/MAPK1", help="Prepared receptor output directory.")
    parser.add_argument("--report", default="results/reports/mapk1_receptor_prep_manifest.json", help="Preparation manifest JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepare_receptors(args.selection, args.paths, args.out_dir, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
