from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from Bio.PDB import MMCIF2Dict, MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import is_aa


from screensift.common.io import ensure_dir, load_yaml
from screensift.common.logging_utils import setup_logger
from screensift.receptors.fetch_pdbs import get_receptor_config


OUTPUT_COLUMNS = [
    "pdb_id",
    "file_path",
    "method",
    "resolution",
    "chains",
    "ligands",
    "has_ligand",
    "selected",
    "reason",
]
WATER_RESNAMES = {"HOH", "WAT", "DOD", "H2O"}


def structure_files(raw_pdb_dir: str | Path) -> list[Path]:
    raw_dir = Path(raw_pdb_dir)
    if not raw_dir.exists():
        return []
    allowed = {".cif", ".mmcif", ".pdb", ".ent"}
    return [path for path in sorted(raw_dir.rglob("*")) if path.is_file() and path.suffix.lower() in allowed]


def parser_for_path(path: Path):
    if path.suffix.lower() in {".cif", ".mmcif"}:
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def parse_structure(path: Path):
    return parser_for_path(path).get_structure(path.stem.upper(), path)


def first_value(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def parse_float(value: Any) -> float | None:
    value = first_value(value)
    if value in {None, "?", "."}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metadata_from_mmcif(path: Path) -> dict[str, Any]:
    try:
        data = MMCIF2Dict.MMCIF2Dict(str(path))
    except Exception:
        return {}
    return {
        "pdb_id": first_value(data.get("_entry.id")),
        "method": first_value(data.get("_exptl.method")),
        "resolution": parse_float(data.get("_refine.ls_d_res_high")) or parse_float(data.get("_em_3d_reconstruction.resolution")),
    }


def metadata_from_structure(path: Path, structure: Any) -> dict[str, Any]:
    if path.suffix.lower() in {".cif", ".mmcif"}:
        metadata = metadata_from_mmcif(path)
    else:
        header = getattr(structure, "header", {}) or {}
        metadata = {
            "pdb_id": header.get("idcode"),
            "method": header.get("structure_method"),
            "resolution": header.get("resolution"),
        }

    pdb_id = str(metadata.get("pdb_id") or path.stem).upper()
    return {
        "pdb_id": pdb_id,
        "method": metadata.get("method") or "unknown",
        "resolution": parse_float(metadata.get("resolution")),
    }


def ligand_residue_names(structure: Any) -> list[str]:
    ligands: set[str] = set()
    for residue in structure.get_residues():
        hetero_flag = str(residue.id[0]).strip()
        resname = residue.get_resname().strip()
        if not hetero_flag:
            continue
        if resname in WATER_RESNAMES:
            continue
        if is_aa(residue, standard=True):
            continue
        ligands.add(resname)
    return sorted(ligands)


def chain_ids(structure: Any) -> list[str]:
    return sorted({chain.id for model in structure for chain in model})


def inspect_structure(path: Path) -> dict[str, Any]:
    structure = parse_structure(path)
    metadata = metadata_from_structure(path, structure)
    ligands = ligand_residue_names(structure)
    return {
        "pdb_id": metadata["pdb_id"],
        "file_path": str(path),
        "method": metadata["method"],
        "resolution": metadata["resolution"],
        "chains": ";".join(chain_ids(structure)),
        "ligands": ";".join(ligands),
        "has_ligand": bool(ligands),
        "selected": False,
        "reason": "not_selected",
    }


def select_rows(rows: list[dict[str, Any]], receptor_config: dict[str, Any]) -> list[dict[str, Any]]:
    if not rows:
        return rows

    criteria = receptor_config.get("selection_criteria", {})
    preferred_count = int(receptor_config.get("preferred_receptor_count", 5))
    max_resolution = float(criteria.get("max_resolution_angstrom", 3.0))
    prefer_holo = bool(criteria.get("prefer_holo", True))

    eligible: list[dict[str, Any]] = []
    for row in rows:
        resolution = row.get("resolution")
        if resolution is not None and float(resolution) > max_resolution:
            row["reason"] = f"resolution_above_{max_resolution:g}_angstrom"
            continue
        eligible.append(row)

    eligible.sort(
        key=lambda row: (
            0 if (prefer_holo and row.get("has_ligand")) else 1,
            float(row["resolution"]) if row.get("resolution") is not None else 99.0,
            str(row.get("pdb_id", "")),
        )
    )

    selected_ids = {id(row) for row in eligible[:preferred_count]}
    for row in rows:
        if id(row) in selected_ids:
            row["selected"] = True
            row["reason"] = "selected_holo_preferred" if row.get("has_ligand") else "selected_no_ligand_available"
        elif row["reason"] == "not_selected":
            row["reason"] = "lower_ranked_candidate"
    return rows


def write_report(path: Path, rows: list[dict[str, Any]], warnings: list[str]) -> None:
    ensure_dir(path.parent)
    selected = [row for row in rows if row.get("selected")]
    lines = [
        "# MAPK1 Receptor Selection",
        "",
        f"- Structures inspected: `{len(rows)}`",
        f"- Structures selected: `{len(selected)}`",
        "",
        "## Notes",
    ]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No receptor selection warnings.")
    if selected:
        lines.extend(["", "## Selected Structures"])
        for row in selected:
            resolution = row.get("resolution")
            resolution_text = "unknown" if pd.isna(resolution) else f"{float(resolution):.2f} A"
            lines.append(f"- {row['pdb_id']}: {resolution_text}; ligands: {row.get('ligands') or 'none'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_receptors(
    raw_pdb_dir: str | Path,
    receptor_config_path: str | Path,
    out_path: str | Path,
    report_path: str | Path,
    target: str = "MAPK1",
) -> pd.DataFrame:
    logger = setup_logger("select_receptors")
    receptor_config = get_receptor_config(load_yaml(receptor_config_path), target)
    out_path = Path(out_path)
    report_path = Path(report_path)
    ensure_dir(out_path.parent)
    ensure_dir(report_path.parent)

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in structure_files(raw_pdb_dir):
        try:
            row = inspect_structure(path)
            rows.append(row)
            logger.info("Inspected receptor candidate %s", path)
        except Exception as exc:
            warning = f"Could not inspect {path}: {exc}"
            warnings.append(warning)
            logger.warning(warning)

    if not rows:
        warnings.append(
            f"No PDB/mmCIF structures found under {raw_pdb_dir}. Fetch structures or place receptor files there manually."
        )
        frame = pd.DataFrame(columns=OUTPUT_COLUMNS)
        frame.to_csv(out_path, index=False)
        write_report(report_path, rows, warnings)
        return frame

    rows = select_rows(rows, receptor_config)
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    frame.to_csv(out_path, index=False)
    write_report(report_path, rows, warnings)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select MAPK1 receptor structures for ensemble preparation.")
    parser.add_argument("--raw-pdb-dir", default="data/raw/pdb/MAPK1", help="Directory with raw PDB/mmCIF files.")
    parser.add_argument("--receptor-config", default="configs/receptors.yml", help="Receptor configuration YAML.")
    parser.add_argument("--out", default="results/tables/mapk1_receptor_selection.csv", help="Selection CSV.")
    parser.add_argument("--report", default="results/reports/mapk1_receptor_selection.md", help="Selection Markdown report.")
    parser.add_argument("--target", default="MAPK1", help="Target identifier.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    select_receptors(args.raw_pdb_dir, args.receptor_config, args.out, args.report, target=args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
