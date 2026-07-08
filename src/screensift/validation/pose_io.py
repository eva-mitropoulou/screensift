from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem


SUPPORTED_POSE_SUFFIXES = {".sdf", ".pdbqt", ".pdb", ".mol2"}


@dataclass(frozen=True)
class PoseLoadResult:
    path: Path
    format: str
    success: bool
    mol: Any | None = None
    failure_reason: str = ""


@dataclass(frozen=True)
class AtomRecord:
    serial: int
    name: str
    residue_name: str
    chain_id: str
    residue_id: str
    element: str
    x: float
    y: float
    z: float


def infer_pose_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdbqt":
        return "pdbqt"
    if suffix == ".pdb":
        return "pdb"
    if suffix == ".sdf":
        return "sdf"
    if suffix == ".mol2":
        return "mol2"
    return "unknown"


def extract_ligand_id_from_pose_path(path: str | Path) -> str:
    """Recover the ligand id from a pose file name.

    Pose files are named ``{split_index:07d}_{ligand_id}[_out].pdbqt`` (see
    ``collect_docking_inputs``/``run_unidock``), so the ligand id is the field
    after the leading zero-padded split index -- NOT the first numeric token,
    which is the index itself. This mirrors ``ligand_id_from_ligand_stem`` in
    ``docking.audit_unidock_scores``.
    """
    stem = Path(path).stem
    # Drop a trailing pose-role suffix such as "_out" or "_pose".
    stem = re.sub(r"_(out|pose|docked|redocked)$", "", stem)
    # Drop an optional leading PDB-id token (digit + letters, e.g. "4QTA_"),
    # which prefixes docking_id-style pose names but never an index.
    stem = re.sub(r"^[0-9][A-Za-z][A-Za-z0-9]{2}_", "", stem)
    # Strip the leading zero-padded split index and keep the rest, matching
    # ligand_id_from_ligand_stem in docking.audit_unidock_scores:
    #   "0000032_842954"        -> "842954"
    #   "0000005_ligand_0000005" -> "ligand_0000005"
    match = re.match(r"^\d+_(.+)$", stem)
    if match:
        return match.group(1)
    # No index prefix: fall back to the first >=4-digit token, then the stem.
    for token in re.split(r"[_\-.]", stem):
        if token.isdigit() and len(token) >= 4:
            return str(int(token))
    fallback = re.search(r"(\d{4,})", stem)
    return str(int(fallback.group(1))) if fallback else stem


def _mol_from_sdf(path: Path):
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    for mol in supplier:
        if mol is not None:
            return mol
    return None


def _mol_from_pdbqt_with_openbabel(path: Path):
    obabel = shutil.which("obabel") or shutil.which("openbabel")
    if not obabel:
        return None, "openbabel_not_available_for_pdbqt_conversion"
    with tempfile.TemporaryDirectory() as tmpdir:
        out_sdf = Path(tmpdir) / "pose.sdf"
        completed = subprocess.run(
            [obabel, str(path), "-O", str(out_sdf)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if completed.returncode != 0 or not out_sdf.exists() or out_sdf.stat().st_size == 0:
            return None, f"openbabel_conversion_failed: {completed.stderr.strip()[:300]}"
        mol = _mol_from_sdf(out_sdf)
        if mol is None:
            return None, "openbabel_output_not_readable_by_rdkit"
        return mol, ""


def safe_rdkit_mol_from_pose(path: str | Path) -> PoseLoadResult:
    pose_path = Path(path)
    fmt = infer_pose_format(pose_path)
    if not pose_path.exists():
        return PoseLoadResult(path=pose_path, format=fmt, success=False, failure_reason="missing_pose_file")
    try:
        if fmt == "sdf":
            mol = _mol_from_sdf(pose_path)
        elif fmt == "pdb":
            mol = Chem.MolFromPDBFile(str(pose_path), sanitize=False, removeHs=False)
        elif fmt == "mol2":
            mol = Chem.MolFromMol2File(str(pose_path), sanitize=False, removeHs=False)
        elif fmt == "pdbqt":
            mol, failure = _mol_from_pdbqt_with_openbabel(pose_path)
            if mol is None:
                return PoseLoadResult(path=pose_path, format=fmt, success=False, failure_reason=failure)
        else:
            return PoseLoadResult(path=pose_path, format=fmt, success=False, failure_reason="unsupported_pose_format")
    except Exception as exc:
        return PoseLoadResult(path=pose_path, format=fmt, success=False, failure_reason=f"pose_load_exception: {exc}")
    if mol is None:
        return PoseLoadResult(path=pose_path, format=fmt, success=False, failure_reason="rdkit_returned_none")
    return PoseLoadResult(path=pose_path, format=fmt, success=True, mol=mol)


def load_pose_file(path: str | Path) -> PoseLoadResult:
    return safe_rdkit_mol_from_pose(path)


def safe_mdanalysis_universe(receptor_path: str | Path, pose_path: str | Path):
    try:
        import MDAnalysis as mda

        return mda.Universe(str(receptor_path), str(pose_path)), ""
    except Exception as exc:
        return None, f"mdanalysis_universe_failed: {exc}"


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def stable_ligand_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def discover_pose_files(search_dirs: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for search_dir in search_dirs:
        root = Path(search_dir)
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_POSE_SUFFIXES)
    return sorted(set(files))


def pose_source(path: str | Path) -> str:
    text = str(path).lower()
    if "gnina" in text:
        return "gnina"
    if "unidock" in text:
        return "unidock"
    return "other"


def pose_priority(path: Path) -> tuple[int, str]:
    source = pose_source(path)
    return {"gnina": 0, "unidock": 1, "other": 2}.get(source, 3), str(path)


def resolve_pose_path(row: pd.Series, pose_index: dict[str, list[Path]] | None = None) -> tuple[Path | None, str]:
    selected = str(row.get("selected_pose_file", "") or "").strip()
    if selected and selected.lower() != "nan" and Path(selected).exists():
        return Path(selected), ""
    ligand_id = stable_ligand_id(row.get("ligand_id"))
    if pose_index and ligand_id in pose_index and pose_index[ligand_id]:
        return sorted(pose_index[ligand_id], key=pose_priority)[0], "resolved_from_pose_tree"
    return None, "pose_missing"


def build_pose_index(search_dirs: list[str | Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in discover_pose_files(search_dirs):
        ligand_id = extract_ligand_id_from_pose_path(path)
        if ligand_id:
            index.setdefault(ligand_id, []).append(path)
    return index


def infer_pdb_id_from_path(path: str | Path) -> str:
    parts = [part.lower() for part in Path(path).parts]
    for part in parts:
        if re.fullmatch(r"[0-9][a-z0-9]{3}", part):
            return part
    text = str(path).lower()
    match = re.search(r"/([0-9][a-z0-9]{3})/", text)
    return match.group(1) if match else ""


def find_receptor_for_pose(row: pd.Series, prepared_receptors_df: pd.DataFrame | None = None) -> tuple[Path | None, str]:
    pose_path = str(row.get("selected_pose_file", "") or "")
    pdb_id = str(row.get("pdb_id", "") or "").lower()
    if not pdb_id:
        pdb_id = infer_pdb_id_from_path(pose_path)

    if prepared_receptors_df is not None and not prepared_receptors_df.empty and pdb_id:
        lower_cols = {col.lower(): col for col in prepared_receptors_df.columns}
        pdb_col = lower_cols.get("pdb_id")
        if pdb_col:
            matches = prepared_receptors_df[prepared_receptors_df[pdb_col].fillna("").astype(str).str.lower().eq(pdb_id)]
            for col in ["receptor_clean_pdb", "receptor_pdb", "receptor_file", "receptor_clean", "file_path"]:
                actual = lower_cols.get(col)
                if actual and not matches.empty:
                    candidate = Path(str(matches.iloc[0].get(actual, "")))
                    if candidate.exists():
                        return candidate, ""

    if pdb_id:
        candidate_paths: list[Path] = []
        receptor_root = Path("data/processed/receptors")
        if receptor_root.exists():
            for target_dir in receptor_root.iterdir():
                if target_dir.is_dir():
                    candidate_paths.extend(
                        [
                            target_dir / pdb_id / "receptor_clean.pdb",
                            target_dir / pdb_id.upper() / "receptor_clean.pdb",
                            target_dir / pdb_id / "receptor.pdbqt",
                        ]
                    )
        for candidate in candidate_paths:
            if candidate.exists():
                return candidate, ""
    return None, "receptor_not_found"


def parse_structure_atoms(path: str | Path) -> list[AtomRecord]:
    structure_path = Path(path)
    if not structure_path.exists():
        return []
    atoms: list[AtomRecord] = []
    for line in structure_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            serial = int(line[6:11].strip() or len(atoms) + 1)
            name = line[12:16].strip()
            residue_name = line[17:20].strip()
            chain_id = line[21:22].strip()
            residue_id = line[22:26].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except Exception:
            continue
        element = line[76:78].strip() if len(line) >= 78 else ""
        if not element:
            tokens = line.split()
            element = tokens[-1] if tokens else ""
        if not element or len(element) > 2 or not element[0].isalpha():
            element = "".join(ch for ch in name if ch.isalpha())[:2]
        element = element.upper()
        atoms.append(
            AtomRecord(
                serial=serial,
                name=name,
                residue_name=residue_name,
                chain_id=chain_id,
                residue_id=residue_id,
                element=element,
                x=x,
                y=y,
                z=z,
            )
        )
    return atoms


def heavy_atoms(atoms: list[AtomRecord]) -> list[AtomRecord]:
    return [atom for atom in atoms if atom.element.upper() != "H"]
