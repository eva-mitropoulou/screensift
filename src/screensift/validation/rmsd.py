from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D


class RmsdResult(NamedTuple):
    rmsd_angstrom: float
    n_atoms: int
    status: str
    failure_reason: str


# AutoDock/Vina (PDBQT) atom types -> chemical element. PDBQT stores docking
# atom types (e.g. "A" aromatic carbon, "OA" H-bond-acceptor oxygen, "HD" polar
# hydrogen) in columns 77-78 instead of the plain element, so redocked PDBQT
# poses must be normalized before their elements can be compared with the
# crystal-frame native ligand (usually a .pdb/.sdf).
_AUTODOCK_ELEMENT = {
    "A": "C", "C": "C", "CG": "C", "G": "C",
    "N": "N", "NA": "N", "NS": "N",
    "O": "O", "OA": "O", "OS": "O",
    "S": "S", "SA": "S",
    "H": "H", "HD": "H", "HS": "H",
    "P": "P", "F": "F", "CL": "CL", "BR": "BR", "I": "I",
    "MG": "MG", "MN": "MN", "ZN": "ZN", "CA": "CA", "FE": "FE",
}

_TWO_LETTER_ELEMENTS = {"CL", "BR", "SI", "MG", "MN", "ZN", "CA", "FE", "NA", "SE"}


def _normalize_element(raw: str, atom_name: str) -> str:
    """Map a raw element/atom-type field to a chemical element symbol."""
    token = raw.strip().upper()
    if token in _AUTODOCK_ELEMENT:
        return _AUTODOCK_ELEMENT[token]
    if token in _TWO_LETTER_ELEMENTS:
        return token
    if len(token) == 1 and token.isalpha():
        return token
    # Fall back to parsing the atom name (e.g. "CL1" -> "CL", "C12" -> "C").
    letters = "".join(char for char in atom_name if char.isalpha()).upper()
    if letters[:2] in _TWO_LETTER_ELEMENTS:
        return letters[:2]
    return letters[:1] if letters else (token[:1] if token else "")


def _element_from_pdb_line(line: str) -> str:
    element = line[76:78] if len(line) >= 78 else ""
    atom_name = line[12:16].strip()
    return _normalize_element(element, atom_name)


def _first_model_heavy_atoms(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Load heavy-atom coordinates and elements from the FIRST model only.

    Docking engines write multi-model pose files (Uni-Dock emits ``num_modes``
    poses per ligand as consecutive ``MODEL``/``ENDMDL`` blocks). RMSD must be
    computed against a single pose, so everything after the first ``ENDMDL`` is
    ignored; otherwise the atom count is a multiple of the true count and every
    comparison fails with a spurious mismatch.
    """
    pose_path = Path(path)
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose file not found: {pose_path}")

    if pose_path.suffix.lower() in {".sdf", ".mol"}:
        mol = Chem.MolFromMolFile(str(pose_path), removeHs=False, sanitize=False)
        if mol is None:
            supplier = Chem.SDMolSupplier(str(pose_path), removeHs=False, sanitize=False)
            mol = next((candidate for candidate in supplier if candidate is not None), None)
        if mol is None or mol.GetNumConformers() == 0:
            raise ValueError(f"No molecule coordinates found in {pose_path}")
        conformer = mol.GetConformer()
        coords: list[list[float]] = []
        elements: list[str] = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue
            position = conformer.GetAtomPosition(atom.GetIdx())
            coords.append([position.x, position.y, position.z])
            elements.append(atom.GetSymbol().upper())
        if not coords:
            raise ValueError(f"No heavy-atom coordinates found in {pose_path}")
        return np.asarray(coords, dtype=float), elements

    coords = []
    elements = []
    seen_atoms = False
    for line in pose_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            seen_atoms = True
            if _element_from_pdb_line(line) == "H":
                continue
            try:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
            elements.append(_element_from_pdb_line(line))
        elif line.startswith(("ENDMDL", "MODEL")) and seen_atoms:
            # Stop at the end of the first model. Break on a fresh MODEL record
            # too, so pose files that delimit models with MODEL only (no ENDMDL
            # between them) are not read as all models concatenated.
            break
    if not coords:
        raise ValueError(f"No heavy-atom coordinates found in {pose_path}")
    return np.asarray(coords, dtype=float), elements


def load_heavy_atom_coords(path: str | Path) -> np.ndarray:
    """Return first-model heavy-atom coordinates (kept for backward use)."""
    coords, _ = _first_model_heavy_atoms(path)
    return coords


def symmetry_corrected_rmsd(
    reference: np.ndarray,
    mobile: np.ndarray,
    reference_elements: list[str] | None = None,
    mobile_elements: list[str] | None = None,
) -> float:
    """RMSD in the shared coordinate frame, WITHOUT superposition.

    For native-ligand redocking validation the reference and the docked pose are
    already in the same receptor frame, so the poses must NOT be re-superposed:
    optimal Kabsch alignment would remove exactly the displacement the 2 A gate
    is meant to detect, letting a pose in the wrong sub-pocket pass.

    Atom correspondence is resolved by an optimal (Hungarian) assignment on the
    in-place squared-distance matrix rather than by file order, which makes the
    metric robust to differing atom ordering (SDF vs PDBQT) and tolerant of
    molecular symmetry. When both element lists are available and describe the
    same multiset, the assignment is constrained to match like elements.
    """
    if reference.shape != mobile.shape:
        raise ValueError(f"Coordinate arrays must have the same shape: {reference.shape} != {mobile.shape}")
    n_atoms = len(reference)
    if n_atoms == 0:
        raise ValueError("Cannot compute RMSD over zero atoms.")

    diff = reference[:, None, :] - mobile[None, :, :]
    sq_dist = (diff * diff).sum(axis=2)

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        # Fall back to file-order pairing; still no superposition.
        return float(np.sqrt(np.einsum("ii->i", sq_dist).mean()))

    cost = sq_dist
    if (
        reference_elements is not None
        and mobile_elements is not None
        and Counter(reference_elements) == Counter(mobile_elements)
    ):
        ref_el = np.asarray(reference_elements)
        mob_el = np.asarray(mobile_elements)
        same_element = ref_el[:, None] == mob_el[None, :]
        penalty = float(sq_dist.max()) * n_atoms + 1.0
        cost = np.where(same_element, sq_dist, penalty)

    row_idx, col_idx = linear_sum_assignment(cost)
    return float(np.sqrt(sq_dist[row_idx, col_idx].mean()))


def pose_rmsd(reference_pose: str | Path, mobile_pose: str | Path) -> RmsdResult:
    try:
        reference, ref_elements = _first_model_heavy_atoms(reference_pose)
        mobile, mob_elements = _first_model_heavy_atoms(mobile_pose)
        if len(reference) != len(mobile):
            return RmsdResult(
                float("nan"),
                min(len(reference), len(mobile)),
                "failed",
                f"atom_count_mismatch:{len(reference)}!={len(mobile)}",
            )
        value = symmetry_corrected_rmsd(reference, mobile, ref_elements, mob_elements)
        return RmsdResult(value, len(reference), "complete", "")
    except Exception as exc:
        return RmsdResult(float("nan"), 0, "failed", str(exc))


def _translate_sdf(source: Path, destination: Path, delta: tuple[float, float, float]) -> int:
    dx, dy, dz = delta
    supplier = Chem.SDMolSupplier(str(source), removeHs=False, sanitize=False)
    mol = next((candidate for candidate in supplier if candidate is not None), None)
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"Could not read a molecule with coordinates from {source}")
    conformer = mol.GetConformer()
    moved = 0
    for atom in mol.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        conformer.SetAtomPosition(atom.GetIdx(), Point3D(position.x + dx, position.y + dy, position.z + dz))
        moved += 1
    writer = Chem.SDWriter(str(destination))
    try:
        writer.write(mol)
    finally:
        writer.close()
    return moved


def translate_pose(input_path: str | Path, output_path: str | Path, delta: tuple[float, float, float]) -> Path:
    """Rigidly translate a pose by ``delta``, format-aware.

    Used to build a displaced-decoy negative control. PDB/PDBQT poses are
    shifted by rewriting coordinate columns; SDF/MOL poses are translated via
    RDKit (a plain column rewrite would silently copy an SDF unchanged, making
    the decoy identical to the native). Raises if no atom was moved so a broken
    negative control can never pass silently.
    """
    source = Path(input_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.suffix.lower() in {".sdf", ".mol"}:
        moved = _translate_sdf(source, destination, delta)
        if moved == 0:
            raise ValueError(f"Decoy translation moved no atoms for {source}")
        return destination

    dx, dy, dz = delta
    lines: list[str] = []
    moved = 0
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                x = float(line[30:38]) + dx
                y = float(line[38:46]) + dy
                z = float(line[46:54]) + dz
                line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
                moved += 1
            except ValueError:
                pass
        lines.append(line)
    if moved == 0:
        raise ValueError(f"Decoy translation moved no atoms for {source}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def pose_centroid_distance(pose_a: str | Path, pose_b: str | Path) -> RmsdResult:
    try:
        coords_a, _ = _first_model_heavy_atoms(pose_a)
        coords_b, _ = _first_model_heavy_atoms(pose_b)
        distance = float(np.linalg.norm(coords_a.mean(axis=0) - coords_b.mean(axis=0)))
        return RmsdResult(distance, min(len(coords_a), len(coords_b)), "complete", "")
    except Exception as exc:
        return RmsdResult(float("nan"), 0, "failed", str(exc))
