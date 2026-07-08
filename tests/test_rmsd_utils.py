from pathlib import Path

import numpy as np

from screensift.validation.rmsd import (
    pose_centroid_distance,
    pose_rmsd,
    symmetry_corrected_rmsd,
    translate_pose,
)


PDB = """\
HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C
HETATM    2  C2  LIG A   1       1.000   0.000   0.000  1.00  0.00           C
HETATM    3  O1  LIG A   1       0.000   1.000   0.000  1.00  0.00           O
END
"""


def test_pose_rmsd_is_zero_for_identical_pose(tmp_path: Path) -> None:
    ref = tmp_path / "ref.pdb"
    copy = tmp_path / "copy.pdb"
    ref.write_text(PDB, encoding="utf-8")
    copy.write_text(PDB, encoding="utf-8")

    result = pose_rmsd(ref, copy)

    assert result.status == "complete"
    assert result.rmsd_angstrom < 1e-6


def test_pose_rmsd_reports_displacement_without_superposition(tmp_path: Path) -> None:
    # A rigidly displaced pose must report the displacement magnitude, NOT ~0.
    # (Optimal superposition would erase it and let wrong-pocket poses pass the
    # redocking gate -- the bug this rewrite fixes.)
    ref = tmp_path / "ref.pdb"
    moved = tmp_path / "moved.pdb"
    ref.write_text(PDB, encoding="utf-8")
    translate_pose(ref, moved, (5.0, 2.0, -1.0))

    result = pose_rmsd(ref, moved)

    expected = float(np.sqrt(5.0**2 + 2.0**2 + 1.0**2))
    assert result.status == "complete"
    assert abs(result.rmsd_angstrom - expected) < 1e-3


def test_symmetry_corrected_rmsd_is_atom_order_independent() -> None:
    # Same three points supplied in a different order must give RMSD 0.
    ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    permuted = ref[[2, 0, 1]]
    value = symmetry_corrected_rmsd(ref, permuted, ["C", "C", "C"], ["C", "C", "C"])
    assert value < 1e-9


def test_pose_centroid_distance_reports_translation(tmp_path: Path) -> None:
    ref = tmp_path / "ref.pdb"
    moved = tmp_path / "moved.pdb"
    ref.write_text(PDB, encoding="utf-8")
    translate_pose(ref, moved, (3.0, 0.0, 0.0))

    result = pose_centroid_distance(ref, moved)

    assert result.status == "complete"
    assert abs(result.rmsd_angstrom - 3.0) < 1e-6


def test_pose_rmsd_uses_first_model_only(tmp_path: Path) -> None:
    # Two-model pose file: RMSD against the single-model reference must succeed
    # (compare first model), not fail with an atom-count mismatch.
    multi = tmp_path / "multi.pdb"
    multi.write_text(
        "MODEL        1\n" + PDB.replace("END\n", "ENDMDL\n")
        + "MODEL        2\n" + PDB.replace("END\n", "ENDMDL\n"),
        encoding="utf-8",
    )
    ref = tmp_path / "ref.pdb"
    ref.write_text(PDB, encoding="utf-8")

    result = pose_rmsd(ref, multi)

    assert result.status == "complete"
    assert result.n_atoms == 3
    assert result.rmsd_angstrom < 1e-6


def test_pose_rmsd_first_model_with_bare_model_delimiters(tmp_path: Path) -> None:
    # Models delimited by MODEL only (no ENDMDL between them) must still be read
    # as the first model, not concatenated into an inflated atom count.
    body = "".join(line + "\n" for line in PDB.splitlines() if line.startswith("HETATM"))
    multi = tmp_path / "multi_bare.pdb"
    multi.write_text("MODEL 1\n" + body + "MODEL 2\n" + body, encoding="utf-8")
    ref = tmp_path / "ref.pdb"
    ref.write_text(PDB, encoding="utf-8")

    result = pose_rmsd(ref, multi)

    assert result.status == "complete"
    assert result.n_atoms == 3
