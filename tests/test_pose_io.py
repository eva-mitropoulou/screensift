from pathlib import Path

import pandas as pd


from screensift.validation.pose_io import (  # noqa: E402
    extract_ligand_id_from_pose_path,
    find_receptor_for_pose,
    infer_pose_format,
    parse_structure_atoms,
    resolve_pose_path,
    safe_rdkit_mol_from_pose,
)


def test_missing_pose_file_does_not_crash(tmp_path: Path) -> None:
    result = safe_rdkit_mol_from_pose(tmp_path / "missing.sdf")

    assert result.success is False
    assert result.failure_reason == "missing_pose_file"


def test_mock_pose_location_is_loaded_and_parsed(tmp_path: Path) -> None:
    pose = tmp_path / "12345_pose.pdb"
    pose.write_text(
        "HETATM    1  C1  LIG A   1       1.000   2.000   3.000  1.00  0.00           C\nEND\n",
        encoding="utf-8",
    )

    assert infer_pose_format(pose) == "pdb"
    assert extract_ligand_id_from_pose_path(pose) == "12345"
    assert len(parse_structure_atoms(pose)) == 1
    assert safe_rdkit_mol_from_pose(pose).success is True


def test_ligand_id_recovered_from_indexed_pose_names() -> None:
    # Real pose files are "{split_index:07d}_{ligand_id}[_out].pdbqt" and the
    # docking_id variant additionally prefixes the PDB id. The ligand id must be
    # the field after the index, never the zero-padded index itself.
    assert extract_ligand_id_from_pose_path("0000032_842954.pdbqt") == "842954"
    assert extract_ligand_id_from_pose_path("0000032_842954_out.pdbqt") == "842954"
    assert extract_ligand_id_from_pose_path("4QTA_0000032_842954.pdbqt") == "842954"
    # Fallback ligand ids that themselves contain underscores survive intact.
    assert extract_ligand_id_from_pose_path("0000005_ligand_0000005.pdbqt") == "ligand_0000005"


def test_resolve_pose_and_receptor_from_mock_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    pose_dir = tmp_path / "results" / "poses" / "unidock" / "MAPK1" / "phase1" / "4qta"
    receptor_dir = tmp_path / "data" / "processed" / "receptors" / "MAPK1" / "4qta"
    pose_dir.mkdir(parents=True)
    receptor_dir.mkdir(parents=True)
    pose = pose_dir / "0001_12345_out.pdbqt"
    receptor = receptor_dir / "receptor_clean.pdb"
    pose.write_text("HETATM    1  C1  LIG A   1       1.000   2.000   3.000  1.00  0.00           C\n", encoding="utf-8")
    receptor.write_text("ATOM      1  CA  ALA A   1       1.000   2.000   4.000  1.00  0.00           C\n", encoding="utf-8")

    resolved, note = resolve_pose_path(pd.Series({"ligand_id": "12345"}), {"12345": [pose]})
    receptor_path, receptor_note = find_receptor_for_pose(pd.Series({"selected_pose_file": str(resolved)}), pd.DataFrame())

    assert resolved == pose
    assert note == "resolved_from_pose_tree"
    assert receptor_path == Path("data/processed/receptors/MAPK1/4qta/receptor_clean.pdb")
    assert receptor_note == ""
