from pathlib import Path

import pandas as pd


from screensift.docking.collect_docking_inputs import collect_docking_inputs


def test_collect_docking_inputs_limit_and_single_receptor(tmp_path: Path) -> None:
    ligand_dir = tmp_path / "ligands"
    ligand_dir.mkdir()
    for index in range(3):
        (ligand_dir / f"lig_{index}.pdbqt").write_text("REMARK ligand\n", encoding="utf-8")

    receptor_a = tmp_path / "rec_a.pdbqt"
    receptor_b = tmp_path / "rec_b.pdbqt"
    receptor_a.write_text("REMARK receptor\n", encoding="utf-8")
    receptor_b.write_text("REMARK receptor\n", encoding="utf-8")
    boxes = pd.DataFrame(
        [
            {
                "pdb_id": "AAAA",
                "center_x": 1.0,
                "center_y": 2.0,
                "center_z": 3.0,
                "size_x": 18.0,
                "size_y": 18.0,
                "size_z": 18.0,
                "receptor_pdbqt": str(receptor_a),
                "status": "complete",
            },
            {
                "pdb_id": "BBBB",
                "center_x": 1.0,
                "center_y": 2.0,
                "center_z": 3.0,
                "size_x": 18.0,
                "size_y": 18.0,
                "size_z": 18.0,
                "receptor_pdbqt": str(receptor_b),
                "status": "complete",
            },
        ]
    )
    boxes_path = tmp_path / "boxes.csv"
    boxes.to_csv(boxes_path, index=False)

    out_path = tmp_path / "inputs.csv"
    frame = collect_docking_inputs(
        ligand_dir,
        boxes_path,
        out_path,
        limit=2,
        single_receptor=True,
        output_root=tmp_path / "poses",
    )

    assert out_path.exists()
    assert frame.shape[0] == 2
    assert set(frame["pdb_id"]) == {"AAAA"}
    assert all(frame["output_dir"].str.contains("aaaa"))
