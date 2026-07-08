from pathlib import Path

import pandas as pd


from screensift.validation.generate_pose_panel_scripts import generate_scripts  # noqa: E402
from screensift.validation.write_manual_pose_review_guide import write_guide  # noqa: E402


def test_pymol_script_is_generated_for_mock_receptor_and_pose(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    pose_dir = tmp_path / "results" / "poses" / "unidock" / "MAPK1" / "phase1" / "4qta"
    receptor_dir = tmp_path / "data" / "processed" / "receptors" / "MAPK1" / "4qta"
    out_dir = tmp_path / "scripts"
    pose_dir.mkdir(parents=True)
    receptor_dir.mkdir(parents=True)
    pose = pose_dir / "0001_12345_out.pdbqt"
    receptor = receptor_dir / "receptor_clean.pdb"
    pose.write_text("HETATM    1  C1  LIG A   1       1.000   2.000   3.000  1.00  0.00           C\n", encoding="utf-8")
    receptor.write_text("ATOM      1  CA  ALA A   1       1.000   2.000   4.000  1.00  0.00           C\n", encoding="utf-8")
    summary = pd.DataFrame([{"ligand_id": "12345", "inspection_categories": "novel_sav_tanimoto_lt_0_30"}])
    locations = pd.DataFrame([{"ligand_id": "12345", "pose_found": True, "selected_pose_file": str(pose)}])

    rows = generate_scripts(summary, locations, pd.DataFrame(), out_dir)

    assert bool(rows.loc[0, "script_generated"]) is True
    assert Path(rows.loc[0, "script_file"]).exists()


def test_manual_review_guide_is_written(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {
                "ligand_id": "12345",
                "activity_label": "active",
                "inspection_categories": "novel_sav_tanimoto_lt_0_30",
                "manual_priority": 1,
                "n_total_interactions": 3,
                "recommended_action": "inspect_manually",
                "pose_found": True,
                "anomaly_flags": "",
            }
        ]
    )
    flags = pd.DataFrame([{"ligand_id": "12345", "plausibility_flags": ""}])
    out = tmp_path / "guide.md"

    write_guide(summary, flags, out)

    assert out.exists()
    assert "Pose inspection is used to prioritize retrospective cases" in out.read_text(encoding="utf-8")
