from pathlib import Path

import pandas as pd


from screensift.validation.locate_selected_poses import locate_pose_files  # noqa: E402


def test_pose_locator_finds_mock_pose_by_ligand_id(tmp_path: Path) -> None:
    pose_dir = tmp_path / "results" / "poses" / "gnina"
    pose_dir.mkdir(parents=True)
    pose_path = pose_dir / "ligA_gnina_score.sdf"
    pose_path.write_text("mock pose\n", encoding="utf-8")
    inspection = pd.DataFrame({"ligand_id": ["ligA"]})

    located = locate_pose_files(inspection, [str(tmp_path / "results" / "poses")])

    assert bool(located.loc[0, "pose_found"]) is True
    assert located.loc[0, "selected_pose_file"] == str(pose_path)
    assert located.loc[0, "pose_source"] == "gnina"


def test_missing_pose_files_do_not_crash_locator(tmp_path: Path) -> None:
    inspection = pd.DataFrame({"ligand_id": ["missing_ligand"]})

    located = locate_pose_files(inspection, [str(tmp_path / "missing")])

    assert bool(located.loc[0, "pose_found"]) is False
    assert bool(located.loc[0, "transfer_needed"]) is True
