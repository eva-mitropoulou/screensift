from pathlib import Path

import pandas as pd


from screensift.docking.parse_unidock_scores import parse_unidock_scores
from screensift.docking.run_unidock import parse_best_score_from_pose


def test_parse_best_score_from_pose_result_line(tmp_path: Path) -> None:
    pose = tmp_path / "pose.pdbqt"
    pose.write_text("MODEL 1\nREMARK VINA RESULT: -8.7 0.000 0.000\nENDMDL\n", encoding="utf-8")

    assert parse_best_score_from_pose(pose) == -8.7


def test_parse_unidock_scores_fills_missing_score(tmp_path: Path) -> None:
    pose = tmp_path / "pose.pdbqt"
    pose.write_text("REMARK VINA RESULT: -7.1 0.000 0.000\n", encoding="utf-8")
    raw = pd.DataFrame(
        [
            {
                "docking_id": "AAAA_lig_1",
                "ligand_pdbqt": str(tmp_path / "lig_1.pdbqt"),
                "receptor_pdbqt": str(tmp_path / "rec.pdbqt"),
                "pdb_id": "AAAA",
                "output_pose_file": str(pose),
                "best_score": None,
                "status": "complete",
                "error_message": "",
            }
        ]
    )
    raw_path = tmp_path / "raw.csv"
    out_path = tmp_path / "scores.csv"
    raw.to_csv(raw_path, index=False)

    parsed = parse_unidock_scores(raw_path, out_path)

    assert out_path.exists()
    assert parsed.loc[0, "ligand_id"] == "lig_1"
    assert parsed.loc[0, "best_score"] == -7.1
