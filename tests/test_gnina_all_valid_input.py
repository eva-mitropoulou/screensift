from pathlib import Path

import pandas as pd


from screensift.rescoring.build_gnina_all_valid_input import build_gnina_all_valid_input


def test_gnina_all_valid_input_marks_valid_and_invalid_rows(tmp_path: Path) -> None:
    receptor_root = tmp_path / "receptors" / "MAPK1"
    (receptor_root / "aaaa").mkdir(parents=True)
    (receptor_root / "aaaa" / "receptor_clean.pdb").write_text("ATOM\n", encoding="utf-8")

    pose_good = tmp_path / "pose_good.pdbqt"
    pose_positive = tmp_path / "pose_positive.pdbqt"
    pose_low = tmp_path / "pose_low.pdbqt"
    for pose in [pose_good, pose_positive, pose_low]:
        pose.write_text("REMARK VINA RESULT\n", encoding="utf-8")

    unidock = pd.DataFrame(
        [
            {
                "ligand_id": "valid",
                "activity_label": "active",
                "best_score": -8.0,
                "best_pdb_id": "AAAA",
                "best_receptor_pdbqt": "rec.pdbqt",
                "best_output_pose_file": str(pose_good),
            },
            {
                "ligand_id": "missing_receptor",
                "activity_label": "inactive",
                "best_score": -7.0,
                "best_pdb_id": "BBBB",
                "best_receptor_pdbqt": "rec.pdbqt",
                "best_output_pose_file": str(pose_good),
            },
            {
                "ligand_id": "missing_pose",
                "activity_label": "inactive",
                "best_score": -6.0,
                "best_pdb_id": "AAAA",
                "best_receptor_pdbqt": "rec.pdbqt",
                "best_output_pose_file": str(tmp_path / "missing.pdbqt"),
            },
            {
                "ligand_id": "positive_score",
                "activity_label": "inactive",
                "best_score": 0.1,
                "best_pdb_id": "AAAA",
                "best_receptor_pdbqt": "rec.pdbqt",
                "best_output_pose_file": str(pose_positive),
            },
            {
                "ligand_id": "too_low",
                "activity_label": "inactive",
                "best_score": -31.0,
                "best_pdb_id": "AAAA",
                "best_receptor_pdbqt": "rec.pdbqt",
                "best_output_pose_file": str(pose_low),
            },
        ]
    )
    unidock_path = tmp_path / "unidock.csv"
    out_path = tmp_path / "all_valid_input.csv"
    report_path = tmp_path / "report.md"
    unidock.to_csv(unidock_path, index=False)

    result = build_gnina_all_valid_input(
        unidock_path,
        out_path,
        report_path,
        receptor_root=receptor_root,
        gnina_output_root=tmp_path / "phase1_all_valid",
    )

    assert out_path.exists()
    assert report_path.exists()
    valid_row = result[result["ligand_id"] == "valid"].iloc[0]
    assert bool(valid_row["valid_for_gnina"])
    assert "phase1_all_valid" in valid_row["gnina_output_file"]

    missing_receptor = result[result["ligand_id"] == "missing_receptor"].iloc[0]
    assert not bool(missing_receptor["valid_for_gnina"])
    assert "missing_receptor_clean_pdb" in missing_receptor["invalid_reason"]

    missing_pose = result[result["ligand_id"] == "missing_pose"].iloc[0]
    assert not bool(missing_pose["valid_for_gnina"])
    assert "missing_ligand_pose" in missing_pose["invalid_reason"]

    for ligand_id in ["positive_score", "too_low"]:
        row = result[result["ligand_id"] == ligand_id].iloc[0]
        assert not bool(row["valid_for_gnina"])
        assert "score_outside_valid_range" in row["invalid_reason"]
