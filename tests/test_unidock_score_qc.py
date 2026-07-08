from pathlib import Path

import pandas as pd


from screensift.docking.audit_unidock_scores import audit_unidock_scores


def test_unidock_score_qc_flags_and_validity(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    pose_dir = tmp_path / "poses"
    pose_dir.mkdir()
    existing_poses = {}
    for stem in ["0000000_a", "0000001_b", "0000002_c", "0000003_d"]:
        pose = pose_dir / f"{stem}_out.pdbqt"
        pose.write_text("REMARK VINA RESULT: -7.0 0.000 0.000\n", encoding="utf-8")
        existing_poses[stem] = pose

    scores = pd.DataFrame(
        [
            {
                "docking_id": "AAAA_0000000_a",
                "ligand_pdbqt": "ligands/0000000_a.pdbqt",
                "receptor_pdbqt": "rec.pdbqt",
                "pdb_id": "AAAA",
                "output_pose_file": str(existing_poses["0000000_a"]),
                "best_score": -7.2,
                "status": "complete",
            },
            {
                "docking_id": "AAAA_0000001_b",
                "ligand_pdbqt": "ligands/0000001_b.pdbqt",
                "receptor_pdbqt": "rec.pdbqt",
                "pdb_id": "AAAA",
                "output_pose_file": str(existing_poses["0000001_b"]),
                "best_score": 12.0,
                "status": "complete",
            },
            {
                "docking_id": "AAAA_0000002_c",
                "ligand_pdbqt": "ligands/0000002_c.pdbqt",
                "receptor_pdbqt": "rec.pdbqt",
                "pdb_id": "AAAA",
                "output_pose_file": str(existing_poses["0000002_c"]),
                "best_score": None,
                "status": "complete",
            },
            {
                "docking_id": "AAAA_0000003_d",
                "ligand_pdbqt": "ligands/0000003_d.pdbqt",
                "receptor_pdbqt": "rec.pdbqt",
                "pdb_id": "AAAA",
                "output_pose_file": str(existing_poses["0000003_d"]),
                "best_score": -31.0,
                "status": "complete",
            },
            {
                "docking_id": "AAAA_0000004_e",
                "ligand_pdbqt": "ligands/0000004_e.pdbqt",
                "receptor_pdbqt": "rec.pdbqt",
                "pdb_id": "AAAA",
                "output_pose_file": str(pose_dir / "missing_out.pdbqt"),
                "best_score": -6.4,
                "status": "complete",
            },
        ]
    )
    splits = pd.DataFrame(
        [
            {"ligand_id": "a", "activity_label": "active"},
            {"ligand_id": "b", "activity_label": "inactive"},
            {"ligand_id": "c", "activity_label": "inactive"},
            {"ligand_id": "d", "activity_label": "inactive"},
            {"ligand_id": "e", "activity_label": "inactive"},
        ]
    )

    scores_path = tmp_path / "scores.csv"
    splits_path = tmp_path / "splits.csv"
    clean_path = tmp_path / "clean.csv"
    flagged_path = tmp_path / "flagged.csv"
    best_path = tmp_path / "best.csv"
    report_path = tmp_path / "report.md"
    scores.to_csv(scores_path, index=False)
    splits.to_csv(splits_path, index=False)

    summary = audit_unidock_scores(scores_path, splits_path, clean_path, flagged_path, best_path, report_path)

    clean = pd.read_csv(clean_path)
    flagged = pd.read_csv(flagged_path)
    best = pd.read_csv(best_path)

    assert summary["total_rows"] == 5
    assert int(clean["valid_for_ranking"].sum()) == 1
    assert flagged.shape[0] == 4
    assert best.shape[0] == 1
    assert best.loc[0, "ligand_id"] == "a"
    assert clean.loc[clean["ligand_id"] == "b", "score_positive"].item()
    assert clean.loc[clean["ligand_id"] == "c", "score_missing"].item()
    assert clean.loc[clean["ligand_id"] == "d", "score_extreme_low"].item()
    assert clean.loc[clean["ligand_id"] == "e", "pose_file_missing"].item()
    assert report_path.exists()
