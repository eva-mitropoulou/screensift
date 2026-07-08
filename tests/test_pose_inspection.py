from pathlib import Path

import pandas as pd


from screensift.validation.analyze_pose_interactions import analyze_poses, fallback_interaction_counts  # noqa: E402
from screensift.validation.pose_plausibility_flags import build_plausibility_flags  # noqa: E402


def test_fallback_interaction_summary_handles_missing_atoms(tmp_path: Path) -> None:
    receptor = tmp_path / "receptor.pdb"
    pose = tmp_path / "pose.pdb"
    receptor.write_text("", encoding="utf-8")
    pose.write_text("", encoding="utf-8")

    result = fallback_interaction_counts(receptor, pose, {})

    assert result["success"] is False
    assert "receptor_atoms_not_parsed" in result["failure_reason"]


def test_analyze_pose_interactions_with_mock_pose(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    pose_dir = tmp_path / "results" / "poses" / "gnina" / "MAPK1" / "phase1_all_valid_inputs" / "4qta"
    receptor_dir = tmp_path / "data" / "processed" / "receptors" / "MAPK1" / "4qta"
    pose_dir.mkdir(parents=True)
    receptor_dir.mkdir(parents=True)
    pose = pose_dir / "12345_gnina_input.pdb"
    receptor = receptor_dir / "receptor_clean.pdb"
    pose.write_text("HETATM    1  O1  LIG A   1       0.000   0.000   0.000  1.00  0.00           O\n", encoding="utf-8")
    receptor.write_text("ATOM      1  N   ALA A   1       0.000   0.000   3.000  1.00  0.00           N\n", encoding="utf-8")
    inspection = pd.DataFrame(
        [{"ligand_id": "12345", "activity_label": "active", "inspection_categories": "low_similarity_active_tanimoto_lt_0_30"}]
    )
    locations = pd.DataFrame([{"ligand_id": "12345", "pose_found": True, "selected_pose_file": str(pose)}])

    interactions, summary = analyze_poses(inspection, locations, pd.DataFrame(), {"pose_inspection": {}, "interaction_checks": {}})

    assert len(interactions) == 1
    assert bool(interactions.loc[0, "interaction_analysis_success"]) is True
    assert interactions.loc[0, "n_hbond_interactions"] >= 1
    assert summary.loc[0, "recommended_action"] == "credible_structure_added_case"


def test_plausibility_flags_mark_missing_pose() -> None:
    summary = pd.DataFrame(
        [
            {
                "ligand_id": "missing",
                "pose_found": False,
                "interaction_analysis_success": False,
                "n_total_interactions": 0,
                "inspection_categories": "",
                "anomaly_flags": "",
                "recommended_action": "missing_pose_needs_transfer",
            }
        ]
    )

    flags = build_plausibility_flags(summary)

    assert "missing_pose" in flags.loc[0, "plausibility_flags"]
