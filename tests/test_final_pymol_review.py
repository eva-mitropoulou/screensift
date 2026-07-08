from pathlib import Path

import pandas as pd


from screensift.validation.generate_final_pymol_review import TIER_A_LIGAND_IDS, VERDICT_COLUMNS, generate_final_review  # noqa: E402


def test_final_pymol_review_archives_old_outputs_and_writes_clean_scripts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for old_dir in ["pml", "png", "annotated_pml", "annotated_png"]:
        path = tmp_path / "results" / "figures" / "pose_review_panels" / old_dir
        path.mkdir(parents=True)
        (path / "old_file.txt").write_text("old", encoding="utf-8")

    receptor_dir = tmp_path / "data" / "processed" / "receptors" / "MAPK1" / "4qta"
    pose_dir = tmp_path / "results" / "poses" / "gnina" / "MAPK1" / "phase1_all_valid_inputs" / "4qta"
    receptor_dir.mkdir(parents=True)
    pose_dir.mkdir(parents=True)
    receptor = receptor_dir / "receptor_clean.pdb"
    pose = pose_dir / "26747800_gnina_input.pdb"
    receptor.write_text("ATOM      1  CA  ALA A   1       1.000   2.000   4.000  1.00  0.00           C\n", encoding="utf-8")
    pose.write_text("HETATM    1  C1  LIG A   1       1.000   2.000   3.000  1.00  0.00           C\n", encoding="utf-8")

    tables = tmp_path / "results" / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ligand_id": ligand_id,
                "triage_tier": "A_analog_seed",
                "inspection_categories": "novel_sav_tanimoto_lt_0_30",
            }
            for ligand_id in TIER_A_LIGAND_IDS
        ]
    ).to_csv(tables / "mapk1_phase1_candidate_triage.csv", index=False)
    pd.DataFrame(
        [
            {
                "ligand_id": ligand_id,
                "triage_tier": "A_analog_seed",
            }
            for ligand_id in TIER_A_LIGAND_IDS
        ]
    ).to_csv(tables / "mapk1_phase1_step10_seed_ligands.csv", index=False)
    pd.DataFrame([{"ligand_id": "26747800", "selected_pose_file": str(pose), "pdb_id": "4qta"}]).to_csv(
        tables / "mapk1_phase1_selected_pose_locations.csv", index=False
    )
    pd.DataFrame([{"pdb_id": "4qta", "receptor_clean_pdb": str(receptor)}]).to_csv(
        tables / "mapk1_prepared_receptors.csv", index=False
    )

    rows, archived = generate_final_review(run_pymol=False)

    assert len(rows) == 5
    assert len(archived) == 4
    assert (tmp_path / "results" / "figures" / "pose_review_panels" / "deprecated_old_panels" / "pml").exists()

    full_pml = tmp_path / "results" / "figures" / "final_pymol_review" / "pml" / "26747800_01_full_receptor.pml"
    contact_pml = tmp_path / "results" / "figures" / "final_pymol_review" / "pml" / "26747800_03_contact_only.pml"
    rotated_pml = tmp_path / "results" / "figures" / "final_pymol_review" / "pml" / "26747800_04_contact_y90.pml"
    assert full_pml.exists()
    assert contact_pml.exists()
    assert rotated_pml.exists()
    assert "zoom receptor, 1.5" in full_pml.read_text(encoding="utf-8")
    contact_text = contact_pml.read_text(encoding="utf-8")
    assert "receptor within 3.5" in contact_text
    assert "distance polar_contacts_26747800" in contact_text
    assert 'label (name CA and contact_res_26747800)' in contact_text
    assert "turn y, 90" in rotated_pml.read_text(encoding="utf-8")

    verdict = pd.read_csv(tables / "mapk1_phase1_final_manual_pose_verdict.csv")
    assert list(verdict.columns) == VERDICT_COLUMNS
    assert verdict["manual_pose_verdict"].eq("unclear").all()
