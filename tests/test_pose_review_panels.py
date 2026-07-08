from pathlib import Path

import pandas as pd


from screensift.validation.generate_pose_review_panels import build_panel_rows, select_panel_ligands  # noqa: E402


def _config(tmp_path: Path) -> dict:
    return {
        "pose_panel": {
            "include_tiers": ["A_analog_seed"],
            "include_controls": {
                "false_positive_cases": 1,
                "false_negative_cases": 1,
                "score_anomaly_cases": 1,
            },
            "output_dir": str(tmp_path / "panels"),
            "pml_dir": str(tmp_path / "panels" / "pml"),
            "png_dir": str(tmp_path / "panels" / "png"),
            "image": {"width": 400, "height": 300, "dpi": 72, "ray": False, "background": "white", "opaque_background": True},
            "views": {
                "full_receptor": {
                    "show_surface": False,
                    "pocket_cutoff_angstrom": 4.0,
                },
                "pocket_overview": {
                    "ligand_zoom_buffer": 12,
                    "receptor_surface_transparency": 0.80,
                    "pocket_cutoff_angstrom": 4.0,
                    "show_surface": True,
                },
                "interactions": {
                    "ligand_zoom_buffer": 8,
                    "pocket_cutoff_angstrom": 4.0,
                    "polar_contact_cutoff_angstrom": 3.5,
                    "show_surface": False,
                    "label_pocket_residues": True,
                    "hide_pocket_cartoon": True,
                },
            },
            "styling": {
                "receptor_color": "gray80",
                "ligand_color": "green",
                "pocket_color": "yellow",
                "contact_color": "black",
                "label_color": "black",
                "label_size": 16,
                "cartoon_transparency": 0.0,
                "stick_radius_ligand": 0.20,
                "stick_radius_pocket": 0.14,
                "dash_radius": 0.08,
            },
            "pymol": {
                "executable_candidates": ["definitely_missing_pymol_for_test"],
                "run_headless_if_available": True,
                "generate_pml_even_if_pymol_missing": True,
            },
        }
    }


def _triage_row(ligand_id: str, tier: str, activity: str, categories: str, action: str, anomaly: str = "") -> dict:
    return {
        "ligand_id": ligand_id,
        "triage_tier": tier,
        "activity_label": activity,
        "inspection_categories": categories,
        "recommended_action": action,
        "anomaly_flags": anomaly,
        "ecfp4_active_similarity": 0.2,
        "n_total_interactions": 20,
        "unidock_best_score": -8.0,
        "CNNscore": 0.7,
        "CNNaffinity": 6.0,
        "gnina_affinity": -8.0,
    }


def test_selection_includes_tier_a_and_controls() -> None:
    triage = pd.DataFrame(
        [
            _triage_row("a1", "A_analog_seed", "active", "novel_sav_tanimoto_lt_0_30", "credible_structure_added_case"),
            _triage_row("fp1", "D_failure_analysis", "inactive", "consensus_inactive_false_positive", "false_positive_failure_case"),
            _triage_row("fn1", "D_failure_analysis", "active", "active_false_negative", "false_negative_failure_case"),
            _triage_row("anom1", "D_failure_analysis", "inactive", "score_anomaly_top_hit", "inspect_manually", "extreme_unidock_negative"),
        ]
    )

    selected = select_panel_ligands(triage, _config(Path(".")))

    assert set(selected["ligand_id"]) == {"a1", "fp1", "fn1", "anom1"}
    assert set(selected["_selection_category"]) == {
        "tier_a_seed",
        "false_positive_control",
        "false_negative_control",
        "score_anomaly_control",
    }


def test_pml_scripts_are_generated_without_pymol(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    pose_dir = tmp_path / "results" / "poses" / "gnina" / "MAPK1" / "phase1_all_valid_inputs" / "4qta"
    receptor_dir = tmp_path / "data" / "processed" / "receptors" / "MAPK1" / "4qta"
    pose_dir.mkdir(parents=True)
    receptor_dir.mkdir(parents=True)
    pose = pose_dir / "a1_gnina_input.pdb"
    receptor = receptor_dir / "receptor_clean.pdb"
    pose.write_text("HETATM    1  C1  LIG A   1       1.000   2.000   3.000  1.00  0.00           C\n", encoding="utf-8")
    receptor.write_text("ATOM      1  CA  ALA A   1       1.000   2.000   4.000  1.00  0.00           C\n", encoding="utf-8")
    triage = pd.DataFrame(
        [_triage_row("a1", "A_analog_seed", "active", "novel_sav_tanimoto_lt_0_30", "credible_structure_added_case")]
    )
    locations = pd.DataFrame([{"ligand_id": "a1", "selected_pose_file": str(pose), "pdb_id": "4qta"}])

    config = _config(tmp_path)
    config["pose_panel"]["pymol"]["run_headless_if_available"] = False
    rows = build_panel_rows(triage, locations, pd.DataFrame(), pd.DataFrame(), config)

    assert len(rows) == 1
    assert bool(rows.loc[0, "pml_generated"]) is True
    assert rows.loc[0, "png_status"] == "skipped_pymol_unavailable"
    assert Path(rows.loc[0, "full_receptor_pml"]).exists()
    assert Path(rows.loc[0, "pocket_overview_pml"]).exists()
    assert Path(rows.loc[0, "interactions_pml"]).exists()
    assert "zoom receptor, 1.5" in Path(rows.loc[0, "full_receptor_pml"]).read_text(encoding="utf-8")
    assert "set ray_opaque_background, on" in Path(rows.loc[0, "pocket_overview_pml"]).read_text(encoding="utf-8")
    interaction_text = Path(rows.loc[0, "interactions_pml"]).read_text(encoding="utf-8")
    assert "hide cartoon, pocket_a1" in interaction_text
    assert 'label (name CA and pocket_a1), "%s%s" % (resn, resi)' in interaction_text
