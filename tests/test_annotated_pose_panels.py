from pathlib import Path

import pandas as pd


from screensift.validation.generate_annotated_pose_panels import CONTACT_REVIEW_COLUMNS, build_panel_rows, select_ligands  # noqa: E402


def _config(tmp_path: Path) -> dict:
    return {
        "annotated_pose_panel": {
            "include_tiers": ["A_analog_seed"],
            "include_controls": {
                "false_positive_cases": 1,
                "false_negative_cases": 1,
                "score_anomaly_cases": 1,
            },
            "output_dir": str(tmp_path / "panels"),
            "pml_dir": str(tmp_path / "panels" / "annotated_pml"),
            "png_dir": str(tmp_path / "panels" / "annotated_png"),
            "image": {
                "width": 400,
                "height": 300,
                "dpi": 72,
                "ray": False,
                "background": "white",
                "opaque_background": True,
            },
            "cutoffs": {
                "pocket_residue_cutoff_angstrom": 4.0,
                "polar_contact_cutoff_angstrom": 3.5,
            },
            "views": {
                "full_receptor": {"zoom_target": "receptor", "zoom_buffer": 1.5, "show_surface": False, "show_labels": False},
                "clean_pocket": {
                    "zoom_target": "ligand",
                    "zoom_buffer": 10,
                    "show_surface": True,
                    "surface_transparency": 0.82,
                },
                "annotated_contacts": {
                    "zoom_target": "ligand",
                    "zoom_buffer": 8,
                    "show_surface": False,
                    "show_labels": True,
                    "show_polar_contacts": True,
                    "show_distance_labels": True,
                },
                "residue_type_view": {
                    "zoom_target": "ligand",
                    "zoom_buffer": 8,
                    "show_surface": False,
                    "color_residues_by_type": True,
                    "show_labels": True,
                },
            },
            "styling": {
                "receptor_color": "gray80",
                "ligand_color": "green",
                "pocket_color": "yellow",
                "polar_contact_color": "cyan",
                "stick_radius_ligand": 0.22,
                "stick_radius_pocket": 0.14,
                "dash_radius": 0.08,
                "label_size": 16,
                "label_color": "black",
            },
            "residue_type_colors": {
                "acidic": "red",
                "basic": "blue",
                "polar": "cyan",
                "hydrophobic": "orange",
                "aromatic": "magenta",
                "glycine_proline": "yellow",
            },
            "residue_groups": {
                "acidic": ["ASP", "GLU"],
                "basic": ["LYS", "ARG", "HIS"],
                "polar": ["SER", "THR", "ASN", "GLN", "CYS", "TYR"],
                "hydrophobic": ["ALA", "VAL", "LEU", "ILE", "MET"],
                "aromatic": ["PHE", "TYR", "TRP", "HIS"],
                "glycine_proline": ["GLY", "PRO"],
            },
            "pymol": {
                "executable_candidates": ["definitely_missing_pymol_for_test"],
                "run_headless_if_available": False,
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
        "manual_priority": 1,
        "ecfp4_active_similarity": 0.2,
        "n_total_interactions": 20,
    }


def test_select_ligands_includes_tier_a_and_controls() -> None:
    triage = pd.DataFrame(
        [
            _triage_row("a1", "A_analog_seed", "active", "novel_sav_tanimoto_lt_0_30", "credible_structure_added_case"),
            _triage_row("fp1", "D_failure_analysis", "inactive", "consensus_inactive_false_positive", "false_positive_failure_case"),
            _triage_row("fn1", "D_failure_analysis", "active", "active_false_negative", "false_negative_failure_case"),
            _triage_row("anom1", "D_failure_analysis", "inactive", "score_anomaly_top_hit", "inspect_manually", "extreme_unidock_negative"),
        ]
    )

    selected = select_ligands(triage, _config(Path(".")))

    assert set(selected["ligand_id"]) == {"a1", "fp1", "fn1", "anom1"}
    assert set(selected["_selection_category"]) == {
        "tier_a_seed",
        "false_positive_control",
        "false_negative_control",
        "score_anomaly_control",
    }


def test_annotated_pml_scripts_and_review_table_columns(tmp_path: Path, monkeypatch) -> None:
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
    interactions = pd.DataFrame(
        [
            {
                "ligand_id": "a1",
                "n_hbond_interactions": 1,
                "n_hydrophobic_interactions": 2,
                "n_pi_interactions": 0,
                "n_total_interactions": 3,
            }
        ]
    )

    rows = build_panel_rows(triage, locations, pd.DataFrame(), interactions, pd.DataFrame(), _config(tmp_path))

    assert len(rows) == 1
    for col in CONTACT_REVIEW_COLUMNS:
        assert col in rows.columns
    assert bool(rows.loc[0, "pml_generated"]) is True
    assert rows.loc[0, "png_status"] == "skipped_pymol_unavailable"
    assert Path(rows.loc[0, "full_receptor_pml"]).exists()
    assert Path(rows.loc[0, "clean_pocket_pml"]).exists()
    assert Path(rows.loc[0, "annotated_contacts_pml"]).exists()
    assert Path(rows.loc[0, "residue_type_view_pml"]).exists()

    full_text = Path(rows.loc[0, "full_receptor_pml"]).read_text(encoding="utf-8")
    annotated_text = Path(rows.loc[0, "annotated_contacts_pml"]).read_text(encoding="utf-8")
    residue_text = Path(rows.loc[0, "residue_type_view_pml"]).read_text(encoding="utf-8")

    assert "zoom receptor, 1.5" in full_text
    assert "distance polar_contacts_a1" in annotated_text
    assert "show labels, polar_contacts_a1" in annotated_text
    assert "select acidic_a1, pocket_a1 and resn ASP+GLU" in residue_text
    assert "select aromatic_a1, pocket_a1 and resn PHE+TYR+TRP+HIS" in residue_text
