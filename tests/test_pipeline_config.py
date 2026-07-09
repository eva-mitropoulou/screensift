from pathlib import Path

import yaml

import pandas as pd

from screensift.pipeline import (
    docking_boxes_for_screening,
    output_paths,
    prepare_score_population,
    redocking_gate_enabled,
    run_pipeline,
    selected_stages,
    write_redocking_filtered_boxes,
)
from screensift.validation.run_native_redocking import box_for_attempt, redocking_attempts


def test_pipeline_dry_run_writes_manifest(tmp_path: Path) -> None:
    config = {
        "target": "TOY",
        "run_id": "smoke",
        "target_slug": "toy",
        "pipeline": {"stages": ["rank_candidates", "native_redocking"]},
        "paths": {"run_dir": str(tmp_path / "run"), "receptor_root": str(tmp_path / "receptors")},
        "dataset": {"schema": "example/mapk1/schema.yml"},
        "ranking": {
            "score_table": "example/mapk1/expected_outputs/mapk1_2receptor_score_population.csv",
            "structure_score_columns": {"unidock_best_score": "lower"},
            "n_candidates": 5,
        },
        "docking": {"boxes": "example/mapk1/mapk1_docking_boxes.csv"},
    }
    config_path = tmp_path / "pipeline.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    manifest = run_pipeline(config_path, dry_run=True)

    assert manifest["dry_run"] is True
    assert manifest["stages"]["rank_candidates"]["status"] == "planned"
    assert Path(manifest["outputs"]["manifest"]).exists()


def test_pipeline_ranking_only_workflow_selects_rank_stage(tmp_path: Path) -> None:
    config = {
        "target": "TOY",
        "run_id": "smoke",
        "target_slug": "toy",
        "workflow": {"mode": "ranking_only"},
        "paths": {"run_dir": str(tmp_path / "run")},
        "dataset": {"schema": "example/mapk1/schema.yml"},
        "ranking": {
            "score_table": "example/mapk1/expected_outputs/mapk1_2receptor_score_population.csv",
            "evidence_mode": "similarity",
            "n_candidates": 5,
        },
    }
    config_path = tmp_path / "pipeline.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    manifest = run_pipeline(config_path, dry_run=True)

    assert list(manifest["stages"]) == ["rank_candidates"]


def test_empty_score_table_uses_generated_gnina_scores(tmp_path: Path) -> None:
    config = {
        "target": "TOY",
        "run_id": "smoke",
        "target_slug": "toy",
        "paths": {"run_dir": str(tmp_path / "run")},
        "ranking": {"score_table": ""},
    }

    paths = output_paths(config)

    assert paths["score_population"] == paths["gnina_scores"]


def test_full_screening_redocking_runs_before_docking_inputs() -> None:
    stages = selected_stages(
        {
            "workflow": {"mode": "full_screening"},
            "candidate_pose_qc": {"run": False},
            "redocking": {"native": {"run": True}, "gnina_score_sanity": {"run": False}},
        }
    )

    assert stages.index("native_redocking") < stages.index("ligand_3d")
    assert stages.index("native_redocking") < stages.index("docking_inputs")
    assert stages.index("native_redocking") < stages.index("unidock_screen")


def test_redocking_filter_keeps_only_passing_receptors(tmp_path: Path) -> None:
    boxes = tmp_path / "boxes.csv"
    out = tmp_path / "passed_boxes.csv"
    pd.DataFrame(
        [
            {"pdb_id": "PASS", "status": "complete", "receptor_pdbqt": "pass.pdbqt"},
            {"pdb_id": "FAIL", "status": "complete", "receptor_pdbqt": "fail.pdbqt"},
        ]
    ).to_csv(boxes, index=False)
    redocking = pd.DataFrame(
        [
            {"pdb_id": "PASS", "redocking_success": True},
            {"pdb_id": "FAIL", "redocking_success": False},
        ]
    )

    filtered = write_redocking_filtered_boxes(boxes, redocking, out)

    assert filtered["pdb_id"].tolist() == ["PASS"]
    assert pd.read_csv(out)["pdb_id"].tolist() == ["PASS"]


def test_redocking_auto_tune_attempts_are_user_defined() -> None:
    attempts = redocking_attempts(
        [
            {"name": "baseline", "exhaustiveness": 8, "box_scale": 1.0},
            {"name": "wide", "exhaustiveness": 16, "box_padding_angstrom": 2.0},
        ],
        exhaustiveness=4,
        num_modes=3,
        energy_range=2,
        seed=7,
    )

    assert [attempt["name"] for attempt in attempts] == ["baseline", "wide"]
    assert [attempt["exhaustiveness"] for attempt in attempts] == [8, 16]
    assert attempts[0]["num_modes"] == 3
    assert attempts[1]["box_padding_angstrom"] == 2.0


def test_redocking_attempt_box_size_can_scale_or_pad() -> None:
    box = {"size_x": 10.0, "size_y": 20.0, "size_z": 30.0}

    scaled = box_for_attempt(box, {"box_scale": 1.2, "box_padding_angstrom": 1.0})
    explicit = box_for_attempt(box, {"size_x": 15.0, "box_scale": 2.0})

    assert scaled["size_x"] == 14.0
    assert scaled["size_y"] == 26.0
    assert scaled["size_z"] == 38.0
    assert explicit["size_x"] == 15.0
    assert explicit["size_y"] == 40.0


def test_prepare_score_population_enriches_gnina_scores(tmp_path: Path) -> None:
    config = {
        "target": "TOY",
        "run_id": "smoke",
        "target_slug": "toy",
        "paths": {"run_dir": str(tmp_path / "run")},
        "ranking": {"score_table": ""},
    }
    paths = output_paths(config)
    pd.DataFrame(
        [
            {
                "ligand_id": "lig1",
                "activity_label": "inactive",
                "pdb_id": "1ABC",
                "best_score_unidock": -9.0,
                "cnnscore": 0.8,
                "cnnaffinity": 7.1,
                "affinity": -8.2,
                "status": "complete",
            }
        ]
    ).to_csv(paths["gnina_scores"], index=False)
    pd.DataFrame(
        [
            {
                "ligand_id": "lig1",
                "canonical_smiles": "CCO",
                "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "best_output_pose_file": "pose.pdbqt",
            }
        ]
    ).to_csv(paths["unidock_best"], index=False)
    pd.DataFrame(
        [
            {
                "ligand_id": "lig1",
                "pdb_id": "1ABC",
                "output_pose_file": "pose.pdbqt",
                "gnina_ligand_input": "pose.pdbqt",
                "gnina_receptor_input": "rec.pdb",
            }
        ]
    ).to_csv(paths["gnina_input"], index=False)

    enriched_path = prepare_score_population(paths)
    enriched = pd.read_csv(enriched_path)

    assert enriched.loc[0, "smiles"] == "CCO"
    assert enriched.loc[0, "unidock_best_score"] == -9.0
    assert enriched.loc[0, "gnina_cnnscore"] == 0.8
    assert enriched.loc[0, "gnina_affinity"] == -8.2
    assert enriched.loc[0, "output_pose_file"] == "pose.pdbqt"


def test_redocking_gate_is_off_by_default() -> None:
    # Report-only by default: redocking runs early but must NOT drop receptors
    # until the user validates the docking config against a known pose.
    assert redocking_gate_enabled({}) is False
    assert redocking_gate_enabled({"redocking": {"native": {}}}) is False
    assert redocking_gate_enabled({"redocking": {"native": {"filter_receptors": True}}}) is True


def test_enabled_gate_without_filtered_boxes_fails_loudly(tmp_path: Path) -> None:
    # With the gate on but the native_redocking stage not yet run (no filtered
    # boxes on disk), screening must error rather than silently use all receptors.
    import pytest

    config = {"redocking": {"native": {"filter_receptors": True}}, "docking": {"boxes": str(tmp_path / "boxes.csv")}}
    paths = {"redocking_filtered_boxes": tmp_path / "missing_passed_boxes.csv"}
    with pytest.raises(FileNotFoundError):
        docking_boxes_for_screening(config, paths)


def test_disabled_gate_uses_configured_boxes(tmp_path: Path) -> None:
    boxes = tmp_path / "boxes.csv"
    boxes.write_text("pdb_id\n", encoding="utf-8")
    config = {"docking": {"boxes": str(boxes)}}  # filter_receptors unset -> off
    paths = {"redocking_filtered_boxes": tmp_path / "passed_boxes.csv"}  # does not exist
    assert docking_boxes_for_screening(config, paths) == boxes
