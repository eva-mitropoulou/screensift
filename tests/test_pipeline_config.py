from pathlib import Path

import yaml

import pandas as pd

from screensift.pipeline import output_paths, prepare_score_population, run_pipeline


def test_pipeline_dry_run_writes_manifest(tmp_path: Path) -> None:
    config = {
        "target": "TOY",
        "run_id": "smoke",
        "target_slug": "toy",
        "pipeline": {"stages": ["rank_candidates", "native_redocking"]},
        "paths": {"run_dir": str(tmp_path / "run"), "receptor_root": str(tmp_path / "receptors")},
        "dataset": {"schema": "example/mapk1/schema.yml"},
        "ranking": {
            "score_table": "example/mapk1/mapk1_phase1_score_population.csv",
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
            "score_table": "example/mapk1/mapk1_phase1_score_population.csv",
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
