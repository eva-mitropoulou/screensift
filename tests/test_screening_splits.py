from pathlib import Path

import pandas as pd


from screensift.curation.make_screening_splits import make_splits


def write_screening_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "random_seed: 7",
                "phase1_inactives: 3",
                "require_all_actives: true",
                "deduplicate_by: inchikey",
                "activity_label_column: activity_label",
                "active_label: active",
                "inactive_label: inactive",
                "split_id_column: split_id",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_tiny_curated(path: Path) -> pd.DataFrame:
    rows = []
    for index in range(2):
        rows.append(
            {
                "ligand_id": f"active_{index}",
                "activity_label": "active",
                "inchikey": f"ACTIVEKEY{index}",
                "canonical_smiles": "CCO",
                "valid": True,
            }
        )
    for index in range(10):
        rows.append(
            {
                "ligand_id": f"inactive_{index}",
                "activity_label": "inactive",
                "inchikey": f"INACTIVEKEY{index}",
                "canonical_smiles": "CCC",
                "valid": True,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def test_screening_splits_include_all_actives_and_requested_inactives(tmp_path: Path) -> None:
    curated_path = tmp_path / "mapk1_ligands_curated.csv"
    config_path = tmp_path / "screening.yml"
    out_dir = tmp_path / "splits"
    report_path = tmp_path / "manifest.json"
    curated = write_tiny_curated(curated_path)
    write_screening_config(config_path)

    manifest = make_splits(curated_path, config_path, out_dir, report_path, target_slug="mapk1")
    phase1 = pd.read_csv(out_dir / "mapk1_phase1_screening_set.csv")

    active_keys = set(curated.loc[curated["activity_label"] == "active", "inchikey"])
    assert set(phase1.loc[phase1["activity_label"] == "active", "inchikey"]) == active_keys
    assert (phase1["activity_label"] == "inactive").sum() == 3
    assert manifest["phase1_actives"] == 2
    assert manifest["phase1_inactives"] == 3
    assert sorted(path.name for path in out_dir.glob("*.csv")) == ["mapk1_phase1_screening_set.csv"]


def test_screening_splits_can_include_all_inactives(tmp_path: Path) -> None:
    curated_path = tmp_path / "mapk1_ligands_curated.csv"
    config_path = tmp_path / "screening.yml"
    out_dir = tmp_path / "splits"
    report_path = tmp_path / "manifest.json"
    curated = write_tiny_curated(curated_path)
    config_path.write_text(
        "\n".join(
            [
                "random_seed: 7",
                "include_all_inactives: true",
                "phase1_inactives: 3",
                "activity_label_column: activity_label",
                "active_label: active",
                "inactive_label: inactive",
                "split_id_column: split_id",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = make_splits(curated_path, config_path, out_dir, report_path, target_slug="mapk1", run_id="full")
    split = pd.read_csv(out_dir / "mapk1_full_screening_set.csv")

    assert len(split) == len(curated)
    assert int((split["activity_label"] == "inactive").sum()) == 10
    assert manifest["include_all_inactives"] is True
    assert manifest["phase1_inactives"] == 10


def test_screening_splits_are_reproducible(tmp_path: Path) -> None:
    curated_path = tmp_path / "mapk1_ligands_curated.csv"
    config_path = tmp_path / "screening.yml"
    curated = write_tiny_curated(curated_path)
    write_screening_config(config_path)
    assert len(curated) == 12

    make_splits(curated_path, config_path, tmp_path / "splits_a", tmp_path / "manifest_a.json", target_slug="mapk1")
    make_splits(curated_path, config_path, tmp_path / "splits_b", tmp_path / "manifest_b.json", target_slug="mapk1")

    phase1_a = pd.read_csv(tmp_path / "splits_a" / "mapk1_phase1_screening_set.csv")
    phase1_b = pd.read_csv(tmp_path / "splits_b" / "mapk1_phase1_screening_set.csv")

    assert phase1_a.equals(phase1_b)


def test_scaffold_folds_are_disjoint_and_reproducible(tmp_path: Path) -> None:
    curated_path = tmp_path / "mapk1_ligands_curated.csv"
    config_path = tmp_path / "screening.yml"
    # Distinct scaffolds so folding is meaningful.
    rows = [
        {"ligand_id": "a0", "activity_label": "active", "inchikey": "K0", "canonical_smiles": "c1ccccc1", "valid": True},
        {"ligand_id": "a1", "activity_label": "active", "inchikey": "K1", "canonical_smiles": "c1ccncc1", "valid": True},
        {"ligand_id": "i0", "activity_label": "inactive", "inchikey": "K2", "canonical_smiles": "c1ccc2ccccc2c1", "valid": True},
        {"ligand_id": "i1", "activity_label": "inactive", "inchikey": "K3", "canonical_smiles": "C1CCCCC1", "valid": True},
        {"ligand_id": "i2", "activity_label": "inactive", "inchikey": "K4", "canonical_smiles": "c1ccc(-c2ccccc2)cc1", "valid": True},
    ]
    pd.DataFrame(rows).to_csv(curated_path, index=False)
    config_path.write_text(
        "\n".join(
            [
                "random_seed: 7",
                "include_all_inactives: true",
                "activity_label_column: activity_label",
                "active_label: active",
                "inactive_label: inactive",
                "split_id_column: split_id",
                "scaffold_folds: 3",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = make_splits(curated_path, config_path, tmp_path / "s", tmp_path / "m.json", target_slug="mapk1")
    split = pd.read_csv(tmp_path / "s" / "mapk1_phase1_screening_set.csv")

    assert "scaffold_fold" in split.columns
    assert split["scaffold_fold"].between(0, 2).all()
    # No scaffold may span more than one fold (holdout-safe).
    assert manifest["scaffold_split"]["scaffolds_spanning_multiple_folds"] == 0


def test_screening_splits_support_custom_run_id(tmp_path: Path) -> None:
    curated_path = tmp_path / "chembl_target_ligands_curated.csv"
    config_path = tmp_path / "screening.yml"
    out_dir = tmp_path / "splits"
    write_tiny_curated(curated_path)
    write_screening_config(config_path)

    make_splits(curated_path, config_path, out_dir, tmp_path / "manifest.json", target_slug="chembl_target", run_id="pilot")

    split = pd.read_csv(out_dir / "chembl_target_pilot_screening_set.csv")
    assert split["split_id"].str.startswith("CHEMBL_TARGET_pilot").all()
