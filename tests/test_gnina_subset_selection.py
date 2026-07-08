from pathlib import Path

import pandas as pd


from screensift.rescoring.select_gnina_subset import select_gnina_subset


def write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    receptor_root = tmp_path / "receptors" / "MAPK1"
    (receptor_root / "aaaa").mkdir(parents=True)
    (receptor_root / "aaaa" / "receptor_clean.pdb").write_text("ATOM\n", encoding="utf-8")

    pose_dir = tmp_path / "poses"
    pose_dir.mkdir()
    pose_paths = {}
    for ligand_id in ["a", "b", "d", "e", "f"]:
        pose = pose_dir / f"{ligand_id}_out.pdbqt"
        pose.write_text("REMARK VINA RESULT: -7.0 0.000 0.000\n", encoding="utf-8")
        pose_paths[ligand_id] = pose

    best = pd.DataFrame(
        [
            {
                "ligand_id": "a",
                "activity_label": "active",
                "best_score": -10.0,
                "best_pdb_id": "AAAA",
                "best_receptor_pdbqt": "rec_a.pdbqt",
                "best_output_pose_file": str(pose_paths["a"]),
                "canonical_smiles": "CCO",
            },
            {
                "ligand_id": "d",
                "activity_label": "inactive",
                "best_score": -9.0,
                "best_pdb_id": "BBBB",
                "best_receptor_pdbqt": "rec_b.pdbqt",
                "best_output_pose_file": str(pose_paths["d"]),
                "canonical_smiles": "CCN",
            },
            {
                "ligand_id": "b",
                "activity_label": "inactive",
                "best_score": -8.0,
                "best_pdb_id": "AAAA",
                "best_receptor_pdbqt": "rec_a.pdbqt",
                "best_output_pose_file": str(pose_paths["b"]),
                "canonical_smiles": "CCC",
            },
            {
                "ligand_id": "c",
                "activity_label": "active",
                "best_score": -7.0,
                "best_pdb_id": "AAAA",
                "best_receptor_pdbqt": "rec_a.pdbqt",
                "best_output_pose_file": str(pose_dir / "c_missing_out.pdbqt"),
                "canonical_smiles": "CCCl",
            },
            {
                "ligand_id": "e",
                "activity_label": "inactive",
                "best_score": -6.0,
                "best_pdb_id": "AAAA",
                "best_receptor_pdbqt": "rec_a.pdbqt",
                "best_output_pose_file": str(pose_paths["e"]),
                "canonical_smiles": "CCBr",
            },
            {
                "ligand_id": "f",
                "activity_label": "inactive",
                "best_score": 1.0,
                "best_pdb_id": "AAAA",
                "best_receptor_pdbqt": "rec_a.pdbqt",
                "best_output_pose_file": str(pose_paths["f"]),
                "canonical_smiles": "CCI",
            },
        ]
    )
    split = best[["ligand_id", "activity_label", "canonical_smiles"]].copy()

    best_path = tmp_path / "best.csv"
    split_path = tmp_path / "split.csv"
    best.to_csv(best_path, index=False)
    split.to_csv(split_path, index=False)
    return best_path, split_path, receptor_root, tmp_path / "gnina"


def test_gnina_subset_selection_rules_and_invalid_inputs(tmp_path: Path) -> None:
    best_path, split_path, receptor_root, gnina_root = write_inputs(tmp_path)

    out_path = tmp_path / "subset.csv"
    report_path = tmp_path / "report.md"
    subset, metadata = select_gnina_subset(
        best_path,
        split_path,
        out_path,
        report_path,
        top_unidock=2,
        random_inactives=2,
        include_all_actives=True,
        seed=11,
        top_ecfp_similarity=0,
        receptor_root=receptor_root,
        gnina_output_root=gnina_root,
    )

    assert out_path.exists()
    assert report_path.exists()
    assert metadata["valid_candidate_rows"] == 5
    assert "f" not in set(subset["ligand_id"])

    top_reason = dict(zip(subset["ligand_id"], subset["selection_reasons"], strict=False))
    assert "top_unidock" in top_reason["a"]
    assert "top_unidock" in top_reason["d"]
    assert "all_valid_actives" in top_reason["a"]
    assert "all_valid_actives" in top_reason["c"]

    d_row = subset[subset["ligand_id"] == "d"].iloc[0]
    assert not bool(d_row["valid_for_gnina"])
    assert "missing_receptor_clean_pdb" in d_row["invalid_reason"]

    c_row = subset[subset["ligand_id"] == "c"].iloc[0]
    assert not bool(c_row["valid_for_gnina"])
    assert "missing_ligand_pose" in c_row["invalid_reason"]


def test_gnina_random_controls_are_reproducible(tmp_path: Path) -> None:
    best_path, split_path, receptor_root, gnina_root = write_inputs(tmp_path)

    kwargs = {
        "top_unidock": 1,
        "random_inactives": 2,
        "include_all_actives": True,
        "seed": 23,
        "top_ecfp_similarity": 0,
        "receptor_root": receptor_root,
        "gnina_output_root": gnina_root,
    }
    first, _metadata_first = select_gnina_subset(best_path, split_path, tmp_path / "first.csv", tmp_path / "first.md", **kwargs)
    second, _metadata_second = select_gnina_subset(best_path, split_path, tmp_path / "second.csv", tmp_path / "second.md", **kwargs)

    first_reasons = first[["ligand_id", "selection_reasons"]].sort_values("ligand_id").reset_index(drop=True)
    second_reasons = second[["ligand_id", "selection_reasons"]].sort_values("ligand_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(first_reasons, second_reasons)
