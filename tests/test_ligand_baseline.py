from pathlib import Path

import pandas as pd


from screensift.metrics.ligand_baseline import (
    compute_ecfp4,
    compute_leave_one_active_out_similarity,
    mol_from_smiles_safe,
)


def test_ecfp4_fingerprint_generated_for_aspirin() -> None:
    mol = mol_from_smiles_safe("CC(=O)Oc1ccccc1C(=O)O")
    fp = compute_ecfp4(mol)

    assert mol is not None
    assert fp is not None
    assert fp.GetNumBits() == 2048


def test_invalid_smiles_is_handled_safely() -> None:
    df = pd.DataFrame(
        [
            {"ligand_id": "a", "activity_label": "active", "canonical_smiles": "CCO"},
            {"ligand_id": "bad", "activity_label": "inactive", "canonical_smiles": "not_smiles"},
        ]
    )

    result = compute_leave_one_active_out_similarity(df)
    bad = result[result["ligand_id"] == "bad"].iloc[0]

    assert not bool(bad["ecfp4_valid"])
    assert bad["ecfp4_failure_reason"] == "invalid_or_missing_smiles"


def test_inactive_nearest_active_similarity_works() -> None:
    df = pd.DataFrame(
        [
            {"ligand_id": "active1", "activity_label": "active", "canonical_smiles": "CCO"},
            {"ligand_id": "inactive1", "activity_label": "inactive", "canonical_smiles": "CCO"},
        ]
    )

    result = compute_leave_one_active_out_similarity(df)
    inactive = result[result["ligand_id"] == "inactive1"].iloc[0]

    assert inactive["ecfp4_active_similarity"] == 1.0
    assert inactive["ecfp4_nearest_active_ligand_id"] == "active1"


def test_active_self_match_and_exact_duplicate_are_excluded() -> None:
    df = pd.DataFrame(
        [
            {"ligand_id": "active1", "activity_label": "active", "canonical_smiles": "CCO", "inchikey": "same"},
            {"ligand_id": "active_duplicate", "activity_label": "active", "canonical_smiles": "CCO", "inchikey": "same"},
            {"ligand_id": "active_other", "activity_label": "active", "canonical_smiles": "c1ccccc1", "inchikey": "other"},
        ]
    )

    result = compute_leave_one_active_out_similarity(df)
    active1 = result[result["ligand_id"] == "active1"].iloc[0]

    assert active1["ecfp4_nearest_active_ligand_id"] == "active_other"
    assert active1["ecfp4_active_similarity"] < 1.0
