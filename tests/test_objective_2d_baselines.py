from pathlib import Path

import pandas as pd


from screensift.metrics.objective_2d_baselines import (
    compute_all_active_leave_one_out,
    compute_few_active_similarity,
    compute_inactive_reference_control,
    compute_label_shuffle_control,
    compute_near_analog_subset,
    compute_scaffold_holdout_similarity,
    prepare_fingerprints,
)


def _toy_population() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ligand_id": "a1", "activity_label": "active", "is_active": 1, "canonical_smiles": "CCO", "inchikey": "same"},
            {"ligand_id": "a_dup", "activity_label": "active", "is_active": 1, "canonical_smiles": "CCO", "inchikey": "same"},
            {"ligand_id": "a2", "activity_label": "active", "is_active": 1, "canonical_smiles": "c1ccccc1", "inchikey": "benzene"},
            {"ligand_id": "a3", "activity_label": "active", "is_active": 1, "canonical_smiles": "C1CCCCC1", "inchikey": "cyclohexane"},
            {"ligand_id": "i1", "activity_label": "inactive", "is_active": 0, "canonical_smiles": "CCO", "inchikey": "inactive"},
            {"ligand_id": "i2", "activity_label": "inactive", "is_active": 0, "canonical_smiles": "CCN", "inchikey": "inactive2"},
        ]
    )


def test_all_active_loo_excludes_self_and_duplicate_but_inactive_can_match() -> None:
    df = prepare_fingerprints(_toy_population())

    result = compute_all_active_leave_one_out(df)
    a1 = result[result["ligand_id"] == "a1"].iloc[0]
    i1 = result[result["ligand_id"] == "i1"].iloc[0]

    assert a1["all_active_loo_nearest_ligand_id"] != "a1"
    assert a1["all_active_loo_nearest_ligand_id"] != "a_dup"
    assert i1["all_active_loo_ecfp4_analog_neighborhood"] == 1.0


def test_few_active_removes_reference_actives_and_is_deterministic() -> None:
    df = prepare_fingerprints(_toy_population())

    first, refs_first = compute_few_active_similarity(df, k=2, repeat_seed=7)
    second, refs_second = compute_few_active_similarity(df, k=2, repeat_seed=7)

    assert refs_first == refs_second
    assert len(refs_first) == 2
    assert not set(refs_first) & set(first["ligand_id"])
    pd.testing.assert_series_equal(first["few_active_ecfp4"].reset_index(drop=True), second["few_active_ecfp4"].reset_index(drop=True))


def test_scaffold_holdout_excludes_reference_scaffolds_from_evaluation() -> None:
    df = pd.DataFrame(
        [
            {"ligand_id": "a_benz1", "activity_label": "active", "is_active": 1, "canonical_smiles": "Cc1ccccc1"},
            {"ligand_id": "a_benz2", "activity_label": "active", "is_active": 1, "canonical_smiles": "Oc1ccccc1"},
            {"ligand_id": "a_cyclo1", "activity_label": "active", "is_active": 1, "canonical_smiles": "CC1CCCCC1"},
            {"ligand_id": "a_cyclo2", "activity_label": "active", "is_active": 1, "canonical_smiles": "OC1CCCCC1"},
            {"ligand_id": "a_pyr1", "activity_label": "active", "is_active": 1, "canonical_smiles": "Cc1ccncc1"},
            {"ligand_id": "a_pyr2", "activity_label": "active", "is_active": 1, "canonical_smiles": "Oc1ccncc1"},
            {"ligand_id": "i1", "activity_label": "inactive", "is_active": 0, "canonical_smiles": "CCO"},
            {"ligand_id": "i2", "activity_label": "inactive", "is_active": 0, "canonical_smiles": "CCN"},
        ]
    )
    df = prepare_fingerprints(df)

    heldout, ref_scaffolds = compute_scaffold_holdout_similarity(
        df,
        repeat_seed=11,
        reference_scaffold_fraction=0.34,
        min_reference_active_scaffolds=1,
        min_test_active_scaffolds=1,
        min_test_actives_required=1,
    )

    assert ref_scaffolds
    assert not set(heldout["scaffold_smiles"].dropna()) & set(ref_scaffolds)


def test_near_analog_filter_and_negative_controls() -> None:
    df = prepare_fingerprints(_toy_population())
    df = compute_all_active_leave_one_out(df)

    low = compute_near_analog_subset(df, "all_active_loo_ecfp4_analog_neighborhood", 0.50)
    shuffled = compute_label_shuffle_control(df, repeat_seed=13)
    inactive_ref, refs = compute_inactive_reference_control(df, k=2, repeat_seed=3)

    assert (low["all_active_loo_ecfp4_analog_neighborhood"] < 0.50).all()
    assert len(shuffled) == len(df)
    assert len(refs) == 2
    assert inactive_ref["inactive_reference_ecfp4"].notna().any()
