from pathlib import Path

import pandas as pd


from screensift.metrics.scaffold_leakage import compute_scaffold_leakage, scaffold_from_smiles


def test_scaffold_generation_returns_non_empty_for_normal_molecule() -> None:
    scaffold = scaffold_from_smiles("CC(=O)Oc1ccccc1C(=O)O")

    assert scaffold


def test_mixed_scaffold_counts_are_correct_on_toy_data() -> None:
    df = pd.DataFrame(
        [
            {"ligand_id": "a1", "activity_label": "active", "canonical_smiles": "Cc1ccccc1"},
            {"ligand_id": "i1", "activity_label": "inactive", "canonical_smiles": "Oc1ccccc1"},
            {"ligand_id": "i2", "activity_label": "inactive", "canonical_smiles": "C1CCCCC1"},
        ]
    )

    _, summary = compute_scaffold_leakage(df)
    mixed = summary[summary["scaffold_class"] == "mixed"]

    assert not mixed.empty
    assert int(mixed.iloc[0]["n_active"]) == 1
    assert int(mixed.iloc[0]["n_inactive"]) == 1
