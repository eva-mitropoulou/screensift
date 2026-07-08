from pathlib import Path


from screensift.common.chemistry import DESCRIPTOR_KEYS, canonicalize_smiles, compute_basic_descriptors, inchikey_from_smiles


def test_aspirin_smiles_canonicalizes() -> None:
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    assert canonicalize_smiles(aspirin) is not None


def test_invalid_smiles_returns_none() -> None:
    assert canonicalize_smiles("not_a_smiles") is None


def test_descriptors_include_expected_keys() -> None:
    descriptors = compute_basic_descriptors("CC(=O)Oc1ccccc1C(=O)O")
    assert descriptors is not None
    assert set(DESCRIPTOR_KEYS).issubset(descriptors)


def test_aspirin_inchikey_is_generated() -> None:
    assert inchikey_from_smiles("CC(=O)Oc1ccccc1C(=O)O") is not None


def test_descriptors_are_numeric() -> None:
    descriptors = compute_basic_descriptors("CC(=O)Oc1ccccc1C(=O)O")
    assert descriptors is not None
    assert all(isinstance(descriptors[key], int | float) for key in DESCRIPTOR_KEYS)
