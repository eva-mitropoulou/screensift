from pathlib import Path

from rdkit import Chem


from screensift.validation.check_native_ligand_references import (
    analyze_reference,
    fetch_ccd_reference,
    sanity_flags_for_mol,
)


def test_valid_simple_molecule_is_marked_sane() -> None:
    mol = Chem.MolFromSmiles("NCCc1ccccc1")

    flags = sanity_flags_for_mol(mol, "ligand.sdf", "LIG")

    assert "chemically_suspicious" not in flags
    assert "likely_solvent_or_ion" not in flags


def test_missing_file_is_handled_safely(tmp_path: Path) -> None:
    row = analyze_reference(tmp_path / "missing.pdb")

    assert row["load_success"] is False
    assert "missing_file" in row["sanity_flags"]
    assert row["recommendation"] == "exclude"


def test_disconnected_molecule_is_flagged() -> None:
    mol = Chem.MolFromSmiles("CC.CC")

    flags = sanity_flags_for_mol(mol, "ligand.sdf", "LIG")

    assert "disconnected_fragments" in flags
    assert "chemically_suspicious" in flags


def test_tiny_ion_or_solvent_like_molecule_is_flagged() -> None:
    mol = Chem.MolFromSmiles("[Na+]")

    flags = sanity_flags_for_mol(mol, "native_ligand.sdf", "NA")

    assert "likely_solvent_or_ion" in flags
    assert "too_few_heavy_atoms" in flags


def test_duplicate_native_reference_is_detected() -> None:
    mol = Chem.MolFromSmiles("NCCc1ccccc1")

    flags = sanity_flags_for_mol(mol, "ligand.sdf", "LIG", duplicate=True)

    assert "duplicate_reference" in flags


def test_ccd_lookup_failure_does_not_crash(tmp_path: Path) -> None:
    class FailingSession:
        def get(self, *args, **kwargs):
            raise RuntimeError("network unavailable")

    path, failure = fetch_ccd_reference("ABC", tmp_path, session=FailingSession())

    assert path is None
    assert failure is not None
    assert "download_failed" in failure
