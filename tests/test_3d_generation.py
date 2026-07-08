from pathlib import Path

from rdkit import Chem


from screensift.ligands.generate_3d_conformers import generate_3d_mol, generate_one_conformer


def test_generate_3d_for_aspirin_has_conformer() -> None:
    mol = generate_3d_mol("CC(=O)Oc1ccccc1C(=O)O", seed=42)

    assert mol.GetNumConformers() == 1


def test_generate_one_conformer_writes_sdf(tmp_path: Path) -> None:
    result = generate_one_conformer(
        {"ligand_id": "aspirin", "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O"},
        index=0,
        out_dir=tmp_path,
        seed=42,
    )

    assert result.success
    assert result.output_path is not None
    sdf_path = Path(result.output_path)
    assert sdf_path.exists()
    mol = next(mol for mol in Chem.SDMolSupplier(str(sdf_path), removeHs=False) if mol is not None)
    assert mol.GetNumConformers() == 1
