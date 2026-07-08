from __future__ import annotations

from typing import Any

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


DESCRIPTOR_KEYS = [
    "mol_wt",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "heavy_atoms",
]


def mol_from_smiles_safe(smiles: str) -> Chem.Mol | None:
    """Parse a SMILES string without raising RDKit errors to callers."""
    if smiles is None:
        return None
    cleaned = str(smiles).strip()
    if not cleaned:
        return None
    try:
        return Chem.MolFromSmiles(cleaned)
    except Exception:
        return None


def canonicalize_smiles(smiles: str) -> str | None:
    mol = mol_from_smiles_safe(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def inchikey_from_smiles(smiles: str) -> str | None:
    mol = mol_from_smiles_safe(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def compute_basic_descriptors(smiles: str) -> dict[str, Any] | None:
    mol = mol_from_smiles_safe(smiles)
    if mol is None:
        return None

    try:
        return {
            "mol_wt": float(Descriptors.MolWt(mol)),
            "logp": float(Crippen.MolLogP(mol)),
            "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
            "hbd": int(Lipinski.NumHDonors(mol)),
            "hba": int(Lipinski.NumHAcceptors(mol)),
            "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
            "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        }
    except Exception:
        return None

