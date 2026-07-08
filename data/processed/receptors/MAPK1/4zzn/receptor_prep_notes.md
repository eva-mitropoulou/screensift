# Receptor Prep Notes: 4ZZN

- Source file: `data/raw/pdb/MAPK1/4zzn.cif`
- Receptor PDBQT status: `complete`

## Non-water Hetero Residues
- CQ8 chain A residue 1355 (25 heavy atoms; hetero flag `H_CQ8`)
- CME chain A residue 159 (10 heavy atoms; hetero flag `H_CME`)
- SO4 chain A residue 1354 (5 heavy atoms; hetero flag `H_SO4`)

## Preparation Log
/home/ubuntu/miniforge3/envs/cadd-mapk1/bin/mk_prepare_receptor.py -i data/processed/receptors/MAPK1/4zzn/receptor_clean.pdb -o data/processed/receptors/MAPK1/4zzn/receptor -p -a --default_altloc A
