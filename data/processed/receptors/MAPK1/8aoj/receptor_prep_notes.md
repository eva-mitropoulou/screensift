# Receptor Prep Notes: 8AOJ

- Source file: `data/raw/pdb/MAPK1/8aoj.cif`
- Receptor PDBQT status: `complete`

## Non-water Hetero Residues
- N8L chain A residue 407 (21 heavy atoms; hetero flag `H_N8L`)
- CME chain A residue 161 (10 heavy atoms; hetero flag `H_CME`)
- SO4 chain A residue 401 (5 heavy atoms; hetero flag `H_SO4`)
- SO4 chain A residue 406 (5 heavy atoms; hetero flag `H_SO4`)
- EDO chain A residue 402 (4 heavy atoms; hetero flag `H_EDO`)
- EDO chain A residue 403 (4 heavy atoms; hetero flag `H_EDO`)
- EDO chain A residue 404 (4 heavy atoms; hetero flag `H_EDO`)
- DMS chain A residue 405 (4 heavy atoms; hetero flag `H_DMS`)

## Preparation Log
/home/ubuntu/miniforge3/envs/cadd-mapk1/bin/mk_prepare_receptor.py -i data/processed/receptors/MAPK1/8aoj/receptor_clean.pdb -o data/processed/receptors/MAPK1/8aoj/receptor -p -a --default_altloc A
