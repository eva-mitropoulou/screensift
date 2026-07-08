# Receptor Prep Notes: 5WP1

- Source file: `data/raw/pdb/MAPK1/5wp1.cif`
- Receptor PDBQT status: `complete`

## Non-water Hetero Residues
- B7S chain A residue 401 (13 heavy atoms; hetero flag `H_B7S`)
- CME chain A residue 127 (10 heavy atoms; hetero flag `H_CME`)
- CME chain A residue 161 (10 heavy atoms; hetero flag `H_CME`)
- CME chain A residue 166 (10 heavy atoms; hetero flag `H_CME`)
- BEZ chain A residue 403 (9 heavy atoms; hetero flag `H_BEZ`)
- SO4 chain A residue 402 (5 heavy atoms; hetero flag `H_SO4`)

## Preparation Log
/home/ubuntu/miniforge3/envs/cadd-mapk1/bin/mk_prepare_receptor.py -i data/processed/receptors/MAPK1/5wp1/receptor_clean.pdb -o data/processed/receptors/MAPK1/5wp1/receptor -p -a --default_altloc A
