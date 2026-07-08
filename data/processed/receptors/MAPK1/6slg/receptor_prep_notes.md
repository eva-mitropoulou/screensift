# Receptor Prep Notes: 6SLG

- Source file: `data/raw/pdb/MAPK1/6slg.cif`
- Receptor PDBQT status: `complete`

## Non-water Hetero Residues
- LHZ chain A residue 401 (36 heavy atoms; hetero flag `H_LHZ`)
- SO4 chain A residue 402 (5 heavy atoms; hetero flag `H_SO4`)
- SO4 chain A residue 403 (5 heavy atoms; hetero flag `H_SO4`)
- EDO chain A residue 404 (4 heavy atoms; hetero flag `H_EDO`)
- EDO chain A residue 405 (4 heavy atoms; hetero flag `H_EDO`)
- EDO chain A residue 406 (4 heavy atoms; hetero flag `H_EDO`)
- EDO chain A residue 407 (4 heavy atoms; hetero flag `H_EDO`)

## Preparation Log
/home/ubuntu/miniforge3/envs/cadd-mapk1/bin/mk_prepare_receptor.py -i data/processed/receptors/MAPK1/6slg/receptor_clean.pdb -o data/processed/receptors/MAPK1/6slg/receptor -p -a --default_altloc A
