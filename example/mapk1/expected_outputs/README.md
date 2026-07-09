# Expected outputs — MAPK1 two-receptor reference run

These are the small reference artifacts used by the shipped MAPK1 example. The
example uses only two receptor setups, `4QTA` and `6SLG`, and ranks the existing
two-receptor docking/GNINA score table with saved ECFP4/Tanimoto similarity.

| file | what it is |
| --- | --- |
| `mapk1_2receptor_score_population.csv` | Existing score population for 51,507 ligands docked/rescored on 4QTA and 6SLG, with `ecfp4_active_similarity` populated for every row. |
| `mapk1_candidates.csv` | Final top-100 triage shortlist from `find_candidates(...)` using saved ECFP4/Tanimoto similarity with 0.75 similarity / 0.25 structure weighting; 6 known actives and 94 known inactives. |

Numbers are a retrospective benchmark on a public dataset, not a claim of new
MAPK1 inhibitors.
