# Expected outputs — MAPK1 reference run

These are reference artifacts from the full LIT-PCBA MAPK1 run described in the
top-level `README.md`, committed so you can sanity-check your own run against a
known-good result. They are small summaries, not the multi-GB pose/score dumps
(those are regenerable — see the pipeline stages).

| file | what it is |
| --- | --- |
| `mapk1_candidates.csv` | Final top-100 triage shortlist from `find_candidates(...)` (2 known actives, 98 known inactives in this retrospective benchmark). |
| `native_redocking.csv` | Uni-Dock native-ligand redocking RMSD per receptor (symmetry-aware, receptor-frame RMSD; 2/5 pass the 2.0 Å gate). |
| `candidate_pose_qc.csv` | Uni-Dock-vs-GNINA predicted-pose agreement for the top 25 candidates. |
| `gnina_rescoring_validation.csv` | GNINA score-only checks on native / redocked / displaced-decoy poses. |
| `reports/*.md`, `reports/*.json` | Human-readable QC and manifest reports for each stage. |

RMSD values here are computed with the symmetry-aware, no-superposition metric
in `screensift.validation.rmsd` (poses are already in the receptor frame, so they
must NOT be re-superposed). Numbers are a retrospective benchmark on a public
dataset — not a claim of new MAPK1 inhibitors.
