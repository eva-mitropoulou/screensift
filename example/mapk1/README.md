# MAPK1 Example

This directory contains a runnable MAPK1 ScreenSift example built from the
existing two-receptor scored ligand table.

Input files:

- `schema.yml`: maps the MAPK1 table columns to ScreenSift fields.
- `expected_outputs/mapk1_2receptor_score_population.csv`: scored MAPK1 ligand table for 4QTA and 6SLG with canonical molecules, activity labels, Uni-Dock scores, GNINA scores, and saved ECFP4/Tanimoto similarity.
- `expected_outputs/mapk1_candidates.csv`: top-100 candidate table produced from the two-receptor score population.
- `mapk1_docking_boxes.csv`: docking-box definitions for 4QTA and 6SLG, used only if you switch the example to `full_screening`.
- `lit_pcba_full_ligands.csv`: full MAPK1 LIT-PCBA ligand table made from `actives.smi` and `inactives.smi`.
- `lit_pcba_full_schema.yml`: schema for the full LIT-PCBA ligand table.
- `full_lit_pcba_screening.yml`: screening config that keeps all actives and all inactives.
- `full_lit_pcba_pipeline.yml`: full Uni-Dock/GNINA Docker GPU pipeline config for review before running.

Run the example:

```bash
PYTHONPATH=src python example/mapk1/run_example.py
```

This writes one candidate table using `pipeline.yml`:

- `runs/mapk1_example/tables/mapk1_example_candidates.csv`

The equivalent CLI command is:

```bash
PYTHONPATH=src python -m screensift.cli \
  --schema example/mapk1/schema.yml \
  --data example/mapk1/expected_outputs/mapk1_2receptor_score_population.csv \
  --target MAPK1 \
  --similarity-score ecfp4_active_similarity \
  --structure-aggregation max \
  --candidate-aggregation weighted_mean \
  --candidate-weight similarity_score_norm:0.75 \
  --candidate-weight structure_score_norm:0.25 \
  --structure-cutoff 0.70 \
  --similarity-cutoff 0.70 \
  --n-candidates 100 \
  --out runs/mapk1_example/tables/mapk1_example_candidates.csv
```

The command does not list Uni-Dock/GNINA score directions because ScreenSift auto-detects the standard columns in this table.

The run returns the requested top `n` candidates by `candidate_score`. Evidence buckets in the output explain why each row ranked, but they are not quotas.

You can also run the config-driven example:

```bash
PYTHONPATH=src python -m screensift.pipeline --config example/mapk1/pipeline.yml
```

The config has two main choices:

```yaml
workflow:
  mode: ranking_only

ranking:
  evidence_mode: combined
```

`workflow.mode` says what work to run. `ranking_only` writes one candidate CSV from the included scored table. `full_screening` is the larger Uni-Dock/GNINA/redocking workflow.

`ranking.evidence_mode` says how the candidates are scored: `similarity`, `structure`, or `combined`.

The included example uses weighted candidate aggregation:

```yaml
similarity_score_column: ecfp4_active_similarity
candidate_aggregation: weighted_mean
candidate_weights:
  similarity_score_norm: 0.75
  structure_score_norm: 0.25
```
