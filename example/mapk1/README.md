# MAPK1 Example

This directory contains a runnable MAPK1 ScreenSift example built from the existing phase 1 scored ligand table.

Input files:

- `schema.yml`: maps the MAPK1 table columns to ScreenSift fields.
- `mapk1_phase1_score_population.csv`: scored MAPK1 ligand table with canonical molecules, activity labels, Uni-Dock scores, and GNINA scores.
- `mapk1_docking_boxes.csv`: MAPK1 docking-box definitions used only if you switch the example to `full_screening`.
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
  --target MAPK1 \
  --structure-aggregation max \
  --candidate-aggregation max \
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

Weighted aggregation is also supported, but the user must supply the weights explicitly. Example:

```bash
PYTHONPATH=src python -m screensift.cli \
  --schema example/mapk1/schema.yml \
  --target MAPK1 \
  --structure-aggregation weighted_mean \
  --structure-weight unidock_best_score:0.40 \
  --structure-weight CNNscore:0.30 \
  --structure-weight CNNaffinity:0.20 \
  --structure-weight gnina_affinity:0.10 \
  --candidate-aggregation weighted_mean \
  --candidate-weight structure_score_norm:0.80 \
  --candidate-weight similarity_score_norm:0.20 \
  --n-candidates 100 \
  --out runs/mapk1_example/tables/mapk1_example_candidates.csv
```
