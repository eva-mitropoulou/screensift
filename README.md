# ScreenSift

[![CI](https://github.com/eva-mitropoulou/mapk1-leakage-aware-screening/actions/workflows/ci.yml/badge.svg)](https://github.com/eva-mitropoulou/mapk1-leakage-aware-screening/actions/workflows/ci.yml)

<p align="center">
  <img src="docs/assets/screensift-logo.png" alt="ScreenSift logo" width="220">
</p>

**Screen wider. Sift smarter.** ScreenSift is an end-to-end virtual-screening
triage pipeline that turns a raw ligand table into a transparent, ranked
candidate shortlist — packaged as a tested, pip-installable Python API.

I ran it on the full LIT-PCBA **MAPK1** benchmark end to end:

- **61,971** ligands curated (RDKit canonicalization, InChIKey dedup, activity conflict quarantine)
- docked across a **5-receptor holo ensemble** — **309,230** Uni-Dock jobs
- **GNINA** CNN rescoring on **60,960** ligands (Docker/GPU)
- native-ligand **redocking** and score-only **rescoring** validation
- triaged to a transparent **top-100** shortlist with per-candidate evidence buckets

...then packaged the whole thing behind one function, `find_candidates(...)`, and a
config-driven CLI, with **114 passing tests** and CI.

> ScreenSift ranks and *explains* candidates from heterogeneous evidence
> (chemical similarity + docking/rescoring). It does **not** claim experimental
> activity — it produces a structured shortlist with explicit, per-channel
> evidence. See [Scope](#scope) and [Limitations](#limitations).

## Quickstart

```bash
pip install -e .                     # core find_candidates API (no PYTHONPATH tricks)
pip install -e ".[workflow]"         # + docking/prep tools for the full pipeline

# Rank the shipped MAPK1 example table in seconds:
python example/mapk1/run_example.py
# or via the installed CLI / config-driven orchestrator:
screensift-run-pipeline --config example/mapk1/pipeline.yml
```

```python
from screensift import find_candidates

candidates = find_candidates(
    schema="example/mapk1/schema.yml",
    data="example/mapk1/mapk1_phase1_score_population.csv",
    target="MAPK1",
    evidence_mode="combined",
    n_candidates=100,
)
```

## Overview

`ScreenSift` turns a ligand table into a ranked candidate table. In the full workflow, the user provides a dataset, a schema that explains the dataset columns, and a target with prepared receptor inputs. The package standardizes the molecules, prepares ligands, supports Uni-Dock/GNINA score generation, computes ligand-similarity evidence, normalizes the evidence channels, assigns interpretable support buckets, and returns the top candidates.

## Scientific Aim

Virtual-screening projects often combine heterogeneous evidence: chemical similarity, docking scores, machine-learning rescoring, and manual pose-review information. These signals are not always commensurate. A weighted sum can hide useful candidates when one score is weak or missing.

`ScreenSift` is designed as a transparent candidate-selection layer. It asks:

> Given a ligand dataset and available prioritization signals, which compounds should be carried forward, and why?

The output is not a claim of experimental activity. It is a structured shortlist with explicit evidence channels.

## Scope

`ScreenSift` is a candidate triage tool. It is not intended to benchmark docking or rescoring methods against ligand similarity baselines, and it does not claim that one score type is scientifically superior to another. Similarity and structure scores are treated as separate evidence channels used to explain why a ligand was prioritized.

## Main Workflow

The intended end-to-end path is:

```text
dataset + schema + target
  -> molecular audit and canonical ligand table
  -> ligand 3D/PDBQT preparation
  -> receptor/box-driven Uni-Dock screening
  -> optional GNINA rescoring
  -> ECFP4/Tanimoto similarity evidence
  -> normalized structure and similarity evidence
  -> candidate table
```

For a config-driven run, use:

```bash
screensift-run-pipeline --config configs/pipeline.example.yml
```

The two most important config choices are:

```yaml
workflow:
  mode: ranking_only     # or full_screening

ranking:
  evidence_mode: combined  # similarity, structure, or combined
```

`workflow.mode` controls how much work the pipeline runs. `ranking_only` reads an existing scored table and writes one candidate CSV. `full_screening` runs ligand preparation, Uni-Dock, GNINA rescoring, validation, and final ranking.

`ranking.evidence_mode` controls how candidates are ranked:

| mode | meaning |
| --- | --- |
| `similarity` | rank only by ligand-similarity evidence |
| `structure` | rank only by docking/rescoring evidence |
| `combined` | rank by both evidence channels |

The internal stage named `rank_candidates` is not a fourth ranking mode. It is simply the pipeline step that writes the final candidate table.

Use dry-run mode to inspect the planned work without executing external tools:

```bash
screensift-run-pipeline --config example/mapk1/pipeline.yml --dry-run
```

The final candidate selection step is exposed as a Python API:

```python
from screensift import find_candidates

candidates = find_candidates(
    schema="configs/my_dataset_schema.yml",
    data="example/mapk1/mapk1_phase1_score_population.csv",
    target="MY_TARGET",
    evidence_mode="combined",
    structure_aggregation="max",
    candidate_aggregation="max",
    structure_score_cutoff=0.70,
    similarity_score_cutoff=0.70,
    n_candidates=100,
)
```

If a similarity column is not supplied, `ScreenSift` computes ECFP4/Tanimoto similarity to known actives when active labels are available. Standard Uni-Dock/GNINA score columns are auto-detected with their known score direction. For custom external score columns, the user supplies whether higher or lower values are better.

The same operation is available from the command line:

```bash
screensift \
  --schema configs/my_dataset_schema.yml \
  --data example/mapk1/mapk1_phase1_score_population.csv \
  --target MY_TARGET \
  --evidence-mode combined \
  --structure-aggregation max \
  --candidate-aggregation max \
  --similarity-cutoff 0.70 \
  --structure-cutoff 0.70 \
  --n-candidates 100 \
  --out example/mapk1/my_target_candidates.csv
```

## How It Works Internally

### 1. Schema Mapping

Different datasets use different column names. A ChEMBL export might use `molecule_chembl_id`, `canonical_smiles`, and `standard_value`; a local CSV might use `compound_id`, `smiles`, and `label`.

`ScreenSift` uses a YAML schema to describe those columns:

```yaml
input:
  path: data/raw/my_target/ligands.csv
  sep: ","

columns:
  smiles: smiles
  ligand_id: compound_id
  activity_label: label

activity:
  mode: label
  active_values: ["active"]
  inactive_values: ["inactive"]
```

For ChEMBL-like data, labels can be derived from activity thresholds:

```yaml
activity:
  mode: threshold
  active_if:
    column: standard_value
    operator: "<="
    value: 1000
    allowed_relations: ["=", "<", "<="]
    allowed_units: ["nM"]
  inactive_if:
    column: standard_value
    operator: ">"
    value: 10000
    allowed_relations: ["=", ">", ">="]
    allowed_units: ["nM"]
```

### 2. Molecular Audit

Each input row is converted into an audit record. The audit step:

- reads the raw SMILES;
- assigns `ligand_id`;
- assigns `activity_label`;
- canonicalizes SMILES with RDKit;
- generates an InChIKey;
- computes basic descriptors;
- marks invalid rows with explicit `failure_reason`.

Rows can fail because of missing SMILES, invalid SMILES, failed InChIKey generation, descriptor failure, unrecognized activity labels, intermediate ChEMBL activity values, or conflicting activity rules.

The canonical internal fields are:

| field | purpose |
| --- | --- |
| `ligand_id` | stable compound identifier |
| `source_row` | original input row number |
| `activity_label` | `active`, `inactive`, or `unlabeled` |
| `raw_smiles` | original SMILES |
| `canonical_smiles` | RDKit-canonicalized SMILES |
| `inchikey` | molecular identity key used for deduplication |
| `valid` | whether the row is usable |
| `failure_reason` | reason a row was excluded |

### 3. Deduplication And Conflict Handling

Valid rows are deduplicated by `inchikey` by default. If the same molecular identity appears with conflicting active/inactive labels, the conflicting records are moved to the failure table instead of being silently kept.

This matters scientifically because duplicated or contradictory molecules can inflate apparent candidate quality or make the shortlist unstable.

### 4. Ligand Similarity Evidence

If the input table already contains a similarity column, the user can pass it:

```python
find_candidates(..., similarity_score_column="ecfp4_active_similarity")
```

If no similarity column is supplied, `ScreenSift` computes ECFP4/Tanimoto similarity to known actives:

1. Convert canonical SMILES to RDKit molecules.
2. Generate Morgan fingerprints with radius 2 and 2048 bits.
3. Identify active reference molecules from `activity_label == "active"`.
4. For each ligand, compute the maximum Tanimoto similarity to the active reference set.
5. Exclude self-matches for active molecules.

This produces `similarity_score_raw` and `similarity_score_norm`.

### 5. Structure-Score Evidence

In the full workflow, structure-score evidence is generated before final candidate selection. The included workflow supports Uni-Dock docking and GNINA score-only rescoring. Those steps produce score columns such as `unidock_best_score`, `CNNscore`, `CNNaffinity`, and `gnina_affinity`.

The final candidate-selection API is scoring-method agnostic. Uni-Dock and GNINA scores are supported by the workflow, but any numeric score column can be used for triage.

For standard Uni-Dock/GNINA columns, the user does not need to say whether higher or lower is better. If no structure-score mapping is provided, the API auto-detects common column names such as:

- `unidock_best_score`, `best_score`, `unidock_score`, `score`;
- `CNNscore`, `cnnscore`, `gnina_cnnscore`;
- `CNNaffinity`, `cnnaffinity`, `gnina_cnnaffinity`;
- `gnina_affinity`, `affinity`, `GNINA_affinity`.

For custom score columns, the user must provide the direction:

```python
structure_score_columns={
    "my_docking_score": "lower",
    "my_ml_score": "higher",
}
```

### 6. Score Normalization

Raw scores may have different directions and units. Docking scores are often better when lower; classifier-like scores are often better when higher.

`ScreenSift` does not blindly compare raw GNINA, Uni-Dock, and Tanimoto values. Structure-score columns are normalized independently with their auto-detected or user-declared score direction. Computed Tanimoto similarity is already bounded from 0 to 1, and supplied similarity columns are normalized as higher-is-better evidence.

For a score where higher is better:

```text
normalized = (value - min) / (max - min)
```

For a score where lower is better:

```text
normalized = (max - value) / (max - min)
```

By default, the best normalized structure score across the selected structure-score columns becomes `structure_score_norm`. The contributing column is stored in `primary_structure_score`.

Users can also request weighted structure aggregation:

```python
find_candidates(
    ...,
    structure_aggregation="weighted_mean",
    structure_weights={
        "unidock_best_score": 0.40,
        "CNNscore": 0.30,
        "CNNaffinity": 0.20,
        "gnina_affinity": 0.10,
    },
)
```

Weighted aggregation requires explicit user-supplied weights for every selected score column.

### 7. Evidence Buckets

`ScreenSift` compares normalized scores against user-defined cutoffs:

```python
structure_score_cutoff=0.70
similarity_score_cutoff=0.70
```

The similarity cutoff is a policy choice, not a universal rule. A high value such as `0.70` favors analog-like candidates that resemble known actives. Users should tune `similarity_score_cutoff` and `structure_score_cutoff` based on the intended triage policy.

The evidence bucket is assigned as:

| condition | bucket |
| --- | --- |
| similarity passes, structure passes | `consensus_supported` |
| similarity passes, structure does not pass | `analog_supported` |
| structure passes, similarity does not pass | `structure_supported` |
| neither passes | `deprioritized` |

This is deliberately not a quota system. The user requests `n_candidates`, and `ScreenSift` returns the top `n_candidates` rows by the unified ranking score. Evidence buckets explain why each returned row ranked; they do not reserve slots for similarity-supported or structure-supported candidates.

### 8. Candidate Ranking

The user chooses which evidence channels are allowed to control ranking:

| `evidence_mode` | ranking behavior |
| --- | --- |
| `similarity` | rank only by ligand-similarity evidence |
| `structure` | rank only by docking/rescoring evidence |
| `combined` | rank by both evidence channels |

The default final candidate score is:

```text
candidate_score = max(structure_score_norm, similarity_score_norm)
```

That default applies when `evidence_mode="combined"`.

Users can also request weighted candidate aggregation:

```python
find_candidates(
    ...,
    candidate_aggregation="weighted_mean",
    candidate_weights={
        "structure_score_norm": 0.80,
        "similarity_score_norm": 0.20,
    },
)
```

Weighted candidate aggregation requires explicit user-supplied weights for both `structure_score_norm` and `similarity_score_norm`.

Rows are sorted by:

1. `candidate_score`;
2. `structure_score_norm`;
3. `similarity_score_norm`.

The top `n_candidates` rows are returned.

The output table includes:

| column | meaning |
| --- | --- |
| `target` | target identifier passed by the user |
| `ligand_id` | canonical ligand ID |
| `activity_label` | activity label if available |
| `canonical_smiles` | canonical SMILES |
| `inchikey` | molecular identity key |
| `similarity_score_raw` | raw or computed similarity evidence |
| `similarity_score_norm` | normalized similarity evidence |
| `structure_score_norm` | best normalized structure evidence |
| `primary_structure_score` | structure-score column contributing the best value |
| `candidate_score` | max available normalized evidence |
| `evidence_bucket` | interpretable candidate-support category |

## End-To-End Screening Stages

The repository contains workflow modules for generating the structure-score evidence used by `find_candidates(...)`.

### Ligand Preparation

The ligand-preparation stage generates 3D conformers from SMILES using RDKit ETKDGv3, adds hydrogens, optimizes with MMFF or UFF, and converts prepared ligand structures to PDBQT using Meeko.

### Docking

The docking stage forms receptor-ligand docking jobs from ligand PDBQT files and docking-box definitions, executes Uni-Dock, parses raw docking outputs into score tables, and checks score validity, pose-file availability, duplicates, and suspicious values.

### Rescoring

The rescoring stage prepares subsets for GNINA score-only rescoring, runs GNINA, and parses GNINA output into tabular scores.

### Review And Reporting

The review stage creates richer candidate-triage reports when pose-inspection tables are available and supports pose-level review assets.

### Validation Strategy

For the Uni-Dock + GNINA workflow, the recommended validation path is:

1. Uni-Dock native-ligand redocking validates the receptor boxes and main docking setup.
2. GNINA score-only validation evaluates native, redocked, and displaced decoy poses to sanity-check the rescoring signal.
3. Optional candidate pose QC redocks selected candidates with GNINA and compares Uni-Dock-vs-GNINA predicted poses. This is a pose-agreement check, not experimental RMSD.

This keeps the roles clear: Uni-Dock is the high-throughput docking engine, GNINA is a rescoring and optional second-pose engine, and ScreenSift ranks normalized evidence channels.

The full workflow path is:

```text
schema + ligand table
  -> curated ligands
  -> ligand/receptor preparation
  -> Uni-Dock docking
  -> Uni-Dock native redocking validation
  -> GNINA score-only rescoring and rescoring sanity checks
  -> find_candidates(...)
  -> candidate table
```

## Tests And Validation

The test suite checks both the public package API and lower-level workflow pieces, including schema adaptation, split generation, docking input construction, score parsing, candidate ranking, and triage behavior.

Run:

```bash
python -m pytest tests/
```

Current local result: 114 tests passed, with 2 upstream MDAnalysis/numpy warnings.

The current workflow has also been run end to end on the LIT-PCBA `MAPK1` target. That run produced:

| stage | result |
| --- | --- |
| dataset audit | 61,971 curated ligands: 302 active and 61,669 inactive |
| screening set | full curated LIT-PCBA MAPK1 set: 61,971 ligands |
| ligand 3D generation | 61,846 successful structures, 125 timed-out conformer failures |
| ligand PDBQT preparation | 61,846 successful PDBQT files, 0 conversion failures |
| receptor preparation | 5 MAPK1 receptors prepared: 4QTA, 4ZZN, 5WP1, 6SLG, and 8AOJ |
| Uni-Dock screening | 309,230 receptor-ligand docking jobs completed |
| Uni-Dock best-per-ligand table | 60,960 ligands with at least one valid docked pose |
| native redocking validation | 5/5 native redocking jobs completed; 2/5 passed the 2.0 A RMSD threshold (symmetry-aware, receptor-frame RMSD) |
| GNINA rescoring | 60,960 ligands submitted with Docker/GPU GNINA; 60,856 complete and 104 failed |
| GNINA rescoring validation | 15/15 native, redocked, and displaced-decoy score-only checks completed |
| `find_candidates(...)` validation | top 100 candidates written by the ranking API: 2 known actives and 98 known inactives in this retrospective benchmark |
| candidate pose QC | top 25 candidates redocked with GNINA; 25/25 docking jobs completed, 24 comparable RMSDs, and 1 atom-count mismatch |

Candidate pose QC is an agreement check between predicted Uni-Dock and GNINA poses. In this run, the 24 comparable candidates had a median Uni-Dock-vs-GNINA RMSD of 2.74 A (symmetry-aware, receptor-frame): 11 consistent poses, 5 moderate shifts, and 8 discordant poses.

The `find_candidates(...)` validation used the full MAPK1 scored ligand table with auto-detected Uni-Dock/GNINA score directions. The output is a ranked triage table, not a claim of new MAPK1 inhibitors.

## Reference Outputs

Small known-good artifacts from the full MAPK1 run are committed under
[](example/mapk1/expected_outputs/) (final
candidate shortlist, native-redocking and pose-QC tables, and per-stage QC
reports) so you can compare your own run against a reference. The multi-GB
pose/score dumps are regenerable and are not committed.

## MAPK1 Example

A runnable MAPK1 example is included in `example/mapk1/`. It contains the example schema, the MAPK1 scored ligand table, and a small runner script:

```bash
PYTHONPATH=src python example/mapk1/run_example.py
```

The script reads `example/mapk1/pipeline.yml` and writes one candidate table for the selected ranking policy:

```text
runs/mapk1_example/tables/mapk1_example_candidates.csv
```

It returns the requested number of best-ranked candidates; evidence buckets are explanatory labels, not allocation rules.

The same directory also includes `pipeline.yml`, a commented config for the end-to-end orchestrator. It defaults to a portable ranking-only example, but shows the Uni-Dock/GNINA/redocking options needed for a full run.

## Installation

The core `find_candidates(...)` API installs from PyPI wheels alone (numpy,
pandas, scikit-learn, scipy, rdkit, pyyaml, joblib):

```bash
python -m pip install -e .          # core API + CLI
python -m pip install -e ".[test]"  # + pytest
```

The full end-to-end docking/prep workflow additionally needs external tools
(Meeko, Open Babel, Uni-Dock, GNINA). The reproducible conda environment
installs everything:

```bash
mamba env create -f environment.yml
mamba activate screensift
python -m pip install -e ".[workflow]"
```

## License

MIT — see [LICENSE](LICENSE).

## Design Choices

- Schema files are used because public screening datasets rarely share the same column names.
- RDKit canonicalization and InChIKey deduplication reduce duplicate and conflicting records.
- Similarity is computed automatically when active labels are available, so the package is not dependent on a precomputed similarity column.
- Structure-score columns are direction-aware, so Uni-Dock, GNINA, or external scoring tools can be plugged in without changing code.
- Weighted aggregation is available, but weights must be supplied by the user rather than inferred silently.
- Evidence buckets are categorical, not a weighted sum, so candidates can be retained for different scientific reasons.

## Limitations

- `ScreenSift` ranks candidates; it does not validate binding experimentally.
- Computed similarity requires active labels. Without known actives or a supplied similarity column, similarity evidence is unavailable.
- Structure evidence requires supplied score columns or scores generated by an external workflow.
- Min-max normalization is simple and interpretable, but it is sensitive to extreme score values.
- Candidate buckets depend on user-selected cutoffs.
- Docking, rescoring, ADMET modeling, selectivity profiling, MD, and free-energy calculations are outside the core `find_candidates(...)` API.

## Direct-Triage Mode

The main workflow above assumes that `ScreenSift` generates or helps generate structure-score columns through the included docking/rescoring modules. There is also a shortcut mode for users who already have a scored ligand table from another source.

In that case, the user can call only:

```python
find_candidates(
    schema="configs/my_dataset_schema.yml",
    data="my_scored_ligands.csv",
    target="MY_TARGET",
    evidence_mode="structure",
    structure_score_columns={"external_model_score": "higher"},
)
```

If no structure-score columns are provided in this direct mode, `ScreenSift` can still rank candidates from computed or supplied ligand-similarity evidence alone. That shortcut is useful for quick ligand-only triage, but it is not the full end-to-end screening workflow.
