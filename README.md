# ScreenSift


<p align="center">
  <img src="docs/assets/screensift-logo.png" alt="ScreenSift logo" width="220">
</p>

**Screen wider. Sift smarter.** ScreenSift is an end-to-end virtual-screening
triage pipeline that turns a raw ligand table into a transparent, ranked
candidate shortlist — packaged as a tested, pip-installable Python API.

I ran it on the LIT-PCBA **MAPK1** benchmark end to end using two validated
MAPK1 receptor setups, **4QTA** and **6SLG**:

- **61,971** ligands curated (RDKit canonicalization, InChIKey dedup, activity conflict quarantine)
- two receptor/box setups validated by native-ligand redocking
- **51,507** ligand score rows from the 4QTA/6SLG docking and GNINA rescoring set
- saved **ECFP4/Tanimoto similarity** for all 51,507 ligand rows
- triaged to a transparent **top-100** shortlist with 0.75 similarity / 0.25 structure weighting

...then packaged the whole thing behind one function, `find_candidates(...)`, and a
config-driven CLI, with **122 passing tests** and CI.

> ScreenSift ranks and *explains* candidates from heterogeneous evidence
> (chemical similarity + docking/rescoring). 
## Quickstart

```bash
pip install -e .                     # core find_candidates API 
pip install -e ".[workflow]"         # + docking/prep tools for the full pipeline

# Rank the shipped MAPK1 example table:
python example/mapk1/run_example.py
# or via the installed CLI / config-driven orchestrator:
screensift-run-pipeline --config example/mapk1/pipeline.yml
```

```python
from screensift import find_candidates

candidates = find_candidates(
    schema="example/mapk1/schema.yml",
    data="example/mapk1/expected_outputs/mapk1_2receptor_score_population.csv",
    target="MAPK1",
    evidence_mode="combined",
    similarity_score_column="ecfp4_active_similarity",
    candidate_aggregation="weighted_mean",
    candidate_weights={
        "similarity_score_norm": 0.75,
        "structure_score_norm": 0.25,
    },
    n_candidates=100,
)
```



## Scientific Aim

Virtual-screening projects often combine heterogeneous evidence: chemical similarity, docking scores, machine-learning rescoring, and manual pose-review information. These signals are not always commensurate. A weighted sum can hide useful candidates when one score is weak or missing.

`ScreenSift` is designed as a transparent candidate-selection layer. It asks:

> Given a ligand dataset and available prioritization signals, which compounds should be carried forward, and why?

Similarity and structure scores are treated as separate evidence used to explain why a ligand was prioritized. The output is not a claim of experimental activity. It is a structured shortlist with explicit evidence channels.



## Main Workflow

The intended end-to-end path is:

```text
dataset + schema + target
  -> molecular audit and canonical ligand table
  -> native-ligand redocking validation
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

`workflow.mode` controls how much work the pipeline runs. `ranking_only` reads an existing scored table and writes one candidate CSV. `full_screening` runs native redocking validation, ligand preparation, Uni-Dock, GNINA rescoring, validation checks, and final ranking. Redocking-based receptor filtering is optional and must be enabled explicitly.

`ranking.evidence_mode` controls how candidates are ranked:

| mode | meaning |
| --- | --- |
| `similarity` | rank only by ligand-similarity evidence |
| `structure` | rank only by docking/rescoring evidence |
| `combined` | rank by both evidence channels |

Use dry-run mode to inspect the planned work without executing external tools:

```bash
screensift-run-pipeline --config example/mapk1/pipeline.yml --dry-run
```

The final candidate selection step is exposed as a Python API:

```python
from screensift import find_candidates

candidates = find_candidates(
    schema="configs/my_dataset_schema.yml",
    data="example/mapk1/expected_outputs/mapk1_2receptor_score_population.csv",
    target="MY_TARGET",
    evidence_mode="combined",
    structure_aggregation="max",
    candidate_aggregation="weighted_mean",
    candidate_weights={
        "similarity_score_norm": 0.75,
        "structure_score_norm": 0.25,
    },
    structure_score_cutoff=0.70,
    similarity_score_cutoff=0.70,
    n_candidates=100,
)
```

If the user does not provide a similarity score column, ScreenSift computes ECFP4/Tanimoto similarity to known actives when active labels are available. If the user does not provide docking or rescoring score columns, the full pipeline can generate them by running Uni-Dock docking and GNINA rescoring. Standard Uni-Dock/GNINA score columns are then auto-detected with their known score direction; for custom external score columns, the user supplies whether higher or lower values are better.

The same operation is available from the command line:

```bash
screensift \
  --schema configs/my_dataset_schema.yml \
  --data example/mapk1/expected_outputs/mapk1_2receptor_score_population.csv \
  --target MY_TARGET \
  --evidence-mode combined \
  --structure-aggregation max \
  --candidate-aggregation weighted_mean \
  --candidate-weight similarity_score_norm:0.75 \
  --candidate-weight structure_score_norm:0.25 \
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

For standard Uni-Dock/GNINA columns, the user does not need to say whether higher or lower is better. If the user does not specify which structure-score columns to use, ScreenSift automatically looks for common Uni-Dock and GNINA column names and applies the correct score direction for each one:

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

Raw evidence scores are not compared directly. ScreenSift first handles the direction of each score, then rescales each score column independently to a 0-1 range.

For example, lower Uni-Dock or GNINA affinity values are treated as better, while higher GNINA CNN scores are treated as better. Computed Tanimoto similarity is already between 0 and 1; supplied similarity columns are also treated as higher-is-better and normalized.

After normalizing the structure-based scores (Uni-Dock score, GNINA CNNscore, GNINA CNNaffinity, and GNINA affinity), ScreenSift creates one overall structure evidence score per ligand: `structure_score_norm`. By default, this is the best normalized structure-based score for that ligand; the source column is recorded in `primary_structure_score`.

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

The similarity cutoff is a policy choice. A high value such as `0.70` favors analog-like candidates that resemble known actives. Users should tune `similarity_score_cutoff` and `structure_score_cutoff` based on the intended triage policy.

The evidence bucket is assigned as:

| condition | bucket |
| --- | --- |
| similarity passes, structure passes | `consensus_supported` |
| similarity passes, structure does not pass | `analog_supported` |
| structure passes, similarity does not pass | `structure_supported` |
| neither passes | `deprioritized` |

The user requests `n_candidates`, and `ScreenSift` returns the top `n_candidates` rows by the unified ranking score. Evidence buckets explain why each returned row ranked, they do not reserve slots for similarity-supported or structure-supported candidates.

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

The standard validation path for the Uni-Dock + GNINA workflow is:

1. Uni-Dock native-ligand redocking validates the receptor boxes and main docking setup before large ligand docking.
2. Optional native-redocking auto-tuning can try user-defined box/exhaustiveness attempts before giving up on a receptor. ScreenSift never invents these combinations; the user lists them in `redocking.native.auto_tune.attempts`.
3. By default, redocking is report-only: it tells the user which receptor setups passed or failed, but keeps all complete receptors for screening.
4. If `redocking.native.filter_receptors: true`, ScreenSift writes a redocking-passed docking-box table and screens only receptor setups that passed the configured RMSD threshold after the configured attempts.
5. GNINA score-only validation evaluates native, redocked, and displaced decoy poses to sanity-check the rescoring signal.
6. Optional candidate pose QC redocks selected candidates with GNINA and compares Uni-Dock-vs-GNINA predicted poses. This is a pose-agreement check, not experimental RMSD.

This keeps the roles clear: Uni-Dock is the high-throughput docking engine, GNINA is a rescoring and optional second-pose engine, and ScreenSift ranks normalized evidence channels.

The full workflow path is:

```text
schema + ligand table
  -> curated ligands
  -> Uni-Dock native redocking validation
  -> optional user-defined redocking auto-tune attempts
  -> optional receptor filtering from redocking pass/fail
  -> ligand preparation
  -> Uni-Dock docking
  -> GNINA score-only rescoring and rescoring sanity checks
  -> find_candidates(...)
  -> candidate table
```

Native redocking auto-tuning is configured explicitly:

```yaml
redocking:
  native:
    run: true
    filter_receptors: true
    rmsd_threshold_angstrom: 2.0
    auto_tune:
      enabled: true
      attempts:
        - name: baseline
          exhaustiveness: 8
          box_scale: 1.0
          box_padding_angstrom: 0.0
        - name: wider_box
          exhaustiveness: 16
          box_scale: 1.2
          box_padding_angstrom: 0.0
        - name: wider_box_more_search
          exhaustiveness: 32
          box_scale: 1.2
          box_padding_angstrom: 2.0
```

For each receptor, ScreenSift tries those attempts in order and stops at the
first attempt that passes the RMSD threshold. Failed receptors remain visible in
the redocking report. The final one-row-per-receptor status is written to
`*_native_redocking.csv`, and every attempted combination is written to
`*_native_redocking_attempts.csv`. If filtering is enabled and no receptor
passes, the pipeline stops instead of screening against an invalid receptor set.

## Tests And Validation

Run the package tests with:

```bash
PYTHONPATH=src python -m pytest tests/
```

Current local result: 122 tests passed, with 2 upstream MDAnalysis/numpy warnings.

The current example workflow has also been run on the LIT-PCBA `MAPK1` target
using the two validated receptor setups, `4QTA` and `6SLG`. That run produced:

| stage | result |
| --- | --- |
| dataset audit | 61,971 curated ligands: 302 active and 61,669 inactive |
| screening set | full curated LIT-PCBA MAPK1 set: 61,971 ligands |
| receptor setup | two MAPK1 receptors used: 4QTA and 6SLG |
| score population | 51,507 ligand rows from existing 4QTA/6SLG docking and GNINA rescoring outputs |
| similarity evidence | ECFP4/Tanimoto similarity to known actives saved for all 51,507 ligand rows as `ecfp4_active_similarity` |
| complete score rows | 51,430 complete rows and 77 failed rows |
| score population labels | 262 known actives and 51,245 known inactives |
| ranking policy | `combined` evidence with `similarity_score_norm: 0.75` and `structure_score_norm: 0.25` |
| `find_candidates(...)` output | top 100 candidates: 6 known actives and 94 known inactives |

The 6/100 active count is the retrospective label composition of this ranked
shortlist. It is not presented as a prospective hit rate or a claim of new MAPK1
inhibitors. The example intentionally ranks one final candidate table from the
available similarity, Uni-Dock, and GNINA evidence.



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
