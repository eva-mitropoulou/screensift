# ScreenSift


<p align="center">
  <img src="docs/assets/screensift-logo.png" alt="ScreenSift logo" width="220">
</p>

ScreenSift is a Python package built around a practical virtual-screening problem: how do we combine 2D ligand-based evidence and 3D structure-based evidence into one ranked ligand table while still preserving the reason each compound was selected?

The workflow starts from a ligand table, curates molecular structures with RDKit, prepares docking inputs with Meeko, screens compounds with Uni-Dock, rescores docked poses with GNINA, and optionally adds ECFP4/Tanimoto similarity against known actives. The final output is a transparent triage table that ranks compounds while preserving the evidence behind each selection.

I validated ScreenSift on the LIT-PCBA MAPK1 benchmark using native-ligand redocking, two MAPK1 receptor setups, Uni-Dock virtual screening, GNINA rescoring, ligand-similarity scoring, and final candidate ranking. The details of that benchmark run are reported below.


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

## Main Workflow

The intended end-to-end path is:

```text
dataset + schema + target
  -> molecular audit and canonical ligand table
  -> receptor/box/native-ligand setup
  -> native-ligand redocking validation
  -> optional receptor filtering from redocking pass/fail
  -> ligand 3D/PDBQT preparation
  -> receptor/box-driven Uni-Dock screening
  -> optional GNINA rescoring
  -> ECFP4/Tanimoto similarity evidence
  -> normalized docking/rescoring and similarity evidence
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
    structure_score_cutoff=0.70,   # cutoff after normalization to 0-1
    similarity_score_cutoff=0.70,  # cutoff after normalization to 0-1
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

## How The Workflow Works

### 1. Dataset Schema And Curation

Different datasets use different column names. A ChEMBL export might use `molecule_chembl_id`, `canonical_smiles`, and `standard_value`; a local CSV might use `compound_id`, `smiles`, and `label`. ScreenSift uses a YAML schema to map those fields:

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

For ChEMBL-like data, labels can also be derived from activity thresholds:

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

The curation step reads SMILES, assigns ligand IDs and activity labels, canonicalizes molecules with RDKit, generates InChIKeys, computes basic descriptors, deduplicates by molecular identity, and writes explicit failure reasons for invalid or conflicting rows.

### 2. Native Redocking Before Screening

Before the many-ligand screen, ScreenSift can redock each receptor's known native ligand with Uni-Dock. This validates the receptor box and docking setup before spending time on candidate-library docking.

By default, native redocking is report-only: it tells the user which receptor setups passed or failed, but does not remove receptors from screening. If `redocking.native.filter_receptors: true`, ScreenSift writes a redocking-passed docking-box table and screens only receptor setups that pass the configured RMSD threshold.

Native redocking auto-tuning is optional and fully user-defined:

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

For each receptor, ScreenSift tries those attempts in order and stops at the first attempt that passes the RMSD threshold. Failed receptors remain visible in the redocking report. The final one-row-per-receptor status is written to `*_native_redocking.csv`, and every attempted combination is written to `*_native_redocking_attempts.csv`. If filtering is enabled and no receptor passes, the pipeline stops instead of screening against an invalid receptor set.

### 3. Ligand Preparation, Docking, And Rescoring

After redocking validation, the ligand-preparation stage generates 3D conformers from SMILES using RDKit ETKDGv3, adds hydrogens, optimizes with MMFF or UFF, and converts prepared ligand structures to PDBQT using Meeko.

The docking stage then forms receptor-ligand docking jobs from ligand PDBQT files and docking-box definitions, runs Uni-Dock, parses raw docking outputs into score tables, and checks score validity, pose-file availability, duplicates, and suspicious values.

The GNINA rescoring stage prepares valid Uni-Dock poses for GNINA score-only rescoring, runs GNINA, and parses outputs into tabular scores. GNINA score-only validation can also compare native, redocked, and displaced-decoy poses to sanity-check the rescoring signal.

This keeps the roles clear: Uni-Dock is the high-throughput docking engine, GNINA is the rescoring engine, and ScreenSift ranks the normalized evidence.

### 4. Similarity Evidence

If the input table already contains a similarity column, the user can pass it:

```python
find_candidates(..., similarity_score_column="ecfp4_active_similarity")
```

If no similarity column is supplied, ScreenSift computes ECFP4/Tanimoto similarity to known actives by generating Morgan fingerprints with radius 2 and 2048 bits, comparing each ligand to the active reference set, and excluding self-matches for active molecules. This produces `similarity_score_raw` and `similarity_score_norm`.

### 5. Structure-Score Evidence

Uni-Dock and GNINA produce structure-score columns such as `unidock_best_score`, `CNNscore`, `CNNaffinity`, and `gnina_affinity`. The final candidate-selection API is scoring-method agnostic, so users can also provide external numeric score columns.

For standard Uni-Dock/GNINA columns, the user does not need to say whether higher or lower is better. If the user does not specify which structure-score columns to use, ScreenSift automatically looks for common Uni-Dock and GNINA column names and applies the correct score direction:

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

### 6. Normalization And Ranking

Raw evidence scores are not compared directly. ScreenSift handles the direction of each score, then rescales each score column independently to a 0-1 range. Lower Uni-Dock or GNINA affinity values are treated as better, while higher GNINA CNN scores are treated as better. Computed Tanimoto similarity is already between 0 and 1; supplied similarity columns are also treated as higher-is-better evidence.

After normalizing the structure-based scores, ScreenSift creates one overall structure evidence score per ligand: `structure_score_norm`. By default, this is the best normalized structure-based score for that ligand; the source column is recorded in `primary_structure_score`.

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

The final candidate score depends on `evidence_mode`:

| `evidence_mode` | ranking behavior |
| --- | --- |
| `similarity` | rank only by ligand-similarity evidence |
| `structure` | rank only by docking/rescoring evidence |
| `combined` | rank by both evidence channels |

For `combined` mode, the default is:

```text
candidate_score = max(structure_score_norm, similarity_score_norm)
```

Users can instead request weighted candidate aggregation:

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

Weighted aggregation requires explicit user-supplied weights.

### 7. Evidence Buckets And Output

ScreenSift compares normalized scores against user-defined cutoffs:

```python
structure_score_cutoff=0.70   # cutoff after normalization to 0-1
similarity_score_cutoff=0.70  # cutoff after normalization to 0-1
```

The evidence bucket is assigned as:

| condition | bucket |
| --- | --- |
| similarity passes, structure passes | `consensus_supported` |
| similarity passes, structure does not pass | `analog_supported` |
| structure passes, similarity does not pass | `structure_supported` |
| neither passes | `deprioritized` |

The user requests `n_candidates`, and ScreenSift returns the top `n_candidates` rows sorted by `candidate_score`, then `structure_score_norm`, then `similarity_score_norm`. Evidence buckets explain why each returned row ranked; they do not reserve slots for similarity-supported or structure-supported candidates.

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
| `candidate_score` | final normalized ranking score |
| `evidence_bucket` | interpretable candidate-support category |

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
shortlist. The scored population contains 262 known actives among 51,507 rows
(`0.51%`), so a random top-100 sample would contain about 0.5 known actives on
average; this example shortlist is about `11.8x` enriched over that random
baseline. 


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
