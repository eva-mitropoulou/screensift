PYTHON ?= python
UNIDOCK_BIN ?= unidock

TARGET ?= MY_TARGET
RUN_ID ?= pilot
SCHEMA ?= configs/dataset_schema.example.yml
N_JOBS ?= 16
TARGET_SLUG ?= $(shell printf '%s' '$(TARGET)' | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_' '_')
RUN_PREFIX ?= $(TARGET_SLUG)_$(RUN_ID)
TARGET_DIR ?= $(TARGET)

TABLES_DIR ?= results/tables
REPORTS_DIR ?= results/reports
FIGURES_DIR ?= results/figures
POSES_DIR ?= results/poses
SPLITS_DIR ?= data/processed/splits
LIGANDS_DIR ?= data/processed/ligands
RECEPTORS_DIR ?= data/processed/receptors

CURATED ?= $(TABLES_DIR)/$(TARGET_SLUG)_ligands_curated.csv
SCREENING_SET ?= $(SPLITS_DIR)/$(RUN_PREFIX)_screening_set.csv
SDF_DIR ?= $(LIGANDS_DIR)/$(TARGET_DIR)/$(RUN_ID)/sdf
PDBQT_DIR ?= $(LIGANDS_DIR)/$(TARGET_DIR)/$(RUN_ID)/pdbqt
DOCKING_BOXES ?= $(TABLES_DIR)/$(TARGET_SLUG)_docking_boxes.csv
UNIDOCK_POSE_DIR ?= $(POSES_DIR)/unidock/$(TARGET_DIR)/$(RUN_ID)

.PHONY: env-create install-dev check adapt-dataset split-dataset ligand-3d ligand-pdbqt docking-inputs unidock-screen parse-unidock unidock-qc find-candidates clean-results

env-create:
	mamba env create -f environment.yml

install-dev:
	$(PYTHON) -m pip install -e .

check:
	$(PYTHON) -m pytest tests/

adapt-dataset:
	$(PYTHON) -m screensift.curation.adapt_dataset_schema \
		--schema $(SCHEMA) \
		--target-slug $(TARGET_SLUG) \
		--out-audit $(TABLES_DIR)/$(TARGET_SLUG)_dataset_audit.csv \
		--out-curated $(CURATED) \
		--out-failures $(TABLES_DIR)/$(TARGET_SLUG)_curation_failures.csv \
		--manifest $(REPORTS_DIR)/$(TARGET_SLUG)_dataset_manifest.json

split-dataset:
	$(PYTHON) -m screensift.curation.make_screening_splits \
		--curated $(CURATED) \
		--screening-config configs/screening.yml \
		--out-dir $(SPLITS_DIR) \
		--report $(REPORTS_DIR)/$(RUN_PREFIX)_screening_splits_manifest.json \
		--target-slug $(TARGET_SLUG) \
		--run-id $(RUN_ID)

ligand-3d:
	$(PYTHON) -m screensift.ligands.generate_3d_conformers \
		--input $(SCREENING_SET) \
		--out-dir $(SDF_DIR) \
		--report $(REPORTS_DIR)/$(RUN_PREFIX)_3d_manifest.json \
		--failures $(TABLES_DIR)/$(RUN_PREFIX)_3d_failures.csv \
		--n-jobs $(N_JOBS)

ligand-pdbqt:
	$(PYTHON) -m screensift.ligands.prepare_pdbqt_ligands \
		--sdf-dir $(SDF_DIR) \
		--out-dir $(PDBQT_DIR) \
		--report $(REPORTS_DIR)/$(RUN_PREFIX)_pdbqt_manifest.json \
		--failures $(TABLES_DIR)/$(RUN_PREFIX)_pdbqt_failures.csv \
		--n-jobs $(N_JOBS)

docking-inputs:
	$(PYTHON) -m screensift.docking.collect_docking_inputs \
		--ligand-pdbqt-dir $(PDBQT_DIR) \
		--boxes $(DOCKING_BOXES) \
		--out $(TABLES_DIR)/$(RUN_PREFIX)_docking_inputs.csv \
		--output-root $(UNIDOCK_POSE_DIR)

unidock-screen:
	$(PYTHON) -m screensift.docking.run_unidock \
		--inputs $(TABLES_DIR)/$(RUN_PREFIX)_docking_inputs.csv \
		--out-root $(UNIDOCK_POSE_DIR) \
		--scores $(TABLES_DIR)/$(RUN_PREFIX)_unidock_raw.csv \
		--unidock-bin $(UNIDOCK_BIN)

parse-unidock:
	$(PYTHON) -m screensift.docking.parse_unidock_scores \
		--raw $(TABLES_DIR)/$(RUN_PREFIX)_unidock_raw.csv \
		--out $(TABLES_DIR)/$(RUN_PREFIX)_unidock_scores.csv

unidock-qc:
	$(PYTHON) -m screensift.docking.audit_unidock_scores \
		--scores $(TABLES_DIR)/$(RUN_PREFIX)_unidock_scores.csv \
		--splits $(SCREENING_SET) \
		--out-clean $(TABLES_DIR)/$(RUN_PREFIX)_unidock_scores_clean.csv \
		--out-flagged $(TABLES_DIR)/$(RUN_PREFIX)_unidock_score_flags.csv \
		--out-best $(TABLES_DIR)/$(RUN_PREFIX)_unidock_best_per_ligand.csv \
		--report $(REPORTS_DIR)/$(RUN_PREFIX)_unidock_score_qc.md

find-candidates:
	$(PYTHON) -m screensift.cli \
		--schema $(SCHEMA) \
		--target $(TARGET) \
		--n-candidates 100 \
		--out $(TABLES_DIR)/$(RUN_PREFIX)_candidates.csv

clean-results:
	rm -f $(TABLES_DIR)/* $(REPORTS_DIR)/*
