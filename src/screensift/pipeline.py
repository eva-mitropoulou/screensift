from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from screensift.common.io import ensure_dir, load_yaml, write_json
from screensift.curation.adapt_dataset_schema import adapt_dataset
from screensift.curation.make_screening_splits import make_splits
from screensift.docking.audit_unidock_scores import audit_unidock_scores
from screensift.docking.collect_docking_inputs import collect_docking_inputs
from screensift.docking.parse_unidock_scores import parse_unidock_scores
from screensift.docking.run_unidock import run_unidock_inputs
from screensift.ligands.generate_3d_conformers import generate_conformers
from screensift.ligands.prepare_pdbqt_ligands import prepare_pdbqt_ligands
from screensift.rescoring.build_gnina_all_valid_input import build_gnina_all_valid_input
from screensift.rescoring.parse_gnina_scores import parse_gnina_scores
from screensift.rescoring.run_gnina_rescore import run_gnina_rescore
from screensift import find_candidates
from screensift.validation.run_candidate_pose_qc import candidate_pose_qc
from screensift.validation.run_native_redocking import redock_native_ligands
from screensift.validation.validate_gnina_rescoring import validate_gnina_rescoring


STAGES = [
    "adapt_dataset",
    "split_dataset",
    "native_redocking",
    "ligand_3d",
    "ligand_pdbqt",
    "docking_inputs",
    "unidock_screen",
    "parse_unidock",
    "unidock_qc",
    "gnina_input",
    "gnina_rescore",
    "parse_gnina",
    "gnina_validation",
    "rank_candidates",
    "candidate_pose_qc",
]


def get_nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def output_paths(config: dict[str, Any]) -> dict[str, Path]:
    target = str(config.get("target", "TARGET"))
    run_id = str(config.get("run_id", "run"))
    slug = str(config.get("target_slug", target.lower()))
    run_dir = Path(get_nested(config, "paths", "run_dir", default=f"runs/{slug}_{run_id}"))
    tables = ensure_dir(run_dir / "tables")
    reports = ensure_dir(run_dir / "reports")
    logs = ensure_dir(run_dir / "logs")
    poses = ensure_dir(run_dir / "poses")
    ligands = ensure_dir(run_dir / "ligands")
    score_table = str(get_nested(config, "ranking", "score_table", default="") or "").strip()
    paths = {
        "run_dir": run_dir,
        "tables": tables,
        "reports": reports,
        "logs": logs,
        "poses": poses,
        "ligands": ligands,
        "audit": tables / f"{slug}_dataset_audit.csv",
        "curated": tables / f"{slug}_ligands_curated.csv",
        "failures": tables / f"{slug}_curation_failures.csv",
        "dataset_manifest": reports / f"{slug}_dataset_manifest.json",
        "split": run_dir / "splits" / f"{slug}_{run_id}_screening_set.csv",
        "split_manifest": reports / f"{slug}_{run_id}_screening_splits_manifest.json",
        "sdf_dir": ligands / "sdf",
        "pdbqt_dir": ligands / "pdbqt",
        "conformer_manifest": reports / f"{slug}_{run_id}_3d_manifest.json",
        "conformer_failures": tables / f"{slug}_{run_id}_3d_failures.csv",
        "pdbqt_manifest": reports / f"{slug}_{run_id}_pdbqt_manifest.json",
        "pdbqt_failures": tables / f"{slug}_{run_id}_pdbqt_failures.csv",
        "docking_inputs": tables / f"{slug}_{run_id}_docking_inputs.csv",
        "unidock_raw": tables / f"{slug}_{run_id}_unidock_raw.csv",
        "unidock_scores": tables / f"{slug}_{run_id}_unidock_scores.csv",
        "unidock_clean": tables / f"{slug}_{run_id}_unidock_scores_clean.csv",
        "unidock_flags": tables / f"{slug}_{run_id}_unidock_score_flags.csv",
        "unidock_best": tables / f"{slug}_{run_id}_unidock_best_per_ligand.csv",
        "unidock_qc_report": reports / f"{slug}_{run_id}_unidock_score_qc.md",
        "native_redocking": tables / f"{slug}_{run_id}_native_redocking.csv",
        "native_redocking_report": reports / f"{slug}_{run_id}_native_redocking_report.md",
        "redocking_filtered_boxes": tables / f"{slug}_{run_id}_redocking_passed_boxes.csv",
        "gnina_input": tables / f"{slug}_{run_id}_gnina_input.csv",
        "gnina_raw": tables / f"{slug}_{run_id}_gnina_raw.csv",
        "gnina_scores": tables / f"{slug}_{run_id}_gnina_scores.csv",
        "gnina_validation": tables / f"{slug}_{run_id}_gnina_rescoring_validation.csv",
        "gnina_validation_report": reports / f"{slug}_{run_id}_gnina_rescoring_validation_report.md",
        "score_population": Path(score_table) if score_table else tables / f"{slug}_{run_id}_gnina_scores.csv",
        "score_population_enriched": tables / f"{slug}_{run_id}_score_population.csv",
        "candidates": tables / f"{slug}_{run_id}_candidates.csv",
        "candidate_pose_qc": tables / f"{slug}_{run_id}_candidate_pose_qc.csv",
        "candidate_pose_qc_report": reports / f"{slug}_{run_id}_candidate_pose_qc_report.md",
        "manifest": run_dir / "pipeline_manifest.json",
    }
    return paths


def selected_stages(config: dict[str, Any]) -> list[str]:
    explicit = get_nested(config, "pipeline", "stages", default=None)
    workflow_mode = str(get_nested(config, "workflow", "mode", default="full_screening")).strip().lower()
    disabled = set(get_nested(config, "pipeline", "skip_stages", default=[]) or [])
    optional_flags = {
        "native_redocking": bool(get_nested(config, "redocking", "native", "run", default=True)),
        "gnina_validation": bool(get_nested(config, "redocking", "gnina_score_sanity", "run", default=True)),
        "candidate_pose_qc": bool(get_nested(config, "candidate_pose_qc", "run", default=False)),
    }
    for stage, enabled in optional_flags.items():
        if not enabled:
            disabled.add(stage)
    if explicit:
        return [stage for stage in explicit if stage in STAGES and stage not in disabled]
    if workflow_mode == "ranking_only":
        return ["rank_candidates"]
    if workflow_mode not in {"full_screening", "full"}:
        raise ValueError("workflow.mode must be 'ranking_only' or 'full_screening'.")
    return [stage for stage in STAGES if stage not in disabled]


def _copy_missing_columns(frame: pd.DataFrame, source: str, aliases: list[str]) -> None:
    if source not in frame.columns:
        return
    for alias in aliases:
        if alias not in frame.columns:
            frame[alias] = frame[source]


def _merge_optional_metadata(
    frame: pd.DataFrame,
    metadata_path: Path,
    columns: list[str],
    on: list[str],
) -> pd.DataFrame:
    if not metadata_path.exists() or not set(on).issubset(frame.columns):
        return frame
    metadata = pd.read_csv(metadata_path, dtype={"ligand_id": str})
    if not set(on).issubset(metadata.columns):
        return frame
    selected = [column for column in columns if column in metadata.columns and column not in on]
    if not selected:
        return frame
    return frame.merge(metadata[on + selected].drop_duplicates(on), on=on, how="left", suffixes=("", "_metadata"))


def prepare_score_population(paths: dict[str, Path]) -> Path:
    """Build the single score table used by ranking and candidate pose QC."""
    score_path = paths["score_population"]
    frame = pd.read_csv(score_path, dtype={"ligand_id": str})

    frame = _merge_optional_metadata(
        frame,
        paths["unidock_best"],
        [
            "canonical_smiles",
            "inchikey",
            "best_score",
            "best_pdb_id",
            "best_receptor_pdbqt",
            "best_output_pose_file",
            "n_valid_receptors",
        ],
        ["ligand_id"],
    )
    frame = _merge_optional_metadata(
        frame,
        paths["gnina_input"],
        [
            "receptor_pdbqt",
            "receptor_clean_pdb",
            "output_pose_file",
            "gnina_receptor_input",
            "gnina_ligand_input",
            "gnina_output_file",
        ],
        ["ligand_id", "pdb_id"],
    )

    for left, right in [
        ("activity_label", "activity_label_metadata"),
        ("canonical_smiles", "canonical_smiles_metadata"),
        ("inchikey", "inchikey_metadata"),
    ]:
        if right in frame.columns:
            if left in frame.columns:
                frame[left] = frame[left].where(frame[left].notna(), frame[right])
            else:
                frame[left] = frame[right]

    _copy_missing_columns(frame, "canonical_smiles", ["smiles"])
    _copy_missing_columns(frame, "best_score_unidock", ["unidock_best_score"])
    _copy_missing_columns(frame, "best_score", ["unidock_best_score"])
    _copy_missing_columns(frame, "best_pdb_id", ["pdb_id"])
    _copy_missing_columns(frame, "cnnscore", ["gnina_cnnscore"])
    _copy_missing_columns(frame, "cnnaffinity", ["gnina_cnnaffinity"])
    _copy_missing_columns(frame, "CNN_VS", ["gnina_cnn_vs"])
    _copy_missing_columns(frame, "affinity", ["gnina_affinity"])

    frame = frame.drop(columns=[column for column in frame.columns if column.endswith("_metadata")])
    ensure_dir(paths["score_population_enriched"].parent)
    frame.to_csv(paths["score_population_enriched"], index=False)
    return paths["score_population_enriched"]


def redocking_gate_enabled(config: dict[str, Any]) -> bool:
    # Off by default: native redocking runs early and REPORTS, but does not
    # drop receptors until the user has validated the docking config against a
    # known pose. Set redocking.native.filter_receptors: true to gate the
    # many-ligand screen on the RMSD threshold once the setup is trusted.
    return bool(get_nested(config, "redocking", "native", "filter_receptors", default=False))


def docking_boxes_for_screening(config: dict[str, Any], paths: dict[str, Path]) -> Path:
    if redocking_gate_enabled(config):
        filtered = paths["redocking_filtered_boxes"]
        if not filtered.exists():
            # The gate is on but no filtered-boxes table was written. Rather
            # than silently screen every receptor (defeating the gate), fail
            # loudly: the native_redocking stage must run before this one.
            raise FileNotFoundError(
                "redocking.native.filter_receptors is enabled but "
                f"{filtered} was not written. Run the native_redocking stage "
                "before docking_inputs, or set filter_receptors: false."
            )
        return filtered
    return Path(get_nested(config, "docking", "boxes"))


def write_redocking_filtered_boxes(
    boxes_path: str | Path,
    redocking: pd.DataFrame,
    out_path: str | Path,
    require_pass: bool = True,
) -> pd.DataFrame:
    boxes = pd.read_csv(boxes_path)
    if "pdb_id" not in boxes.columns:
        raise ValueError("Docking boxes table must contain a pdb_id column.")
    if redocking.empty or "pdb_id" not in redocking.columns or "redocking_success" not in redocking.columns:
        passed_ids: set[str] = set()
    else:
        passed = redocking[redocking["redocking_success"].astype(str).str.lower().isin({"true", "1", "yes"})]
        passed_ids = set(passed["pdb_id"].astype(str).str.upper())
    filtered = boxes[boxes["pdb_id"].astype(str).str.upper().isin(passed_ids)].copy()
    if require_pass and filtered.empty:
        raise ValueError("Native redocking gate removed all receptors; no receptor passed the configured RMSD threshold.")
    out = Path(out_path)
    ensure_dir(out.parent)
    filtered.to_csv(out, index=False)
    return filtered


def run_pipeline(config_path: str | Path, dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    config = load_yaml(config_path)
    paths = output_paths(config)
    stages = selected_stages(config)
    manifest: dict[str, Any] = {
        "config": str(config_path),
        "target": config.get("target"),
        "run_id": config.get("run_id"),
        "dry_run": dry_run,
        "stages": {},
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    for stage in stages:
        if dry_run:
            manifest["stages"][stage] = {"status": "planned"}
            continue
        try:
            _run_stage(stage, config, paths, force=force)
            manifest["stages"][stage] = {"status": "complete"}
        except Exception as exc:
            manifest["stages"][stage] = {"status": "failed", "error": str(exc)}
            write_json(manifest, paths["manifest"])
            raise
        write_json(manifest, paths["manifest"])
    write_json(manifest, paths["manifest"])
    return manifest


def _run_stage(stage: str, config: dict[str, Any], paths: dict[str, Path], force: bool = False) -> None:
    target = str(config.get("target", "TARGET"))
    slug = str(config.get("target_slug", target.lower()))
    run_id = str(config.get("run_id", "run"))
    if stage == "adapt_dataset":
        adapt_dataset(get_nested(config, "dataset", "schema"), slug, paths["audit"], paths["curated"], paths["failures"], paths["dataset_manifest"])
    elif stage == "split_dataset":
        make_splits(paths["curated"], get_nested(config, "screening", "config", default="configs/screening.yml"), paths["split"].parent, paths["split_manifest"], target_slug=slug, run_id=run_id)
    elif stage == "ligand_3d":
        generate_conformers(paths["split"], paths["sdf_dir"], paths["conformer_manifest"], paths["conformer_failures"], n_jobs=int(get_nested(config, "ligands", "n_jobs", default=1)), limit=get_nested(config, "limits", "ligands", default=None), force=force, timeout_seconds=get_nested(config, "ligands", "conformer_timeout_seconds", default=None))
    elif stage == "ligand_pdbqt":
        prepare_pdbqt_ligands(paths["sdf_dir"], paths["pdbqt_dir"], paths["pdbqt_manifest"], paths["pdbqt_failures"], n_jobs=int(get_nested(config, "ligands", "n_jobs", default=1)), force=force)
    elif stage == "docking_inputs":
        collect_docking_inputs(paths["pdbqt_dir"], docking_boxes_for_screening(config, paths), paths["docking_inputs"], limit=get_nested(config, "limits", "ligands", default=None), single_receptor=bool(get_nested(config, "limits", "single_receptor", default=False)), output_root=paths["poses"] / "unidock")
    elif stage == "unidock_screen":
        run_unidock_inputs(paths["docking_inputs"], paths["poses"] / "unidock", paths["unidock_raw"], unidock_bin=get_nested(config, "docking", "unidock_bin", default="unidock"), exhaustiveness=int(get_nested(config, "docking", "exhaustiveness", default=8)), max_step=int(get_nested(config, "docking", "max_step", default=10)), num_modes=int(get_nested(config, "docking", "num_modes", default=10)), energy_range=float(get_nested(config, "docking", "energy_range", default=3)), cpu=int(get_nested(config, "docking", "cpu", default=1)), log_dir=paths["logs"] / "unidock")
    elif stage == "parse_unidock":
        parse_unidock_scores(paths["unidock_raw"], paths["unidock_scores"])
    elif stage == "unidock_qc":
        audit_unidock_scores(paths["unidock_scores"], paths["split"], paths["unidock_clean"], paths["unidock_flags"], paths["unidock_best"], paths["unidock_qc_report"])
    elif stage == "native_redocking":
        redocking = redock_native_ligands(get_nested(config, "docking", "boxes"), paths["native_redocking"], paths["native_redocking_report"], paths["poses"] / "native_redocking", paths["logs"] / "native_redocking", unidock_bin=get_nested(config, "docking", "unidock_bin", default="unidock"), rmsd_threshold_angstrom=float(get_nested(config, "redocking", "native", "rmsd_threshold_angstrom", default=2.0)), cpu=int(get_nested(config, "docking", "cpu", default=1)), exhaustiveness=int(get_nested(config, "docking", "exhaustiveness", default=8)), num_modes=int(get_nested(config, "docking", "num_modes", default=10)), energy_range=float(get_nested(config, "docking", "energy_range", default=3)))
        if redocking_gate_enabled(config):
            write_redocking_filtered_boxes(get_nested(config, "docking", "boxes"), redocking, paths["redocking_filtered_boxes"], require_pass=True)
    elif stage == "gnina_input":
        build_gnina_all_valid_input(paths["unidock_best"], paths["gnina_input"], paths["reports"] / "gnina_input_report.md", receptor_root=get_nested(config, "paths", "receptor_root"), gnina_output_root=paths["poses"] / "gnina")
    elif stage == "gnina_rescore":
        run_gnina_rescore(
            paths["gnina_input"],
            paths["gnina_raw"],
            score_only=True,
            gnina_bin=get_nested(config, "gnina", "binary", default="gnina"),
            log_dir=paths["logs"] / "gnina",
            converted_ligand_root=paths["poses"] / "gnina_inputs",
            timeout_seconds=int(get_nested(config, "gnina", "timeout_seconds", default=1800)),
            resume=bool(get_nested(config, "gnina", "resume", default=True)),
            checkpoint_every=int(get_nested(config, "gnina", "checkpoint_every", default=50)),
            max_workers=int(get_nested(config, "gnina", "max_workers", default=1)),
            gnina_mode=get_nested(config, "gnina", "mode", default="native"),
            gnina_image=get_nested(config, "gnina", "image"),
            gpu=get_nested(config, "gnina", "gpu"),
            docker_args=get_nested(config, "gnina", "docker_args", default=[]),
            extra_args=get_nested(config, "gnina", "extra_args", default=[]),
        )
    elif stage == "parse_gnina":
        parse_gnina_scores(paths["gnina_raw"], paths["gnina_scores"])
    elif stage == "gnina_validation":
        validate_gnina_rescoring(paths["native_redocking"], paths["gnina_validation"], paths["gnina_validation_report"], paths["logs"] / "gnina_validation", gnina_mode=get_nested(config, "gnina", "mode", default="native"), gnina_bin=get_nested(config, "gnina", "binary", default="gnina"), gnina_image=get_nested(config, "gnina", "image"), gpu=get_nested(config, "gnina", "gpu"), docker_args=get_nested(config, "gnina", "docker_args", default=[]), extra_args=get_nested(config, "gnina", "extra_args", default=[]), decoy_translation_angstrom=float(get_nested(config, "redocking", "gnina_score_sanity", "decoy_translation_angstrom", default=12.0)))
    elif stage == "rank_candidates":
        score_table = prepare_score_population(paths)
        ranking = config.get("ranking", {})
        candidates = find_candidates(
            schema=get_nested(config, "dataset", "schema"),
            data=score_table,
            target=target,
            evidence_mode=ranking.get("evidence_mode", "combined"),
            structure_score_columns=ranking.get("structure_score_columns"),
            similarity_score_column=ranking.get("similarity_score_column"),
            structure_aggregation=ranking.get("structure_aggregation", "max"),
            structure_weights=ranking.get("structure_weights"),
            candidate_aggregation=ranking.get("candidate_aggregation", "max"),
            candidate_weights=ranking.get("candidate_weights"),
            n_candidates=int(ranking.get("n_candidates", 100)),
        )
        ensure_dir(paths["candidates"].parent)
        candidates.to_csv(paths["candidates"], index=False)
    elif stage == "candidate_pose_qc":
        score_table = prepare_score_population(paths)
        candidate_pose_qc(paths["candidates"], score_table, get_nested(config, "docking", "boxes"), paths["candidate_pose_qc"], paths["candidate_pose_qc_report"], paths["poses"] / "candidate_gnina", paths["logs"] / "candidate_pose_qc", top_n=int(get_nested(config, "candidate_pose_qc", "top_n", default=25)), gnina_mode=get_nested(config, "gnina", "mode", default="native"), gnina_bin=get_nested(config, "gnina", "binary", default="gnina"), gnina_image=get_nested(config, "gnina", "image"), gpu=get_nested(config, "gnina", "gpu"), docker_args=get_nested(config, "gnina", "docker_args", default=[]), extra_args=get_nested(config, "gnina", "extra_args", default=[]), converted_ligand_root=paths["poses"] / "gnina_inputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ScreenSift end-to-end pipeline from a YAML config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_pipeline(args.config, dry_run=args.dry_run, force=args.force)
    print(f"Pipeline manifest: {manifest['outputs']['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
