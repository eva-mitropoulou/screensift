from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from screensift.common.io import ensure_dir, load_yaml


POSE_EXTENSIONS = {".sdf", ".pdbqt", ".pdb", ".mol2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locate pose files for selected inspection ligands.")
    parser.add_argument("--inspection-table", default="results/tables/mapk1_phase1_inspection_shortlist.csv")
    parser.add_argument("--inspection-config", default="configs/inspection.yml")
    parser.add_argument("--out-table", default="results/tables/mapk1_phase1_selected_pose_locations.csv")
    parser.add_argument("--transfer-manifest", default="results/reports/mapk1_phase1_selected_pose_transfer_manifest.txt")
    parser.add_argument("--report", default="results/reports/mapk1_phase1_selected_pose_location_report.md")
    return parser.parse_args()


def stable_ligand_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def discover_pose_files(search_dirs: list[str]) -> list[Path]:
    files: list[Path] = []
    for search_dir in search_dirs:
        root = Path(search_dir)
        if not root.exists():
            continue
        files.extend(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in POSE_EXTENSIONS)
    return sorted(set(files))


def pose_source(path: Path) -> str:
    text = str(path).lower()
    if "gnina" in text:
        return "gnina"
    if "unidock" in text:
        return "unidock"
    return "other"


def pose_priority(path: Path) -> tuple[int, str]:
    source = pose_source(path)
    priority = {"gnina": 0, "unidock": 1, "other": 2}.get(source, 3)
    return priority, str(path)


def locate_pose_files(inspection: pd.DataFrame, search_dirs: list[str]) -> pd.DataFrame:
    all_poses = discover_pose_files(search_dirs)
    rows: list[dict[str, Any]] = []
    if inspection.empty:
        return pd.DataFrame(
            columns=["ligand_id", "pose_found", "selected_pose_file", "pose_source", "all_candidate_pose_files", "transfer_needed", "notes"]
        )

    for ligand_id in inspection["ligand_id"].map(stable_ligand_id):
        candidates = [path for path in all_poses if ligand_id and ligand_id in path.name]
        candidates = sorted(candidates, key=pose_priority)
        selected = candidates[0] if candidates else None
        rows.append(
            {
                "ligand_id": ligand_id,
                "pose_found": selected is not None,
                "selected_pose_file": str(selected) if selected else "",
                "pose_source": pose_source(selected) if selected else "",
                "all_candidate_pose_files": ";".join(str(path) for path in candidates),
                "transfer_needed": selected is None,
                "notes": "" if selected else "No selected pose found on this machine; transfer from GPU results/poses may be needed.",
            }
        )
    return pd.DataFrame(rows)


def write_transfer_manifest(locations: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    missing = locations[~locations["pose_found"].astype(bool)] if not locations.empty else locations
    lines = [
        "# MAPK1 Phase 1 Selected Pose Transfer Manifest",
        "",
        f"Selected ligands: {len(locations)}",
        f"Pose files missing on this machine: {len(missing)}",
        "",
        "Missing ligand IDs:",
    ]
    lines.extend(f"- {ligand_id}" for ligand_id in missing.get("ligand_id", pd.Series(dtype=str)).astype(str))
    lines.extend(
        [
            "",
            "Use scripts/rsync_selected_poses_from_gpu_TEMPLATE.sh as a starting point if these poses remain on the GPU machine.",
            "Do not run the template unchanged; set GPU_USER, GPU_HOST, and GPU_REPO first.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_rsync_template(path: Path) -> None:
    ensure_dir(path.parent)
    content = """#!/usr/bin/env bash
set -euo pipefail

GPU_USER=<set_me>
GPU_HOST=<set_me>
GPU_REPO=/path/to/cadd-screen-triage
VM_REPO=/home/ubuntu/cadd-screen-triage

# Fill in GPU_USER/GPU_HOST/GPU_REPO before running.
# This script is a template and should not be run unchanged.

rsync -avP \\
  "${GPU_USER}@${GPU_HOST}:${GPU_REPO}/results/poses/" \\
  "${VM_REPO}/results/poses/"
"""
    path.write_text(content, encoding="utf-8")


def write_pose_location_report(locations: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    found = int(locations["pose_found"].sum()) if "pose_found" in locations.columns else 0
    missing = int(len(locations) - found)
    source_counts = locations["pose_source"].replace("", "missing").value_counts().to_dict() if "pose_source" in locations.columns else {}
    lines = [
        "# MAPK1 Phase 1 Selected Pose Location Report",
        "",
        f"- Inspection ligands: {len(locations)}",
        f"- Pose files found on this machine: {found}",
        f"- Pose files missing / transfer needed: {missing}",
        f"- Pose source counts: {source_counts}",
        "",
        "Missing pose files do not invalidate the shortlist. They only mean selected pose files should be synced from the GPU machine before manual pose inspection.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.inspection_config)
    search_dirs = config.get("pose_sources", {}).get("search_dirs", ["results/poses"])
    inspection_path = Path(args.inspection_table)
    inspection = pd.read_csv(inspection_path) if inspection_path.exists() else pd.DataFrame(columns=["ligand_id"])
    locations = locate_pose_files(inspection, search_dirs)

    out_table = Path(args.out_table)
    ensure_dir(out_table.parent)
    locations.to_csv(out_table, index=False)
    write_transfer_manifest(locations, Path(args.transfer_manifest))
    write_pose_location_report(locations, Path(args.report))
    write_rsync_template(Path("scripts/rsync_selected_poses_from_gpu_TEMPLATE.sh"))
    print(f"Pose locations written: {out_table}; found={int(locations['pose_found'].sum()) if not locations.empty else 0}")


if __name__ == "__main__":
    main()
