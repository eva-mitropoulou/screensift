from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from screensift.common.io import ensure_dir, markdown_table
from screensift.validation.pose_io import build_pose_index, find_receptor_for_pose, resolve_pose_path, stable_ligand_id


TIER_A_LIGAND_IDS = ("26747800", "864048", "844054", "26749352", "4244861")
OLD_PANEL_DIRS = ("pml", "png", "annotated_pml", "annotated_png")
REQUIRED_WORDING = (
    "Manual pose review does not validate binding or potency. It only decides whether a retrospective "
    "docking pose is plausible enough to show or use as an analog-prioritization seed."
)
VERDICT_COLUMNS = [
    "ligand_id",
    "inside_pocket",
    "plausible_contacts",
    "contact_distances_reasonable",
    "obvious_clash",
    "ligand_chemically_intact",
    "manual_pose_verdict",
    "analog_seed_decision",
    "pose_review_showcase",
    "manual_notes",
]


@dataclass(frozen=True)
class FinalReviewPaths:
    root: Path
    pml_dir: Path
    png_dir: Path
    pse_dir: Path


@dataclass(frozen=True)
class ViewSpec:
    suffix: str
    style: str
    y_turn_degrees: int | None = None


VIEW_SPECS = (
    ViewSpec("01_full_receptor", "full_receptor"),
    ViewSpec("02_clean_pocket", "clean_pocket"),
    ViewSpec("03_contact_only", "contact_only"),
    ViewSpec("04_contact_y90", "contact_only", 90),
    ViewSpec("05_contact_y180", "contact_only", 180),
)


def read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(table_path)
    if "ligand_id" in df.columns:
        df["ligand_id"] = df["ligand_id"].map(stable_ligand_id)
    return df


def clean_comment(value: Any) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    return text.replace("\n", " ").replace(";", ",").strip()


def pml_quote(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def final_review_paths(root: str | Path = "results/figures/final_pymol_review") -> FinalReviewPaths:
    root_path = Path(root)
    paths = FinalReviewPaths(
        root=root_path,
        pml_dir=root_path / "pml",
        png_dir=root_path / "png",
        pse_dir=root_path / "pse",
    )
    ensure_dir(paths.pml_dir)
    ensure_dir(paths.png_dir)
    ensure_dir(paths.pse_dir)
    return paths


def archive_old_outputs(panel_root: str | Path = "results/figures/pose_review_panels") -> list[tuple[Path, Path]]:
    root = Path(panel_root)
    archive_root = root / "deprecated_old_panels"
    ensure_dir(archive_root)
    moved: list[tuple[Path, Path]] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for dirname in OLD_PANEL_DIRS:
        source = root / dirname
        if not source.exists():
            continue
        target = archive_root / dirname
        if target.exists():
            target = archive_root / f"{dirname}_{timestamp}"
        shutil.move(str(source), str(target))
        moved.append((source, target))
    return moved


def select_tier_a_rows(triage: pd.DataFrame, seed_table: pd.DataFrame) -> pd.DataFrame:
    sources = []
    if not triage.empty:
        sources.append(triage)
    if not seed_table.empty:
        sources.append(seed_table)
    if not sources:
        return pd.DataFrame({"ligand_id": list(TIER_A_LIGAND_IDS)})

    merged = pd.concat(sources, ignore_index=True, sort=False)
    merged["ligand_id"] = merged["ligand_id"].map(stable_ligand_id)
    rows: list[pd.Series] = []
    for ligand_id in TIER_A_LIGAND_IDS:
        matches = merged[merged["ligand_id"].eq(ligand_id)]
        if matches.empty:
            rows.append(pd.Series({"ligand_id": ligand_id}))
        else:
            tier_matches = matches[matches.get("triage_tier", pd.Series(index=matches.index, dtype=object)).eq("A_analog_seed")]
            rows.append((tier_matches if not tier_matches.empty else matches).iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def merge_pose_context(rows: pd.DataFrame, pose_locations: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or pose_locations.empty:
        return rows.copy()
    keep_cols = [col for col in pose_locations.columns if col == "ligand_id" or col not in rows.columns]
    return rows.merge(pose_locations[keep_cols].drop_duplicates("ligand_id"), on="ligand_id", how="left")


def candidate_executables() -> list[str]:
    candidates = [
        "pymol",
        "pymol-open-source",
        str(Path(sys.executable).parent / "pymol"),
    ]
    return list(dict.fromkeys(candidates))


def find_working_pymol() -> tuple[str | None, str]:
    last_error = "No PyMOL executable candidate was found."
    for candidate in candidate_executables():
        executable = candidate if Path(candidate).is_file() else shutil.which(candidate)
        if not executable:
            continue
        try:
            completed = subprocess.run([executable, "-cq"], check=False, capture_output=True, text=True, timeout=30)
        except Exception as exc:
            last_error = f"{candidate}: {exc}"
            continue
        if completed.returncode == 0:
            return executable, ""
        last_error = f"{candidate}: {(completed.stderr or completed.stdout).strip()[:500]}"
    return None, last_error


def run_pymol_script(executable: str, script_path: Path, log_dir: Path) -> tuple[bool, str]:
    ensure_dir(log_dir)
    log_path = log_dir / f"{script_path.stem}.log"
    try:
        completed = subprocess.run([executable, "-cq", str(script_path)], check=False, capture_output=True, text=True, timeout=240)
    except Exception as exc:
        log_path.write_text(str(exc), encoding="utf-8")
        return False, f"pymol_exception: {exc}"
    log_path.write_text((completed.stdout or "") + "\n" + (completed.stderr or ""), encoding="utf-8")
    if completed.returncode != 0:
        return False, f"pymol_failed: see {log_path}"
    return True, ""


def output_paths(ligand_id: str, spec: ViewSpec, paths: FinalReviewPaths) -> dict[str, Path]:
    stem = f"{ligand_id}_{spec.suffix}"
    return {
        "pml": paths.pml_dir / f"{stem}.pml",
        "png": paths.png_dir / f"{stem}.png",
        "pse": paths.pse_dir / f"{stem}.pse",
    }


def shared_scene_lines(ligand_id: str, receptor: Path, pose: Path) -> tuple[list[str], str, str]:
    ligand_obj = f"ligand_{ligand_id}"
    contact_obj = f"contact_res_{ligand_id}"
    lines = [
        "reinitialize",
        f"load {pml_quote(receptor)}, receptor",
        f"load {pml_quote(pose)}, {ligand_obj}",
        "",
        "hide everything",
        "show cartoon, receptor",
        f"show sticks, {ligand_obj}",
        f"select {contact_obj}, byres (receptor within 3.5 of {ligand_obj})",
        f"show sticks, {contact_obj}",
        "",
        "color gray80, receptor",
        f"color green, {ligand_obj}",
        f"color yellow, {contact_obj}",
        f"set stick_radius, 0.22, {ligand_obj}",
        f"set stick_radius, 0.14, {contact_obj}",
        "",
        "bg_color white",
        "set ray_opaque_background, on",
        "set orthoscopic, on",
        "hide labels, all",
    ]
    return lines, ligand_obj, contact_obj


def pml_text(row: pd.Series, receptor: Path, pose: Path, spec: ViewSpec, paths: FinalReviewPaths) -> str:
    ligand_id = stable_ligand_id(row.get("ligand_id"))
    out = output_paths(ligand_id, spec, paths)
    lines, ligand_obj, contact_obj = shared_scene_lines(ligand_id, receptor, pose)
    lines.insert(0, f"# MAPK1 final manual review panel for ligand {ligand_id}")
    lines.insert(1, f"# View: {spec.suffix}")
    lines.insert(2, f"# Triage tier: {clean_comment(row.get('triage_tier', ''))}")
    lines.insert(3, f"# Categories: {clean_comment(row.get('inspection_categories', ''))}")

    if spec.style == "full_receptor":
        lines.extend(
            [
                "orient receptor",
                "zoom receptor, 1.5",
            ]
        )
    elif spec.style == "clean_pocket":
        lines.extend(
            [
                f"orient {ligand_obj}",
                f"zoom {ligand_obj}, 9",
            ]
        )
    else:
        lines.extend(
            [
                f"distance polar_contacts_{ligand_id}, {ligand_obj}, {contact_obj}, 3.5, mode=2",
                f"color cyan, polar_contacts_{ligand_id}",
                f"set dash_radius, 0.08, polar_contacts_{ligand_id}",
                f"show labels, polar_contacts_{ligand_id}",
                f'label (name CA and {contact_obj}), "%s%s" % (resn, resi)',
                "set label_size, 14",
                "set label_color, black",
                f"orient {ligand_obj}",
                f"zoom {ligand_obj}, 7",
            ]
        )
        if spec.y_turn_degrees is not None:
            lines.append(f"turn y, {spec.y_turn_degrees}")

    lines.extend(
        [
            f"png {pml_quote(out['png'])}, 2400, 1800, dpi=300, ray=1",
            f"save {pml_quote(out['pse'])}",
            "",
        ]
    )
    return "\n".join(lines)


def write_manual_verdict_template(ligand_ids: list[str], out_path: str | Path) -> None:
    ensure_dir(Path(out_path).parent)
    rows = []
    for ligand_id in ligand_ids:
        row = {col: "unclear" for col in VERDICT_COLUMNS}
        row["ligand_id"] = ligand_id
        rows.append(row)
    pd.DataFrame(rows, columns=VERDICT_COLUMNS).to_csv(out_path, index=False)


def write_manifest(
    rows: list[dict[str, Any]],
    archived: list[tuple[Path, Path]],
    pymol_executable: str | None,
    pymol_message: str,
    out_path: str | Path,
) -> None:
    ensure_dir(Path(out_path).parent)
    pml_count = sum(1 for row in rows for key in row if key.endswith("_pml") and row[key])
    png_count = sum(1 for row in rows for key in row if key.endswith("_png") and row.get(key) and Path(str(row[key])).exists())
    lines = [
        "# MAPK1 Phase 1 Final PyMOL Review Manifest",
        "",
        REQUIRED_WORDING,
        "",
        "## Summary",
        "",
        f"- Tier A ligands requested: {', '.join(TIER_A_LIGAND_IDS)}",
        f"- Tier A ligands processed: {len(rows)}",
        f"- PML files generated: {pml_count}",
        f"- PNG files generated: {png_count}",
        f"- PyMOL executable: {pymol_executable or 'not available'}",
        f"- PyMOL status: {pymol_message or 'available'}",
        f"- Old visualization folders archived: {'yes' if archived else 'no'}",
        "",
        "## Archived Old Panel Folders",
        "",
    ]
    if archived:
        lines.extend(f"- `{src}` -> `{dst}`" for src, dst in archived)
    else:
        lines.append("- No old panel folders were present to archive.")

    lines.extend(
        [
            "",
            "## Generated Files",
            "",
        ]
    )
    if rows:
        display = pd.DataFrame(rows)
        lines.append(markdown_table(display))
    else:
        lines.append("No Tier A ligands were processed.")

    lines.extend(
        [
            "",
            "## Manual Review Instructions",
            "",
            "Open the PNGs under `results/figures/final_pymol_review/png/`.",
            "For each ligand, inspect `01_full_receptor`, `02_clean_pocket`, and the three contact-only views.",
            "Then fill `results/tables/mapk1_phase1_final_manual_pose_verdict.csv` with KEEP, BACKUP, or REJECT decisions.",
            "",
            "Use KEEP only when the ligand is inside the pocket, has plausible contacts, has no obvious clash, and remains useful as a low-similarity / structure-added-value case.",
        ]
    )
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def generate_final_review(
    triage_table: str | Path = "results/tables/mapk1_phase1_candidate_triage.csv",
    seed_table: str | Path = "results/tables/mapk1_phase1_step10_seed_ligands.csv",
    pose_locations_table: str | Path = "results/tables/mapk1_phase1_selected_pose_locations.csv",
    prepared_receptors_table: str | Path = "results/tables/mapk1_prepared_receptors.csv",
    out_root: str | Path = "results/figures/final_pymol_review",
    verdict_csv: str | Path = "results/tables/mapk1_phase1_final_manual_pose_verdict.csv",
    manifest: str | Path = "results/reports/mapk1_phase1_final_pymol_review_manifest.md",
    run_pymol: bool = True,
    archive_old: bool = True,
) -> tuple[list[dict[str, Any]], list[tuple[Path, Path]]]:
    archived = archive_old_outputs() if archive_old else []
    paths = final_review_paths(out_root)
    triage = read_table(triage_table)
    seeds = read_table(seed_table)
    locations = read_table(pose_locations_table)
    receptors = read_table(prepared_receptors_table)
    selected = merge_pose_context(select_tier_a_rows(triage, seeds), locations)
    pose_index = build_pose_index(["results/poses"])
    pymol_executable, pymol_message = find_working_pymol() if run_pymol else (None, "headless PyMOL disabled")
    log_dir = Path("results/reports/final_pymol_review_logs")
    manifest_rows: list[dict[str, Any]] = []

    for _, row in selected.iterrows():
        ligand_id = stable_ligand_id(row.get("ligand_id"))
        pose_path, pose_note = resolve_pose_path(row, pose_index=pose_index)
        receptor_path, receptor_note = find_receptor_for_pose(pd.Series({**dict(row), "selected_pose_file": str(pose_path or "")}), receptors)
        row_record: dict[str, Any] = {
            "ligand_id": ligand_id,
            "pose_file": str(pose_path or ""),
            "receptor_file": str(receptor_path or ""),
            "pymol_status": "not_attempted",
            "notes": "; ".join(note for note in [pose_note, receptor_note] if note),
        }

        if not pose_path or not receptor_path or not pose_path.exists() or not receptor_path.exists():
            row_record["pymol_status"] = "missing_pose_or_receptor"
            manifest_rows.append(row_record)
            continue

        for spec in VIEW_SPECS:
            out = output_paths(ligand_id, spec, paths)
            out["pml"].write_text(pml_text(row, receptor_path, pose_path, spec, paths), encoding="utf-8")
            row_record[f"{spec.suffix}_pml"] = str(out["pml"])
            row_record[f"{spec.suffix}_png"] = str(out["png"])

        if pymol_executable:
            statuses = []
            for spec in VIEW_SPECS:
                out = output_paths(ligand_id, spec, paths)
                ok, error = run_pymol_script(pymol_executable, out["pml"], log_dir)
                statuses.append(ok and out["png"].exists())
                if error:
                    row_record["notes"] = "; ".join(part for part in [row_record.get("notes", ""), f"{spec.suffix}: {error}"] if part)
            row_record["pymol_status"] = "generated" if all(statuses) else "partial_or_failed"
        else:
            row_record["pymol_status"] = "pml_only_pymol_unavailable"
            row_record["notes"] = "; ".join(part for part in [row_record.get("notes", ""), pymol_message] if part)

        manifest_rows.append(row_record)

    write_manual_verdict_template(list(TIER_A_LIGAND_IDS), verdict_csv)
    write_manifest(manifest_rows, archived, pymol_executable, pymol_message, manifest)
    return manifest_rows, archived


def main() -> None:
    rows, archived = generate_final_review()
    pml_count = sum(1 for row in rows for key in row if key.endswith("_pml") and row[key])
    png_count = sum(1 for row in rows for key in row if key.endswith("_png") and row.get(key) and Path(str(row[key])).exists())
    print(
        "Final PyMOL review: "
        f"ligands={len(rows)} pml={pml_count} png={png_count} old_archived={'yes' if archived else 'no'}"
    )


if __name__ == "__main__":
    main()
