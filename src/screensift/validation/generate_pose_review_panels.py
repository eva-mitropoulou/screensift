from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from screensift.common.io import ensure_dir, load_yaml, markdown_table
from screensift.validation.pose_io import build_pose_index, find_receptor_for_pose, resolve_pose_path, stable_ligand_id


@dataclass(frozen=True)
class PanelPaths:
    output_dir: Path
    pml_dir: Path
    png_dir: Path


VIEW_NAMES = ("full_receptor", "pocket_overview", "interactions")
LEGACY_VIEW_NAMES = ("overview",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pose-review PyMOL panels for MAPK1 inspection ligands.")
    parser.add_argument("--pose-panel-config", default="configs/pose_panel.yml")
    parser.add_argument("--triage-table", default="results/tables/mapk1_phase1_candidate_triage.csv")
    parser.add_argument("--seed-table", default="results/tables/mapk1_phase1_step10_seed_ligands.csv")
    parser.add_argument("--pose-locations", default="results/tables/mapk1_phase1_selected_pose_locations.csv")
    parser.add_argument("--prepared-receptors", default="results/tables/mapk1_prepared_receptors.csv")
    parser.add_argument("--inspection-summary", default="results/tables/mapk1_phase1_pose_inspection_summary.csv")
    parser.add_argument("--out-report", default="results/reports/mapk1_phase1_pose_review_panel_manifest.md")
    return parser.parse_args()


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
    return text.replace(";", ", ").replace("\n", " ").strip()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def numeric_value(value: Any, default: float = 0.0) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(default if pd.isna(parsed) else parsed)


def pml_quote(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def _category_contains(row: pd.Series, needle: str) -> bool:
    return needle in str(row.get("inspection_categories", ""))


def _has_anomaly(row: pd.Series) -> bool:
    flags = str(row.get("anomaly_flags", "") or "").strip().lower()
    return bool(flags and flags != "nan")


def select_panel_ligands(triage: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if triage.empty:
        return pd.DataFrame()
    panel_cfg = config.get("pose_panel", {})
    include_tiers = set(panel_cfg.get("include_tiers", ["A_analog_seed"]))
    controls = panel_cfg.get("include_controls", {})
    work = triage.copy()
    work["ligand_id"] = work["ligand_id"].map(stable_ligand_id)
    work["_selection_category"] = ""

    selected_parts: list[pd.DataFrame] = []
    seed_rows = work[work.get("triage_tier", pd.Series(index=work.index, dtype=object)).isin(include_tiers)].copy()
    seed_rows["_selection_category"] = "tier_a_seed"
    selected_parts.append(seed_rows)

    fp_n = int(controls.get("false_positive_cases", 0) or 0)
    fn_n = int(controls.get("false_negative_cases", 0) or 0)
    anomaly_n = int(controls.get("score_anomaly_cases", 0) or 0)

    if fp_n:
        fp = work[
            work.apply(lambda row: _category_contains(row, "consensus_inactive_false_positive"), axis=1)
            | work.get("recommended_action", pd.Series(index=work.index, dtype=object)).fillna("").eq("false_positive_failure_case")
        ].copy()
        fp = fp.sort_values(["ecfp4_active_similarity", "n_total_interactions"], ascending=[False, False], na_position="last").head(fp_n)
        fp["_selection_category"] = "false_positive_control"
        selected_parts.append(fp)

    if fn_n:
        fn = work[
            work.apply(lambda row: _category_contains(row, "active_false_negative"), axis=1)
            | work.get("recommended_action", pd.Series(index=work.index, dtype=object)).fillna("").eq("false_negative_failure_case")
        ].copy()
        fn = fn.sort_values(["n_total_interactions", "ecfp4_active_similarity"], ascending=[False, True], na_position="last").head(fn_n)
        fn["_selection_category"] = "false_negative_control"
        selected_parts.append(fn)

    if anomaly_n:
        anomalies = work[work.apply(_has_anomaly, axis=1)].copy()
        anomalies["_severe_score"] = anomalies["anomaly_flags"].fillna("").astype(str).str.contains(
            "extreme_unidock_negative|extreme_positive_gnina_affinity|suspicious_unidock_extreme_negative|suspicious_ligand_efficiency_extreme",
            regex=True,
        )
        anomalies = anomalies.sort_values(["_severe_score", "n_total_interactions"], ascending=[False, False], na_position="last").head(anomaly_n)
        anomalies["_selection_category"] = "score_anomaly_control"
        selected_parts.append(anomalies.drop(columns=["_severe_score"], errors="ignore"))

    if not selected_parts:
        return pd.DataFrame()
    selected = pd.concat(selected_parts, ignore_index=True)
    selected["_selection_order"] = range(len(selected))
    selected = selected.sort_values("_selection_order").drop_duplicates("ligand_id", keep="first")
    return selected.drop(columns=["_selection_order"], errors="ignore").reset_index(drop=True)


def _merge_context(selected: pd.DataFrame, pose_locations: pd.DataFrame, inspection_summary: pd.DataFrame) -> pd.DataFrame:
    work = selected.copy()
    if not pose_locations.empty:
        cols = [col for col in pose_locations.columns if col == "ligand_id" or col not in work.columns]
        work = work.merge(pose_locations[cols].drop_duplicates("ligand_id"), on="ligand_id", how="left")
    if not inspection_summary.empty:
        cols = [col for col in inspection_summary.columns if col == "ligand_id" or col not in work.columns]
        work = work.merge(inspection_summary[cols].drop_duplicates("ligand_id"), on="ligand_id", how="left")
    return work


def _panel_paths(config: dict[str, Any]) -> PanelPaths:
    panel_cfg = config.get("pose_panel", {})
    output_dir = Path(panel_cfg.get("output_dir", "results/figures/pose_review_panels"))
    pml_dir = Path(panel_cfg.get("pml_dir", output_dir / "pml"))
    png_dir = Path(panel_cfg.get("png_dir", output_dir / "png"))
    ensure_dir(output_dir)
    ensure_dir(pml_dir)
    ensure_dir(png_dir)
    return PanelPaths(output_dir=output_dir, pml_dir=pml_dir, png_dir=png_dir)


def _scores_comment(row: pd.Series) -> str:
    return (
        f"Uni-Dock={row.get('unidock_best_score', '')}, CNNscore={row.get('CNNscore', '')}, "
        f"CNNaffinity={row.get('CNNaffinity', '')}, GNINA_affinity={row.get('gnina_affinity', '')}"
    )


def pml_text(row: pd.Series, receptor: Path, pose: Path, config: dict[str, Any], view: str, paths: PanelPaths) -> str:
    ligand_id = stable_ligand_id(row.get("ligand_id"))
    ligand_obj = f"ligand_{ligand_id}"
    pocket_obj = f"pocket_{ligand_id}"
    panel_cfg = config.get("pose_panel", {})
    image_cfg = panel_cfg.get("image", {})
    view_cfg = panel_cfg.get("views", {}).get(view, {})
    styling = panel_cfg.get("styling", {})
    width = int(image_cfg.get("width", 2400))
    height = int(image_cfg.get("height", 1800))
    dpi = int(image_cfg.get("dpi", 300))
    ray = 1 if parse_bool(image_cfg.get("ray", True)) else 0
    bg = image_cfg.get("background", "white")
    opaque = "on" if parse_bool(image_cfg.get("opaque_background", True)) else "off"
    cutoff = float(view_cfg.get("pocket_cutoff_angstrom", 4.0))
    zoom_buffer = float(view_cfg.get("ligand_zoom_buffer", 12 if view == "pocket_overview" else 8))
    surface_transparency = float(view_cfg.get("receptor_surface_transparency", 0.80))
    polar_cutoff = float(view_cfg.get("polar_contact_cutoff_angstrom", 3.5))
    show_surface = parse_bool(view_cfg.get("show_surface", view == "pocket_overview"))
    label_pocket_residues = parse_bool(view_cfg.get("label_pocket_residues", view == "interactions"))
    hide_pocket_cartoon = parse_bool(view_cfg.get("hide_pocket_cartoon", view == "interactions"))
    receptor_color = styling.get("receptor_color", "gray80")
    ligand_color = styling.get("ligand_color", "green")
    pocket_color = styling.get("pocket_color", "yellow")
    contact_color = styling.get("contact_color", "black")
    label_color = styling.get("label_color", "black")
    label_size = int(styling.get("label_size", 16))
    cartoon_transparency = float(styling.get("cartoon_transparency", 0.0))
    ligand_stick_radius = float(styling.get("stick_radius_ligand", 0.20))
    pocket_stick_radius = float(styling.get("stick_radius_pocket", 0.14))
    dash_radius = float(styling.get("dash_radius", 0.08))
    png_path = paths.png_dir / f"{ligand_id}_{view}.png"
    pse_path = paths.output_dir / f"{ligand_id}_{view}.pse"
    lines = [
        f"# MAPK1 pose-review panel for ligand {ligand_id}",
        f"# Selection category: {clean_comment(row.get('_selection_category', ''))}",
        f"# Triage tier: {clean_comment(row.get('triage_tier', ''))}",
        f"# Inspection categories: {clean_comment(row.get('inspection_categories', ''))}",
        f"# Scores: {clean_comment(_scores_comment(row))}",
        "reinitialize",
        f"load {pml_quote(receptor)}, receptor",
        f"load {pml_quote(pose)}, {ligand_obj}",
        "",
        "hide everything",
        "show cartoon, receptor",
        f"set cartoon_transparency, {cartoon_transparency:.2f}, receptor",
        f"show sticks, {ligand_obj}",
        f"select {pocket_obj}, byres (receptor within {cutoff:.1f} of {ligand_obj})",
        f"show sticks, {pocket_obj}",
        f"set stick_radius, {ligand_stick_radius:.2f}, {ligand_obj}",
        f"set stick_radius, {pocket_stick_radius:.2f}, {pocket_obj}",
    ]
    if show_surface:
        lines.extend(
            [
                "show surface, receptor",
                f"set transparency, {surface_transparency:.2f}, receptor",
            ]
        )
    if view == "interactions":
        if hide_pocket_cartoon:
            lines.extend(
                [
                    f"hide cartoon, {pocket_obj}",
                    f"hide lines, {pocket_obj}",
                ]
            )
        lines.extend(
            [
                f"distance polar_contacts_{ligand_id}, {ligand_obj}, {pocket_obj}, {polar_cutoff:.1f}, mode=2",
                f"hide labels, polar_contacts_{ligand_id}",
                f"color {contact_color}, polar_contacts_{ligand_id}",
                f"set dash_radius, {dash_radius:.2f}",
            ]
        )
        if label_pocket_residues:
            lines.extend(
                [
                    f'label (name CA and {pocket_obj}), "%s%s" % (resn, resi)',
                    f"set label_size, {label_size}",
                    f"set label_color, {label_color}",
                ]
            )
    lines.extend(
        [
            f"color {receptor_color}, receptor",
            f"color {ligand_color}, {ligand_obj}",
            f"color {pocket_color}, {pocket_obj}",
            f"bg_color {bg}",
            f"set ray_opaque_background, {opaque}",
            "set orthoscopic, on",
        ]
    )
    if view == "full_receptor":
        lines.extend(["orient receptor", "zoom receptor, 1.5"])
    else:
        lines.extend([f"orient {ligand_obj}", f"zoom {ligand_obj}, {zoom_buffer:g}"])
    lines.extend(
        [
            f"png {pml_quote(png_path)}, {width}, {height}, dpi={dpi}, ray={ray}",
            f"save {pml_quote(pse_path)}",
            "",
        ]
    )
    return "\n".join(lines)


def output_paths_for_ligand(ligand_id: str, paths: PanelPaths, view: str) -> dict[str, Path]:
    return {
        "pml": paths.pml_dir / f"{ligand_id}_{view}.pml",
        "png": paths.png_dir / f"{ligand_id}_{view}.png",
        "pse": paths.output_dir / f"{ligand_id}_{view}.pse",
    }


def cleanup_existing_panel_outputs(ligand_id: str, paths: PanelPaths) -> None:
    for view in (*VIEW_NAMES, *LEGACY_VIEW_NAMES):
        for output_path in output_paths_for_ligand(ligand_id, paths, view).values():
            output_path.unlink(missing_ok=True)


def _candidate_executables(config: dict[str, Any]) -> list[str]:
    panel_cfg = config.get("pose_panel", {})
    configured = panel_cfg.get("pymol", {}).get("executable_candidates", ["pymol", "pymol-open-source"])
    candidates: list[str] = []
    env_bin = os.environ.get("PYMOL_BIN")
    if env_bin:
        candidates.append(env_bin)
    candidates.extend(str(candidate) for candidate in configured)
    candidates.append(str(Path(sys.executable).parent / "pymol"))
    return list(dict.fromkeys(candidates))


def find_working_pymol(config: dict[str, Any]) -> tuple[str | None, str]:
    for candidate in _candidate_executables(config):
        executable = candidate if Path(candidate).is_file() else shutil.which(candidate)
        if not executable:
            continue
        try:
            completed = subprocess.run(
                [executable, "-cq"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception as exc:
            last_error = f"{candidate}: {exc}"
            continue
        if completed.returncode == 0:
            return executable, ""
        last_error = f"{candidate}: {(completed.stderr or completed.stdout).strip()[:500]}"
    return None, locals().get("last_error", "No PyMOL executable candidate was found.")


def run_pymol_script(executable: str, script_path: Path, log_dir: Path) -> tuple[bool, str]:
    ensure_dir(log_dir)
    log_path = log_dir / f"{script_path.stem}.log"
    try:
        completed = subprocess.run(
            [executable, "-cq", str(script_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=240,
        )
    except Exception as exc:
        log_path.write_text(str(exc), encoding="utf-8")
        return False, f"pymol_exception: {exc}"
    log_path.write_text((completed.stdout or "") + "\n" + (completed.stderr or ""), encoding="utf-8")
    if completed.returncode != 0:
        return False, f"pymol_failed: see {log_path}"
    return True, ""


def build_panel_rows(
    triage: pd.DataFrame,
    pose_locations: pd.DataFrame,
    prepared_receptors: pd.DataFrame,
    inspection_summary: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    selected = select_panel_ligands(triage, config)
    if selected.empty:
        return pd.DataFrame()
    work = _merge_context(selected, pose_locations, inspection_summary)
    pose_index = build_pose_index(["results/poses"])
    rows: list[dict[str, Any]] = []
    paths = _panel_paths(config)
    panel_cfg = config.get("pose_panel", {})
    run_headless = parse_bool(panel_cfg.get("pymol", {}).get("run_headless_if_available", True))
    pymol_exe, pymol_warning = find_working_pymol(config) if run_headless else (None, "headless PyMOL disabled")
    log_dir = Path("results/reports/pose_review_panel_logs")

    for _, row in work.iterrows():
        ligand_id = stable_ligand_id(row.get("ligand_id"))
        pose_path, pose_note = resolve_pose_path(row, pose_index=pose_index)
        receptor_path, receptor_note = find_receptor_for_pose(pd.Series({**dict(row), "selected_pose_file": str(pose_path or "")}), prepared_receptors)
        cleanup_existing_panel_outputs(ligand_id, paths)
        view_paths = {view: output_paths_for_ligand(ligand_id, paths, view) for view in VIEW_NAMES}
        pml_generated = False
        png_status = "not_attempted"
        png_files_generated = 0
        notes: list[str] = []
        if pose_note:
            notes.append(pose_note)
        if receptor_note:
            notes.append(receptor_note)
        if pose_path and receptor_path and pose_path.exists() and receptor_path.exists():
            for view in VIEW_NAMES:
                view_paths[view]["pml"].write_text(pml_text(row, receptor_path, pose_path, config, view, paths), encoding="utf-8")
            pml_generated = True
            if pymol_exe:
                view_success: list[bool] = []
                for view in VIEW_NAMES:
                    ok, view_error = run_pymol_script(pymol_exe, view_paths[view]["pml"], log_dir)
                    png_exists = view_paths[view]["png"].exists()
                    view_success.append(ok and png_exists)
                    if png_exists:
                        png_files_generated += 1
                    if view_error:
                        notes.append(f"{view}: {view_error}")
                png_status = "generated" if all(view_success) else "partial_or_failed"
            else:
                png_status = "skipped_pymol_unavailable"
                notes.append(pymol_warning)
        else:
            notes.append("missing_pose_or_receptor")
        rows.append(
            {
                "ligand_id": ligand_id,
                "selection_category": row.get("_selection_category", ""),
                "triage_tier": row.get("triage_tier", ""),
                "activity_label": row.get("activity_label", ""),
                "inspection_categories": row.get("inspection_categories", ""),
                "pose_file": str(pose_path or ""),
                "receptor_file": str(receptor_path or ""),
                "full_receptor_pml": str(view_paths["full_receptor"]["pml"]),
                "full_receptor_png": str(view_paths["full_receptor"]["png"]),
                "pocket_overview_pml": str(view_paths["pocket_overview"]["pml"]),
                "pocket_overview_png": str(view_paths["pocket_overview"]["png"]),
                "interactions_pml": str(view_paths["interactions"]["pml"]),
                "interactions_png": str(view_paths["interactions"]["png"]),
                "pml_generated": pml_generated,
                "png_status": png_status,
                "png_files_generated": png_files_generated,
                "pymol_executable": pymol_exe or "",
                "caption_full_receptor": caption_for(row, "full_receptor"),
                "caption_pocket_overview": caption_for(row, "pocket_overview"),
                "caption_interactions": caption_for(row, "interactions"),
                "notes": "; ".join(note for note in notes if note),
            }
        )
    return pd.DataFrame(rows)


def caption_for(row: pd.Series, view: str) -> str:
    ligand_id = stable_ligand_id(row.get("ligand_id"))
    if view == "full_receptor":
        return (
            f"Ligand {ligand_id} shown in the context of the full MAPK1 receptor. "
            "This view highlights the overall binding-site location within the protein fold."
        )
    if view == "pocket_overview":
        return (
            f"Ligand {ligand_id} shown within the MAPK1 binding pocket with nearby residues highlighted. "
            "This is a retrospective pose-inspection view, not proof of binding."
        )
    return (
        f"Ligand {ligand_id} close-up view with nearby pocket residues and possible polar contacts. "
        "This panel is used for manual inspection of pose plausibility."
    )


def write_manifest(rows: pd.DataFrame, out_report: Path) -> None:
    ensure_dir(out_report.parent)
    if rows.empty:
        out_report.write_text("# MAPK1 Phase 1 Portfolio Pose Panel Manifest\n\nNo ligands were selected.\n", encoding="utf-8")
        return
    selected_ids = ", ".join(rows["ligand_id"].astype(str).tolist())
    table_cols = [
        "ligand_id",
        "selection_category",
        "triage_tier",
        "activity_label",
        "pose_file",
        "receptor_file",
        "full_receptor_pml",
        "full_receptor_png",
        "pocket_overview_pml",
        "pocket_overview_png",
        "interactions_pml",
        "interactions_png",
        "png_status",
        "caption_full_receptor",
        "caption_pocket_overview",
        "caption_interactions",
    ]
    pml_count = int(rows["pml_generated"].sum()) * len(VIEW_NAMES)
    png_count = int(rows["png_files_generated"].sum()) if "png_files_generated" in rows.columns else 0
    lines = [
        "# MAPK1 Phase 1 Portfolio Pose Panel Manifest",
        "",
        f"<!-- selected_ligands: {selected_ids} -->",
        "",
        "These panels are retrospective pose-inspection material, not evidence of validated inhibition.",
        "",
        "## Summary",
        "",
        f"- Selected ligands: {len(rows)}",
        f"- PML scripts generated: {pml_count}",
        f"- PNG files generated: {png_count}",
        f"- PNG status counts: {rows['png_status'].value_counts(dropna=False).to_dict()}",
        "",
        "## Selected Ligands",
        "",
        markdown_table(rows, columns=table_cols),
        "",
        "## Captions",
        "",
    ]
    for _, row in rows.iterrows():
        lines.extend(
            [
                f"### Ligand {row['ligand_id']}",
                "",
                f"- Full receptor: {row['caption_full_receptor']}",
                f"- Pocket overview: {row['caption_pocket_overview']}",
                f"- Interactions: {row['caption_interactions']}",
                "",
            ]
        )
    out_report.write_text("\n".join(lines), encoding="utf-8")


def run(
    pose_panel_config: str | Path,
    triage_table: str | Path,
    seed_table: str | Path,
    pose_locations: str | Path,
    prepared_receptors: str | Path,
    inspection_summary: str | Path,
    out_report: str | Path,
) -> pd.DataFrame:
    config = load_yaml(pose_panel_config)
    triage = read_table(triage_table)
    _seed = read_table(seed_table)
    locations = read_table(pose_locations)
    receptors = read_table(prepared_receptors)
    summary = read_table(inspection_summary)
    rows = build_panel_rows(triage, locations, receptors, summary, config)
    write_manifest(rows, Path(out_report))
    return rows


def main() -> None:
    args = parse_args()
    rows = run(
        args.pose_panel_config,
        args.triage_table,
        args.seed_table,
        args.pose_locations,
        args.prepared_receptors,
        args.inspection_summary,
        args.out_report,
    )
    pml_count = int(rows["pml_generated"].sum()) * len(VIEW_NAMES) if not rows.empty else 0
    png_count = int(rows["png_files_generated"].sum()) if not rows.empty and "png_files_generated" in rows.columns else 0
    print(f"Portfolio pose panels: ligands={len(rows)} pml={pml_count} png={png_count}")


if __name__ == "__main__":
    main()
