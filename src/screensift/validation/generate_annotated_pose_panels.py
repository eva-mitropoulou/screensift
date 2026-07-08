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


VIEW_NAMES = ("full_receptor", "clean_pocket", "annotated_contacts", "residue_type_view")
REQUIRED_WORDING = (
    "PyMOL polar contacts are geometric candidate interactions. They are used for pose review and "
    "visualization, not as proof of binding or potency."
)

CONTACT_REVIEW_COLUMNS = [
    "ligand_id",
    "triage_tier",
    "activity_label",
    "inspection_categories",
    "manual_priority",
    "pose_file",
    "receptor_file",
    "full_receptor_png",
    "clean_pocket_png",
    "annotated_contacts_png",
    "residue_type_view_png",
    "full_receptor_pml",
    "clean_pocket_pml",
    "annotated_contacts_pml",
    "residue_type_view_pml",
    "n_hbond_interactions",
    "n_hydrophobic_interactions",
    "n_pi_interactions",
    "n_total_interactions",
    "anomaly_flags",
    "recommended_action",
    "manual_review_priority",
    "notes",
]


@dataclass(frozen=True)
class AnnotatedPanelPaths:
    output_dir: Path
    pml_dir: Path
    png_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate annotated PyMOL contact-review panels for MAPK1 ligands.")
    parser.add_argument("--config", default="configs/annotated_pose_panel.yml")
    parser.add_argument("--triage-table", default="results/tables/mapk1_phase1_candidate_triage.csv")
    parser.add_argument("--seed-table", default="results/tables/mapk1_phase1_step10_seed_ligands.csv")
    parser.add_argument("--pose-locations", default="results/tables/mapk1_phase1_selected_pose_locations.csv")
    parser.add_argument("--prepared-receptors", default="results/tables/mapk1_prepared_receptors.csv")
    parser.add_argument("--pose-interactions", default="results/tables/mapk1_phase1_pose_interactions.csv")
    parser.add_argument("--pose-flags", default="results/tables/mapk1_phase1_pose_plausibility_flags.csv")
    parser.add_argument("--out-contact-table", default="results/tables/mapk1_phase1_pose_panel_contact_review_table.csv")
    parser.add_argument("--out-report", default="results/reports/mapk1_phase1_annotated_pose_panel_manifest.md")
    return parser.parse_args()


def read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(table_path)
    if "ligand_id" in df.columns:
        df["ligand_id"] = df["ligand_id"].map(stable_ligand_id)
    return df


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def clean_comment(value: Any) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    return text.replace(";", ", ").replace("\n", " ").strip()


def pml_quote(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def panel_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("annotated_pose_panel", {})


def panel_paths(config: dict[str, Any]) -> AnnotatedPanelPaths:
    cfg = panel_config(config)
    output_dir = Path(cfg.get("output_dir", "results/figures/pose_review_panels"))
    pml_dir = Path(cfg.get("pml_dir", output_dir / "annotated_pml"))
    png_dir = Path(cfg.get("png_dir", output_dir / "annotated_png"))
    ensure_dir(output_dir)
    ensure_dir(pml_dir)
    ensure_dir(png_dir)
    return AnnotatedPanelPaths(output_dir=output_dir, pml_dir=pml_dir, png_dir=png_dir)


def category_contains(row: pd.Series, needle: str) -> bool:
    return needle in str(row.get("inspection_categories", ""))


def has_anomaly(row: pd.Series) -> bool:
    flags = str(row.get("anomaly_flags", "") or "").strip().lower()
    return bool(flags and flags != "nan")


def select_ligands(triage: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = panel_config(config)
    if triage.empty:
        return pd.DataFrame()
    work = triage.copy()
    work["ligand_id"] = work["ligand_id"].map(stable_ligand_id)
    include_tiers = set(cfg.get("include_tiers", ["A_analog_seed"]))
    controls = cfg.get("include_controls", {})
    selected_parts: list[pd.DataFrame] = []

    tier_rows = work[work.get("triage_tier", pd.Series(index=work.index, dtype=object)).isin(include_tiers)].copy()
    tier_rows["_selection_category"] = "tier_a_seed"
    selected_parts.append(tier_rows)

    false_positive_n = int(controls.get("false_positive_cases", 0) or 0)
    false_negative_n = int(controls.get("false_negative_cases", 0) or 0)
    anomaly_n = int(controls.get("score_anomaly_cases", 0) or 0)

    if false_positive_n:
        fp = work[
            work.apply(lambda row: category_contains(row, "consensus_inactive_false_positive"), axis=1)
            | work.get("recommended_action", pd.Series(index=work.index, dtype=object)).fillna("").eq("false_positive_failure_case")
        ].copy()
        fp = fp.sort_values(["ecfp4_active_similarity", "n_total_interactions"], ascending=[False, False], na_position="last")
        fp = fp.head(false_positive_n)
        fp["_selection_category"] = "false_positive_control"
        selected_parts.append(fp)

    if false_negative_n:
        fn = work[
            work.apply(lambda row: category_contains(row, "active_false_negative"), axis=1)
            | work.get("recommended_action", pd.Series(index=work.index, dtype=object)).fillna("").eq("false_negative_failure_case")
        ].copy()
        fn = fn.sort_values(["n_total_interactions", "ecfp4_active_similarity"], ascending=[False, True], na_position="last")
        fn = fn.head(false_negative_n)
        fn["_selection_category"] = "false_negative_control"
        selected_parts.append(fn)

    if anomaly_n:
        anomalies = work[work.apply(has_anomaly, axis=1)].copy()
        anomalies["_severe_score"] = anomalies["anomaly_flags"].fillna("").astype(str).str.contains(
            "extreme_unidock_negative|extreme_positive_gnina_affinity|suspicious_unidock_extreme_negative|suspicious_ligand_efficiency_extreme",
            regex=True,
        )
        anomalies = anomalies.sort_values(["_severe_score", "n_total_interactions"], ascending=[False, False], na_position="last")
        anomalies = anomalies.head(anomaly_n)
        anomalies["_selection_category"] = "score_anomaly_control"
        selected_parts.append(anomalies.drop(columns=["_severe_score"], errors="ignore"))

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    if selected.empty:
        return selected
    selected["_selection_order"] = range(len(selected))
    selected = selected.sort_values("_selection_order").drop_duplicates("ligand_id", keep="first")
    return selected.drop(columns=["_selection_order"], errors="ignore").reset_index(drop=True)


def merge_context(selected: pd.DataFrame, interactions: pd.DataFrame, flags: pd.DataFrame, locations: pd.DataFrame) -> pd.DataFrame:
    work = selected.copy()
    if not locations.empty:
        cols = [col for col in locations.columns if col == "ligand_id" or col not in work.columns]
        work = work.merge(locations[cols].drop_duplicates("ligand_id"), on="ligand_id", how="left")
    if not interactions.empty:
        cols = [
            col
            for col in [
                "ligand_id",
                "pose_file",
                "receptor_file",
                "n_hbond_interactions",
                "n_hydrophobic_interactions",
                "n_pi_interactions",
                "n_total_interactions",
            ]
            if col in interactions.columns
        ]
        work = work.merge(interactions[cols].drop_duplicates("ligand_id"), on="ligand_id", how="left", suffixes=("", "_interaction"))
        if "n_total_interactions_interaction" in work.columns:
            work["n_total_interactions"] = work["n_total_interactions"].fillna(work["n_total_interactions_interaction"])
    if not flags.empty:
        cols = [col for col in ["ligand_id", "plausibility_flags", "n_plausibility_flags"] if col in flags.columns]
        work = work.merge(flags[cols].drop_duplicates("ligand_id"), on="ligand_id", how="left")
    return work


def output_paths_for_ligand(ligand_id: str, paths: AnnotatedPanelPaths, view: str) -> dict[str, Path]:
    return {
        "pml": paths.pml_dir / f"{ligand_id}_{view}.pml",
        "png": paths.png_dir / f"{ligand_id}_{view}.png",
        "pse": paths.output_dir / f"{ligand_id}_{view}.pse",
    }


def cleanup_existing_outputs(ligand_id: str, paths: AnnotatedPanelPaths) -> None:
    for view in VIEW_NAMES:
        for output_path in output_paths_for_ligand(ligand_id, paths, view).values():
            output_path.unlink(missing_ok=True)


def score_comment(row: pd.Series) -> str:
    return (
        f"Uni-Dock={row.get('unidock_best_score', '')}, CNNscore={row.get('CNNscore', '')}, "
        f"CNNaffinity={row.get('CNNaffinity', '')}, GNINA_affinity={row.get('gnina_affinity', '')}"
    )


def color_residue_groups_lines(ligand_id: str, config: dict[str, Any]) -> list[str]:
    cfg = panel_config(config)
    groups = cfg.get("residue_groups", {})
    colors = cfg.get("residue_type_colors", {})
    name_map = {"glycine_proline": "glypro"}
    lines: list[str] = []
    for group_name, residues in groups.items():
        selection_name = f"{name_map.get(group_name, group_name)}_{ligand_id}"
        residue_expr = "+".join(residues)
        lines.append(f"select {selection_name}, pocket_{ligand_id} and resn {residue_expr}")
    for group_name in groups:
        selection_name = f"{name_map.get(group_name, group_name)}_{ligand_id}"
        color = colors.get(group_name, "yellow")
        lines.append(f"color {color}, {selection_name}")
    return lines


def base_pml_header(row: pd.Series, receptor: Path, pose: Path, config: dict[str, Any], view: str) -> tuple[list[str], str, str]:
    ligand_id = stable_ligand_id(row.get("ligand_id"))
    ligand_obj = f"ligand_{ligand_id}"
    pocket_obj = f"pocket_{ligand_id}"
    cfg = panel_config(config)
    cutoffs = cfg.get("cutoffs", {})
    styling = cfg.get("styling", {})
    pocket_cutoff = float(cutoffs.get("pocket_residue_cutoff_angstrom", 4.0))
    ligand_radius = float(styling.get("stick_radius_ligand", 0.22))
    pocket_radius = float(styling.get("stick_radius_pocket", 0.14))
    lines = [
        f"# MAPK1 annotated pose panel for ligand {ligand_id}",
        f"# View: {view}",
        f"# Selection category: {clean_comment(row.get('_selection_category', ''))}",
        f"# Triage tier: {clean_comment(row.get('triage_tier', ''))}",
        f"# Scores: {clean_comment(score_comment(row))}",
        "reinitialize",
        "",
        f"load {pml_quote(receptor)}, receptor",
        f"load {pml_quote(pose)}, {ligand_obj}",
        "",
        "hide everything",
        "show cartoon, receptor",
        f"show sticks, {ligand_obj}",
        f"select {pocket_obj}, byres (receptor within {pocket_cutoff:.1f} of {ligand_obj})",
        f"show sticks, {pocket_obj}",
        f"set stick_radius, {ligand_radius:.2f}, {ligand_obj}",
        f"set stick_radius, {pocket_radius:.2f}, {pocket_obj}",
    ]
    return lines, ligand_obj, pocket_obj


def finish_view_lines(ligand_id: str, ligand_obj: str, config: dict[str, Any], view: str, paths: AnnotatedPanelPaths) -> list[str]:
    cfg = panel_config(config)
    image = cfg.get("image", {})
    view_cfg = cfg.get("views", {}).get(view, {})
    width = int(image.get("width", 2400))
    height = int(image.get("height", 1800))
    dpi = int(image.get("dpi", 300))
    ray = 1 if parse_bool(image.get("ray", True)) else 0
    background = image.get("background", "white")
    opaque = "on" if parse_bool(image.get("opaque_background", True)) else "off"
    zoom_target = view_cfg.get("zoom_target", "ligand")
    zoom_buffer = float(view_cfg.get("zoom_buffer", 8))
    output_paths = output_paths_for_ligand(ligand_id, paths, view)
    lines = [
        f"bg_color {background}",
        f"set ray_opaque_background, {opaque}",
        "set orthoscopic, on",
    ]
    if zoom_target == "receptor":
        lines.extend(["orient receptor", f"zoom receptor, {zoom_buffer:g}"])
    else:
        lines.extend([f"orient {ligand_obj}", f"zoom {ligand_obj}, {zoom_buffer:g}"])
    lines.extend(
        [
            f"png {pml_quote(output_paths['png'])}, {width}, {height}, dpi={dpi}, ray={ray}",
            f"save {pml_quote(output_paths['pse'])}",
            "",
        ]
    )
    return lines


def pml_text(row: pd.Series, receptor: Path, pose: Path, config: dict[str, Any], view: str, paths: AnnotatedPanelPaths) -> str:
    ligand_id = stable_ligand_id(row.get("ligand_id"))
    cfg = panel_config(config)
    styling = cfg.get("styling", {})
    views = cfg.get("views", {})
    view_cfg = views.get(view, {})
    cutoffs = cfg.get("cutoffs", {})
    receptor_color = styling.get("receptor_color", "gray80")
    ligand_color = styling.get("ligand_color", "green")
    pocket_color = styling.get("pocket_color", "yellow")
    polar_color = styling.get("polar_contact_color", "cyan")
    dash_radius = float(styling.get("dash_radius", 0.08))
    label_size = int(styling.get("label_size", 16))
    label_color = styling.get("label_color", "black")
    polar_cutoff = float(cutoffs.get("polar_contact_cutoff_angstrom", 3.5))
    lines, ligand_obj, pocket_obj = base_pml_header(row, receptor, pose, config, view)

    lines.extend(
        [
            f"color {receptor_color}, receptor",
            f"color {ligand_color}, {ligand_obj}",
            f"color {pocket_color}, {pocket_obj}",
        ]
    )

    if parse_bool(view_cfg.get("show_surface", False)):
        transparency = float(view_cfg.get("surface_transparency", 0.82))
        lines.extend(["show surface, receptor", f"set transparency, {transparency:.2f}, receptor"])

    if view == "annotated_contacts":
        lines.extend(
            [
                f"hide cartoon, {pocket_obj}",
                f"hide lines, {pocket_obj}",
                f"distance polar_contacts_{ligand_id}, {ligand_obj}, {pocket_obj}, {polar_cutoff:.1f}, mode=2",
                f"set dash_radius, {dash_radius:.2f}, polar_contacts_{ligand_id}",
                f"color {polar_color}, polar_contacts_{ligand_id}",
                f"show labels, polar_contacts_{ligand_id}",
                f'label (name CA and {pocket_obj}), "%s%s" % (resn, resi)',
                f"set label_size, {label_size}",
                f"set label_color, {label_color}",
            ]
        )

    if view == "residue_type_view":
        lines.extend(color_residue_groups_lines(ligand_id, config))
        lines.extend(
            [
                f"hide cartoon, {pocket_obj}",
                f'label (name CA and {pocket_obj}), "%s%s" % (resn, resi)',
                f"set label_size, {label_size}",
                f"set label_color, {label_color}",
            ]
        )

    lines.extend(finish_view_lines(ligand_id, ligand_obj, config, view, paths))
    return "\n".join(lines)


def candidate_executables(config: dict[str, Any]) -> list[str]:
    cfg = panel_config(config)
    configured = cfg.get("pymol", {}).get("executable_candidates", ["pymol", "pymol-open-source"])
    candidates: list[str] = []
    env_bin = os.environ.get("PYMOL_BIN")
    if env_bin:
        candidates.append(env_bin)
    candidates.extend(str(candidate) for candidate in configured)
    candidates.append(str(Path(sys.executable).parent / "pymol"))
    return list(dict.fromkeys(candidates))


def find_working_pymol(config: dict[str, Any]) -> tuple[str | None, str]:
    last_error = "No PyMOL executable candidate was found."
    for candidate in candidate_executables(config):
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


def manual_review_priority(row: pd.Series) -> str:
    tier = str(row.get("triage_tier", ""))
    category = str(row.get("_selection_category", ""))
    if tier == "A_analog_seed":
        return "high"
    if category == "score_anomaly_control":
        return "score_anomaly_control"
    if category in {"false_positive_control", "false_negative_control"}:
        return "failure_control"
    return "review"


def build_panel_rows(
    triage: pd.DataFrame,
    locations: pd.DataFrame,
    receptors: pd.DataFrame,
    interactions: pd.DataFrame,
    flags: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    selected = select_ligands(triage, config)
    if selected.empty:
        return pd.DataFrame(columns=CONTACT_REVIEW_COLUMNS)
    work = merge_context(selected, interactions, flags, locations)
    pose_index = build_pose_index(["results/poses"])
    paths = panel_paths(config)
    cfg = panel_config(config)
    run_headless = parse_bool(cfg.get("pymol", {}).get("run_headless_if_available", True))
    pymol_exe, pymol_warning = find_working_pymol(config) if run_headless else (None, "headless PyMOL disabled")
    log_dir = Path("results/reports/annotated_pose_panel_logs")
    rows: list[dict[str, Any]] = []

    for _, row in work.iterrows():
        ligand_id = stable_ligand_id(row.get("ligand_id"))
        pose_path, pose_note = resolve_pose_path(row, pose_index=pose_index)
        receptor_path, receptor_note = find_receptor_for_pose(pd.Series({**dict(row), "selected_pose_file": str(pose_path or "")}), receptors)
        cleanup_existing_outputs(ligand_id, paths)
        view_paths = {view: output_paths_for_ligand(ligand_id, paths, view) for view in VIEW_NAMES}
        pml_generated = False
        png_files_generated = 0
        png_status = "not_attempted"
        notes = [note for note in [pose_note, receptor_note] if note]

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
                "triage_tier": row.get("triage_tier", ""),
                "activity_label": row.get("activity_label", ""),
                "inspection_categories": row.get("inspection_categories", ""),
                "manual_priority": row.get("manual_priority", ""),
                "pose_file": str(pose_path or ""),
                "receptor_file": str(receptor_path or ""),
                "full_receptor_png": str(view_paths["full_receptor"]["png"]),
                "clean_pocket_png": str(view_paths["clean_pocket"]["png"]),
                "annotated_contacts_png": str(view_paths["annotated_contacts"]["png"]),
                "residue_type_view_png": str(view_paths["residue_type_view"]["png"]),
                "full_receptor_pml": str(view_paths["full_receptor"]["pml"]),
                "clean_pocket_pml": str(view_paths["clean_pocket"]["pml"]),
                "annotated_contacts_pml": str(view_paths["annotated_contacts"]["pml"]),
                "residue_type_view_pml": str(view_paths["residue_type_view"]["pml"]),
                "n_hbond_interactions": row.get("n_hbond_interactions", pd.NA),
                "n_hydrophobic_interactions": row.get("n_hydrophobic_interactions", pd.NA),
                "n_pi_interactions": row.get("n_pi_interactions", pd.NA),
                "n_total_interactions": row.get("n_total_interactions", pd.NA),
                "anomaly_flags": row.get("anomaly_flags", ""),
                "recommended_action": row.get("recommended_action", ""),
                "manual_review_priority": manual_review_priority(row),
                "notes": "; ".join(note for note in notes if note),
                "selection_category": row.get("_selection_category", ""),
                "pml_generated": pml_generated,
                "png_status": png_status,
                "png_files_generated": png_files_generated,
                "pymol_executable": pymol_exe or "",
            }
        )

    out = pd.DataFrame(rows)
    for col in CONTACT_REVIEW_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def write_manifest(rows: pd.DataFrame, out_report: Path) -> None:
    ensure_dir(out_report.parent)
    if rows.empty:
        out_report.write_text("# MAPK1 Phase 1 Annotated Pose Panel Manifest\n\nNo ligands selected.\n", encoding="utf-8")
        return
    pml_count = int(rows["pml_generated"].sum()) * len(VIEW_NAMES)
    png_count = int(rows["png_files_generated"].sum()) if "png_files_generated" in rows.columns else 0
    table_cols = [
        "ligand_id",
        "selection_category",
        "triage_tier",
        "activity_label",
        "full_receptor_png",
        "clean_pocket_png",
        "annotated_contacts_png",
        "residue_type_view_png",
        "n_hbond_interactions",
        "n_hydrophobic_interactions",
        "n_pi_interactions",
        "n_total_interactions",
        "recommended_action",
        "png_status",
    ]
    lines = [
        "# MAPK1 Phase 1 Annotated Pose Panel Manifest",
        "",
        REQUIRED_WORDING,
        "",
        "## Summary",
        "",
        f"- Ligands selected: {len(rows)}",
        f"- PML scripts generated: {pml_count}",
        f"- PNG files generated: {png_count}",
        f"- PNG status counts: {rows['png_status'].value_counts(dropna=False).to_dict()}",
        "",
        "## Selected Ligands",
        "",
        markdown_table(rows, columns=table_cols),
        "",
        "## Recommended Portfolio Image",
        "",
        "Use `clean_pocket_png` for uncluttered clean review views and `annotated_contacts_png` for manual scientific review.",
        "",
    ]
    out_report.write_text("\n".join(lines), encoding="utf-8")


def write_pose_reading_guide(out_path: Path) -> None:
    ensure_dir(out_path.parent)
    lines = [
        "# MAPK1 Phase 1 Pose Reading Guide",
        "",
        REQUIRED_WORDING,
        "",
        "## Visual Legend",
        "",
        "- gray cartoon = protein backbone",
        "- spirals = alpha helices",
        "- arrows = beta strands",
        "- thin gray lines = loops",
        "- green sticks = docked ligand",
        "- yellow or colored sticks = nearby pocket residues",
        "- cyan dashed lines = candidate polar contacts / H-bonds",
        "- labels such as ASP106 = residue name + residue number",
        "",
        "## Verdict Rules",
        "",
        "KEEP:",
        "",
        "- ligand inside pocket",
        "- plausible contacts",
        "- no obvious clash",
        "- no severe score anomaly",
        "- useful low-similarity / SAV case",
        "",
        "BACKUP:",
        "",
        "- inside pocket but contacts unclear or weak",
        "",
        "REJECT/HOLD:",
        "",
        "- outside pocket",
        "- obvious clash",
        "- broken ligand",
        "- scoring anomaly",
        "- no meaningful contacts",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run(
    config_path: str | Path,
    triage_table: str | Path,
    seed_table: str | Path,
    pose_locations: str | Path,
    prepared_receptors: str | Path,
    pose_interactions: str | Path,
    pose_flags: str | Path,
    out_contact_table: str | Path,
    out_report: str | Path,
) -> pd.DataFrame:
    config = load_yaml(config_path)
    triage = read_table(triage_table)
    _seed = read_table(seed_table)
    locations = read_table(pose_locations)
    receptors = read_table(prepared_receptors)
    interactions = read_table(pose_interactions)
    flags = read_table(pose_flags)
    rows = build_panel_rows(triage, locations, receptors, interactions, flags, config)

    out_table = Path(out_contact_table)
    ensure_dir(out_table.parent)
    rows[CONTACT_REVIEW_COLUMNS].to_csv(out_table, index=False)
    write_manifest(rows, Path(out_report))
    write_pose_reading_guide(Path("results/reports/mapk1_phase1_pose_reading_guide.md"))
    return rows


def main() -> None:
    args = parse_args()
    rows = run(
        args.config,
        args.triage_table,
        args.seed_table,
        args.pose_locations,
        args.prepared_receptors,
        args.pose_interactions,
        args.pose_flags,
        args.out_contact_table,
        args.out_report,
    )
    pml_count = int(rows["pml_generated"].sum()) * len(VIEW_NAMES) if not rows.empty else 0
    png_count = int(rows["png_files_generated"].sum()) if not rows.empty and "png_files_generated" in rows.columns else 0
    print(f"Annotated pose panels: ligands={len(rows)} pml={pml_count} png={png_count}")


if __name__ == "__main__":
    main()
