from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from screensift.common.io import ensure_dir, load_yaml
from screensift.validation.pose_io import build_pose_index, find_receptor_for_pose, resolve_pose_path, stable_ligand_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PyMOL scripts for selected MAPK1 inspection poses.")
    parser.add_argument("--inspection-summary", default="results/tables/mapk1_phase1_pose_inspection_summary.csv")
    parser.add_argument("--pose-locations", default="results/tables/mapk1_phase1_selected_pose_locations.csv")
    parser.add_argument("--prepared-receptors", default="results/tables/mapk1_prepared_receptors.csv")
    parser.add_argument("--pose-config", default="configs/pose_inspection.yml")
    parser.add_argument("--out-dir", default="results/figures/pose_panels/pymol_scripts")
    parser.add_argument("--report", default="results/reports/mapk1_phase1_pose_panel_script_report.md")
    return parser.parse_args()


def read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(table_path)
    except Exception:
        return pd.DataFrame()
    if "ligand_id" in df.columns:
        df["ligand_id"] = df["ligand_id"].map(stable_ligand_id)
    return df


def pymol_script_text(row: pd.Series, receptor: Path, pose: Path) -> str:
    ligand_id = stable_ligand_id(row.get("ligand_id"))
    # PyMOL treats semicolons as command separators even in comment-like lines.
    categories = str(row.get("inspection_categories", "")).replace(";", ", ")
    scores = (
        f"Uni-Dock={row.get('unidock_best_score', '')}, CNNscore={row.get('CNNscore', '')}, "
        f"CNNaffinity={row.get('CNNaffinity', '')}, GNINA_affinity={row.get('gnina_affinity', '')}"
    )
    return f"""# MAPK1 Step 9 pose panel for ligand {ligand_id}
# Categories: {categories}
# Scores: {scores}
reinitialize
load {receptor}, receptor
load {pose}, ligand_{ligand_id}
hide everything
show cartoon, receptor
set cartoon_transparency, 0.25, receptor
show surface, receptor
set transparency, 0.65, receptor
show sticks, ligand_{ligand_id}
util.cbag ligand_{ligand_id}
color gray70, receptor
orient ligand_{ligand_id}
zoom ligand_{ligand_id}, 8
set ray_opaque_background, off
save results/figures/pose_panels/{ligand_id}_pose_review.pse
"""


def generate_scripts(summary: pd.DataFrame, locations: pd.DataFrame, receptors: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    ensure_dir(out_dir)
    pose_index = build_pose_index(["results/poses"])
    loc_map = {}
    if not locations.empty and "ligand_id" in locations.columns:
        loc_map = {stable_ligand_id(row["ligand_id"]): row for _, row in locations.iterrows()}
    rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        ligand_id = stable_ligand_id(row.get("ligand_id"))
        pose_row = loc_map.get(ligand_id, row)
        pose_path, pose_note = resolve_pose_path(pose_row, pose_index=pose_index)
        receptor_path, receptor_note = find_receptor_for_pose(pd.Series({**dict(row), "selected_pose_file": str(pose_path or "")}), receptors)
        if pose_path and receptor_path and pose_path.exists() and receptor_path.exists():
            script_path = out_dir / f"{ligand_id}_view.pml"
            script_path.write_text(pymol_script_text(row, receptor_path, pose_path), encoding="utf-8")
            rows.append(
                {
                    "ligand_id": ligand_id,
                    "script_generated": True,
                    "script_file": str(script_path),
                    "pose_file": str(pose_path),
                    "receptor_file": str(receptor_path),
                    "notes": pose_note,
                }
            )
        else:
            rows.append(
                {
                    "ligand_id": ligand_id,
                    "script_generated": False,
                    "script_file": "",
                    "pose_file": str(pose_path or ""),
                    "receptor_file": str(receptor_path or ""),
                    "notes": pose_note or receptor_note or "missing_pose_or_receptor",
                }
            )
    return pd.DataFrame(rows)


def write_report(script_rows: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    generated = int(script_rows["script_generated"].sum()) if "script_generated" in script_rows.columns else 0
    lines = [
        "# MAPK1 Phase 1 Pose Panel Script Report",
        "",
        f"- Rows: {len(script_rows)}",
        f"- PyMOL scripts generated: {generated}",
        "",
        "PyMOL is not required to generate these scripts. Open a `.pml` file in PyMOL to review a selected receptor-ligand pose.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    _config = load_yaml(args.pose_config)
    summary = read_table(args.inspection_summary)
    locations = read_table(args.pose_locations)
    receptors = read_table(args.prepared_receptors)
    rows = generate_scripts(summary, locations, receptors, Path(args.out_dir))
    write_report(rows, Path(args.report))
    print(f"PyMOL scripts generated: {int(rows['script_generated'].sum()) if not rows.empty else 0}")


if __name__ == "__main__":
    main()
