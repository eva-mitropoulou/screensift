from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from screensift.common.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight QC for selected inspection poses.")
    parser.add_argument("--pose-locations", default="results/tables/mapk1_phase1_selected_pose_locations.csv")
    parser.add_argument("--prepared-receptors", default="results/tables/mapk1_prepared_receptors.csv")
    parser.add_argument("--out", default="results/tables/mapk1_phase1_selected_pose_qc.csv")
    parser.add_argument("--report", default="results/reports/mapk1_phase1_selected_pose_qc_report.md")
    return parser.parse_args()


def posebusters_available() -> bool:
    try:
        import posebusters  # noqa: F401

        return True
    except Exception:
        return False


def run_pose_qc(locations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pb_available = posebusters_available()
    for _, row in locations.iterrows():
        ligand_id = str(row.get("ligand_id", ""))
        pose_file = str(row.get("selected_pose_file", "") or "")
        path = Path(pose_file) if pose_file else None
        if not path or not path.exists():
            rows.append(
                {
                    "ligand_id": ligand_id,
                    "pose_file": pose_file,
                    "qc_attempted": False,
                    "qc_success": False,
                    "posebusters_pass": pd.NA,
                    "failure_reason": "pose_missing",
                    "notes": "Pose file not present on this machine.",
                }
            )
            continue
        size_ok = path.stat().st_size > 0
        rows.append(
            {
                "ligand_id": ligand_id,
                "pose_file": str(path),
                "qc_attempted": True,
                "qc_success": size_ok,
                "posebusters_pass": pd.NA,
                "failure_reason": "" if size_ok else "empty_pose_file",
                "notes": (
                    "PoseBusters import is available, but Step 8 keeps QC to non-destructive file-level checks."
                    if pb_available
                    else "PoseBusters unavailable; file-level QC only."
                ),
            }
        )
    return pd.DataFrame(rows)


def write_pose_qc_report(qc: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    attempted = int(qc["qc_attempted"].sum()) if "qc_attempted" in qc.columns else 0
    success = int(qc["qc_success"].sum()) if "qc_success" in qc.columns else 0
    missing = int((qc.get("failure_reason", pd.Series(dtype=str)).fillna("") == "pose_missing").sum())
    lines = [
        "# MAPK1 Phase 1 Selected Pose QC Report",
        "",
        f"- Pose QC rows: {len(qc)}",
        f"- QC attempts: {attempted}",
        f"- File-level QC successes: {success}",
        f"- Missing poses: {missing}",
        "",
        "This is an optional pre-inspection check. Missing or unprocessed poses do not invalidate the inspection shortlist.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    locations_path = Path(args.pose_locations)
    locations = pd.read_csv(locations_path) if locations_path.exists() else pd.DataFrame(columns=["ligand_id", "selected_pose_file"])
    qc = run_pose_qc(locations)
    out = Path(args.out)
    ensure_dir(out.parent)
    qc.to_csv(out, index=False)
    write_pose_qc_report(qc, Path(args.report))
    print(f"Pose QC written: {out}; attempts={int(qc['qc_attempted'].sum()) if not qc.empty else 0}")


if __name__ == "__main__":
    main()
