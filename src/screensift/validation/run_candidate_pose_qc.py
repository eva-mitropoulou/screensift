from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from screensift.common.external import run_command, shell_join, tool_command
from screensift.common.io import ensure_dir, markdown_table
from screensift.rescoring.run_gnina_rescore import prepare_ligand_for_gnina
from screensift.validation.rmsd import pose_centroid_distance, pose_rmsd


OUTPUT_COLUMNS = [
    "ligand_id",
    "pdb_id",
    "unidock_pose_file",
    "gnina_pose_file",
    "unidock_vs_gnina_rmsd",
    "native_pocket_center_distance",
    "pose_consistency_flag",
    "status",
    "failure_reason",
    "command",
]


def build_gnina_docking_command(
    receptor: str | Path,
    ligand: str | Path,
    out_pose: str | Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    gnina_mode: str = "native",
    gnina_bin: str = "gnina",
    gnina_image: str | None = None,
    gpu: str | bool | None = None,
    docker_args: list[str] | str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    args = [
        "-r",
        str(receptor),
        "-l",
        str(ligand),
        "--out",
        str(out_pose),
        "--center_x",
        str(center[0]),
        "--center_y",
        str(center[1]),
        "--center_z",
        str(center[2]),
        "--size_x",
        str(size[0]),
        "--size_y",
        str(size[1]),
        "--size_z",
        str(size[2]),
    ] + list(extra_args or [])
    return tool_command(gnina_bin, args, mode=gnina_mode, image=gnina_image, gpu=gpu, docker_args=docker_args)


def candidate_pose_qc(
    candidates_path: str | Path,
    score_population_path: str | Path,
    boxes_path: str | Path,
    out_path: str | Path,
    report_path: str | Path,
    pose_dir: str | Path,
    log_dir: str | Path,
    top_n: int = 25,
    gnina_mode: str = "native",
    gnina_bin: str = "gnina",
    gnina_image: str | None = None,
    gpu: str | bool | None = None,
    docker_args: list[str] | str | None = None,
    extra_args: list[str] | None = None,
    consistent_threshold: float = 2.0,
    moderate_threshold: float = 4.0,
    timeout_seconds: int = 1800,
    converted_ligand_root: str | Path | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    candidates = pd.read_csv(candidates_path, dtype={"ligand_id": str}).head(top_n)
    population = pd.read_csv(score_population_path, dtype={"ligand_id": str})
    boxes = pd.read_csv(boxes_path)
    merged = candidates[["ligand_id"]].merge(population, on="ligand_id", how="left")
    rows: list[dict[str, Any]] = []
    poses = ensure_dir(pose_dir)
    logs = ensure_dir(log_dir)
    converted_root = ensure_dir(converted_ligand_root or Path(pose_dir).parent / "gnina_inputs")
    for row in merged.to_dict(orient="records"):
        pdb_id = str(row.get("pdb_id", "")).upper()
        box = boxes[boxes["pdb_id"].astype(str).str.upper().eq(pdb_id)]
        box_row = box.iloc[0].to_dict() if not box.empty else {}
        unidock_pose = row.get("best_output_pose_file") or row.get("output_pose_file") or row.get("unidock_pose_file")
        ligand_source = row.get("gnina_ligand_input") or unidock_pose
        ligand_input = ligand_source
        receptor = box_row.get("ligand_file", "")
        receptor = str(box_row.get("receptor_clean_pdb", "")) if box_row.get("receptor_clean_pdb") else str(box_row.get("receptor_pdbqt", ""))
        out_pose = poses / pdb_id.lower() / f"{row.get('ligand_id')}_gnina_redocked.sdf"
        ensure_dir(out_pose.parent)
        reasons: list[str] = []
        if not box_row:
            reasons.append("missing_box")
        if not unidock_pose or not Path(str(unidock_pose)).exists():
            reasons.append("missing_unidock_pose")
        if not receptor or not Path(str(receptor)).exists():
            reasons.append("missing_receptor")
        if ligand_source and Path(str(ligand_source)).exists():
            try:
                conversion_row = pd.Series({**row, "gnina_ligand_input": ligand_source})
                ligand_input, _conversion_status = prepare_ligand_for_gnina(conversion_row, converted_root)
            except Exception as exc:
                reasons.append(f"ligand_conversion_failed:{exc}")
        else:
            reasons.append("missing_gnina_ligand_input")
        command = build_gnina_docking_command(
            receptor,
            ligand_input or "",
            out_pose,
            center=(float(box_row.get("center_x", 0.0)), float(box_row.get("center_y", 0.0)), float(box_row.get("center_z", 0.0))),
            size=(float(box_row.get("size_x", 0.0)), float(box_row.get("size_y", 0.0)), float(box_row.get("size_z", 0.0))),
            gnina_mode=gnina_mode,
            gnina_bin=gnina_bin,
            gnina_image=gnina_image,
            gpu=gpu,
            docker_args=docker_args,
            extra_args=extra_args,
        )
        if reasons:
            rows.append(_row(row, unidock_pose, out_pose, None, None, "missing_pose", "failed", ";".join(reasons), command))
            continue
        result = run_command(command, logs / f"{row.get('ligand_id')}_gnina_dock.stdout.log", logs / f"{row.get('ligand_id')}_gnina_dock.stderr.log", timeout_seconds=timeout_seconds, dry_run=dry_run)
        rmsd = pose_rmsd(ligand_input, out_pose) if result.status == "complete" else None
        native_distance = pose_centroid_distance(box_row.get("ligand_file", ""), out_pose) if result.status == "complete" and box_row.get("ligand_file") else None
        flag = _pose_flag(rmsd.rmsd_angstrom if rmsd and rmsd.status == "complete" else None, consistent_threshold, moderate_threshold)
        failure = result.error_message or (rmsd.failure_reason if rmsd and rmsd.status != "complete" else "")
        rows.append(_row(row, unidock_pose, out_pose, rmsd.rmsd_angstrom if rmsd else None, native_distance.rmsd_angstrom if native_distance else None, flag, result.status, failure, command))
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out = Path(out_path)
    ensure_dir(out.parent)
    frame.to_csv(out, index=False)
    write_report(frame, report_path)
    return frame


def _pose_flag(rmsd: float | None, consistent: float, moderate: float) -> str:
    if rmsd is None or pd.isna(rmsd):
        return "missing_pose"
    if rmsd <= consistent:
        return "consistent_pose"
    if rmsd <= moderate:
        return "moderate_shift"
    return "discordant_pose"


def _row(row: dict[str, Any], unidock_pose: Any, gnina_pose: Path, rmsd: float | None, pocket_distance: float | None, flag: str, status: str, failure_reason: str, command: list[str]) -> dict[str, Any]:
    return {
        "ligand_id": row.get("ligand_id"),
        "pdb_id": row.get("pdb_id"),
        "unidock_pose_file": str(unidock_pose or ""),
        "gnina_pose_file": str(gnina_pose),
        "unidock_vs_gnina_rmsd": rmsd,
        "native_pocket_center_distance": pocket_distance,
        "pose_consistency_flag": flag,
        "status": status,
        "failure_reason": failure_reason,
        "command": shell_join(command),
    }


def write_report(frame: pd.DataFrame, report_path: str | Path) -> None:
    path = Path(report_path)
    ensure_dir(path.parent)
    lines = [
        "# Candidate Pose QC Report",
        "",
        "Candidate pose RMSD is an agreement check between predicted poses, not experimental pose accuracy.",
        "",
        markdown_table(frame, ["ligand_id", "pdb_id", "unidock_vs_gnina_rmsd", "native_pocket_center_distance", "pose_consistency_flag", "status", "failure_reason"], max_rows=100),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optional candidate pose agreement QC by GNINA redocking.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--score-population", required=True)
    parser.add_argument("--boxes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--pose-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--gnina-mode", choices=["native", "docker"], default="native")
    parser.add_argument("--gnina-bin", default="gnina")
    parser.add_argument("--gnina-image", default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--docker-arg", action="append", default=[])
    parser.add_argument("--gnina-extra-arg", action="append", default=[])
    parser.add_argument("--converted-ligand-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = candidate_pose_qc(
        args.candidates,
        args.score_population,
        args.boxes,
        args.out,
        args.report,
        args.pose_dir,
        args.log_dir,
        top_n=args.top_n,
        gnina_mode=args.gnina_mode,
        gnina_bin=args.gnina_bin,
        gnina_image=args.gnina_image,
        gpu=args.gpu,
        docker_args=args.docker_arg,
        extra_args=args.gnina_extra_arg,
        converted_ligand_root=args.converted_ligand_root,
        dry_run=args.dry_run,
    )
    print(f"Candidate pose QC rows={len(frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
