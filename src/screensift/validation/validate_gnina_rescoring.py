from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from screensift.common.external import run_command, shell_join, tool_command
from screensift.common.io import ensure_dir, markdown_table
from screensift.rescoring.parse_gnina_scores import parse_gnina_stdout_text
from screensift.rescoring.run_gnina_rescore import prepare_ligand_for_gnina
from screensift.validation.rmsd import translate_pose


OUTPUT_COLUMNS = [
    "pdb_id",
    "pose_type",
    "pose_file",
    "CNNscore",
    "CNNaffinity",
    "gnina_affinity",
    "intramolecular_energy",
    "status",
    "failure_reason",
    "command",
]


def build_gnina_score_command(
    receptor: str | Path,
    ligand: str | Path,
    gnina_mode: str = "native",
    gnina_bin: str = "gnina",
    gnina_image: str | None = None,
    gpu: str | bool | None = None,
    docker_args: list[str] | str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    args = ["--score_only", "-r", str(receptor), "-l", str(ligand)] + list(extra_args or [])
    return tool_command(
        gnina_bin,
        args,
        mode=gnina_mode,
        image=gnina_image,
        gpu=gpu,
        docker_args=docker_args,
    )


def validate_gnina_rescoring(
    redocking_table: str | Path,
    out_path: str | Path,
    report_path: str | Path,
    log_dir: str | Path,
    gnina_mode: str = "native",
    gnina_bin: str = "gnina",
    gnina_image: str | None = None,
    gpu: str | bool | None = None,
    docker_args: list[str] | str | None = None,
    extra_args: list[str] | None = None,
    decoy_translation_angstrom: float = 12.0,
    timeout_seconds: int = 1800,
    dry_run: bool = False,
) -> pd.DataFrame:
    redocking = pd.read_csv(redocking_table)
    rows: list[dict[str, Any]] = []
    logs = ensure_dir(log_dir)
    converted_poses = ensure_dir(logs / "converted_pose_inputs")
    for row in redocking.to_dict(orient="records"):
        pdb_id = str(row.get("pdb_id", "")).lower()
        receptor = row.get("receptor_pdbqt") or row.get("receptor_file") or row.get("gnina_receptor_input")
        native_pose = row.get("native_ligand_file")
        redocked_pose = row.get("redocked_pose_file")
        # Match the decoy file's extension to the native pose so translate_pose
        # writes a readable file (SDF content into .sdf, PDB into .pdb).
        native_suffix = Path(str(native_pose)).suffix.lower() if native_pose else ".pdb"
        decoy_suffix = native_suffix if native_suffix in {".pdb", ".sdf", ".mol"} else ".pdb"
        decoy_pose = logs / f"{pdb_id}_native_decoy{decoy_suffix}"
        if native_pose and Path(str(native_pose)).exists():
            translate_pose(native_pose, decoy_pose, (decoy_translation_angstrom, 0.0, 0.0))
        for pose_type, pose_file in [("native", native_pose), ("redocked", redocked_pose), ("decoy", str(decoy_pose))]:
            rows.append(
                _score_pose(
                    pdb_id=pdb_id,
                    receptor=receptor,
                    pose_type=pose_type,
                    pose_file=pose_file,
                    logs=logs,
                    gnina_mode=gnina_mode,
                    gnina_bin=gnina_bin,
                    gnina_image=gnina_image,
                    gpu=gpu,
                    docker_args=docker_args,
                    extra_args=extra_args,
                    converted_pose_dir=converted_poses,
                    timeout_seconds=timeout_seconds,
                    dry_run=dry_run,
                )
            )
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out = Path(out_path)
    ensure_dir(out.parent)
    frame.to_csv(out, index=False)
    write_report(frame, report_path)
    return frame


def _score_pose(
    pdb_id: str,
    receptor: Any,
    pose_type: str,
    pose_file: Any,
    logs: Path,
    gnina_mode: str,
    gnina_bin: str,
    gnina_image: str | None,
    gpu: str | bool | None,
    docker_args: list[str] | str | None,
    extra_args: list[str] | None,
    converted_pose_dir: Path,
    timeout_seconds: int,
    dry_run: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not receptor or not Path(str(receptor)).exists():
        reasons.append("missing_receptor")
    if not pose_file or not Path(str(pose_file)).exists():
        reasons.append("missing_pose")
    ligand_for_gnina = pose_file
    if not reasons and Path(str(pose_file)).suffix.lower() == ".pdbqt":
        try:
            ligand_for_gnina, _conversion_status = prepare_ligand_for_gnina(
                pd.Series({"gnina_ligand_input": pose_file, "pdb_id": pdb_id, "ligand_id": pose_type}),
                converted_pose_dir,
            )
        except Exception as exc:
            reasons.append(f"pose_conversion_failed:{exc}")
    command = build_gnina_score_command(
        receptor or "",
        ligand_for_gnina or "",
        gnina_mode=gnina_mode,
        gnina_bin=gnina_bin,
        gnina_image=gnina_image,
        gpu=gpu,
        docker_args=docker_args,
        extra_args=extra_args,
    )
    if reasons:
        return _row(pdb_id, pose_type, pose_file, None, None, None, None, "failed", ";".join(reasons), command)
    result = run_command(
        command,
        logs / f"{pdb_id}_{pose_type}_gnina_score.stdout.log",
        logs / f"{pdb_id}_{pose_type}_gnina_score.stderr.log",
        timeout_seconds=timeout_seconds,
        dry_run=dry_run,
    )
    parsed = {"cnnscore": None, "cnnaffinity": None, "affinity": None, "intramolecular_energy": None}
    if result.status == "complete":
        parsed = parse_gnina_stdout_text(Path(result.stdout_path).read_text(encoding="utf-8", errors="ignore"))
    return _row(
        pdb_id,
        pose_type,
        pose_file,
        parsed.get("cnnscore"),
        parsed.get("cnnaffinity"),
        parsed.get("affinity"),
        parsed.get("intramolecular_energy"),
        result.status,
        result.error_message,
        command,
    )


def _row(
    pdb_id: str,
    pose_type: str,
    pose_file: Any,
    cnnscore: float | None,
    cnnaffinity: float | None,
    affinity: float | None,
    intramolecular_energy: float | None,
    status: str,
    failure_reason: str,
    command: list[str],
) -> dict[str, Any]:
    return {
        "pdb_id": pdb_id.upper(),
        "pose_type": pose_type,
        "pose_file": str(pose_file or ""),
        "CNNscore": cnnscore,
        "CNNaffinity": cnnaffinity,
        "gnina_affinity": affinity,
        "intramolecular_energy": intramolecular_energy,
        "status": status,
        "failure_reason": failure_reason,
        "command": shell_join(command),
    }


def write_report(frame: pd.DataFrame, report_path: str | Path) -> None:
    path = Path(report_path)
    ensure_dir(path.parent)
    complete = int((frame["status"] == "complete").sum()) if "status" in frame.columns else 0
    lines = [
        "# GNINA Score-Only Validation Report",
        "",
        "GNINA score-only evaluates supplied poses; it does not create independent poses.",
        "",
        f"- rows: {len(frame)}",
        f"- complete_rows: {complete}",
        "",
        markdown_table(frame, ["pdb_id", "pose_type", "CNNscore", "CNNaffinity", "gnina_affinity", "status", "failure_reason"], max_rows=100),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GNINA score-only rescoring on native, redocked, and decoy poses.")
    parser.add_argument("--redocking-table", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--gnina-mode", choices=["native", "docker"], default="native")
    parser.add_argument("--gnina-bin", default="gnina")
    parser.add_argument("--gnina-image", default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--docker-arg", action="append", default=[])
    parser.add_argument("--gnina-extra-arg", action="append", default=[])
    parser.add_argument("--decoy-translation", type=float, default=12.0)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = validate_gnina_rescoring(
        args.redocking_table,
        args.out,
        args.report,
        args.log_dir,
        gnina_mode=args.gnina_mode,
        gnina_bin=args.gnina_bin,
        gnina_image=args.gnina_image,
        gpu=args.gpu,
        docker_args=args.docker_arg,
        extra_args=args.gnina_extra_arg,
        decoy_translation_angstrom=args.decoy_translation,
        timeout_seconds=args.timeout_seconds,
        dry_run=args.dry_run,
    )
    print(f"GNINA validation rows={len(frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
