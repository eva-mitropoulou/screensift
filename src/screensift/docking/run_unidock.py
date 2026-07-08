from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


from screensift.common.io import ensure_dir
from screensift.docking.collect_docking_inputs import safe_id


RAW_SCORE_COLUMNS = [
    "docking_id",
    "ligand_pdbqt",
    "receptor_pdbqt",
    "pdb_id",
    "output_pose_file",
    "best_score",
    "status",
    "error_message",
]


def resolve_unidock_bin(value: str | None = None) -> str:
    candidate = value or os.environ.get("UNIDOCK_BIN") or shutil.which("unidock")
    if not candidate:
        raise FileNotFoundError("Uni-Dock binary not found. Set UNIDOCK_BIN or put unidock on PATH.")
    path = Path(candidate)
    if path.exists():
        if not os.access(path, os.X_OK):
            raise PermissionError(f"Uni-Dock binary is not executable: {path}")
        return str(path)
    found = shutil.which(candidate)
    if found:
        return found
    raise FileNotFoundError(f"Uni-Dock binary not found: {candidate}")


def validate_unidock(unidock_bin: str) -> str:
    completed = subprocess.run([unidock_bin, "--help"], check=False, capture_output=True, text=True, timeout=30)
    help_text = completed.stdout + completed.stderr
    required_flags = ["--receptor", "--ligand_index", "--center_x", "--size_x", "--dir", "--max_step"]
    missing = [flag for flag in required_flags if flag not in help_text]
    if completed.returncode != 0 or missing:
        raise RuntimeError(
            f"Uni-Dock command syntax check failed for {unidock_bin}. "
            f"returncode={completed.returncode}; missing_flags={missing}; output={help_text[:1000]}"
        )
    return help_text


def parse_best_score_from_pose(path: str | Path) -> float | None:
    pose_path = Path(path)
    if not pose_path.exists():
        return None
    pattern = re.compile(r"RESULT:\s*([-+]?\d+(?:\.\d+)?)")
    for line in pose_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            return float(match.group(1))
    return None


def output_pose_path(row: pd.Series, out_root: Path) -> Path:
    pdb_id = safe_id(row["pdb_id"]).lower()
    docking_id = safe_id(row["docking_id"])
    return out_root / pdb_id / f"{docking_id}.pdbqt"


def batch_output_pose_path(row: pd.Series) -> Path:
    output_dir = Path(str(row["output_dir"]))
    ligand_stem = Path(str(row["ligand_pdbqt"])).stem
    return output_dir / f"{ligand_stem}_out.pdbqt"


def run_one_docking(
    row: pd.Series,
    unidock_bin: str,
    out_root: Path,
    log_dir: Path,
    exhaustiveness: int,
    num_modes: int,
    energy_range: float,
    cpu: int,
    seed: int,
) -> dict[str, Any]:
    docking_id = safe_id(row["docking_id"])
    output_pose = output_pose_path(row, out_root)
    ensure_dir(output_pose.parent)
    ensure_dir(log_dir)
    stdout_path = log_dir / f"{docking_id}.stdout.log"
    stderr_path = log_dir / f"{docking_id}.stderr.log"

    command = [
        unidock_bin,
        "--receptor",
        str(row["receptor_pdbqt"]),
        "--ligand",
        str(row["ligand_pdbqt"]),
        "--center_x",
        str(float(row["center_x"])),
        "--center_y",
        str(float(row["center_y"])),
        "--center_z",
        str(float(row["center_z"])),
        "--size_x",
        str(float(row["size_x"])),
        "--size_y",
        str(float(row["size_y"])),
        "--size_z",
        str(float(row["size_z"])),
        "--out",
        str(output_pose),
        "--exhaustiveness",
        str(int(exhaustiveness)),
        "--num_modes",
        str(int(num_modes)),
        "--energy_range",
        str(float(energy_range)),
        "--cpu",
        str(int(cpu)),
        "--seed",
        str(int(seed)),
    ]

    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=1800)
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        score = parse_best_score_from_pose(output_pose)
        if completed.returncode == 0 and output_pose.exists():
            status = "complete"
            error_message = ""
        else:
            status = "failed"
            error_message = f"returncode={completed.returncode}; stderr={completed.stderr.strip()[:1000]}"
    except Exception as exc:
        score = None
        status = "failed"
        error_message = str(exc)
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(str(exc), encoding="utf-8")

    return {
        "docking_id": docking_id,
        "ligand_pdbqt": str(row["ligand_pdbqt"]),
        "receptor_pdbqt": str(row["receptor_pdbqt"]),
        "pdb_id": str(row["pdb_id"]),
        "output_pose_file": str(output_pose),
        "best_score": score,
        "status": status,
        "error_message": error_message,
    }


def batch_group_columns() -> list[str]:
    return [
        "receptor_pdbqt",
        "pdb_id",
        "center_x",
        "center_y",
        "center_z",
        "size_x",
        "size_y",
        "size_z",
        "output_dir",
    ]


def run_one_receptor_batch(
    group: pd.DataFrame,
    unidock_bin: str,
    log_dir: Path,
    exhaustiveness: int,
    max_step: int,
    num_modes: int,
    energy_range: float,
    cpu: int,
    seed: int,
) -> list[dict[str, Any]]:
    first = group.iloc[0]
    pdb_id = safe_id(first["pdb_id"])
    output_dir = ensure_dir(first["output_dir"])
    ensure_dir(log_dir)
    ligand_index_path = log_dir / f"{pdb_id.lower()}_ligand_index.txt"
    stdout_path = log_dir / f"{pdb_id.lower()}_batch.stdout.log"
    stderr_path = log_dir / f"{pdb_id.lower()}_batch.stderr.log"
    ligand_paths = [str(value) for value in group["ligand_pdbqt"].tolist()]
    ligand_index_path.write_text("\n".join(ligand_paths) + "\n", encoding="utf-8")

    command = [
        unidock_bin,
        "--receptor",
        str(first["receptor_pdbqt"]),
        "--ligand_index",
        str(ligand_index_path),
        "--dir",
        str(output_dir),
        "--center_x",
        str(float(first["center_x"])),
        "--center_y",
        str(float(first["center_y"])),
        "--center_z",
        str(float(first["center_z"])),
        "--size_x",
        str(float(first["size_x"])),
        "--size_y",
        str(float(first["size_y"])),
        "--size_z",
        str(float(first["size_z"])),
        "--exhaustiveness",
        str(int(exhaustiveness)),
        "--max_step",
        str(int(max_step)),
        "--num_modes",
        str(int(num_modes)),
        "--energy_range",
        str(float(energy_range)),
        "--cpu",
        str(int(cpu)),
        "--seed",
        str(int(seed)),
        "--verbosity",
        "1",
    ]

    error_message = ""
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=86400)
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            error_message = f"batch_returncode={completed.returncode}; stderr={completed.stderr.strip()[:1000]}"
    except Exception as exc:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(str(exc), encoding="utf-8")
        error_message = str(exc)

    rows: list[dict[str, Any]] = []
    for _index, row in group.iterrows():
        output_pose = batch_output_pose_path(row)
        score = parse_best_score_from_pose(output_pose)
        status = "complete" if output_pose.exists() and score is not None and not error_message else "failed"
        message = "" if status == "complete" else error_message or f"Missing or unparseable output pose: {output_pose}"
        rows.append(
            {
                "docking_id": safe_id(row["docking_id"]),
                "ligand_pdbqt": str(row["ligand_pdbqt"]),
                "receptor_pdbqt": str(row["receptor_pdbqt"]),
                "pdb_id": str(row["pdb_id"]),
                "output_pose_file": str(output_pose),
                "best_score": score,
                "status": status,
                "error_message": message,
            }
        )
    return rows


def run_unidock_inputs(
    inputs_path: str | Path,
    out_root: str | Path,
    scores_path: str | Path,
    unidock_bin: str | None = None,
    exhaustiveness: int = 8,
    max_step: int = 10,
    num_modes: int = 10,
    energy_range: float = 3.0,
    cpu: int = 4,
    seed: int = 42,
    log_dir: str | Path = "results/reports/unidock_logs",
) -> pd.DataFrame:
    resolved_bin = resolve_unidock_bin(unidock_bin)
    validate_unidock(resolved_bin)
    inputs = pd.read_csv(inputs_path)
    out_root = ensure_dir(out_root)
    scores_path = Path(scores_path)
    ensure_dir(scores_path.parent)
    log_dir = ensure_dir(log_dir)

    rows: list[dict[str, Any]] = []
    if inputs.empty:
        frame = pd.DataFrame(columns=RAW_SCORE_COLUMNS)
        frame.to_csv(scores_path, index=False)
        return frame

    grouped_inputs = list(inputs.groupby(batch_group_columns(), sort=True, dropna=False))
    total_groups = len(grouped_inputs)
    for group_index, (_group_key, group) in enumerate(grouped_inputs, start=1):
        pdb_id = str(group.iloc[0]["pdb_id"]) if not group.empty else f"group_{group_index}"
        print(f"Running Uni-Dock batch {group_index}/{total_groups}: {pdb_id} ({len(group)} ligands)", flush=True)
        rows.extend(
            run_one_receptor_batch(
                group.copy(),
                resolved_bin,
                log_dir,
                exhaustiveness=exhaustiveness,
                max_step=max_step,
                num_modes=num_modes,
                energy_range=energy_range,
                cpu=cpu,
                seed=seed,
            )
        )
        partial_frame = pd.DataFrame(rows, columns=RAW_SCORE_COLUMNS)
        partial_frame.to_csv(scores_path, index=False)
        complete_count = int((partial_frame["status"] == "complete").sum()) if not partial_frame.empty else 0
        print(f"Finished batch {group_index}/{total_groups}: cumulative complete={complete_count}/{len(partial_frame)}", flush=True)

    frame = pd.DataFrame(rows, columns=RAW_SCORE_COLUMNS)
    frame.to_csv(scores_path, index=False)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Uni-Dock on a prepared docking input table.")
    parser.add_argument("--inputs", default="results/tables/mapk1_phase1_docking_inputs_smoke.csv")
    parser.add_argument("--out-root", default="results/poses/unidock/MAPK1/phase1_smoke")
    parser.add_argument("--scores", default="results/tables/mapk1_phase1_unidock_smoke_raw.csv")
    parser.add_argument("--unidock-bin", default=None)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--max-step", type=int, default=10)
    parser.add_argument("--num-modes", type=int, default=10)
    parser.add_argument("--energy-range", type=float, default=3.0)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", default="results/reports/unidock_logs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_unidock_inputs(
        args.inputs,
        args.out_root,
        args.scores,
        unidock_bin=args.unidock_bin,
        exhaustiveness=args.exhaustiveness,
        max_step=args.max_step,
        num_modes=args.num_modes,
        energy_range=args.energy_range,
        cpu=args.cpu,
        seed=args.seed,
        log_dir=args.log_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
