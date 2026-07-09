from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem

from screensift.common.external import run_command, shell_join
from screensift.common.io import ensure_dir, markdown_table
from screensift.docking.run_unidock import parse_best_score_from_pose
from screensift.validation.rmsd import pose_rmsd


OUTPUT_COLUMNS = [
    "pdb_id",
    "engine",
    "attempt_name",
    "attempt_index",
    "exhaustiveness",
    "box_scale",
    "box_padding_angstrom",
    "size_x",
    "size_y",
    "size_z",
    "native_ligand_file",
    "native_ligand_pdbqt",
    "receptor_pdbqt",
    "redocked_pose_file",
    "redocking_score",
    "rmsd_angstrom",
    "redocking_success",
    "status",
    "failure_reason",
    "command",
]


def _attempt_slug(value: Any, index: int) -> str:
    text = str(value or f"attempt_{index}").strip().lower()
    safe = "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")
    return safe or f"attempt_{index}"


def redocking_attempts(
    attempts: list[dict[str, Any]] | None,
    exhaustiveness: int,
    num_modes: int,
    energy_range: float,
    seed: int,
) -> list[dict[str, Any]]:
    if attempts is None:
        return [
            {
                "name": "configured",
                "exhaustiveness": exhaustiveness,
                "num_modes": num_modes,
                "energy_range": energy_range,
                "seed": seed,
                "box_scale": 1.0,
                "box_padding_angstrom": 0.0,
                "size_x": None,
                "size_y": None,
                "size_z": None,
            }
        ]
    if not attempts:
        raise ValueError("redocking.native.auto_tune.attempts must contain at least one user-defined attempt.")

    normalized: list[dict[str, Any]] = []
    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, dict):
            raise ValueError("Each redocking auto_tune attempt must be a mapping.")
        normalized.append(
            {
                "name": attempt.get("name") or f"attempt_{index}",
                "exhaustiveness": int(attempt.get("exhaustiveness", exhaustiveness)),
                "num_modes": int(attempt.get("num_modes", num_modes)),
                "energy_range": float(attempt.get("energy_range", energy_range)),
                "seed": int(attempt.get("seed", seed)),
                "box_scale": float(attempt.get("box_scale", 1.0)),
                "box_padding_angstrom": float(attempt.get("box_padding_angstrom", 0.0)),
                "size_x": attempt.get("size_x"),
                "size_y": attempt.get("size_y"),
                "size_z": attempt.get("size_z"),
            }
        )
    return normalized


def box_for_attempt(box: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
    tuned = dict(box)
    scale = float(attempt.get("box_scale", 1.0))
    padding = float(attempt.get("box_padding_angstrom", 0.0))
    for axis in ["x", "y", "z"]:
        key = f"size_{axis}"
        explicit = attempt.get(key)
        if explicit is not None:
            tuned[key] = float(explicit)
        else:
            tuned[key] = (float(box[key]) * scale) + (2.0 * padding)
    return tuned


def _native_pdbqt_path(native_ligand_file: str | Path) -> Path:
    path = Path(native_ligand_file)
    return path.with_suffix(".pdbqt")


def prepare_native_ligand_pdbqt(native_ligand_file: str | Path, out_path: str | Path) -> tuple[Path | None, str]:
    native = Path(native_ligand_file)
    out = Path(out_path)
    if out.exists() and out.stat().st_size > 0:
        return out, "existing"
    if not native.exists():
        return None, "missing_native_ligand_file"

    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
    except Exception as exc:
        return None, f"meeko_import_failed:{exc}"

    suffix = native.suffix.lower()
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(native), removeHs=False, sanitize=True)
        mol = next((candidate for candidate in supplier if candidate is not None), None)
    elif suffix in {".pdb", ".ent"}:
        mol = Chem.MolFromPDBFile(str(native), removeHs=False, sanitize=True)
    else:
        mol = Chem.MolFromMolFile(str(native), removeHs=False, sanitize=True)
    if mol is None:
        return None, "rdkit_read_failed"

    try:
        mol = Chem.AddHs(mol, addCoords=True)
        setups = MoleculePreparation().prepare(mol)
        if not setups:
            return None, "meeko_no_setup"
        written = PDBQTWriterLegacy.write_string(setups[0])
        if isinstance(written, tuple):
            pdbqt_text = written[0]
            is_ok = bool(written[1]) if len(written) > 1 else True
            error_msg = str(written[2]) if len(written) > 2 else ""
            if not is_ok:
                return None, error_msg or "meeko_write_failed"
        else:
            pdbqt_text = str(written)
        ensure_dir(out.parent)
        out.write_text(pdbqt_text, encoding="utf-8")
        return out, "meeko_api"
    except Exception as exc:
        return None, f"native_pdbqt_conversion_failed:{exc}"


def build_unidock_redocking_command(
    unidock_bin: str,
    receptor_pdbqt: str | Path,
    ligand_pdbqt: str | Path,
    box: dict[str, Any],
    output_pose: str | Path,
    cpu: int = 1,
    exhaustiveness: int = 8,
    num_modes: int = 10,
    energy_range: float = 3,
    seed: int = 42,
) -> list[str]:
    return [
        unidock_bin,
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(ligand_pdbqt),
        "--center_x",
        str(float(box["center_x"])),
        "--center_y",
        str(float(box["center_y"])),
        "--center_z",
        str(float(box["center_z"])),
        "--size_x",
        str(float(box["size_x"])),
        "--size_y",
        str(float(box["size_y"])),
        "--size_z",
        str(float(box["size_z"])),
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


def redock_native_ligands(
    boxes_path: str | Path,
    out_path: str | Path,
    report_path: str | Path,
    pose_dir: str | Path,
    log_dir: str | Path,
    unidock_bin: str = "unidock",
    rmsd_threshold_angstrom: float = 2.0,
    cpu: int = 1,
    exhaustiveness: int = 8,
    num_modes: int = 10,
    energy_range: float = 3,
    seed: int = 42,
    auto_tune_attempts: list[dict[str, Any]] | None = None,
    attempts_out_path: str | Path | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    boxes = pd.read_csv(boxes_path)
    attempts = redocking_attempts(auto_tune_attempts, exhaustiveness, num_modes, energy_range, seed)
    final_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    pose_root = ensure_dir(pose_dir)
    log_root = ensure_dir(log_dir)

    for box in boxes.to_dict(orient="records"):
        pdb_id = str(box.get("pdb_id", "")).lower()
        native_ligand = Path(str(box.get("ligand_file", "")))
        native_pdbqt = _native_pdbqt_path(native_ligand)
        receptor_pdbqt = Path(str(box.get("receptor_pdbqt", "")))
        reasons: list[str] = []
        if not native_ligand.exists():
            reasons.append("missing_native_ligand_file")
        else:
            prepared_pdbqt, prep_status = prepare_native_ligand_pdbqt(native_ligand, native_pdbqt)
            if prepared_pdbqt is None:
                reasons.append(f"native_ligand_pdbqt_failed:{prep_status}")
        if not receptor_pdbqt.exists():
            reasons.append("missing_receptor_pdbqt")

        if reasons:
            attempt = attempts[0]
            tuned_box = box_for_attempt(box, attempt)
            output_pose = pose_root / pdb_id / f"{_attempt_slug(attempt.get('name'), 1)}_native_redocked.pdbqt"
            command = build_unidock_redocking_command(
                unidock_bin,
                receptor_pdbqt,
                native_pdbqt,
                tuned_box,
                output_pose,
                cpu=cpu,
                exhaustiveness=int(attempt["exhaustiveness"]),
                num_modes=int(attempt["num_modes"]),
                energy_range=float(attempt["energy_range"]),
                seed=int(attempt["seed"]),
            )
            row = _row(box, attempt, 1, tuned_box, native_ligand, native_pdbqt, receptor_pdbqt, output_pose, None, None, False, "failed", ";".join(reasons), command)
            final_rows.append(row)
            attempt_rows.append(row)
            continue

        final_row: dict[str, Any] | None = None
        for attempt_index, attempt in enumerate(attempts, start=1):
            attempt_slug = _attempt_slug(attempt.get("name"), attempt_index)
            tuned_box = box_for_attempt(box, attempt)
            output_pose = pose_root / pdb_id / f"{attempt_slug}_native_redocked.pdbqt"
            ensure_dir(output_pose.parent)
            command = build_unidock_redocking_command(
                unidock_bin,
                receptor_pdbqt,
                native_pdbqt,
                tuned_box,
                output_pose,
                cpu=cpu,
                exhaustiveness=int(attempt["exhaustiveness"]),
                num_modes=int(attempt["num_modes"]),
                energy_range=float(attempt["energy_range"]),
                seed=int(attempt["seed"]),
            )

            result = run_command(
                command,
                stdout_path=log_root / f"{pdb_id}_{attempt_slug}_native_redocking.stdout.log",
                stderr_path=log_root / f"{pdb_id}_{attempt_slug}_native_redocking.stderr.log",
                dry_run=dry_run,
            )
            score = parse_best_score_from_pose(output_pose)
            rmsd = pose_rmsd(native_ligand, output_pose) if result.status == "complete" else None
            rmsd_value = rmsd.rmsd_angstrom if rmsd else None
            success = bool(result.status == "complete" and rmsd and rmsd.status == "complete" and rmsd.rmsd_angstrom <= rmsd_threshold_angstrom)
            failure = result.error_message or (rmsd.failure_reason if rmsd and rmsd.status != "complete" else "")
            row = _row(box, attempt, attempt_index, tuned_box, native_ligand, native_pdbqt, receptor_pdbqt, output_pose, score, rmsd_value, success, result.status, failure, command)
            attempt_rows.append(row)
            final_row = row
            if success:
                break
        if final_row is not None:
            final_rows.append(final_row)

    attempt_frame = pd.DataFrame(attempt_rows, columns=OUTPUT_COLUMNS)
    frame = pd.DataFrame(final_rows, columns=OUTPUT_COLUMNS)
    out = Path(out_path)
    ensure_dir(out.parent)
    frame.to_csv(out, index=False)
    if attempts_out_path is not None:
        attempts_out = Path(attempts_out_path)
        ensure_dir(attempts_out.parent)
        attempt_frame.to_csv(attempts_out, index=False)
    write_report(frame, report_path, rmsd_threshold_angstrom, attempt_frame=attempt_frame)
    return frame


def _row(
    box: dict[str, Any],
    attempt: dict[str, Any],
    attempt_index: int,
    tuned_box: dict[str, Any],
    native_ligand: Path,
    native_pdbqt: Path,
    receptor_pdbqt: Path,
    output_pose: Path,
    score: float | None,
    rmsd: float | None,
    success: bool,
    status: str,
    failure_reason: str,
    command: list[str],
) -> dict[str, Any]:
    return {
        "pdb_id": box.get("pdb_id"),
        "engine": "unidock",
        "attempt_name": attempt.get("name"),
        "attempt_index": attempt_index,
        "exhaustiveness": attempt.get("exhaustiveness"),
        "box_scale": attempt.get("box_scale"),
        "box_padding_angstrom": attempt.get("box_padding_angstrom"),
        "size_x": tuned_box.get("size_x"),
        "size_y": tuned_box.get("size_y"),
        "size_z": tuned_box.get("size_z"),
        "native_ligand_file": str(native_ligand),
        "native_ligand_pdbqt": str(native_pdbqt),
        "receptor_pdbqt": str(receptor_pdbqt),
        "redocked_pose_file": str(output_pose),
        "redocking_score": score,
        "rmsd_angstrom": rmsd,
        "redocking_success": success,
        "status": status,
        "failure_reason": failure_reason,
        "command": shell_join(command),
    }


def write_report(frame: pd.DataFrame, report_path: str | Path, threshold: float, attempt_frame: pd.DataFrame | None = None) -> None:
    path = Path(report_path)
    ensure_dir(path.parent)
    success = int(frame["redocking_success"].sum()) if "redocking_success" in frame.columns else 0
    attempts = attempt_frame if attempt_frame is not None else frame
    failed_receptors: list[str] = []
    if not frame.empty and "redocking_success" in frame.columns:
        failed_receptors = frame.loc[~frame["redocking_success"].astype(bool), "pdb_id"].astype(str).tolist()
    lines = [
        "# Native Ligand Redocking Report",
        "",
        f"- rows: {len(frame)}",
        f"- attempted_redocking_rows: {len(attempts)}",
        f"- redocking_success_count: {success}",
        f"- rmsd_success_threshold_angstrom: {threshold}",
        "",
    ]
    if failed_receptors:
        lines.extend([f"- failed_after_all_attempts: {', '.join(failed_receptors)}", ""])
    lines.extend(
        [
            "## Final receptor status",
            "",
            markdown_table(frame, ["pdb_id", "attempt_name", "exhaustiveness", "size_x", "size_y", "size_z", "redocking_score", "rmsd_angstrom", "redocking_success", "status", "failure_reason"], max_rows=50),
            "",
            "## Attempt log",
            "",
            markdown_table(attempts, ["pdb_id", "attempt_name", "attempt_index", "exhaustiveness", "size_x", "size_y", "size_z", "redocking_score", "rmsd_angstrom", "redocking_success", "status", "failure_reason"], max_rows=200),
        ]
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redock native ligands into their receptors with Uni-Dock and compute RMSD.")
    parser.add_argument("--boxes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--pose-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--unidock-bin", default="unidock")
    parser.add_argument("--rmsd-threshold", type=float, default=2.0)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--num-modes", type=int, default=10)
    parser.add_argument("--energy-range", type=float, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = redock_native_ligands(
        args.boxes,
        args.out,
        args.report,
        args.pose_dir,
        args.log_dir,
        unidock_bin=args.unidock_bin,
        rmsd_threshold_angstrom=args.rmsd_threshold,
        cpu=args.cpu,
        exhaustiveness=args.exhaustiveness,
        num_modes=args.num_modes,
        energy_range=args.energy_range,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    print(f"Native redocking rows={len(frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
