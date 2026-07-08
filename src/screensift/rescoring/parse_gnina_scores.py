from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


from screensift.common.io import ensure_dir


OUTPUT_COLUMNS = [
    "ligand_id",
    "activity_label",
    "pdb_id",
    "best_score_unidock",
    "cnnscore",
    "cnnaffinity",
    "CNN_VS",
    "affinity",
    "intramolecular_energy",
    "status",
    "error_message",
]


FLOAT_PATTERN = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
LABEL_PATTERNS = {
    "cnnscore": [re.compile(rf"\bCNNscore\b\s*[:=]\s*{FLOAT_PATTERN}", re.IGNORECASE)],
    "cnnaffinity": [re.compile(rf"\bCNNaffinity\b\s*[:=]\s*{FLOAT_PATTERN}", re.IGNORECASE)],
    "affinity": [re.compile(rf"(?<!CNN)\bAffinity\b\s*[:=]\s*{FLOAT_PATTERN}", re.IGNORECASE)],
    "intramolecular_energy": [
        re.compile(rf"\bIntramolecular\s+energy\b\s*[:=]\s*{FLOAT_PATTERN}", re.IGNORECASE),
        re.compile(rf"\bintra(?:molecular)?(?:[-_ ]energy)?\b\s*[:=]\s*{FLOAT_PATTERN}", re.IGNORECASE),
    ],
}


def to_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def parse_score_table_line(text: str, parsed: dict[str, float | None]) -> None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header_columns: list[str] | None = None
    for line in lines:
        if line.startswith("#"):
            normalized = line.lstrip("#").strip()
            if "CNNscore" in normalized and "CNNaffinity" in normalized:
                header_columns = normalized.split()
            continue
        if not header_columns:
            continue
        values = line.split()
        if len(values) < len(header_columns):
            continue
        value_map = dict(zip(header_columns, values, strict=False))
        for output_key, header_key in [
            ("cnnscore", "CNNscore"),
            ("cnnaffinity", "CNNaffinity"),
            ("affinity", "Affinity"),
            ("intramolecular_energy", "Intramolecular"),
        ]:
            if parsed.get(output_key) is None and header_key in value_map:
                parsed[output_key] = to_float(value_map[header_key])


def parse_gnina_stdout_text(text: str) -> dict[str, float | None]:
    parsed: dict[str, float | None] = {
        "cnnscore": None,
        "cnnaffinity": None,
        "affinity": None,
        "intramolecular_energy": None,
    }
    for key, patterns in LABEL_PATTERNS.items():
        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                parsed[key] = to_float(matches[-1])
                break
    parse_score_table_line(text, parsed)
    return parsed


def parse_gnina_scores(raw_path: str | Path, out_path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, dtype={"ligand_id": str})
    rows: list[dict[str, Any]] = []
    for row in raw.to_dict(orient="records"):
        stdout_log = row.get("gnina_stdout_log")
        text = ""
        if stdout_log and not pd.isna(stdout_log):
            path = Path(str(stdout_log))
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="ignore")
        parsed = parse_gnina_stdout_text(text)
        cnn_vs = None
        if parsed["cnnscore"] is not None and parsed["cnnaffinity"] is not None:
            cnn_vs = parsed["cnnscore"] * parsed["cnnaffinity"]
        rows.append(
            {
                "ligand_id": row.get("ligand_id"),
                "activity_label": row.get("activity_label"),
                "pdb_id": row.get("pdb_id"),
                "best_score_unidock": row.get("best_score_unidock"),
                "cnnscore": parsed["cnnscore"],
                "cnnaffinity": parsed["cnnaffinity"],
                "CNN_VS": cnn_vs,
                "affinity": parsed["affinity"],
                "intramolecular_energy": parsed["intramolecular_energy"],
                "status": row.get("status"),
                "error_message": row.get("error_message"),
            }
        )

    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out = Path(out_path)
    ensure_dir(out.parent)
    frame.to_csv(out, index=False)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse GNINA stdout logs into a score table.")
    parser.add_argument("--raw", default="results/tables/mapk1_phase1_gnina_smoke_raw.csv")
    parser.add_argument("--out", default="results/tables/mapk1_phase1_gnina_smoke_scores.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = parse_gnina_scores(args.raw, args.out)
    parsed_count = int(pd.to_numeric(frame["cnnscore"], errors="coerce").notna().sum()) if not frame.empty else 0
    print(f"Parsed GNINA scores: rows={len(frame)} cnnscore_non_null={parsed_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
