from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return an empty dict for empty documents."""
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML config not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML config: {yaml_path}")
    return data


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(data: Any, path: str | Path) -> None:
    """Write JSON with deterministic formatting."""
    json_path = Path(path)
    if json_path.parent:
        ensure_dir(json_path.parent)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    """Render a simple GitHub-flavored Markdown table without optional dependencies."""
    if df.empty:
        return "_None._"
    work = df.copy()
    if columns is not None:
        work = work[[col for col in columns if col in work.columns]]
    if max_rows is not None:
        work = work.head(max_rows)
    if work.empty:
        return "_None._"

    def clean(value: Any) -> str:
        if value is None or pd.isna(value):
            return ""
        text = str(value)
        return text.replace("|", "\\|").replace("\n", " ").strip()

    headers = [str(col) for col in work.columns]
    rows = work.astype(object).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def read_text_table_auto(path: str | Path) -> pd.DataFrame:
    """Read a text table by trying common separators and selecting the best parse."""
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"Table not found: {table_path}")

    attempts: list[str] = []
    parsed: list[tuple[str, pd.DataFrame]] = []
    suffixes = [suffix.lower() for suffix in table_path.suffixes]
    if ".csv" in suffixes:
        separators = (("comma", ","), ("tab", "\t"), ("whitespace", r"\s+"))
    elif ".tsv" in suffixes:
        separators = (("tab", "\t"), ("comma", ","), ("whitespace", r"\s+"))
    else:
        separators = (("whitespace", r"\s+"), ("tab", "\t"), ("comma", ","))

    for sep_name, sep in separators:
        try:
            frame = pd.read_csv(table_path, sep=sep, engine="python")
        except Exception as exc:  # pandas raises several parser-specific errors.
            attempts.append(f"{sep_name}: {exc}")
            continue

        if frame.shape[1] > 0:
            parsed.append((sep_name, frame))
            continue
        attempts.append(f"{sep_name}: parsed zero columns")

    if parsed:
        return max(parsed, key=lambda item: item[1].shape[1])[1]

    detail = "; ".join(attempts) if attempts else "no parser attempts were made"
    raise ValueError(f"Could not parse table {table_path}. Tried comma, tab, and whitespace. Details: {detail}")
