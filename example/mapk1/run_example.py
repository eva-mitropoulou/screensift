from __future__ import annotations

from pathlib import Path
from typing import Any

from rdkit import RDLogger

from screensift.common.io import ensure_dir, load_yaml
from screensift import find_candidates


RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
CONFIG = HERE / "pipeline.yml"


def resolve_path(value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    config: dict[str, Any] = load_yaml(CONFIG)
    ranking = config.get("ranking", {})
    target = str(config.get("target", "MAPK1"))
    target_slug = str(config.get("target_slug", target.lower()))
    run_id = str(config.get("run_id", "example"))

    run_dir = resolve_path(config.get("paths", {}).get("run_dir")) or (PROJECT_ROOT / "runs" / f"{target_slug}_{run_id}")
    out = ensure_dir(run_dir / "tables") / f"{target_slug}_{run_id}_candidates.csv"

    candidates = find_candidates(
        schema=resolve_path(config.get("dataset", {}).get("schema")),
        data=resolve_path(ranking.get("score_table")),
        target=target,
        evidence_mode=ranking.get("evidence_mode", "combined"),
        structure_score_columns=ranking.get("structure_score_columns"),
        similarity_score_column=ranking.get("similarity_score_column"),
        structure_aggregation=ranking.get("structure_aggregation", "max"),
        structure_weights=ranking.get("structure_weights"),
        candidate_aggregation=ranking.get("candidate_aggregation", "max"),
        candidate_weights=ranking.get("candidate_weights"),
        n_candidates=int(ranking.get("n_candidates", 100)),
    )
    candidates.to_csv(out, index=False)
    print(f"Wrote {len(candidates)} candidates to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
