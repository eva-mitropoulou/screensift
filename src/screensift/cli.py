from __future__ import annotations

import argparse
from pathlib import Path

from screensift import find_candidates
from screensift.common.io import ensure_dir


def parse_score_column(value: str) -> tuple[str, str]:
    if ":" not in value:
        return value, "higher"
    column, direction = value.split(":", 1)
    direction = direction.strip().lower()
    if direction not in {"higher", "lower"}:
        raise argparse.ArgumentTypeError("Score direction must be 'higher' or 'lower'.")
    return column.strip(), direction


def parse_weight(value: str) -> tuple[str, float]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Weights must be formatted as NAME:WEIGHT.")
    name, weight = value.split(":", 1)
    try:
        parsed = float(weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Weight must be numeric.") from exc
    return name.strip(), parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find candidates from a schema-adapted ligand dataset.")
    parser.add_argument("--schema", required=True, help="Dataset schema YAML.")
    parser.add_argument("--data", default=None, help="Ligand table. If omitted, schema input.path is used.")
    parser.add_argument("--target", default="", help="Target identifier.")
    parser.add_argument(
        "--evidence-mode",
        choices=["similarity", "structure", "combined"],
        default="combined",
        help="Evidence channel used for ranking. 'combined' can use both similarity and structure evidence.",
    )
    parser.add_argument(
        "--structure-score",
        action="append",
        type=parse_score_column,
        default=[],
        help="Structure score column as COLUMN:higher or COLUMN:lower. Can be repeated.",
    )
    parser.add_argument("--similarity-score", default=None, help="Optional precomputed similarity score column.")
    parser.add_argument("--structure-aggregation", choices=["max", "weighted_mean"], default="max")
    parser.add_argument(
        "--structure-weight",
        action="append",
        type=parse_weight,
        default=[],
        help="Structure evidence weight as COLUMN:WEIGHT. Required for --structure-aggregation weighted_mean.",
    )
    parser.add_argument("--candidate-aggregation", choices=["max", "weighted_mean"], default="max")
    parser.add_argument(
        "--candidate-weight",
        action="append",
        type=parse_weight,
        default=[],
        help="Candidate evidence weight as structure_score_norm:WEIGHT or similarity_score_norm:WEIGHT. Required for --candidate-aggregation weighted_mean.",
    )
    parser.add_argument("--structure-cutoff", type=float, default=0.70)
    parser.add_argument("--similarity-cutoff", type=float, default=0.70)
    parser.add_argument("--n-candidates", type=int, default=100)
    parser.add_argument("--out", required=True, help="Output candidate CSV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    structure_scores = dict(args.structure_score)
    candidates = find_candidates(
        schema=args.schema,
        data=args.data,
        target=args.target,
        evidence_mode=args.evidence_mode,
        structure_score_columns=structure_scores or None,
        similarity_score_column=args.similarity_score,
        structure_aggregation=args.structure_aggregation,
        structure_weights=dict(args.structure_weight) or None,
        candidate_aggregation=args.candidate_aggregation,
        candidate_weights=dict(args.candidate_weight) or None,
        structure_score_cutoff=args.structure_cutoff,
        similarity_score_cutoff=args.similarity_cutoff,
        n_candidates=args.n_candidates,
    )
    out = Path(args.out)
    ensure_dir(out.parent)
    candidates.to_csv(out, index=False)
    print(f"Wrote {len(candidates)} candidates to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
