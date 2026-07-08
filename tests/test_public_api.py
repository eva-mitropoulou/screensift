from pathlib import Path

import pandas as pd


from screensift import find_candidates


def _schema() -> dict:
    return {
        "columns": {
            "smiles": "smiles",
            "ligand_id": "compound_id",
            "activity_label": "label",
        },
        "activity": {
            "mode": "label",
            "active_values": ["active"],
            "inactive_values": ["inactive"],
        },
        "deduplicate_by": "inchikey",
    }


def test_find_candidates_returns_ranked_evidence_buckets() -> None:
    data = pd.DataFrame(
        [
            {"compound_id": "analog", "smiles": "CCO", "label": "active", "tanimoto": 0.95, "CNNscore": 0.2},
            {"compound_id": "structure", "smiles": "CCN", "label": "inactive", "tanimoto": 0.10, "CNNscore": 0.9},
            {"compound_id": "consensus", "smiles": "CCC", "label": "active", "tanimoto": 0.85, "CNNscore": 0.8},
            {"compound_id": "low", "smiles": "CCCC", "label": "inactive", "tanimoto": 0.05, "CNNscore": 0.1},
        ]
    )

    candidates = find_candidates(
        schema=_schema(),
        data=data,
        target="toy",
        structure_score_columns={"CNNscore": "higher"},
        similarity_score_column="tanimoto",
        structure_score_cutoff=0.70,
        similarity_score_cutoff=0.70,
        n_candidates=3,
    )

    assert candidates["ligand_id"].tolist() == ["structure", "analog", "consensus"]
    assert set(candidates["evidence_bucket"]) == {"analog_supported", "structure_supported", "consensus_supported"}
    assert candidates["candidate_score"].is_monotonic_decreasing
    assert candidates["target"].tolist() == ["toy", "toy", "toy"]


def test_find_candidates_uses_common_score_columns_by_default() -> None:
    data = pd.DataFrame(
        [
            {"compound_id": "best_unidock", "smiles": "CCO", "label": "active", "unidock_best_score": -11.0},
            {"compound_id": "weak_unidock", "smiles": "CCN", "label": "inactive", "unidock_best_score": -6.0},
        ]
    )

    candidates = find_candidates(schema=_schema(), data=data, target="toy", n_candidates=1)

    assert candidates.loc[0, "ligand_id"] == "best_unidock"
    assert candidates.loc[0, "primary_structure_score"] == "unidock_best_score"


def test_find_candidates_computes_similarity_when_score_columns_are_missing() -> None:
    data = pd.DataFrame(
        [
            {"compound_id": "active_ref", "smiles": "CCO", "label": "active"},
            {"compound_id": "near_active", "smiles": "CCCO", "label": "inactive"},
            {"compound_id": "far", "smiles": "c1ccccc1", "label": "inactive"},
        ]
    )

    candidates = find_candidates(
        schema=_schema(),
        data=data,
        target="toy",
        structure_score_columns={},
        similarity_score_cutoff=0.20,
        n_candidates=2,
    )

    assert candidates.loc[0, "ligand_id"] == "near_active"
    assert candidates.loc[0, "evidence_bucket"] == "analog_supported"
    assert candidates.loc[0, "similarity_score_norm"] > 0


def test_similarity_mode_ignores_structure_columns() -> None:
    data = pd.DataFrame(
        [
            {"compound_id": "similarity_hit", "smiles": "CCO", "label": "inactive", "tanimoto": 0.95, "CNNscore": 0.0},
            {"compound_id": "structure_hit", "smiles": "CCN", "label": "inactive", "tanimoto": 0.10, "CNNscore": 1.0},
        ]
    )

    candidates = find_candidates(
        schema=_schema(),
        data=data,
        target="toy",
        evidence_mode="similarity",
        structure_score_columns={"CNNscore": "higher"},
        similarity_score_column="tanimoto",
        n_candidates=1,
    )

    assert candidates.loc[0, "ligand_id"] == "similarity_hit"
    assert candidates.loc[0, "primary_structure_score"] == ""


def test_structure_mode_ignores_similarity_columns() -> None:
    data = pd.DataFrame(
        [
            {"compound_id": "similarity_hit", "smiles": "CCO", "label": "inactive", "tanimoto": 0.95, "CNNscore": 0.0},
            {"compound_id": "structure_hit", "smiles": "CCN", "label": "inactive", "tanimoto": 0.10, "CNNscore": 1.0},
        ]
    )

    candidates = find_candidates(
        schema=_schema(),
        data=data,
        target="toy",
        evidence_mode="structure",
        structure_score_columns={"CNNscore": "higher"},
        similarity_score_column="tanimoto",
        n_candidates=1,
    )

    assert candidates.loc[0, "ligand_id"] == "structure_hit"
    assert pd.isna(candidates.loc[0, "similarity_score_norm"])


def test_find_candidates_supports_weighted_structure_aggregation() -> None:
    data = pd.DataFrame(
        [
            {"compound_id": "cnn_only", "smiles": "CCO", "label": "inactive", "CNNscore": 1.0, "unidock_best_score": -1.0},
            {"compound_id": "balanced", "smiles": "CCN", "label": "inactive", "CNNscore": 0.7, "unidock_best_score": -7.0},
            {"compound_id": "dock_only", "smiles": "CCC", "label": "inactive", "CNNscore": 0.0, "unidock_best_score": -10.0},
        ]
    )

    candidates = find_candidates(
        schema=_schema(),
        data=data,
        target="toy",
        structure_score_columns={"CNNscore": "higher", "unidock_best_score": "lower"},
        structure_aggregation="weighted_mean",
        structure_weights={"CNNscore": 0.25, "unidock_best_score": 0.75},
        compute_similarity_if_missing=False,
        n_candidates=1,
    )

    assert candidates.loc[0, "ligand_id"] == "dock_only"


def test_find_candidates_supports_weighted_candidate_aggregation() -> None:
    data = pd.DataFrame(
        [
            {"compound_id": "structure_only", "smiles": "CCO", "label": "inactive", "tanimoto": 0.0, "CNNscore": 1.0},
            {"compound_id": "balanced", "smiles": "CCN", "label": "inactive", "tanimoto": 0.5, "CNNscore": 0.7},
            {"compound_id": "similarity_only", "smiles": "CCC", "label": "inactive", "tanimoto": 1.0, "CNNscore": 0.0},
        ]
    )

    candidates = find_candidates(
        schema=_schema(),
        data=data,
        target="toy",
        structure_score_columns={"CNNscore": "higher"},
        similarity_score_column="tanimoto",
        candidate_aggregation="weighted_mean",
        candidate_weights={"structure_score_norm": 0.25, "similarity_score_norm": 0.75},
        n_candidates=1,
    )

    assert candidates.loc[0, "ligand_id"] == "similarity_only"


def test_combined_mode_handles_ligands_with_no_structure_score() -> None:
    # A ligand whose every structure column is NaN (failed docking) must not
    # crash find_candidates via idxmax(axis=1) "all NA" on modern pandas.
    data = pd.DataFrame(
        [
            {"compound_id": "docked", "smiles": "CCO", "label": "active", "CNNscore": 0.9, "tanimoto": 0.2},
            {"compound_id": "no_dock", "smiles": "CCN", "label": "inactive", "CNNscore": None, "tanimoto": 0.9},
        ]
    )

    candidates = find_candidates(
        schema=_schema(),
        data=data,
        target="toy",
        structure_score_columns={"CNNscore": "higher"},
        similarity_score_column="tanimoto",
        n_candidates=2,
    )

    assert len(candidates) == 2
    no_dock = candidates.set_index("ligand_id").loc["no_dock"]
    assert no_dock["primary_structure_score"] == ""
    assert pd.isna(no_dock["structure_score_norm"])


def test_supplied_bounded_similarity_is_not_population_rescaled() -> None:
    # A supplied similarity column that is already in [0, 1] must pass through
    # unchanged so the similarity cutoff keeps its absolute meaning, rather than
    # being min-max stretched so the row-max becomes 1.0.
    data = pd.DataFrame(
        [
            {"compound_id": "a", "smiles": "CCO", "label": "inactive", "tanimoto": 0.10},
            {"compound_id": "b", "smiles": "CCN", "label": "inactive", "tanimoto": 0.30},
        ]
    )

    candidates = find_candidates(
        schema=_schema(),
        data=data,
        target="toy",
        evidence_mode="similarity",
        similarity_score_column="tanimoto",
        compute_similarity_if_missing=False,
        n_candidates=2,
    )

    norms = candidates.set_index("ligand_id")["similarity_score_norm"].to_dict()
    assert abs(norms["a"] - 0.10) < 1e-9
    assert abs(norms["b"] - 0.30) < 1e-9  # not rescaled to 1.0


def test_weighted_aggregation_requires_user_weights() -> None:
    data = pd.DataFrame(
        [
            {"compound_id": "a", "smiles": "CCO", "label": "inactive", "CNNscore": 1.0},
            {"compound_id": "b", "smiles": "CCN", "label": "inactive", "CNNscore": 0.0},
        ]
    )

    try:
        find_candidates(
            schema=_schema(),
            data=data,
            target="toy",
            structure_score_columns={"CNNscore": "higher"},
            structure_aggregation="weighted_mean",
        )
    except ValueError as exc:
        assert "requires explicit weights" in str(exc)
    else:
        raise AssertionError("weighted structure aggregation should require explicit weights")
