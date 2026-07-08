from pathlib import Path

import pandas as pd
import yaml


from screensift.metrics.anomaly_utils import load_or_reconstruct_anomaly_flags


def test_clean_population_exclusion_from_reconstructed_flags(tmp_path: Path) -> None:
    score_population = pd.DataFrame(
        [
            {
                "ligand_id": "extreme_unidock",
                "unidock_best_score": -25.0,
                "CNNscore": 0.5,
                "CNNaffinity": 5.0,
                "gnina_affinity": -8.0,
                "heavy_atoms": 20,
            },
            {
                "ligand_id": "suspicious_unidock",
                "unidock_best_score": -16.0,
                "CNNscore": 0.5,
                "CNNaffinity": 5.0,
                "gnina_affinity": -8.0,
                "heavy_atoms": 20,
            },
            {
                "ligand_id": "lig_eff",
                "unidock_best_score": -18.0,
                "CNNscore": 0.5,
                "CNNaffinity": 5.0,
                "gnina_affinity": -8.0,
                "heavy_atoms": 20,
            },
            {
                "ligand_id": "extreme_gnina",
                "unidock_best_score": -7.0,
                "CNNscore": 0.5,
                "CNNaffinity": 5.0,
                "gnina_affinity": 25.0,
                "heavy_atoms": 20,
            },
            {
                "ligand_id": "positive_only",
                "unidock_best_score": -7.0,
                "CNNscore": 0.5,
                "CNNaffinity": 5.0,
                "gnina_affinity": 5.0,
                "heavy_atoms": 20,
            },
        ]
    )
    metrics_config = tmp_path / "metrics.yml"
    metrics_config.write_text(
        yaml.safe_dump(
            {
                "anomaly_thresholds": {
                    "cnnscore_min": 0.0,
                    "cnnscore_max": 1.0,
                    "cnnaffinity_min": -5.0,
                    "cnnaffinity_max": 20.0,
                    "gnina_affinity_positive_threshold": 0.0,
                    "gnina_affinity_extreme_positive_threshold": 20.0,
                    "unidock_suspicious_negative_threshold": -15.0,
                    "unidock_extreme_negative_threshold": -20.0,
                    "unidock_positive_threshold": 0.0,
                    "ligand_efficiency_extreme_negative_threshold": -0.8,
                }
            }
        ),
        encoding="utf-8",
    )

    result = load_or_reconstruct_anomaly_flags(score_population, tmp_path, metrics_config)
    by_id = result.set_index("ligand_id")

    assert bool(by_id.loc["extreme_unidock", "clean_population_excluded"])
    assert bool(by_id.loc["suspicious_unidock", "clean_population_excluded"])
    assert bool(by_id.loc["lig_eff", "clean_population_excluded"])
    assert bool(by_id.loc["extreme_gnina", "clean_population_excluded"])
    assert not bool(by_id.loc["positive_only", "clean_population_excluded"])
