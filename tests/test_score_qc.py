from pathlib import Path

import pandas as pd


from screensift.metrics.score_qc import add_score_anomaly_flags, anomaly_counts


def test_score_qc_flags_expected_anomalies() -> None:
    df = pd.DataFrame(
        [
            {
                "ligand_id": "ok",
                "CNNscore": 0.4,
                "CNNaffinity": 5.0,
                "gnina_affinity": -8.0,
                "unidock_best_score": -7.0,
                "heavy_atoms": 20,
            },
            {
                "ligand_id": "bad",
                "CNNscore": 1.2,
                "CNNaffinity": 25.0,
                "gnina_affinity": 84.0,
                "unidock_best_score": -29.0,
                "heavy_atoms": 20,
            },
        ]
    )

    result = add_score_anomaly_flags(df)
    bad = result[result["ligand_id"] == "bad"].iloc[0]
    counts = anomaly_counts(result)

    assert bool(bad["out_of_range_cnnscore"])
    assert bool(bad["suspicious_cnnaffinity_extreme"])
    assert bool(bad["positive_gnina_affinity"])
    assert bool(bad["extreme_positive_gnina_affinity"])
    assert bool(bad["suspicious_unidock_extreme_negative"])
    assert bool(bad["extreme_unidock_negative"])
    assert bool(bad["suspicious_ligand_efficiency_extreme"])
    assert counts["extreme_positive_gnina_affinity"] == 1
    assert counts["extreme_unidock_negative"] == 1
