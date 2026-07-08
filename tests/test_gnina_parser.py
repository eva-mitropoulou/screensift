from pathlib import Path

import pandas as pd


from screensift.rescoring.parse_gnina_scores import parse_gnina_scores, parse_gnina_stdout_text


def test_parse_gnina_labeled_scores() -> None:
    text = """
Affinity: -8.24
CNNscore: 0.731
CNNaffinity: 7.12
Intramolecular energy: 1.42
"""
    parsed = parse_gnina_stdout_text(text)

    assert parsed["affinity"] == -8.24
    assert parsed["cnnscore"] == 0.731
    assert parsed["cnnaffinity"] == 7.12
    assert parsed["intramolecular_energy"] == 1.42


def test_parse_gnina_missing_fields_are_none() -> None:
    parsed = parse_gnina_stdout_text("No scores here\n")

    assert parsed["affinity"] is None
    assert parsed["cnnscore"] is None
    assert parsed["cnnaffinity"] is None


def test_parse_gnina_scores_from_log_file(tmp_path: Path) -> None:
    log = tmp_path / "gnina.stdout.log"
    log.write_text("CNNscore: 0.52\nCNNaffinity: 6.4\nAffinity: -7.8\n", encoding="utf-8")
    raw = pd.DataFrame(
        [
            {
                "ligand_id": "lig1",
                "activity_label": "active",
                "pdb_id": "AAAA",
                "best_score_unidock": -9.1,
                "gnina_stdout_log": str(log),
                "status": "complete",
                "error_message": "",
            }
        ]
    )
    raw_path = tmp_path / "raw.csv"
    out_path = tmp_path / "scores.csv"
    raw.to_csv(raw_path, index=False)

    scores = parse_gnina_scores(raw_path, out_path)

    assert out_path.exists()
    assert scores.loc[0, "cnnscore"] == 0.52
    assert scores.loc[0, "cnnaffinity"] == 6.4
    assert scores.loc[0, "CNN_VS"] == 0.52 * 6.4
    assert scores.loc[0, "affinity"] == -7.8
