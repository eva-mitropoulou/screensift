from pathlib import Path

import pandas as pd


from screensift.validation.pre_step8_baseline_cleanup import (  # noqa: E402
    CCD_NATIVE_STATUS,
    PDB_NATIVE_STATUS,
    cleanup_baseline_tables,
    mark_baseline_status,
)


def test_pdb_native_baseline_rows_are_marked_deprecated() -> None:
    df = pd.DataFrame(
        {
            "baseline": ["native_ligand_ecfp4", "all_active_loo_ecfp4_analog_neighborhood"],
            "method": ["native_ligand_ecfp4", "unidock_best"],
        }
    )

    marked, counts = mark_baseline_status(df)

    assert counts["pdb_native_rows_deprecated"] == 1
    assert marked.loc[0, "baseline_status"] == PDB_NATIVE_STATUS


def test_ccd_native_baseline_rows_are_marked_official() -> None:
    df = pd.DataFrame(
        {
            "population": ["full"],
            "method": ["native_ligand_ecfp4_ccd"],
            "score_col": ["native_ligand_ecfp4_ccd"],
        }
    )

    marked, counts = mark_baseline_status(df)

    assert counts["ccd_native_rows_marked_official"] == 1
    assert marked.loc[0, "baseline_status"] == CCD_NATIVE_STATUS


def test_cleanup_updates_matching_tables(tmp_path: Path) -> None:
    table = tmp_path / "mapk1_phase1_objective_2d_single_baseline_metrics_full.csv"
    pd.DataFrame(
        {
            "baseline": ["native_ligand_ecfp4", "native_ligand_ecfp4_ccd"],
            "method": ["native_ligand_ecfp4", "native_ligand_ecfp4_ccd"],
        }
    ).to_csv(table, index=False)

    summary = cleanup_baseline_tables(tmp_path)
    updated = pd.read_csv(table)

    assert summary["pdb_native_deprecated"] is True
    assert summary["ccd_native_official"] is True
    assert PDB_NATIVE_STATUS in set(updated["baseline_status"])
    assert CCD_NATIVE_STATUS in set(updated["baseline_status"])
