from pathlib import Path

import pandas as pd


from screensift.validation.write_manual_pose_verdict_template import REQUIRED_COLUMNS, build_template, run, selected_ligands_from_report  # noqa: E402


def test_selected_ligands_are_parsed_from_manifest(tmp_path: Path) -> None:
    report = tmp_path / "manifest.md"
    report.write_text("<!-- selected_ligands: 26747800, 864048, fp1 -->\n", encoding="utf-8")

    assert selected_ligands_from_report(report) == ["26747800", "864048", "fp1"]


def test_manual_verdict_template_has_required_columns_and_tier_a(tmp_path: Path) -> None:
    triage = pd.DataFrame(
        [
            {
                "ligand_id": "26747800",
                "triage_tier": "A_analog_seed",
                "activity_label": "active",
                "inspection_categories": "novel_sav_tanimoto_lt_0_30",
            },
            {
                "ligand_id": "fp1",
                "triage_tier": "D_failure_analysis",
                "activity_label": "inactive",
                "inspection_categories": "consensus_inactive_false_positive",
            },
        ]
    )

    template = build_template(triage, ["26747800", "fp1"])

    assert list(template.columns) == REQUIRED_COLUMNS
    assert template["ligand_id"].tolist() == ["26747800", "fp1"]
    assert template.loc[0, "manual_pose_verdict"] == "unclear"
    assert template.loc[0, "triage_tier"] == "A_analog_seed"


def test_manual_verdict_template_run_writes_files(tmp_path: Path) -> None:
    triage_path = tmp_path / "triage.csv"
    report_path = tmp_path / "manifest.md"
    out_csv = tmp_path / "manual.csv"
    out_md = tmp_path / "manual.md"
    pd.DataFrame(
        [
            {
                "ligand_id": "26747800",
                "triage_tier": "A_analog_seed",
                "activity_label": "active",
                "inspection_categories": "novel_sav_tanimoto_lt_0_30",
            }
        ]
    ).to_csv(triage_path, index=False)
    report_path.write_text("<!-- selected_ligands: 26747800 -->\n", encoding="utf-8")

    template = run(triage_path, report_path, out_csv, out_md)

    assert len(template) == 1
    assert out_csv.exists()
    assert out_md.exists()
    assert "Manual visual pose review is used" in out_md.read_text(encoding="utf-8")
