from pathlib import Path

import pandas as pd


from screensift.curation.adapt_dataset_schema import adapt_dataset


def test_adapts_labeled_csv_schema(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "\n".join(
            [
                "compound,smiles,label",
                "a,CCO,active",
                "b,CCC,inactive",
                "bad,not_a_smiles,active",
            ]
        ),
        encoding="utf-8",
    )
    schema = tmp_path / "schema.yml"
    schema.write_text(
        "\n".join(
            [
                "input:",
                f"  path: {raw}",
                "  sep: ','",
                "columns:",
                "  smiles: smiles",
                "  ligand_id: compound",
                "  activity_label: label",
                "activity:",
                "  mode: label",
                "  active_values: ['active']",
                "  inactive_values: ['inactive']",
                "deduplicate_by: inchikey",
            ]
        ),
        encoding="utf-8",
    )

    manifest = adapt_dataset(
        schema,
        "toy",
        out_audit=tmp_path / "audit.csv",
        out_curated=tmp_path / "curated.csv",
        out_failures=tmp_path / "failures.csv",
        manifest_path=tmp_path / "manifest.json",
    )

    curated = pd.read_csv(tmp_path / "curated.csv")
    failures = pd.read_csv(tmp_path / "failures.csv")

    assert manifest["curated_rows"] == 2
    assert set(curated["ligand_id"]) == {"a", "b"}
    assert set(curated["activity_label"]) == {"active", "inactive"}
    assert failures["failure_reason"].tolist() == ["invalid_smiles"]


def test_adapts_chembl_like_threshold_schema(tmp_path: Path) -> None:
    raw = tmp_path / "chembl.csv"
    raw.write_text(
        "\n".join(
            [
                "molecule_chembl_id,canonical_smiles,standard_value,standard_relation,standard_units",
                "CHEMBL1,CCO,50,=,nM",
                "CHEMBL2,CCC,50000,=,nM",
                "CHEMBL3,CCN,5000,=,nM",
                "CHEMBL4,CCCl,1000,>,nM",
            ]
        ),
        encoding="utf-8",
    )
    schema = tmp_path / "chembl_schema.yml"
    schema.write_text(
        "\n".join(
            [
                "input:",
                f"  path: {raw}",
                "  sep: ','",
                "columns:",
                "  smiles: canonical_smiles",
                "  ligand_id: molecule_chembl_id",
                "  activity_value: standard_value",
                "  activity_relation: standard_relation",
                "  activity_units: standard_units",
                "activity:",
                "  mode: threshold",
                "  active_if:",
                "    column: standard_value",
                "    operator: '<='",
                "    value: 1000",
                "    allowed_relations: ['=', '<', '<=']",
                "    allowed_units: ['nM']",
                "  inactive_if:",
                "    column: standard_value",
                "    operator: '>'",
                "    value: 10000",
                "    allowed_relations: ['=', '>', '>=']",
                "    allowed_units: ['nM']",
                "  drop_intermediate: true",
                "deduplicate_by: inchikey",
            ]
        ),
        encoding="utf-8",
    )

    adapt_dataset(
        schema,
        "chembltoy",
        out_audit=tmp_path / "audit.csv",
        out_curated=tmp_path / "curated.csv",
        out_failures=tmp_path / "failures.csv",
        manifest_path=tmp_path / "manifest.json",
    )

    curated = pd.read_csv(tmp_path / "curated.csv")
    failures = pd.read_csv(tmp_path / "failures.csv")

    assert set(curated["ligand_id"]) == {"CHEMBL1", "CHEMBL2"}
    assert dict(zip(curated["ligand_id"], curated["activity_label"], strict=True)) == {
        "CHEMBL1": "active",
        "CHEMBL2": "inactive",
    }
    assert set(failures["failure_reason"]) == {"intermediate_or_unassigned_activity"}
