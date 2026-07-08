from pathlib import Path


from screensift.common.io import ensure_dir, load_yaml, read_text_table_auto


def test_load_yaml_works(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("answer: 42\n", encoding="utf-8")
    assert load_yaml(config_path) == {"answer": 42}


def test_ensure_dir_creates_dir(tmp_path: Path) -> None:
    directory = ensure_dir(tmp_path / "nested" / "dir")
    assert directory.exists()
    assert directory.is_dir()


def test_read_text_table_auto_reads_comma_and_tab(tmp_path: Path) -> None:
    comma_path = tmp_path / "table.csv"
    tab_path = tmp_path / "table.tsv"
    comma_path.write_text("smiles,id\nCCO,ethanol\n", encoding="utf-8")
    tab_path.write_text("smiles\tid\nCCO\tethanol\n", encoding="utf-8")

    comma = read_text_table_auto(comma_path)
    tab = read_text_table_auto(tab_path)

    assert list(comma.columns) == ["smiles", "id"]
    assert list(tab.columns) == ["smiles", "id"]
    assert comma.loc[0, "smiles"] == "CCO"
    assert tab.loc[0, "id"] == "ethanol"


def test_read_text_table_auto_preserves_smiles_triple_bonds(tmp_path: Path) -> None:
    table_path = tmp_path / "table.smi"
    table_path.write_text("smiles id\nCc1nc(c(C#N)c(C)c1Cl)S 1\n", encoding="utf-8")

    table = read_text_table_auto(table_path)

    assert table.loc[0, "smiles"] == "Cc1nc(c(C#N)c(C)c1Cl)S"
