from pathlib import Path


from screensift.ligands.generate_3d_conformers import safe_filename


def test_safe_filename_removes_path_separators() -> None:
    filename = safe_filename("../bad ligand/id with spaces")

    assert "/" not in filename
    assert "\\" not in filename
    assert ".." not in filename
    assert filename


def test_safe_filename_is_length_limited() -> None:
    filename = safe_filename("x" * 500, max_length=25)

    assert len(filename) == 25
