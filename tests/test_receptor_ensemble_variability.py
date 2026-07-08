from pathlib import Path


from screensift.receptors.ensemble_variability import receptor_clean_files, variability_table  # noqa: E402


def _atom_line(serial: int, resname: str, chain: str, resnum: int, x: float, y: float, z: float) -> str:
    return (
        f"ATOM  {serial:5d}  CA  {resname:>3s} {chain}{resnum:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
    )


def _write_receptor(path: Path, start_resnum: int, x_offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    residues = ["GLY", "ALA", "SER", "VAL"]
    lines = [
        _atom_line(index + 1, resname, "A", start_resnum + index, float(index) + x_offset, 0.0, 0.0)
        for index, resname in enumerate(residues)
    ]
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")


def test_sequence_alignment_handles_shifted_residue_numbering(tmp_path: Path) -> None:
    receptor_dir = tmp_path / "receptors"
    _write_receptor(receptor_dir / "ref" / "receptor_clean.pdb", start_resnum=10, x_offset=0.0)
    _write_receptor(receptor_dir / "shifted" / "receptor_clean.pdb", start_resnum=101, x_offset=0.2)

    frame, warnings = variability_table(receptor_clean_files(receptor_dir), tmp_path / "missing_boxes.csv")

    assert len(frame) == 4
    assert frame["residue_number"].tolist() == [10, 11, 12, 13]
    assert frame["coordinate_dispersion_angstrom"].max() > 0
    assert not any("No common" in warning for warning in warnings)
