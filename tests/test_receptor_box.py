from pathlib import Path

import numpy as np


from screensift.receptors.define_docking_boxes import calculate_docking_box


def test_calculate_docking_box_from_ligand_coordinates() -> None:
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 4.0, 6.0],
        ]
    )

    box = calculate_docking_box(coordinates, padding=2.0, min_box_size=5.0)

    assert box["center_x"] == 1.0
    assert box["center_y"] == 2.0
    assert box["center_z"] == 3.0
    assert box["size_x"] == 6.0
    assert box["size_y"] == 8.0
    assert box["size_z"] == 10.0


def test_calculate_docking_box_enforces_minimum_size() -> None:
    coordinates = np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    box = calculate_docking_box(coordinates, padding=1.0, min_box_size=18.0)

    assert box["size_x"] == 18.0
    assert box["size_y"] == 18.0
    assert box["size_z"] == 18.0
