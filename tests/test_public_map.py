from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.datasets import load_target_map_csv, scenario_from_target_map_csv


def test_load_target_map_csv_defaults_weights(tmp_path: Path):
    path = tmp_path / "map.csv"
    pd.DataFrame({
        "x": [10.0, 20.0],
        "y": [30.0, 40.0],
    }).to_csv(path, index=False)

    targets, weights = load_target_map_csv(path)

    np.testing.assert_allclose(targets, [[10.0, 30.0], [20.0, 40.0]])
    np.testing.assert_allclose(weights, [1.0, 1.0])


def test_external_scenario_rejects_out_of_bounds_target(tmp_path: Path):
    path = tmp_path / "map.csv"
    pd.DataFrame({
        "x": [120.0],
        "y": [20.0],
        "weight": [2.0],
    }).to_csv(path, index=False)

    with pytest.raises(ValueError, match="outside map bounds"):
        scenario_from_target_map_csv(
            path,
            name="external_case",
            width=100.0,
            height=100.0,
            n_uavs=4,
            sensing_radius=20.0,
            communication_radius=40.0,
            min_separation=5.0,
        )
