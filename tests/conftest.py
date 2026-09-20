import numpy as np
import pytest

from src.problem import Scenario


@pytest.fixture
def simple_scenario():
    return Scenario(
        name="test",
        pattern="test",
        width=1000.0,
        height=1000.0,
        targets=np.array([
            [100.0, 100.0],
            [300.0, 100.0],
            [500.0, 500.0],
            [700.0, 700.0],
        ]),
        target_weights=np.ones(4),
        n_uavs=4,
        sensing_radius=180.0,
        communication_radius=350.0,
        min_separation=40.0,
        seed=0,
    )
