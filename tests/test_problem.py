import numpy as np
import pytest

from src.problem import Scenario


def test_impossible_min_separation_is_rejected():
    with pytest.raises(ValueError):
        Scenario(
            name="impossible",
            pattern="test",
            width=100.0,
            height=100.0,
            targets=np.array([[50.0, 50.0]]),
            target_weights=np.ones(1),
            n_uavs=2,
            sensing_radius=10.0,
            communication_radius=10.0,
            min_separation=11.0,
        )
