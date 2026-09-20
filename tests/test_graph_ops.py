import numpy as np

from src.graph_ops import articulation_points
from src.problem import Scenario


def make_scenario():
    return Scenario(
        name="graph-test",
        pattern="test",
        width=10.0,
        height=10.0,
        targets=np.array([[5.0, 5.0]]),
        target_weights=np.ones(1),
        n_uavs=4,
        sensing_radius=1.0,
        communication_radius=1.1,
        min_separation=0.0,
    )


def test_chain_articulation_points():
    scenario = make_scenario()
    positions = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ])

    assert articulation_points(scenario, positions) == {1, 2}


def test_cycle_has_no_articulation():
    scenario = make_scenario()
    positions = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])

    assert articulation_points(scenario, positions) == set()
