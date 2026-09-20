import numpy as np
import pytest

from src.metrics import evaluate, evaluate_positions
from src.problem import Solution


def test_partial_placement_is_allowed_but_not_feasible(simple_scenario):
    positions = np.array([
        [100.0, 100.0],
        [250.0, 100.0],
    ])

    metrics = evaluate_positions(simple_scenario, positions)

    assert not metrics.complete
    assert not metrics.feasible
    assert metrics.constraint_violation > 0.0


def test_final_evaluate_requires_all_uavs(simple_scenario):
    solution = Solution(np.array([[100.0, 100.0]]))

    with pytest.raises(ValueError):
        evaluate(simple_scenario, solution)


def test_full_connected_collision_free_placement_is_feasible(simple_scenario):
    positions = np.array([
        [200.0, 200.0],
        [400.0, 200.0],
        [600.0, 200.0],
        [800.0, 200.0],
    ])

    metrics = evaluate_positions(simple_scenario, positions)

    assert metrics.complete
    assert metrics.in_bounds
    assert metrics.connected
    assert metrics.collision_violations == 0
    assert metrics.feasible
    assert metrics.constraint_violation == 0.0
