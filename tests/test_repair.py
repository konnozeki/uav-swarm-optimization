import numpy as np

from src.metrics import evaluate_positions
from src.repair import repair_solution


def test_full_repair_returns_feasible_solution(simple_scenario):
    positions = np.array([
        [100.0, 100.0],
        [105.0, 100.0],
        [800.0, 800.0],
        [805.0, 800.0],
    ])

    repaired = repair_solution(positions, simple_scenario)
    metrics = evaluate_positions(simple_scenario, repaired)

    assert metrics.feasible
