import numpy as np

from src.metrics import evaluate
from src.problem import ForbiddenRegion, Scenario, positions_in_forbidden_regions
from src.proposed import ConnectedFrontierConfig, ConnectedFrontierLeafSwap
from src.static_candidates import static_candidate_points


def obstacle_scenario():
    targets = np.array([
        [100.0, 100.0],
        [220.0, 100.0],
        [460.0, 110.0],
        [580.0, 120.0],
        [340.0, 420.0],
        [500.0, 430.0],
    ])
    return Scenario(
        name="obstacle_static",
        pattern="test",
        width=700.0,
        height=550.0,
        targets=targets,
        target_weights=np.ones(len(targets)),
        n_uavs=3,
        sensing_radius=150.0,
        communication_radius=280.0,
        min_separation=20.0,
        forbidden_regions=(
            ForbiddenRegion.rectangle("block", 280.0, 40.0, 400.0, 260.0),
        ),
    )


def test_static_candidates_exclude_forbidden_regions():
    scenario = obstacle_scenario()
    points = static_candidate_points(scenario, grid_size=6)

    assert len(points) > 0
    assert not np.any(positions_in_forbidden_regions(scenario, points))


def test_connected_frontier_returns_obstacle_feasible_solution():
    scenario = obstacle_scenario()
    solution, _ = ConnectedFrontierLeafSwap(
        ConnectedFrontierConfig(grid_size=7, local_rounds=4)
    ).solve(scenario)
    metrics = evaluate(scenario, solution)

    assert metrics.feasible
    assert metrics.forbidden_violations == 0
