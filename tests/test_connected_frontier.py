import numpy as np

from src.metrics import evaluate
from src.problem import Scenario
from src.proposed import (
    ConnectedFrontierConfig,
    ConnectedFrontierLeafSwap,
)


def scenario(seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([
        [200.0, 250.0],
        [800.0, 250.0],
        [200.0, 750.0],
        [800.0, 750.0],
    ])
    labels = rng.integers(0, len(centers), size=96)
    targets = centers[labels] + rng.normal(0.0, 55.0, size=(96, 2))
    targets = np.clip(targets, 0.0, 1000.0)

    return Scenario(
        name="cfg_test",
        pattern="clustered",
        width=1000.0,
        height=1000.0,
        targets=targets,
        target_weights=np.ones(len(targets)),
        n_uavs=6,
        sensing_radius=170.0,
        communication_radius=300.0,
        min_separation=25.0,
        seed=seed,
    )


def test_connected_frontier_returns_hard_feasible_formation():
    s = scenario()
    solution, runtime = ConnectedFrontierLeafSwap().solve(s)
    metrics = evaluate(s, solution)

    assert runtime >= 0.0
    assert metrics.feasible
    assert metrics.connected
    assert metrics.collision_violations == 0
    assert len(solution.positions) == s.n_uavs


def test_leaf_swap_never_reduces_primary_coverage():
    s = scenario(2)
    cfg, _ = ConnectedFrontierLeafSwap(
        ConnectedFrontierConfig(local_rounds=0)
    ).solve(s)
    cfg_ls, _ = ConnectedFrontierLeafSwap(
        ConnectedFrontierConfig(local_rounds=12)
    ).solve(s)

    before = evaluate(s, cfg)
    after = evaluate(s, cfg_ls)

    assert before.feasible
    assert after.feasible
    assert after.weighted_coverage_ratio >= before.weighted_coverage_ratio - 1e-12


def test_connected_frontier_is_deterministic_across_solver_seeds():
    s = scenario(4)
    algorithm = ConnectedFrontierLeafSwap()

    first, _ = algorithm.solve(s, seed=0)
    second, _ = algorithm.solve(s, seed=999)

    assert np.array_equal(first.positions, second.positions)
