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


def test_multi_start_improves_centroid_gap_case_for_cfg():
    targets = []
    for center in (
        np.array([150.0, 250.0]),
        np.array([850.0, 250.0]),
        np.array([150.0, 750.0]),
        np.array([850.0, 750.0]),
    ):
        rng = np.random.default_rng(int(np.sum(center)))
        targets.extend(center + rng.normal(0.0, 20.0, size=(18, 2)))
    targets = np.clip(np.asarray(targets, dtype=float), 0.0, 1000.0)
    s = Scenario(
        name="centroid_gap",
        pattern="centroid_gap",
        width=1000.0,
        height=1000.0,
        targets=targets,
        target_weights=np.ones(len(targets)),
        n_uavs=4,
        sensing_radius=120.0,
        communication_radius=310.0,
        min_separation=25.0,
        seed=0,
    )

    centroid_only, _ = ConnectedFrontierLeafSwap(
        ConnectedFrontierConfig(
            grid_size=12,
            local_rounds=0,
            multi_start_roots=1,
            lookahead_depth=1,
        )
    ).solve(s)
    multi_start, _ = ConnectedFrontierLeafSwap(
        ConnectedFrontierConfig(
            grid_size=12,
            local_rounds=0,
            multi_start_roots=5,
        )
    ).solve(s)

    centroid_metrics = evaluate(s, centroid_only)
    multi_start_metrics = evaluate(s, multi_start)

    assert centroid_metrics.feasible
    assert multi_start_metrics.feasible
    assert (
        multi_start_metrics.weighted_coverage_ratio
        > centroid_metrics.weighted_coverage_ratio
    )


def test_lookahead_accepts_relay_steps_for_future_coverage():
    rng = np.random.default_rng(2)
    targets = []
    weights = []
    for center, count, weight in (
        (np.array([120.0, 500.0]), 20, 1.0),
        (np.array([880.0, 500.0]), 20, 1.5),
        (np.array([500.0, 200.0]), 8, 0.2),
    ):
        targets.extend(center + rng.normal(0.0, 35.0, size=(count, 2)))
        weights.extend([weight] * count)
    targets = np.clip(np.asarray(targets, dtype=float), 0.0, 1000.0)
    s = Scenario(
        name="relay_lookahead",
        pattern="relay_lookahead",
        width=1000.0,
        height=1000.0,
        targets=targets,
        target_weights=np.asarray(weights, dtype=float),
        n_uavs=5,
        sensing_radius=115.0,
        communication_radius=230.0,
        min_separation=25.0,
        seed=2,
    )

    greedy_construct, _ = ConnectedFrontierLeafSwap(
        ConnectedFrontierConfig(
            grid_size=16,
            local_rounds=0,
            multi_start_roots=1,
            lookahead_depth=1,
        )
    ).solve(s)
    lookahead_construct, _ = ConnectedFrontierLeafSwap(
        ConnectedFrontierConfig(
            grid_size=16,
            local_rounds=0,
            multi_start_roots=1,
            lookahead_depth=2,
            lookahead_branch=4,
            lookahead_beam=2,
        )
    ).solve(s)

    greedy_metrics = evaluate(s, greedy_construct)
    lookahead_metrics = evaluate(s, lookahead_construct)

    assert greedy_metrics.feasible
    assert lookahead_metrics.feasible
    assert (
        lookahead_metrics.weighted_coverage_ratio
        > greedy_metrics.weighted_coverage_ratio + 0.05
    )
