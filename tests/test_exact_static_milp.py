import numpy as np

from src.baselines import ExactStaticCoverageMILP, ExactStaticMILPConfig
from src.metrics import evaluate
from src.problem import Scenario
from src.proposed import ConnectedFrontierConfig, ConnectedFrontierLeafSwap


def tiny_scenario():
    targets = np.array([
        [120.0, 150.0],
        [180.0, 160.0],
        [380.0, 180.0],
        [430.0, 200.0],
        [620.0, 180.0],
        [680.0, 150.0],
        [360.0, 420.0],
        [460.0, 430.0],
    ])
    return Scenario(
        name="tiny_exact",
        pattern="clustered",
        width=800.0,
        height=600.0,
        targets=targets,
        target_weights=np.ones(len(targets)),
        n_uavs=3,
        sensing_radius=150.0,
        communication_radius=300.0,
        min_separation=20.0,
        seed=0,
    )


def test_exact_milp_returns_feasible_optimum_and_bounds_cfg():
    scenario = tiny_scenario()
    grid_size = 5

    exact, _ = ExactStaticCoverageMILP(
        ExactStaticMILPConfig(
            grid_size=grid_size,
            time_limit_sec=10.0,
        )
    ).solve(scenario)
    proposed, _ = ConnectedFrontierLeafSwap(
        ConnectedFrontierConfig(
            grid_size=grid_size,
            local_rounds=8,
        )
    ).solve(scenario)

    exact_metrics = evaluate(scenario, exact)
    proposed_metrics = evaluate(scenario, proposed)

    assert exact_metrics.feasible
    assert proposed_metrics.feasible
    assert (
        exact_metrics.weighted_coverage_ratio
        + 1e-12
        >= proposed_metrics.weighted_coverage_ratio
    )
