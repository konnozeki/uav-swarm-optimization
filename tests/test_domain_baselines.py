import numpy as np

from src.baselines import (
    JOCCGradientConfig,
    JOCCCentralizedProjectedGradient,
    JOCCDistributedProjectedGradient,
    R2CBufferedForceConfig,
    R2CBufferedVirtualForce,
)
from src.metrics import evaluate
from src.scenarios import make_scenario


def _scenario():
    return make_scenario(
        pattern="clustered",
        seed=123,
        n_uavs=4,
        n_targets=30,
        communication_radius=360.0,
        sensing_radius=180.0,
        min_separation=45.0,
    )


def test_jocc_cpgs_smoke():
    scenario = _scenario()
    algorithm = JOCCCentralizedProjectedGradient(
        JOCCGradientConfig(iterations=4)
    )
    solution, runtime = algorithm.solve(scenario, seed=0)

    assert solution.positions.shape == (scenario.n_uavs, 2)
    assert np.all(np.isfinite(solution.positions))
    assert runtime >= 0.0
    assert evaluate(scenario, solution).complete


def test_jocc_dpgs_smoke():
    scenario = _scenario()
    algorithm = JOCCDistributedProjectedGradient(
        JOCCGradientConfig(iterations=4)
    )
    solution, runtime = algorithm.solve(scenario, seed=1)

    assert solution.positions.shape == (scenario.n_uavs, 2)
    assert np.all(np.isfinite(solution.positions))
    assert runtime >= 0.0
    assert evaluate(scenario, solution).complete


def test_r2c_buffered_force_smoke():
    scenario = _scenario()
    algorithm = R2CBufferedVirtualForce(
        R2CBufferedForceConfig(iterations=4)
    )
    solution, runtime = algorithm.solve(scenario, seed=2)

    assert solution.positions.shape == (scenario.n_uavs, 2)
    assert np.all(np.isfinite(solution.positions))
    assert runtime >= 0.0
    assert evaluate(scenario, solution).complete
