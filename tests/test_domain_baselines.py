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


def test_r2c_pair_force_matches_three_paper_zones():
    scenario = _scenario()
    algorithm = R2CBufferedVirtualForce(
        R2CBufferedForceConfig(iterations=1)
    )
    rc = scenario.communication_radius
    unit = np.array([1.0, 0.0])

    repulsive = algorithm._pair_force(0.5 * rc, unit, rc)
    neutral = algorithm._pair_force(0.85 * rc, unit, rc)
    attractive = algorithm._pair_force(0.95 * rc, unit, rc)

    assert repulsive[0] < 0.0
    assert np.allclose(neutral, 0.0)
    assert attractive[0] > 0.0


def test_r2c_boundary_force_points_inward():
    scenario = _scenario()
    algorithm = R2CBufferedVirtualForce(
        R2CBufferedForceConfig(
            iterations=1,
            edge_buffer_ratio=0.10,
        )
    )

    left = algorithm._boundary_force(
        scenario,
        np.array([1.0, scenario.height / 2.0]),
    )
    right = algorithm._boundary_force(
        scenario,
        np.array([
            scenario.width - 1.0,
            scenario.height / 2.0,
        ]),
    )
    bottom = algorithm._boundary_force(
        scenario,
        np.array([scenario.width / 2.0, 1.0]),
    )
    top = algorithm._boundary_force(
        scenario,
        np.array([
            scenario.width / 2.0,
            scenario.height - 1.0,
        ]),
    )

    assert left[0] > 0.0
    assert right[0] < 0.0
    assert bottom[1] > 0.0
    assert top[1] < 0.0


def test_r2c_velocity_uses_explicit_speed_cap():
    scenario = _scenario()
    algorithm = R2CBufferedVirtualForce(
        R2CBufferedForceConfig(
            iterations=1,
            max_speed=1.0,
        )
    )

    positions = np.array([
        [100.0, 100.0],
        [140.0, 100.0],
        [180.0, 100.0],
        [220.0, 100.0],
    ])
    velocity = algorithm._velocity(scenario, positions)
    speed = np.linalg.norm(velocity, axis=1)

    assert np.all(speed <= 1.0 + 1e-9)


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
