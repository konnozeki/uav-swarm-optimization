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


def test_r2c_pair_force_matches_equation_9():
    scenario = _scenario()
    config = R2CBufferedForceConfig(iterations=1)
    algorithm = R2CBufferedVirtualForce(config)
    rc = scenario.communication_radius
    unit = np.array([1.0, 0.0])

    repulsive_distance = 0.5 * rc
    repulsive = algorithm._pair_force(repulsive_distance, unit, rc)
    expected_repulsive = -np.exp(
        -config.alpha
        * repulsive_distance
        / (config.lower_ratio * rc)
    )
    assert np.allclose(repulsive, [expected_repulsive, 0.0])

    assert np.allclose(algorithm._pair_force(0.85 * rc, unit, rc), 0.0)

    attractive_distance = 0.95 * rc
    attractive = algorithm._pair_force(attractive_distance, unit, rc)
    expected_attractive = np.exp(
        -config.beta
        * (rc - attractive_distance)
        / (rc - config.upper_ratio * rc)
    )
    assert np.allclose(attractive, [expected_attractive, 0.0])
    assert np.allclose(algorithm._pair_force(1.01 * rc, unit, rc), 0.0)


def test_r2c_boundary_force_matches_equation_10():
    scenario = _scenario()
    config = R2CBufferedForceConfig(
        iterations=1,
        boundary_buffer_ratio=0.10,
    )
    algorithm = R2CBufferedVirtualForce(config)
    distance = 10.0

    force = algorithm._boundary_force(
        scenario,
        np.array([distance, scenario.height / 2.0]),
    )

    assert np.allclose(
        force,
        [np.exp(-config.boundary_decay * distance), 0.0],
    )


def test_r2c_velocity_matches_equation_12_and_speed_cap():
    scenario = _scenario()
    config = R2CBufferedForceConfig(
        iterations=1,
        max_speed=1.0,
        epsilon=1e-12,
    )
    algorithm = R2CBufferedVirtualForce(config)
    positions = np.array([
        [100.0, 100.0],
        [140.0, 100.0],
        [180.0, 100.0],
        [220.0, 100.0],
    ])

    force = algorithm._force(scenario, positions)
    velocity = algorithm._velocity(scenario, positions)
    norm = np.linalg.norm(force, axis=1, keepdims=True)
    expected = config.max_speed * force / (norm + config.epsilon)

    assert np.allclose(velocity, expected)
    assert np.all(np.linalg.norm(velocity, axis=1) <= config.max_speed + 1e-12)


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
