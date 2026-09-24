import numpy as np

from src.obstacles import point_in_obstacle
from src.simulator.city_map import (
    CityMap,
    generate_city_map,
    load_city_map,
    save_city_map,
)
from src.simulator.engine import _make_algorithm, execute_simulation
from src.transition import formation_connected, formation_collision_free


def test_city_generation_is_reproducible_and_clear_of_obstacles():
    first = generate_city_map(
        n_targets=30,
        n_obstacles=6,
        difficulty="Trung bình",
        seed=19,
    )
    second = generate_city_map(
        n_targets=30,
        n_obstacles=6,
        difficulty="Trung bình",
        seed=19,
    )

    assert np.allclose(first.targets, second.targets)
    assert np.allclose(first.target_weights, second.target_weights)
    assert first.obstacles == second.obstacles
    assert all(
        not point_in_obstacle(target, obstacle, 8.0)
        for target in first.targets
        for obstacle in first.obstacles
    )


def test_city_map_json_round_trip(tmp_path):
    original = generate_city_map(
        n_targets=12,
        n_obstacles=3,
        difficulty="Dễ",
        seed=7,
    )
    path = tmp_path / "city.json"

    save_city_map(original, path)
    restored = load_city_map(path)

    assert isinstance(restored, CityMap)
    assert restored.name == original.name
    assert restored.difficulty == original.difficulty
    assert np.allclose(restored.targets, original.targets)
    assert np.allclose(restored.target_weights, original.target_weights)
    assert restored.obstacles == original.obstacles


def test_generated_city_builds_valid_initial_swarm_problem():
    city_map = generate_city_map(
        n_targets=20,
        n_obstacles=5,
        difficulty="Khó",
        seed=42,
    )
    problem = city_map.to_problem(
        n_uavs=8,
        sensing_radius=175.0,
        communication_radius=280.0,
        min_separation=3.0,
        max_speed=40.0,
    )

    assert formation_connected(
        problem.start_positions,
        problem.scenario.communication_radius,
    )
    assert formation_collision_free(
        problem.start_positions,
        problem.scenario.min_separation,
    )
    assert len(problem.obstacles) == 5


def test_large_interactive_ga_disables_expensive_post_processing():
    algorithm = _make_algorithm(
        "transition-aware",
        population=18,
        generations=20,
        large_interactive_scene=True,
    )

    assert algorithm.finalist_count == 4
    assert algorithm.coverage_refine_rounds == 0
    assert not algorithm.use_deterministic_refinement
    assert not algorithm.use_global_planning


def test_large_interactive_ga_does_not_fall_back_to_zero_motion():
    city = generate_city_map(
        n_targets=800,
        n_obstacles=18,
        difficulty="Trung bình",
        seed=42,
    )
    result = execute_simulation(
        city.to_dict(),
        {
            "n_uavs": 14,
            "sensing_radius": 175.0,
            "communication_radius": 280.0,
            "min_separation": 3.0,
            "max_speed": 40.0,
            "obstacle_clearance": 8.0,
            "algorithm": "transition-aware",
            "seed": 0,
            "population": 18,
            "generations": 20,
        },
    )

    trajectory = result["trajectory"]
    total_movement = float(
        np.sum(np.linalg.norm(trajectory[-1] - trajectory[0], axis=1))
    )
    assert len(trajectory) > 1
    assert total_movement > 0.0
    assert result["metrics"]["transition_feasible"]
    assert result["metrics"]["weighted_coverage_ratio"] > 0.90
