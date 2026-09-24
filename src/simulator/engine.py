from __future__ import annotations

import numpy as np

from ..baselines import (
    JOCCCentralizedProjectedGradient,
    JOCCDistributedProjectedGradient,
    JOCCGradientConfig,
    LiteratureStaticThenTransition,
    R2CBufferedForceConfig,
    R2CBufferedVirtualForce,
    StaticThenTransition,
)
from ..proposed import TransitionAwareGA
from .city_map import CityMap


ALGORITHM_LABELS = {
    "static": "Formation trước, đường đi sau",
    "jocc-cpgs": "JOCC CPGS 2026",
    "jocc-dpgs": "JOCC DPGS 2026",
    "r2c-ise": "R2C-ISE AAAI-26",
    "transition-aware": "GA xét quá trình chuyển đội hình",
}


def _make_algorithm(
    name: str,
    population: int,
    generations: int,
    *,
    large_interactive_scene: bool = False,
):
    if name == "static":
        return StaticThenTransition(
            population_size=population,
            generations=generations,
        )
    if name == "jocc-cpgs":
        return LiteratureStaticThenTransition(
            JOCCCentralizedProjectedGradient(JOCCGradientConfig(iterations=80))
        )
    if name == "jocc-dpgs":
        return LiteratureStaticThenTransition(
            JOCCDistributedProjectedGradient(
                JOCCGradientConfig(
                    iterations=80,
                    step_fraction=0.05,
                    connectivity_weight=0.80,
                )
            )
        )
    if name == "r2c-ise":
        return LiteratureStaticThenTransition(
            R2CBufferedVirtualForce(R2CBufferedForceConfig(iterations=400))
        )
    if name == "transition-aware":
        return TransitionAwareGA(
            population_size=population,
            generations=generations,
            # The desktop simulator must remain interactive on dense city
            # maps. Research runners instantiate TransitionAwareGA directly
            # and therefore keep the exhaustive finalist/refinement campaign.
            # Four candidates are still fast with the cached obstacle graph,
            # and avoid treating one optimistic-but-unreachable estimate as a
            # reason to fall straight back to the zero-motion start formation.
            finalist_count=4 if large_interactive_scene else 2,
            coverage_refine_rounds=0 if large_interactive_scene else 1,
            use_deterministic_refinement=not large_interactive_scene,
            use_global_planning=not large_interactive_scene,
        )
    raise ValueError(f"unknown algorithm: {name}")


def execute_simulation(city_payload: dict, settings: dict) -> dict:
    """Worker-process entry point; accepts and returns picklable values."""
    city_map = CityMap.from_dict(city_payload)
    problem = city_map.to_problem(
        n_uavs=int(settings["n_uavs"]),
        sensing_radius=float(settings["sensing_radius"]),
        communication_radius=float(settings["communication_radius"]),
        min_separation=float(settings["min_separation"]),
        max_speed=float(settings["max_speed"]),
        obstacle_clearance=float(settings["obstacle_clearance"]),
    )
    large_interactive_scene = (
        len(city_map.targets) >= 400
        or len(city_map.obstacles) >= 15
    )
    algorithm = _make_algorithm(
        str(settings["algorithm"]),
        int(settings["population"]),
        int(settings["generations"]),
        large_interactive_scene=large_interactive_scene,
    )
    solution, runtime = algorithm.solve(problem, seed=int(settings["seed"]))
    transition = solution.transition_solution

    if transition is None:
        # Never animate an unverified straight line as though it were a safe
        # planner result. Hold formation A and expose the failed feasibility
        # metrics to the UI instead.
        trajectory = np.asarray([problem.start_positions], dtype=float)
    else:
        trajectory = np.asarray(transition.trajectory, dtype=float)

    return {
        "algorithm_key": settings["algorithm"],
        "algorithm": algorithm.name,
        "label": (
            ALGORITHM_LABELS[str(settings["algorithm"])]
            + (
                " (nhanh cho map lớn)"
                if large_interactive_scene
                and settings["algorithm"] == "transition-aware"
                else ""
            )
        ),
        "seed": int(settings["seed"]),
        "runtime": float(runtime),
        "dt": float(problem.dt),
        "trajectory": trajectory,
        "transition_available": transition is not None,
        "final_positions": np.asarray(solution.final_positions, dtype=float),
        "metrics": solution.evaluation.to_dict(),
        "communication_radius": problem.scenario.communication_radius,
        "sensing_radius": problem.scenario.sensing_radius,
        "min_separation": problem.scenario.min_separation,
        "interactive_mode": (
            "fast_large_scene"
            if large_interactive_scene and settings["algorithm"] == "transition-aware"
            else "standard"
        ),
    }
