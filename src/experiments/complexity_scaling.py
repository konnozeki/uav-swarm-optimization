from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from ..proposed import TransitionAwareGA
from ..reconfiguration import ReconfigurationProblem
from ..scenarios import make_scenario
from .transition_cases import ring_formation


DEFAULT_GRID = {
    "n_uavs": [4, 6, 8, 10, 12],
    "n_targets": [50, 100, 200, 400],
    "population_size": [8, 12, 18, 28],
    "generations": [6, 10, 20, 30],
}


def _problem(
    *,
    n_uavs: int,
    n_targets: int,
    communication_radius: float = 280.0,
    min_separation: float = 55.0,
    seed: int = 9100,
) -> ReconfigurationProblem:
    scenario = make_scenario(
        pattern="split",
        seed=seed,
        n_uavs=n_uavs,
        n_targets=n_targets,
        communication_radius=communication_radius,
        sensing_radius=175.0,
        min_separation=min_separation,
    )

    min_ring_radius = (
        min_separation
        / max(2.0 * math.sin(math.pi / n_uavs), 1e-12)
        * 1.10
    )
    radius = max(0.40 * communication_radius, min_ring_radius)
    start = ring_formation(
        (scenario.width / 2.0, scenario.height / 2.0),
        n_uavs,
        radius,
    )

    return ReconfigurationProblem(
        name=f"complexity_u{n_uavs}_m{n_targets}",
        scenario=scenario,
        start_positions=start,
        max_speed=40.0,
        allow_reassignment=True,
    )


def run_cp3_complexity_scaling(
    seeds,
    output_dir: str | Path,
    *,
    grid: dict | None = None,
    base_n_uavs: int = 6,
    base_n_targets: int = 100,
    base_population: int = 18,
    base_generations: int = 20,
):
    """Empirically isolate the main CP3 runtime scaling axes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = DEFAULT_GRID if grid is None else grid
    seeds = list(seeds)

    rows = []

    for axis, values in grid.items():
        for value in values:
            n_uavs = value if axis == "n_uavs" else base_n_uavs
            n_targets = value if axis == "n_targets" else base_n_targets
            population = value if axis == "population_size" else base_population
            generations = value if axis == "generations" else base_generations

            problem = _problem(
                n_uavs=int(n_uavs),
                n_targets=int(n_targets),
                seed=9100 + int(value),
            )
            algorithm = TransitionAwareGA(
                population_size=int(population),
                generations=int(generations),
            )

            for seed in seeds:
                solution, runtime = algorithm.solve(problem, seed=seed)
                rows.append({
                    "axis": axis,
                    "value": value,
                    "seed": seed,
                    "n_uavs": n_uavs,
                    "n_targets": n_targets,
                    "population_size": population,
                    "generations": generations,
                    "runtime_sec": runtime,
                    "coverage": solution.evaluation.weighted_coverage_ratio,
                    "transition_feasible": (
                        solution.evaluation.transition_feasible
                    ),
                    "formation_time_sec": (
                        solution.evaluation.formation_time_sec
                    ),
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "results.csv", index=False)

    summary = (
        df.groupby(["axis", "value"], as_index=False)
        .agg(
            runtime_mean_sec=("runtime_sec", "mean"),
            runtime_std_sec=("runtime_sec", "std"),
            coverage_mean=("coverage", "mean"),
            transition_feasible_rate=("transition_feasible", "mean"),
            formation_time_mean_sec=("formation_time_sec", "mean"),
        )
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    return df, summary
