from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..proposed import TransitionAwareGA
from .reconfiguration_benchmark import run_reconfiguration_benchmark


DEFAULT_SENSITIVITY_GRID = {
    "performance_tolerance": [0.0025, 0.005, 0.010],
    "warm_start_fraction": [0.0, 0.35, 0.70],
    "group_mutation_probability": [0.20, 0.45, 0.70],
    "max_group_hops": [1, 2, 3, 4],
    "coverage_refine_rounds": [0, 3, 6],
    "coverage_refine_exact_candidates": [4, 8, 12],
}


def _algorithm_for(parameter: str, value, population_size: int, generations: int):
    kwargs = {
        "population_size": population_size,
        "generations": generations,
        "selection_mode": "coverage_first",
        parameter: value,
        "name": f"sensitivity_{parameter}_{value}",
    }
    return TransitionAwareGA(**kwargs)


def run_reconfiguration_sensitivity(
    problems,
    seeds,
    output_dir: str | Path,
    *,
    grid: dict | None = None,
    population_size: int = 18,
    generations: int = 20,
):
    """Run one-factor-at-a-time sensitivity around the final CP3 defaults.

    Scenario sweeps (N, target count, communication radius, pattern) are treated
    as robustness/scalability experiments elsewhere. This function is only for
    algorithm-parameter sensitivity.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = DEFAULT_SENSITIVITY_GRID if grid is None else grid

    all_rows = []
    all_summaries = []

    for parameter, values in grid.items():
        parameter_dir = output_dir / parameter
        algorithms = [
            _algorithm_for(
                parameter,
                value,
                population_size=population_size,
                generations=generations,
            )
            for value in values
        ]

        df, summary = run_reconfiguration_benchmark(
            problems=problems,
            algorithms=algorithms,
            seeds=seeds,
            output_dir=parameter_dir,
        )

        df.insert(0, "sensitivity_parameter", parameter)
        summary.insert(0, "sensitivity_parameter", parameter)
        all_rows.append(df)
        all_summaries.append(summary)

    results = pd.concat(all_rows, ignore_index=True)
    summary = pd.concat(all_summaries, ignore_index=True)

    results.to_csv(output_dir / "results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    return results, summary
