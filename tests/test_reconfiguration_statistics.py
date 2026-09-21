from pathlib import Path

import pandas as pd

from src.experiments.reconfiguration_statistics import (
    run_reconfiguration_statistics,
)


def test_reconfiguration_statistics_compare_all_baselines(tmp_path: Path):
    rows = []
    for seed in range(4):
        rows.extend([
            {
                "problem": "p0",
                "seed": seed,
                "algorithm": "transition_aware_ga",
                "joint_fitness": 0.9 + 0.01 * seed,
                "weighted_coverage_ratio": 0.95,
                "formation_time_sec": 10.0,
                "total_travel_distance": 100.0,
                "runtime_sec": 2.0,
                "transition_feasible": True,
            },
            {
                "problem": "p0",
                "seed": seed,
                "algorithm": "baseline_a",
                "joint_fitness": 0.7 + 0.01 * seed,
                "weighted_coverage_ratio": 0.80,
                "formation_time_sec": 14.0,
                "total_travel_distance": 140.0,
                "runtime_sec": 1.0,
                "transition_feasible": seed != 0,
            },
            {
                "problem": "p0",
                "seed": seed,
                "algorithm": "baseline_b",
                "joint_fitness": 0.75 + 0.01 * seed,
                "weighted_coverage_ratio": 0.84,
                "formation_time_sec": 12.0,
                "total_travel_distance": 125.0,
                "runtime_sec": 1.5,
                "transition_feasible": True,
            },
        ])

    path = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    continuous, feasibility = run_reconfiguration_statistics(path)

    assert set(continuous["baseline"]) == {"baseline_a", "baseline_b"}
    assert set(feasibility["baseline"]) == {"baseline_a", "baseline_b"}
    assert "p_holm" in continuous.columns
    assert "mcnemar_p_holm" in feasibility.columns
