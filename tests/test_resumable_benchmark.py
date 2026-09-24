from types import SimpleNamespace

import pandas as pd
import pytest

from src.experiments.reconfiguration_benchmark import (
    run_reconfiguration_benchmark,
)


class _Evaluation:
    weighted_coverage_ratio = 0.75
    final_fitness = 0.70
    joint_fitness = 0.65
    feasible = True
    transition_feasible = True
    formation_time_sec = 2.0
    total_travel_distance = 10.0

    def to_dict(self):
        return {
            "weighted_coverage_ratio": self.weighted_coverage_ratio,
            "final_fitness": self.final_fitness,
            "joint_fitness": self.joint_fitness,
            "feasible": self.feasible,
            "static_feasible": True,
            "transition_feasible": self.transition_feasible,
            "final_obstacle_free": True,
            "transition_obstacle_free": True,
            "formation_time_sec": self.formation_time_sec,
            "total_travel_distance": self.total_travel_distance,
            "time_efficiency": 0.9,
        }


class _Algorithm:
    name = "fake_algorithm"

    def __init__(self, fail_seed=None):
        self.fail_seed = fail_seed
        self.calls = []

    def solve(self, problem, seed):
        self.calls.append(seed)
        if seed == self.fail_seed:
            raise RuntimeError("deliberate interruption")
        solution = SimpleNamespace(
            evaluation=_Evaluation(),
            transition_solution=None,
            final_positions=None,
        )
        return solution, 0.01


def _problem():
    scenario = SimpleNamespace(
        pattern="test",
        n_uavs=2,
        targets=[(0.0, 0.0)],
        communication_radius=10.0,
        sensing_radius=5.0,
    )
    return SimpleNamespace(name="case", scenario=scenario, max_speed=1.0)


def test_benchmark_persists_each_run_and_resumes_after_failure(tmp_path):
    interrupted = _Algorithm(fail_seed=1)

    with pytest.raises(RuntimeError, match="deliberate interruption"):
        run_reconfiguration_benchmark(
            [_problem()],
            [interrupted],
            [0, 1],
            tmp_path,
        )

    partial = pd.read_csv(tmp_path / "results.csv")
    assert list(partial["seed"]) == [0]
    assert (tmp_path / "failures.json").exists()

    resumed = _Algorithm()
    results, _ = run_reconfiguration_benchmark(
        [_problem()],
        [resumed],
        [0, 1],
        tmp_path,
    )

    assert resumed.calls == [1]
    assert list(results["seed"]) == [0, 1]


def test_benchmark_refuses_to_overwrite_when_resume_is_disabled(tmp_path):
    pd.DataFrame([{
        "problem": "case",
        "algorithm": "fake_algorithm",
        "seed": 0,
    }]).to_csv(tmp_path / "results.csv", index=False)

    with pytest.raises(FileExistsError):
        run_reconfiguration_benchmark(
            [_problem()],
            [_Algorithm()],
            [0],
            tmp_path,
            resume=False,
        )


def test_benchmark_rejects_results_from_another_experiment(tmp_path):
    pd.DataFrame([{
        "problem": "another_case",
        "algorithm": "fake_algorithm",
        "seed": 0,
    }]).to_csv(tmp_path / "results.csv", index=False)

    with pytest.raises(ValueError, match="outside this experiment"):
        run_reconfiguration_benchmark(
            [_problem()],
            [_Algorithm()],
            [0],
            tmp_path,
        )
