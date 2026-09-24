"""Reproduce the fixed-endpoint obstacle regression and export its trajectory."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from src.experiments.reconfiguration_cases import showcase_reconfiguration_profile
from src.proposed import BackboneTransitionPlanner, TransitionAwareGA
from src.reconfiguration import evaluate_reconfiguration
from src.transition import evaluate_transition
from src.transition_visualization import save_transition_plot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/global_formation_path"))
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    problem = showcase_reconfiguration_profile()[0]
    started = time.perf_counter()
    goal = TransitionAwareGA()._obstacle_aware_greedy_goal(problem)
    greedy_runtime = time.perf_counter() - started
    rows = []
    for global_planning in (False, True):
        started = time.perf_counter()
        evaluation, solution = evaluate_reconfiguration(
            problem, goal,
            BackboneTransitionPlanner(enable_global_planning=global_planning),
        )
        runtime = greedy_runtime + time.perf_counter() - started
        label = "greedy_global" if global_planning else "greedy_local"
        rows.append(dict(algorithm=label, seed=0, runtime_sec=runtime, **evaluation.to_dict()))
        print(label, f"runtime={runtime:.3f}s", f"coverage={evaluation.weighted_coverage_ratio:.2%}",
              f"feasible={evaluation.feasible}", f"travel={evaluation.total_travel_distance:.1f}", flush=True)
        if global_planning and evaluation.feasible:
            metrics = evaluate_transition(problem.transition_problem(goal), solution)
            np.savez_compressed(args.output_dir / "greedy_global_trajectory.npz",
                                trajectory=solution.trajectory, goal=goal, assignment=solution.assignment)
            (args.output_dir / "trajectory_validation.json").write_text(
                json.dumps(metrics.to_dict(), indent=2) + "\n"
            )
            save_transition_plot(
                problem.transition_problem(goal), solution,
                args.output_dir / "greedy_global.png",
                title=f"Greedy + joint waypoint planner\nCoverage {evaluation.weighted_coverage_ratio:.1%} | compute {runtime:.2f}s\n",
                targets=problem.scenario.targets,
                sensing_radius=problem.scenario.sensing_radius,
                target_weights=problem.scenario.target_weights,
            )
    for seed in range(args.seeds):
        algorithm = TransitionAwareGA(population_size=8, generations=4, elite_size=2,
                                      finalist_count=2, coverage_refine_rounds=1,
                                      coverage_refine_exact_candidates=2)
        solution, runtime = algorithm.solve(problem, seed=seed)
        evaluation = solution.evaluation
        rows.append(dict(algorithm="cp3_global", seed=seed, runtime_sec=runtime, **evaluation.to_dict()))
        print("cp3_global", seed, f"runtime={runtime:.3f}s", f"coverage={evaluation.weighted_coverage_ratio:.2%}",
              f"feasible={evaluation.feasible}", flush=True)
    with (args.output_dir / "results.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
