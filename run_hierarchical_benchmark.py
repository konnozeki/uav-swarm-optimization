"""Compare identical Greedy seeds with feedback-driven formation refinement."""
import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy

from src.experiments.reconfiguration_cases import (
    quick_reconfiguration_profile, showcase_reconfiguration_profile,
)
from src.proposed import BackboneTransitionPlanner, HierarchicalReconfiguration
from src.proposed.formation_search import greedy_formation
from src.reconfiguration import evaluate_reconfiguration, ReconfigurationObjectiveConfig
from src.transition import evaluate_transition
from src.transition_visualization import save_transition_plot
from src.plotly_transition_visualization import save_transition_html


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--budget', type=float, default=3.0)
    parser.add_argument('--target-coverage', type=float, default=None,
                        help='Optional early-stop coverage ratio, e.g. 0.90; default disabled')
    parser.add_argument('--profile', choices=['showcase', 'all'], default='all')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/hierarchical'))
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error('--seeds must be positive')
    if args.target_coverage is not None and not 0 <= args.target_coverage <= 1:
        parser.error('--target-coverage must be in [0, 1]')
    problems = showcase_reconfiguration_profile()
    if args.profile == 'all':
        problems = quick_reconfiguration_profile() + problems
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    traces = []
    for problem in problems:
        started = time.perf_counter()
        goal = greedy_formation(problem)
        evaluation, route = evaluate_reconfiguration(
            problem, goal, BackboneTransitionPlanner(),
            objective_config=ReconfigurationObjectiveConfig(time_weight=0, travel_weight=0),
        )
        baseline_time = time.perf_counter()-started
        rows.append(dict(problem=problem.name, algorithm='greedy_global', seed=0,
                         runtime_sec=baseline_time, **evaluation.to_dict()))
        print(problem.name, 'greedy_global', f'{baseline_time:.3f}s',
              f'coverage={evaluation.weighted_coverage_ratio:.2%}', f'feasible={evaluation.feasible}', flush=True)
        for seed in range(args.seeds):
            algorithm = HierarchicalReconfiguration(time_budget_sec=args.budget,
                                                    target_coverage=args.target_coverage)
            solution, runtime = algorithm.solve(problem, seed)
            evaluation = solution.evaluation
            rows.append(dict(problem=problem.name, algorithm=algorithm.name, seed=seed,
                             runtime_sec=runtime, **evaluation.to_dict()))
            traces.append(dict(problem=problem.name, seed=seed, **algorithm.last_diagnostics))
            print(problem.name, seed, f'{runtime:.3f}s',
                  f'coverage={evaluation.weighted_coverage_ratio:.2%}', f'feasible={evaluation.feasible}', flush=True)
            tp = problem.transition_problem(solution.final_positions)
            verified = evaluate_transition(tp, solution.transition_solution)
            if not verified.feasible:
                raise RuntimeError('independent route validation failed')
            if problem.obstacles and seed == 0:
                stem = args.output_dir / f'{problem.name}__hierarchical_seed0'
                title = f'Hierarchical formation + path planning<br>Coverage {evaluation.weighted_coverage_ratio:.1%} | compute {runtime:.2f}s<br>'
                options = dict(targets=problem.scenario.targets,
                               sensing_radius=problem.scenario.sensing_radius,
                               target_weights=problem.scenario.target_weights)
                save_transition_plot(tp, solution.transition_solution, stem.with_suffix('.png'),
                                     title=title.replace('<br>', '\n'), **options)
                save_transition_html(tp, solution.transition_solution, stem.with_suffix('.html'),
                                     title=title, subframes_per_step=5, frame_duration_ms=200,
                                     enable_manual_edit=False, **options)
                np.savez_compressed(stem.with_suffix('.npz'),
                                    trajectory=solution.transition_solution.trajectory,
                                    goal=solution.final_positions,
                                    assignment=solution.transition_solution.assignment)
                stem.with_suffix('.json').write_text(json.dumps(verified.to_dict(), indent=2)+'\n')
    with (args.output_dir / 'results.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / 'diagnostics.json').write_text(json.dumps(traces, indent=2)+'\n')
    (args.output_dir / 'manifest.json').write_text(json.dumps(dict(
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        profile=args.profile, seeds=list(range(args.seeds)), time_budget_sec=args.budget,
        search_rounds=4, restarts=2, shortlist_size=4, endpoint_radius=20,
        attempt_time_sec=1.5, coverage_loss_tolerance=0.01, motion_cost_in_selection=False,
        planning_policy='fast_screen_diverse_recovery_v3',
        target_coverage=args.target_coverage, recovery_budget_fraction=0.25,
        fast_screen_budget_fraction=0.20, fast_attempt_sec=0.08, search_budget_fraction=0.20,
        greedy_baseline='deterministic, run once per scenario, no wall-clock cutoff',
        min_separation_m={p.name: p.scenario.min_separation for p in problems},
        note='Coverage uses sensing radius only; obstacles constrain UAV motion, not sensing or radio line-of-sight.',
    ), indent=2)+'\n')


if __name__ == '__main__':
    main()
