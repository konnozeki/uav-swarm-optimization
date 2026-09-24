"""Repeated online planning ticks; optional idealized execution between ticks."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from src.experiments.reconfiguration_cases import showcase_reconfiguration_profile
from src.proposed import IncrementalReconfiguration
from src.metrics import evaluate_positions
from src.transition import evaluate_transition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cycles', type=int, default=50)
    parser.add_argument('--budget-ms', type=float, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rc', type=float, default=150)
    parser.add_argument('--sensing-radius', type=float, default=110)
    parser.add_argument('--execute-steps', type=int, default=0,
                        help='Ideal execution of this many dt waypoints per cycle; 0 = static convergence')
    parser.add_argument('--target-coverage', type=float, default=None)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/incremental'))
    args = parser.parse_args()
    if args.cycles < 1 or args.execute_steps < 0:
        parser.error('cycles must be positive and execute-steps nonnegative')
    problem = showcase_reconfiguration_profile()[0]
    problem = replace(problem, scenario=replace(problem.scenario,
                      communication_radius=args.rc, sensing_radius=args.sensing_radius))
    optimizer = IncrementalReconfiguration(step_budget_sec=args.budget_ms/1000,
                                           target_coverage=args.target_coverage)
    started = time.perf_counter()
    optimizer.initialize(problem, args.seed)
    initialization_sec = time.perf_counter()-started
    records = []
    executed = [problem.start_positions.copy()]
    for cycle in range(args.cycles):
        cycle_started = time.perf_counter()
        update_sec = 0.0
        if cycle and args.execute_steps:
            previous = optimizer.get_plan().transition_solution.trajectory
            index = min(args.execute_steps, len(previous)-1)
            executed.extend(previous[1:index+1].copy())
            problem = replace(problem, start_positions=previous[index].copy())
            update_started = time.perf_counter()
            optimizer.update_state(problem, path_index=index)
            update_sec = time.perf_counter()-update_started
        plan, runtime = optimizer.improve_step()
        metrics = evaluate_transition(problem.transition_problem(plan.final_positions),
                                      plan.transition_solution)
        if not metrics.feasible:
            raise RuntimeError('independent route validation failed')
        current_coverage = evaluate_positions(problem.scenario, problem.start_positions).weighted_coverage_ratio
        record = dict(optimizer.last_diagnostics, update_sec=update_sec,
                      current_coverage=current_coverage,
                      goal_coverage=plan.evaluation.weighted_coverage_ratio,
                      cycle_wall_sec=time.perf_counter()-cycle_started,
                      feasible=metrics.feasible)
        records.append(record)
        print(f"cycle={cycle} improve={runtime:.3f}s update={update_sec:.3f}s "
              f"current_coverage={current_coverage:.2%} "
              f"goal_coverage={plan.evaluation.weighted_coverage_ratio:.2%} "
              f"pending={record['pending_goals']} status={record['status']}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'diagnostics.json').write_text(json.dumps(records, indent=2)+'\n')
    (args.output_dir/'manifest.json').write_text(json.dumps(dict(
        cycles=args.cycles, budget_ms=args.budget_ms, seed=args.seed,
        rc=args.rc, sensing_radius=args.sensing_radius, min_separation=problem.scenario.min_separation,
        execute_steps=args.execute_steps, target_coverage=args.target_coverage,
        initialization_sec=initialization_sec,
        note='Cooperative planning budget, not hard realtime. Execution is ideal waypoint playback, not dynamics. '
             'cycle_wall_sec includes update, plan copies and independent validation; coverage/goal_coverage '
             'refer to the planned endpoint, current_coverage to the measured formation.'
    ), indent=2)+'\n')
    np.savez_compressed(args.output_dir/'plan.npz', executed=np.asarray(executed),
                        remaining=plan.transition_solution.trajectory,
                        goal=plan.final_positions)


if __name__ == '__main__':
    main()
