"""Persistent anytime geometric planning, not a flight controller or hard RT solver."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import replace
import hashlib
import pickle
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..metrics import evaluate_positions
from ..obstacles import formation_obstacle_free
from ..transition import TransitionSolution, assign_goals
from .backbone_transition import BackboneTransitionPlanner
from .formation_path import optimize_path
from .formation_search import greedy_steps, improve_formations, sensing_key
from .hierarchical_reconfiguration import HierarchicalReconfiguration


class IncrementalReconfiguration:
    """Keep endpoint generators, pending goals and waypoint progress across ticks.

    Only independently certified plans are exposed. Numerical waypoint iterates
    are private warm starts, NEVER executable plans. L-BFGS history itself is
    not retained. Budgets are cooperative; one search/validation operation may
    overrun. No finite-time global optimality or physical tracking guarantee.
    """

    name = 'incremental_reconfiguration'

    def __init__(self, step_budget_sec=0.1, search_rounds=4, restarts=8,
                 max_pending=32, path_slices=6, target_coverage=None):
        if not np.isfinite(step_budget_sec) or step_budget_sec <= 0:
            raise ValueError('step_budget_sec must be finite and positive')
        if any(not isinstance(x, int) or x < 1 for x in
               (search_rounds, restarts, max_pending, path_slices)):
            raise ValueError('search_rounds, restarts, max_pending, path_slices must be positive integers')
        self.step_budget_sec = float(step_budget_sec)
        self.search_rounds, self.restarts = search_rounds, restarts
        self.max_pending, self.path_slices = max_pending, path_slices
        self.engine = HierarchicalReconfiguration(target_coverage=target_coverage)
        self.problem = None
        self.incumbent = None
        self.last_diagnostics = {}

    @staticmethod
    def _signature(problem):
        # Excludes measured positions, includes targets/weights, geometry,
        # assignment policy and all motion constraints.
        payload = (problem.scenario, problem.obstacles, problem.obstacle_clearance,
                   problem.max_speed, problem.dt, problem.allow_reassignment)
        return hashlib.sha256(pickle.dumps(payload)).digest()

    def _hold(self, problem):
        start = problem.start_positions
        route = TransitionSolution(start[None].copy(), start.copy(),
                                   np.arange(len(start)), self.name, True)
        solution = self.engine._evaluate_route(problem, route, self.seed)
        if solution is None:
            raise ValueError('measured formation has no certified stationary fallback')
        solution.algorithm = self.name
        return solution

    def _reset_search(self):
        self.pending = deque()
        self.seen = set()
        self.greedy = None
        self.greedy_generator = greedy_steps(self.problem)
        self.search_generator = None
        self.restart_index = 0
        self.search_done = False
        self.work_turn = 0
        self.completed_search_units = 0
        self.completed_path_units = 0

    def initialize(self, problem, seed=0):
        self.incumbent = None
        self.problem = deepcopy(problem)
        self.seed = seed
        self.incumbent = self._hold(self.problem)
        self.signature = self._signature(self.problem)
        self._reset_search()
        self.cycles = 0
        self.last_diagnostics = dict(status='initialized', route_reused=False)
        return self.get_plan()

    def get_plan(self):
        """Return a defensive copy of the current certified plan."""
        if self.incumbent is None:
            raise RuntimeError('no certified plan; initialize/update_state with a valid formation')
        return deepcopy(self.incumbent)

    def update_state(self, problem, *, path_index=None):
        """Revalidate the remaining route from measured positions.

        UAV array order must preserve physical identities. If available, supply
        the executed waypoint index to disambiguate loops/pauses. Otherwise the
        nearest joint configuration is used and the resulting path is rechecked.
        Geometry/target/constraint changes reset stale search evaluations. A
        position-only change keeps endpoint search and waypoint warm starts,
        but resets attempt counters and rechecks paths from the new state.
        """
        if self.problem is None:
            return self.initialize(problem)
        old = self.incumbent
        if path_index is not None and (old is None or not isinstance(path_index, int)
                                      or not 0 <= path_index < len(old.transition_solution.trajectory)):
            raise ValueError('path_index outside previous trajectory')
        previous_positions = self.problem.start_positions
        self.incumbent = None  # Never expose an old route after failed validation.
        self.problem = deepcopy(problem)
        hold = self._hold(self.problem)
        signature = self._signature(self.problem)
        changed = signature != self.signature
        self.incumbent = hold
        if (not changed and old is not None
                and np.array_equal(previous_positions, problem.start_positions)
                and path_index in (None, 0)):
            self.incumbent = old
            self.last_diagnostics = dict(status='state_unchanged', route_reused=True,
                                         search_reset=False)
            return self.get_plan()
        reused = False
        if old is not None:
            trajectory = old.transition_solution.trajectory
            if trajectory.shape[1:] == problem.start_positions.shape:
                if path_index is None:
                    index = int(np.argmin(np.sum((trajectory-problem.start_positions)**2, axis=(1, 2))))
                else:
                    if not isinstance(path_index, int) or not 0 <= path_index < len(trajectory):
                        raise ValueError('path_index outside previous trajectory')
                    index = path_index
                # A reached waypoint is already the measured start: prepending
                # it again would insert a new wait on every execution cycle.
                # Off-route telemetry still needs an explicitly checked bridge.
                if np.array_equal(problem.start_positions, trajectory[index]):
                    tail = trajectory[index:].copy()
                else:
                    tail = np.concatenate([problem.start_positions[None], trajectory[index:]])
                try:
                    tp = problem.transition_problem(tail[-1])
                    route = self.engine.planner._certify_waypoints(tp, tail, np.arange(len(tail[0])))
                    solution = self.engine._evaluate_route(problem, route, self.seed) if route else None
                except ValueError:
                    solution = None
                if solution is not None and self.engine._key(solution) >= self.engine._key(hold):
                    solution.algorithm = self.name
                    self.incumbent = solution
                    reused = True
        if changed:
            self._reset_search()
        else:
            old_tasks = {task['goal'].tobytes(): task for task in self.pending}
            endpoints = [task['goal'] for task in self.pending]
            if old is not None:
                endpoints.append(old.final_positions)
            self.pending.clear()
            self.seen.clear()  # A goal that failed from A may work from new A.
            for endpoint in endpoints:
                self._enqueue(endpoint)
            for task in self.pending:
                previous = old_tasks.get(task['goal'].tobytes())
                if previous is not None:
                    task['path'] = previous['path']
                    task['labeled'] = previous['labeled']
        self.signature = signature
        self.last_diagnostics = dict(status='state_updated', route_reused=reused,
                                     search_reset=changed)
        return self.get_plan()

    def _enqueue(self, goal):
        if goal.shape != self.problem.start_positions.shape:
            return
        key = goal.tobytes()
        if key in self.seen:
            return
        metrics = evaluate_positions(self.problem.scenario, goal)
        if (not metrics.feasible or not formation_obstacle_free(
                goal, self.problem.obstacles, self.problem.obstacle_clearance)):
            return
        self.seen.add(key)
        self.pending.append(dict(goal=goal.copy(), quality=sensing_key(metrics),
                                 phase='local', slices=0, path=None, labeled=None))
        if len(self.pending) > self.max_pending:
            worst = min(range(len(self.pending)), key=lambda i: self.pending[i]['quality'])
            del self.pending[worst]

    def _search_unit(self):
        if self.greedy_generator is not None:
            try:
                partial = next(self.greedy_generator)
                if len(partial) == self.problem.scenario.n_uavs:
                    self.greedy = partial
                    self._enqueue(partial)
            except StopIteration:
                self.greedy_generator = None
            return
        if self.search_generator is None:
            if self.restart_index >= self.restarts:
                self.search_done = True
                return
            initial = (self.greedy if self.greedy is not None and self.restart_index % 2 == 0
                       else self.incumbent.final_positions)
            self.search_generator = improve_formations(
                self.problem, initial, seed=self.seed+self.restart_index,
                rounds=self.search_rounds, yield_every_uav=True)
            self.restart_index += 1
        try:
            self._enqueue(next(self.search_generator))
        except StopIteration:
            self.search_generator = None

    def _path_unit(self, deadline):
        # Screen new high-quality endpoints before spending another numerical
        # slice on an old blocked goal. Rotate numerical tasks fairly afterward.
        local_indices = [i for i, task in enumerate(self.pending) if task['phase'] == 'local']
        if local_indices:
            index = max(local_indices, key=lambda i: self.pending[i]['quality'])
            task = self.pending[index]
            del self.pending[index]
        else:
            task = self.pending.popleft()
        if task['quality'] <= self.engine._key(self.incumbent):
            return
        if task['labeled'] is None:
            tp = self.problem.transition_problem(task['goal'])
            warm = self.incumbent.transition_solution.trajectory
            if len(warm) > 1:
                if self.problem.allow_reassignment:
                    _, order = linear_sum_assignment(np.linalg.norm(
                        warm[-1, :, None]-task['goal'][None], axis=2))
                    task['labeled'] = task['goal'][order]
                else:
                    task['labeled'] = task['goal'].copy()
                task['path'] = warm.copy()
            else:
                task['labeled'], _, _ = assign_goals(tp)
        tp = replace(self.problem.transition_problem(task['labeled']), allow_reassignment=False)
        route = None
        if task['phase'] == 'local':
            candidate = BackboneTransitionPlanner(enable_global_planning=False).solve(tp, deadline=deadline)
            if candidate.reached_goal:
                route = candidate
            task['phase'] = 'waypoints'
        else:
            def checkpoint(path):
                task['path'] = path

            route = optimize_path(
                tp, tp.goal_positions,
                lambda path: self.engine.planner._certify_waypoints(
                    tp, path, np.arange(tp.n_uavs), deadline=deadline),
                max_iterations=20, initial_trajectory=task['path'], deadline=deadline,
                progress_callback=checkpoint)
            task['slices'] += 1
        if route is not None:
            solution = self.engine._evaluate_route(self.problem, route, self.seed)
            if solution is not None and self.engine._key(solution) > self.engine._key(self.incumbent):
                solution.algorithm = self.name
                self.incumbent = solution
        elif task['slices'] < self.path_slices:
            self.pending.append(task)

    def improve_step(self, budget_sec=None, *, max_work_units=None):
        """Continue saved work; returns (certified plan, elapsed seconds).

        max_work_units supports deterministic experiments independent of CPU
        speed. State update/copy/validation cost is not hidden in the budget.
        """
        if self.incumbent is None:
            raise RuntimeError('initialize a valid problem first')
        budget = self.step_budget_sec if budget_sec is None else budget_sec
        if not np.isfinite(budget) or budget <= 0:
            raise ValueError('budget_sec must be finite and positive')
        if max_work_units is not None and (not isinstance(max_work_units, int) or max_work_units < 1):
            raise ValueError('max_work_units must be a positive integer')
        started = time.perf_counter()
        deadline = started+budget
        units = 0
        while time.perf_counter() < deadline and not self.engine._target_met(self.incumbent):
            if max_work_units is not None and units >= max_work_units:
                break
            if self.search_done and not self.pending:
                break
            # One coordinate sweep is much cheaper than one path solve. Do
            # not spend a full planning slice after every single-UAV proposal.
            period = self.problem.scenario.n_uavs+1
            if self.pending and (self.search_done or self.work_turn % period == period-1):
                # Do not turn the last few milliseconds of a tick into a
                # spurious failed path attempt. Resume this task next tick.
                if deadline-time.perf_counter() < min(0.04, budget*0.5):
                    break
                self._path_unit(deadline)
                self.completed_path_units += 1
            else:
                self._search_unit()
                self.completed_search_units += 1
            self.work_turn += 1
            units += 1
        self.cycles += 1
        result = self.get_plan()
        elapsed = time.perf_counter()-started
        self.last_diagnostics = dict(
            cycle=self.cycles, runtime_sec=elapsed, budget_sec=budget,
            budget_exhausted=elapsed >= budget, work_units=units,
            search_units=self.completed_search_units, path_units=self.completed_path_units,
            pending_goals=len(self.pending), coverage=result.evaluation.weighted_coverage_ratio,
            status=('target_coverage' if self.engine._target_met(self.incumbent) else
                    'search_exhausted' if self.search_done and not self.pending else 'in_progress'))
        return result, elapsed

    def solve(self, problem, seed=0, *, time_budget_sec=3.0):
        """One-shot adapter; repeated online work should use improve_step."""
        if not np.isfinite(time_budget_sec) or time_budget_sec <= 0:
            raise ValueError('time_budget_sec must be finite and positive')
        started = time.perf_counter()
        self.initialize(problem, seed)
        remaining = time_budget_sec-(time.perf_counter()-started)
        result = self.improve_step(remaining)[0] if remaining > 0 else self.get_plan()
        return result, time.perf_counter()-started
