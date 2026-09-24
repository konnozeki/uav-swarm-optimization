"""Anytime formation search with trajectory feedback and a verified incumbent."""
from __future__ import annotations

from dataclasses import replace
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..metrics import evaluate_positions
from ..obstacles import formation_obstacle_free
from ..reconfiguration import (ReconfigurationObjectiveConfig, ReconfigurationSolution,
                               evaluate_reconfiguration)
from ..transition import TransitionSolution, assign_goals
from .backbone_transition import BackboneTransitionPlanner
from .formation_search import greedy_formation, improve_formations, sensing_key
from .formation_path import group_paths


class HierarchicalReconfiguration:
    """Greedy -> multi-start local search -> certified joint planning.

    Time is a cooperative wall-clock budget, not a hard real-time guarantee.
    Cost does not influence acceptance: actual coverage/fitness must improve.
    The incumbent always contains an independently validated complete route.
    `warm_start` supports replanning from updated UAV positions; its route must
    pass validation again against the current problem before becoming incumbent.
    """
    name = "hierarchical_reconfiguration"

    def __init__(self, time_budget_sec=3.0, search_rounds=4, restarts=2,
                 shortlist_size=4, endpoint_radius=20.0, attempt_time_sec=1.5,
                 coverage_loss_tolerance=0.01, target_coverage=None):
        if not np.isfinite(time_budget_sec) or time_budget_sec <= 0:
            raise ValueError("time_budget_sec must be finite and positive")
        if search_rounds < 0 or restarts < 1 or shortlist_size < 1:
            raise ValueError("search_rounds >= 0, restarts and shortlist_size >= 1 required")
        if not np.isfinite(endpoint_radius) or endpoint_radius < 0:
            raise ValueError("endpoint_radius must be finite and non-negative")
        if not np.isfinite(attempt_time_sec) or attempt_time_sec <= 0:
            raise ValueError("attempt_time_sec must be finite and positive")
        if not np.isfinite(coverage_loss_tolerance) or not 0 <= coverage_loss_tolerance <= 1:
            raise ValueError("coverage_loss_tolerance must be in [0, 1]")
        self.time_budget_sec = float(time_budget_sec)
        self.search_rounds = search_rounds
        self.restarts = restarts
        self.shortlist_size = shortlist_size
        self.endpoint_radius = float(endpoint_radius)
        self.attempt_time_sec = float(attempt_time_sec)
        self.coverage_loss_tolerance = float(coverage_loss_tolerance)
        if target_coverage is not None and (not np.isfinite(target_coverage)
                                           or not 0 <= target_coverage <= 1):
            raise ValueError("target_coverage must be None or a ratio in [0, 1]")
        self.target_coverage = target_coverage
        self.planner = BackboneTransitionPlanner()
        self.objective = ReconfigurationObjectiveConfig(time_weight=0, travel_weight=0)
        self.last_diagnostics = {}

    def _evaluate_route(self, problem, route, seed):
        # Convert the already identity-assigned endpoint to canonical slot order.
        route = replace(route, assigned_goals=route.trajectory[-1].copy(),
                        assignment=np.arange(problem.scenario.n_uavs))
        endpoint = route.assigned_goals
        evaluation, validated = evaluate_reconfiguration(
            problem, endpoint, self.planner, objective_config=self.objective,
            transition_solution=route,
        )
        if not evaluation.feasible:
            return None
        return ReconfigurationSolution(endpoint, evaluation, validated, self.name, seed)

    @staticmethod
    def _key(solution):
        e = solution.evaluation
        return e.weighted_coverage_ratio, e.final_fitness

    def _target_met(self, solution):
        return (self.target_coverage is not None
                and solution.evaluation.weighted_coverage_ratio >= self.target_coverage)

    def _attempt(self, problem, candidate, incumbent, seed, deadline, phase, diagnostics):
        """One bounded attempt; exact and relaxed endpoints share its budget."""
        started = time.perf_counter()
        candidate_key = sensing_key(evaluate_positions(problem.scenario, candidate))
        if (started >= deadline or self._target_met(incumbent)
                or candidate_key <= self._key(incumbent)):
            return incumbent
        tp = problem.transition_problem(candidate)
        warm = incumbent.transition_solution.trajectory
        warm = warm if len(warm) > 1 else None
        if phase != 'fast_local' and warm is not None and problem.allow_reassignment:
            _, indices = linear_sum_assignment(np.linalg.norm(
                warm[-1, :, None] - candidate[None], axis=2))
            candidate = candidate[indices]
        else:
            candidate, _, _ = assign_goals(tp)
        tp = replace(tp, goal_positions=candidate, allow_reassignment=False)
        route = None
        method = 'local'
        flexible = False
        if phase == 'fast_local':
            local = BackboneTransitionPlanner(enable_global_planning=False)
            proposal = local.solve(tp, deadline=deadline)
            if proposal.reached_goal:
                route = proposal
        elif phase == 'deep' and warm is not None:
            method = 'warm'
            route = self.planner.refine_path(tp, initial_trajectory=warm, deadline=deadline)
        elif phase == 'deep':
            # Same full pipeline as standalone Greedy: no internal time slices.
            method = 'full_planner'
            proposal = self.planner.solve(tp, deadline=deadline)
            if proposal.reached_goal:
                route = proposal
        elif phase == 'group':
            method = 'group'
            # Check coordinated seeds exactly before optimizing them. In
            # particular, straight compact transit can already be feasible.
            for path in group_paths(tp, candidate, self.planner.global_knots):
                if time.perf_counter() >= deadline:
                    break
                route = self.planner._certify_waypoints(tp, path, np.arange(tp.n_uavs), deadline=deadline)
                if route is not None:
                    method = 'group_certified'
                    break
                route = self.planner.refine_path(
                    tp, initial_trajectory=path,
                    deadline=deadline)
                if route is not None:
                    method = 'group_refined'
                    break
        elif phase == 'flexible' and self.endpoint_radius > 0:
            flexible = True
            method = 'flexible'

            def accept(endpoint):
                m = evaluate_positions(problem.scenario, endpoint)
                return (m.feasible and sensing_key(m) > self._key(incumbent)
                        and m.weighted_coverage_ratio >= candidate_key[0]-self.coverage_loss_tolerance
                        and formation_obstacle_free(endpoint, problem.obstacles, problem.obstacle_clearance))

            route = self.planner.refine_path(
                tp, initial_trajectory=warm, endpoint_radius=self.endpoint_radius,
                endpoint_accept=accept, endpoint_terms=self._coverage_terms(problem, candidate),
                deadline=deadline)
        verified = self._evaluate_route(problem, route, seed) if route is not None else None
        diagnostics['attempts'].append(dict(
            phase=phase, method=method, proposed_coverage=candidate_key[0],
            flexible_used=flexible,
            status='certified' if verified is not None else 'not_found_within_budget',
            runtime_sec=time.perf_counter()-started, allocated_sec=deadline-started))
        if verified is not None and self._key(verified) > self._key(incumbent):
            diagnostics['accepted'].append(verified.evaluation.weighted_coverage_ratio)
            return verified
        return incumbent

    @staticmethod
    def _coverage_terms(problem, preferred):
        """Smooth penalty protects targets covered by the proposed endpoint."""
        scenario = problem.scenario
        covered = (np.linalg.norm(scenario.targets[:, None] - preferred[None], axis=2)
                   <= scenario.sensing_radius).any(axis=1)
        targets = scenario.targets[covered]

        def terms(endpoint):
            gradient = np.zeros_like(endpoint)
            if not len(targets):
                return 0.0, gradient
            delta = endpoint[None] - targets[:, None]
            distance = np.linalg.norm(delta, axis=2)
            nearest = np.argmin(distance, axis=1)
            rows = np.arange(len(targets))
            length = distance[rows, nearest]
            violation = np.maximum(length - (scenario.sensing_radius - 0.5), 0)
            force = 2 * violation[:, None] * delta[rows, nearest] / np.maximum(length[:, None], 1e-12)
            np.add.at(gradient, nearest, force)
            return float(np.sum(violation ** 2)), gradient
        return terms

    def solve(self, problem, seed=0, *, warm_start=None):
        started = time.perf_counter()
        deadline = started + self.time_budget_sec
        n = problem.scenario.n_uavs
        hold = TransitionSolution(problem.start_positions[None].copy(),
                                  problem.start_positions.copy(), np.arange(n),
                                  self.name, True)
        incumbent = self._evaluate_route(problem, hold, seed)
        if incumbent is None:
            raise ValueError("current formation cannot be validated")
        diagnostics = dict(attempts=[], accepted=[], warm_start_reused=False)
        if self._target_met(incumbent):
            runtime = time.perf_counter()-started
            diagnostics.update(runtime_sec=runtime, budget_sec=self.time_budget_sec,
                               budget_exhausted=runtime >= self.time_budget_sec,
                               target_coverage=self.target_coverage, stop_reason='target_coverage',
                               fast_screen_elapsed_sec=runtime,
                               fast_screen_coverage=incumbent.evaluation.weighted_coverage_ratio)
            self.last_diagnostics = diagnostics
            return incumbent, runtime

        if warm_start is not None and warm_start.transition_solution is not None:
            old = warm_start.transition_solution.trajectory
            if old.shape[1:] == problem.start_positions.shape and len(old) and np.isfinite(old).all():
                closest = int(np.argmin(np.sum((old-problem.start_positions)**2, axis=(1, 2))))
                tail = old[closest:].copy()
                tail[0] = problem.start_positions
                try:
                    route = self.planner._certify_waypoints(
                        problem.transition_problem(tail[-1]), tail,
                        np.arange(n), deadline=deadline,
                    )
                except ValueError:
                    # New bounds/obstacles may invalidate yesterday's endpoint.
                    route = None
                if route is not None:
                    reused = self._evaluate_route(problem, route, seed)
                    if reused is not None and self._key(reused) > self._key(incumbent):
                        incumbent = reused
                        diagnostics['warm_start_reused'] = True

        proposals = []
        greedy = greedy_formation(problem, deadline=deadline)
        if greedy is not None:
            proposals.append(greedy)
            search_deadline = min(deadline, time.perf_counter()+0.20*self.time_budget_sec)
            # Search both Greedy and the incumbent to expose different basins.
            for restart in range(self.restarts):
                initial = greedy if restart % 2 == 0 else incumbent.final_positions
                proposals.extend(improve_formations(
                    problem, initial, seed=seed+restart, rounds=self.search_rounds,
                    deadline=search_deadline,
                ))
        unique = {p.tobytes(): p for p in proposals}
        ranked = sorted(unique.values(), key=lambda p: sensing_key(
            evaluate_positions(problem.scenario, p)), reverse=True)
        shortlist = ranked[:self.shortlist_size]
        # Reserve a distinct Greedy attempt rather than filling every slot with
        # similar local optima. A failed path attempt does not poison its score.
        if (greedy is not None and self.shortlist_size > 1
                and not any(np.array_equal(greedy, p) for p in shortlist)):
            shortlist = shortlist[:max(0, self.shortlist_size-1)] + [greedy]

        # Cheap screening across ALL distinct proposals, including intermediate
        # formations discarded by the expensive shortlist. Easy routes win early.
        fast_deadline = min(deadline, time.perf_counter()+0.20*self.time_budget_sec)
        for candidate in ranked:
            now = time.perf_counter()
            if now >= fast_deadline or self._target_met(incumbent):
                break
            incumbent = self._attempt(problem, candidate, incumbent, seed,
                                      min(fast_deadline, now+0.08), 'fast_local', diagnostics)

        diagnostics['fast_screen_elapsed_sec'] = time.perf_counter()-started
        diagnostics['fast_screen_coverage'] = incumbent.evaluation.weighted_coverage_ratio

        # A failed family of endpoints needs new endpoints, not repeated long
        # attempts at the same Greedy goal. Diversify only on total screening
        # failure; test each new endpoint immediately and keep certified gains.
        recovery_started = time.perf_counter()
        if (greedy is not None and len(incumbent.transition_solution.trajectory) == 1
                and not self._target_met(incumbent)):
            recovery_deadline = min(deadline, recovery_started+0.25*self.time_budget_sec)
            for restart in range(3):
                if time.perf_counter() >= recovery_deadline or self._target_met(incumbent):
                    break
                for candidate in improve_formations(
                        problem, greedy if restart % 2 == 0 else problem.start_positions,
                        seed=seed+self.restarts+restart,
                        rounds=self.search_rounds, deadline=recovery_deadline):
                    if candidate.tobytes() in unique:
                        continue
                    unique[candidate.tobytes()] = candidate
                    proposals.append(candidate)
                    now = time.perf_counter()
                    incumbent = self._attempt(problem, candidate, incumbent, seed,
                                              min(recovery_deadline, now+0.08),
                                              'fast_local', diagnostics)
                    if self._target_met(incumbent):
                        break
                if len(incumbent.transition_solution.trajectory) > 1:
                    break
            ranked = sorted(unique.values(), key=lambda p: sensing_key(
                evaluate_positions(problem.scenario, p)), reverse=True)
            shortlist = ranked[:self.shortlist_size]
        diagnostics['recovery_elapsed_sec'] = time.perf_counter()-recovery_started
        diagnostics['recovery_coverage'] = incumbent.evaluation.weighted_coverage_ratio

        # Prefer high-coverage goals; reserve time for a different path family
        # if still holding, instead of automatically retrying Greedy first.
        deep_candidates = shortlist
        now = time.perf_counter()
        deep_deadline = (now+0.65*(deadline-now)
                         if len(incumbent.transition_solution.trajectory) == 1 else deadline)
        for candidate in deep_candidates:
            if time.perf_counter() >= deep_deadline or self._target_met(incumbent):
                break
            now = time.perf_counter()
            incumbent = self._attempt(problem, candidate, incumbent, seed,
                                      min(deep_deadline, now+self.attempt_time_sec), 'deep', diagnostics)

        # Optional fallback only after the primary methods had full attempts.
        for phase in ('group', 'flexible'):
            for candidate in shortlist:
                now = time.perf_counter()
                if now >= deadline or self._target_met(incumbent):
                    break
                incumbent = self._attempt(problem, candidate, incumbent, seed,
                                          min(deadline, now+self.attempt_time_sec), phase, diagnostics)

        runtime = time.perf_counter()-started
        diagnostics.update(runtime_sec=runtime, budget_sec=self.time_budget_sec,
                           budget_exhausted=runtime >= self.time_budget_sec,
                           target_coverage=self.target_coverage,
                           stop_reason=('target_coverage' if self._target_met(incumbent) else
                                        'budget' if runtime >= self.time_budget_sec else 'search_exhausted'))
        self.last_diagnostics = diagnostics
        return incumbent, runtime
