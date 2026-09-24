from dataclasses import replace
import time

import numpy as np

from src.experiments.reconfiguration_cases import showcase_reconfiguration_profile
from src.metrics import evaluate_positions
from src.proposed import BackboneTransitionPlanner, HierarchicalReconfiguration
from src.proposed.formation_search import greedy_formation, improve_formations, sensing_key
from src.reconfiguration import evaluate_reconfiguration
from src.transition import TransitionProblem, evaluate_transition
from src.proposed.formation_path import group_paths


def test_local_search_improves_greedy_and_preserves_static_constraints():
    problem = showcase_reconfiguration_profile()[0]
    greedy = greedy_formation(problem)
    key = sensing_key(evaluate_positions(problem.scenario, greedy))
    initial_coverage = key[0]
    for proposal in improve_formations(problem, greedy):
        metrics = evaluate_positions(problem.scenario, proposal)
        assert metrics.feasible
        assert sensing_key(metrics) > key
        key = sensing_key(metrics)
    assert key[0] > initial_coverage


def test_hierarchy_returns_certified_improvement_and_reuses_remaining_route():
    problem = showcase_reconfiguration_profile()[0]
    algorithm = HierarchicalReconfiguration(time_budget_sec=10)
    solution, _ = algorithm.solve(problem, seed=0)
    assert solution.evaluation.weighted_coverage_ratio > 0.8389
    trajectory = solution.transition_solution.trajectory
    assert evaluate_transition(problem.transition_problem(solution.final_positions),
                               solution.transition_solution).feasible
    # A new control cycle starts part-way through the last certified plan.
    updated = replace(problem, start_positions=trajectory[len(trajectory)//2].copy())
    replanned, _ = algorithm.solve(updated, seed=0, warm_start=solution)
    assert algorithm.last_diagnostics['warm_start_reused']
    assert replanned.evaluation.weighted_coverage_ratio >= solution.evaluation.weighted_coverage_ratio
    assert np.allclose(replanned.transition_solution.trajectory[0], updated.start_positions)
    assert evaluate_transition(updated.transition_problem(replanned.final_positions),
                               replanned.transition_solution).feasible


def test_tiny_budget_returns_valid_hold_instead_of_unverified_proposal():
    problem = showcase_reconfiguration_profile()[0]
    solution, _ = HierarchicalReconfiguration(time_budget_sec=1e-9).solve(problem)
    assert solution.evaluation.feasible
    assert np.array_equal(solution.final_positions, problem.start_positions)
    assert solution.evaluation.total_travel_distance == 0


def test_fast_screening_precedes_unsliced_deep_attempts(monkeypatch):
    import src.proposed.hierarchical_reconfiguration as module
    problem = showcase_reconfiguration_profile()[0]
    algorithm = HierarchicalReconfiguration(time_budget_sec=3)
    calls = []
    search_starts = []
    original_search = module.improve_formations

    def search(problem, initial, **kwargs):
        search_starts.append(initial.copy())
        yield from original_search(problem, initial, **kwargs)

    def attempt(problem, candidate, incumbent, seed, deadline, phase, diagnostics):
        calls.append((phase, deadline-time.perf_counter()))
        return incumbent

    monkeypatch.setattr(algorithm, '_attempt', attempt)
    monkeypatch.setattr(module, 'improve_formations', search)
    solution, _ = algorithm.solve(problem)
    assert calls[0][0] == 'fast_local'
    assert 0 < calls[0][1] <= 0.08
    phases = [phase for phase, _ in calls]
    first_deep = phases.index('deep')
    assert all(phase == 'fast_local' for phase in phases[:first_deep])
    assert calls[first_deep][1] > 1.0
    assert len(calls) > 1
    assert all(0 < allowance <= algorithm.attempt_time_sec for _, allowance in calls)
    assert solution.evaluation.feasible
    # Failed screening triggers fresh searches, including the measured start,
    # not only permutations of a trapped Greedy endpoint.
    assert len(search_starts) > algorithm.restarts
    assert any(np.array_equal(p, problem.start_positions)
               for p in search_starts[algorithm.restarts:])


def test_target_coverage_stops_after_certified_fast_result():
    problem = showcase_reconfiguration_profile()[0]
    algorithm = HierarchicalReconfiguration(time_budget_sec=3, target_coverage=0.90)
    solution, _ = algorithm.solve(problem, seed=0)
    assert solution.evaluation.weighted_coverage_ratio >= 0.90
    assert evaluate_transition(problem.transition_problem(solution.final_positions),
                               solution.transition_solution).feasible
    assert algorithm.last_diagnostics['stop_reason'] == 'target_coverage'
    assert all(a['phase'] == 'fast_local' for a in algorithm.last_diagnostics['attempts'])


def test_target_coverage_validation_and_hold_stop():
    import pytest
    for value in (-0.1, 1.1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            HierarchicalReconfiguration(target_coverage=value)
    algorithm = HierarchicalReconfiguration(target_coverage=0.0)
    solution, _ = algorithm.solve(showcase_reconfiguration_profile()[0])
    assert solution.evaluation.feasible
    assert not algorithm.last_diagnostics['attempts']
    assert algorithm.last_diagnostics['stop_reason'] == 'target_coverage'


def test_deep_attempt_uses_full_baseline_pipeline(monkeypatch):
    from src.transition import TransitionSolution
    problem = showcase_reconfiguration_profile()[0]
    problem = replace(problem, scenario=replace(problem.scenario, sensing_radius=110,
                                                communication_radius=180))
    algorithm = HierarchicalReconfiguration(time_budget_sec=3)
    n = problem.scenario.n_uavs
    hold = TransitionSolution(problem.start_positions[None].copy(),
                              problem.start_positions.copy(), np.arange(n), 'hold', True)
    incumbent = algorithm._evaluate_route(problem, hold, 0)
    deadline = time.perf_counter()+2
    calls = []

    def solve(tp, *, deadline):
        calls.append(deadline)
        return replace(hold, reached_goal=False)

    monkeypatch.setattr(algorithm.planner, 'solve', solve)
    diagnostics = dict(attempts=[], accepted=[])
    result = algorithm._attempt(problem, greedy_formation(problem), incumbent, 0,
                                deadline, 'deep', diagnostics)
    assert calls == [deadline]  # No hidden fraction of the allocated deadline.
    assert result is incumbent
    assert diagnostics['attempts'][0]['method'] == 'full_planner'


def test_group_translation_certifies_all_uavs():
    start = np.array([[20., 20.], [24., 20.], [22., 24.]])
    tp = TransitionProblem(name='group', width=100, height=100,
                           start_positions=start, goal_positions=start+[40., 30.],
                           communication_radius=10, min_separation=3, max_speed=10,
                           allow_reassignment=False)
    path = next(group_paths(tp, tp.goal_positions))
    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], tp.goal_positions)
    route = BackboneTransitionPlanner()._certify_waypoints(tp, path, np.arange(3))
    assert route is not None
    assert evaluate_transition(tp, route).feasible


def test_group_seed_is_not_a_collision_certificate():
    from src.obstacles import AxisAlignedRectangle
    start = np.array([[20., 20.], [24., 20.]])
    tp = TransitionProblem(name='crossing', width=100, height=100,
                           start_positions=start, goal_positions=start+[50., 0.],
                           communication_radius=10, min_separation=3, max_speed=10,
                           allow_reassignment=False,
                           obstacles=(AxisAlignedRectangle(40, 10, 50, 30),))
    planner = BackboneTransitionPlanner()
    assert planner._certify_waypoints(tp, np.array([start, tp.goal_positions]),
                                     np.arange(2)) is None
    # Group routing is allowed to go around the obstacle, but accepted routes
    # must still pass the same independent continuous check.
    path = next(group_paths(tp, tp.goal_positions))
    route = planner._certify_waypoints(tp, path, np.arange(2))
    if route is not None:
        assert evaluate_transition(tp, route).feasible


def test_flexible_endpoint_moves_requested_positions_but_still_certifies_route():
    problem = TransitionProblem(
        name='flexible_endpoint', width=100, height=100,
        start_positions=np.array([[10., 50.], [25., 50.]]),
        goal_positions=np.array([[80., 50.], [88., 50.]]),
        communication_radius=25, min_separation=10, max_speed=10,
        allow_reassignment=False,
    )
    planner = BackboneTransitionPlanner(global_knots=10)
    route = planner.refine_path(problem, endpoint_radius=5)
    assert route is not None
    assert not np.allclose(route.assigned_goals, problem.goal_positions)
    assert np.all(np.abs(route.assigned_goals - problem.goal_positions) <= 5 + 1e-9)
    actual = replace(problem, goal_positions=route.assigned_goals)
    assert evaluate_transition(actual, route).feasible
    assert planner.refine_path(problem, deadline=time.perf_counter()-1) is None
    rejecting = BackboneTransitionPlanner(global_knots=10, global_iterations=1)
    assert rejecting.refine_path(actual, endpoint_radius=5,
                                 endpoint_accept=lambda endpoint: False) is None


def test_supplied_route_cannot_be_scored_against_a_different_destination():
    problem = showcase_reconfiguration_profile()[0]
    planner = BackboneTransitionPlanner()
    route = planner.solve(problem.transition_problem(problem.start_positions))
    wrong_goal = greedy_formation(problem)
    evaluation, _ = evaluate_reconfiguration(problem, wrong_goal, planner, transition_solution=route)
    assert not evaluation.feasible
