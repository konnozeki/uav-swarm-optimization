from dataclasses import replace

import numpy as np
import pytest

from src.experiments.reconfiguration_cases import showcase_reconfiguration_profile
from src.proposed import IncrementalReconfiguration
from src.transition import evaluate_transition


def problem():
    return showcase_reconfiguration_profile()[0]


def test_search_resumes_one_uav_at_a_time_and_plan_copy_is_private():
    p = problem()
    optimizer = IncrementalReconfiguration()
    optimizer.initialize(p)
    generator = optimizer.greedy_generator
    for count in range(3):
        plan, _ = optimizer.improve_step(1, max_work_units=1)
        assert optimizer.greedy_generator is generator
        assert optimizer.last_diagnostics['search_units'] == count+1
        assert evaluate_transition(p.transition_problem(plan.final_positions), plan.transition_solution).feasible
    plan.final_positions[:] = 0
    assert not np.all(optimizer.get_plan().final_positions == 0)


def test_static_steps_never_degrade_certified_coverage():
    p = problem()
    optimizer = IncrementalReconfiguration(step_budget_sec=0.1, restarts=2)
    optimizer.initialize(p)
    previous = optimizer.incumbent.evaluation.weighted_coverage_ratio
    for _ in range(20):
        plan, _ = optimizer.improve_step()
        assert plan.evaluation.weighted_coverage_ratio >= previous
        assert evaluate_transition(p.transition_problem(plan.final_positions), plan.transition_solution).feasible
        previous = plan.evaluation.weighted_coverage_ratio
    assert optimizer.completed_search_units > p.scenario.n_uavs


def test_midflight_update_reuses_route_and_preserves_search():
    p = problem()
    optimizer = IncrementalReconfiguration()
    optimizer.initialize(p)
    goal = p.start_positions + [0., -10.]
    tp = p.transition_problem(goal)
    route = optimizer.engine.planner._certify_waypoints(tp, np.array([p.start_positions, goal]), np.arange(10))
    assert route is not None
    optimizer.incumbent = optimizer.engine._evaluate_route(p, route, 0)
    generator = optimizer.greedy_generator
    index = len(route.trajectory)-1
    updated = replace(p, start_positions=route.trajectory[index].copy())
    plan = optimizer.update_state(updated, path_index=index)
    assert optimizer.last_diagnostics['route_reused']
    assert not optimizer.last_diagnostics['search_reset']
    assert optimizer.greedy_generator is generator
    assert np.allclose(plan.transition_solution.trajectory[0], updated.start_positions)
    assert evaluate_transition(updated.transition_problem(plan.final_positions), plan.transition_solution).feasible


def test_target_change_resets_search_and_rescores_hold():
    p = problem()
    optimizer = IncrementalReconfiguration()
    optimizer.initialize(p)
    optimizer.improve_step(1, max_work_units=1)
    old_generator = optimizer.greedy_generator
    updated = replace(p, scenario=replace(p.scenario, target_weights=p.scenario.target_weights*2))
    optimizer.update_state(updated)
    assert optimizer.last_diagnostics['search_reset']
    assert optimizer.greedy_generator is not old_generator
    assert optimizer.completed_search_units == 0


def transit_fixture(with_wait=False):
    base = problem()
    start = np.array([[10., 10.], [14., 10.]])
    goal = start+[30., 0.]
    scenario = replace(base.scenario, n_uavs=2, targets=goal.copy(),
                       target_weights=np.ones(2), sensing_radius=1)
    p = replace(base, scenario=scenario, start_positions=start,
                obstacles=(), max_speed=5)
    optimizer = IncrementalReconfiguration()
    optimizer.initialize(p)
    path = np.array([start, start, goal] if with_wait else [start, goal])
    route = optimizer.engine.planner._certify_waypoints(p.transition_problem(goal), path, np.arange(2))
    assert route is not None
    optimizer.incumbent = optimizer.engine._evaluate_route(p, route, 0)
    return p, optimizer, goal


@pytest.mark.parametrize('explicit_index', [False, True])
def test_repeated_execution_reaches_goal_without_inserting_waits(explicit_index):
    p, optimizer, goal = transit_fixture()
    steps = len(optimizer.get_plan().transition_solution.trajectory)-1
    for _ in range(steps):
        path = optimizer.get_plan().transition_solution.trajectory
        measured = path[1].copy()
        assert not np.array_equal(measured, p.start_positions)
        p = replace(p, start_positions=measured)
        plan = optimizer.update_state(p, **({'path_index': 1} if explicit_index else {}))
        assert len(plan.transition_solution.trajectory) == len(path)-1
        assert evaluate_transition(p.transition_problem(plan.final_positions), plan.transition_solution).feasible
    assert np.array_equal(p.start_positions, goal)
    assert len(optimizer.get_plan().transition_solution.trajectory) == 1


def test_executed_wait_advances_even_when_telemetry_is_unchanged():
    p, optimizer, _ = transit_fixture(with_wait=True)
    path = optimizer.get_plan().transition_solution.trajectory
    assert np.array_equal(path[0], path[1])
    plan = optimizer.update_state(p, path_index=1)
    assert len(plan.transition_solution.trajectory) == len(path)-1
    assert not np.array_equal(plan.transition_solution.trajectory[0],
                              plan.transition_solution.trajectory[1])


def test_off_route_measurement_keeps_checked_bridge():
    p, optimizer, _ = transit_fixture()
    path = optimizer.get_plan().transition_solution.trajectory
    p = replace(p, start_positions=path[1]+[0., 0.1])
    plan = optimizer.update_state(p, path_index=1)
    assert np.array_equal(plan.transition_solution.trajectory[0], p.start_positions)
    assert np.allclose(plan.transition_solution.trajectory[1], path[1])
    assert evaluate_transition(p.transition_problem(plan.final_positions), plan.transition_solution).feasible


def test_invalid_index_is_checked_even_for_unchanged_telemetry():
    p, optimizer, _ = transit_fixture()
    with pytest.raises(ValueError):
        optimizer.update_state(p, path_index=10000)


def test_certified_connectivity_boundary_is_accepted_as_next_start():
    p, _, _ = transit_fixture()
    start = np.array([[10., 10.], [20.+0.5e-9, 10.]])
    p = replace(p, scenario=replace(p.scenario, communication_radius=10), start_positions=start)
    assert IncrementalReconfiguration().initialize(p).evaluation.transition_feasible
    with pytest.raises(ValueError, match='connected'):
        replace(p, start_positions=np.array([[10., 10.], [20.+1e-6, 10.]]))


def test_new_obstacle_invalidates_old_route_and_returns_certified_hold():
    from src.obstacles import AxisAlignedRectangle
    p = problem()
    optimizer = IncrementalReconfiguration()
    optimizer.initialize(p)
    goal = p.start_positions+[0., -60.]
    route = optimizer.engine.planner._certify_waypoints(
        p.transition_problem(goal), np.array([p.start_positions, goal]), np.arange(10))
    assert route is not None
    optimizer.incumbent = optimizer.engine._evaluate_route(p, route, 0)
    x, y = goal[0]
    updated = replace(p, obstacles=p.obstacles+(AxisAlignedRectangle(x-1, y-1, x+1, y+1),))
    plan = optimizer.update_state(updated)
    assert not optimizer.last_diagnostics['route_reused']
    assert optimizer.last_diagnostics['search_reset']
    assert np.array_equal(plan.final_positions, updated.start_positions)
    assert evaluate_transition(updated.transition_problem(plan.final_positions), plan.transition_solution).feasible


def test_identical_telemetry_does_not_restart_pending_work():
    p = problem()
    optimizer = IncrementalReconfiguration()
    optimizer.initialize(p)
    optimizer._enqueue(p.start_positions+[0., 10.])
    optimizer.pending[0]['slices'] = 2
    generator = optimizer.greedy_generator
    optimizer.update_state(p)
    assert optimizer.pending[0]['slices'] == 2
    assert optimizer.greedy_generator is generator


def test_invalid_measured_state_never_exposes_stale_route():
    p = problem()
    optimizer = IncrementalReconfiguration()
    optimizer.initialize(p)
    positions = p.start_positions.copy()
    measured = replace(p, start_positions=positions)
    positions[1] = positions[0]
    with pytest.raises(ValueError):
        optimizer.update_state(measured)
    with pytest.raises(RuntimeError):
        optimizer.get_plan()


def test_waypoint_progress_is_saved_and_used_on_next_slice(monkeypatch):
    import src.proposed.incremental_reconfiguration as module
    p = problem()
    optimizer = IncrementalReconfiguration()
    optimizer.initialize(p)
    goal = p.start_positions+[0., 10.]
    optimizer._enqueue(goal)
    task = optimizer.pending[0]
    task['quality'] = (2., 2.)  # Force a search task regardless of this map's coverage.
    task['phase'] = 'waypoints'
    checkpoint = np.linspace(p.start_positions, goal, 25)
    seeds = []

    def optimize(tp, goals, certify, **kwargs):
        seeds.append(kwargs['initial_trajectory'])
        kwargs['progress_callback'](checkpoint.copy())
        return None

    monkeypatch.setattr(module, 'optimize_path', optimize)
    optimizer._path_unit(float('inf'))
    optimizer._path_unit(float('inf'))
    assert seeds[0] is None
    assert np.array_equal(seeds[1], checkpoint)
    assert optimizer.pending[0]['slices'] == 2
    assert np.array_equal(optimizer.get_plan().final_positions, p.start_positions)
