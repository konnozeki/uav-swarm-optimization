import numpy as np
from dataclasses import replace

from src.experiments.reconfiguration_cases import showcase_reconfiguration_profile
from src.proposed import BackboneTransitionPlanner, TransitionAwareGA
from src.transition import evaluate_transition


def legacy_wide_spacing_problem():
    # Preserve the original 75 m deadlock regression, independent of defaults.
    problem = showcase_reconfiguration_profile()[0]
    return replace(problem, scenario=replace(problem.scenario, min_separation=75.0))


def test_joint_path_reaches_greedy_formation_where_local_controller_deadlocks():
    problem = legacy_wide_spacing_problem()
    goal = TransitionAwareGA()._obstacle_aware_greedy_goal(problem)
    transition_problem = problem.transition_problem(goal)
    local = BackboneTransitionPlanner(enable_global_planning=False).solve(transition_problem)
    assert not evaluate_transition(transition_problem, local).feasible

    solution = BackboneTransitionPlanner().solve(transition_problem)
    metrics = evaluate_transition(transition_problem, solution)
    assert metrics.feasible
    assert metrics.backbone_certified
    assert metrics.obstacle_segment_violations == 0
    assert metrics.min_continuous_pair_distance >= problem.scenario.min_separation - 1e-9
    assert np.allclose(solution.trajectory[0], problem.start_positions)
    assert np.allclose(solution.trajectory[-1], goal[solution.assignment])


def test_exhausted_global_budget_does_not_claim_unverified_success():
    problem = legacy_wide_spacing_problem()
    goal = TransitionAwareGA()._obstacle_aware_greedy_goal(problem)
    transition_problem = problem.transition_problem(goal)
    solution = BackboneTransitionPlanner(global_iterations=1).solve(transition_problem)
    assert not solution.reached_goal
    assert not evaluate_transition(transition_problem, solution).feasible
