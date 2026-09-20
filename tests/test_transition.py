import numpy as np

from src.baselines import DirectTransitionPlanner
from src.obstacles import (
    AxisAlignedRectangle,
    segment_intersects_obstacle,
)
from src.proposed import BackboneTransitionPlanner
from src.transition import (
    TransitionProblem,
    bottleneck_goal_assignment,
    continuous_min_pair_distance,
    evaluate_transition,
)


def _simple_problem():
    start = np.array([
        [200.0, 400.0],
        [320.0, 400.0],
        [440.0, 400.0],
        [560.0, 400.0],
    ])
    goal = start + np.array([180.0, 120.0])

    return TransitionProblem(
        name="simple_transition",
        width=1000.0,
        height=1000.0,
        start_positions=start,
        goal_positions=goal,
        communication_radius=180.0,
        min_separation=50.0,
        max_speed=40.0,
    )


def test_bottleneck_assignment_can_relabel_goal_slots():
    start = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
    ])
    goals = np.array([
        [10.0, 0.0],
        [0.0, 0.0],
    ])

    assigned, assignment, bottleneck = bottleneck_goal_assignment(
        start,
        goals,
    )

    assert np.allclose(assigned, start)
    assert np.array_equal(assignment, np.array([1, 0]))
    assert bottleneck == 0.0


def test_continuous_collision_check_detects_crossing_paths():
    start = np.array([
        [-1.0, 0.0],
        [1.0, 0.0],
    ])
    end = np.array([
        [1.0, 0.0],
        [-1.0, 0.0],
    ])

    assert continuous_min_pair_distance(start, end) == 0.0


def test_direct_transition_reaches_simple_goal():
    problem = _simple_problem()

    solution = DirectTransitionPlanner().solve(problem)
    metrics = evaluate_transition(problem, solution)

    assert metrics.reached_goal
    assert metrics.speed_feasible


def test_backbone_transition_is_certified_and_feasible():
    problem = _simple_problem()

    solution = BackboneTransitionPlanner().solve(problem)
    metrics = evaluate_transition(problem, solution)

    assert metrics.reached_goal
    assert metrics.backbone_certified
    assert metrics.collision_free
    assert metrics.feasible



def test_segment_obstacle_check_detects_crossing():
    obstacle = AxisAlignedRectangle(
        4.0,
        -1.0,
        6.0,
        1.0,
        name="wall",
    )

    assert segment_intersects_obstacle(
        np.array([0.0, 0.0]),
        np.array([10.0, 0.0]),
        obstacle,
    )
    assert not segment_intersects_obstacle(
        np.array([0.0, 3.0]),
        np.array([10.0, 3.0]),
        obstacle,
    )


def test_backbone_transition_detours_around_rectangle():
    obstacle = AxisAlignedRectangle(
        400.0,
        400.0,
        600.0,
        600.0,
        name="center_block",
    )

    problem = TransitionProblem(
        name="single_uav_obstacle_detour",
        width=1000.0,
        height=1000.0,
        start_positions=np.array([[150.0, 500.0]]),
        goal_positions=np.array([[850.0, 500.0]]),
        communication_radius=200.0,
        min_separation=0.0,
        max_speed=40.0,
        obstacles=(obstacle,),
        obstacle_clearance=20.0,
    )

    direct = DirectTransitionPlanner().solve(problem)
    direct_metrics = evaluate_transition(problem, direct)

    proposed = BackboneTransitionPlanner().solve(problem)
    proposed_metrics = evaluate_transition(problem, proposed)

    assert not direct_metrics.obstacle_free
    assert not direct_metrics.feasible

    assert proposed_metrics.reached_goal
    assert proposed_metrics.obstacle_free
    assert proposed_metrics.feasible



def test_backbone_smoother_removes_redundant_zigzag():
    problem = TransitionProblem(
        name="shortcut_unit",
        width=100.0,
        height=100.0,
        start_positions=np.array([[10.0, 10.0]]),
        goal_positions=np.array([[30.0, 30.0]]),
        communication_radius=50.0,
        min_separation=0.0,
        max_speed=15.0,
    )

    raw = np.array([
        [[10.0, 10.0]],
        [[10.0, 25.0]],
        [[25.0, 25.0]],
        [[30.0, 30.0]],
    ])

    planner = BackboneTransitionPlanner()
    smoothed, backbones = planner._smooth_trajectory(
        raw,
        problem,
    )

    raw_travel = planner._trajectory_travel(raw)
    smooth_travel = planner._trajectory_travel(smoothed)

    assert smooth_travel < raw_travel
    assert len(smoothed) <= len(raw)
    assert len(backbones) == len(smoothed) - 1
