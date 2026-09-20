from __future__ import annotations

from dataclasses import dataclass, field
import math

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from .graph_ops import communication_graph_from_positions, pairwise_distances
from .obstacles import (
    AxisAlignedRectangle,
    formation_obstacle_free,
    obstacles_within_bounds,
    swarm_segment_obstacle_free,
    swarm_segment_obstacle_violations,
)


TRANSITION_TOL = 1e-9


@dataclass(frozen=True)
class TransitionProblem:
    """A generic formation-A -> formation-B transition problem.

    Checkpoint 03 deliberately does not assume that formation A is a deployment
    base. A and B may be any two valid swarm formations. This makes the
    transition planner reusable later for dynamic reconfiguration, UAV failure,
    moving hotspots, or a base-deployment scenario as a special case.

    The goal formation is treated as an unlabelled set by default because UAVs
    are interchangeable in the sensing model. When allow_reassignment is true,
    a planner may assign UAV identities to goal slots before moving.
    """

    name: str
    width: float
    height: float
    start_positions: np.ndarray
    goal_positions: np.ndarray
    communication_radius: float
    min_separation: float
    max_speed: float
    dt: float = 1.0
    allow_reassignment: bool = True
    obstacles: tuple[AxisAlignedRectangle, ...] = ()
    obstacle_clearance: float = 0.0

    def __post_init__(self) -> None:
        start = np.asarray(self.start_positions, dtype=float)
        goal = np.asarray(self.goal_positions, dtype=float)

        if start.ndim != 2 or start.shape[1] != 2:
            raise ValueError("start_positions must have shape (N, 2)")
        if goal.shape != start.shape:
            raise ValueError("goal_positions must have the same shape as start_positions")
        if len(start) == 0:
            raise ValueError("a transition must contain at least one UAV")
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
            raise ValueError("formation coordinates must be finite")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("map dimensions must be positive")
        if self.communication_radius <= 0:
            raise ValueError("communication_radius must be positive")
        if self.min_separation < 0:
            raise ValueError("min_separation must be non-negative")
        if self.max_speed <= 0 or self.dt <= 0:
            raise ValueError("max_speed and dt must be positive")
        if self.obstacle_clearance < 0:
            raise ValueError("obstacle_clearance must be non-negative")

        obstacles = tuple(self.obstacles)

        if not obstacles_within_bounds(
            obstacles,
            self.width,
            self.height,
        ):
            raise ValueError("all obstacles must lie inside map bounds")

        for label, positions in (("start", start), ("goal", goal)):
            if np.any(positions[:, 0] < 0) or np.any(positions[:, 0] > self.width):
                raise ValueError(f"{label} formation leaves horizontal map bounds")
            if np.any(positions[:, 1] < 0) or np.any(positions[:, 1] > self.height):
                raise ValueError(f"{label} formation leaves vertical map bounds")
            if not formation_obstacle_free(
                positions,
                obstacles,
                self.obstacle_clearance,
            ):
                raise ValueError(
                    f"{label} formation intersects a no-fly obstacle"
                )

        object.__setattr__(self, "start_positions", start)
        object.__setattr__(self, "goal_positions", goal)
        object.__setattr__(self, "obstacles", obstacles)

    @property
    def n_uavs(self) -> int:
        return int(len(self.start_positions))

    @property
    def max_step(self) -> float:
        """Maximum distance one UAV may travel during one discrete time step."""
        return float(self.max_speed * self.dt)


@dataclass
class TransitionSolution:
    """Trajectory returned by a Checkpoint-03 transition planner."""

    trajectory: np.ndarray
    assigned_goals: np.ndarray
    assignment: np.ndarray
    planner: str
    reached_goal: bool
    deadlocked: bool = False

    # For a backbone planner, element t contains the tree edges certified for
    # the segment trajectory[t] -> trajectory[t + 1].
    backbones: list[tuple[tuple[int, int], ...]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.trajectory = np.asarray(self.trajectory, dtype=float)
        self.assigned_goals = np.asarray(self.assigned_goals, dtype=float)
        self.assignment = np.asarray(self.assignment, dtype=int)

        if self.trajectory.ndim != 3 or self.trajectory.shape[2] != 2:
            raise ValueError("trajectory must have shape (T + 1, N, 2)")
        if self.assigned_goals.shape != self.trajectory.shape[1:]:
            raise ValueError("assigned_goals must have shape (N, 2)")
        if self.assignment.shape != (self.trajectory.shape[1],):
            raise ValueError("assignment must have shape (N,)")


@dataclass(frozen=True)
class TransitionMetrics:
    """Metrics used to compare formation-transition planners."""

    reached_goal: bool
    deadlocked: bool
    n_steps: int
    elapsed_time_sec: float
    formation_time_sec: float
    total_travel_distance: float
    max_uav_travel_distance: float
    max_step_distance: float
    max_speed_observed: float
    assignment_bottleneck_distance: float
    straight_line_time_lower_bound_sec: float
    time_efficiency: float
    discrete_connected_rate: float
    sampled_continuous_connected_rate: float
    min_continuous_pair_distance: float
    collision_free: bool
    obstacle_free: bool
    obstacle_segment_violations: int
    in_bounds: bool
    speed_feasible: bool
    backbone_certified: bool
    feasible: bool

    def to_dict(self) -> dict:
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
        }


def formation_connected(positions: np.ndarray, communication_radius: float) -> bool:
    """Return whether the unit-disk communication graph is connected."""
    graph = communication_graph_from_positions(positions, communication_radius)
    return nx.is_connected(graph)


def formation_min_distance(positions: np.ndarray) -> float:
    """Smallest pairwise UAV distance in a formation."""
    positions = np.asarray(positions, dtype=float)
    if len(positions) <= 1:
        return math.inf

    distances = pairwise_distances(positions, positions)
    upper = distances[np.triu_indices(len(positions), k=1)]
    return float(np.min(upper))


def formation_collision_free(
    positions: np.ndarray,
    min_separation: float,
) -> bool:
    return formation_min_distance(positions) >= min_separation - TRANSITION_TOL


def _perfect_matching_exists(distance: np.ndarray, threshold: float) -> bool:
    """Check if every UAV can be matched to a goal within threshold.

    This feasibility test is the core of the bottleneck assignment. It avoids
    optimizing total distance when our main lower bound is the slowest UAV.
    """
    n = distance.shape[0]
    graph = nx.Graph()

    left = [("u", i) for i in range(n)]
    right = [("g", j) for j in range(n)]
    graph.add_nodes_from(left, bipartite=0)
    graph.add_nodes_from(right, bipartite=1)

    for i in range(n):
        for j in range(n):
            if distance[i, j] <= threshold + TRANSITION_TOL:
                graph.add_edge(("u", i), ("g", j))

    matching = nx.algorithms.bipartite.maximum_matching(
        graph,
        top_nodes=set(left),
    )
    return all(node in matching for node in left)


def bottleneck_goal_assignment(
    start_positions: np.ndarray,
    goal_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Assign indistinguishable UAVs to goal slots by min-max distance.

    Primary objective:
        min_pi max_i ||start_i - goal_pi(i)||.

    Once the minimum bottleneck threshold is known, Hungarian assignment is used
    as a deterministic tie-breaker to minimize total distance inside that
    threshold. This also gives the straight-line formation-time lower bound.
    """
    start = np.asarray(start_positions, dtype=float)
    goal = np.asarray(goal_positions, dtype=float)

    if start.shape != goal.shape:
        raise ValueError("start and goal formations must have the same shape")

    distance = pairwise_distances(start, goal)
    thresholds = np.unique(distance.ravel())

    lo = 0
    hi = len(thresholds) - 1
    best = hi

    while lo <= hi:
        mid = (lo + hi) // 2
        threshold = float(thresholds[mid])

        if _perfect_matching_exists(distance, threshold):
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1

    threshold = float(thresholds[best])
    allowed = distance <= threshold + TRANSITION_TOL

    # Use a cost much larger than every real path so the Hungarian tie-breaker
    # cannot pick an edge outside the bottleneck-feasible graph.
    large_cost = max(float(np.max(distance)), 1.0) * 1e6
    cost = np.where(allowed, distance, large_cost)

    rows, cols = linear_sum_assignment(cost)

    if not np.all(allowed[rows, cols]):
        raise RuntimeError("failed to recover a bottleneck-feasible assignment")

    assignment = np.empty(len(start), dtype=int)
    assignment[rows] = cols
    assigned_goals = goal[assignment]

    bottleneck = float(
        np.max(np.linalg.norm(start - assigned_goals, axis=1))
    )
    return assigned_goals, assignment, bottleneck


def assign_goals(
    problem: TransitionProblem,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return goal positions ordered by UAV identity."""
    if problem.allow_reassignment:
        return bottleneck_goal_assignment(
            problem.start_positions,
            problem.goal_positions,
        )

    assignment = np.arange(problem.n_uavs, dtype=int)
    assigned = problem.goal_positions.copy()
    bottleneck = float(
        np.max(
            np.linalg.norm(
                problem.start_positions - assigned,
                axis=1,
            )
        )
    )
    return assigned, assignment, bottleneck


def clip_to_bounds(
    positions: np.ndarray,
    problem: TransitionProblem,
) -> np.ndarray:
    out = np.asarray(positions, dtype=float).copy()
    out[:, 0] = np.clip(out[:, 0], 0.0, problem.width)
    out[:, 1] = np.clip(out[:, 1], 0.0, problem.height)
    return out


def bounded_goal_step(
    current: np.ndarray,
    goals: np.ndarray,
    max_step: float,
) -> np.ndarray:
    """Move every UAV directly toward its goal by at most max_step."""
    current = np.asarray(current, dtype=float)
    goals = np.asarray(goals, dtype=float)

    delta = goals - current
    distance = np.linalg.norm(delta, axis=1)

    scale = np.ones(len(current), dtype=float)
    mask = distance > max_step

    if np.any(mask):
        scale[mask] = max_step / distance[mask]

    return current + delta * scale[:, None]


def clip_displacement(
    start: np.ndarray,
    candidate: np.ndarray,
    max_step: float,
) -> np.ndarray:
    """Project each endpoint into its per-step speed disk."""
    start = np.asarray(start, dtype=float)
    candidate = np.asarray(candidate, dtype=float)

    delta = candidate - start
    distance = np.linalg.norm(delta, axis=1)

    scale = np.ones(len(start), dtype=float)
    mask = distance > max_step

    if np.any(mask):
        scale[mask] = max_step / distance[mask]

    return start + delta * scale[:, None]


def continuous_min_pair_distance(
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    """Exact minimum UAV separation during one linear transition segment.

    Relative position of a UAV pair is affine in normalized time tau:
        r(tau) = r0 + tau * dr, tau in [0, 1].

    Squared distance is a convex quadratic, so its minimum is obtained by
    projecting -r0 dot dr / ||dr||^2 onto [0, 1]. Checking only segment
    endpoints would miss crossing trajectories; this function does not.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    n = len(start)
    if n <= 1:
        return math.inf

    minimum = math.inf

    for i in range(n):
        for j in range(i + 1, n):
            r0 = start[j] - start[i]
            dr = (end[j] - end[i]) - r0
            denom = float(np.dot(dr, dr))

            if denom <= 1e-15:
                tau = 0.0
            else:
                tau = float(
                    np.clip(
                        -np.dot(r0, dr) / denom,
                        0.0,
                        1.0,
                    )
                )

            separation = r0 + tau * dr
            minimum = min(
                minimum,
                float(np.linalg.norm(separation)),
            )

    return float(minimum)


def segment_collision_free(
    start: np.ndarray,
    end: np.ndarray,
    min_separation: float,
) -> bool:
    return (
        continuous_min_pair_distance(start, end)
        >= min_separation - TRANSITION_TOL
    )


def sampled_segment_connected(
    start: np.ndarray,
    end: np.ndarray,
    communication_radius: float,
    samples: int = 9,
) -> bool:
    """Sample connectivity along a linear segment.

    This is intentionally used only as a generic diagnostic/baseline guard.
    The proposed backbone planner has a stronger certificate: every edge of one
    spanning tree is valid at both segment endpoints, and convexity of Euclidean
    norm then guarantees those edges remain valid throughout the segment.
    """
    if samples < 2:
        samples = 2

    for tau in np.linspace(0.0, 1.0, samples):
        positions = (1.0 - tau) * start + tau * end

        if not formation_connected(
            positions,
            communication_radius,
        ):
            return False

    return True


def backbone_segment_certified(
    start: np.ndarray,
    end: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    communication_radius: float,
) -> bool:
    """Certify continuous connectivity for one linear segment.

    If every spanning-tree edge has length <= Rc at both endpoints, convexity
    of Euclidean norm guarantees that every such edge survives for the complete
    linear interpolation between those endpoints.
    """
    n = len(start)
    if n <= 1:
        return True

    if len(edges) != n - 1:
        return False

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)

    if not nx.is_tree(graph):
        return False

    for i, j in edges:
        if (
            np.linalg.norm(start[i] - start[j])
            > communication_radius + TRANSITION_TOL
        ):
            return False
        if (
            np.linalg.norm(end[i] - end[j])
            > communication_radius + TRANSITION_TOL
        ):
            return False

    return True


def evaluate_transition(
    problem: TransitionProblem,
    solution: TransitionSolution,
    goal_tolerance: float = 1e-6,
    connectivity_samples_per_segment: int = 9,
) -> TransitionMetrics:
    """Evaluate a complete formation transition."""
    trajectory = solution.trajectory
    n_steps = len(trajectory) - 1

    if trajectory.shape[1:] != problem.start_positions.shape:
        raise ValueError("trajectory UAV shape does not match transition problem")
    if not np.allclose(
        trajectory[0],
        problem.start_positions,
        atol=TRANSITION_TOL,
    ):
        raise ValueError("trajectory must start at problem.start_positions")

    step_delta = np.diff(trajectory, axis=0)
    step_distance = np.linalg.norm(step_delta, axis=2)

    if n_steps:
        total_per_uav = np.sum(step_distance, axis=0)
        total_travel = float(np.sum(total_per_uav))
        max_uav_travel = float(np.max(total_per_uav))
        max_step_distance = float(np.max(step_distance))
    else:
        total_travel = 0.0
        max_uav_travel = 0.0
        max_step_distance = 0.0

    max_speed = max_step_distance / problem.dt
    speed_feasible = (
        max_step_distance
        <= problem.max_step + TRANSITION_TOL
    )

    in_bounds = bool(
        np.all(trajectory[:, :, 0] >= -TRANSITION_TOL)
        and np.all(
            trajectory[:, :, 0]
            <= problem.width + TRANSITION_TOL
        )
        and np.all(trajectory[:, :, 1] >= -TRANSITION_TOL)
        and np.all(
            trajectory[:, :, 1]
            <= problem.height + TRANSITION_TOL
        )
    )

    discrete_connected = [
        formation_connected(state, problem.communication_radius)
        for state in trajectory
    ]
    discrete_connected_rate = float(np.mean(discrete_connected))

    sampled_connected_checks = []
    minimum_separation = math.inf
    obstacle_segment_violations = 0

    for t in range(n_steps):
        start = trajectory[t]
        end = trajectory[t + 1]

        sampled_connected_checks.append(
            sampled_segment_connected(
                start,
                end,
                problem.communication_radius,
                samples=connectivity_samples_per_segment,
            )
        )
        minimum_separation = min(
            minimum_separation,
            continuous_min_pair_distance(start, end),
        )
        obstacle_segment_violations += (
            swarm_segment_obstacle_violations(
                start,
                end,
                problem.obstacles,
                problem.obstacle_clearance,
            )
        )

    if n_steps == 0:
        sampled_connected_rate = float(discrete_connected_rate)
        minimum_separation = formation_min_distance(trajectory[0])
    else:
        sampled_connected_rate = float(
            np.mean(sampled_connected_checks)
        )

    collision_free = (
        minimum_separation
        >= problem.min_separation - TRANSITION_TOL
    )

    if n_steps == 0:
        obstacle_free = formation_obstacle_free(
            trajectory[0],
            problem.obstacles,
            problem.obstacle_clearance,
        )
    else:
        obstacle_free = obstacle_segment_violations == 0

    reached_goal = bool(
        solution.reached_goal
        and np.all(
            np.linalg.norm(
                trajectory[-1] - solution.assigned_goals,
                axis=1,
            )
            <= goal_tolerance
        )
    )

    elapsed_time = n_steps * problem.dt
    formation_time = (
        float(elapsed_time)
        if reached_goal
        else float("nan")
    )

    bottleneck = float(
        np.max(
            np.linalg.norm(
                problem.start_positions - solution.assigned_goals,
                axis=1,
            )
        )
    )
    lower_bound = bottleneck / problem.max_speed

    if reached_goal and formation_time > 0:
        time_efficiency = lower_bound / formation_time
    elif reached_goal and lower_bound <= TRANSITION_TOL:
        time_efficiency = 1.0
    else:
        time_efficiency = 0.0

    backbone_certified = False

    if n_steps > 0 and len(solution.backbones) == n_steps:
        backbone_certified = all(
            backbone_segment_certified(
                trajectory[t],
                trajectory[t + 1],
                solution.backbones[t],
                problem.communication_radius,
            )
            for t in range(n_steps)
        )

    continuous_connectivity_ok = (
        backbone_certified
        or sampled_connected_rate >= 1.0 - TRANSITION_TOL
    )

    feasible = bool(
        reached_goal
        and in_bounds
        and speed_feasible
        and collision_free
        and obstacle_free
        and continuous_connectivity_ok
    )

    return TransitionMetrics(
        reached_goal=reached_goal,
        deadlocked=solution.deadlocked,
        n_steps=n_steps,
        elapsed_time_sec=float(elapsed_time),
        formation_time_sec=float(formation_time),
        total_travel_distance=total_travel,
        max_uav_travel_distance=max_uav_travel,
        max_step_distance=max_step_distance,
        max_speed_observed=float(max_speed),
        assignment_bottleneck_distance=bottleneck,
        straight_line_time_lower_bound_sec=float(lower_bound),
        time_efficiency=float(time_efficiency),
        discrete_connected_rate=discrete_connected_rate,
        sampled_continuous_connected_rate=sampled_connected_rate,
        min_continuous_pair_distance=float(minimum_separation),
        collision_free=collision_free,
        obstacle_free=bool(obstacle_free),
        obstacle_segment_violations=int(obstacle_segment_violations),
        in_bounds=in_bounds,
        speed_feasible=speed_feasible,
        backbone_certified=backbone_certified,
        feasible=feasible,
    )
