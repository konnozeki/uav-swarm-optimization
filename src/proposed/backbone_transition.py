from __future__ import annotations

import networkx as nx
import numpy as np
import time

from ..graph_ops import communication_graph_from_positions
from ..obstacles import (
    formation_obstacle_free,
    next_visibility_waypoint,
    swarm_segment_obstacle_free,
)
from ..transition import (
    TRANSITION_TOL,
    TransitionProblem,
    TransitionSolution,
    assign_goals,
    bounded_goal_step,
    clip_displacement,
    clip_to_bounds,
    formation_collision_free,
    formation_connected,
    segment_collision_free,
)


class BackboneTransitionPlanner:
    """Connectivity-preserving formation-transition planner.

    At every state, the current communication graph is reduced to a
    maximum-slack spanning tree. Those N-1 edges form a temporary connectivity
    backbone for the next motion segment.

    A maximum-speed direct step is generated first, then projected so that every
    backbone edge is valid at the next endpoint. If a backbone edge is valid at
    both endpoints of a linear motion segment, convexity of Euclidean norm
    guarantees it remains valid for the complete segment. Since the protected
    edges form a spanning tree, swarm connectivity is continuously certified.

    The tree is rebuilt at the next state. Therefore the swarm is not permanently
    tied to the topology of formation A.

    This planner is a heuristic for minimizing transition time. It does not claim
    a globally optimal transition.
    """

    name = "backbone_transition"

    def __init__(
        self,
        max_steps: int = 500,
        goal_tolerance: float = 1e-6,
        communication_margin: float = 0.0,
        projection_iterations: int = 20,
        backtracking_factor: float = 0.5,
        minimum_scale: float = 1e-4,
        stall_limit: int = 12,
        shortcut_lookahead: int = 12,
        enable_smoothing: bool = False,
        enable_global_planning: bool = True,
        global_knots: int = 25,
        global_iterations: int = 800,
    ) -> None:
        if communication_margin < 0:
            raise ValueError("communication_margin must be non-negative")
        if not 0 < backtracking_factor < 1:
            raise ValueError("backtracking_factor must be in (0, 1)")
        if global_knots < 3 or global_iterations < 1:
            raise ValueError("global_knots >= 3 and global_iterations >= 1 required")

        self.max_steps = max_steps
        self.goal_tolerance = goal_tolerance
        self.communication_margin = communication_margin
        self.projection_iterations = projection_iterations
        self.backtracking_factor = backtracking_factor
        self.minimum_scale = minimum_scale
        self.stall_limit = stall_limit
        self.shortcut_lookahead = max(2, int(shortcut_lookahead))
        # Smoothing is useful when exporting a presentation trajectory, but it
        # is not part of finding a feasible trajectory.  Running its O(T *
        # lookahead) set of extra certified plans for every GA chromosome was
        # the dominant cost of CP3.  Keep it opt-in for offline visualisation.
        self.enable_smoothing = bool(enable_smoothing)
        self.enable_global_planning = bool(enable_global_planning)
        self.global_knots = int(global_knots)
        self.global_iterations = int(global_iterations)

    def _global_path(self, problem, assigned_goals, assignment, *,
                     initial_trajectory=None, endpoint_radius=0.0,
                     endpoint_accept=None, endpoint_terms=None, deadline=None):
        from .formation_path import optimize_path

        def certify(path):
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            if endpoint_accept is not None and not endpoint_accept(path[-1]):
                return None
            return self._certify_waypoints(problem, path, assignment, deadline=deadline)

        return optimize_path(
            problem, assigned_goals, certify,
            knots=self.global_knots, max_iterations=self.global_iterations,
            initial_trajectory=initial_trajectory, endpoint_radius=endpoint_radius,
            endpoint_terms=endpoint_terms, deadline=deadline,
        )

    def _certify_waypoints(self, problem, path, assignment, *, deadline=None):
        from ..transition import evaluate_transition

        if not len(path) or not np.isfinite(path).all():
            return None
        states = [path[0].copy()]
        backbones = []
        for start, end in zip(path[:-1], path[1:]):
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            segment = self._linear_shortcut(start, end, problem)
            if segment is None:
                return None
            states.extend(segment[0])
            backbones.extend(segment[1])
            if len(states) - 1 > self.max_steps:
                return None
        solution = TransitionSolution(
            trajectory=np.asarray(states), assigned_goals=path[-1].copy(),
            assignment=assignment, planner=self.name, reached_goal=True,
            backbones=backbones,
        )
        if evaluate_transition(problem, solution).feasible:
            return solution
        return None

    def refine_path(self, problem, *, initial_trajectory=None, endpoint_radius=0.0,
                    endpoint_accept=None, endpoint_terms=None, deadline=None):
        """Plan a proposed endpoint, optionally allowing per-coordinate slack.

        UAV identity must already be assigned by the caller. With endpoint slack,
        callers must evaluate the *returned* endpoint, not the requested one.
        This explicit API keeps fixed-goal `solve` semantics unchanged.
        """
        if problem.allow_reassignment:
            raise ValueError("refine_path requires an identity-assigned goal")
        return self._global_path(
            problem, problem.goal_positions, np.arange(problem.n_uavs),
            initial_trajectory=initial_trajectory, endpoint_radius=endpoint_radius,
            endpoint_accept=endpoint_accept, endpoint_terms=endpoint_terms,
            deadline=deadline,
        )

    def _backbone(
        self,
        positions: np.ndarray,
        problem: TransitionProblem,
    ) -> tuple[tuple[int, int], ...]:
        """Return a maximum-slack spanning tree of the current graph."""
        effective_radius = (
            problem.communication_radius
            - self.communication_margin
        )

        if effective_radius <= 0:
            raise ValueError(
                "communication_margin leaves no usable communication radius"
            )

        graph = communication_graph_from_positions(
            positions,
            effective_radius + TRANSITION_TOL,
        )

        if not nx.is_connected(graph):
            raise ValueError(
                "current formation is not connected under the configured "
                "communication margin"
            )

        # Larger slack means a shorter, more robust communication edge.
        for _, _, data in graph.edges(data=True):
            distance = float(data["distance"])
            data["slack"] = effective_radius - distance

        tree = nx.maximum_spanning_tree(
            graph,
            weight="slack",
        )

        return tuple(
            (int(i), int(j))
            for i, j in tree.edges()
        )

    def _project_backbone(
        self,
        current: np.ndarray,
        candidate: np.ndarray,
        backbone: tuple[tuple[int, int], ...],
        problem: TransitionProblem,
    ) -> np.ndarray:
        """Project a candidate endpoint toward the backbone-feasible set.

        Every violated tree edge receives a symmetric pairwise correction.
        Per-UAV speed disks and map bounds are re-applied after every sweep
        because correcting one edge can perturb its neighboring tree edges.
        """
        out = candidate.copy()
        max_edge_length = (
            problem.communication_radius
            - self.communication_margin
        )

        for _ in range(self.projection_iterations):
            changed = False

            for i, j in backbone:
                delta = out[j] - out[i]
                distance = float(np.linalg.norm(delta))

                if distance <= max_edge_length + TRANSITION_TOL:
                    continue

                changed = True

                if distance <= 1e-15:
                    continue

                direction = delta / distance
                excess = distance - max_edge_length

                correction = 0.5 * excess * direction
                out[i] += correction
                out[j] -= correction

            out = clip_displacement(
                current,
                out,
                problem.max_step,
            )
            out = clip_to_bounds(out, problem)

            if not changed:
                break

        return out

    def _backbone_endpoint_ok(
        self,
        positions: np.ndarray,
        backbone: tuple[tuple[int, int], ...],
        problem: TransitionProblem,
    ) -> bool:
        max_edge_length = (
            problem.communication_radius
            - self.communication_margin
        )

        return all(
            np.linalg.norm(positions[i] - positions[j])
            <= max_edge_length + TRANSITION_TOL
            for i, j in backbone
        )

    def _steering_targets(
        self,
        current: np.ndarray,
        assigned_goals: np.ndarray,
        problem: TransitionProblem,
    ) -> np.ndarray:
        """Return per-UAV obstacle-aware local targets.

        Without obstacles the local targets are simply the assigned goals. When
        a direct path is blocked, a tiny visibility graph around expanded
        rectangle corners returns the first waypoint of a shortest polygonal
        detour. The graph is recomputed at every planner step, so UAVs naturally
        leave a waypoint as soon as the final goal becomes directly visible.
        """
        if not problem.obstacles:
            return assigned_goals

        targets = assigned_goals.copy()

        for i in range(len(current)):
            waypoint = next_visibility_waypoint(
                current[i],
                assigned_goals[i],
                problem.obstacles,
                problem.obstacle_clearance,
                problem.width,
                problem.height,
            )

            if waypoint is None:
                # No geometric detour is currently visible. Keeping this UAV in
                # place lets the rest of the connected formation reshape; stall
                # detection still bounds planner runtime if no progress emerges.
                targets[i] = current[i]
            else:
                targets[i] = waypoint

        return targets

    def _safe_backtracked_candidate(
        self,
        current: np.ndarray,
        projected: np.ndarray,
        backbone: tuple[tuple[int, int], ...],
        problem: TransitionProblem,
    ) -> np.ndarray | None:
        """Uniformly shorten a projected move until all constraints are safe."""
        displacement = projected - current
        scale = 1.0

        while scale >= self.minimum_scale:
            candidate = current + scale * displacement

            if not self._backbone_endpoint_ok(
                candidate,
                backbone,
                problem,
            ):
                scale *= self.backtracking_factor
                continue

            if not segment_collision_free(
                current,
                candidate,
                problem.min_separation,
            ):
                scale *= self.backtracking_factor
                continue

            if not formation_obstacle_free(
                candidate,
                problem.obstacles,
                problem.obstacle_clearance,
            ):
                scale *= self.backtracking_factor
                continue

            if not swarm_segment_obstacle_free(
                current,
                candidate,
                problem.obstacles,
                problem.obstacle_clearance,
            ):
                scale *= self.backtracking_factor
                continue

            return candidate

        return None

    def _common_backbone(
        self,
        start: np.ndarray,
        end: np.ndarray,
        problem: TransitionProblem,
    ) -> tuple[tuple[int, int], ...] | None:
        """Find one spanning tree valid at both ends of a linear segment.

        If an edge is within communication range at both endpoints, convexity
        guarantees it stays within range for the whole interpolation. A spanning
        tree made only from such edges therefore certifies the shortcut.
        """
        n = len(start)

        if n <= 1:
            return ()

        effective_radius = (
            problem.communication_radius
            - self.communication_margin
        )

        graph = nx.Graph()
        graph.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                d0 = float(
                    np.linalg.norm(
                        start[i] - start[j]
                    )
                )
                d1 = float(
                    np.linalg.norm(
                        end[i] - end[j]
                    )
                )

                if (
                    d0 <= effective_radius + TRANSITION_TOL
                    and d1 <= effective_radius + TRANSITION_TOL
                ):
                    graph.add_edge(
                        i,
                        j,
                        slack=min(
                            effective_radius - d0,
                            effective_radius - d1,
                        ),
                    )

        if not nx.is_connected(graph):
            return None

        tree = nx.maximum_spanning_tree(
            graph,
            weight="slack",
        )

        return tuple(
            (int(i), int(j))
            for i, j in tree.edges()
        )

    def _linear_shortcut(
        self,
        start: np.ndarray,
        end: np.ndarray,
        problem: TransitionProblem,
    ):
        """Build the shortest-time straight interpolation if it is fully safe."""
        displacement = end - start
        max_distance = float(
            np.max(
                np.linalg.norm(
                    displacement,
                    axis=1,
                )
            )
        )

        n_steps = max(
            1,
            int(
                np.ceil(
                    max_distance
                    / max(problem.max_step, 1e-12)
                    - 1e-12
                )
            ),
        )

        states = []
        backbones = []
        previous = start

        for step in range(1, n_steps + 1):
            tau = step / n_steps
            current = (
                (1.0 - tau) * start
                + tau * end
            )

            if not segment_collision_free(
                previous,
                current,
                problem.min_separation,
            ):
                return None

            if not swarm_segment_obstacle_free(
                previous,
                current,
                problem.obstacles,
                problem.obstacle_clearance,
            ):
                return None

            backbone = self._common_backbone(
                previous,
                current,
                problem,
            )

            if backbone is None:
                return None

            states.append(current.copy())
            backbones.append(backbone)
            previous = current

        return states, backbones

    @staticmethod
    def _trajectory_travel(
        states: np.ndarray,
    ) -> float:
        if len(states) <= 1:
            return 0.0

        return float(
            np.sum(
                np.linalg.norm(
                    np.diff(states, axis=0),
                    axis=2,
                )
            )
        )

    def _smooth_trajectory(
        self,
        trajectory: np.ndarray,
        problem: TransitionProblem,
    ) -> tuple[np.ndarray, list[tuple[tuple[int, int], ...]]]:
        """Shortcut zig-zags while preserving every hard transition constraint.

        The raw online planner may take small corrective or waypoint moves.
        Starting from each state, this routine tries to connect to a later state
        with a straight, speed-limited interpolation. The shortcut is accepted
        only when it reduces step count or total travel and every new segment has
        an obstacle-free, collision-free, continuously connected certificate.
        """
        trajectory = np.asarray(
            trajectory,
            dtype=float,
        )

        if len(trajectory) <= 1:
            return trajectory.copy(), []

        result = [trajectory[0].copy()]
        result_backbones = []
        i = 0
        last = len(trajectory) - 1

        while i < last:
            chosen_j = i + 1
            chosen_states = None
            chosen_backbones = None

            max_j = min(
                last,
                i + self.shortcut_lookahead,
            )

            for j in range(max_j, i, -1):
                shortcut = self._linear_shortcut(
                    trajectory[i],
                    trajectory[j],
                    problem,
                )

                if shortcut is None:
                    continue

                states, backbones = shortcut
                old_steps = j - i
                new_steps = len(states)

                old_travel = self._trajectory_travel(
                    trajectory[i : j + 1]
                )
                shortcut_array = np.asarray(
                    [
                        trajectory[i],
                        *states,
                    ],
                    dtype=float,
                )
                new_travel = self._trajectory_travel(
                    shortcut_array
                )

                if (
                    new_steps < old_steps
                    or new_travel
                    < old_travel - 1e-6
                ):
                    chosen_j = j
                    chosen_states = states
                    chosen_backbones = backbones
                    break

            if chosen_states is None:
                shortcut = self._linear_shortcut(
                    trajectory[i],
                    trajectory[i + 1],
                    problem,
                )

                if shortcut is None:
                    # The raw planner already validated this segment, so this is
                    # only a defensive fallback for numerical edge cases.
                    chosen_states = [
                        trajectory[i + 1].copy()
                    ]
                    chosen_backbones = [
                        self._backbone(
                            trajectory[i],
                            problem,
                        )
                    ]
                else:
                    chosen_states, chosen_backbones = shortcut

            result.extend(chosen_states)
            result_backbones.extend(
                chosen_backbones
            )
            i = chosen_j

        return (
            np.asarray(result, dtype=float),
            result_backbones,
        )

    def solve(self, problem: TransitionProblem, *, deadline=None) -> TransitionSolution:
        if not formation_connected(
            problem.start_positions,
            problem.communication_radius,
        ):
            raise ValueError("start formation must be connected")

        if not formation_collision_free(
            problem.start_positions,
            problem.min_separation,
        ):
            raise ValueError("start formation must be collision-free")

        if not formation_obstacle_free(
            problem.start_positions,
            problem.obstacles,
            problem.obstacle_clearance,
        ):
            raise ValueError("start formation intersects a no-fly obstacle")

        assigned_goals, assignment, _ = assign_goals(problem)

        current = problem.start_positions.copy()
        trajectory = [current.copy()]
        backbones: list[tuple[tuple[int, int], ...]] = []

        deadlocked = False
        stall_count = 0

        for _ in range(self.max_steps):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            goal_distance = np.linalg.norm(
                assigned_goals - current,
                axis=1,
            )

            if np.all(goal_distance <= self.goal_tolerance):
                break

            backbone = self._backbone(current, problem)

            steering_targets = self._steering_targets(
                current,
                assigned_goals,
                problem,
            )

            desired = bounded_goal_step(
                current,
                steering_targets,
                problem.max_step,
            )
            desired = clip_to_bounds(desired, problem)

            projected = self._project_backbone(
                current,
                desired,
                backbone,
                problem,
            )

            candidate = self._safe_backtracked_candidate(
                current,
                projected,
                backbone,
                problem,
            )

            if candidate is None:
                deadlocked = True
                break

            previous_total_distance = float(np.sum(goal_distance))

            current = candidate
            trajectory.append(current.copy())
            backbones.append(backbone)

            new_total_distance = float(
                np.sum(
                    np.linalg.norm(
                        assigned_goals - current,
                        axis=1,
                    )
                )
            )

            if (
                previous_total_distance - new_total_distance
                <= 1e-8
            ):
                stall_count += 1
            else:
                stall_count = 0

            if stall_count >= self.stall_limit:
                deadlocked = True
                break

        reached = bool(
            np.all(
                np.linalg.norm(
                    assigned_goals - current,
                    axis=1,
                )
                <= self.goal_tolerance
            )
        )

        final_trajectory = np.asarray(
            trajectory,
            dtype=float,
        )
        final_backbones = backbones

        # Collisions and connectivity can deadlock the local controller even
        # in open space, so joint planning is not restricted to obstacle maps.
        if not reached and self.enable_global_planning:
            global_solution = self._global_path(problem, assigned_goals, assignment, deadline=deadline)
            if global_solution is not None:
                return global_solution

        if (
            self.enable_smoothing
            and reached
            and len(final_trajectory) > 1
        ):
            (
                final_trajectory,
                final_backbones,
            ) = self._smooth_trajectory(
                final_trajectory,
                problem,
            )

        return TransitionSolution(
            trajectory=final_trajectory,
            assigned_goals=assigned_goals,
            assignment=assignment,
            planner=self.name,
            reached_goal=reached,
            deadlocked=deadlocked or not reached,
            backbones=final_backbones,
        )
