from __future__ import annotations

import numpy as np

from ..obstacles import (
    formation_obstacle_free,
    swarm_segment_obstacle_free,
)
from ..transition import (
    TransitionProblem,
    TransitionSolution,
    assign_goals,
    bounded_goal_step,
    clip_to_bounds,
    formation_connected,
    sampled_segment_connected,
    segment_collision_free,
)


class EndpointGuardTransitionPlanner:
    """Reactive feasibility baseline.

    A maximum-speed direct step is proposed first. If the candidate endpoint is
    disconnected, a collision occurs during the linear segment, or sampled
    connectivity is lost between endpoints, the complete swarm step is scaled
    down uniformly.

    This planner only reacts after generating an invalid move. It does not use
    graph structure to shape the move itself, which makes it a useful baseline
    against the proposed backbone-preserving planner.
    """

    name = "endpoint_guard_transition"

    def __init__(
        self,
        max_steps: int = 500,
        goal_tolerance: float = 1e-6,
        backtracking_factor: float = 0.5,
        minimum_scale: float = 1e-4,
        connectivity_samples: int = 9,
        stall_limit: int = 12,
    ) -> None:
        if not 0 < backtracking_factor < 1:
            raise ValueError("backtracking_factor must be in (0, 1)")

        self.max_steps = max_steps
        self.goal_tolerance = goal_tolerance
        self.backtracking_factor = backtracking_factor
        self.minimum_scale = minimum_scale
        self.connectivity_samples = connectivity_samples
        self.stall_limit = stall_limit

    def _step_is_safe(
        self,
        current: np.ndarray,
        candidate: np.ndarray,
        problem: TransitionProblem,
    ) -> bool:
        if not formation_connected(
            candidate,
            problem.communication_radius,
        ):
            return False

        if not segment_collision_free(
            current,
            candidate,
            problem.min_separation,
        ):
            return False

        if not formation_obstacle_free(
            candidate,
            problem.obstacles,
            problem.obstacle_clearance,
        ):
            return False

        if not swarm_segment_obstacle_free(
            current,
            candidate,
            problem.obstacles,
            problem.obstacle_clearance,
        ):
            return False

        return sampled_segment_connected(
            current,
            candidate,
            problem.communication_radius,
            samples=self.connectivity_samples,
        )

    def solve(self, problem: TransitionProblem) -> TransitionSolution:
        assigned_goals, assignment, _ = assign_goals(problem)

        current = problem.start_positions.copy()
        trajectory = [current.copy()]
        stall_count = 0
        deadlocked = False

        for _ in range(self.max_steps):
            goal_distance = np.linalg.norm(
                assigned_goals - current,
                axis=1,
            )

            if np.all(goal_distance <= self.goal_tolerance):
                break

            desired = bounded_goal_step(
                current,
                assigned_goals,
                problem.max_step,
            )
            desired = clip_to_bounds(desired, problem)
            displacement = desired - current

            previous_total_distance = float(np.sum(goal_distance))
            accepted = None
            scale = 1.0

            while scale >= self.minimum_scale:
                candidate = current + scale * displacement

                if self._step_is_safe(
                    current,
                    candidate,
                    problem,
                ):
                    accepted = candidate
                    break

                scale *= self.backtracking_factor

            if accepted is None:
                deadlocked = True
                break

            current = accepted
            trajectory.append(current.copy())

            new_total_distance = float(
                np.sum(
                    np.linalg.norm(
                        assigned_goals - current,
                        axis=1,
                    )
                )
            )

            # A planner can technically keep accepting microscopic moves forever.
            # Mark a deadlock when progress disappears for several consecutive
            # steps so benchmark runtime remains bounded and interpretable.
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

        return TransitionSolution(
            trajectory=np.asarray(trajectory),
            assigned_goals=assigned_goals,
            assignment=assignment,
            planner=self.name,
            reached_goal=reached,
            deadlocked=deadlocked or not reached,
        )
