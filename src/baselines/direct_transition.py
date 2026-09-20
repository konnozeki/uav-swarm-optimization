from __future__ import annotations

import numpy as np

from ..transition import (
    TransitionProblem,
    TransitionSolution,
    assign_goals,
    bounded_goal_step,
)


class DirectTransitionPlanner:
    """Naive lower-bound baseline.

    Every UAV moves directly toward its assigned goal at the maximum allowed
    per-step speed. No connectivity or collision handling is performed.

    The planner is intentionally unsafe: it tells us how fast deployment could
    be if formation-transition constraints were ignored.
    """

    name = "direct_transition"

    def __init__(
        self,
        max_steps: int = 500,
        goal_tolerance: float = 1e-6,
    ) -> None:
        self.max_steps = max_steps
        self.goal_tolerance = goal_tolerance

    def solve(self, problem: TransitionProblem) -> TransitionSolution:
        assigned_goals, assignment, _ = assign_goals(problem)

        current = problem.start_positions.copy()
        trajectory = [current.copy()]
        reached = False

        for _ in range(self.max_steps):
            distance = np.linalg.norm(
                assigned_goals - current,
                axis=1,
            )

            if np.all(distance <= self.goal_tolerance):
                reached = True
                break

            current = bounded_goal_step(
                current,
                assigned_goals,
                problem.max_step,
            )
            trajectory.append(current.copy())

        if np.all(
            np.linalg.norm(
                assigned_goals - current,
                axis=1,
            )
            <= self.goal_tolerance
        ):
            reached = True

        return TransitionSolution(
            trajectory=np.asarray(trajectory),
            assigned_goals=assigned_goals,
            assignment=assignment,
            planner=self.name,
            reached_goal=reached,
            deadlocked=not reached,
        )
