from __future__ import annotations

import time

from ..obstacles import project_positions_outside_obstacles
from ..repair import repair_solution
from ..proposed.backbone_transition import BackboneTransitionPlanner
from ..reconfiguration import (
    DEFAULT_RECONFIGURATION_OBJECTIVE,
    ReconfigurationObjectiveConfig,
    ReconfigurationProblem,
    ReconfigurationSolution,
    evaluate_reconfiguration,
)


class LiteratureStaticThenTransition:
    """Run a literature/domain static deployment baseline, then CP3 transition.

    The same BackboneTransitionPlanner is used for every static baseline so the
    comparison isolates destination-formation quality rather than giving each
    method a different lower-level motion planner.
    """

    def __init__(
        self,
        static_optimizer,
        *,
        reconfiguration_objective_config: ReconfigurationObjectiveConfig = (
            DEFAULT_RECONFIGURATION_OBJECTIVE
        ),
    ) -> None:
        self.static_optimizer = static_optimizer
        self.reconfiguration_objective_config = (
            reconfiguration_objective_config
        )
        self.transition_planner = BackboneTransitionPlanner()
        self.name = f"{static_optimizer.name}_then_transition"

    def solve(
        self,
        problem: ReconfigurationProblem,
        seed: int = 0,
    ) -> tuple[ReconfigurationSolution, float]:
        started = time.perf_counter()

        static_solution, _ = self.static_optimizer.solve(
            problem.scenario,
            seed=seed,
        )
        goal_positions = static_solution.positions.copy()

        if problem.obstacles:
            for _ in range(3):
                goal_positions = project_positions_outside_obstacles(
                    goal_positions,
                    problem.obstacles,
                    problem.obstacle_clearance,
                    problem.scenario.width,
                    problem.scenario.height,
                )
                goal_positions = repair_solution(
                    goal_positions,
                    problem.scenario,
                    connectivity=True,
                    collisions=True,
                    max_rounds=8,
                )

            goal_positions = project_positions_outside_obstacles(
                goal_positions,
                problem.obstacles,
                problem.obstacle_clearance,
                problem.scenario.width,
                problem.scenario.height,
            )

        evaluation, transition_solution = evaluate_reconfiguration(
            problem,
            goal_positions,
            self.transition_planner,
            objective_config=self.reconfiguration_objective_config,
        )

        runtime = time.perf_counter() - started

        return (
            ReconfigurationSolution(
                final_positions=goal_positions,
                evaluation=evaluation,
                transition_solution=transition_solution,
                algorithm=self.name,
                seed=seed,
            ),
            runtime,
        )
