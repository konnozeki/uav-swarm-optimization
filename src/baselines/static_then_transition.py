from __future__ import annotations

import time

from ..objectives import ObjectiveConfig, DEFAULT_OBJECTIVE
from ..obstacles import project_positions_outside_obstacles
from ..proposed.graph_aware_ga import GraphAwareGA
from ..proposed.backbone_transition import BackboneTransitionPlanner
from ..reconfiguration import (
    DEFAULT_RECONFIGURATION_OBJECTIVE,
    ReconfigurationObjectiveConfig,
    ReconfigurationProblem,
    ReconfigurationSolution,
    evaluate_reconfiguration,
)


class StaticThenTransition:
    """Two-stage baseline: optimize formation B first, then plan A -> B.

    This is the direct continuation of Checkpoint 02. It has no transition cost
    inside evolutionary selection, so it can discover a high-quality final
    sensing formation that is unnecessarily slow or even impossible to reach
    from the current formation A.
    """

    name = "static_then_transition"

    def __init__(
        self,
        population_size: int = 24,
        generations: int = 28,
        static_objective_config: ObjectiveConfig = DEFAULT_OBJECTIVE,
        reconfiguration_objective_config: ReconfigurationObjectiveConfig = (
            DEFAULT_RECONFIGURATION_OBJECTIVE
        ),
    ) -> None:
        self.static_objective_config = static_objective_config
        self.reconfiguration_objective_config = (
            reconfiguration_objective_config
        )

        self.static_optimizer = GraphAwareGA(
            population_size=population_size,
            generations=generations,
            objective_config=static_objective_config,
        )
        self.transition_planner = BackboneTransitionPlanner()

    def solve(
        self,
        problem: ReconfigurationProblem,
        seed: int = 0,
    ) -> tuple[ReconfigurationSolution, float]:
        start_time = time.perf_counter()

        static_solution, _ = self.static_optimizer.solve(
            problem.scenario,
            seed=seed,
        )

        goal_positions = static_solution.positions

        # StaticThenTransition remains unaware of transition cost during search,
        # but environmental no-fly regions are hard constraints rather than an
        # optimization preference. Push the chosen endpoint formation outside
        # obstacles before asking the transition planner to reach it.
        if problem.obstacles:
            for _ in range(3):
                goal_positions = project_positions_outside_obstacles(
                    goal_positions,
                    problem.obstacles,
                    problem.obstacle_clearance,
                    problem.scenario.width,
                    problem.scenario.height,
                )
                goal_positions = self.static_optimizer._repair(
                    goal_positions,
                    problem.scenario,
                )

            # End on the hard environmental constraint. The evaluation below
            # will still reject the result if this last push damages graph or
            # collision feasibility.
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
            static_objective_config=self.static_objective_config,
            objective_config=self.reconfiguration_objective_config,
        )

        runtime = time.perf_counter() - start_time

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
