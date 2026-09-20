from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .metrics import (
    evaluate_positions,
    soft_coverage_potential,
)
from .objectives import ObjectiveConfig, DEFAULT_OBJECTIVE
from .problem import Scenario
from .obstacles import (
    AxisAlignedRectangle,
    formation_obstacle_free,
    obstacle_violation_ratio,
    obstacles_within_bounds,
)
from .transition import (
    TransitionMetrics,
    TransitionProblem,
    TransitionSolution,
    evaluate_transition,
    formation_collision_free,
    formation_connected,
)


@dataclass(frozen=True)
class ReconfigurationProblem:
    """Joint sensing-formation and transition problem.

    Formation A is the current swarm formation. Formation B is *not* given:
    an optimizer must choose B so that it is a good sensing formation and is
    also cheap/fast to reach from A.

    This is the second stage of Checkpoint 03. The lower-level transition engine
    remains reusable on its own for fixed A -> B experiments.
    """

    name: str
    scenario: Scenario
    start_positions: np.ndarray
    max_speed: float
    dt: float = 1.0
    allow_reassignment: bool = True
    obstacles: tuple[AxisAlignedRectangle, ...] = ()
    obstacle_clearance: float = 0.0

    def __post_init__(self) -> None:
        start = np.asarray(self.start_positions, dtype=float)

        if start.shape != (self.scenario.n_uavs, 2):
            raise ValueError(
                "start_positions must have shape "
                f"({self.scenario.n_uavs}, 2)"
            )
        if not np.all(np.isfinite(start)):
            raise ValueError("start_positions must contain finite coordinates")
        if self.max_speed <= 0 or self.dt <= 0:
            raise ValueError("max_speed and dt must be positive")
        if self.obstacle_clearance < 0:
            raise ValueError("obstacle_clearance must be non-negative")

        obstacles = tuple(self.obstacles)

        if not obstacles_within_bounds(
            obstacles,
            self.scenario.width,
            self.scenario.height,
        ):
            raise ValueError("all obstacles must lie inside map bounds")

        if np.any(start[:, 0] < 0) or np.any(start[:, 0] > self.scenario.width):
            raise ValueError("start formation leaves horizontal map bounds")
        if np.any(start[:, 1] < 0) or np.any(start[:, 1] > self.scenario.height):
            raise ValueError("start formation leaves vertical map bounds")

        if not formation_connected(
            start,
            self.scenario.communication_radius,
        ):
            raise ValueError("start formation must be connected")

        if not formation_collision_free(
            start,
            self.scenario.min_separation,
        ):
            raise ValueError("start formation must be collision-free")

        if not formation_obstacle_free(
            start,
            obstacles,
            self.obstacle_clearance,
        ):
            raise ValueError(
                "start formation must not intersect a no-fly obstacle"
            )

        object.__setattr__(self, "start_positions", start)
        object.__setattr__(self, "obstacles", obstacles)

    @property
    def reference_distance(self) -> float:
        """Map diagonal used to normalize time and travel costs."""
        return float(
            math.hypot(
                self.scenario.width,
                self.scenario.height,
            )
        )

    @property
    def reference_time(self) -> float:
        """Time required to cross one map diagonal at maximum speed."""
        return self.reference_distance / self.max_speed

    def transition_problem(
        self,
        goal_positions: np.ndarray,
    ) -> TransitionProblem:
        """Create the lower-level fixed A -> B transition problem."""
        return TransitionProblem(
            name=f"{self.name}_transition",
            width=self.scenario.width,
            height=self.scenario.height,
            start_positions=self.start_positions,
            goal_positions=goal_positions,
            communication_radius=self.scenario.communication_radius,
            min_separation=self.scenario.min_separation,
            max_speed=self.max_speed,
            dt=self.dt,
            allow_reassignment=self.allow_reassignment,
            obstacles=self.obstacles,
            obstacle_clearance=self.obstacle_clearance,
        )


@dataclass(frozen=True)
class ReconfigurationObjectiveConfig:
    """Weights for the joint formation + transition objective.

    Static formation fitness is inherited unchanged from Checkpoint 02. CP3 only
    subtracts normalized transition costs. This separation keeps the source of
    improvement interpretable.
    """

    time_weight: float = 0.20
    travel_weight: float = 0.05
    transition_infeasible_penalty: float = 2.0


DEFAULT_RECONFIGURATION_OBJECTIVE = ReconfigurationObjectiveConfig()


@dataclass(frozen=True)
class ReconfigurationEvaluation:
    """Flattened metrics for one candidate destination formation."""

    final_fitness: float
    weighted_coverage_ratio: float
    coverage_potential: float
    redundancy_excess: float
    static_feasible: bool
    static_constraint_violation: float
    final_obstacle_free: bool

    transition_attempted: bool
    transition_feasible: bool
    reached_goal: bool
    formation_time_sec: float
    total_travel_distance: float
    time_efficiency: float
    continuous_connected_rate: float
    min_continuous_pair_distance: float
    transition_obstacle_free: bool

    normalized_time: float
    normalized_travel: float
    joint_fitness: float
    feasible: bool

    def to_dict(self) -> dict:
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
        }


@dataclass
class ReconfigurationSolution:
    """Final output of a joint reconfiguration algorithm."""

    final_positions: np.ndarray
    evaluation: ReconfigurationEvaluation
    transition_solution: TransitionSolution | None
    algorithm: str
    seed: int | None = None

    def __post_init__(self) -> None:
        self.final_positions = np.asarray(
            self.final_positions,
            dtype=float,
        )


def _failed_transition_evaluation(
    static_metrics,
    objective_config: ReconfigurationObjectiveConfig,
    *,
    static_feasible: bool | None = None,
    additional_constraint_violation: float = 0.0,
    final_obstacle_free: bool = True,
    coverage_potential: float = 0.0,
) -> ReconfigurationEvaluation:
    """Create a deterministic score when transition simulation is impossible."""
    effective_static_feasible = (
        bool(static_metrics.feasible)
        if static_feasible is None
        else bool(static_feasible)
    )
    effective_violation = (
        float(static_metrics.constraint_violation)
        + float(additional_constraint_violation)
    )

    joint = (
        static_metrics.fitness
        - objective_config.transition_infeasible_penalty
        - effective_violation
    )

    return ReconfigurationEvaluation(
        final_fitness=float(static_metrics.fitness),
        weighted_coverage_ratio=float(
            static_metrics.weighted_coverage_ratio
        ),
        coverage_potential=float(coverage_potential),
        redundancy_excess=float(static_metrics.redundancy_excess),
        static_feasible=effective_static_feasible,
        static_constraint_violation=effective_violation,
        final_obstacle_free=bool(final_obstacle_free),
        transition_attempted=False,
        transition_feasible=False,
        reached_goal=False,
        formation_time_sec=float("nan"),
        total_travel_distance=float("nan"),
        time_efficiency=0.0,
        continuous_connected_rate=0.0,
        min_continuous_pair_distance=float("nan"),
        transition_obstacle_free=False,
        normalized_time=1.0,
        normalized_travel=1.0,
        joint_fitness=float(joint),
        feasible=False,
    )


def evaluate_reconfiguration(
    problem: ReconfigurationProblem,
    goal_positions: np.ndarray,
    transition_planner,
    static_objective_config: ObjectiveConfig = DEFAULT_OBJECTIVE,
    objective_config: ReconfigurationObjectiveConfig = (
        DEFAULT_RECONFIGURATION_OBJECTIVE
    ),
) -> tuple[ReconfigurationEvaluation, TransitionSolution | None]:
    """Evaluate a candidate formation B from the current formation A.

    Static infeasibility is rejected before path simulation. For a statically
    feasible B, the supplied transition planner is run exactly and contributes
    normalized formation time and travel distance to the joint fitness.
    """
    goal_positions = np.asarray(goal_positions, dtype=float)

    static_metrics = evaluate_positions(
        problem.scenario,
        goal_positions,
        static_objective_config,
    )

    coverage_potential = soft_coverage_potential(
        problem.scenario,
        goal_positions,
    )

    obstacle_violation = obstacle_violation_ratio(
        goal_positions,
        problem.obstacles,
        problem.obstacle_clearance,
    )
    final_obstacle_free = obstacle_violation <= 1e-12
    static_feasible = bool(
        static_metrics.feasible
        and final_obstacle_free
    )

    if not static_feasible:
        return (
            _failed_transition_evaluation(
                static_metrics,
                objective_config,
                static_feasible=False,
                additional_constraint_violation=obstacle_violation,
                final_obstacle_free=final_obstacle_free,
                coverage_potential=coverage_potential,
            ),
            None,
        )

    try:
        transition_problem = problem.transition_problem(goal_positions)
        transition_solution = transition_planner.solve(
            transition_problem
        )
        transition_metrics: TransitionMetrics = evaluate_transition(
            transition_problem,
            transition_solution,
        )
    except (ValueError, RuntimeError):
        # Search should treat planner failure as an infeasible candidate rather
        # than aborting the entire evolutionary run.
        failed = _failed_transition_evaluation(
            static_metrics,
            objective_config,
            coverage_potential=coverage_potential,
        )
        failed = ReconfigurationEvaluation(
            **{
                **failed.to_dict(),
                "transition_attempted": True,
            }
        )
        return failed, None

    reference_time = max(problem.reference_time, problem.dt)
    reference_travel = max(
        problem.scenario.n_uavs * problem.reference_distance,
        1e-12,
    )

    if transition_metrics.reached_goal:
        normalized_time = (
            transition_metrics.formation_time_sec
            / reference_time
        )
        normalized_travel = (
            transition_metrics.total_travel_distance
            / reference_travel
        )
    else:
        # Failed trajectories must not gain an artificial advantage from being
        # short simply because the planner stopped early.
        normalized_time = 1.0
        normalized_travel = 1.0

    penalty = (
        0.0
        if transition_metrics.feasible
        else objective_config.transition_infeasible_penalty
    )

    joint_fitness = (
        static_metrics.fitness
        - objective_config.time_weight * normalized_time
        - objective_config.travel_weight * normalized_travel
        - penalty
    )

    evaluation = ReconfigurationEvaluation(
        final_fitness=float(static_metrics.fitness),
        weighted_coverage_ratio=float(
            static_metrics.weighted_coverage_ratio
        ),
        coverage_potential=float(coverage_potential),
        redundancy_excess=float(static_metrics.redundancy_excess),
        static_feasible=bool(static_metrics.feasible),
        static_constraint_violation=float(
            static_metrics.constraint_violation
        ),
        final_obstacle_free=True,
        transition_attempted=True,
        transition_feasible=bool(transition_metrics.feasible),
        reached_goal=bool(transition_metrics.reached_goal),
        formation_time_sec=float(
            transition_metrics.formation_time_sec
        ),
        total_travel_distance=float(
            transition_metrics.total_travel_distance
        ),
        time_efficiency=float(transition_metrics.time_efficiency),
        continuous_connected_rate=float(
            transition_metrics.sampled_continuous_connected_rate
        ),
        min_continuous_pair_distance=float(
            transition_metrics.min_continuous_pair_distance
        ),
        transition_obstacle_free=bool(
            transition_metrics.obstacle_free
        ),
        normalized_time=float(normalized_time),
        normalized_travel=float(normalized_travel),
        joint_fitness=float(joint_fitness),
        feasible=bool(
            static_metrics.feasible
            and transition_metrics.feasible
        ),
    )

    return evaluation, transition_solution
