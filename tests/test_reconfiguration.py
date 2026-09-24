import numpy as np

from src.baselines import StaticThenTransition
from src.metrics import soft_coverage_potential
from src.problem import Scenario
from src.experiments.transition_cases import ring_formation
from src.proposed import TransitionAwareGA
from src.reconfiguration import (
    ReconfigurationEvaluation,
    ReconfigurationProblem,
    evaluate_reconfiguration,
)
from src.proposed.backbone_transition import BackboneTransitionPlanner


def _problem(simple_scenario):
    start = ring_formation(
        (500.0, 500.0),
        simple_scenario.n_uavs,
        radius=90.0,
    )

    return ReconfigurationProblem(
        name="simple_reconfiguration",
        scenario=simple_scenario,
        start_positions=start,
        max_speed=60.0,
    )


def test_start_formation_itself_has_zero_transition_cost(simple_scenario):
    problem = _problem(simple_scenario)

    evaluation, transition = evaluate_reconfiguration(
        problem,
        problem.start_positions,
        BackboneTransitionPlanner(),
    )

    # The start formation is already the goal, so no motion is required.
    assert transition is not None
    assert evaluation.reached_goal
    assert evaluation.formation_time_sec == 0.0
    assert evaluation.total_travel_distance == 0.0


def test_static_then_transition_returns_joint_solution(simple_scenario):
    problem = _problem(simple_scenario)

    algorithm = StaticThenTransition(
        population_size=8,
        generations=4,
    )
    solution, _ = algorithm.solve(problem, seed=2)

    assert solution.final_positions.shape == (
        simple_scenario.n_uavs,
        2,
    )
    assert np.isfinite(solution.evaluation.joint_fitness)


def test_transition_aware_ga_returns_evaluable_solution(simple_scenario):
    problem = _problem(simple_scenario)

    algorithm = TransitionAwareGA(
        population_size=8,
        generations=4,
        elite_size=2,
        warm_start_fraction=0.5,
    )
    solution, _ = algorithm.solve(problem, seed=3)

    assert solution.final_positions.shape == (
        simple_scenario.n_uavs,
        2,
    )
    assert np.isfinite(solution.evaluation.joint_fitness)
    # Real-time search may use lower-bound ranking internally, but must never
    # return that estimate in place of a certified transition.
    assert solution.evaluation.transition_attempted
    assert solution.evaluation.transition_feasible
    assert solution.transition_solution is not None



def _evaluation(
    *,
    feasible=True,
    coverage=0.8,
    coverage_potential=0.85,
    final_fitness=0.75,
    normalized_time=0.4,
    normalized_travel=0.3,
    joint_fitness=0.6,
):
    """Small helper for deterministic CP3 selection-order tests."""
    return ReconfigurationEvaluation(
        final_fitness=final_fitness,
        weighted_coverage_ratio=coverage,
        coverage_potential=coverage_potential,
        redundancy_excess=0.0,
        static_feasible=feasible,
        static_constraint_violation=0.0 if feasible else 1.0,
        final_obstacle_free=feasible,
        transition_attempted=True,
        transition_feasible=feasible,
        reached_goal=feasible,
        formation_time_sec=10.0,
        total_travel_distance=100.0,
        time_efficiency=1.0,
        continuous_connected_rate=1.0 if feasible else 0.0,
        min_continuous_pair_distance=100.0,
        transition_obstacle_free=feasible,
        normalized_time=normalized_time,
        normalized_travel=normalized_travel,
        joint_fitness=joint_fitness,
        feasible=feasible,
    )


def test_transition_aware_default_disables_articulation_protection():
    algorithm = TransitionAwareGA(
        population_size=4,
        generations=0,
    )

    assert not algorithm.operator.graph_config.use_articulation_awareness


def test_coverage_first_prefers_more_coverage_over_cheaper_motion():
    algorithm = TransitionAwareGA(
        population_size=4,
        generations=0,
        selection_mode="coverage_first",
    )

    more_coverage = _evaluation(
        coverage=0.90,
        normalized_time=0.90,
        normalized_travel=0.90,
        joint_fitness=0.50,
    )
    cheaper_motion = _evaluation(
        coverage=0.89,
        normalized_time=0.10,
        normalized_travel=0.10,
        joint_fitness=0.80,
    )

    assert (
        algorithm._selection_key(more_coverage)
        > algorithm._selection_key(cheaper_motion)
    )


def test_coverage_first_uses_time_only_after_sensing_quality_ties():
    algorithm = TransitionAwareGA(
        population_size=4,
        generations=0,
        selection_mode="coverage_first",
    )

    faster = _evaluation(
        coverage=0.90,
        final_fitness=0.86,
        normalized_time=0.20,
        normalized_travel=0.50,
    )
    slower = _evaluation(
        coverage=0.90,
        final_fitness=0.86,
        normalized_time=0.40,
        normalized_travel=0.20,
    )

    assert (
        algorithm._selection_key(faster)
        > algorithm._selection_key(slower)
    )


def test_coverage_first_allows_cost_to_break_near_performance_ties():
    algorithm = TransitionAwareGA(
        population_size=4,
        generations=0,
        selection_mode="coverage_first",
        performance_tolerance=0.005,
    )

    cheaper_near_tie = _evaluation(
        coverage=0.901,
        final_fitness=0.861,
        normalized_time=0.20,
        normalized_travel=0.20,
    )
    costlier_near_tie = _evaluation(
        coverage=0.904,
        final_fitness=0.864,
        normalized_time=0.60,
        normalized_travel=0.60,
    )

    assert (
        algorithm._selection_key(cheaper_near_tie)
        > algorithm._selection_key(costlier_near_tie)
    )


def test_weighted_joint_mode_reproduces_scalar_tradeoff_order():
    algorithm = TransitionAwareGA(
        population_size=4,
        generations=0,
        selection_mode="weighted_joint",
    )

    high_joint = _evaluation(
        coverage=0.80,
        joint_fitness=0.90,
    )
    low_joint = _evaluation(
        coverage=0.95,
        joint_fitness=0.70,
    )

    assert (
        algorithm._selection_key(high_joint)
        > algorithm._selection_key(low_joint)
    )



def test_soft_coverage_potential_rewards_progress_before_binary_coverage():
    scenario = Scenario(
        name="potential_plateau",
        pattern="unit",
        width=1000.0,
        height=1000.0,
        targets=np.array([[500.0, 500.0]]),
        target_weights=np.ones(1),
        n_uavs=1,
        sensing_radius=100.0,
        communication_radius=200.0,
        min_separation=0.0,
        seed=1,
    )

    far = np.array([[100.0, 500.0]])
    closer = np.array([[350.0, 500.0]])

    # Both are still outside Rs=100, so binary coverage is unchanged. The
    # smooth signal must nevertheless recognize that the second is much closer.
    assert (
        soft_coverage_potential(scenario, closer)
        > soft_coverage_potential(scenario, far)
    )


def test_search_uses_potential_but_final_selection_does_not():
    algorithm = TransitionAwareGA(
        population_size=4,
        generations=0,
        selection_mode="coverage_first",
    )

    closer_but_costlier = _evaluation(
        coverage=0.80,
        coverage_potential=0.90,
        final_fitness=0.75,
        normalized_time=0.60,
        normalized_travel=0.60,
    )
    farther_but_cheaper = _evaluation(
        coverage=0.80,
        coverage_potential=0.82,
        final_fitness=0.75,
        normalized_time=0.30,
        normalized_travel=0.30,
    )

    # During search, proximity can keep an "almost there" offspring alive.
    assert (
        algorithm._search_key(closer_but_costlier)
        > algorithm._search_key(farther_but_cheaper)
    )

    # Final output must not pay extra motion merely for proximity.
    assert (
        algorithm._selection_key(farther_but_cheaper)
        > algorithm._selection_key(closer_but_costlier)
    )
