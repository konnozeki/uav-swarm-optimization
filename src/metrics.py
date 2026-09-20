from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np

from .problem import Scenario, Solution
from .objectives import (
    ObjectiveConfig,
    DEFAULT_OBJECTIVE,
    scalar_fitness,
    minimization_objectives,
)
from .graph_ops import pairwise_distances, connectivity_deficit


FEASIBILITY_TOL = 1e-9


@dataclass
class Metrics:
    coverage_ratio: float
    weighted_coverage_ratio: float
    redundancy_excess: float
    connected: bool
    n_components: int
    connectivity_deficit: float
    collision_violations: int
    collision_ratio: float
    min_uav_distance: float
    complete: bool
    in_bounds: bool
    feasible: bool
    constraint_violation: float
    fitness: float

    def to_dict(self) -> dict:
        return asdict(self)


def _validated_positions(
    scenario: Scenario,
    positions: np.ndarray,
) -> np.ndarray:
    """Validate a complete or partial placement.

    Partial placements are intentionally allowed because ConnectedGreedy evaluates
    an intermediate swarm while constructing the final solution.
    """
    p = np.asarray(positions, dtype=float)

    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"positions must have shape (n, 2), got {p.shape}")

    if len(p) == 0 or len(p) > scenario.n_uavs:
        raise ValueError(
            "positions must contain between 1 and "
            f"{scenario.n_uavs} UAVs, got {len(p)}"
        )

    if not np.all(np.isfinite(p)):
        raise ValueError("positions must contain only finite values")

    return p


def target_coverage_matrix(scenario: Scenario, positions: np.ndarray) -> np.ndarray:
    """shape (M, N), True if target m is inside UAV n's sensing radius."""
    p = _validated_positions(scenario, positions)
    d = pairwise_distances(scenario.targets, p)
    return d <= scenario.sensing_radius


def coverage_counts(scenario: Scenario, positions: np.ndarray) -> np.ndarray:
    return np.sum(target_coverage_matrix(scenario, positions), axis=1)


def per_uav_redundancy_scores(scenario: Scenario, positions: np.ndarray) -> np.ndarray:
    """Score is high when a UAV covers targets already covered by other UAVs."""
    p = _validated_positions(scenario, positions)
    matrix = target_coverage_matrix(scenario, p)
    counts = np.sum(matrix, axis=1)

    scores = np.zeros(len(p), dtype=float)
    total_weight = max(float(np.sum(scenario.target_weights)), 1e-12)

    for i in range(len(p)):
        redundant_targets = matrix[:, i] & (counts >= 2)
        scores[i] = np.sum(scenario.target_weights[redundant_targets]) / total_weight

    return scores


def uncovered_target_indices(scenario: Scenario, positions: np.ndarray) -> np.ndarray:
    counts = coverage_counts(scenario, positions)
    return np.flatnonzero(counts == 0)


def soft_coverage_potential(
    scenario: Scenario,
    positions: np.ndarray,
    decay_scale: float | None = None,
) -> float:
    """Smooth guidance score for targets that are not covered yet.

    Binary coverage gives exactly the same reward to a UAV that is 500 m and
    1 m outside the sensing radius. This potential keeps the research metric
    unchanged, but gives CP3 search a useful tie-break signal on that plateau.

    Covered targets contribute 1. For an uncovered target, contribution decays
    exponentially with its positive distance gap outside the sensing radius.
    """
    p = _validated_positions(scenario, positions)

    if len(scenario.targets) == 0:
        return 0.0

    if decay_scale is None:
        decay_scale = max(
            0.75 * scenario.sensing_radius,
            1.0,
        )
    if decay_scale <= 0:
        raise ValueError("decay_scale must be positive")

    distances = pairwise_distances(
        scenario.targets,
        p,
    )
    nearest = np.min(distances, axis=1)
    gap = np.maximum(
        nearest - scenario.sensing_radius,
        0.0,
    )
    potential = np.exp(-gap / decay_scale)

    weights = scenario.target_weights.astype(float)
    total_weight = float(np.sum(weights))

    if total_weight <= 0:
        return float(np.mean(potential))

    return float(
        np.sum(weights * potential)
        / total_weight
    )



def collision_stats(
    scenario: Scenario,
    positions: np.ndarray,
) -> tuple[int, float, float]:
    p = _validated_positions(scenario, positions)
    n = len(p)
    if n <= 1:
        return 0, math.inf, 0.0

    d = pairwise_distances(p, p)
    upper = d[np.triu_indices(n, k=1)]

    violations = int(
        np.sum(upper < scenario.min_separation - FEASIBILITY_TOL)
    )
    min_distance = float(np.min(upper))
    n_pairs = n * (n - 1) / 2
    collision_ratio = violations / n_pairs

    return violations, min_distance, float(collision_ratio)


def bounds_violation_ratio(
    scenario: Scenario,
    positions: np.ndarray,
) -> float:
    """Normalized amount by which UAV coordinates lie outside the map."""
    p = _validated_positions(scenario, positions)

    x = p[:, 0]
    y = p[:, 1]

    x_violation = (
        np.maximum(-x, 0.0) + np.maximum(x - scenario.width, 0.0)
    ) / scenario.width
    y_violation = (
        np.maximum(-y, 0.0) + np.maximum(y - scenario.height, 0.0)
    ) / scenario.height

    return float(np.mean(x_violation + y_violation))


def feasibility_key(metrics: Metrics) -> tuple[int, float, float]:
    """Comparison key for final-solution selection.

    Feasible solutions dominate infeasible ones. Among infeasible solutions, lower
    normalized constraint violation is preferred; scalar fitness breaks ties.
    """
    return (
        1 if metrics.feasible else 0,
        -float(metrics.constraint_violation),
        float(metrics.fitness),
    )


def evaluate_positions(
    scenario: Scenario,
    positions: np.ndarray,
    objective_config: ObjectiveConfig = DEFAULT_OBJECTIVE,
) -> Metrics:
    p = _validated_positions(scenario, positions)

    counts = coverage_counts(scenario, p)
    covered = counts >= 1

    coverage_ratio = float(np.mean(covered)) if len(counts) else 0.0

    total_weight = float(np.sum(scenario.target_weights))
    if total_weight <= 0:
        weighted_coverage_ratio = coverage_ratio
        redundancy_excess = float(np.mean(np.maximum(counts - 1, 0)))
    else:
        weighted_coverage_ratio = float(
            np.sum(scenario.target_weights * covered) / total_weight
        )
        redundancy_excess = float(
            np.sum(scenario.target_weights * np.maximum(counts - 1, 0))
            / total_weight
        )

    conn_deficit, n_components = connectivity_deficit(scenario, p)
    connected = n_components == 1

    collision_violations, min_distance, collision_ratio = collision_stats(
        scenario,
        p,
    )

    complete = len(p) == scenario.n_uavs
    bounds_violation = bounds_violation_ratio(scenario, p)
    in_bounds = bounds_violation <= FEASIBILITY_TOL

    feasible = (
        complete
        and in_bounds
        and connected
        and collision_violations == 0
    )

    missing_uav_ratio = (scenario.n_uavs - len(p)) / scenario.n_uavs
    constraint_violation = float(
        missing_uav_ratio
        + bounds_violation
        + conn_deficit
        + collision_ratio
    )

    fitness = scalar_fitness(
        weighted_coverage=weighted_coverage_ratio,
        redundancy_excess=redundancy_excess,
        connectivity_deficit=conn_deficit,
        collision_ratio=collision_ratio,
        config=objective_config,
    )

    return Metrics(
        coverage_ratio=coverage_ratio,
        weighted_coverage_ratio=weighted_coverage_ratio,
        redundancy_excess=redundancy_excess,
        connected=connected,
        n_components=n_components,
        connectivity_deficit=conn_deficit,
        collision_violations=collision_violations,
        collision_ratio=collision_ratio,
        min_uav_distance=min_distance,
        complete=complete,
        in_bounds=in_bounds,
        feasible=feasible,
        constraint_violation=constraint_violation,
        fitness=fitness,
    )


def evaluate(
    scenario: Scenario,
    solution: Solution,
    objective_config: ObjectiveConfig = DEFAULT_OBJECTIVE,
) -> Metrics:
    if len(solution.positions) != scenario.n_uavs:
        raise ValueError(
            "final solution must contain exactly "
            f"{scenario.n_uavs} UAVs, got {len(solution.positions)}"
        )

    return evaluate_positions(scenario, solution.positions, objective_config)


def objective_vector(
    scenario: Scenario,
    positions: np.ndarray,
) -> tuple[float, float, float, float]:
    m = evaluate_positions(scenario, positions)
    return minimization_objectives(
        weighted_coverage=m.weighted_coverage_ratio,
        redundancy_excess=m.redundancy_excess,
        connectivity_deficit=m.connectivity_deficit,
        collision_ratio=m.collision_ratio,
    )
