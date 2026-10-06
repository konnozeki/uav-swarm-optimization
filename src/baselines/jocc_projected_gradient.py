from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..metrics import evaluate_positions, feasibility_key
from ..problem import Scenario, Solution, clip_positions
from ..repair import repair_solution
from ..graph_ops import pairwise_distances


@dataclass(frozen=True)
class JOCCGradientConfig:
    """Adapter parameters for the 2026 JOCC projected-gradient baselines.

    These implementations preserve the published algorithmic structure:
    coverage responsibility partitioning + differentiable soft communication
    graph + projected gradient updates. The repository's target-point sensing
    model replaces the paper's continuous ground-area/camera model, so these
    classes are literature adapters rather than bit-for-bit reproductions.
    """

    iterations: int = 80
    step_fraction: float = 0.06
    coverage_weight: float = 1.0
    connectivity_weight: float = 0.65
    separation_weight: float = 0.45
    responsibility_sigma_factor: float = 0.85
    soft_graph_sigma_factor: float = 0.60
    min_step_fraction: float = 0.015


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(
        vectors,
        np.maximum(norm, 1e-12),
        out=np.zeros_like(vectors),
        where=norm > 1e-12,
    )


def _target_weight_vector(scenario: Scenario) -> np.ndarray:
    weights = scenario.target_weights.astype(float)
    if float(np.sum(weights)) <= 0:
        return np.ones(len(weights), dtype=float)
    return weights


def _seeded_connected_initialization(
    scenario: Scenario,
    rng: np.random.Generator,
) -> np.ndarray:
    """Target-aware randomized start followed by hard feasibility repair."""
    n = scenario.n_uavs
    m = len(scenario.targets)
    weights = _target_weight_vector(scenario)
    probabilities = weights / float(np.sum(weights))

    if m >= n:
        ids = rng.choice(
            m,
            size=n,
            replace=False,
            p=probabilities,
        )
        positions = scenario.targets[ids].copy()
        positions += rng.normal(
            0.0,
            0.08 * scenario.sensing_radius,
            size=positions.shape,
        )
    else:
        total = float(np.sum(weights))
        center = (
            np.average(scenario.targets, axis=0, weights=weights)
            if total > 0
            else np.mean(scenario.targets, axis=0)
        )
        angles = (
            rng.uniform(0.0, 2.0 * np.pi)
            + 2.0 * np.pi * np.arange(n) / n
        )
        radius = min(
            0.38 * scenario.communication_radius,
            0.18 * min(scenario.width, scenario.height),
        )
        positions = center + np.column_stack([
            np.cos(angles),
            np.sin(angles),
        ]) * radius

    positions = clip_positions(positions, scenario)
    return repair_solution(
        positions,
        scenario,
        connectivity=True,
        collisions=True,
        max_rounds=10,
    )


def _soft_responsibility_force(
    scenario: Scenario,
    positions: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Soft Voronoi/responsibility coverage direction using all UAV states."""
    distances = pairwise_distances(
        scenario.targets,
        positions,
    )
    logits = -0.5 * (distances / max(sigma, 1e-12)) ** 2
    logits -= np.max(logits, axis=1, keepdims=True)
    responsibility = np.exp(logits)
    responsibility /= np.maximum(
        np.sum(responsibility, axis=1, keepdims=True),
        1e-12,
    )

    target_weights = _target_weight_vector(scenario)
    effective = target_weights[:, None] * responsibility
    mass = np.sum(effective, axis=0)

    centroids = positions.copy()
    active = mass > 1e-12
    if np.any(active):
        centroids[active] = (
            effective[:, active].T @ scenario.targets
        ) / mass[active, None]

    return centroids - positions


def _centralized_spectral_gradient(
    positions: np.ndarray,
    communication_radius: float,
    sigma: float,
) -> np.ndarray:
    """Gradient of soft-graph algebraic connectivity lambda_2.

    For Gaussian edge weight w_ij, d lambda_2 / d w_ij equals the
    squared Fiedler-vector difference. Chaining that derivative through
    w_ij(position) yields a smooth connectivity ascent direction.
    """
    n = len(positions)
    if n <= 1:
        return np.zeros_like(positions)

    distances = pairwise_distances(positions, positions)
    sigma = max(float(sigma), 1e-12)

    weights = np.exp(
        -0.5 * (distances / sigma) ** 2
    )
    np.fill_diagonal(weights, 0.0)

    # Suppress extremely long virtual edges. The hard graph is still enforced
    # by projection/repair after the gradient step.
    weights[distances > 1.35 * communication_radius] = 0.0

    laplacian = np.diag(np.sum(weights, axis=1)) - weights

    try:
        _, eigenvectors = np.linalg.eigh(laplacian)
    except np.linalg.LinAlgError:
        return np.zeros_like(positions)

    fiedler = eigenvectors[:, 1]
    gradient = np.zeros_like(positions)

    for i in range(n):
        for j in range(i + 1, n):
            weight = float(weights[i, j])
            if weight <= 1e-15:
                continue

            delta = positions[i] - positions[j]
            coefficient = float(
                (fiedler[i] - fiedler[j]) ** 2
            )
            grad_i = (
                -coefficient
                * weight
                * delta
                / (sigma ** 2)
            )
            gradient[i] += grad_i
            gradient[j] -= grad_i

    return gradient


def _separation_force(
    scenario: Scenario,
    positions: np.ndarray,
) -> np.ndarray:
    """Smooth short-range repulsion to complement hard collision repair."""
    n = len(positions)
    out = np.zeros_like(positions)
    desired = max(
        1.25 * scenario.min_separation,
        0.12 * scenario.communication_radius,
    )

    for i in range(n):
        for j in range(i + 1, n):
            delta = positions[i] - positions[j]
            distance = float(np.linalg.norm(delta))

            if distance >= desired:
                continue

            if distance <= 1e-12:
                # Deterministic direction avoids seed-dependent NaNs.
                angle = 2.0 * np.pi * (i + 1) / max(n, 1)
                direction = np.array([
                    np.cos(angle),
                    np.sin(angle),
                ])
            else:
                direction = delta / distance

            strength = (desired - distance) / max(desired, 1e-12)
            out[i] += strength * direction
            out[j] -= strength * direction

    return out


class JOCCCentralizedProjectedGradient:
    """CPGS-style centralized JOCC baseline (Wang et al., Computer Networks 2026).

    Published structure retained:
      * joint coverage/connectivity deployment;
      * coverage responsibility partitioning;
      * differentiable soft communication graph;
      * algebraic-connectivity gradient;
      * projected updates under hard feasibility constraints.

    The original paper optimizes continuous ground-area coverage quality in 3D.
    This adapter uses the project's weighted 2D target-point coverage semantics.
    """

    name = "jocc_cpgs_2026"

    def __init__(
        self,
        config: JOCCGradientConfig = JOCCGradientConfig(),
    ) -> None:
        self.config = config

    def solve(
        self,
        scenario: Scenario,
        seed: int = 0,
    ) -> tuple[Solution, float]:
        rng = np.random.default_rng(seed)
        started = time.perf_counter()

        positions = _seeded_connected_initialization(
            scenario,
            rng,
        )
        best = positions.copy()
        best_metrics = evaluate_positions(scenario, best)

        responsibility_sigma = max(
            self.config.responsibility_sigma_factor
            * scenario.sensing_radius,
            1.0,
        )
        graph_sigma = max(
            self.config.soft_graph_sigma_factor
            * scenario.communication_radius,
            1.0,
        )

        for iteration in range(self.config.iterations):
            coverage = _soft_responsibility_force(
                scenario,
                positions,
                responsibility_sigma,
            )
            connectivity = _centralized_spectral_gradient(
                positions,
                scenario.communication_radius,
                graph_sigma,
            )
            separation = _separation_force(
                scenario,
                positions,
            )

            direction = (
                self.config.coverage_weight
                * _normalize_rows(coverage)
                + self.config.connectivity_weight
                * _normalize_rows(connectivity)
                + self.config.separation_weight
                * _normalize_rows(separation)
            )
            direction = _normalize_rows(direction)

            progress = (
                iteration
                / max(self.config.iterations - 1, 1)
            )
            step_fraction = (
                self.config.step_fraction * (1.0 - progress)
                + self.config.min_step_fraction * progress
            )
            candidate = (
                positions
                + step_fraction
                * scenario.communication_radius
                * direction
            )
            candidate = clip_positions(candidate, scenario)
            candidate = repair_solution(
                candidate,
                scenario,
                connectivity=True,
                collisions=True,
                max_rounds=8,
            )

            positions = candidate
            metrics = evaluate_positions(scenario, positions)

            if feasibility_key(metrics) > feasibility_key(best_metrics):
                best = positions.copy()
                best_metrics = metrics

        runtime = time.perf_counter() - started
        return (
            Solution(best, algorithm=self.name, seed=seed),
            runtime,
        )


class JOCCDistributedProjectedGradient:
    """DPGS-style distributed JOCC baseline (Wang et al., Computer Networks 2026).

    Each UAV computes its coverage responsibility against its current one-hop
    communication neighbors and applies only local cohesion/separation forces.
    Synchronous projection keeps the comparison deterministic and compatible
    with the repository's hard graph/collision constraints.
    """

    name = "jocc_dpgs_2026"

    def __init__(
        self,
        config: JOCCGradientConfig = JOCCGradientConfig(
            iterations=100,
            step_fraction=0.05,
            connectivity_weight=0.80,
        ),
    ) -> None:
        self.config = config

    def _local_coverage_force(
        self,
        scenario: Scenario,
        positions: np.ndarray,
    ) -> np.ndarray:
        n = len(positions)
        target_weights = _target_weight_vector(scenario)
        distances_to_targets = pairwise_distances(
            scenario.targets,
            positions,
        )
        pairwise = pairwise_distances(
            positions,
            positions,
        )
        out = np.zeros_like(positions)

        for i in range(n):
            neighbors = np.flatnonzero(
                (pairwise[i] <= scenario.communication_radius + 1e-9)
                & (np.arange(n) != i)
            )
            local_nodes = np.concatenate([
                np.array([i], dtype=int),
                neighbors.astype(int),
            ])

            local_distances = distances_to_targets[:, local_nodes]
            owner_local = np.argmin(local_distances, axis=1)
            owned = owner_local == 0

            if not np.any(owned):
                continue

            weights = target_weights[owned]
            mass = float(np.sum(weights))
            if mass <= 1e-12:
                continue

            centroid = np.average(
                scenario.targets[owned],
                axis=0,
                weights=weights,
            )
            out[i] = centroid - positions[i]

        return out

    def _local_connectivity_force(
        self,
        scenario: Scenario,
        positions: np.ndarray,
    ) -> np.ndarray:
        n = len(positions)
        pairwise = pairwise_distances(positions, positions)
        out = np.zeros_like(positions)

        neutral_low = max(
            1.35 * scenario.min_separation,
            0.45 * scenario.communication_radius,
        )
        neutral_high = 0.78 * scenario.communication_radius

        for i in range(n):
            for j in range(i + 1, n):
                distance = float(pairwise[i, j])

                if distance > scenario.communication_radius:
                    continue

                delta = positions[j] - positions[i]
                if distance <= 1e-12:
                    continue
                direction = delta / distance

                if distance < neutral_low:
                    strength = (
                        neutral_low - distance
                    ) / max(neutral_low, 1e-12)
                    force = -strength * direction
                elif distance > neutral_high:
                    strength = (
                        distance - neutral_high
                    ) / max(
                        scenario.communication_radius - neutral_high,
                        1e-12,
                    )
                    force = strength * direction
                else:
                    continue

                out[i] += force
                out[j] -= force

        return out

    def solve(
        self,
        scenario: Scenario,
        seed: int = 0,
    ) -> tuple[Solution, float]:
        rng = np.random.default_rng(seed)
        started = time.perf_counter()

        positions = _seeded_connected_initialization(
            scenario,
            rng,
        )
        best = positions.copy()
        best_metrics = evaluate_positions(scenario, best)

        for iteration in range(self.config.iterations):
            coverage = self._local_coverage_force(
                scenario,
                positions,
            )
            connectivity = self._local_connectivity_force(
                scenario,
                positions,
            )
            separation = _separation_force(
                scenario,
                positions,
            )

            direction = (
                self.config.coverage_weight
                * _normalize_rows(coverage)
                + self.config.connectivity_weight
                * _normalize_rows(connectivity)
                + self.config.separation_weight
                * _normalize_rows(separation)
            )
            direction = _normalize_rows(direction)

            progress = (
                iteration
                / max(self.config.iterations - 1, 1)
            )
            step_fraction = (
                self.config.step_fraction * (1.0 - progress)
                + self.config.min_step_fraction * progress
            )

            candidate = (
                positions
                + step_fraction
                * scenario.communication_radius
                * direction
            )
            candidate = clip_positions(candidate, scenario)
            candidate = repair_solution(
                candidate,
                scenario,
                connectivity=True,
                collisions=True,
                max_rounds=8,
            )

            positions = candidate
            metrics = evaluate_positions(scenario, positions)

            if feasibility_key(metrics) > feasibility_key(best_metrics):
                best = positions.copy()
                best_metrics = metrics

        runtime = time.perf_counter() - started
        return (
            Solution(best, algorithm=self.name, seed=seed),
            runtime,
        )
