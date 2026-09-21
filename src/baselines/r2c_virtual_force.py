from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..metrics import evaluate_positions, feasibility_key
from ..problem import Scenario, Solution, clip_positions
from ..repair import repair_solution
from ..graph_ops import pairwise_distances


@dataclass(frozen=True)
class R2CBufferedForceConfig:
    """Parameters for the published R2C intra-subnetwork expansion mechanism."""

    iterations: int = 120
    lower_ratio: float = 0.80
    upper_ratio: float = 0.90
    alpha: float = 0.01
    beta: float = 0.01
    step_fraction: float = 0.035


class R2CBufferedVirtualForce:
    """R2C-ISE literature-component baseline from Peng et al., AAAI 2026.

    This implements the paper's buffered dynamic virtual-force expansion:
    repulsive below d_l, zero in [d_l, d_u], attractive above d_u and within
    communication range. It intentionally does NOT claim to reproduce full R2C,
    whose inter-subnetwork module is a trained multipartite GCN designed for
    post-failure recovery.

    For compatibility with this project's static target-coverage problem, the
    connected swarm is initialized around the weighted target centroid; after
    that, the motion law is the published force mechanism plus hard feasibility
    projection only.
    """

    name = "r2c_ise_aaai26"

    def __init__(
        self,
        config: R2CBufferedForceConfig = R2CBufferedForceConfig(),
    ) -> None:
        self.config = config

    def _initial_positions(
        self,
        scenario: Scenario,
        rng: np.random.Generator,
    ) -> np.ndarray:
        weights = scenario.target_weights.astype(float)
        if float(np.sum(weights)) > 0:
            center = np.average(
                scenario.targets,
                axis=0,
                weights=weights,
            )
        else:
            center = np.mean(scenario.targets, axis=0)

        n = scenario.n_uavs
        phase = rng.uniform(0.0, 2.0 * np.pi)
        angles = phase + 2.0 * np.pi * np.arange(n) / n

        # Start well inside the communication radius; R2C-ISE then expands into
        # its [0.8 Rc, 0.9 Rc] buffered neutral region.
        radius = min(
            0.30 * scenario.communication_radius,
            0.16 * min(scenario.width, scenario.height),
        )
        positions = center + np.column_stack([
            np.cos(angles),
            np.sin(angles),
        ]) * radius
        positions += rng.normal(
            0.0,
            0.015 * scenario.communication_radius,
            size=positions.shape,
        )

        positions = clip_positions(positions, scenario)
        return repair_solution(
            positions,
            scenario,
            connectivity=True,
            collisions=True,
            max_rounds=10,
        )

    def _force(
        self,
        scenario: Scenario,
        positions: np.ndarray,
    ) -> np.ndarray:
        n = len(positions)
        out = np.zeros_like(positions)
        distances = pairwise_distances(positions, positions)

        rc = scenario.communication_radius
        dl = self.config.lower_ratio * rc
        du = self.config.upper_ratio * rc

        for i in range(n):
            for j in range(i + 1, n):
                distance = float(distances[i, j])

                if distance <= 1e-12 or distance > rc:
                    continue

                unit_ij = (
                    positions[j] - positions[i]
                ) / distance

                if distance < dl:
                    magnitude = -np.exp(
                        -self.config.alpha
                        * distance
                        / max(dl, 1e-12)
                    )
                elif distance <= du:
                    magnitude = 0.0
                else:
                    magnitude = np.exp(
                        -self.config.beta
                        * (rc - distance)
                        / max(rc - du, 1e-12)
                    )

                force = magnitude * unit_ij
                out[i] += force
                out[j] -= force

        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return np.divide(
            out,
            np.maximum(norm, 1e-12),
            out=np.zeros_like(out),
            where=norm > 1e-12,
        )

    def solve(
        self,
        scenario: Scenario,
        seed: int = 0,
    ) -> tuple[Solution, float]:
        rng = np.random.default_rng(seed)
        started = time.perf_counter()

        positions = self._initial_positions(
            scenario,
            rng,
        )
        best = positions.copy()
        best_metrics = evaluate_positions(scenario, best)

        for _ in range(self.config.iterations):
            direction = self._force(
                scenario,
                positions,
            )

            if not np.any(np.linalg.norm(direction, axis=1) > 1e-12):
                break

            candidate = (
                positions
                + self.config.step_fraction
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
