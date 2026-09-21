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
    boundary_decay: float = 0.01
    edge_buffer_ratio: float = 0.10
    max_speed: float | None = None
    speed_fraction: float = 0.035
    dt: float = 1.0


class R2CBufferedVirtualForce:
    """R2C-ISE literature-component baseline from Peng et al., AAAI 2026.

    This implements the paper's buffered dynamic virtual-force expansion:
    Eq. (9) pairwise spring, Eq. (10) boundary repulsion, Eq. (11) total force
    and Eq. (12) normalized velocity. It intentionally does NOT claim to
    reproduce full R2C,
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

    def _pair_force(
        self,
        distance: float,
        unit_ij: np.ndarray,
        communication_radius: float,
    ) -> np.ndarray:
        """Exact R2C intra-subnetwork virtual spring from Eq. (9).

        The sign convention follows the paper: u_ij points from i to j,
        therefore the negative branch is repulsive and the positive branch is
        attractive.
        """
        rc = float(communication_radius)
        dl = self.config.lower_ratio * rc
        du = self.config.upper_ratio * rc

        if distance <= 1e-12 or distance > rc:
            return np.zeros(2, dtype=float)

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

        return magnitude * unit_ij

    def _boundary_force(
        self,
        scenario: Scenario,
        position: np.ndarray,
    ) -> np.ndarray:
        """R2C edge repulsion from Eq. (10).

        Every boundary within the edge buffer contributes an exponentially
        decaying inward normal force.
        """
        buffer_width = (
            self.config.edge_buffer_ratio
            * min(scenario.width, scenario.height)
        )
        if buffer_width <= 0:
            return np.zeros(2, dtype=float)

        x, y = map(float, position)
        terms = (
            (x, np.array([1.0, 0.0])),
            (scenario.width - x, np.array([-1.0, 0.0])),
            (y, np.array([0.0, 1.0])),
            (scenario.height - y, np.array([0.0, -1.0])),
        )

        out = np.zeros(2, dtype=float)
        for distance, inward_normal in terms:
            if distance <= buffer_width:
                out += np.exp(
                    -self.config.boundary_decay
                    * max(distance, 0.0)
                ) * inward_normal

        return out

    def _force(
        self,
        scenario: Scenario,
        positions: np.ndarray,
    ) -> np.ndarray:
        """Total ISE force, matching R2C Eqs. (9)-(11)."""
        n = len(positions)
        out = np.zeros_like(positions)
        distances = pairwise_distances(positions, positions)

        for i in range(n):
            for j in range(i + 1, n):
                distance = float(distances[i, j])
                if distance <= 1e-12:
                    continue

                unit_ij = (
                    positions[j] - positions[i]
                ) / distance
                force = self._pair_force(
                    distance,
                    unit_ij,
                    scenario.communication_radius,
                )
                out[i] += force
                out[j] -= force

        for i in range(n):
            out[i] += self._boundary_force(
                scenario,
                positions[i],
            )

        return out

    def _velocity(
        self,
        scenario: Scenario,
        positions: np.ndarray,
    ) -> np.ndarray:
        """R2C Eq. (12): capped velocity in the total-force direction."""
        force = self._force(scenario, positions)
        norm = np.linalg.norm(force, axis=1, keepdims=True)
        unit = np.divide(
            force,
            norm + 1e-12,
            out=np.zeros_like(force),
            where=norm > 1e-12,
        )

        # Eq. (12) uses the physical speed cap directly. The common CP3
        # Scenario has no speed field, so the adapter keeps the historical
        # scale-relative fallback unless max_speed is supplied explicitly.
        vmax = (
            float(self.config.max_speed)
            if self.config.max_speed is not None
            else (
                self.config.speed_fraction
                * scenario.communication_radius
            )
        )
        return vmax * unit

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
            velocity = self._velocity(
                scenario,
                positions,
            )

            if not np.any(np.linalg.norm(velocity, axis=1) > 1e-12):
                break

            candidate = (
                positions
                + self.config.dt * velocity
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
