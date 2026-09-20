from __future__ import annotations

import time
import numpy as np

from ..problem import Scenario, Solution
from ..metrics import evaluate_positions
from ..objectives import ObjectiveConfig, DEFAULT_OBJECTIVE


class ConnectedGreedy:
    name = "greedy"

    def __init__(
        self,
        grid_size: int = 8,
        objective_config: ObjectiveConfig = DEFAULT_OBJECTIVE,
    ):
        self.grid_size = grid_size
        self.objective_config = objective_config

    def _candidate_points(self, scenario: Scenario) -> np.ndarray:
        xs = np.linspace(0, scenario.width, self.grid_size)
        ys = np.linspace(0, scenario.height, self.grid_size)
        grid = np.array([(x, y) for x in xs for y in ys], dtype=float)

        total_w = float(np.sum(scenario.target_weights))
        if total_w > 0:
            centroid = np.average(
                scenario.targets,
                axis=0,
                weights=scenario.target_weights,
            )
        else:
            centroid = np.mean(scenario.targets, axis=0)

        return np.vstack([
            scenario.targets,
            grid,
            centroid[None, :],
        ])

    def solve(self, scenario: Scenario, seed: int = 0) -> tuple[Solution, float]:
        # The algorithm is deterministic; seed is kept for a common benchmark API.
        _ = seed
        start = time.perf_counter()

        candidates = self._candidate_points(scenario)

        total_w = float(np.sum(scenario.target_weights))
        if total_w > 0:
            first = np.average(
                scenario.targets,
                axis=0,
                weights=scenario.target_weights,
            )
        else:
            first = np.mean(scenario.targets, axis=0)

        positions = [np.asarray(first, dtype=float)]

        while len(positions) < scenario.n_uavs:
            current = np.asarray(positions)

            d = np.linalg.norm(
                candidates[:, None, :] - current[None, :, :],
                axis=2,
            )

            safe = np.all(
                d >= scenario.min_separation - 1e-9,
                axis=1,
            )
            connected_candidate = np.any(
                d <= scenario.communication_radius + 1e-9,
                axis=1,
            )

            valid = candidates[safe & connected_candidate]

            if len(valid) == 0:
                # Ring fallback around the latest UAV.
                base = positions[-1]
                angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
                radius = 0.8 * scenario.communication_radius

                valid = base + np.column_stack([
                    radius * np.cos(angles),
                    radius * np.sin(angles),
                ])
                valid[:, 0] = np.clip(valid[:, 0], 0, scenario.width)
                valid[:, 1] = np.clip(valid[:, 1], 0, scenario.height)

                # Clipping can collapse candidates onto existing UAVs, so all
                # constraints must be recomputed after clipping.
                fallback_d = np.linalg.norm(
                    valid[:, None, :] - current[None, :, :],
                    axis=2,
                )
                fallback_safe = np.all(
                    fallback_d >= scenario.min_separation - 1e-9,
                    axis=1,
                )
                fallback_connected = np.any(
                    fallback_d <= scenario.communication_radius + 1e-9,
                    axis=1,
                )
                valid = valid[fallback_safe & fallback_connected]

            if len(valid) == 0:
                raise RuntimeError(
                    "ConnectedGreedy could not construct another "
                    "collision-free connected UAV position"
                )

            best_candidate = None
            best_fitness = -np.inf

            for candidate in valid:
                trial = np.vstack([current, candidate[None, :]])
                fitness = evaluate_positions(
                    scenario,
                    trial,
                    self.objective_config,
                ).fitness

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_candidate = candidate

            positions.append(best_candidate.copy())

        runtime = time.perf_counter() - start

        return (
            Solution(
                np.asarray(positions),
                algorithm=self.name,
                seed=seed,
            ),
            runtime,
        )
