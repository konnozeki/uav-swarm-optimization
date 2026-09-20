from __future__ import annotations

import time
import numpy as np

from ..problem import Scenario, Solution
from ..metrics import evaluate_positions
from ..objectives import ObjectiveConfig, DEFAULT_OBJECTIVE


class RandomSearch:
    name = "random"

    def __init__(
        self,
        n_samples: int = 1200,
        objective_config: ObjectiveConfig = DEFAULT_OBJECTIVE,
    ):
        self.n_samples = n_samples
        self.objective_config = objective_config

    def solve(self, scenario: Scenario, seed: int = 0) -> tuple[Solution, float]:
        rng = np.random.default_rng(seed)
        start = time.perf_counter()

        best_positions = None
        best_fitness = -np.inf

        for _ in range(self.n_samples):
            positions = np.column_stack([
                rng.uniform(0, scenario.width, scenario.n_uavs),
                rng.uniform(0, scenario.height, scenario.n_uavs),
            ])

            fitness = evaluate_positions(
                scenario,
                positions,
                self.objective_config,
            ).fitness

            if fitness > best_fitness:
                best_fitness = fitness
                best_positions = positions.copy()

        runtime = time.perf_counter() - start

        return (
            Solution(best_positions, algorithm=self.name, seed=seed),
            runtime,
        )
