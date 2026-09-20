from __future__ import annotations

import time
import numpy as np

from ..problem import Scenario, Solution, clip_positions
from ..metrics import evaluate_positions
from ..objectives import ObjectiveConfig, DEFAULT_OBJECTIVE


class VanillaGA:
    name = "ga"

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 70,
        elite_size: int = 5,
        mutation_rate: float = 0.25,
        mutation_sigma: float = 65.0,
        tournament_size: int = 3,
        objective_config: ObjectiveConfig = DEFAULT_OBJECTIVE,
    ):
        self.population_size = population_size
        self.generations = generations
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.tournament_size = tournament_size
        self.objective_config = objective_config

    def _random_individual(self, scenario, rng):
        return np.column_stack([
            rng.uniform(0, scenario.width, scenario.n_uavs),
            rng.uniform(0, scenario.height, scenario.n_uavs),
        ])

    def _fitness(self, scenario, chromosome):
        return evaluate_positions(
            scenario,
            chromosome,
            self.objective_config,
        ).fitness

    def _tournament(self, population, fitnesses, rng):
        ids = rng.integers(
            0,
            len(population),
            size=self.tournament_size,
        )
        best = max(ids, key=lambda i: fitnesses[i])
        return population[best]

    def _crossover(self, a, b, rng):
        mask = rng.random((len(a), 1)) < 0.5
        return np.where(mask, a, b).copy()

    def _mutate(self, x, scenario, rng):
        child = x.copy()

        mask = rng.random(len(child)) < self.mutation_rate
        noise = rng.normal(
            0.0,
            self.mutation_sigma,
            size=child.shape,
        )

        child[mask] += noise[mask]
        return clip_positions(child, scenario)

    def solve(self, scenario: Scenario, seed: int = 0) -> tuple[Solution, float]:
        rng = np.random.default_rng(seed)
        start = time.perf_counter()

        population = [
            self._random_individual(scenario, rng)
            for _ in range(self.population_size)
        ]

        for _ in range(self.generations):
            fitnesses = [
                self._fitness(scenario, ind)
                for ind in population
            ]

            order = np.argsort(fitnesses)[::-1]
            next_population = [
                population[i].copy()
                for i in order[:self.elite_size]
            ]

            while len(next_population) < self.population_size:
                a = self._tournament(population, fitnesses, rng)
                b = self._tournament(population, fitnesses, rng)

                child = self._crossover(a, b, rng)
                child = self._mutate(child, scenario, rng)
                next_population.append(child)

            population = next_population

        fitnesses = [
            self._fitness(scenario, ind)
            for ind in population
        ]
        best = int(np.argmax(fitnesses))

        runtime = time.perf_counter() - start

        return (
            Solution(
                population[best],
                algorithm=self.name,
                seed=seed,
            ),
            runtime,
        )
