from __future__ import annotations

import time
import numpy as np

from ..problem import Scenario, Solution, clip_positions
from ..metrics import objective_vector, evaluate_positions
from ..objectives import ObjectiveConfig, DEFAULT_OBJECTIVE


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(objectives: np.ndarray):
    n = len(objectives)

    domination_count = np.zeros(n, dtype=int)
    dominated = [[] for _ in range(n)]
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue

            if dominates(objectives[p], objectives[q]):
                dominated[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                domination_count[p] += 1

        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []

        for p in fronts[i]:
            for q in dominated[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)

        if next_front:
            fronts.append(next_front)

        i += 1

    return fronts


def crowding_distance(front: list[int], objectives: np.ndarray) -> dict[int, float]:
    if not front:
        return {}

    distance = {i: 0.0 for i in front}

    if len(front) <= 2:
        for i in front:
            distance[i] = float("inf")
        return distance

    m = objectives.shape[1]

    for obj_idx in range(m):
        ordered = sorted(front, key=lambda i: objectives[i, obj_idx])

        distance[ordered[0]] = float("inf")
        distance[ordered[-1]] = float("inf")

        min_v = objectives[ordered[0], obj_idx]
        max_v = objectives[ordered[-1], obj_idx]

        if abs(max_v - min_v) <= 1e-12:
            continue

        for k in range(1, len(ordered) - 1):
            prev_v = objectives[ordered[k - 1], obj_idx]
            next_v = objectives[ordered[k + 1], obj_idx]
            distance[ordered[k]] += (next_v - prev_v) / (max_v - min_v)

    return distance


class NSGA2:
    name = "nsga2"

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 60,
        mutation_rate: float = 0.25,
        mutation_sigma: float = 65.0,
        reporting_objective: ObjectiveConfig = DEFAULT_OBJECTIVE,
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.reporting_objective = reporting_objective

    def _random_individual(self, scenario, rng):
        return np.column_stack([
            rng.uniform(0, scenario.width, scenario.n_uavs),
            rng.uniform(0, scenario.height, scenario.n_uavs),
        ])

    def _evaluate_population(self, scenario, population):
        return np.asarray([
            objective_vector(scenario, ind)
            for ind in population
        ])

    def _rank_and_crowding(self, objectives):
        fronts = fast_non_dominated_sort(objectives)

        rank = np.full(len(objectives), 10**9, dtype=int)
        crowd = np.zeros(len(objectives), dtype=float)

        for r, front in enumerate(fronts):
            cd = crowding_distance(front, objectives)
            for i in front:
                rank[i] = r
                crowd[i] = cd[i]

        return rank, crowd, fronts

    def _tournament(self, population, rank, crowd, rng):
        a, b = rng.integers(0, len(population), size=2)

        if rank[a] < rank[b]:
            return population[a]
        if rank[b] < rank[a]:
            return population[b]

        if crowd[a] >= crowd[b]:
            return population[a]
        return population[b]

    def _crossover(self, a, b, rng):
        alpha = rng.random((len(a), 1))
        child = alpha * a + (1.0 - alpha) * b
        return child

    def _mutate(self, child, scenario, rng):
        out = child.copy()
        mask = rng.random(len(out)) < self.mutation_rate
        out[mask] += rng.normal(
            0.0,
            self.mutation_sigma,
            size=(np.sum(mask), 2),
        )
        return clip_positions(out, scenario)

    def solve(self, scenario: Scenario, seed: int = 0) -> tuple[Solution, float]:
        rng = np.random.default_rng(seed)
        start = time.perf_counter()

        population = [
            self._random_individual(scenario, rng)
            for _ in range(self.population_size)
        ]

        for _ in range(self.generations):
            objectives = self._evaluate_population(scenario, population)
            rank, crowd, _ = self._rank_and_crowding(objectives)

            offspring = []

            while len(offspring) < self.population_size:
                a = self._tournament(population, rank, crowd, rng)
                b = self._tournament(population, rank, crowd, rng)

                child = self._crossover(a, b, rng)
                child = self._mutate(child, scenario, rng)
                offspring.append(child)

            combined = population + offspring
            combined_objectives = self._evaluate_population(
                scenario,
                combined,
            )

            _, _, fronts = self._rank_and_crowding(combined_objectives)

            next_population = []

            for front in fronts:
                if len(next_population) + len(front) <= self.population_size:
                    next_population.extend(
                        combined[i].copy()
                        for i in front
                    )
                else:
                    cd = crowding_distance(front, combined_objectives)
                    ordered = sorted(
                        front,
                        key=lambda i: cd[i],
                        reverse=True,
                    )
                    remaining = self.population_size - len(next_population)
                    next_population.extend(
                        combined[i].copy()
                        for i in ordered[:remaining]
                    )
                    break

            population = next_population

        # Chọn một nghiệm từ Pareto population bằng scalar fitness chung,
        # chỉ để benchmark chung với các thuật toán scalar.
        scores = [
            evaluate_positions(
                scenario,
                ind,
                self.reporting_objective,
            ).fitness
            for ind in population
        ]
        best = int(np.argmax(scores))

        runtime = time.perf_counter() - start

        return (
            Solution(
                population[best],
                algorithm=self.name,
                seed=seed,
            ),
            runtime,
        )
