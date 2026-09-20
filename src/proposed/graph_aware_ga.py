from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

from ..problem import Scenario, Solution, clip_positions
from ..metrics import (
    evaluate_positions,
    feasibility_key,
    per_uav_redundancy_scores,
    uncovered_target_indices,
)
from ..objectives import ObjectiveConfig, DEFAULT_OBJECTIVE
from ..graph_ops import articulation_points
from ..repair import repair_solution


@dataclass(frozen=True)
class GraphAwareConfig:
    use_articulation_awareness: bool = True
    use_redundancy_aware_mutation: bool = True

    use_connectivity_repair: bool = True
    use_collision_repair: bool = True
    use_coverage_guided_mutation: bool = True

    articulation_mutation_factor: float = 0.20
    articulation_sigma_factor: float = 0.35

    redundancy_mutation_boost: float = 1.25
    guided_mutation_probability: float = 0.40


class GraphAwareGA:
    """Graph-aware evolutionary mechanism for Checkpoint 02."""

    name = "graph_ga"

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 70,
        elite_size: int = 5,
        mutation_rate: float = 0.25,
        mutation_sigma: float = 65.0,
        tournament_size: int = 3,
        graph_config: GraphAwareConfig = GraphAwareConfig(),
        objective_config: ObjectiveConfig = DEFAULT_OBJECTIVE,
        name: str | None = None,
    ):
        self.population_size = population_size
        self.generations = generations
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.tournament_size = tournament_size
        self.graph_config = graph_config
        self.objective_config = objective_config

        if name is not None:
            self.name = name

    def _random_individual(self, scenario, rng):
        return np.column_stack([
            rng.uniform(0, scenario.width, scenario.n_uavs),
            rng.uniform(0, scenario.height, scenario.n_uavs),
        ])

    def _repair(self, positions, scenario):
        return repair_solution(
            positions,
            scenario,
            connectivity=self.graph_config.use_connectivity_repair,
            collisions=self.graph_config.use_collision_repair,
        )

    def _fitness(self, scenario, positions):
        return evaluate_positions(
            scenario,
            positions,
            self.objective_config,
        ).fitness

    def _tournament(self, population, fitnesses, rng):
        ids = rng.integers(
            0,
            len(population),
            size=self.tournament_size,
        )
        best = max(ids, key=lambda i: fitnesses[i])
        return population[best], float(fitnesses[best])

    def _crossover(self, a, a_score, b, b_score, scenario, rng):
        if a_score >= b_score:
            anchor, other = a, b
        else:
            anchor, other = b, a

        child = anchor.copy()

        if self.graph_config.use_articulation_awareness:
            protected = articulation_points(scenario, anchor)
        else:
            protected = set()

        for i in range(len(child)):
            if i in protected:
                continue

            if rng.random() < 0.5:
                child[i] = other[i]

        return child

    def _guided_step(self, scenario, positions, uav_idx, rng):
        uncovered = uncovered_target_indices(scenario, positions)
        if len(uncovered) == 0:
            return None

        weights = scenario.target_weights[uncovered].astype(float)
        total = float(np.sum(weights))

        if total > 0:
            probs = weights / total
            target_idx = rng.choice(uncovered, p=probs)
        else:
            target_idx = rng.choice(uncovered)

        target = scenario.targets[target_idx]
        vector = target - positions[uav_idx]
        norm = float(np.linalg.norm(vector))

        if norm <= 1e-12:
            return None

        direction = vector / norm
        step = abs(
            rng.normal(
                self.mutation_sigma,
                0.35 * self.mutation_sigma,
            )
        )
        return direction * step

    def _mutate(self, chromosome, scenario, rng):
        child = chromosome.copy()

        if self.graph_config.use_articulation_awareness:
            protected = articulation_points(scenario, child)
        else:
            protected = set()

        if self.graph_config.use_redundancy_aware_mutation:
            redundancy = per_uav_redundancy_scores(
                scenario,
                child,
            )
            max_red = max(float(np.max(redundancy)), 1e-12)
            redundancy_norm = redundancy / max_red
        else:
            redundancy_norm = np.zeros(len(child), dtype=float)

        for i in range(len(child)):
            rate = self.mutation_rate
            sigma = self.mutation_sigma

            if i in protected:
                rate *= self.graph_config.articulation_mutation_factor
                sigma *= self.graph_config.articulation_sigma_factor
            else:
                rate *= (
                    1.0
                    + self.graph_config.redundancy_mutation_boost
                    * redundancy_norm[i]
                )

            rate = min(rate, 0.95)

            if rng.random() >= rate:
                continue

            step = None

            if (
                self.graph_config.use_coverage_guided_mutation
                and i not in protected
                and rng.random()
                < self.graph_config.guided_mutation_probability
            ):
                step = self._guided_step(
                    scenario,
                    child,
                    i,
                    rng,
                )

            if step is None:
                step = rng.normal(0.0, sigma, size=2)

            child[i] += step

        return clip_positions(child, scenario)

    def solve(self, scenario: Scenario, seed: int = 0) -> tuple[Solution, float]:
        rng = np.random.default_rng(seed)
        start = time.perf_counter()

        population = []

        for _ in range(self.population_size):
            individual = self._random_individual(scenario, rng)

            if (
                self.graph_config.use_connectivity_repair
                or self.graph_config.use_collision_repair
            ):
                individual = self._repair(individual, scenario)

            population.append(individual)

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
                a, a_score = self._tournament(
                    population,
                    fitnesses,
                    rng,
                )
                b, b_score = self._tournament(
                    population,
                    fitnesses,
                    rng,
                )

                child = self._crossover(
                    a,
                    a_score,
                    b,
                    b_score,
                    scenario,
                    rng,
                )

                child = self._mutate(child, scenario, rng)

                if (
                    self.graph_config.use_connectivity_repair
                    or self.graph_config.use_collision_repair
                ):
                    child = self._repair(child, scenario)

                next_population.append(child)

            population = next_population

        evaluations = [
            evaluate_positions(
                scenario,
                ind,
                self.objective_config,
            )
            for ind in population
        ]

        best = max(
            range(len(population)),
            key=lambda i: feasibility_key(evaluations[i]),
        )

        runtime = time.perf_counter() - start

        return (
            Solution(
                population[best],
                algorithm=self.name,
                seed=seed,
            ),
            runtime,
        )
