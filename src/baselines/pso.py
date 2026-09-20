from __future__ import annotations

import time
import numpy as np

from ..problem import Scenario, Solution, clip_positions
from ..metrics import evaluate_positions
from ..objectives import ObjectiveConfig, DEFAULT_OBJECTIVE


class ParticleSwarm:
    name = "pso"

    def __init__(
        self,
        swarm_size: int = 45,
        iterations: int = 70,
        inertia: float = 0.72,
        cognitive: float = 1.45,
        social: float = 1.45,
        objective_config: ObjectiveConfig = DEFAULT_OBJECTIVE,
    ):
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.objective_config = objective_config

    def solve(self, scenario: Scenario, seed: int = 0) -> tuple[Solution, float]:
        rng = np.random.default_rng(seed)
        start = time.perf_counter()

        particles = np.stack([
            np.column_stack([
                rng.uniform(0, scenario.width, scenario.n_uavs),
                rng.uniform(0, scenario.height, scenario.n_uavs),
            ])
            for _ in range(self.swarm_size)
        ])

        velocity_scale = 0.08 * max(scenario.width, scenario.height)
        velocities = rng.normal(
            0.0,
            velocity_scale,
            size=particles.shape,
        )

        pbest = particles.copy()
        pbest_scores = np.array([
            evaluate_positions(
                scenario,
                p,
                self.objective_config,
            ).fitness
            for p in particles
        ])

        g_idx = int(np.argmax(pbest_scores))
        gbest = pbest[g_idx].copy()
        gbest_score = float(pbest_scores[g_idx])

        for _ in range(self.iterations):
            r1 = rng.random(particles.shape)
            r2 = rng.random(particles.shape)

            velocities = (
                self.inertia * velocities
                + self.cognitive * r1 * (pbest - particles)
                + self.social * r2 * (gbest[None, :, :] - particles)
            )

            particles = particles + velocities

            for i in range(self.swarm_size):
                particles[i] = clip_positions(particles[i], scenario)

            scores = np.array([
                evaluate_positions(
                    scenario,
                    p,
                    self.objective_config,
                ).fitness
                for p in particles
            ])

            improved = scores > pbest_scores
            pbest[improved] = particles[improved]
            pbest_scores[improved] = scores[improved]

            g_idx = int(np.argmax(pbest_scores))
            if pbest_scores[g_idx] > gbest_score:
                gbest_score = float(pbest_scores[g_idx])
                gbest = pbest[g_idx].copy()

        runtime = time.perf_counter() - start

        return (
            Solution(gbest, algorithm=self.name, seed=seed),
            runtime,
        )
