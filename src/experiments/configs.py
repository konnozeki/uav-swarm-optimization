from __future__ import annotations

from ..scenarios import quick_profile, stress_profile, full_profile
from ..baselines import (
    RandomSearch,
    ConnectedGreedy,
    VanillaGA,
    ParticleSwarm,
    NSGA2,
)
from ..proposed import GraphAwareGA


def get_scenarios(profile: str):
    if profile == "quick":
        return quick_profile()
    if profile == "stress":
        return stress_profile()
    if profile == "full":
        return full_profile()
    raise ValueError(f"Unknown profile: {profile}")


def get_algorithms(profile: str = "quick"):
    if profile == "quick":
        return [
            RandomSearch(n_samples=500),
            ConnectedGreedy(grid_size=7),
            VanillaGA(population_size=30, generations=35),
            ParticleSwarm(swarm_size=30, iterations=35),
            NSGA2(population_size=30, generations=25),
            GraphAwareGA(population_size=30, generations=35),
        ]

    if profile == "stress":
        return [
            RandomSearch(n_samples=800),
            ConnectedGreedy(grid_size=8),
            VanillaGA(population_size=40, generations=50),
            ParticleSwarm(swarm_size=40, iterations=50),
            NSGA2(population_size=36, generations=40),
            GraphAwareGA(population_size=40, generations=50),
        ]

    if profile == "full":
        return [
            RandomSearch(n_samples=700),
            ConnectedGreedy(grid_size=8),
            VanillaGA(population_size=36, generations=45),
            ParticleSwarm(swarm_size=36, iterations=45),
            NSGA2(population_size=32, generations=35),
            GraphAwareGA(population_size=36, generations=45),
        ]

    raise ValueError(f"Unknown profile: {profile}")
