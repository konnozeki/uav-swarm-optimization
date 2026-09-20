from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveConfig:
    redundancy_weight: float = 0.10
    connectivity_weight: float = 2.00
    collision_weight: float = 2.00


DEFAULT_OBJECTIVE = ObjectiveConfig()


def scalar_fitness(
    weighted_coverage: float,
    redundancy_excess: float,
    connectivity_deficit: float,
    collision_ratio: float,
    config: ObjectiveConfig = DEFAULT_OBJECTIVE,
) -> float:
    return float(
        weighted_coverage
        - config.redundancy_weight * redundancy_excess
        - config.connectivity_weight * connectivity_deficit
        - config.collision_weight * collision_ratio
    )


def minimization_objectives(
    weighted_coverage: float,
    redundancy_excess: float,
    connectivity_deficit: float,
    collision_ratio: float,
) -> tuple[float, float, float, float]:
    """
    Vector dùng cho NSGA-II. Tất cả đều ở dạng MINIMIZE.
    """
    return (
        -float(weighted_coverage),
        float(redundancy_excess),
        float(connectivity_deficit),
        float(collision_ratio),
    )
