from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

DEFAULT_MIN_SEPARATION = 3.0  # Metres: hard collision clearance, not formation spacing.


@dataclass(frozen=True)
class Scenario:
    name: str
    pattern: str
    width: float
    height: float
    targets: np.ndarray
    target_weights: np.ndarray
    n_uavs: int
    sensing_radius: float
    communication_radius: float
    min_separation: float
    seed: int = 0

    def __post_init__(self) -> None:
        targets = np.asarray(self.targets, dtype=float)
        weights = np.asarray(self.target_weights, dtype=float)

        if targets.ndim != 2 or targets.shape[1] != 2:
            raise ValueError("targets must have shape (n_targets, 2)")
        if weights.ndim != 1 or len(weights) != len(targets):
            raise ValueError("target_weights must have shape (n_targets,)")
        if self.n_uavs <= 0:
            raise ValueError("n_uavs must be > 0")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("map dimensions must be > 0")
        if self.sensing_radius <= 0 or self.communication_radius <= 0:
            raise ValueError("radii must be > 0")
        if self.min_separation < 0:
            raise ValueError("min_separation must be >= 0")
        if (
            self.n_uavs > 1
            and self.min_separation > self.communication_radius
        ):
            raise ValueError(
                "min_separation cannot exceed communication_radius when "
                "n_uavs > 1: no collision-free connected graph can exist"
            )

        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "target_weights", weights)


@dataclass
class Solution:
    positions: np.ndarray
    algorithm: str = ""
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float)
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError("positions must have shape (n_uavs, 2)")


def clip_positions(positions: np.ndarray, scenario: Scenario) -> np.ndarray:
    p = np.asarray(positions, dtype=float).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, scenario.width)
    p[:, 1] = np.clip(p[:, 1], 0.0, scenario.height)
    return p
