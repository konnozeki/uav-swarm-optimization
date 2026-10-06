from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

DEFAULT_MIN_SEPARATION = 3.0  # Metres: hard collision clearance, not formation spacing.
GEOMETRY_TOL = 1e-9


@dataclass(frozen=True)
class ForbiddenRegion:
    """Static no-deploy region used by obstacle-aware static variants.

    The region is deliberately geometric and lightweight. It constrains UAV
    positions, while coverage and communication keep their existing target-point
    semantics unless a dedicated obstacle experiment changes those models.
    """

    name: str
    kind: str
    bounds: tuple[float, float, float, float] | None = None
    center: tuple[float, float] | None = None
    radius: float | None = None

    @staticmethod
    def rectangle(
        name: str,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> "ForbiddenRegion":
        return ForbiddenRegion(
            name=name,
            kind="rectangle",
            bounds=(xmin, ymin, xmax, ymax),
        )

    @staticmethod
    def circle(
        name: str,
        x: float,
        y: float,
        radius: float,
    ) -> "ForbiddenRegion":
        return ForbiddenRegion(
            name=name,
            kind="circle",
            center=(x, y),
            radius=radius,
        )

    def __post_init__(self) -> None:
        if self.kind not in {"rectangle", "circle"}:
            raise ValueError("forbidden region kind must be rectangle or circle")

        if self.kind == "rectangle":
            if self.bounds is None:
                raise ValueError("rectangle forbidden region requires bounds")
            xmin, ymin, xmax, ymax = map(float, self.bounds)
            if xmax <= xmin or ymax <= ymin:
                raise ValueError("rectangle bounds must satisfy xmax > xmin and ymax > ymin")
            object.__setattr__(self, "bounds", (xmin, ymin, xmax, ymax))
            object.__setattr__(self, "center", None)
            object.__setattr__(self, "radius", None)
        else:
            if self.center is None or self.radius is None:
                raise ValueError("circle forbidden region requires center and radius")
            x, y = map(float, self.center)
            radius = float(self.radius)
            if radius <= 0:
                raise ValueError("circle forbidden region radius must be > 0")
            object.__setattr__(self, "center", (x, y))
            object.__setattr__(self, "radius", radius)
            object.__setattr__(self, "bounds", None)

    def contains(self, points: np.ndarray, clearance: float = 0.0) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        if p.ndim == 1:
            p = p[None, :]
        if p.ndim != 2 or p.shape[1] != 2:
            raise ValueError("points must have shape (n, 2)")
        if clearance < 0:
            raise ValueError("clearance must be >= 0")

        if self.kind == "rectangle":
            assert self.bounds is not None
            xmin, ymin, xmax, ymax = self.bounds
            return (
                (p[:, 0] >= xmin - clearance - GEOMETRY_TOL)
                & (p[:, 0] <= xmax + clearance + GEOMETRY_TOL)
                & (p[:, 1] >= ymin - clearance - GEOMETRY_TOL)
                & (p[:, 1] <= ymax + clearance + GEOMETRY_TOL)
            )

        assert self.center is not None and self.radius is not None
        center = np.asarray(self.center, dtype=float)
        distance = np.linalg.norm(p - center[None, :], axis=1)
        return distance <= self.radius + clearance + GEOMETRY_TOL

    def project_out(self, points: np.ndarray, clearance: float = 0.0) -> np.ndarray:
        p = np.asarray(points, dtype=float).copy()
        if p.ndim == 1:
            p = p[None, :]
        mask = self.contains(p, clearance=clearance)
        if not np.any(mask):
            return p

        margin = clearance + 1e-6
        if self.kind == "circle":
            assert self.center is not None and self.radius is not None
            center = np.asarray(self.center, dtype=float)
            delta = p[mask] - center[None, :]
            norm = np.linalg.norm(delta, axis=1, keepdims=True)
            fallback = np.array([[1.0, 0.0]])
            direction = np.divide(
                delta,
                np.maximum(norm, 1e-12),
                out=np.repeat(fallback, len(delta), axis=0),
                where=norm > 1e-12,
            )
            p[mask] = center[None, :] + direction * (self.radius + margin)
            return p

        assert self.bounds is not None
        xmin, ymin, xmax, ymax = self.bounds
        inside = p[mask]
        distances = np.column_stack([
            inside[:, 0] - xmin,
            xmax - inside[:, 0],
            inside[:, 1] - ymin,
            ymax - inside[:, 1],
        ])
        nearest = np.argmin(distances, axis=1)
        for local, side in enumerate(nearest):
            if side == 0:
                inside[local, 0] = xmin - margin
            elif side == 1:
                inside[local, 0] = xmax + margin
            elif side == 2:
                inside[local, 1] = ymin - margin
            else:
                inside[local, 1] = ymax + margin
        p[mask] = inside
        return p


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
    forbidden_regions: tuple[ForbiddenRegion, ...] = ()

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
        object.__setattr__(self, "forbidden_regions", tuple(self.forbidden_regions))


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

    if scenario.forbidden_regions:
        p = project_positions_out_of_forbidden_regions(p, scenario)
        p[:, 0] = np.clip(p[:, 0], 0.0, scenario.width)
        p[:, 1] = np.clip(p[:, 1], 0.0, scenario.height)

    return p


def forbidden_region_mask(
    points: np.ndarray,
    scenario: Scenario,
    clearance: float = 0.0,
) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")

    mask = np.zeros(len(p), dtype=bool)
    for region in scenario.forbidden_regions:
        mask |= region.contains(p, clearance=clearance)
    return mask


def positions_in_forbidden_regions(
    scenario: Scenario,
    positions: np.ndarray,
    clearance: float = 0.0,
) -> np.ndarray:
    return forbidden_region_mask(positions, scenario, clearance=clearance)


def project_positions_out_of_forbidden_regions(
    positions: np.ndarray,
    scenario: Scenario,
    clearance: float = 0.0,
    passes: int = 4,
) -> np.ndarray:
    p = np.asarray(positions, dtype=float).copy()
    for _ in range(max(1, passes)):
        before = p.copy()
        for region in scenario.forbidden_regions:
            p = region.project_out(p, clearance=clearance)
        if np.allclose(before, p, atol=1e-10, rtol=0.0):
            break
    return p
