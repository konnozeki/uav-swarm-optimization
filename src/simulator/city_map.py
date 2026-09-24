from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from ..obstacles import AxisAlignedRectangle, point_in_obstacle
from ..problem import Scenario
from ..reconfiguration import ReconfigurationProblem
from ..experiments.transition_cases import ring_formation


SCHEMA_VERSION = 1


@dataclass
class CityMap:
    """Serializable 2D target map with rectangular no-fly regions."""

    name: str
    width: float
    height: float
    targets: np.ndarray
    target_weights: np.ndarray
    obstacles: tuple[AxisAlignedRectangle, ...] = ()
    seed: int = 0
    difficulty: str = "custom"

    def __post_init__(self) -> None:
        self.targets = np.asarray(self.targets, dtype=float)
        self.target_weights = np.asarray(self.target_weights, dtype=float)
        self.obstacles = tuple(self.obstacles)

        if self.width <= 0 or self.height <= 0:
            raise ValueError("map width and height must be positive")
        if self.targets.ndim != 2 or self.targets.shape[1] != 2:
            raise ValueError("targets must have shape (n_targets, 2)")
        if len(self.targets) == 0:
            raise ValueError("city map must contain at least one target")
        if self.target_weights.shape != (len(self.targets),):
            raise ValueError("target_weights must match targets")
        if np.any(self.target_weights <= 0):
            raise ValueError("target weights must be positive")
        if np.any(self.targets < 0):
            raise ValueError("target coordinates must be non-negative")
        if len(self.targets) and (
            np.any(self.targets[:, 0] > self.width)
            or np.any(self.targets[:, 1] > self.height)
        ):
            raise ValueError("targets must lie inside map bounds")
        for obstacle in self.obstacles:
            if (
                obstacle.x_min < 0
                or obstacle.y_min < 0
                or obstacle.x_max > self.width
                or obstacle.y_max > self.height
            ):
                raise ValueError("obstacles must lie inside map bounds")

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "coordinate_system": "local_meters",
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "difficulty": self.difficulty,
            "targets": [
                {"x": float(p[0]), "y": float(p[1]), "weight": float(w)}
                for p, w in zip(self.targets, self.target_weights)
            ],
            "obstacles": [
                {
                    "name": obstacle.name,
                    "x_min": obstacle.x_min,
                    "y_min": obstacle.y_min,
                    "x_max": obstacle.x_max,
                    "y_max": obstacle.y_max,
                }
                for obstacle in self.obstacles
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CityMap":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported city-map schema: {payload.get('schema_version')}"
            )
        targets = payload.get("targets", [])
        obstacles = payload.get("obstacles", [])
        return cls(
            name=str(payload.get("name", "custom_city")),
            width=float(payload["width"]),
            height=float(payload["height"]),
            targets=np.asarray(
                [[target["x"], target["y"]] for target in targets],
                dtype=float,
            ).reshape((-1, 2)),
            target_weights=np.asarray(
                [target.get("weight", 1.0) for target in targets],
                dtype=float,
            ),
            obstacles=tuple(
                AxisAlignedRectangle(
                    float(obstacle["x_min"]),
                    float(obstacle["y_min"]),
                    float(obstacle["x_max"]),
                    float(obstacle["y_max"]),
                    name=str(obstacle.get("name", f"obstacle_{index}")),
                )
                for index, obstacle in enumerate(obstacles)
            ),
            seed=int(payload.get("seed", 0)),
            difficulty=str(payload.get("difficulty", "custom")),
        )

    def to_problem(
        self,
        *,
        n_uavs: int,
        sensing_radius: float,
        communication_radius: float,
        min_separation: float,
        max_speed: float,
        obstacle_clearance: float = 8.0,
    ) -> ReconfigurationProblem:
        scenario = Scenario(
            name=self.name,
            pattern=f"synthetic_city_{self.difficulty}",
            width=self.width,
            height=self.height,
            targets=self.targets,
            target_weights=self.target_weights,
            n_uavs=n_uavs,
            sensing_radius=sensing_radius,
            communication_radius=communication_radius,
            min_separation=min_separation,
            seed=self.seed,
        )
        center = (0.15 * self.width, 0.16 * self.height)
        radius = min(
            0.35 * communication_radius,
            0.10 * min(self.width, self.height),
        )
        start = ring_formation(center, n_uavs, radius)
        return ReconfigurationProblem(
            name=f"{self.name}_reconfiguration",
            scenario=scenario,
            start_positions=start,
            max_speed=max_speed,
            dt=1.0,
            allow_reassignment=True,
            obstacles=self.obstacles,
            obstacle_clearance=obstacle_clearance,
        )


DIFFICULTY_CONFIG = {
    "Dễ": {"obstacle_ratio": 0.06, "clusters": 2},
    "Trung bình": {"obstacle_ratio": 0.10, "clusters": 3},
    "Khó": {"obstacle_ratio": 0.14, "clusters": 4},
}


def _rectangles_overlap(a: AxisAlignedRectangle, b: AxisAlignedRectangle) -> bool:
    return not (
        a.x_max <= b.x_min
        or b.x_max <= a.x_min
        or a.y_max <= b.y_min
        or b.y_max <= a.y_min
    )


def generate_city_map(
    *,
    width: float = 1200.0,
    height: float = 900.0,
    n_targets: int = 80,
    n_obstacles: int = 10,
    difficulty: str = "Trung bình",
    seed: int = 42,
) -> CityMap:
    """Generate a reproducible 2D city benchmark with feasible open space."""
    if difficulty not in DIFFICULTY_CONFIG:
        raise ValueError(f"unknown difficulty: {difficulty}")
    if n_targets <= 0 or n_obstacles < 0:
        raise ValueError("target count must be positive and obstacles non-negative")

    rng = np.random.default_rng(seed)
    config = DIFFICULTY_CONFIG[difficulty]
    reserved_start = AxisAlignedRectangle(
        0.02 * width,
        0.02 * height,
        0.29 * width,
        0.31 * height,
        name="reserved_start",
    )
    obstacles: list[AxisAlignedRectangle] = []
    attempts = 0
    max_attempts = max(200, 80 * n_obstacles)

    while len(obstacles) < n_obstacles and attempts < max_attempts:
        attempts += 1
        base = config["obstacle_ratio"]
        obstacle_width = rng.uniform(0.55 * base, 1.25 * base) * width
        obstacle_height = rng.uniform(0.55 * base, 1.25 * base) * height
        x_min = rng.uniform(0.04 * width, 0.96 * width - obstacle_width)
        y_min = rng.uniform(0.04 * height, 0.96 * height - obstacle_height)
        candidate = AxisAlignedRectangle(
            x_min,
            y_min,
            x_min + obstacle_width,
            y_min + obstacle_height,
            name=f"city_block_{len(obstacles) + 1:02d}",
        )
        if _rectangles_overlap(candidate, reserved_start):
            continue
        if any(_rectangles_overlap(candidate, current) for current in obstacles):
            continue
        obstacles.append(candidate)

    if len(obstacles) != n_obstacles:
        raise RuntimeError("could not place requested obstacle count")

    cluster_count = int(config["clusters"])
    cluster_centres = rng.uniform(
        [0.35 * width, 0.30 * height],
        [0.90 * width, 0.88 * height],
        size=(cluster_count, 2),
    )
    targets: list[np.ndarray] = []
    weights: list[float] = []
    attempts = 0
    max_attempts = max(1000, 100 * n_targets)

    while len(targets) < n_targets and attempts < max_attempts:
        attempts += 1
        centre = cluster_centres[int(rng.integers(cluster_count))]
        candidate = rng.normal(
            centre,
            [0.13 * width, 0.13 * height],
        )
        candidate = np.clip(candidate, [15.0, 15.0], [width - 15.0, height - 15.0])
        if any(point_in_obstacle(candidate, obstacle, 8.0) for obstacle in obstacles):
            continue
        targets.append(candidate)
        weights.append(float(rng.integers(1, 6)))

    if len(targets) != n_targets:
        raise RuntimeError("could not place requested target count")

    return CityMap(
        name=f"city_{difficulty.lower().replace(' ', '_')}_{seed}",
        width=width,
        height=height,
        targets=np.asarray(targets),
        target_weights=np.asarray(weights),
        obstacles=tuple(obstacles),
        seed=seed,
        difficulty=difficulty,
    )


def save_city_map(city_map: CityMap, path: str | Path) -> None:
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError("city map output must be a .json file")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(city_map.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_city_map(path: str | Path) -> CityMap:
    path = Path(path)
    return CityMap.from_dict(json.loads(path.read_text(encoding="utf-8")))
