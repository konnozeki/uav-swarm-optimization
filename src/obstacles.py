from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import heapq
import math

import numpy as np


GEOMETRY_TOL = 1e-9


@dataclass(frozen=True)
class AxisAlignedRectangle:
    """Static rectangular no-fly obstacle in the 2D workspace.

    UAVs are modeled as points for CP3. obstacle_clearance in the transition
    problem expands every rectangle uniformly, which is the simple way to give
    the point model a physical safety margin.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    name: str = "obstacle"

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.x_min, self.y_min, self.x_max, self.y_max],
            dtype=float,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError("obstacle coordinates must be finite")
        if self.x_max <= self.x_min:
            raise ValueError("obstacle x_max must be greater than x_min")
        if self.y_max <= self.y_min:
            raise ValueError("obstacle y_max must be greater than y_min")

    def expanded(self, clearance: float) -> tuple[float, float, float, float]:
        if clearance < 0:
            raise ValueError("obstacle clearance must be non-negative")

        return (
            self.x_min - clearance,
            self.y_min - clearance,
            self.x_max + clearance,
            self.y_max + clearance,
        )


def obstacles_within_bounds(
    obstacles: tuple[AxisAlignedRectangle, ...],
    width: float,
    height: float,
) -> bool:
    """Return whether all physical obstacle rectangles lie inside the map."""
    return all(
        obstacle.x_min >= -GEOMETRY_TOL
        and obstacle.y_min >= -GEOMETRY_TOL
        and obstacle.x_max <= width + GEOMETRY_TOL
        and obstacle.y_max <= height + GEOMETRY_TOL
        for obstacle in obstacles
    )


def point_in_obstacle(
    point: np.ndarray,
    obstacle: AxisAlignedRectangle,
    clearance: float = 0.0,
) -> bool:
    """Treat the obstacle boundary as blocked."""
    x, y = np.asarray(point, dtype=float)
    x_min, y_min, x_max, y_max = obstacle.expanded(clearance)

    return bool(
        x_min - GEOMETRY_TOL <= x <= x_max + GEOMETRY_TOL
        and y_min - GEOMETRY_TOL <= y <= y_max + GEOMETRY_TOL
    )


def formation_obstacle_free(
    positions: np.ndarray,
    obstacles: tuple[AxisAlignedRectangle, ...],
    clearance: float = 0.0,
) -> bool:
    positions = np.asarray(positions, dtype=float)

    return not any(
        point_in_obstacle(point, obstacle, clearance)
        for point in positions
        for obstacle in obstacles
    )


def obstacle_violation_ratio(
    positions: np.ndarray,
    obstacles: tuple[AxisAlignedRectangle, ...],
    clearance: float = 0.0,
) -> float:
    """Fraction of UAV endpoints currently inside a no-fly region."""
    positions = np.asarray(positions, dtype=float)

    if len(positions) == 0:
        return 0.0

    violations = sum(
        any(
            point_in_obstacle(point, obstacle, clearance)
            for obstacle in obstacles
        )
        for point in positions
    )

    return float(violations / len(positions))


def segment_intersects_obstacle(
    start: np.ndarray,
    end: np.ndarray,
    obstacle: AxisAlignedRectangle,
    clearance: float = 0.0,
) -> bool:
    """Exact segment-vs-axis-aligned-rectangle intersection.

    A slab/Liang-Barsky style test is enough here and keeps the implementation
    dependency-free. Touching the expanded obstacle boundary counts as a
    collision, which is appropriate for a no-fly safety constraint.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    x_min, y_min, x_max, y_max = obstacle.expanded(clearance)
    direction = end - start

    t_enter = 0.0
    t_exit = 1.0

    for value, delta, lower, upper in (
        (start[0], direction[0], x_min, x_max),
        (start[1], direction[1], y_min, y_max),
    ):
        if abs(delta) <= 1e-15:
            if value < lower - GEOMETRY_TOL or value > upper + GEOMETRY_TOL:
                return False
            continue

        t1 = (lower - value) / delta
        t2 = (upper - value) / delta

        if t1 > t2:
            t1, t2 = t2, t1

        t_enter = max(t_enter, float(t1))
        t_exit = min(t_exit, float(t2))

        if t_enter > t_exit + GEOMETRY_TOL:
            return False

    return bool(
        t_exit >= -GEOMETRY_TOL
        and t_enter <= 1.0 + GEOMETRY_TOL
    )


def line_obstacle_free(
    start: np.ndarray,
    end: np.ndarray,
    obstacles: tuple[AxisAlignedRectangle, ...],
    clearance: float = 0.0,
) -> bool:
    return not any(
        segment_intersects_obstacle(
            start,
            end,
            obstacle,
            clearance,
        )
        for obstacle in obstacles
    )


def swarm_segment_obstacle_free(
    start: np.ndarray,
    end: np.ndarray,
    obstacles: tuple[AxisAlignedRectangle, ...],
    clearance: float = 0.0,
) -> bool:
    """Check every UAV's continuous linear segment against all obstacles."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    return all(
        line_obstacle_free(
            start[i],
            end[i],
            obstacles,
            clearance,
        )
        for i in range(len(start))
    )


def swarm_segment_obstacle_violations(
    start: np.ndarray,
    end: np.ndarray,
    obstacles: tuple[AxisAlignedRectangle, ...],
    clearance: float = 0.0,
) -> int:
    """Count UAV motion segments intersecting at least one obstacle."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    return int(
        sum(
            not line_obstacle_free(
                start[i],
                end[i],
                obstacles,
                clearance,
            )
            for i in range(len(start))
        )
    )


def project_positions_outside_obstacles(
    positions: np.ndarray,
    obstacles: tuple[AxisAlignedRectangle, ...],
    clearance: float,
    width: float,
    height: float,
    max_passes: int = 4,
) -> np.ndarray:
    """Push endpoint candidates to the nearest side of blocking rectangles.

    This is a lightweight endpoint repair used by formation optimizers. It does
    not plan a trajectory around an obstacle; that remains the transition
    planner's responsibility.
    """
    out = np.asarray(positions, dtype=float).copy()
    epsilon = 1e-6

    for _ in range(max_passes):
        changed = False

        for i, point in enumerate(out):
            for obstacle in obstacles:
                if not point_in_obstacle(
                    point,
                    obstacle,
                    clearance,
                ):
                    continue

                x_min, y_min, x_max, y_max = obstacle.expanded(clearance)
                x, y = float(point[0]), float(point[1])

                options = [
                    (abs(x - x_min), np.array([x_min - epsilon, y])),
                    (abs(x_max - x), np.array([x_max + epsilon, y])),
                    (abs(y - y_min), np.array([x, y_min - epsilon])),
                    (abs(y_max - y), np.array([x, y_max + epsilon])),
                ]

                _, repaired = min(
                    options,
                    key=lambda item: item[0],
                )

                out[i] = repaired
                out[i, 0] = np.clip(out[i, 0], 0.0, width)
                out[i, 1] = np.clip(out[i, 1], 0.0, height)
                changed = True

        if not changed:
            break

    return out


def _visibility_nodes(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: tuple[AxisAlignedRectangle, ...],
    clearance: float,
    width: float,
    height: float,
) -> list[np.ndarray]:
    """Create start/goal plus slightly offset expanded obstacle corners."""
    nodes = [
        np.asarray(start, dtype=float),
        np.asarray(goal, dtype=float),
    ]

    route_margin = max(1.0, 0.05 * max(clearance, 1.0))

    for obstacle in obstacles:
        x_min, y_min, x_max, y_max = obstacle.expanded(
            clearance + route_margin
        )

        for corner in (
            (x_min, y_min),
            (x_min, y_max),
            (x_max, y_min),
            (x_max, y_max),
        ):
            point = np.asarray(corner, dtype=float)

            if (
                -GEOMETRY_TOL <= point[0] <= width + GEOMETRY_TOL
                and -GEOMETRY_TOL <= point[1] <= height + GEOMETRY_TOL
            ):
                nodes.append(
                    np.array([
                        np.clip(point[0], 0.0, width),
                        np.clip(point[1], 0.0, height),
                    ])
                )

    return nodes


@lru_cache(maxsize=128)
def _cached_corner_visibility_graph(
    obstacles: tuple[AxisAlignedRectangle, ...],
    clearance: float,
    width: float,
    height: float,
) -> tuple[
    tuple[tuple[float, float], ...],
    tuple[tuple[tuple[int, float], ...], ...],
]:
    """Cache obstacle-corner visibility that is constant during planning.

    Only the current UAV position and its goal move between planner steps. The
    obstacle corners and every visible corner-to-corner edge remain unchanged,
    so rebuilding that quadratic graph for every UAV at every step is wasteful.
    """
    placeholder = np.zeros(2, dtype=float)
    nodes = _visibility_nodes(
        placeholder,
        placeholder,
        obstacles,
        clearance,
        width,
        height,
    )[2:]
    coordinates = tuple(
        (float(node[0]), float(node[1]))
        for node in nodes
    )
    adjacency: list[list[tuple[int, float]]] = [
        [] for _ in coordinates
    ]

    for i in range(len(coordinates)):
        first = np.asarray(coordinates[i], dtype=float)
        for j in range(i + 1, len(coordinates)):
            second = np.asarray(coordinates[j], dtype=float)
            if not line_obstacle_free(
                first,
                second,
                obstacles,
                clearance,
            ):
                continue
            distance = float(np.linalg.norm(first - second))
            adjacency[i].append((j, distance))
            adjacency[j].append((i, distance))

    return coordinates, tuple(tuple(edges) for edges in adjacency)


def next_visibility_waypoint(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: tuple[AxisAlignedRectangle, ...],
    clearance: float,
    width: float,
    height: float,
) -> np.ndarray | None:
    """Return the first waypoint on a shortest obstacle-free visibility path.

    With axis-aligned rectangles, their expanded corners are sufficient
    visibility-graph vertices for a simple polygonal detour. Corner-to-corner
    visibility is cached; each call only connects the moving start and goal.
    """
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    if not obstacles or line_obstacle_free(
        start,
        goal,
        obstacles,
        clearance,
    ):
        return goal.copy()

    corner_coordinates, corner_adjacency = _cached_corner_visibility_graph(
        obstacles,
        float(clearance),
        float(width),
        float(height),
    )
    corners = [
        np.asarray(point, dtype=float)
        for point in corner_coordinates
    ]
    start_links: list[tuple[int, float]] = []
    goal_links: dict[int, float] = {}

    for corner_index, corner in enumerate(corners):
        node_index = corner_index + 2
        if line_obstacle_free(
            start,
            corner,
            obstacles,
            clearance,
        ):
            start_links.append((
                node_index,
                float(np.linalg.norm(start - corner)),
            ))
        if line_obstacle_free(
            corner,
            goal,
            obstacles,
            clearance,
        ):
            goal_links[corner_index] = float(
                np.linalg.norm(corner - goal)
            )

    n = len(corners) + 2
    distances = [math.inf] * n
    previous = [-1] * n
    distances[0] = 0.0
    queue = [(0.0, 0)]

    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances[node] + GEOMETRY_TOL:
            continue
        if node == 1:
            break

        if node == 0:
            neighbours = start_links
        else:
            corner_index = node - 2
            neighbours = [
                (other + 2, weight)
                for other, weight in corner_adjacency[corner_index]
            ]
            if corner_index in goal_links:
                neighbours.append((1, goal_links[corner_index]))

        for neighbour, weight in neighbours:
            candidate = distance + weight
            if candidate + GEOMETRY_TOL < distances[neighbour]:
                distances[neighbour] = candidate
                previous[neighbour] = node
                heapq.heappush(queue, (candidate, neighbour))

    if not math.isfinite(distances[1]):
        return None

    node = 1
    while previous[node] not in (-1, 0):
        node = previous[node]

    if previous[node] == -1 or node < 2:
        return None

    return corners[node - 2].copy()


def _legacy_next_visibility_waypoint(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: tuple[AxisAlignedRectangle, ...],
    clearance: float,
    width: float,
    height: float,
) -> np.ndarray | None:
    """Previous full-graph implementation retained for equivalence tests."""
    nodes = _visibility_nodes(
        start,
        goal,
        obstacles,
        clearance,
        width,
        height,
    )

    n = len(nodes)
    adjacency: list[list[tuple[int, float]]] = [
        []
        for _ in range(n)
    ]

    for i in range(n):
        for j in range(i + 1, n):
            if not line_obstacle_free(
                nodes[i],
                nodes[j],
                obstacles,
                clearance,
            ):
                continue

            distance = float(
                np.linalg.norm(nodes[i] - nodes[j])
            )
            adjacency[i].append((j, distance))
            adjacency[j].append((i, distance))

    distances = [math.inf] * n
    previous = [-1] * n
    distances[0] = 0.0
    queue = [(0.0, 0)]

    while queue:
        distance, node = heapq.heappop(queue)

        if distance > distances[node] + GEOMETRY_TOL:
            continue
        if node == 1:
            break

        for neighbor, weight in adjacency[node]:
            candidate = distance + weight

            if candidate + GEOMETRY_TOL < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(
                    queue,
                    (candidate, neighbor),
                )

    if not math.isfinite(distances[1]):
        return None

    path = [1]
    node = 1

    while node != 0:
        node = previous[node]

        if node < 0:
            return None

        path.append(node)

    path.reverse()

    if len(path) < 2:
        return goal.copy()

    return nodes[path[1]].copy()
