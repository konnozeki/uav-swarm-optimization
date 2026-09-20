from __future__ import annotations

import numpy as np

from .problem import Scenario, clip_positions
from .graph_ops import connected_components, pairwise_distances


REPAIR_TOL = 1e-9


def _translation_capacity(
    component_positions: np.ndarray,
    direction: np.ndarray,
    scenario: Scenario,
) -> float:
    """Maximum whole-component translation before leaving the map."""
    eps = 1e-12
    cap = float("inf")

    dx, dy = float(direction[0]), float(direction[1])

    for x, y in component_positions:
        if dx > eps:
            cap = min(cap, (scenario.width - x) / dx)
        elif dx < -eps:
            cap = min(cap, (0.0 - x) / dx)

        if dy > eps:
            cap = min(cap, (scenario.height - y) / dy)
        elif dy < -eps:
            cap = min(cap, (0.0 - y) / dy)

    return max(0.0, float(cap))


def repair_connectivity(
    positions: np.ndarray,
    scenario: Scenario,
    max_passes: int | None = None,
    target_fraction: float = 0.92,
) -> np.ndarray:
    """Heuristically reconnect disconnected communication components.

    The closest pair of components is selected. Whole-component translation is
    preferred because it preserves internal pairwise distances. If one component
    hits the map boundary, the remaining required translation is attempted with
    the other component.
    """
    p = clip_positions(np.asarray(positions, dtype=float), scenario)
    n = len(p)

    if n <= 1:
        return p

    if max_passes is None:
        max_passes = 2 * n

    desired_link = target_fraction * scenario.communication_radius

    for _ in range(max_passes):
        components = connected_components(scenario, p)
        if len(components) <= 1:
            break

        best = None

        for a_idx in range(len(components)):
            for b_idx in range(a_idx + 1, len(components)):
                a = sorted(components[a_idx])
                b = sorted(components[b_idx])

                d = pairwise_distances(p[a], p[b])
                local = np.unravel_index(np.argmin(d), d.shape)
                distance = float(d[local])

                i = a[local[0]]
                j = b[local[1]]

                if best is None or distance < best[0]:
                    best = (distance, a, b, i, j)

        if best is None:
            break

        distance, comp_a, comp_b, i, j = best
        if distance <= scenario.communication_radius + REPAIR_TOL:
            continue

        vector = p[j] - p[i]
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            break

        direction = vector / norm
        required = max(0.0, distance - desired_link)
        remaining = required

        choices = [
            (comp_a, direction),
            (comp_b, -direction),
        ]
        choices.sort(key=lambda x: len(x[0]))

        moved = False

        for component, move_dir in choices:
            idx = np.array(component, dtype=int)
            capacity = _translation_capacity(p[idx], move_dir, scenario)
            step = min(remaining, capacity)

            if step > REPAIR_TOL:
                p[idx] += move_dir * step
                moved = True
                remaining -= step

                if remaining <= REPAIR_TOL:
                    break

        if not moved and remaining > REPAIR_TOL:
            shift = direction * (remaining / 2.0)
            p[i] += shift
            p[j] -= shift

        p = clip_positions(p, scenario)

    return p


def repair_collisions(
    positions: np.ndarray,
    scenario: Scenario,
    max_passes: int = 8,
) -> np.ndarray:
    p = clip_positions(np.asarray(positions, dtype=float), scenario)
    n = len(p)

    if n <= 1 or scenario.min_separation <= 0:
        return p

    rng = np.random.default_rng(123456)

    for _ in range(max_passes):
        changed = False

        for i in range(n):
            for j in range(i + 1, n):
                delta = p[j] - p[i]
                d = float(np.linalg.norm(delta))

                if d >= scenario.min_separation - REPAIR_TOL:
                    continue

                changed = True

                if d <= 1e-12:
                    direction = rng.normal(size=2)
                    direction /= np.linalg.norm(direction)
                else:
                    direction = delta / d

                overlap = scenario.min_separation - d
                move = 0.51 * overlap * direction

                p[i] -= move
                p[j] += move

        p = clip_positions(p, scenario)

        if not changed:
            break

    return p


def _connectivity_ok(positions: np.ndarray, scenario: Scenario) -> bool:
    return len(connected_components(scenario, positions)) == 1


def _collision_ok(positions: np.ndarray, scenario: Scenario) -> bool:
    if len(positions) <= 1 or scenario.min_separation <= 0:
        return True

    d = pairwise_distances(positions, positions)
    upper = d[np.triu_indices(len(positions), k=1)]
    return bool(
        np.all(upper >= scenario.min_separation - REPAIR_TOL)
    )


def repair_solution(
    positions: np.ndarray,
    scenario: Scenario,
    connectivity: bool = True,
    collisions: bool = True,
    max_rounds: int = 6,
) -> np.ndarray:
    """Alternate connectivity and collision repair until stable or budget is used.

    Repair is deliberately heuristic. It does not assert feasibility at the end;
    benchmark metrics remain responsible for exposing repair failures.
    """
    p = clip_positions(np.asarray(positions, dtype=float), scenario)

    for _ in range(max_rounds):
        if connectivity:
            p = repair_connectivity(p, scenario)

        if collisions:
            p = repair_collisions(p, scenario)

        p = clip_positions(p, scenario)

        connectivity_ok = (
            not connectivity or _connectivity_ok(p, scenario)
        )
        collision_ok = (
            not collisions or _collision_ok(p, scenario)
        )

        if connectivity_ok and collision_ok:
            break

    return clip_positions(p, scenario)
