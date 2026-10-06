from __future__ import annotations

import numpy as np

from .problem import Scenario, positions_in_forbidden_regions


def target_weights_or_ones(scenario: Scenario) -> np.ndarray:
    weights = np.asarray(scenario.target_weights, dtype=float)
    if float(np.sum(weights)) <= 0:
        return np.ones(len(scenario.targets), dtype=float)
    return weights


def static_candidate_points(
    scenario: Scenario,
    grid_size: int,
) -> np.ndarray:
    """Shared discrete candidate set for static deployment algorithms.

    Candidate positions are target coordinates, a regular grid, and the weighted
    target centroid. Keeping this helper shared is important for exact-oracle
    comparisons: CFG-LS and MILP then optimize over the same discrete domain.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if len(scenario.targets) == 0:
        raise ValueError("at least one target is required")

    xs = np.linspace(0.0, scenario.width, grid_size)
    ys = np.linspace(0.0, scenario.height, grid_size)
    grid = np.asarray(
        [(x, y) for x in xs for y in ys],
        dtype=float,
    )

    weights = target_weights_or_ones(scenario)
    centroid = np.average(
        scenario.targets,
        axis=0,
        weights=weights,
    )

    points = np.vstack([
        np.asarray(scenario.targets, dtype=float),
        grid,
        centroid[None, :],
    ])
    points[:, 0] = np.clip(points[:, 0], 0.0, scenario.width)
    points[:, 1] = np.clip(points[:, 1], 0.0, scenario.height)

    # Deterministic duplicate removal without perturbing actual coordinates.
    rounded = np.round(points, decimals=9)
    _, first = np.unique(
        rounded,
        axis=0,
        return_index=True,
    )
    unique = points[np.sort(first)]

    if scenario.forbidden_regions:
        unique = unique[~positions_in_forbidden_regions(scenario, unique)]
        if len(unique) == 0:
            raise ValueError("no static candidate points remain outside forbidden regions")

    return unique
