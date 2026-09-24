"""Cheap endpoint proposals; none imply that a trajectory exists."""
from __future__ import annotations

import time
import numpy as np

from ..metrics import evaluate_positions
from ..objectives import DEFAULT_OBJECTIVE
from ..obstacles import formation_obstacle_free


def sensing_key(metrics):
    """Coverage first, redundancy second; motion cost is deliberately absent."""
    return float(metrics.weighted_coverage_ratio), float(metrics.fitness)


def candidate_points(problem, grid_size=8):
    s = problem.scenario
    grid = np.array([(x, y) for x in np.linspace(0, s.width, grid_size)
                     for y in np.linspace(0, s.height, grid_size)])
    centroid = (
        np.average(s.targets, axis=0, weights=s.target_weights)
        if np.sum(s.target_weights) > 0 else
        np.mean(s.targets, axis=0) if len(s.targets) else
        np.array([s.width / 2, s.height / 2])
    )
    return valid_points(problem, np.vstack([s.targets, grid, centroid[None]]))


def valid_points(problem, points):
    s = problem.scenario
    points = np.asarray(points, dtype=float)
    mask = (np.isfinite(points).all(axis=1) & (points >= 0).all(axis=1)
            & (points[:, 0] <= s.width) & (points[:, 1] <= s.height))
    points = points[mask]
    return points[np.array([formation_obstacle_free(q[None], problem.obstacles,
                           problem.obstacle_clearance) for q in points], dtype=bool)]


def greedy_formation(problem, grid_size=8, objective_config=DEFAULT_OBJECTIVE,
                     deadline=None):
    """The same obstacle-filtered Greedy seed used in the fixed-goal benchmark."""
    result = None
    for result in greedy_steps(problem, grid_size, objective_config, deadline):
        pass
    return result if result is not None and len(result) == problem.scenario.n_uavs else None


def greedy_steps(problem, grid_size=8, objective_config=DEFAULT_OBJECTIVE, deadline=None):
    """Resumable construction: yield after assigning each UAV."""
    s = problem.scenario
    candidates = candidate_points(problem, grid_size)
    positions = np.empty((0, 2))
    for _ in range(s.n_uavs):
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        options = candidates
        if len(positions):
            distances = np.linalg.norm(candidates[:, None] - positions[None], axis=2)
            options = candidates[(distances >= s.min_separation - 1e-9).all(axis=1)
                                 & (distances <= s.communication_radius + 1e-9).any(axis=1)]
        if not len(options):
            return None
        scores = [evaluate_positions(s, np.vstack([positions, q]), objective_config).fitness
                  for q in options]
        positions = np.vstack([positions, options[int(np.argmax(scores))]])
        yield positions.copy()


def improve_formations(problem, initial, seed=0, rounds=4, deadline=None,
                       objective_config=DEFAULT_OBJECTIVE, yield_every_uav=False):
    """Yield improving coordinate-search proposals with exact static checks.

    Candidate coverage/redundancy is computed in batches. Connectivity is checked
    on the whole formation, because replacing an articulation UAV can disconnect
    the rest even if the replacement has one neighbor.
    """
    s = problem.scenario
    rng = np.random.default_rng(seed)
    positions = np.array(initial, dtype=float, copy=True)
    current_key = sensing_key(evaluate_positions(s, positions, objective_config))
    fixed_points = candidate_points(problem)
    weights = s.target_weights if np.sum(s.target_weights) > 0 else np.ones(len(s.targets))
    total_weight = max(float(np.sum(weights)), 1e-12)
    directions = np.column_stack([np.cos(np.arange(16)*np.pi/8),
                                  np.sin(np.arange(16)*np.pi/8)])
    for sweep in range(rounds):
        improved = False
        step = s.sensing_radius * (0.5 ** (sweep + 1))
        for uav in rng.permutation(s.n_uavs):
            if deadline is not None and time.perf_counter() >= deadline:
                return
            options = valid_points(problem, np.vstack([
                fixed_points, positions[uav] + step * directions,
                positions[uav] + 0.5 * step * directions,
            ]))
            others = np.delete(positions, uav, axis=0)
            if len(others):
                distance = np.linalg.norm(options[:, None] - others[None], axis=2)
                options = options[(distance >= s.min_separation - 1e-9).all(axis=1)
                                  & (distance <= s.communication_radius).any(axis=1)]
            if not len(options):
                if yield_every_uav:
                    yield positions.copy()
                continue
            counts = (np.linalg.norm(s.targets[:, None] - others[None], axis=2)
                      <= s.sensing_radius).sum(axis=1)
            covered = np.linalg.norm(s.targets[:, None] - options[None], axis=2) <= s.sensing_radius
            candidate_counts = counts[:, None] + covered
            coverage = np.sum(weights[:, None] * (candidate_counts > 0), axis=0) / total_weight
            redundancy = np.sum(weights[:, None] * np.maximum(candidate_counts-1, 0), axis=0) / total_weight
            fitness = coverage - objective_config.redundancy_weight * redundancy
            order = np.lexsort((-fitness, -coverage))
            for index in order:
                if (float(coverage[index]), float(fitness[index])) <= current_key:
                    break
                trial = positions.copy()
                trial[uav] = options[index]
                metrics = evaluate_positions(s, trial, objective_config)
                key = sensing_key(metrics)
                if metrics.feasible and key > current_key:
                    positions, current_key = trial, key
                    improved = True
                    break
            if yield_every_uav:
                yield positions.copy()
        if improved and not yield_every_uav:
            yield positions.copy()
