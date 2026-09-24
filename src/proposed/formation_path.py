"""Joint waypoint optimization for an obstacle-blocked formation transition.

The optimizer is a proposal generator, never a feasibility oracle. Every
returned path must pass the existing continuous segment checks and carry a
common spanning-tree certificate for every segment.
"""
from __future__ import annotations

import numpy as np
import time
from dataclasses import replace
from scipy.optimize import minimize
from scipy.sparse.csgraph import minimum_spanning_tree

from ..obstacles import next_visibility_waypoint, formation_obstacle_free


def group_paths(problem, goals, knots=25):
    """Coordinated proposals only; every UAV segment still needs certification.

    Try translating the original shape, then a compact version. A centroid
    route alone is never considered proof of swarm feasibility.
    """
    start = problem.start_positions
    center = start.mean(axis=0)
    destination = goals.mean(axis=0)
    offsets = start - center
    distances = np.linalg.norm(start[:, None] - start[None], axis=2)
    np.fill_diagonal(distances, np.inf)
    minimum = distances.min()
    compact_scale = min(1.0, max(0.12, (problem.min_separation + 0.5) / max(minimum, 1e-9)))
    for scale in dict.fromkeys((1.0, compact_scale)):
        shape = scale * offsets
        radius = float(np.max(np.linalg.norm(shape, axis=1)))
        clearance = problem.obstacle_clearance + radius
        # A coverage formation may straddle an obstacle: its centroid need not
        # be free. End compact transit at a nearby staging point, then expand.
        angles = np.arange(16)*2*np.pi/16
        staging = [destination]
        staging.extend(destination + r*np.array([np.cos(a), np.sin(a)])
                       for r in (50., 100., 200., 300.) for a in angles)
        staging = [p for p in staging
                   if np.all(p >= radius)
                   and np.all(p <= np.array([problem.width, problem.height])-radius)
                   and formation_obstacle_free(p[None], problem.obstacles, clearance)]
        if not staging or not formation_obstacle_free(center[None], problem.obstacles, clearance):
            continue
        target_center = staging[0]
        proxy = replace(problem, start_positions=center[None],
                        goal_positions=target_center[None], obstacle_clearance=clearance)
        centers = initial_path(proxy, target_center[None], max(3, knots-4))[:, 0]
        transit = centers[:, None] + shape[None]
        yield np.concatenate([start[None], transit, goals[None]])


def initial_path(problem, goals, knots):
    """Seed joint optimization with independently routed, synchronized paths."""
    paths = []
    for start, goal in zip(problem.start_positions, goals):
        vertices = [start.copy()]
        for _ in range(4 * len(problem.obstacles) + 4):
            waypoint = next_visibility_waypoint(
                vertices[-1], goal, problem.obstacles,
                problem.obstacle_clearance + 3.0, problem.width, problem.height,
            )
            if waypoint is None:
                vertices.append(goal.copy())
                break
            vertices.append(waypoint)
            if np.linalg.norm(waypoint - goal) < 1e-8:
                break
        if not np.allclose(vertices[-1], goal):
            vertices.append(goal.copy())
        vertices = np.asarray(vertices)
        length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(vertices, axis=0), axis=1))]
        distances = np.linspace(0.0, length[-1], knots)
        paths.append(np.column_stack([
            np.interp(distances, length, vertices[:, axis]) for axis in range(2)
        ]))
    return np.stack(paths, axis=1)


def optimize_path(problem, goals, certify, knots=25, max_iterations=800, *,
                  initial_trajectory=None, endpoint_radius=0.0,
                  endpoint_terms=None, deadline=None, progress_callback=None):
    """Deform all routes together; return a certified solution or None.

    This searches the complete trajectory but remains a local numerical
    optimizer: failure is not a proof that no feasible route exists.
    """
    if deadline is not None and time.perf_counter() >= deadline:
        return None
    if endpoint_radius < 0 or not np.isfinite(endpoint_radius):
        raise ValueError("endpoint_radius must be finite and non-negative")
    if initial_trajectory is None:
        path = initial_path(problem, goals, knots)
    else:
        initial = np.asarray(initial_trajectory, dtype=float)
        if (initial.ndim != 3 or initial.shape[1:] != goals.shape
                or not len(initial) or not np.isfinite(initial).all()):
            raise ValueError("initial_trajectory must be finite with shape (T, N, 2)")
        # Preserve the old route topology while smoothly moving its endpoints.
        old_times = np.linspace(0, 1, len(initial))
        new_times = np.linspace(0, 1, knots)
        path = np.stack([np.interp(new_times, old_times, column)
                         for column in initial.reshape(len(initial), -1).T], axis=1)
        path = path.reshape(knots, *goals.shape)
        path += (1-new_times[:, None, None]) * (problem.start_positions-path[0])
        path += new_times[:, None, None]**2 * (goals-path[-1])
        path[..., 0] = np.clip(path[..., 0], 0, problem.width)
        path[..., 1] = np.clip(path[..., 1], 0, problem.height)
    path[0], path[-1] = problem.start_positions, goals
    flexible = endpoint_radius > 0
    variable_slice = slice(1, None if flexible else -1)
    n = problem.n_uavs
    ii, jj = np.triu_indices(n, 1)
    radius = problem.communication_radius
    separation = problem.min_separation
    start = problem.start_positions

    class DeadlineReached(Exception):
        pass

    def objective(flat):
        if deadline is not None and time.perf_counter() >= deadline:
            raise DeadlineReached
        states = np.concatenate([start[None], flat.reshape(-1, n, 2)]) if flexible else np.concatenate(
            [start[None], flat.reshape(-1, n, 2), goals[None]])
        grad = np.zeros_like(states)
        value = 0.0
        # Small regularization removes arbitrary oscillations, without assigning
        # sensing or motion cost a role in choosing the destination formation.
        delta = np.diff(states, axis=0)
        value += 1e-5 * np.sum(delta ** 2)
        grad[:-1] -= 2e-5 * delta
        grad[1:] += 2e-5 * delta

        # Exact minimum pair separation over every linear segment.
        r0 = states[:-1, ii] - states[:-1, jj]
        dr = (states[1:, ii] - states[1:, jj]) - r0
        denom = np.sum(dr * dr, axis=-1)
        tau = np.clip(-np.sum(r0 * dr, axis=-1) / np.maximum(denom, 1e-15), 0, 1)
        relative = r0 + tau[..., None] * dr
        distance = np.linalg.norm(relative, axis=-1)
        violation = np.maximum(separation + 0.5 - distance, 0)
        value += np.sum(violation ** 2)
        force = -2 * violation[..., None] * relative / np.maximum(distance[..., None], 1e-12)
        for t in range(knots - 1):
            np.add.at(grad[t], ii, (1-tau[t, :, None])*force[t])
            np.add.at(grad[t], jj, -(1-tau[t, :, None])*force[t])
            np.add.at(grad[t+1], ii, tau[t, :, None]*force[t])
            np.add.at(grad[t+1], jj, -tau[t, :, None]*force[t])

        # A tree in the intersection of endpoint graphs certifies the entire
        # segment. The selected tree may differ across successive segments.
        distances = np.linalg.norm(states[:, ii] - states[:, jj], axis=-1)
        for t in range(knots - 1):
            weights = np.maximum(distances[t], distances[t+1])
            matrix = np.zeros((n, n))
            matrix[ii, jj] = np.maximum(weights, 1e-8)
            tree = minimum_spanning_tree(matrix + matrix.T).tocoo()
            a, b = tree.row, tree.col
            for endpoint in (t, t+1):
                relative = states[endpoint, a] - states[endpoint, b]
                length = np.linalg.norm(relative, axis=1)
                violation = np.maximum(length - (radius - 0.5), 0)
                value += np.sum(violation ** 2)
                force = 2 * violation[:, None] * relative / np.maximum(length[:, None], 1e-12)
                np.add.at(grad[endpoint], a, force)
                np.add.at(grad[endpoint], b, -force)

        # Dense obstacle samples guide optimization; exact segment intersection
        # tests in `certify` decide whether the resulting path is acceptable.
        for tau in np.linspace(0, 1, 7):
            points = (1-tau)*states[:-1] + tau*states[1:]
            for obstacle in problem.obstacles:
                x0, y0, x1, y1 = obstacle.expanded(problem.obstacle_clearance + 2.0)
                gaps = np.stack([points[..., 0]-x0, x1-points[..., 0],
                                 points[..., 1]-y0, y1-points[..., 1]], axis=-1)
                side = np.argmin(gaps, axis=-1)
                depth = np.maximum(np.min(gaps, axis=-1), 0)
                value += np.sum(depth ** 2)
                directions = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
                force = 2 * depth[..., None] * directions[side]
                grad[:-1] += (1-tau) * force
                grad[1:] += tau * force
        if flexible:
            displacement = states[-1] - goals
            value += 0.001 * np.sum(displacement ** 2)
            grad[-1] += 0.002 * displacement
            if endpoint_terms is not None:
                endpoint_value, endpoint_grad = endpoint_terms(states[-1])
                value += endpoint_value
                grad[-1] += endpoint_grad
        return value, grad[variable_slice].ravel()

    bounds = [(0, problem.width), (0, problem.height)] * ((knots-2)*n)
    if flexible:
        for x, y in goals:
            bounds.extend([(max(0, x-endpoint_radius), min(problem.width, x+endpoint_radius)),
                           (max(0, y-endpoint_radius), min(problem.height, y+endpoint_radius))])
    # Reusing a nearby route often needs only endpoint deformation, with no
    # numerical solve at all. The same continuous validator decides this.
    accepted = certify(path)
    if progress_callback is not None:
        progress_callback(path.copy())
    if accepted is not None:
        return accepted
    iterations = 0

    class FeasiblePathFound(Exception):
        pass

    def callback(flat):
        nonlocal accepted, iterations
        iterations += 1
        path[variable_slice] = flat.reshape(-1, n, 2)
        if progress_callback is not None:
            progress_callback(path.copy())
        if iterations % 20 == 0:
            path[variable_slice] = flat.reshape(-1, n, 2)
            accepted = certify(path)
            if accepted is not None:
                raise FeasiblePathFound

    try:
        result = minimize(objective, path[variable_slice].ravel(), jac=True, method="L-BFGS-B",
                          bounds=bounds, callback=callback,
                          options={"maxiter": max_iterations, "ftol": 1e-12, "gtol": 1e-6, "maxls": 40})
    except FeasiblePathFound:
        return accepted
    except DeadlineReached:
        return None
    if deadline is not None and time.perf_counter() >= deadline:
        return None
    path[variable_slice] = result.x.reshape(-1, n, 2)
    if progress_callback is not None:
        progress_callback(path.copy())
    return certify(path)
