from __future__ import annotations

import numpy as np
import networkx as nx

from .problem import Scenario


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.linalg.norm(diff, axis=2)


def communication_graph_from_positions(
    positions: np.ndarray,
    communication_radius: float,
) -> nx.Graph:
    positions = np.asarray(positions, dtype=float)
    n = len(positions)

    g = nx.Graph()
    g.add_nodes_from(range(n))

    if n <= 1:
        return g

    d = pairwise_distances(positions, positions)

    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] <= communication_radius:
                g.add_edge(i, j, distance=float(d[i, j]))

    return g


def communication_graph(scenario: Scenario, positions: np.ndarray) -> nx.Graph:
    return communication_graph_from_positions(
        positions,
        scenario.communication_radius,
    )


def articulation_points(scenario: Scenario, positions: np.ndarray) -> set[int]:
    g = communication_graph(scenario, positions)
    if len(g) <= 2 or nx.number_connected_components(g) != 1:
        return set()
    return set(nx.articulation_points(g))


def connected_components(scenario: Scenario, positions: np.ndarray) -> list[set[int]]:
    g = communication_graph(scenario, positions)
    return [set(c) for c in nx.connected_components(g)]


def connectivity_deficit(scenario: Scenario, positions: np.ndarray) -> tuple[float, int]:
    """
    0 khi connected.
    Tiến dần tới 1 khi graph bị vỡ thành nhiều component.
    """
    g = communication_graph(scenario, positions)
    components = nx.number_connected_components(g)

    n = len(positions)
    if n <= 1:
        return 0.0, components

    deficit = (components - 1) / (n - 1)
    return float(deficit), int(components)
