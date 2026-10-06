from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from ..metrics import evaluate_positions
from ..problem import Scenario, Solution
from ..static_candidates import static_candidate_points, target_weights_or_ones


@dataclass(frozen=True)
class ConnectedFrontierConfig:
    """Configuration for Connected Frontier Greedy + Leaf-Swap.

    The method is deliberately static: it optimizes one deployment formation.
    Connectivity is a construction invariant rather than a penalty that is
    repaired after search.
    """

    grid_size: int = 18
    redundancy_weight: float = 0.10
    potential_weight: float = 0.12
    connectivity_bonus: float = 0.01
    local_rounds: int = 12
    potential_decay_factor: float = 0.75
    multi_start_roots: int = 5


class ConnectedFrontierLeafSwap:
    """Connectivity-preserving constructive coverage optimizer.

    Phase 1 grows a formation one UAV at a time. A candidate can enter the
    frontier only when it is collision-safe and has a communication edge to the
    already selected swarm, so every partial formation is connected.

    Phase 2 repeatedly moves leaves of a spanning tree. Removing a tree leaf
    preserves connectivity of the remaining UAVs; the replacement point only
    needs one communication edge back to that connected remainder. This gives
    a topology-safe local-search neighborhood without connectivity repair.
    """

    name = "connected_frontier_leaf_swap"

    def __init__(
        self,
        config: ConnectedFrontierConfig = ConnectedFrontierConfig(),
    ) -> None:
        if config.grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if config.local_rounds < 0:
            raise ValueError("local_rounds must be non-negative")
        if config.potential_decay_factor <= 0:
            raise ValueError("potential_decay_factor must be positive")
        if config.multi_start_roots < 1:
            raise ValueError("multi_start_roots must be at least 1")
        self.config = config

    @staticmethod
    def _weights(scenario: Scenario) -> np.ndarray:
        return target_weights_or_ones(scenario)

    def _candidate_points(self, scenario: Scenario) -> np.ndarray:
        return static_candidate_points(
            scenario,
            self.config.grid_size,
        )

    def _precompute(self, scenario: Scenario, points: np.ndarray) -> dict:
        target_delta = (
            scenario.targets[None, :, :]
            - points[:, None, :]
        )
        target_distance = np.linalg.norm(target_delta, axis=2)
        covers = target_distance <= scenario.sensing_radius + 1e-9

        decay = max(
            self.config.potential_decay_factor * scenario.sensing_radius,
            1.0,
        )
        gap = np.maximum(
            target_distance - scenario.sensing_radius,
            0.0,
        )
        potential = np.exp(-gap / decay)

        point_delta = points[:, None, :] - points[None, :, :]
        pair_distance = np.linalg.norm(point_delta, axis=2)
        communication = (
            pair_distance <= scenario.communication_radius + 1e-9
        )
        np.fill_diagonal(communication, False)

        return dict(
            covers=covers,
            potential=potential,
            pair_distance=pair_distance,
            communication=communication,
        )

    @staticmethod
    def _coverage_redundancy(
        covers: np.ndarray,
        selected: list[int],
        weights: np.ndarray,
    ) -> tuple[float, float]:
        counts = np.sum(covers[selected], axis=0)
        total = max(float(np.sum(weights)), 1e-12)
        coverage = float(
            np.sum(weights * (counts > 0))
            / total
        )
        redundancy = float(
            np.sum(weights * np.maximum(counts - 1, 0))
            / total
        )
        return coverage, redundancy

    def _potential_score(
        self,
        potential: np.ndarray,
        selected: list[int],
        weights: np.ndarray,
    ) -> float:
        total = max(float(np.sum(weights)), 1e-12)
        best = np.max(potential[selected], axis=0)
        return float(np.sum(weights * best) / total)

    def _solution_key(
        self,
        covers: np.ndarray,
        potential: np.ndarray,
        selected: list[int],
        weights: np.ndarray,
    ) -> tuple[float, float]:
        coverage, redundancy = self._coverage_redundancy(
            covers,
            selected,
            weights,
        )
        smooth = self._potential_score(
            potential,
            selected,
            weights,
        )
        secondary = (
            -self.config.redundancy_weight * redundancy
            + self.config.potential_weight * smooth
        )
        return coverage, secondary

    def _root_candidates(
        self,
        scenario: Scenario,
        points: np.ndarray,
        data: dict,
    ) -> list[int]:
        covers = data["covers"]
        potential = data["potential"]
        weights = self._weights(scenario)
        total = max(float(np.sum(weights)), 1e-12)

        centroid = np.average(
            scenario.targets,
            axis=0,
            weights=weights,
        )
        centroid_seed = int(
            np.argmin(
                np.linalg.norm(
                    points - centroid[None, :],
                    axis=1,
                )
            )
        )

        direct = covers @ weights / total
        smooth = potential @ weights / total
        pool_size = min(
            len(points),
            max(
                self.config.multi_start_roots * 4,
                self.config.multi_start_roots + 4,
            ),
        )

        top_direct = [
            int(candidate_id)
            for candidate_id in np.argsort(direct)[-pool_size:][::-1]
        ]
        top_smooth = [
            int(candidate_id)
            for candidate_id in np.argsort(smooth)[-pool_size:][::-1]
        ]

        # Add spatially diverse roots. This catches split-cluster cases where
        # the weighted centroid lies in empty space and dense starts all come
        # from the same side of the map.
        distances_to_targets = np.linalg.norm(
            points[:, None, :] - scenario.targets[None, :, :],
            axis=2,
        )
        nearest_target = np.min(distances_to_targets, axis=1)
        target_like = nearest_target <= 1e-8
        if np.any(target_like):
            target_ids = np.flatnonzero(target_like)
        else:
            target_ids = np.arange(len(points))

        diverse: list[int] = []
        first = int(target_ids[np.argmax(direct[target_ids])])
        diverse.append(first)
        while len(diverse) < self.config.multi_start_roots and len(diverse) < len(target_ids):
            selected_points = points[np.asarray(diverse, dtype=int)]
            distance_to_selected = np.min(
                np.linalg.norm(
                    points[target_ids, None, :] - selected_points[None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            score = distance_to_selected / max(scenario.communication_radius, 1e-12)
            score += 0.05 * direct[target_ids]
            for existing in diverse:
                score[target_ids == existing] = -np.inf
            diverse.append(int(target_ids[int(np.argmax(score))]))

        seeds: list[int] = []

        def add_seed(candidate_id: int) -> None:
            if candidate_id not in seeds:
                seeds.append(candidate_id)

        add_seed(centroid_seed)
        if top_direct:
            add_seed(top_direct[0])
        if top_smooth:
            add_seed(top_smooth[0])
        for cid in diverse:
            add_seed(cid)
        for candidates in (top_direct, top_smooth):
            for cid in candidates:
                add_seed(cid)

        return seeds[: self.config.multi_start_roots]

    def _construct_from_root(
        self,
        scenario: Scenario,
        points: np.ndarray,
        data: dict,
        root: int,
    ) -> list[int]:
        covers = data["covers"]
        potential = data["potential"]
        pair_distance = data["pair_distance"]
        communication = data["communication"]
        weights = self._weights(scenario)
        total = max(float(np.sum(weights)), 1e-12)

        seed = int(root)
        selected = [seed]

        counts = covers[seed].astype(int)
        current_potential = potential[seed].copy()

        while len(selected) < scenario.n_uavs:
            selected_array = np.asarray(selected, dtype=int)
            distances = pair_distance[:, selected_array]

            safe = np.all(
                distances >= scenario.min_separation - 1e-9,
                axis=1,
            )
            connected = np.any(
                communication[:, selected_array],
                axis=1,
            )
            unused = np.ones(len(points), dtype=bool)
            unused[selected_array] = False
            valid = safe & connected & unused

            if not np.any(valid):
                raise RuntimeError(
                    "ConnectedFrontierLeafSwap could not extend the "
                    "connected frontier; increase grid_size or communication radius"
                )

            uncovered = counts == 0
            marginal = (
                covers[:, uncovered] @ weights[uncovered]
                / total
            )

            already_covered = counts > 0
            redundancy_increment = (
                covers[:, already_covered] @ weights[already_covered]
                / total
            )

            improved_potential = np.maximum(
                potential,
                current_potential[None, :],
            )
            potential_gain = (
                (improved_potential - current_potential[None, :])
                @ weights
                / total
            )

            # Robustness is only a tie-scale bonus: coverage remains dominant.
            within = communication[:, selected_array]
            slack = np.maximum(
                scenario.communication_radius - distances,
                0.0,
            ) / max(scenario.communication_radius, 1e-12)
            robust_link = np.max(
                np.where(within, slack, 0.0),
                axis=1,
            )

            score = (
                marginal
                + self.config.potential_weight * potential_gain
                - self.config.redundancy_weight * redundancy_increment
                + self.config.connectivity_bonus * robust_link
            )
            score[~valid] = -np.inf
            choice = int(np.argmax(score))

            selected.append(choice)
            counts += covers[choice].astype(int)
            current_potential = np.maximum(
                current_potential,
                potential[choice],
            )

        return selected

    def _construct(
        self,
        scenario: Scenario,
        points: np.ndarray,
        data: dict,
    ) -> list[int]:
        best_selected: list[int] | None = None
        best_key: tuple[float, float] | None = None
        last_error: RuntimeError | None = None
        weights = self._weights(scenario)

        for root in self._root_candidates(
            scenario,
            points,
            data,
        ):
            try:
                selected = self._construct_from_root(
                    scenario,
                    points,
                    data,
                    root,
                )
            except RuntimeError as failure:
                last_error = failure
                continue

            candidate_key = self._solution_key(
                data["covers"],
                data["potential"],
                selected,
                weights,
            )
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_selected = selected

        if best_selected is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError(
                "ConnectedFrontierLeafSwap could not construct a formation"
            )

        return best_selected

    @staticmethod
    def _spanning_tree_leaves(
        selected: list[int],
        communication: np.ndarray,
    ) -> list[int]:
        n = len(selected)
        if n <= 1:
            return []

        ids = np.asarray(selected, dtype=int)
        adjacency = communication[np.ix_(ids, ids)]
        degree = np.sum(adjacency, axis=1)
        root = int(np.argmax(degree))

        parent = np.full(n, -1, dtype=int)
        parent[root] = root
        queue = [root]
        order = []

        while queue:
            current = queue.pop(0)
            order.append(current)
            for nxt in np.flatnonzero(adjacency[current]):
                nxt = int(nxt)
                if parent[nxt] != -1:
                    continue
                parent[nxt] = current
                queue.append(nxt)

        if len(order) != n:
            raise RuntimeError("construction invariant broken: swarm disconnected")

        child_count = np.zeros(n, dtype=int)
        for node in range(n):
            if node == root:
                continue
            child_count[parent[node]] += 1

        leaves = [
            i
            for i in range(n)
            if i != root and child_count[i] == 0
        ]
        return leaves

    def _leaf_swap(
        self,
        scenario: Scenario,
        points: np.ndarray,
        data: dict,
        selected: list[int],
    ) -> list[int]:
        covers = data["covers"]
        potential = data["potential"]
        pair_distance = data["pair_distance"]
        communication = data["communication"]
        weights = self._weights(scenario)
        total = max(float(np.sum(weights)), 1e-12)

        current_key = self._solution_key(
            covers,
            potential,
            selected,
            weights,
        )

        for _ in range(self.config.local_rounds):
            leaves = self._spanning_tree_leaves(
                selected,
                communication,
            )
            best_move = None
            best_key = current_key

            for leaf_slot in leaves:
                others = [
                    candidate_id
                    for slot, candidate_id in enumerate(selected)
                    if slot != leaf_slot
                ]
                others_array = np.asarray(others, dtype=int)

                distances = pair_distance[:, others_array]
                safe = np.all(
                    distances >= scenario.min_separation - 1e-9,
                    axis=1,
                )
                connected = np.any(
                    communication[:, others_array],
                    axis=1,
                )
                unused = np.ones(len(points), dtype=bool)
                unused[others_array] = False
                valid = safe & connected & unused

                if not np.any(valid):
                    continue

                base_counts = np.sum(
                    covers[others_array],
                    axis=0,
                )
                candidate_counts = (
                    base_counts[None, :]
                    + covers.astype(int)
                )
                coverage = (
                    (candidate_counts > 0) @ weights
                    / total
                )
                redundancy = (
                    np.maximum(candidate_counts - 1, 0)
                    @ weights
                    / total
                )

                base_potential = np.max(
                    potential[others_array],
                    axis=0,
                )
                smooth = (
                    np.maximum(
                        potential,
                        base_potential[None, :],
                    )
                    @ weights
                    / total
                )
                secondary = (
                    -self.config.redundancy_weight * redundancy
                    + self.config.potential_weight * smooth
                )

                valid_ids = np.flatnonzero(valid)
                order = np.lexsort(
                    (
                        secondary[valid_ids],
                        coverage[valid_ids],
                    )
                )
                candidate_id = int(valid_ids[order[-1]])
                candidate_key = (
                    float(coverage[candidate_id]),
                    float(secondary[candidate_id]),
                )

                if candidate_key > best_key:
                    best_key = candidate_key
                    best_move = (leaf_slot, candidate_id)

            if best_move is None:
                break

            leaf_slot, candidate_id = best_move
            selected[leaf_slot] = candidate_id
            current_key = best_key

        return selected

    def solve(
        self,
        scenario: Scenario,
        seed: int = 0,
    ) -> tuple[Solution, float]:
        # Deterministic by design. Keep seed for the shared benchmark API.
        _ = seed
        started = time.perf_counter()

        if len(scenario.targets) == 0:
            raise ValueError("at least one target is required")

        points = self._candidate_points(scenario)
        data = self._precompute(
            scenario,
            points,
        )

        weights = self._weights(scenario)
        selected: list[int] | None = None
        selected_key: tuple[float, float] | None = None
        last_error: RuntimeError | None = None
        for root in self._root_candidates(
            scenario,
            points,
            data,
        ):
            try:
                candidate = self._construct_from_root(
                    scenario,
                    points,
                    data,
                    root,
                )
                candidate = self._leaf_swap(
                    scenario,
                    points,
                    data,
                    candidate,
                )
            except RuntimeError as failure:
                last_error = failure
                continue

            candidate_key = self._solution_key(
                data["covers"],
                data["potential"],
                candidate,
                weights,
            )
            if selected_key is None or candidate_key > selected_key:
                selected = candidate
                selected_key = candidate_key

        if selected is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError(
                "ConnectedFrontierLeafSwap could not construct a formation"
            )

        positions = points[np.asarray(selected, dtype=int)]

        metrics = evaluate_positions(
            scenario,
            positions,
        )
        if not metrics.feasible:
            raise RuntimeError(
                "construction invariant broken: returned formation is infeasible"
            )

        runtime = time.perf_counter() - started
        return (
            Solution(
                positions,
                algorithm=self.name,
                seed=seed,
            ),
            runtime,
        )
