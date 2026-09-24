from __future__ import annotations

import time

import networkx as nx
import numpy as np

from ..metrics import (
    evaluate_positions,
    soft_coverage_potential,
    target_coverage_matrix,
    uncovered_target_indices,
)
from ..objectives import ObjectiveConfig, DEFAULT_OBJECTIVE
from ..obstacles import (
    formation_obstacle_free,
    project_positions_outside_obstacles,
)
from ..graph_ops import communication_graph_from_positions
from ..reconfiguration import (
    DEFAULT_RECONFIGURATION_OBJECTIVE,
    ReconfigurationEvaluation,
    ReconfigurationObjectiveConfig,
    ReconfigurationProblem,
    ReconfigurationSolution,
    evaluate_reconfiguration,
)
from ..transition import bottleneck_goal_assignment
from .backbone_transition import BackboneTransitionPlanner
from .graph_aware_ga import GraphAwareConfig, GraphAwareGA


class TransitionAwareGA:
    """Graph-aware GA whose selection explicitly accounts for A -> B motion.

    Chromosomes contain only final formation B positions. The trajectory itself
    is not encoded in the chromosome. Each candidate B is evaluated by the
    deterministic lower-level BackboneTransitionPlanner.

    CP3 uses a coverage-first hierarchy by default:

        feasible transition
        > weighted sensing coverage bucket
        > final static CP2 fitness bucket
        > shorter formation time / travel among near-tied formations
        > exact sensing quality

    A separate smooth proximity signal is used only inside evolutionary search
    and local exploration to cross binary-coverage plateaus. It is deliberately
    excluded from final solution ordering so the algorithm does not keep extra
    motion merely to stand closer to targets it still cannot cover.

    This matches the intended semantics of "reach the best sensing formation we
    can, then prefer the cheaper transition among similarly good formations".
    The older weighted-sum selection is still available through
    selection_mode="weighted_joint" for reproducibility/ablation.

    Preliminary CP2 ablation did not show a clear gain from articulation-aware
    protection. CP3 therefore disables articulation protection by default so
    relay UAVs remain free to move while connectivity repair and the transition
    planner enforce the actual graph constraints. The final multi-seed campaign
    remains the source of any statistical claim.
    """

    name = "transition_aware_ga"

    VALID_SELECTION_MODES = {
        "coverage_first",
        "weighted_joint",
    }

    def __init__(
        self,
        population_size: int = 10,
        generations: int = 6,
        elite_size: int = 2,
        mutation_rate: float = 0.25,
        mutation_sigma: float = 65.0,
        tournament_size: int = 3,
        warm_start_fraction: float = 0.35,
        group_mutation_probability: float = 0.45,
        max_group_hops: int = 1,
        coverage_refine_rounds: int = 1,
        coverage_refine_exact_candidates: int = 2,
        finalist_count: int = 2,
        guided_offspring: int = 1,
        performance_tolerance: float = 0.005,
        use_smooth_coverage_potential: bool = True,
        use_connected_group_mutation: bool = True,
        use_guided_offspring: bool = True,
        use_deterministic_refinement: bool = True,
        use_coverage_probe: bool = True,
        use_motion_pruning: bool = True,
        use_obstacle_greedy_guidance: bool = False,
        obstacle_guidance_trials: int = 48,
        realtime: bool = True,
        use_global_planning: bool = True,
        graph_config: GraphAwareConfig | None = None,
        static_objective_config: ObjectiveConfig = DEFAULT_OBJECTIVE,
        reconfiguration_objective_config: ReconfigurationObjectiveConfig = (
            DEFAULT_RECONFIGURATION_OBJECTIVE
        ),
        selection_mode: str = "coverage_first",
        name: str | None = None,
    ) -> None:
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        if generations < 0:
            raise ValueError("generations must be non-negative")
        if not 0.0 <= warm_start_fraction <= 1.0:
            raise ValueError("warm_start_fraction must be in [0, 1]")
        if not 0.0 <= group_mutation_probability <= 1.0:
            raise ValueError("group_mutation_probability must be in [0, 1]")
        if max_group_hops < 0:
            raise ValueError("max_group_hops must be non-negative")
        if coverage_refine_rounds < 0:
            raise ValueError("coverage_refine_rounds must be non-negative")
        if coverage_refine_exact_candidates <= 0:
            raise ValueError(
                "coverage_refine_exact_candidates must be positive"
            )
        if finalist_count <= 0:
            raise ValueError("finalist_count must be positive")
        if guided_offspring < 0:
            raise ValueError("guided_offspring must be non-negative")
        if performance_tolerance <= 0:
            raise ValueError("performance_tolerance must be positive")
        if obstacle_guidance_trials < 0:
            raise ValueError("obstacle_guidance_trials must be non-negative")
        if selection_mode not in self.VALID_SELECTION_MODES:
            raise ValueError(
                "selection_mode must be one of "
                f"{sorted(self.VALID_SELECTION_MODES)}"
            )

        self.population_size = population_size
        self.generations = generations
        self.elite_size = min(elite_size, population_size)
        self.warm_start_fraction = warm_start_fraction
        self.group_mutation_probability = group_mutation_probability
        self.max_group_hops = max_group_hops
        self.coverage_refine_rounds = coverage_refine_rounds
        self.coverage_refine_exact_candidates = (
            coverage_refine_exact_candidates
        )
        self.finalist_count = min(finalist_count, population_size)
        self.guided_offspring = min(guided_offspring, population_size)
        self.performance_tolerance = performance_tolerance
        self.use_smooth_coverage_potential = use_smooth_coverage_potential
        self.use_connected_group_mutation = use_connected_group_mutation
        self.use_guided_offspring = use_guided_offspring
        self.use_deterministic_refinement = use_deterministic_refinement
        self.use_coverage_probe = use_coverage_probe
        self.use_motion_pruning = use_motion_pruning
        self.use_obstacle_greedy_guidance = use_obstacle_greedy_guidance
        self.obstacle_guidance_trials = obstacle_guidance_trials
        # In real-time mode the population is ranked with an admissible motion
        # lower bound; only final contenders run the expensive trajectory
        # planner.  Set False to reproduce the former exhaustive CP3 search.
        self.realtime = bool(realtime)
        self.use_global_planning = bool(use_global_planning)
        self.static_objective_config = static_objective_config
        self.reconfiguration_objective_config = (
            reconfiguration_objective_config
        )
        self.selection_mode = selection_mode

        if name is not None:
            self.name = name

        if graph_config is None:
            # Preliminary CP2 ablation did not show a clear articulation gain.
            # More importantly for CP3, freezing articulation UAVs
            # can prevent a connected relay chain from translating toward
            # uncovered sensing regions.
            graph_config = GraphAwareConfig(
                use_articulation_awareness=False,
            )

        # Reuse CP2 crossover/mutation/repair implementation through
        # composition. Only CP3 selection semantics are changed here.
        self.operator = GraphAwareGA(
            population_size=population_size,
            generations=generations,
            elite_size=self.elite_size,
            mutation_rate=mutation_rate,
            mutation_sigma=mutation_sigma,
            tournament_size=tournament_size,
            graph_config=graph_config,
            objective_config=static_objective_config,
        )

        # The raw backbone path is already continuously certified.  Expensive
        # shortcut smoothing is deliberately reserved for offline rendering,
        # not hundreds of fitness calls in the real-time optimizer.
        self.transition_planner = BackboneTransitionPlanner(
            enable_smoothing=False,
            enable_global_planning=False,
        )

    @staticmethod
    def _cache_key(positions: np.ndarray) -> bytes:
        """Stable exact cache key for elites copied across generations."""
        return np.ascontiguousarray(
            positions,
            dtype=np.float64,
        ).tobytes()

    def _performance_bucket(self, value: float) -> int:
        """Bucket performance so transition cost only wins near ties."""
        return int(
            np.floor(
                float(value)
                / self.performance_tolerance
            )
        )

    def _selection_key(
        self,
        evaluation: ReconfigurationEvaluation,
    ) -> tuple:
        """Return the ordering key used by elites, tournament and final choice.

        coverage_first
            Feasibility is a hard priority. Binary coverage is maximized first.
            Static fitness then handles redundancy quality. Coverage and static
            fitness are bucketed so time and travel can choose between near-tied
            formations, but cannot buy a materially worse sensing result.
            Smooth proximity is intentionally NOT part of the final ordering
            because it is a search aid, not a user objective.

        weighted_joint
            Reproduces the earlier CP3 prototype behavior where a weighted
            scalar joint_fitness directly trades sensing quality for transition
            cost.
        """
        feasible = 1 if evaluation.feasible else 0

        if self.selection_mode == "weighted_joint":
            return (
                feasible,
                float(evaluation.joint_fitness),
            )

        coverage_bucket = self._performance_bucket(
            evaluation.weighted_coverage_ratio
        )
        fitness_bucket = self._performance_bucket(
            evaluation.final_fitness
        )

        return (
            feasible,
            coverage_bucket,
            fitness_bucket,
            -float(evaluation.normalized_time),
            -float(evaluation.normalized_travel),
            float(evaluation.weighted_coverage_ratio),
            float(evaluation.final_fitness),
            float(evaluation.joint_fitness),
        )

    def _search_key(
        self,
        evaluation: ReconfigurationEvaluation,
    ) -> tuple:
        """Ordering used during evolutionary search.

        Coverage potential appears here, after hard feasibility and binary
        coverage, so an offspring can survive a few useful "almost there"
        moves before it actually reaches a new sensing target. Final selection
        uses _selection_key instead and therefore removes this temporary bias.
        """
        feasible = 1 if evaluation.feasible else 0

        if self.selection_mode == "weighted_joint":
            return (
                feasible,
                float(evaluation.joint_fitness),
            )

        return (
            feasible,
            float(evaluation.weighted_coverage_ratio),
            (
                float(evaluation.coverage_potential)
                if self.use_smooth_coverage_potential
                else 0.0
            ),
            float(evaluation.final_fitness),
            -float(evaluation.normalized_time),
            -float(evaluation.normalized_travel),
            float(evaluation.joint_fitness),
        )

    def _rank_population(
        self,
        evaluations: list[ReconfigurationEvaluation],
    ) -> tuple[list[int], list[float]]:
        """Rank a population once and expose rank scores to CP2 operators.

        GraphAwareGA's tournament/crossover interface expects scalar scores.
        Feeding it lexicographic *ranks* rather than joint_fitness keeps parent
        selection and crossover anchoring exactly consistent with CP3's
        hierarchy without modifying the reusable CP2 implementation.
        """
        order = sorted(
            range(len(evaluations)),
            key=lambda i: self._search_key(
                evaluations[i]
            ),
            reverse=True,
        )

        scores = [0.0] * len(evaluations)

        # Worst receives 0, best receives N-1. Only relative ordering matters.
        for rank, index in enumerate(reversed(order)):
            scores[index] = float(rank)

        return order, scores

    def _evaluate_candidate(
        self,
        problem: ReconfigurationProblem,
        positions: np.ndarray,
        cache: dict,
    ):
        key = self._cache_key(positions)

        if key not in cache:
            cache[key] = evaluate_reconfiguration(
                problem,
                positions,
                self.transition_planner,
                static_objective_config=self.static_objective_config,
                objective_config=self.reconfiguration_objective_config,
            )

        return cache[key]

    def _estimate_candidate(
        self,
        problem: ReconfigurationProblem,
        positions: np.ndarray,
        cache: dict,
    ) -> ReconfigurationEvaluation:
        """Cheap transition-aware ranking signal used inside real-time GA.

        The bottleneck assignment gives a hard lower bound on formation time
        and the matched straight-line travel is a lower bound on travel.  It
        preserves the desired preference for nearby formations without running
        the per-step backbone planner for every chromosome.  Finalists are
        always evaluated exactly before they can be returned.
        """
        key = (b"estimate", self._cache_key(positions))
        if key in cache:
            return cache[key]

        scenario = problem.scenario
        static_metrics = evaluate_positions(
            scenario,
            positions,
            self.static_objective_config,
        )
        coverage_potential = soft_coverage_potential(scenario, positions)
        obstacle_free = formation_obstacle_free(
            positions,
            problem.obstacles,
            problem.obstacle_clearance,
        )
        static_feasible = bool(static_metrics.feasible and obstacle_free)

        if static_feasible:
            assigned, _, bottleneck = bottleneck_goal_assignment(
                problem.start_positions,
                positions,
            )
            travel = float(np.sum(np.linalg.norm(
                assigned - problem.start_positions,
                axis=1,
            )))
            # The actual controller is discrete, so use its rounded-up time
            # lower bound instead of an optimistic continuous value.
            steps = int(np.ceil(
                bottleneck / max(problem.max_speed * problem.dt, 1e-12)
                - 1e-12
            ))
            formation_time = float(steps * problem.dt)
            normalized_time = formation_time / max(
                problem.reference_time,
                problem.dt,
            )
            normalized_travel = travel / max(
                scenario.n_uavs * problem.reference_distance,
                1e-12,
            )
            joint = (
                static_metrics.fitness
                - self.reconfiguration_objective_config.time_weight
                * normalized_time
                - self.reconfiguration_objective_config.travel_weight
                * normalized_travel
            )
        else:
            formation_time = float("nan")
            travel = float("nan")
            normalized_time = 1.0
            normalized_travel = 1.0
            joint = (
                static_metrics.fitness
                - self.reconfiguration_objective_config.transition_infeasible_penalty
                - static_metrics.constraint_violation
            )

        estimate = ReconfigurationEvaluation(
            final_fitness=float(static_metrics.fitness),
            weighted_coverage_ratio=float(static_metrics.weighted_coverage_ratio),
            coverage_potential=float(coverage_potential),
            redundancy_excess=float(static_metrics.redundancy_excess),
            static_feasible=static_feasible,
            static_constraint_violation=float(static_metrics.constraint_violation),
            final_obstacle_free=bool(obstacle_free),
            transition_attempted=False,
            transition_feasible=False,
            reached_goal=False,
            formation_time_sec=formation_time,
            total_travel_distance=travel,
            time_efficiency=0.0,
            continuous_connected_rate=0.0,
            min_continuous_pair_distance=float("nan"),
            transition_obstacle_free=False,
            normalized_time=float(normalized_time),
            normalized_travel=float(normalized_travel),
            joint_fitness=float(joint),
            # This means statically feasible for evolutionary ranking only;
            # exact transition feasibility is required for the returned result.
            feasible=static_feasible,
        )
        cache[key] = estimate
        return estimate

    def _repair_candidate(
        self,
        problem: ReconfigurationProblem,
        positions: np.ndarray,
    ) -> np.ndarray:
        """Repair graph/collision constraints and push endpoints out of obstacles.

        CP2 repair knows only the Scenario object, so obstacle repair lives in
        CP3. We alternate the two lightweight repairs a few times because moving
        a UAV out of a no-fly region can perturb connectivity and vice versa.
        """
        candidate = np.asarray(positions, dtype=float).copy()

        for _ in range(3):
            candidate = self.operator._repair(
                candidate,
                problem.scenario,
            )

            if problem.obstacles:
                candidate = project_positions_outside_obstacles(
                    candidate,
                    problem.obstacles,
                    problem.obstacle_clearance,
                    problem.scenario.width,
                    problem.scenario.height,
                )

        return candidate

    def _cluster_centroid(
        self,
        problem: ReconfigurationProblem,
        positions: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Choose one uncovered target cluster and return its weighted centroid.

        The seed target is sampled by target weight. Nearby uncovered targets
        inside roughly one-and-a-half sensing radii form the local cluster. This
        keeps the operator cheap while avoiding a random walk toward isolated
        individual targets.
        """
        scenario = problem.scenario
        uncovered = uncovered_target_indices(
            scenario,
            positions,
        )

        if len(uncovered) == 0:
            return None

        weights = scenario.target_weights[
            uncovered
        ].astype(float)
        total = float(np.sum(weights))

        if total > 0:
            probs = weights / total
            seed_idx = int(
                rng.choice(uncovered, p=probs)
            )
        else:
            seed_idx = int(rng.choice(uncovered))

        seed_target = scenario.targets[seed_idx]
        local_targets = scenario.targets[uncovered]
        distance = np.linalg.norm(
            local_targets - seed_target,
            axis=1,
        )
        mask = (
            distance
            <= 1.5 * scenario.sensing_radius
        )

        cluster_indices = uncovered[mask]
        cluster_targets = scenario.targets[
            cluster_indices
        ]
        cluster_weights = scenario.target_weights[
            cluster_indices
        ].astype(float)

        cluster_total = float(
            np.sum(cluster_weights)
        )

        if cluster_total > 0:
            centroid = np.average(
                cluster_targets,
                axis=0,
                weights=cluster_weights,
            )
        else:
            centroid = np.mean(
                cluster_targets,
                axis=0,
            )

        return (
            np.asarray(centroid, dtype=float),
            cluster_indices,
        )

    @staticmethod
    def _boundary_preserving_translation(
        positions: np.ndarray,
        group: set[int],
        delta: np.ndarray,
        problem: ReconfigurationProblem,
    ) -> np.ndarray:
        """Translate a connected group without immediately tearing graph edges.

        Internal group distances are unchanged by translation. Only edges that
        cross from the moving group to the stationary remainder can break, so a
        short backtracking loop keeps every currently active boundary edge
        inside communication radius.
        """
        scenario = problem.scenario
        graph = communication_graph_from_positions(
            positions,
            scenario.communication_radius,
        )

        boundary_edges = [
            (i, j)
            for i, j in graph.edges()
            if (i in group) != (j in group)
        ]

        scale = 1.0

        while scale >= 1e-3:
            candidate = positions.copy()
            index = np.array(
                sorted(group),
                dtype=int,
            )
            candidate[index] += scale * delta

            in_bounds = bool(
                np.all(candidate[:, 0] >= 0.0)
                and np.all(
                    candidate[:, 0]
                    <= scenario.width
                )
                and np.all(candidate[:, 1] >= 0.0)
                and np.all(
                    candidate[:, 1]
                    <= scenario.height
                )
            )

            if not in_bounds:
                scale *= 0.70
                continue

            if not formation_obstacle_free(
                candidate,
                problem.obstacles,
                problem.obstacle_clearance,
            ):
                scale *= 0.70
                continue

            boundary_ok = all(
                np.linalg.norm(
                    candidate[i] - candidate[j]
                )
                <= scenario.communication_radius + 1e-9
                for i, j in boundary_edges
            )

            if boundary_ok:
                return candidate

            scale *= 0.70

        return positions.copy()

    def _candidate_group_anchors(
        self,
        problem: ReconfigurationProblem,
        positions: np.ndarray,
        centroid: np.ndarray,
    ) -> list[int]:
        """Return a small anchor set: nearby UAVs plus low-contribution UAVs."""
        scenario = problem.scenario
        n = len(positions)

        distance_order = np.argsort(
            np.linalg.norm(
                positions - centroid,
                axis=1,
            )
        )

        matrix = target_coverage_matrix(
            scenario,
            positions,
        )
        counts = np.sum(matrix, axis=1)
        unique = np.zeros(n, dtype=float)

        total_weight = max(
            float(
                np.sum(scenario.target_weights)
            ),
            1e-12,
        )

        for i in range(n):
            mask = (
                matrix[:, i]
                & (counts == 1)
            )
            unique[i] = (
                np.sum(
                    scenario.target_weights[mask]
                )
                / total_weight
            )

        low_contribution_order = np.argsort(unique)

        anchors = []

        for idx in list(
            distance_order[: min(3, n)]
        ) + list(
            low_contribution_order[: min(3, n)]
        ):
            idx = int(idx)
            if idx not in anchors:
                anchors.append(idx)

        return anchors

    def _group_guided_mutation(
        self,
        problem: ReconfigurationProblem,
        positions: np.ndarray,
        rng: np.random.Generator,
        *,
        force: bool = False,
    ) -> np.ndarray:
        """Move connected UAV groups toward uncovered target clusters.

        This operator addresses a CP3-specific failure mode: moving one relay
        UAV toward an uncovered region often breaks connectivity and is undone
        by repair. Translating a connected neighborhood preserves its internal
        geometry and lets a relay chain "climb" toward coverage together.

        Binary coverage remains the primary metric. Smooth coverage potential is
        used only to choose among equal-coverage mutation proposals so progress
        toward a still-uncovered region is visible to search.
        """
        if not self.use_connected_group_mutation:
            return positions

        if (
            not force
            and rng.random()
            >= self.group_mutation_probability
        ):
            return positions

        cluster = self._cluster_centroid(
            problem,
            positions,
            rng,
        )

        if cluster is None:
            return positions

        centroid, _ = cluster
        scenario = problem.scenario
        graph = communication_graph_from_positions(
            positions,
            scenario.communication_radius,
        )

        if len(graph) == 0:
            return positions

        baseline_metrics = evaluate_positions(
            scenario,
            positions,
            self.static_objective_config,
        )
        baseline_key = (
            float(
                baseline_metrics.weighted_coverage_ratio
            ),
            (
                float(
                    soft_coverage_potential(
                        scenario,
                        positions,
                    )
                )
                if self.use_smooth_coverage_potential
                else 0.0
            ),
            float(baseline_metrics.fitness),
        )

        best = positions
        best_key = baseline_key

        anchors = self._candidate_group_anchors(
            problem,
            positions,
            centroid,
        )

        step_base = max(
            self.operator.mutation_sigma,
            0.45 * scenario.sensing_radius,
        )

        for anchor in anchors:
            lengths = nx.single_source_shortest_path_length(
                graph,
                anchor,
                cutoff=self.max_group_hops,
            )

            for hops in range(
                self.max_group_hops + 1
            ):
                group = {
                    node
                    for node, distance in lengths.items()
                    if distance <= hops
                }

                if not group:
                    continue

                group_idx = np.array(
                    sorted(group),
                    dtype=int,
                )
                group_center = np.mean(
                    positions[group_idx],
                    axis=0,
                )
                vector = centroid - group_center
                distance = float(
                    np.linalg.norm(vector)
                )

                if distance <= 1e-12:
                    continue

                direction = vector / distance

                for multiplier in (
                    0.75,
                    1.50,
                    2.25,
                ):
                    step = min(
                        distance,
                        multiplier * step_base,
                    )
                    candidate = (
                        self._boundary_preserving_translation(
                            positions,
                            group,
                            direction * step,
                            problem,
                        )
                    )

                    if np.allclose(
                        candidate,
                        positions,
                    ):
                        continue

                    candidate = self._repair_candidate(
                        problem,
                        candidate,
                    )

                    if not formation_obstacle_free(
                        candidate,
                        problem.obstacles,
                        problem.obstacle_clearance,
                    ):
                        continue

                    metrics = evaluate_positions(
                        scenario,
                        candidate,
                        self.static_objective_config,
                    )

                    if not metrics.feasible:
                        continue

                    key = (
                        float(
                            metrics.weighted_coverage_ratio
                        ),
                        (
                            float(
                                soft_coverage_potential(
                                    scenario,
                                    candidate,
                                )
                            )
                            if self.use_smooth_coverage_potential
                            else 0.0
                        ),
                        float(metrics.fitness),
                    )

                    if key > best_key:
                        best = candidate
                        best_key = key

        return best

    def _deterministic_uncovered_clusters(
        self,
        problem: ReconfigurationProblem,
        positions: np.ndarray,
        max_clusters: int = 6,
    ) -> list[np.ndarray]:
        """Build a few dense uncovered-target centroids deterministically.

        The previous guided operator sampled one uncovered cluster at random.
        For final refinement that is unnecessarily fragile: the human-edit
        counterexamples often reveal an obvious large cluster that simply was
        never sampled. Here we greedily pick the densest remaining uncovered
        region and remove its nearby targets before picking the next one.
        """
        scenario = problem.scenario
        uncovered = uncovered_target_indices(
            scenario,
            positions,
        )

        if len(uncovered) == 0:
            return []

        remaining = set(
            int(index)
            for index in uncovered
        )
        radius = 1.5 * scenario.sensing_radius
        centroids = []

        while remaining and len(centroids) < max_clusters:
            remaining_list = np.array(
                sorted(remaining),
                dtype=int,
            )
            points = scenario.targets[
                remaining_list
            ]
            weights = scenario.target_weights[
                remaining_list
            ].astype(float)

            best_seed = None
            best_mass = -1.0
            best_members = None

            for local_index, seed_index in enumerate(
                remaining_list
            ):
                distance = np.linalg.norm(
                    points - points[local_index],
                    axis=1,
                )
                mask = distance <= radius
                mass = float(
                    np.sum(weights[mask])
                )

                if mass > best_mass:
                    best_mass = mass
                    best_seed = int(seed_index)
                    best_members = remaining_list[
                        mask
                    ]

            if best_seed is None or best_members is None:
                break

            member_weights = scenario.target_weights[
                best_members
            ].astype(float)
            member_points = scenario.targets[
                best_members
            ]
            total_weight = float(
                np.sum(member_weights)
            )

            if total_weight > 0:
                centroid = np.average(
                    member_points,
                    axis=0,
                    weights=member_weights,
                )
            else:
                centroid = np.mean(
                    member_points,
                    axis=0,
                )

            centroids.append(
                np.asarray(
                    centroid,
                    dtype=float,
                )
            )

            for index in best_members:
                remaining.discard(
                    int(index)
                )

        return centroids

    def _elastic_pull_candidate(
        self,
        problem: ReconfigurationProblem,
        positions: np.ndarray,
        anchor: int,
        centroid: np.ndarray,
        hops: int,
        step: float,
    ) -> np.ndarray:
        """Pull a relay neighborhood toward a target cluster like an elastic chain.

        The anchor moves the most. UAVs farther away in graph hops move by a
        smaller fraction of the same vector, so the communication chain bends
        and translates instead of asking one UAV to jump away from its relays.
        """
        scenario = problem.scenario
        graph = communication_graph_from_positions(
            positions,
            scenario.communication_radius,
        )

        lengths = nx.single_source_shortest_path_length(
            graph,
            anchor,
            cutoff=hops,
        )

        if not lengths:
            return positions.copy()

        direction_vector = (
            centroid - positions[anchor]
        )
        distance = float(
            np.linalg.norm(direction_vector)
        )

        if distance <= 1e-12:
            return positions.copy()

        delta = (
            direction_vector
            / distance
            * min(step, distance)
        )

        candidate = positions.copy()

        for node, graph_distance in lengths.items():
            # Keep the outer relay moving too, but less aggressively. This
            # creates the "drag the chain upward" behavior visible in manual
            # counterexamples without translating the whole swarm rigidly.
            weight = (
                hops - graph_distance + 1
            ) / max(hops + 1, 1)
            candidate[node] += weight * delta

        return self._repair_candidate(
            problem,
            candidate,
        )

    def _static_refine_key(
        self,
        problem: ReconfigurationProblem,
        positions: np.ndarray,
    ) -> tuple | None:
        """Cheap pre-screen key before an exact transition simulation."""
        metrics = evaluate_positions(
            problem.scenario,
            positions,
            self.static_objective_config,
        )

        if not metrics.feasible:
            return None
        if not formation_obstacle_free(
            positions,
            problem.obstacles,
            problem.obstacle_clearance,
        ):
            return None

        return (
            float(
                metrics.weighted_coverage_ratio
            ),
            (
                float(
                    soft_coverage_potential(
                        problem.scenario,
                        positions,
                    )
                )
                if self.use_smooth_coverage_potential
                else 0.0
            ),
            float(metrics.fitness),
        )

    def _refinement_proposals(
        self,
        problem: ReconfigurationProblem,
        positions: np.ndarray,
    ) -> list[np.ndarray]:
        """Generate deterministic coordinated moves toward all major gaps."""
        scenario = problem.scenario
        centroids = self._deterministic_uncovered_clusters(
            problem,
            positions,
        )

        if not centroids:
            return []

        proposals = []
        seen = set()
        step_base = max(
            self.operator.mutation_sigma,
            0.55 * scenario.sensing_radius,
        )

        for centroid in centroids:
            anchors = self._candidate_group_anchors(
                problem,
                positions,
                centroid,
            )

            for anchor in anchors:
                for hops in range(
                    self.max_group_hops + 1
                ):
                    for multiplier in (
                        0.75,
                        1.50,
                        2.50,
                        3.50,
                    ):
                        candidate = self._elastic_pull_candidate(
                            problem,
                            positions,
                            anchor,
                            centroid,
                            hops,
                            multiplier * step_base,
                        )

                        if np.allclose(
                            candidate,
                            positions,
                        ):
                            continue

                        key = self._cache_key(
                            candidate
                        )

                        if key in seen:
                            continue

                        seen.add(key)
                        proposals.append(candidate)

        return proposals

    def _guided_generation_children(
        self,
        problem: ReconfigurationProblem,
        elite_positions: list[np.ndarray],
        occupied: set[bytes],
    ) -> list[np.ndarray]:
        """Create deterministic coverage-directed children for one generation.

        Random mutation is useful for combinations, but it can miss the same
        distant target region for an entire short run. These children expose
        every major uncovered cluster to the population on every generation.
        They remain ordinary population members and receive the same exact
        evaluation and selection as stochastic offspring.
        """
        if not self.use_guided_offspring or self.guided_offspring == 0:
            return []

        screened = []
        seen = set(occupied)

        for elite_rank, positions in enumerate(elite_positions[:2]):
            for candidate in self._refinement_proposals(problem, positions):
                key = self._cache_key(candidate)
                if key in seen:
                    continue
                seen.add(key)
                static_key = self._static_refine_key(problem, candidate)
                if static_key is None:
                    continue
                screened.append((static_key, -elite_rank, candidate))

        screened.sort(key=lambda item: item[:2], reverse=True)
        return [
            candidate
            for _, _, candidate in screened[: self.guided_offspring]
        ]

    def _deterministic_coverage_refine(
        self,
        problem: ReconfigurationProblem,
        base_positions: np.ndarray,
        base_evaluation: ReconfigurationEvaluation,
        base_transition,
        cache: dict,
    ):
        """Systematically push the best solution toward uncovered target mass.

        Final acceptance is strictly lexicographic. Any additional binary
        coverage beats any transition-cost saving. Cost matters only after
        coverage and static performance tie.

        A temporary working state may move closer to uncovered targets without
        being committed. This lets several elastic relay moves cross a binary
        coverage plateau, while the returned solution never keeps such a move
        unless the real final objective improves.
        """
        best_positions = base_positions.copy()
        best_evaluation = base_evaluation
        best_transition = base_transition

        working_positions = best_positions.copy()
        working_static_key = self._static_refine_key(
            problem,
            working_positions,
        )

        for _ in range(self.coverage_refine_rounds):
            proposals = self._refinement_proposals(
                problem,
                working_positions,
            )

            if not proposals:
                break

            screened = []

            for candidate in proposals:
                static_key = self._static_refine_key(
                    problem,
                    candidate,
                )

                if static_key is None:
                    continue

                screened.append(
                    (
                        static_key,
                        candidate,
                    )
                )

            if not screened:
                break

            screened.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            exact = []

            for static_key, candidate in screened[
                : self.coverage_refine_exact_candidates
            ]:
                evaluation, transition = (
                    self._evaluate_candidate(
                        problem,
                        candidate,
                        cache,
                    )
                )

                if not evaluation.feasible:
                    continue

                exact.append(
                    (
                        evaluation,
                        transition,
                        candidate,
                        static_key,
                    )
                )

            if not exact:
                break

            exact.sort(
                key=lambda item: self._selection_key(
                    item[0]
                ),
                reverse=True,
            )

            candidate_eval, candidate_transition, candidate_positions, _ = (
                exact[0]
            )

            if (
                self._selection_key(candidate_eval)
                > self._selection_key(best_evaluation)
            ):
                best_positions = (
                    candidate_positions.copy()
                )
                best_evaluation = candidate_eval
                best_transition = candidate_transition

                # Restart the next round from the newly committed improvement.
                working_positions = best_positions.copy()
                working_static_key = (
                    self._static_refine_key(
                        problem,
                        working_positions,
                    )
                )
                continue

            # No real objective improvement yet. Continue from the feasible
            # proposal with the best static search signal if it advances toward
            # uncovered target mass. This state is temporary only.
            search_candidate = max(
                exact,
                key=lambda item: item[3],
            )
            search_key = search_candidate[3]

            if (
                working_static_key is not None
                and search_key
                <= working_static_key
            ):
                break

            working_positions = (
                search_candidate[2].copy()
            )
            working_static_key = search_key

        return (
            best_positions,
            best_evaluation,
            best_transition,
        )

    def _coverage_probe(
        self,
        problem: ReconfigurationProblem,
        base_positions: np.ndarray,
        base_evaluation: ReconfigurationEvaluation,
        rng: np.random.Generator,
        cache: dict,
        trials: int = 8,
        plateau_steps: int = 4,
    ):
        """Try to turn smooth progress into an actual binary coverage gain.

        Potential-only moves are allowed inside a temporary probe chain, but the
        real solution is updated only when the full final ordering improves.
        This prevents the final formation from paying motion cost for a target
        it still does not cover.
        """
        best_positions = base_positions.copy()
        best_evaluation = base_evaluation
        best_transition = self._evaluate_candidate(
            problem,
            best_positions,
            cache,
        )[1]

        for _ in range(trials):
            probe = best_positions.copy()

            for _ in range(plateau_steps):
                next_probe = self._group_guided_mutation(
                    problem,
                    probe,
                    rng,
                    force=True,
                )

                if np.allclose(
                    next_probe,
                    probe,
                ):
                    break

                probe = next_probe
                evaluation, transition = (
                    self._evaluate_candidate(
                        problem,
                        probe,
                        cache,
                    )
                )

                if (
                    evaluation.feasible
                    and self._selection_key(evaluation)
                    > self._selection_key(best_evaluation)
                ):
                    best_positions = probe.copy()
                    best_evaluation = evaluation
                    best_transition = transition
                    break

        return (
            best_positions,
            best_evaluation,
            best_transition,
        )

    def _prune_unnecessary_motion(
        self,
        problem: ReconfigurationProblem,
        base_positions: np.ndarray,
        base_evaluation: ReconfigurationEvaluation,
        base_transition,
        cache: dict,
        rounds: int = 2,
    ):
        """Retract destination slots toward their assigned starts when harmless.

        A final formation can contain a UAV that moved a long way without adding
        coverage. We use the current bottleneck assignment to try shorter goal
        positions along that UAV's start-to-goal ray. A retraction is accepted
        only if the full final ordering improves, so binary coverage and static
        formation quality cannot be sacrificed just to save distance.
        """
        best_positions = base_positions.copy()
        best_evaluation = base_evaluation
        best_transition = base_transition

        if best_transition is None:
            return (
                best_positions,
                best_evaluation,
                best_transition,
            )

        for _ in range(rounds):
            improved = False

            # Recompute the assignment after every accepted change because goal
            # slots are unlabeled and the optimal UAV-slot matching may change.
            if best_transition is None:
                break

            travel = np.linalg.norm(
                best_transition.assigned_goals
                - problem.start_positions,
                axis=1,
            )
            uav_order = np.argsort(travel)[::-1]

            for uav_idx in uav_order:
                if best_transition is None:
                    break

                slot_idx = int(
                    best_transition.assignment[uav_idx]
                )
                start = problem.start_positions[
                    uav_idx
                ].copy()
                goal = best_positions[
                    slot_idx
                ].copy()

                if np.linalg.norm(goal - start) <= 1e-9:
                    continue

                # Try aggressive shortening first. The exact evaluator decides
                # whether coverage/connectivity/obstacle constraints still hold.
                for fraction in (
                    0.25,
                    0.50,
                    0.75,
                ):
                    candidate = best_positions.copy()
                    candidate[slot_idx] = (
                        start
                        + fraction * (goal - start)
                    )
                    candidate = self._repair_candidate(
                        problem,
                        candidate,
                    )

                    evaluation, transition = (
                        self._evaluate_candidate(
                            problem,
                            candidate,
                            cache,
                        )
                    )

                    if (
                        evaluation.feasible
                        and self._selection_key(evaluation)
                        > self._selection_key(
                            best_evaluation
                        )
                    ):
                        best_positions = candidate
                        best_evaluation = evaluation
                        best_transition = transition
                        improved = True
                        break

                if improved:
                    # Recompute slot assignment before considering the next UAV.
                    break

            if not improved:
                break

        return (
            best_positions,
            best_evaluation,
            best_transition,
        )

    def _post_refine_solution(
        self,
        problem: ReconfigurationProblem,
        positions: np.ndarray,
        evaluation: ReconfigurationEvaluation,
        transition,
        rng: np.random.Generator,
        cache: dict,
    ):
        """Coverage expansion followed by motion pruning on the best GA result."""
        if self.use_deterministic_refinement:
            (
                positions,
                evaluation,
                transition,
            ) = self._deterministic_coverage_refine(
                problem,
                positions,
                evaluation,
                transition,
                cache,
            )

        if self.use_coverage_probe and not self.realtime:
            positions, evaluation, transition = self._coverage_probe(
                problem,
                positions,
                evaluation,
                rng,
                cache,
            )

        if self.use_motion_pruning and not self.realtime:
            positions, evaluation, transition = (
                self._prune_unnecessary_motion(
                    problem,
                    positions,
                    evaluation,
                    transition,
                    cache,
                )
            )

        return (
            positions,
            evaluation,
            transition,
        )

    def _warm_individual(
        self,
        problem: ReconfigurationProblem,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Perturb formation A and repair it into a candidate B."""
        candidate = problem.start_positions.copy()
        candidate += rng.normal(
            0.0,
            self.operator.mutation_sigma,
            size=candidate.shape,
        )

        return self._repair_candidate(
            problem,
            candidate,
        )

    def _random_individual(
        self,
        problem: ReconfigurationProblem,
        rng: np.random.Generator,
    ) -> np.ndarray:
        candidate = self.operator._random_individual(
            problem.scenario,
            rng,
        )
        return self._repair_candidate(
            problem,
            candidate,
        )

    def _obstacle_aware_greedy_goal(
        self,
        problem: ReconfigurationProblem,
        grid_size: int = 8,
    ) -> np.ndarray | None:
        """Build a CP1-style sensing formation while excluding no-fly space.

        This is deliberately an endpoint generator, not a claim that Greedy
        alone solves reconfiguration.  The output is subsequently blended with
        a known transition-feasible CP3 formation and every blend is evaluated
        by the real backbone planner.
        """
        scenario = problem.scenario
        xs = np.linspace(0.0, scenario.width, grid_size)
        ys = np.linspace(0.0, scenario.height, grid_size)
        grid = np.array(
            [(x, y) for x in xs for y in ys],
            dtype=float,
        )

        total_weight = float(np.sum(scenario.target_weights))
        if total_weight > 0:
            centroid = np.average(
                scenario.targets,
                axis=0,
                weights=scenario.target_weights,
            )
        else:
            centroid = np.mean(scenario.targets, axis=0)

        candidates = np.vstack([
            scenario.targets,
            grid,
            np.asarray(centroid, dtype=float)[None, :],
        ])
        obstacle_free = np.array([
            formation_obstacle_free(
                point[None, :],
                problem.obstacles,
                problem.obstacle_clearance,
            )
            for point in candidates
        ])
        candidates = candidates[obstacle_free]

        if len(candidates) == 0:
            return None

        positions: list[np.ndarray] = []

        for _ in range(scenario.n_uavs):
            current = np.asarray(
                positions,
                dtype=float,
            ).reshape(-1, 2)

            if len(current) == 0:
                valid = np.ones(len(candidates), dtype=bool)
            else:
                distances = np.linalg.norm(
                    candidates[:, None, :] - current[None, :, :],
                    axis=2,
                )
                valid = (
                    np.all(
                        distances >= scenario.min_separation - 1e-9,
                        axis=1,
                    )
                    & np.any(
                        distances <= scenario.communication_radius + 1e-9,
                        axis=1,
                    )
                )

            options = candidates[valid]
            if len(options) == 0:
                return None

            scores = [
                evaluate_positions(
                    scenario,
                    np.vstack([current, point]),
                    self.static_objective_config,
                ).fitness
                for point in options
            ]
            positions.append(
                options[int(np.argmax(scores))].copy()
            )

        goal = np.asarray(positions, dtype=float)
        return (
            goal
            if formation_obstacle_free(
                goal,
                problem.obstacles,
                problem.obstacle_clearance,
            )
            else None
        )

    def _greedy_guided_obstacle_refine(
        self,
        problem: ReconfigurationProblem,
        base_positions: np.ndarray,
        base_evaluation: ReconfigurationEvaluation,
        base_transition,
        cache: dict,
        seed: int,
    ):
        """Raise coverage toward an obstacle-aware Greedy endpoint safely.

        Greedy supplies a high-coverage destination but may have no connected
        route from formation A.  Rather than returning that unsafe endpoint,
        explore short blends from a certified CP3 result to permutations of the
        Greedy slots.  The exact transition planner is the acceptance oracle.
        This makes the operation monotonic: it can only improve the normal CP3
        selection key and can never replace a feasible solution with a merely
        static one.
        """
        if (
            not self.use_obstacle_greedy_guidance
            or not problem.obstacles
            or self.obstacle_guidance_trials == 0
        ):
            return base_positions, base_evaluation, base_transition

        greedy_goal = self._obstacle_aware_greedy_goal(problem)
        if greedy_goal is None:
            return base_positions, base_evaluation, base_transition

        best_positions = base_positions.copy()
        best_evaluation = base_evaluation
        best_transition = base_transition
        rng = np.random.default_rng(seed)

        # A small fixed schedule is enough to move through coverage plateaus;
        # the planner rejects blends whose obstacle route is not viable.
        fractions = (0.10, 0.20, 0.30)

        for trial in range(self.obstacle_guidance_trials):
            guide = (
                greedy_goal
                if trial == 0
                else greedy_goal[rng.permutation(len(greedy_goal))]
            )

            for fraction in fractions:
                candidate = best_positions + fraction * (
                    guide - best_positions
                )
                candidate = self._repair_candidate(problem, candidate)
                evaluation, transition = self._evaluate_candidate(
                    problem,
                    candidate,
                    cache,
                )

                if (
                    evaluation.feasible
                    and self._selection_key(evaluation)
                    > self._selection_key(best_evaluation)
                ):
                    best_positions = candidate
                    best_evaluation = evaluation
                    best_transition = transition

        return best_positions, best_evaluation, best_transition

    def solve(
        self,
        problem: ReconfigurationProblem,
        seed: int = 0,
    ) -> tuple[ReconfigurationSolution, float]:
        start_time = time.perf_counter()
        cache = {}
        rng = np.random.default_rng(seed)
        (
            best_positions,
            best_evaluation,
            best_transition,
        ) = self._solve_single_run(problem, rng, cache)

        # A failed local controller is not proof that a formation is unreachable.
        # Try the high-coverage, obstacle-aware Greedy destination with joint
        # waypoint planning once, outside the evolutionary fitness loop.
        if problem.obstacles and self.use_global_planning:
            greedy_goal = self._obstacle_aware_greedy_goal(problem)
            if greedy_goal is not None:
                global_evaluation, global_transition = evaluate_reconfiguration(
                    problem, greedy_goal,
                    BackboneTransitionPlanner(enable_global_planning=True),
                    static_objective_config=self.static_objective_config,
                    objective_config=self.reconfiguration_objective_config,
                )
                if (
                    global_evaluation.feasible
                    and self._selection_key(global_evaluation)
                    > self._selection_key(best_evaluation)
                ):
                    best_positions = greedy_goal
                    best_evaluation = global_evaluation
                    best_transition = global_transition

        (
            best_positions,
            best_evaluation,
            best_transition,
        ) = self._greedy_guided_obstacle_refine(
            problem,
            best_positions,
            best_evaluation,
            best_transition,
            cache,
            seed,
        )

        runtime = time.perf_counter() - start_time

        return (
            ReconfigurationSolution(
                final_positions=best_positions,
                evaluation=best_evaluation,
                transition_solution=best_transition,
                algorithm=self.name,
                seed=seed,
            ),
            runtime,
        )

    def _solve_single_run(
        self,
        problem: ReconfigurationProblem,
        rng: np.random.Generator,
        cache: dict,
    ):

        population = []
        n_warm = int(
            round(
                self.population_size
                * self.warm_start_fraction
            )
        )

        # A remains an anchor candidate, but under coverage-first selection it
        # cannot beat a reachable formation with better sensing merely because
        # staying still has zero transition cost.
        population.append(problem.start_positions.copy())

        while len(population) < min(n_warm, self.population_size):
            population.append(
                self._warm_individual(problem, rng)
            )

        while len(population) < self.population_size:
            population.append(
                self._random_individual(problem, rng)
            )

        for _ in range(self.generations):
            evaluated = [
                (
                    self._estimate_candidate(problem, individual, cache),
                    None,
                )
                if self.realtime else self._evaluate_candidate(
                    problem, individual, cache,
                )
                for individual in population
            ]
            evaluations = [
                item[0]
                for item in evaluated
            ]

            order, rank_scores = self._rank_population(
                evaluations
            )

            next_population = [
                population[i].copy()
                for i in order[:self.elite_size]
            ]

            occupied = {
                self._cache_key(individual)
                for individual in next_population
            }
            # The directed children enumerate many relay/cluster combinations.
            # They are valuable in offline research mode, but dominate a
            # short-horizon control cycle even after transition simulation has
            # been deferred to finalists.
            if not self.realtime:
                next_population.extend(
                    self._guided_generation_children(
                        problem,
                        next_population,
                        occupied,
                    )[: max(0, self.population_size - len(next_population))]
                )

            while len(next_population) < self.population_size:
                a, a_score = self.operator._tournament(
                    population,
                    rank_scores,
                    rng,
                )
                b, b_score = self.operator._tournament(
                    population,
                    rank_scores,
                    rng,
                )

                child = self.operator._crossover(
                    a,
                    a_score,
                    b,
                    b_score,
                    problem.scenario,
                    rng,
                )
                child = self.operator._mutate(
                    child,
                    problem.scenario,
                    rng,
                )
                child = self._repair_candidate(
                    problem,
                    child,
                )
                if not self.realtime:
                    child = self._group_guided_mutation(
                        problem,
                        child,
                        rng,
                    )

                next_population.append(child)

            population = next_population

        evaluated = [
            (
                self._estimate_candidate(problem, individual, cache),
                None,
            )
            if self.realtime else self._evaluate_candidate(
                problem, individual, cache,
            )
            for individual in population
        ]
        evaluations = [
            item[0]
            for item in evaluated
        ]

        finalist_order = sorted(
            range(len(population)),
            key=lambda i: self._selection_key(evaluations[i]),
            reverse=True,
        )

        # Refinement belongs to this one evolutionary run. Applying it to a
        # small championship avoids overcommitting to a pre-refinement winner.
        best_positions = None
        best_evaluation = None
        best_transition = None
        seen = set()
        finalists = []

        for index in finalist_order:
            key = self._cache_key(population[index])
            if key in seen:
                continue
            seen.add(key)
            finalists.append(index)
            if len(finalists) >= self.finalist_count:
                break

        for index in finalists:
            positions = population[index].copy()
            if self.realtime:
                evaluation, transition = self._evaluate_candidate(
                    problem,
                    positions,
                    cache,
                )
            else:
                evaluation, transition = evaluated[index]
            if not evaluation.feasible:
                continue
            positions, evaluation, transition = self._post_refine_solution(
                problem,
                positions,
                evaluation,
                transition,
                rng,
                cache,
            )

            if (
                best_evaluation is None
                or self._selection_key(evaluation)
                > self._selection_key(best_evaluation)
            ):
                best_positions = positions
                best_evaluation = evaluation
                best_transition = transition

        if best_evaluation is None:
            # A lower-bound ranking cannot prove that a destination has a
            # connected path.  The current formation is always a valid,
            # zero-motion fallback, so a tight final-contender set must never
            # make a control cycle fail outright.
            fallback = problem.start_positions.copy()
            evaluation, transition = self._evaluate_candidate(
                problem,
                fallback,
                cache,
            )
            if not evaluation.feasible:
                raise RuntimeError(
                    "the current formation could not be evaluated"
                )
            best_positions = fallback
            best_evaluation = evaluation
            best_transition = transition

        return (
            best_positions,
            best_evaluation,
            best_transition,
        )
