from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from ..graph_ops import pairwise_distances
from ..metrics import evaluate_positions
from ..problem import Scenario, Solution
from ..static_candidates import static_candidate_points, target_weights_or_ones


@dataclass(frozen=True)
class ExactStaticMILPConfig:
    """Exact discrete oracle configuration.

    The oracle solves the same discrete candidate problem used by CFG-LS:
    choose exactly N candidate positions, maximize weighted target coverage,
    respect minimum separation, and keep the selected communication graph
    connected. Connectivity is encoded by a single-commodity flow rooted at one
    selected candidate.
    """

    grid_size: int = 7
    time_limit_sec: float = 30.0
    mip_rel_gap: float = 0.0


class ExactStaticCoverageMILP:
    """MILP oracle for small static coverage-connectivity instances."""

    name = "exact_milp"

    def __init__(
        self,
        config: ExactStaticMILPConfig = ExactStaticMILPConfig(),
    ) -> None:
        if config.grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if config.time_limit_sec <= 0:
            raise ValueError("time_limit_sec must be positive")
        if config.mip_rel_gap < 0:
            raise ValueError("mip_rel_gap must be non-negative")
        self.config = config
        self.last_result = None

    def _build_model(self, scenario: Scenario, points: np.ndarray):
        m = len(points)
        k = len(scenario.targets)
        n = scenario.n_uavs
        if n > m:
            raise ValueError("not enough candidate points for all UAVs")

        weights = target_weights_or_ones(scenario)
        target_dist = pairwise_distances(
            scenario.targets,
            points,
        )
        covers = target_dist <= scenario.sensing_radius + 1e-9

        pair_dist = pairwise_distances(points, points)
        communication = (
            pair_dist <= scenario.communication_radius + 1e-9
        )
        np.fill_diagonal(communication, False)

        # Directed communication arcs for single-commodity flow.
        arcs = [
            (i, j)
            for i in range(m)
            for j in range(m)
            if i != j and communication[i, j]
        ]
        arc_index = {
            edge: index
            for index, edge in enumerate(arcs)
        }

        # Variable layout: [x candidates | y targets | z root | flow arcs]
        x0 = 0
        y0 = m
        z0 = m + k
        f0 = 2 * m + k
        nvar = f0 + len(arcs)

        c = np.zeros(nvar, dtype=float)
        c[y0:y0 + k] = -weights

        integrality = np.zeros(nvar, dtype=int)
        integrality[x0:x0 + m] = 1
        integrality[y0:y0 + k] = 1
        integrality[z0:z0 + m] = 1

        lower = np.zeros(nvar, dtype=float)
        upper = np.ones(nvar, dtype=float)
        if arcs:
            upper[f0:] = max(n - 1, 0)

        rows = []
        cols = []
        vals = []
        lb = []
        ub = []

        def add_row(entries, lo=-np.inf, hi=np.inf):
            row = len(lb)
            for column, value in entries:
                rows.append(row)
                cols.append(column)
                vals.append(float(value))
            lb.append(float(lo))
            ub.append(float(hi))

        # Select exactly N positions and exactly one selected root.
        add_row(
            [(x0 + i, 1.0) for i in range(m)],
            n,
            n,
        )
        add_row(
            [(z0 + i, 1.0) for i in range(m)],
            1.0,
            1.0,
        )
        for i in range(m):
            add_row(
                [(z0 + i, 1.0), (x0 + i, -1.0)],
                hi=0.0,
            )

        # A target can be marked covered only when some selected point covers it.
        for target in range(k):
            entries = [(y0 + target, 1.0)]
            entries.extend(
                (x0 + i, -1.0)
                for i in np.flatnonzero(covers[target])
            )
            add_row(entries, hi=0.0)

        # Candidate pairs that violate collision clearance cannot both be chosen.
        for i in range(m):
            for j in range(i + 1, m):
                if pair_dist[i, j] < scenario.min_separation - 1e-9:
                    add_row(
                        [(x0 + i, 1.0), (x0 + j, 1.0)],
                        hi=1.0,
                    )

        # Connectivity: root emits N-1 units, every other selected node consumes
        # one. Unselected nodes have zero balance and cannot carry flow because
        # each arc is capacity-gated by both endpoint selection variables.
        outgoing = [[] for _ in range(m)]
        incoming = [[] for _ in range(m)]
        for arc_id, (i, j) in enumerate(arcs):
            outgoing[i].append(arc_id)
            incoming[j].append(arc_id)

        for i in range(m):
            entries = [(x0 + i, 1.0), (z0 + i, -float(n))]
            entries.extend((f0 + a, 1.0) for a in outgoing[i])
            entries.extend((f0 + a, -1.0) for a in incoming[i])
            add_row(entries, 0.0, 0.0)

        capacity = float(max(n - 1, 0))
        for arc_id, (i, j) in enumerate(arcs):
            flow_var = f0 + arc_id
            add_row(
                [(flow_var, 1.0), (x0 + i, -capacity)],
                hi=0.0,
            )
            add_row(
                [(flow_var, 1.0), (x0 + j, -capacity)],
                hi=0.0,
            )

        matrix = coo_matrix(
            (vals, (rows, cols)),
            shape=(len(lb), nvar),
        ).tocsr()

        return (
            c,
            integrality,
            Bounds(lower, upper),
            LinearConstraint(
                matrix,
                np.asarray(lb, dtype=float),
                np.asarray(ub, dtype=float),
            ),
        )

    def solve(
        self,
        scenario: Scenario,
        seed: int = 0,
    ) -> tuple[Solution, float]:
        _ = seed
        started = time.perf_counter()
        points = static_candidate_points(
            scenario,
            self.config.grid_size,
        )
        c, integrality, bounds, constraint = self._build_model(
            scenario,
            points,
        )

        result = milp(
            c=c,
            integrality=integrality,
            bounds=bounds,
            constraints=constraint,
            options={
                "time_limit": self.config.time_limit_sec,
                "mip_rel_gap": self.config.mip_rel_gap,
                "presolve": True,
            },
        )
        self.last_result = result
        runtime = time.perf_counter() - started

        if not result.success or result.x is None:
            raise RuntimeError(
                "exact MILP did not prove an optimal solution: "
                f"status={result.status}, message={result.message}"
            )

        m = len(points)
        chosen = np.flatnonzero(result.x[:m] > 0.5)
        if len(chosen) != scenario.n_uavs:
            raise RuntimeError(
                "exact MILP returned an invalid number of selected candidates"
            )

        positions = points[chosen]
        metrics = evaluate_positions(scenario, positions)
        if not metrics.feasible:
            raise RuntimeError(
                "exact MILP returned a formation that fails repository validation"
            )

        return (
            Solution(
                positions,
                algorithm=self.name,
                seed=seed,
            ),
            runtime,
        )
