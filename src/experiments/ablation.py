from __future__ import annotations

from ..objectives import ObjectiveConfig
from ..baselines import VanillaGA
from ..proposed import GraphAwareGA, GraphAwareConfig
from .benchmark import run_benchmark


def make_ablation_algorithms():
    full = GraphAwareGA(
        name="graph_ga_full",
        graph_config=GraphAwareConfig(),
    )

    no_articulation = GraphAwareGA(
        name="graph_ga_no_articulation",
        graph_config=GraphAwareConfig(
            use_articulation_awareness=False,
        ),
    )

    no_connectivity_repair = GraphAwareGA(
        name="graph_ga_no_connectivity_repair",
        graph_config=GraphAwareConfig(
            use_connectivity_repair=False,
        ),
    )

    no_guided = GraphAwareGA(
        name="graph_ga_no_guided",
        graph_config=GraphAwareConfig(
            use_coverage_guided_mutation=False,
        ),
    )

    no_redundancy_mutation = GraphAwareGA(
        name="graph_ga_no_redundancy_mutation",
        graph_config=GraphAwareConfig(
            use_redundancy_aware_mutation=False,
        ),
    )

    no_redundancy_objective = GraphAwareGA(
        name="graph_ga_no_redundancy_objective",
        graph_config=GraphAwareConfig(),
        objective_config=ObjectiveConfig(
            redundancy_weight=0.0,
            connectivity_weight=2.0,
            collision_weight=2.0,
        ),
    )

    vanilla = VanillaGA()

    return [
        full,
        no_articulation,
        no_connectivity_repair,
        no_guided,
        no_redundancy_mutation,
        no_redundancy_objective,
        vanilla,
    ]


def run_ablation(scenarios, seeds, output_dir):
    return run_benchmark(
        scenarios=scenarios,
        algorithms=make_ablation_algorithms(),
        seeds=seeds,
        output_dir=output_dir,
    )
