from __future__ import annotations

from ..baselines import StaticThenTransition
from ..proposed import TransitionAwareGA
from .reconfiguration_benchmark import run_reconfiguration_benchmark


def make_reconfiguration_ablation_algorithms():
    """Build compact CP3 variants around the final prototype semantics.

    Variants:

    - full: coverage-first selection + temporal warm start;
    - no warm: same hierarchy without warm initialization around formation A;
    - weighted joint: older scalar trade-off between sensing and motion cost;
    - static baseline: CP2 chooses B without seeing A -> B transition at all.

    This keeps the ablation aligned with the final CP3 algorithm rather than
    testing time/travel weights that coverage-first selection no longer uses as
    primary selection pressure.
    """
    full = TransitionAwareGA(
        name="transition_aware_full",
        selection_mode="coverage_first",
    )

    no_warm = TransitionAwareGA(
        warm_start_fraction=0.0,
        selection_mode="coverage_first",
        name="transition_aware_no_warm",
    )

    weighted_joint = TransitionAwareGA(
        selection_mode="weighted_joint",
        name="transition_aware_weighted_joint",
    )

    static = StaticThenTransition(
        population_size=18,
        generations=20,
    )
    static.name = "static_then_transition"

    return [
        full,
        no_warm,
        weighted_joint,
        static,
    ]


def run_reconfiguration_ablation(
    problems,
    seeds,
    output_dir,
):
    return run_reconfiguration_benchmark(
        problems=problems,
        algorithms=make_reconfiguration_ablation_algorithms(),
        seeds=seeds,
        output_dir=output_dir,
    )
