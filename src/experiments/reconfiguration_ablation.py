from __future__ import annotations

from ..baselines import StaticThenTransition
from ..proposed import TransitionAwareGA
from .reconfiguration_benchmark import run_reconfiguration_benchmark


def make_reconfiguration_ablation_algorithms(
    population_size: int = 18,
    generations: int = 20,
):
    """Build one-component-off variants around the final CP3 algorithm.

    Each variant changes exactly one search/refinement mechanism where possible,
    so the resulting benchmark can attribute gains to concrete components
    instead of comparing only broad prototype generations.
    """
    common = {
        "population_size": population_size,
        "generations": generations,
        "selection_mode": "coverage_first",
    }

    variants = [
        TransitionAwareGA(
            **common,
            name="transition_aware_full",
        ),
        TransitionAwareGA(
            **common,
            warm_start_fraction=0.0,
            name="transition_aware_no_warm",
        ),
        TransitionAwareGA(
            **common,
            use_smooth_coverage_potential=False,
            name="transition_aware_no_smooth_potential",
        ),
        TransitionAwareGA(
            **common,
            use_connected_group_mutation=False,
            name="transition_aware_no_group_mutation",
        ),
        TransitionAwareGA(
            **common,
            use_guided_offspring=False,
            name="transition_aware_no_guided_offspring",
        ),
        TransitionAwareGA(
            **common,
            use_deterministic_refinement=False,
            name="transition_aware_no_refinement",
        ),
        TransitionAwareGA(
            **common,
            use_coverage_probe=False,
            name="transition_aware_no_coverage_probe",
        ),
        TransitionAwareGA(
            **common,
            use_motion_pruning=False,
            name="transition_aware_no_motion_pruning",
        ),
        TransitionAwareGA(
            population_size=population_size,
            generations=generations,
            selection_mode="weighted_joint",
            name="transition_aware_weighted_joint",
        ),
    ]

    static = StaticThenTransition(
        population_size=population_size,
        generations=generations,
    )
    static.name = "static_then_transition"
    variants.append(static)

    return variants


def run_reconfiguration_ablation(
    problems,
    seeds,
    output_dir,
    *,
    population_size: int = 18,
    generations: int = 20,
):
    return run_reconfiguration_benchmark(
        problems=problems,
        algorithms=make_reconfiguration_ablation_algorithms(
            population_size=population_size,
            generations=generations,
        ),
        seeds=seeds,
        output_dir=output_dir,
    )
