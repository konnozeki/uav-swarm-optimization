from src.experiments.reconfiguration_ablation import (
    make_reconfiguration_ablation_algorithms,
)
from src.proposed import TransitionAwareGA


def test_cp3_ablation_variants_cover_final_mechanisms():
    algorithms = make_reconfiguration_ablation_algorithms(
        population_size=4,
        generations=1,
    )
    by_name = {algorithm.name: algorithm for algorithm in algorithms}

    expected = {
        "transition_aware_full",
        "transition_aware_no_warm",
        "transition_aware_no_smooth_potential",
        "transition_aware_no_group_mutation",
        "transition_aware_no_guided_offspring",
        "transition_aware_no_refinement",
        "transition_aware_no_coverage_probe",
        "transition_aware_no_motion_pruning",
        "transition_aware_weighted_joint",
        "static_then_transition",
    }
    assert set(by_name) == expected

    assert by_name["transition_aware_no_warm"].warm_start_fraction == 0.0
    assert not by_name[
        "transition_aware_no_smooth_potential"
    ].use_smooth_coverage_potential
    assert not by_name[
        "transition_aware_no_group_mutation"
    ].use_connected_group_mutation
    assert not by_name[
        "transition_aware_no_guided_offspring"
    ].use_guided_offspring
    assert not by_name[
        "transition_aware_no_refinement"
    ].use_deterministic_refinement
    assert not by_name[
        "transition_aware_no_coverage_probe"
    ].use_coverage_probe
    assert not by_name[
        "transition_aware_no_motion_pruning"
    ].use_motion_pruning
    assert (
        by_name["transition_aware_weighted_joint"].selection_mode
        == "weighted_joint"
    )


def test_transition_aware_defaults_keep_final_cp3_mechanisms_enabled():
    algorithm = TransitionAwareGA(
        population_size=4,
        generations=1,
    )

    assert algorithm.use_smooth_coverage_potential
    assert algorithm.use_connected_group_mutation
    assert algorithm.use_guided_offspring
    assert algorithm.use_deterministic_refinement
    assert algorithm.use_coverage_probe
    assert algorithm.use_motion_pruning
