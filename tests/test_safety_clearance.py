import numpy as np

from src.experiments.reconfiguration_cases import (
    quick_reconfiguration_profile, stress_reconfiguration_profile,
    showcase_reconfiguration_profile,
)
from src.experiments.transition_cases import quick_transition_profile, stress_transition_profile
from src.metrics import evaluate_positions
from src.scenarios import make_scenario
from src.transition import formation_collision_free, segment_collision_free


def test_profiles_use_three_metre_collision_clearance():
    for factory in (quick_reconfiguration_profile, stress_reconfiguration_profile,
                    showcase_reconfiguration_profile):
        for problem in factory():
            assert problem.scenario.min_separation == 3.0
            assert problem.transition_problem(problem.start_positions).min_separation == 3.0
    for factory in (quick_transition_profile, stress_transition_profile):
        assert all(p.min_separation == 3.0 for p in factory())
    assert make_scenario(pattern='uniform', seed=0, n_uavs=2, n_targets=10,
                         communication_radius=150).min_separation == 3.0


def test_compact_formation_and_transit_allowed_but_under_three_metres_rejected():
    scenario = make_scenario(pattern='uniform', seed=0, n_uavs=2, n_targets=10,
                             communication_radius=150)
    compact = np.array([[100., 100.], [103., 100.]])
    assert evaluate_positions(scenario, compact).feasible
    assert segment_collision_free(compact, compact + [30., 20.], 3.0)
    too_close = np.array([[100., 100.], [102.9, 100.]])
    assert not evaluate_positions(scenario, too_close).feasible
    assert not segment_collision_free(too_close, too_close + [30., 20.], 3.0)
    # Safe endpoints must not hide a collision between waypoints.
    swapped = compact[::-1].copy()
    assert formation_collision_free(swapped, 3.0)
    assert not segment_collision_free(compact, swapped, 3.0)
