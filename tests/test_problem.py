import numpy as np
import pytest

from src.problem import ForbiddenRegion, Scenario, clip_positions, positions_in_forbidden_regions


def test_impossible_min_separation_is_rejected():
    with pytest.raises(ValueError):
        Scenario(
            name="impossible",
            pattern="test",
            width=100.0,
            height=100.0,
            targets=np.array([[50.0, 50.0]]),
            target_weights=np.ones(1),
            n_uavs=2,
            sensing_radius=10.0,
            communication_radius=10.0,
            min_separation=11.0,
        )


def test_forbidden_region_validation_rejects_bad_geometry():
    with pytest.raises(ValueError):
        ForbiddenRegion.rectangle("bad", 10.0, 0.0, 5.0, 10.0)

    with pytest.raises(ValueError):
        ForbiddenRegion.circle("bad", 0.0, 0.0, 0.0)


def test_clip_positions_projects_out_of_forbidden_regions():
    scenario = Scenario(
        name="obstacle",
        pattern="test",
        width=100.0,
        height=100.0,
        targets=np.array([[50.0, 50.0]]),
        target_weights=np.ones(1),
        n_uavs=1,
        sensing_radius=10.0,
        communication_radius=20.0,
        min_separation=0.0,
        forbidden_regions=(
            ForbiddenRegion.rectangle("block", 40.0, 40.0, 60.0, 60.0),
        ),
    )

    projected = clip_positions(np.array([[50.0, 50.0]]), scenario)

    assert not positions_in_forbidden_regions(scenario, projected)[0]
