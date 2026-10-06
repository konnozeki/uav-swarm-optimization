import numpy as np

from src.problem import ForbiddenRegion, Scenario
from src.scenario_io import read_scenario, write_scenario


def test_scenario_json_roundtrip(tmp_path):
    scenario = Scenario(
        name="external_case",
        pattern="external_osm_poi",
        width=1000.0,
        height=1000.0,
        targets=np.array([
            [100.0, 120.0],
            [300.0, 320.0],
            [500.0, 520.0],
        ]),
        target_weights=np.array([1.0, 2.0, 3.0]),
        n_uavs=2,
        sensing_radius=150.0,
        communication_radius=280.0,
        min_separation=20.0,
        seed=4,
        forbidden_regions=(
            ForbiddenRegion.rectangle("block", 200.0, 200.0, 260.0, 300.0),
            ForbiddenRegion.circle("disc", 700.0, 700.0, 40.0),
        ),
    )
    path = tmp_path / "scenario.json"

    write_scenario(path, scenario)
    restored = read_scenario(path)

    assert restored.name == scenario.name
    assert restored.pattern == scenario.pattern
    assert restored.seed == scenario.seed
    assert np.allclose(restored.targets, scenario.targets)
    assert np.allclose(restored.target_weights, scenario.target_weights)
    assert len(restored.forbidden_regions) == 2
