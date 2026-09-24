import json

import numpy as np

from src.datasets import (
    load_weighted_wgs84_points,
    project_wgs84_to_local_metres,
    scenario_from_geographic_geojson,
)


def _write_geojson(path):
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-85.0, 30.0]},
                "properties": {
                    "post_impact_capacity": 10,
                    "evacuation_capacity": 100,
                },
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-85.0, 30.0]},
                "properties": {"evacuation_capacity": 20},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-84.99, 30.01]},
                "properties": {},
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_geographic_loader_aggregates_duplicate_sites_and_capacity(tmp_path):
    path = tmp_path / "points.geojson"
    _write_geojson(path)

    points, weights, source_count = load_weighted_wgs84_points(path)

    assert source_count == 3
    assert points.shape == (2, 2)
    assert np.allclose(weights, [30.0, 1.0])


def test_local_projection_uses_metres_and_margin():
    points = np.array([[0.0, 0.0], [1.0, 0.0]])

    projected, width, height, origin_lat, origin_lon = (
        project_wgs84_to_local_metres(points, margin_m=100.0)
    )

    assert np.isclose(projected[1, 0] - projected[0, 0], 111_195, atol=2)
    assert np.allclose(projected[:, 1], 100.0)
    assert np.isclose(width, 111_395, atol=2)
    assert height == 200.0
    assert origin_lat == 0.0
    assert origin_lon == 0.5


def test_geographic_scenario_records_projection_metadata(tmp_path):
    path = tmp_path / "points.geojson"
    _write_geojson(path)

    scenario, metadata = scenario_from_geographic_geojson(
        path,
        name="geographic_test",
        n_uavs=2,
        sensing_radius=500.0,
        communication_radius=2_000.0,
        min_separation=3.0,
        margin_m=250.0,
    )

    assert scenario.pattern == "external_geographic"
    assert len(scenario.targets) == 2
    assert metadata.source_feature_count == 3
    assert metadata.target_count == 2
    assert len(metadata.source_sha256) == 64
    assert np.all(scenario.targets >= 250.0)


def test_bundled_fema_snapshot_is_valid():
    scenario, metadata = scenario_from_geographic_geojson(
        "data/external/fema_bay_county_shelters.geojson",
        name="fema_bay_county",
        n_uavs=6,
        sensing_radius=3_000.0,
        communication_radius=7_000.0,
        min_separation=3.0,
    )

    assert metadata.source_feature_count == 38
    assert metadata.target_count == 35
    assert np.isclose(scenario.target_weights.sum(), 18_647.0)
    assert scenario.width > 40_000.0
    assert scenario.height > 30_000.0
