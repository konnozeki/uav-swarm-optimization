from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from ..problem import Scenario


EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class GeographicMapMetadata:
    source_sha256: str
    source_feature_count: int
    target_count: int
    origin_latitude: float
    origin_longitude: float
    width_m: float
    height_m: float
    margin_m: float
    projection: str = "local_equirectangular_wgs84"

    def to_dict(self) -> dict:
        return {
            "source_sha256": self.source_sha256,
            "source_feature_count": self.source_feature_count,
            "target_count": self.target_count,
            "origin_latitude": self.origin_latitude,
            "origin_longitude": self.origin_longitude,
            "width_m": self.width_m,
            "height_m": self.height_m,
            "margin_m": self.margin_m,
            "projection": self.projection,
        }


def _positive_capacity(properties: dict) -> float:
    """Prefer usable post-impact capacity, then evacuation capacity."""
    for field in ("post_impact_capacity", "evacuation_capacity"):
        value = properties.get(field)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number) and number > 0:
            return number
    return 1.0


def load_weighted_wgs84_points(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load point features and aggregate exact duplicate coordinates."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") != "FeatureCollection":
        raise ValueError("geographic input must be a GeoJSON FeatureCollection")

    features = payload.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("geographic input must contain at least one feature")

    aggregated: dict[tuple[float, float], float] = {}
    for index, feature in enumerate(features):
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates")
        if geometry.get("type") != "Point" or not isinstance(coordinates, list):
            raise ValueError(f"feature {index} must contain Point geometry")
        if len(coordinates) < 2:
            raise ValueError(f"feature {index} has incomplete coordinates")
        longitude, latitude = map(float, coordinates[:2])
        if not np.isfinite(longitude) or not np.isfinite(latitude):
            raise ValueError(f"feature {index} coordinates must be finite")
        if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
            raise ValueError(f"feature {index} is outside WGS84 bounds")

        # Eight decimal places are sub-centimetre at this latitude. This only
        # merges records located at the same physical point, not nearby sites.
        key = (round(longitude, 8), round(latitude, 8))
        aggregated[key] = aggregated.get(key, 0.0) + _positive_capacity(
            feature.get("properties") or {}
        )

    points = np.asarray(list(aggregated), dtype=float)
    weights = np.asarray(list(aggregated.values()), dtype=float)
    return points, weights, len(features)


def project_wgs84_to_local_metres(
    longitude_latitude: np.ndarray,
    *,
    margin_m: float = 1_000.0,
) -> tuple[np.ndarray, float, float, float, float]:
    """Project a county-scale WGS84 point set to a local metric plane."""
    points = np.asarray(longitude_latitude, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        raise ValueError("longitude_latitude must have shape (n, 2)")
    if not np.isfinite(margin_m) or margin_m < 0:
        raise ValueError("margin_m must be finite and non-negative")

    longitude = np.radians(points[:, 0])
    latitude = np.radians(points[:, 1])
    origin_longitude = float(np.mean(longitude))
    origin_latitude = float(np.mean(latitude))

    x = EARTH_RADIUS_M * np.cos(origin_latitude) * (
        longitude - origin_longitude
    )
    y = EARTH_RADIUS_M * (latitude - origin_latitude)
    x = x - np.min(x) + margin_m
    y = y - np.min(y) + margin_m
    projected = np.column_stack([x, y])
    width = float(np.max(x) + margin_m)
    height = float(np.max(y) + margin_m)
    return (
        projected,
        width,
        height,
        float(np.degrees(origin_latitude)),
        float(np.degrees(origin_longitude)),
    )


def scenario_from_geographic_geojson(
    path: str | Path,
    *,
    name: str,
    n_uavs: int,
    sensing_radius: float,
    communication_radius: float,
    min_separation: float,
    margin_m: float = 1_000.0,
    seed: int = 0,
) -> tuple[Scenario, GeographicMapMetadata]:
    """Build a metric Scenario from weighted WGS84 point features."""
    path = Path(path)
    points, weights, source_count = load_weighted_wgs84_points(path)
    projected, width, height, origin_latitude, origin_longitude = (
        project_wgs84_to_local_metres(points, margin_m=margin_m)
    )
    metadata = GeographicMapMetadata(
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_feature_count=source_count,
        target_count=len(projected),
        origin_latitude=origin_latitude,
        origin_longitude=origin_longitude,
        width_m=width,
        height_m=height,
        margin_m=float(margin_m),
    )
    scenario = Scenario(
        name=name,
        pattern="external_geographic",
        width=width,
        height=height,
        targets=projected,
        target_weights=weights,
        n_uavs=n_uavs,
        sensing_radius=sensing_radius,
        communication_radius=communication_radius,
        min_separation=min_separation,
        seed=seed,
    )
    return scenario, metadata
