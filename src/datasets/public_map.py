from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..problem import Scenario


REQUIRED_COLUMNS = {"x", "y"}


def load_target_map_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a public/external target map from a simple interchange CSV.

    Required columns:
        x, y

    Optional column:
        weight

    Dataset-specific adapters (for example a RescueNet/C2A preprocessing step)
    should convert their native annotation format into this interchange schema
    instead of coupling the optimizer to one dataset layout.
    """
    path = Path(path)
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"target map is missing required columns: {sorted(missing)}"
        )

    positions = df[["x", "y"]].to_numpy(dtype=float)

    if "weight" in df.columns:
        weights = df["weight"].to_numpy(dtype=float)
    else:
        weights = np.ones(len(df), dtype=float)

    if len(positions) == 0:
        raise ValueError("target map must contain at least one target")
    if not np.all(np.isfinite(positions)):
        raise ValueError("target coordinates must be finite")
    if not np.all(np.isfinite(weights)):
        raise ValueError("target weights must be finite")
    if np.any(weights < 0):
        raise ValueError("target weights must be non-negative")
    if float(np.sum(weights)) <= 0:
        raise ValueError("target weights must have positive total mass")

    return positions, weights


def scenario_from_target_map_csv(
    path: str | Path,
    *,
    name: str,
    width: float,
    height: float,
    n_uavs: int,
    sensing_radius: float,
    communication_radius: float,
    min_separation: float,
    seed: int = 0,
) -> Scenario:
    """Build a Scenario from an external/public target-map CSV."""
    targets, weights = load_target_map_csv(path)

    if np.any(targets[:, 0] < 0) or np.any(targets[:, 0] > width):
        raise ValueError("target x coordinate lies outside map bounds")
    if np.any(targets[:, 1] < 0) or np.any(targets[:, 1] > height):
        raise ValueError("target y coordinate lies outside map bounds")

    return Scenario(
        name=name,
        pattern="external",
        width=width,
        height=height,
        targets=targets,
        target_weights=weights,
        n_uavs=n_uavs,
        sensing_radius=sensing_radius,
        communication_radius=communication_radius,
        min_separation=min_separation,
        seed=seed,
    )
