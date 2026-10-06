from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .problem import ForbiddenRegion, Scenario


SCENARIO_SCHEMA_VERSION = 1


def forbidden_region_to_dict(region: ForbiddenRegion) -> dict:
    data = dict(
        name=region.name,
        kind=region.kind,
    )
    if region.kind == "rectangle":
        data["bounds"] = list(region.bounds or ())
    else:
        data["center"] = list(region.center or ())
        data["radius"] = region.radius
    return data


def forbidden_region_from_dict(data: dict) -> ForbiddenRegion:
    kind = data["kind"]
    if kind == "rectangle":
        return ForbiddenRegion.rectangle(
            str(data.get("name", "rectangle")),
            *map(float, data["bounds"]),
        )
    if kind == "circle":
        center = data["center"]
        return ForbiddenRegion.circle(
            str(data.get("name", "circle")),
            float(center[0]),
            float(center[1]),
            float(data["radius"]),
        )
    raise ValueError(f"unknown forbidden region kind: {kind}")


def scenario_to_dict(scenario: Scenario) -> dict:
    return dict(
        schema_version=SCENARIO_SCHEMA_VERSION,
        name=scenario.name,
        pattern=scenario.pattern,
        width=scenario.width,
        height=scenario.height,
        targets=scenario.targets.tolist(),
        target_weights=scenario.target_weights.tolist(),
        n_uavs=scenario.n_uavs,
        sensing_radius=scenario.sensing_radius,
        communication_radius=scenario.communication_radius,
        min_separation=scenario.min_separation,
        seed=scenario.seed,
        forbidden_regions=[
            forbidden_region_to_dict(region)
            for region in scenario.forbidden_regions
        ],
    )


def scenario_from_dict(data: dict) -> Scenario:
    version = int(data.get("schema_version", 1))
    if version != SCENARIO_SCHEMA_VERSION:
        raise ValueError(f"unsupported scenario schema_version: {version}")

    return Scenario(
        name=str(data["name"]),
        pattern=str(data["pattern"]),
        width=float(data["width"]),
        height=float(data["height"]),
        targets=np.asarray(data["targets"], dtype=float),
        target_weights=np.asarray(data["target_weights"], dtype=float),
        n_uavs=int(data["n_uavs"]),
        sensing_radius=float(data["sensing_radius"]),
        communication_radius=float(data["communication_radius"]),
        min_separation=float(data["min_separation"]),
        seed=int(data.get("seed", 0)),
        forbidden_regions=tuple(
            forbidden_region_from_dict(region)
            for region in data.get("forbidden_regions", [])
        ),
    )


def write_scenario(path: Path, scenario: Scenario) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            scenario_to_dict(scenario),
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")


def read_scenario(path: Path) -> Scenario:
    with path.open("r", encoding="utf-8") as stream:
        return scenario_from_dict(json.load(stream))


def read_scenarios(paths: list[Path]) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*.json")):
                scenarios.append(read_scenario(child))
        else:
            scenarios.append(read_scenario(path))
    return scenarios
