"""Prepare external UAV benchmark scenarios from local raw datasets.

The script does not download data. Put raw files under data/external_raw or pass
paths explicitly, then convert them into frozen Scenario JSON files that the SOTA
runner can read.

Supported lightweight inputs:
  * OSM/POI CSV: lat, lon, optional category/name/weight columns.
  * OpenCellID CSV: lat, lon, optional radio/samples/range columns.
  * Mobility CSV: lat, lon, optional time/user/weight columns.
  * GeoLife directory: recursive .plt files are read as mobility points.
  * C2A YOLO pose labels: class x_center y_center width height pose.

Examples:
    python prepare_external_datasets.py --osm-poi data/external_raw/osm_poi.csv
    python prepare_external_datasets.py --opencellid data/external_raw/cell_towers.csv
    python prepare_external_datasets.py --mobility-csv data/external_raw/mobility.csv
    python prepare_external_datasets.py --geolife-dir data/external_raw/Geolife
    python prepare_external_datasets.py --c2a-label-dir c2a/C2A_Dataset/new_dataset3/train/labels
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from src.problem import ForbiddenRegion, Scenario
from src.scenario_io import write_scenario


DEFAULT_WIDTH = 1000.0
DEFAULT_HEIGHT = 1000.0
DEFAULT_N_UAVS = 6
DEFAULT_RS = 170.0
DEFAULT_RC = 300.0
DEFAULT_MIN_SEPARATION = 30.0

POI_CATEGORY_WEIGHTS = {
    "hospital": 3.0,
    "clinic": 2.6,
    "school": 2.4,
    "university": 2.4,
    "fire_station": 2.3,
    "police": 2.2,
    "station": 2.0,
    "bus_station": 1.8,
    "marketplace": 1.5,
    "supermarket": 1.4,
    "shop": 1.2,
}

C2A_POSE_WEIGHTS = {
    0: 2.0,  # bent
    1: 2.3,  # kneeling
    2: 3.0,  # lying
    3: 1.8,  # sitting
    4: 1.2,  # upright
}


def _first_present(row: dict, names: tuple[str, ...]) -> str | None:
    lowered = {key.lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name)
        if value not in (None, ""):
            return value
    return None


def _float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def read_latlon_csv(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            lat = _float_or_none(_first_present(row, ("lat", "latitude", "y")))
            lon = _float_or_none(_first_present(row, ("lon", "lng", "longitude", "x")))
            if lat is None or lon is None:
                continue
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                continue
            records.append(dict(row=row, lat=lat, lon=lon))
    return records


def read_geolife_dir(path: Path, max_points: int) -> list[dict]:
    records: list[dict] = []
    for file_path in sorted(path.rglob("*.plt")):
        with file_path.open("r", encoding="utf-8", errors="ignore") as stream:
            for line_id, line in enumerate(stream):
                if line_id < 6:
                    continue
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                lat = _float_or_none(parts[0])
                lon = _float_or_none(parts[1])
                if lat is None or lon is None:
                    continue
                records.append(
                    dict(
                        row={"user": file_path.parent.parent.name},
                        lat=lat,
                        lon=lon,
                    )
                )
                if len(records) >= max_points:
                    return records
    return records


def project_latlon(records: list[dict]) -> np.ndarray:
    if not records:
        raise ValueError("no valid lat/lon records found")

    lat0 = math.radians(float(np.mean([record["lat"] for record in records])))
    lon0 = math.radians(float(np.mean([record["lon"] for record in records])))
    earth_radius = 6_371_000.0
    points = []
    for record in records:
        lat = math.radians(record["lat"])
        lon = math.radians(record["lon"])
        x = earth_radius * (lon - lon0) * math.cos(lat0)
        y = earth_radius * (lat - lat0)
        points.append((x, y))
    return np.asarray(points, dtype=float)


def normalize_points(points: np.ndarray) -> np.ndarray:
    mins = np.min(points, axis=0)
    shifted = points - mins[None, :]
    span = np.max(shifted, axis=0)
    scale = max(float(np.max(span)), 1e-9)
    out = shifted / scale
    out[:, 0] *= DEFAULT_WIDTH
    out[:, 1] *= DEFAULT_HEIGHT
    out[:, 0] += 0.5 * (DEFAULT_WIDTH - float(np.max(out[:, 0])))
    out[:, 1] += 0.5 * (DEFAULT_HEIGHT - float(np.max(out[:, 1])))
    return np.clip(out, [0.0, 0.0], [DEFAULT_WIDTH, DEFAULT_HEIGHT])


def category_weight(row: dict) -> float:
    explicit = _float_or_none(_first_present(row, ("weight", "importance", "demand")))
    if explicit is not None and explicit > 0:
        return explicit

    category = (_first_present(row, ("category", "amenity", "type", "class")) or "").lower()
    return POI_CATEGORY_WEIGHTS.get(category, 1.0)


def opencellid_weight(row: dict) -> float:
    samples = _float_or_none(_first_present(row, ("samples", "sample", "measurements")))
    radio = (_first_present(row, ("radio", "technology")) or "").lower()
    weight = 1.0
    if samples is not None:
        weight += min(math.log1p(max(samples, 0.0)) / 4.0, 2.0)
    if radio in {"lte", "nr", "5g"}:
        weight += 0.5
    return weight


def mobility_weight(row: dict) -> float:
    explicit = _float_or_none(_first_present(row, ("weight", "demand", "count")))
    return explicit if explicit is not None and explicit > 0 else 1.0


def read_c2a_label_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    targets: list[tuple[float, float]] = []
    weights: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            x_center = _float_or_none(parts[1])
            y_center = _float_or_none(parts[2])
            pose_value = _float_or_none(parts[5])
            if x_center is None or y_center is None or pose_value is None:
                continue
            if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
                continue
            pose = int(pose_value)
            targets.append(
                (
                    x_center * DEFAULT_WIDTH,
                    y_center * DEFAULT_HEIGHT,
                )
            )
            weights.append(C2A_POSE_WEIGHTS.get(pose, 1.0))

    return (
        np.asarray(targets, dtype=float),
        np.asarray(weights, dtype=float),
    )


def write_c2a_cases(
    *,
    label_dir: Path,
    output_dir: Path,
    max_cases: int | None,
    min_targets: int,
) -> list[Path]:
    if min_targets < 1:
        raise ValueError("min_targets must be positive")

    written: list[Path] = []
    files = sorted(label_dir.rglob("*.txt"))
    for file_id, label_path in enumerate(files):
        targets, weights = read_c2a_label_file(label_path)
        if len(targets) < min_targets:
            continue

        scenario = Scenario(
            name=label_path.stem,
            pattern="external_c2a_pose",
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            targets=targets,
            target_weights=np.maximum(weights, 1e-6),
            n_uavs=DEFAULT_N_UAVS,
            sensing_radius=DEFAULT_RS,
            communication_radius=DEFAULT_RC,
            min_separation=DEFAULT_MIN_SEPARATION,
            seed=len(written),
        )
        path = output_dir / "c2a_pose" / f"{scenario.name}.json"
        write_scenario(path, scenario)
        written.append(path)

        if max_cases is not None and len(written) >= max_cases:
            break

    return written


def deterministic_case_indices(
    n_points: int,
    *,
    cases: int,
    targets_per_case: int,
    seed: int,
) -> Iterable[np.ndarray]:
    if n_points < targets_per_case:
        raise ValueError(
            f"need at least {targets_per_case} points, found {n_points}"
        )
    rng = np.random.default_rng(seed)
    for _ in range(cases):
        yield np.sort(
            rng.choice(
                n_points,
                size=targets_per_case,
                replace=False,
            )
        )


def write_cases(
    *,
    source: str,
    records: list[dict],
    output_dir: Path,
    cases: int,
    targets_per_case: int,
    seed: int,
    weight_fn,
    obstacles: tuple[ForbiddenRegion, ...] = (),
) -> list[Path]:
    projected = normalize_points(project_latlon(records))
    weights = np.asarray(
        [weight_fn(record["row"]) for record in records],
        dtype=float,
    )
    weights = np.maximum(weights, 1e-6)

    written: list[Path] = []
    for case_id, indices in enumerate(
        deterministic_case_indices(
            len(records),
            cases=cases,
            targets_per_case=targets_per_case,
            seed=seed,
        )
    ):
        scenario = Scenario(
            name=f"{source}_{case_id:03d}",
            pattern=f"external_{source}",
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            targets=projected[indices],
            target_weights=weights[indices],
            n_uavs=DEFAULT_N_UAVS,
            sensing_radius=DEFAULT_RS,
            communication_radius=DEFAULT_RC,
            min_separation=DEFAULT_MIN_SEPARATION,
            seed=case_id,
            forbidden_regions=obstacles,
        )
        path = output_dir / source / f"{scenario.name}.json"
        write_scenario(path, scenario)
        written.append(path)
    return written


def read_obstacles_csv(path: Path | None) -> tuple[ForbiddenRegion, ...]:
    if path is None:
        return ()
    if not path.exists():
        print(f"warning: obstacle file not found, skipping: {path}")
        return ()
    regions: list[ForbiddenRegion] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row_id, row in enumerate(reader):
            name = _first_present(row, ("name", "id")) or f"obstacle_{row_id}"
            kind = (_first_present(row, ("kind", "type")) or "").lower()
            if kind == "rectangle":
                values = [
                    _float_or_none(_first_present(row, (field,)))
                    for field in ("xmin", "ymin", "xmax", "ymax")
                ]
                if any(value is None for value in values):
                    continue
                regions.append(ForbiddenRegion.rectangle(name, *values))
            elif kind == "circle":
                center_x = _float_or_none(_first_present(row, ("x", "cx")))
                center_y = _float_or_none(_first_present(row, ("y", "cy")))
                radius = _float_or_none(_first_present(row, ("radius", "r")))
                if center_x is None or center_y is None or radius is None:
                    continue
                regions.append(ForbiddenRegion.circle(name, center_x, center_y, radius))
    return tuple(regions)


def existing_path(path: Path | None, label: str) -> Path | None:
    if path is None:
        return None
    if path.exists():
        return path
    print(f"warning: {label} not found, skipping: {path}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osm-poi", type=Path)
    parser.add_argument("--osm-obstacles", type=Path)
    parser.add_argument("--opencellid", type=Path)
    parser.add_argument("--mobility-csv", type=Path)
    parser.add_argument("--geolife-dir", type=Path)
    parser.add_argument(
        "--c2a-label-dir",
        type=Path,
        help="Directory containing C2A YOLO pose .txt labels",
    )
    parser.add_argument(
        "--c2a-max-cases",
        type=int,
        help="Optional cap on converted C2A label files",
    )
    parser.add_argument("--c2a-min-targets", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/external_scenarios"))
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--targets-per-case", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--max-geolife-points", type=int, default=250_000)
    args = parser.parse_args()

    if args.cases < 1 or args.targets_per_case < 1:
        parser.error("--cases and --targets-per-case must be positive")

    osm_poi = existing_path(args.osm_poi, "OSM/POI CSV")
    opencellid = existing_path(args.opencellid, "OpenCellID CSV")
    mobility_csv = existing_path(args.mobility_csv, "mobility CSV")
    geolife_dir = existing_path(args.geolife_dir, "GeoLife directory")
    c2a_label_dir = existing_path(args.c2a_label_dir, "C2A label directory")
    if args.c2a_max_cases is not None and args.c2a_max_cases < 1:
        parser.error("--c2a-max-cases must be positive")
    if args.c2a_min_targets < 1:
        parser.error("--c2a-min-targets must be positive")

    total = 0
    if osm_poi:
        obstacles = read_obstacles_csv(args.osm_obstacles)
        written = write_cases(
            source="osm_poi",
            records=read_latlon_csv(osm_poi),
            output_dir=args.output_dir,
            cases=args.cases,
            targets_per_case=args.targets_per_case,
            seed=args.seed,
            weight_fn=category_weight,
            obstacles=obstacles,
        )
        total += len(written)
        print(f"wrote {len(written)} OSM/POI scenarios")

    if opencellid:
        written = write_cases(
            source="opencellid",
            records=read_latlon_csv(opencellid),
            output_dir=args.output_dir,
            cases=args.cases,
            targets_per_case=args.targets_per_case,
            seed=args.seed + 1,
            weight_fn=opencellid_weight,
        )
        total += len(written)
        print(f"wrote {len(written)} OpenCellID scenarios")

    mobility_records: list[dict] = []
    if mobility_csv:
        mobility_records.extend(read_latlon_csv(mobility_csv))
    if geolife_dir:
        mobility_records.extend(
            read_geolife_dir(geolife_dir, args.max_geolife_points)
        )
    if mobility_records:
        written = write_cases(
            source="mobility",
            records=mobility_records,
            output_dir=args.output_dir,
            cases=args.cases,
            targets_per_case=args.targets_per_case,
            seed=args.seed + 2,
            weight_fn=mobility_weight,
        )
        total += len(written)
        print(f"wrote {len(written)} mobility scenarios")

    if c2a_label_dir:
        written = write_c2a_cases(
            label_dir=c2a_label_dir,
            output_dir=args.output_dir,
            max_cases=args.c2a_max_cases,
            min_targets=args.c2a_min_targets,
        )
        total += len(written)
        print(f"wrote {len(written)} C2A pose scenarios")

    if total == 0:
        parser.error(
            "no external scenarios were written; provide at least one existing "
            "raw input file with enough valid lat/lon rows"
        )
    print(f"external scenarios: {args.output_dir}")


if __name__ == "__main__":
    main()
