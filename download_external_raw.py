"""Download or stage raw external datasets for the UAV benchmark.

This script prepares files consumed by prepare_external_datasets.py:
  * data/external_raw/osm_poi.csv
  * data/external_raw/opencellid.csv
  * data/external_raw/mobility.csv or data/external_raw/Geolife

OSM/POI is downloaded from Overpass API using a bounding box. OpenCellID bulk
downloads usually require a token/login, so this script can either copy a local
CSV or download a user-provided URL. GeoLife can be downloaded from Microsoft's
public download endpoint if the URL remains available.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
GEOLIFE_URL = (
    "https://download.microsoft.com/download/F/4/8/"
    "F4894AA5-FDBC-481E-9285-D5F8C4C4F039/"
    "Geolife%20Trajectories%201.3.zip"
)


def download_url(url: str, output: Path, timeout: int = 120) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "uav-swarm-optimization/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with output.open("wb") as stream:
            shutil.copyfileobj(response, stream)


def download_overpass_poi(
    *,
    bbox: tuple[float, float, float, float],
    output: Path,
    endpoint: str,
    timeout: int,
) -> None:
    south, west, north, east = bbox
    query = f"""
[out:csv(::id,::lat,::lon,amenity,shop,tourism,leisure,name;true;",")][timeout:{timeout}];
(
  node["amenity"]({south},{west},{north},{east});
  node["shop"]({south},{west},{north},{east});
  node["tourism"]({south},{west},{north},{east});
  node["leisure"]({south},{west},{north},{east});
);
out center;
"""
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "User-Agent": "uav-swarm-optimization/1.0",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=timeout + 30) as response:
        raw = response.read().decode("utf-8", errors="replace")

    rows = []
    reader = csv.DictReader(raw.splitlines())
    for row in reader:
        lat = row.get("@lat") or row.get("lat")
        lon = row.get("@lon") or row.get("lon")
        amenity = row.get("amenity") or ""
        shop = row.get("shop") or ""
        tourism = row.get("tourism") or ""
        leisure = row.get("leisure") or ""
        category = amenity or shop or tourism or leisure or "poi"
        if not lat or not lon:
            continue
        rows.append(
            dict(
                lat=lat,
                lon=lon,
                category=category,
                name=row.get("name") or "",
            )
        )

    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["lat", "lon", "category", "name"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote OSM/POI rows={len(rows)}: {output}")


def copy_or_download_opencellid(args, output_dir: Path) -> None:
    output = output_dir / "opencellid.csv"
    if args.opencellid_csv:
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.opencellid_csv, output)
        print(f"copied OpenCellID CSV: {output}")
        return
    if args.opencellid_url:
        download_url(args.opencellid_url, output, timeout=args.timeout)
        print(f"downloaded OpenCellID CSV: {output}")
        return

    print(
        "OpenCellID not downloaded. Get an API token/account and download a CSV "
        "from https://opencellid.org/downloads, then rerun with "
        "--opencellid-csv /path/to/cell_towers.csv",
        file=sys.stderr,
    )


def download_geolife(output_dir: Path, timeout: int, keep_zip: bool) -> None:
    zip_path = output_dir / "Geolife.zip"
    geolife_dir = output_dir / "Geolife"
    download_url(GEOLIFE_URL, zip_path, timeout=timeout)
    geolife_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(geolife_dir)
    if not keep_zip:
        zip_path.unlink(missing_ok=True)
    print(f"downloaded GeoLife trajectories: {geolife_dir}")


def parse_bbox(raw: str) -> tuple[float, float, float, float]:
    parts = [float(part.strip()) for part in raw.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be south,west,north,east")
    south, west, north, east = parts
    if not (south < north and west < east):
        raise argparse.ArgumentTypeError("bbox must satisfy south<north and west<east")
    return south, west, north, east


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/external_raw"))
    parser.add_argument(
        "--bbox",
        type=parse_bbox,
        default=parse_bbox("21.000,105.780,21.060,105.860"),
        help="OSM bbox as south,west,north,east. Default is a small Hanoi window.",
    )
    parser.add_argument("--skip-osm", action="store_true")
    parser.add_argument("--overpass-endpoint", default=OVERPASS_ENDPOINT)
    parser.add_argument("--opencellid-csv", type=Path)
    parser.add_argument("--opencellid-url")
    parser.add_argument("--download-geolife", action="store_true")
    parser.add_argument("--keep-geolife-zip", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_osm:
        download_overpass_poi(
            bbox=args.bbox,
            output=args.output_dir / "osm_poi.csv",
            endpoint=args.overpass_endpoint,
            timeout=args.timeout,
        )

    copy_or_download_opencellid(args, args.output_dir)

    if args.download_geolife:
        download_geolife(
            args.output_dir,
            timeout=args.timeout,
            keep_zip=args.keep_geolife_zip,
        )
    else:
        print(
            "GeoLife not downloaded. To download it automatically, rerun with "
            "--download-geolife, or download from Microsoft's GeoLife page and "
            "extract it to data/external_raw/Geolife.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
