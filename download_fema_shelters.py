import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


FEMA_SHELTER_LAYER = (
    "https://gis.fema.gov/arcgis/rest/services/"
    "NSS/FEMA_NSS/FeatureServer/5/query"
)
FIELDS = [
    "shelter_id",
    "shelter_name",
    "city",
    "county_parish",
    "state",
    "evacuation_capacity",
    "post_impact_capacity",
    "latitude",
    "longitude",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a reproducible FEMA shelter-location snapshot."
    )
    parser.add_argument("--state", default="FL")
    parser.add_argument("--county", default="BAY")
    parser.add_argument(
        "--output",
        default="data/external/fema_bay_county_shelters.geojson",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    state = args.state.strip().upper()
    county = args.county.strip().upper()
    parameters = {
        "where": f"state='{state}' AND UPPER(county_parish)='{county}'",
        "outFields": ",".join(FIELDS),
        "returnGeometry": "true",
        "outSR": "4326",
        "orderByFields": "shelter_id",
        "f": "geojson",
    }
    url = f"{FEMA_SHELTER_LAYER}?{urlencode(parameters)}"
    request = Request(
        url,
        headers={"User-Agent": "uav-swarm-optimization-research/1.0"},
    )
    with urlopen(request, timeout=120) as response:
        payload = json.load(response)
    if payload.get("type") != "FeatureCollection" or not payload.get("features"):
        raise RuntimeError(f"FEMA returned no shelter features for {county}, {state}")

    payload["source"] = {
        "agency": "Federal Emergency Management Agency",
        "service": FEMA_SHELTER_LAYER,
        "query": parameters["where"],
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "spatial_reference": "WGS84 / EPSG:4326",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"Saved {len(payload['features'])} FEMA shelter records to {output}")


if __name__ == "__main__":
    main()
