# FEMA geographic benchmark snapshot

`fema_bay_county_shelters.geojson` is a fixed snapshot of shelter locations in
Bay County, Florida, downloaded from the Federal Emergency Management Agency
(FEMA) National Shelter System.

- Source layer: `NSS/FEMA_NSS/FeatureServer/5` (Shelter Locations)
- Query: `state='FL' AND UPPER(county_parish)='BAY'`
- Spatial reference: WGS84 / EPSG:4326
- Snapshot records: 38
- Exact-coordinate targets after deduplication: 35
- Snapshot SHA-256: `2ac7e96ea25461fe616f4fba68ed99f7f941f03fcb8fb890371808bf5b84f383`

The GeoJSON contains a `source` object with the exact service URL, query and
retrieval time. Run `download_fema_shelters.py` to refresh it deliberately.
Refreshing changes the benchmark input and therefore requires a new result
directory; it must not silently replace the snapshot used by existing results.

## Target semantics

Each unique shelter coordinate is one coverage target. Weight uses positive
`post_impact_capacity` when available, otherwise positive
`evacuation_capacity`, otherwise `1`. Records with exactly equal coordinates
are combined and their weights summed.

The benchmark converts longitude/latitude to local metres using an
equirectangular tangent plane centered on the point set. This approximation is
appropriate for this county-scale case. A 1 km margin is added around the
target extent.

The source lists candidate shelter locations, not current occupancy or proof
that a shelter is open. Results must be described as a geographic
shelter-location case study, not an operational emergency deployment.
