# Static UAV Coverage with Connectivity Constraints

This branch now focuses on one report-sized problem:

> Place a static swarm of UAVs to maximize weighted target coverage while preserving hard communication connectivity and minimum separation.

The main SOTA/report suite remains a clean static target-point benchmark. Small
obstacle and realtime mobility variants are available as explicit opt-in
experiments so they do not change the report numbers by accident.

## Algorithms kept

- `cfg_ls`: proposed Connected Frontier Greedy + topology-safe Leaf-Swap local search.
- `cfg`: ablation of the proposed method without Leaf-Swap.
- `graph_ga`: earlier graph-aware genetic algorithm baseline.
- `greedy`: connected greedy baseline.
- `jocc_cpgs`, `jocc_dpgs`: literature-inspired JOCC adapters for the common 2D target-point model.
- `exact_milp`: exact oracle for small discrete instances, using the same candidate set as CFG-LS.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

## Quick benchmark

```powershell
python run_static_coverage_study.py --seeds 20 --output-dir outputs/static_coverage_study
```

Obstacle-aware static variant:

```powershell
python run_static_coverage_study.py --seeds 5 --obstacles blocks --output-dir outputs/static_obstacles
```

## Full report suite

Smoke:

```powershell
python run_static_sota_suite.py --profile smoke
```

Report:

```powershell
python run_static_sota_suite.py --profile report --output-dir outputs/static_sota_report
```

Large SOTA run with hundreds of clean, hard, obstacle, and scalability cases:

```powershell
python run_static_sota_suite.py --profile extended --skip-exact --output-dir outputs/static_sota_extended
```

The report suite evaluates main quality, hard generated cases, obstacle/no-deploy cases, UAV scalability, target scalability, communication-radius stress, CFG/CFG-LS ablation, paired Wilcoxon tests, and small-instance optimality gaps against MILP. The extended profile uses `main_static=600`, `hard_static=400`, and `obstacle_static=300` deterministic scenarios, plus scalability sweeps.

## External Datasets

Three external case families are supported through a local conversion step:

- `osm_poi`: POI/demand points from an OSM-style CSV, with optional no-deploy obstacles.
- `opencellid`: cell-tower or cellular infrastructure points from an OpenCellID-style CSV.
- `mobility`: mobility demand points from a trajectory CSV or GeoLife `.plt` directory.
- `c2a_pose`: C2A YOLO pose labels, converted from normalized bounding-box centers.

The converter expects local raw files and writes frozen scenario JSON files:

Download/stage raw data first:

```powershell
python download_external_raw.py `
  --bbox 21.000,105.780,21.060,105.860 `
  --download-geolife `
  --output-dir data/external_raw
```

OpenCellID bulk files usually require an account/API token. Download a CSV from
`https://opencellid.org/downloads` and stage it with:

```powershell
python download_external_raw.py `
  --skip-osm `
  --opencellid-csv path/to/cell_towers.csv `
  --output-dir data/external_raw
```

Then convert the raw data:

```powershell
python prepare_external_datasets.py `
  --osm-poi data/external_raw/osm_poi.csv `
  --osm-obstacles data/external_raw/osm_obstacles.csv `
  --opencellid data/external_raw/opencellid.csv `
  --mobility-csv data/external_raw/mobility.csv `
  --geolife-dir data/external_raw/Geolife `
  --c2a-label-dir "c2a/C2A_Dataset/new_dataset3/All labels with Pose information/labels" `
  --cases 10 `
  --targets-per-case 120 `
  --output-dir datasets/external_scenarios
```

For C2A only:

```powershell
python prepare_external_datasets.py `
  --c2a-label-dir "c2a/C2A_Dataset/new_dataset3/All labels with Pose information/labels" `
  --c2a-max-cases 100 `
  --output-dir datasets/external_scenarios
```

Run SOTA on those frozen external cases separately from the generated benchmark:

```powershell
python run_static_sota_suite.py --profile smoke --skip-main --skip-exact --skip-hard --skip-obstacles --skip-sweeps --external-scenarios datasets/external_scenarios --output-dir outputs/static_sota_external
```

## Realtime Mobility Demo

Moving target nodes are visualized in a separate demo runner. Each frame updates
the target nodes along mixed circular, oscillating, figure-eight, and patrol-like
trajectories, then replans the static deployment for visual inspection.

```powershell
python run_realtime_mobility_demo.py --frames 80 --output outputs/realtime_mobility_demo.gif
```

## Tests

```powershell
python -m pytest -q
```

## Scope Boundaries

Forbidden regions are currently hard no-deploy regions for UAV positions. They
do not yet model raster maps, line-of-sight radio shadowing, path planning,
trajectory safety, online replanning, or AirSim/PX4 execution. The SOTA runner
does not include moving nodes; use `run_realtime_mobility_demo.py` for realtime
visual stress cases.
