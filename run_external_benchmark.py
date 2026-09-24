import argparse
from pathlib import Path

from src.baselines import (
    JOCCCentralizedProjectedGradient,
    JOCCDistributedProjectedGradient,
    JOCCGradientConfig,
    LiteratureStaticThenTransition,
    R2CBufferedForceConfig,
    R2CBufferedVirtualForce,
    StaticThenTransition,
)
from src.datasets import (
    scenario_from_geographic_geojson,
    scenario_from_target_map_csv,
)
from src.experiments.reconfiguration_benchmark import (
    run_reconfiguration_benchmark,
)
from src.experiments.reproducibility import write_experiment_manifest
from src.experiments.transition_cases import ring_formation
from src.proposed import TransitionAwareGA
from src.reconfiguration import ReconfigurationProblem
from src.problem import DEFAULT_MIN_SEPARATION


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run CP3 on an x,y[,weight] CSV or a WGS84 point GeoJSON map."
        )
    )
    parser.add_argument("target_map")
    parser.add_argument(
        "--input-format",
        choices=["auto", "csv", "geographic-geojson"],
        default="auto",
    )
    parser.add_argument("--name", default="external_map")
    parser.add_argument("--width", type=float)
    parser.add_argument("--height", type=float)
    parser.add_argument("--margin-m", type=float, default=1000.0)
    parser.add_argument("--n-uavs", type=int, default=6)
    parser.add_argument("--sensing-radius", type=float)
    parser.add_argument("--communication-radius", type=float)
    parser.add_argument("--min-separation", type=float, default=DEFAULT_MIN_SEPARATION)
    parser.add_argument("--max-speed", type=float, default=40.0)
    parser.add_argument("--start-x", type=float, default=None)
    parser.add_argument("--start-y", type=float, default=None)
    parser.add_argument("--start-radius", type=float, default=None)
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--population", type=int, default=18)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--domain-iterations", type=int, default=80)
    parser.add_argument("--r2c-iterations", type=int, default=400)
    parser.add_argument(
        "--output-dir",
        default="results/cp3_final/external_benchmark",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow an exploratory run with uncommitted changes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    map_path = Path(args.target_map)
    input_format = args.input_format
    if input_format == "auto":
        input_format = (
            "geographic-geojson"
            if map_path.suffix.lower() in {".geojson", ".json"}
            else "csv"
        )

    geographic_metadata = None
    if input_format == "geographic-geojson":
        sensing_radius = args.sensing_radius or 3000.0
        communication_radius = args.communication_radius or 7000.0
        scenario, geographic_metadata = scenario_from_geographic_geojson(
            map_path,
            name=args.name,
            n_uavs=args.n_uavs,
            sensing_radius=sensing_radius,
            communication_radius=communication_radius,
            min_separation=args.min_separation,
            margin_m=args.margin_m,
        )
    else:
        if args.width is None or args.height is None:
            raise ValueError("--width and --height are required for CSV input")
        sensing_radius = args.sensing_radius or 175.0
        communication_radius = args.communication_radius or 280.0
        scenario = scenario_from_target_map_csv(
            map_path,
            name=args.name,
            width=args.width,
            height=args.height,
            n_uavs=args.n_uavs,
            sensing_radius=sensing_radius,
            communication_radius=communication_radius,
            min_separation=args.min_separation,
        )

    start_x = scenario.width / 2 if args.start_x is None else args.start_x
    start_y = scenario.height / 2 if args.start_y is None else args.start_y
    start_radius = (
        0.45 * scenario.communication_radius
        if args.start_radius is None
        else args.start_radius
    )
    start = ring_formation(
        (start_x, start_y),
        args.n_uavs,
        start_radius,
    )

    problem = ReconfigurationProblem(
        name=f"{args.name}_reconfiguration",
        scenario=scenario,
        start_positions=start,
        max_speed=args.max_speed,
        allow_reassignment=True,
    )

    algorithms = [
        StaticThenTransition(
            population_size=args.population,
            generations=args.generations,
        ),
        LiteratureStaticThenTransition(
            JOCCCentralizedProjectedGradient(
                JOCCGradientConfig(iterations=args.domain_iterations)
            )
        ),
        LiteratureStaticThenTransition(
            JOCCDistributedProjectedGradient(
                JOCCGradientConfig(
                    iterations=args.domain_iterations,
                    step_fraction=0.05,
                    connectivity_weight=0.80,
                )
            )
        ),
        LiteratureStaticThenTransition(
            R2CBufferedVirtualForce(
                R2CBufferedForceConfig(iterations=args.r2c_iterations)
            )
        ),
        TransitionAwareGA(
            population_size=args.population,
            generations=args.generations,
        ),
    ]
    seeds = list(range(args.seeds))
    output_dir = Path(args.output_dir)

    write_experiment_manifest(
        output_dir,
        experiment="cp3_external_target_map",
        seeds=seeds,
        parameters={
            "target_map": str(map_path),
            "input_format": input_format,
            "map_name": args.name,
            "width": scenario.width,
            "height": scenario.height,
            "n_uavs": args.n_uavs,
            "sensing_radius": scenario.sensing_radius,
            "communication_radius": scenario.communication_radius,
            "min_separation": args.min_separation,
            "max_speed": args.max_speed,
            "start_center": [start_x, start_y],
            "start_radius": start_radius,
            "population_size": args.population,
            "generations": args.generations,
            "domain_iterations": args.domain_iterations,
            "r2c_iterations": args.r2c_iterations,
            "algorithms": [algorithm.name for algorithm in algorithms],
            "geographic_projection": (
                geographic_metadata.to_dict()
                if geographic_metadata is not None
                else None
            ),
        },
        require_clean=not args.allow_dirty,
    )

    _, summary = run_reconfiguration_benchmark(
        problems=[problem],
        algorithms=algorithms,
        seeds=seeds,
        output_dir=output_dir,
    )

    print()
    print("=== EXTERNAL MAP SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
