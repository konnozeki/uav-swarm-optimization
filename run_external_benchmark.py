import argparse
from pathlib import Path

from src.baselines import StaticThenTransition
from src.datasets import scenario_from_target_map_csv
from src.experiments.reconfiguration_benchmark import (
    run_reconfiguration_benchmark,
)
from src.experiments.reproducibility import write_experiment_manifest
from src.experiments.transition_cases import ring_formation
from src.proposed import TransitionAwareGA
from src.reconfiguration import ReconfigurationProblem


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run CP3 on an external/public target map converted to x,y[,weight] CSV."
        )
    )
    parser.add_argument("map_csv")
    parser.add_argument("--name", default="external_map")
    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--height", type=float, required=True)
    parser.add_argument("--n-uavs", type=int, default=6)
    parser.add_argument("--sensing-radius", type=float, default=175.0)
    parser.add_argument("--communication-radius", type=float, default=280.0)
    parser.add_argument("--min-separation", type=float, default=55.0)
    parser.add_argument("--max-speed", type=float, default=40.0)
    parser.add_argument("--start-x", type=float, default=None)
    parser.add_argument("--start-y", type=float, default=None)
    parser.add_argument("--start-radius", type=float, default=None)
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--population", type=int, default=18)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        default="results/cp3_final/external_benchmark",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scenario = scenario_from_target_map_csv(
        args.map_csv,
        name=args.name,
        width=args.width,
        height=args.height,
        n_uavs=args.n_uavs,
        sensing_radius=args.sensing_radius,
        communication_radius=args.communication_radius,
        min_separation=args.min_separation,
    )

    start_x = args.width / 2 if args.start_x is None else args.start_x
    start_y = args.height / 2 if args.start_y is None else args.start_y
    start_radius = (
        0.45 * args.communication_radius
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
            "map_csv": str(Path(args.map_csv)),
            "map_name": args.name,
            "width": args.width,
            "height": args.height,
            "n_uavs": args.n_uavs,
            "sensing_radius": args.sensing_radius,
            "communication_radius": args.communication_radius,
            "min_separation": args.min_separation,
            "max_speed": args.max_speed,
            "start_center": [start_x, start_y],
            "start_radius": start_radius,
            "population_size": args.population,
            "generations": args.generations,
        },
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
