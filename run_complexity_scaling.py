import argparse
from pathlib import Path

from src.experiments.complexity_scaling import run_cp3_complexity_scaling
from src.experiments.reproducibility import write_experiment_manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default="results/cp3_final/complexity_scaling",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = list(range(args.seeds))
    output_dir = Path(args.output_dir)

    write_experiment_manifest(
        output_dir,
        experiment="cp3_complexity_scaling",
        seeds=seeds,
        parameters={"design": "one_axis_at_a_time"},
    )

    _, summary = run_cp3_complexity_scaling(
        seeds=seeds,
        output_dir=output_dir,
    )

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
