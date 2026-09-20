import argparse

from src.experiments.reconfiguration_statistics import (
    run_reconfiguration_statistics,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("csv_path")
    parser.add_argument(
        "--proposed",
        default="transition_aware_ga",
    )
    parser.add_argument(
        "--baseline",
        default="static_then_transition",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    continuous, feasibility = run_reconfiguration_statistics(
        args.csv_path,
        proposed=args.proposed,
        baseline=args.baseline,
    )

    print("=== PAIRED CONTINUOUS METRICS ===")
    print(continuous.to_string(index=False))
    print()
    print("=== TRANSITION FEASIBILITY ===")
    print(feasibility.to_string(index=False))


if __name__ == "__main__":
    main()
