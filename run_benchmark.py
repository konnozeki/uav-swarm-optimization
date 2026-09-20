import argparse
from pathlib import Path

from src.experiments.configs import get_scenarios, get_algorithms
from src.experiments.benchmark import run_benchmark


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profile",
        choices=["quick", "stress", "full"],
        default="quick",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Số seed cho mỗi scenario/algorithm.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    scenarios = get_scenarios(args.profile)
    algorithms = get_algorithms(args.profile)

    output_dir = Path("outputs") / f"benchmark_{args.profile}"

    _, summary = run_benchmark(
        scenarios=scenarios,
        algorithms=algorithms,
        seeds=range(args.seeds),
        output_dir=output_dir,
    )

    print()
    print("=== SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
