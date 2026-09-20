import argparse
from pathlib import Path

from src.experiments.reconfiguration_ablation import (
    run_reconfiguration_ablation,
)
from src.experiments.reconfiguration_cases import (
    stress_reconfiguration_profile,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    problems = stress_reconfiguration_profile()
    output_dir = Path("outputs") / "reconfiguration_ablation"

    _, summary = run_reconfiguration_ablation(
        problems=problems,
        seeds=range(args.seeds),
        output_dir=output_dir,
    )

    print()
    print("=== RECONFIGURATION ABLATION ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
