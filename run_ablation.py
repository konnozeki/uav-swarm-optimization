import argparse
from pathlib import Path

from src.scenarios import stress_profile
from src.experiments.ablation import run_ablation


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Ablation dùng stress profile vì ở đây graph mechanism mới lộ rõ nhất.
    scenarios = stress_profile()

    output_dir = Path("outputs") / "ablation"

    _, summary = run_ablation(
        scenarios=scenarios,
        seeds=range(args.seeds),
        output_dir=output_dir,
    )

    print()
    print("=== ABLATION SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
