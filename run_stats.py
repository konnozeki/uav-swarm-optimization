import argparse

from src.experiments.statistics import run_statistics


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("csv_path")
    parser.add_argument(
        "--proposed",
        default="graph_ga",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    result = run_statistics(
        args.csv_path,
        proposed=args.proposed,
    )

    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
