import argparse
from pathlib import Path

from src.baselines import StaticThenTransition
from src.proposed import TransitionAwareGA
from src.experiments.reconfiguration_ablation import (
    run_reconfiguration_ablation,
)
from src.experiments.reconfiguration_benchmark import (
    run_reconfiguration_benchmark,
)
from src.experiments.reconfiguration_cases import (
    final_reconfiguration_profile,
)
from src.experiments.reconfiguration_sensitivity import (
    run_reconfiguration_sensitivity,
)
from src.experiments.reconfiguration_statistics import (
    run_reconfiguration_statistics,
)
from src.experiments.reproducibility import (
    write_experiment_manifest,
)


DEFAULT_SEEDS = 24
DEFAULT_POPULATION = 18
DEFAULT_GENERATIONS = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the frozen CP3 final evidence pipeline."
    )
    parser.add_argument(
        "--phase",
        choices=[
            "benchmark",
            "ablation",
            "sensitivity",
            "statistics",
            "all",
        ],
        default="benchmark",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEEDS,
        help="Paired optimizer seeds. Final evidence defaults to 24.",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=DEFAULT_POPULATION,
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_GENERATIONS,
    )
    parser.add_argument(
        "--output-root",
        default="results/cp3_final",
    )
    return parser.parse_args()


def _seed_values(count: int):
    if count <= 0:
        raise ValueError("--seeds must be positive")
    return list(range(count))


def run_benchmark_phase(args, problems, seeds, root: Path):
    output_dir = root / "benchmark"
    algorithms = [
        StaticThenTransition(
            population_size=args.population,
            generations=args.generations,
        ),
        TransitionAwareGA(
            population_size=args.population,
            generations=args.generations,
            selection_mode="coverage_first",
        ),
    ]

    write_experiment_manifest(
        output_dir,
        experiment="cp3_final_benchmark",
        seeds=seeds,
        parameters={
            "population_size": args.population,
            "generations": args.generations,
            "profile": "final_reconfiguration_profile",
            "algorithms": [a.name for a in algorithms],
        },
    )

    return run_reconfiguration_benchmark(
        problems=problems,
        algorithms=algorithms,
        seeds=seeds,
        output_dir=output_dir,
    )


def run_ablation_phase(args, problems, seeds, root: Path):
    output_dir = root / "ablation"
    write_experiment_manifest(
        output_dir,
        experiment="cp3_final_ablation",
        seeds=seeds,
        parameters={
            "population_size": args.population,
            "generations": args.generations,
            "profile": "final_reconfiguration_profile",
            "design": "one_component_off",
        },
    )

    return run_reconfiguration_ablation(
        problems=problems,
        seeds=seeds,
        output_dir=output_dir,
        population_size=args.population,
        generations=args.generations,
    )


def run_sensitivity_phase(args, problems, seeds, root: Path):
    output_dir = root / "sensitivity"
    write_experiment_manifest(
        output_dir,
        experiment="cp3_final_sensitivity",
        seeds=seeds,
        parameters={
            "population_size": args.population,
            "generations": args.generations,
            "profile": "final_reconfiguration_profile",
            "design": "one_factor_at_a_time",
        },
    )

    return run_reconfiguration_sensitivity(
        problems=problems,
        seeds=seeds,
        output_dir=output_dir,
        population_size=args.population,
        generations=args.generations,
    )


def run_statistics_phase(root: Path):
    csv_path = root / "benchmark" / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} does not exist; run the benchmark phase first"
        )
    return run_reconfiguration_statistics(csv_path)


def main():
    args = parse_args()
    seeds = _seed_values(args.seeds)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    problems = final_reconfiguration_profile()

    phases = (
        ["benchmark", "ablation", "sensitivity", "statistics"]
        if args.phase == "all"
        else [args.phase]
    )

    for phase in phases:
        print()
        print(f"=== CP3 FINAL: {phase.upper()} ===")

        if phase == "benchmark":
            run_benchmark_phase(args, problems, seeds, root)
        elif phase == "ablation":
            run_ablation_phase(args, problems, seeds, root)
        elif phase == "sensitivity":
            run_sensitivity_phase(args, problems, seeds, root)
        elif phase == "statistics":
            run_statistics_phase(root)


if __name__ == "__main__":
    main()
