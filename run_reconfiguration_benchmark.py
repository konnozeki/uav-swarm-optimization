import argparse
from pathlib import Path

from src.baselines import StaticThenTransition
from src.proposed import TransitionAwareGA
from src.experiments.reconfiguration_cases import (
    quick_reconfiguration_profile,
    stress_reconfiguration_profile,
)
from src.experiments.reconfiguration_benchmark import (
    run_reconfiguration_benchmark,
)


BUDGETS = {
    # Smoke is for prototype validation: import paths, feasibility logic,
    # trajectory generation and the direction of the trade-off. It is not meant
    # to support final statistical claims.
    "smoke": {
        "population_size": 8,
        "generations": 6,
    },

    # Quick is still intentionally small, but large enough to see whether the
    # transition-aware objective is behaving sensibly across all quick cases.
    "quick": {
        "population_size": 12,
        "generations": 10,
    },

    # Standard keeps the original CP3 prototype budget.
    "standard": {
        "population_size": 18,
        "generations": 20,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profile",
        choices=["quick", "stress"],
        default="quick",
    )
    parser.add_argument(
        "--budget",
        choices=list(BUDGETS),
        default="quick",
        help=(
            "Evolutionary budget. Use smoke while finishing prototype features; "
            "quick for short comparisons; standard for the original CP3 budget."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help=(
            "Optimizer seeds per problem. Joint transition-aware fitness is "
            "substantially more expensive than CP2 static fitness."
        ),
    )
    parser.add_argument(
        "--limit-problems",
        type=int,
        default=None,
        help=(
            "Optional prefix length of the selected profile. Useful for a very "
            "fast smoke run, e.g. --limit-problems 2."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.profile == "quick":
        problems = quick_reconfiguration_profile()
    else:
        problems = stress_reconfiguration_profile()

    if args.limit_problems is not None:
        if args.limit_problems <= 0:
            raise ValueError("--limit-problems must be positive")
        problems = problems[:args.limit_problems]

    budget = BUDGETS[args.budget]

    # Keep the two algorithms on exactly the same evolutionary budget. The
    # proposed method still does more work per candidate because it simulates
    # A -> B transition inside fitness; runtime is reported explicitly.
    algorithms = [
        StaticThenTransition(
            population_size=budget["population_size"],
            generations=budget["generations"],
        ),
        TransitionAwareGA(
            population_size=budget["population_size"],
            generations=budget["generations"],
        ),
    ]

    n_algorithm_runs = (
        len(problems)
        * len(algorithms)
        * args.seeds
    )

    print(
        f"profile={args.profile} | "
        f"budget={args.budget} | "
        f"problems={len(problems)} | "
        f"seeds={args.seeds} | "
        f"algorithm_runs={n_algorithm_runs} | "
        f"population={budget['population_size']} | "
        f"generations={budget['generations']}"
    )

    output_dir = (
        Path("outputs")
        / (
            f"reconfiguration_{args.profile}"
            f"_{args.budget}"
        )
    )

    _, summary = run_reconfiguration_benchmark(
        problems=problems,
        algorithms=algorithms,
        seeds=range(args.seeds),
        output_dir=output_dir,
    )

    print()
    print("=== JOINT RECONFIGURATION SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
