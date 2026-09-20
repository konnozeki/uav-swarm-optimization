import argparse
from pathlib import Path

from src.baselines import (
    DirectTransitionPlanner,
    EndpointGuardTransitionPlanner,
)
from src.proposed import BackboneTransitionPlanner
from src.experiments.transition_cases import (
    quick_transition_profile,
    stress_transition_profile,
)
from src.experiments.transition_benchmark import (
    run_transition_benchmark,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profile",
        choices=["quick", "stress"],
        default="quick",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.profile == "quick":
        problems = quick_transition_profile()
    else:
        problems = stress_transition_profile()

    planners = [
        DirectTransitionPlanner(),
        EndpointGuardTransitionPlanner(),
        BackboneTransitionPlanner(),
    ]

    output_dir = (
        Path("outputs")
        / f"transition_{args.profile}"
    )

    _, summary = run_transition_benchmark(
        problems=problems,
        planners=planners,
        output_dir=output_dir,
    )

    print()
    print("=== TRANSITION SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
