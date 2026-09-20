import argparse
from pathlib import Path

from src.baselines import StaticThenTransition
from src.proposed import TransitionAwareGA
from src.reconfiguration import ReconfigurationObjectiveConfig
from src.experiments.reconfiguration_cases import (
    quick_reconfiguration_profile,
    stress_reconfiguration_profile,
    showcase_reconfiguration_profile,
)
from src.plotly_transition_visualization import (
    save_transition_html,
)
from src.transition_visualization import (
    save_transition_animation,
    save_transition_plot,
)


BUDGETS = {
    "smoke": {
        "population_size": 8,
        "generations": 6,
    },
    "quick": {
        "population_size": 12,
        "generations": 10,
    },
    "standard": {
        "population_size": 36,
        "generations": 32,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one CP3 reconfiguration case and export an interactive HTML "
            "visualization. PNG is also saved; GIF is optional."
        )
    )

    parser.add_argument(
        "--profile",
        choices=["quick", "stress", "showcase"],
        default="quick",
    )
    parser.add_argument(
        "--problem-index",
        type=int,
        default=None,
        help=(
            "Zero-based index inside the selected profile. "
            "Defaults to quick[2] for the familiar split demo and to 0 for "
            "stress/showcase profiles."
        ),
    )
    parser.add_argument(
        "--algorithm",
        choices=["aware", "static", "both"],
        default="both",
    )
    parser.add_argument(
        "--budget",
        choices=list(BUDGETS),
        default="quick",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--selection-mode",
        choices=["coverage_first", "weighted_joint"],
        default="coverage_first",
        help=(
            "coverage_first prioritizes feasible sensing quality before motion "
            "cost. weighted_joint reproduces the older weighted-sum behavior."
        ),
    )
    parser.add_argument(
        "--time-weight",
        type=float,
        default=0.20,
        help=(
            "Weight of normalized formation time in reported joint_fitness and "
            "in weighted_joint selection. coverage_first uses time only as a "
            "tie-break after sensing quality."
        ),
    )
    parser.add_argument(
        "--travel-weight",
        type=float,
        default=0.05,
        help=(
            "Weight of normalized total travel distance in reported "
            "joint_fitness and in weighted_joint selection."
        ),
    )
    parser.add_argument(
        "--group-mutation-probability",
        type=float,
        default=0.45,
        help=(
            "Probability of CP3 connected-group coverage mutation for each "
            "offspring. Set 0 to disable it for comparison."
        ),
    )
    parser.add_argument(
        "--max-group-hops",
        type=int,
        default=3,
        help=(
            "Maximum communication-graph neighborhood radius moved together "
            "by group mutation."
        ),
    )
    parser.add_argument(
        "--coverage-refine-rounds",
        type=int,
        default=6,
        help=(
            "Deterministic post-GA coverage expansion rounds. "
            "Higher values search harder for coordinated relay moves."
        ),
    )
    parser.add_argument(
        "--coverage-refine-exact-candidates",
        type=int,
        default=12,
        help=(
            "Top static refinement proposals per round that receive an exact "
            "A->B transition evaluation."
        ),
    )
    parser.add_argument(
        "--finalists",
        type=int,
        default=4,
        help=(
            "Distinct members of the final population refined before the "
            "single run returns its best solution."
        ),
    )
    parser.add_argument(
        "--performance-tolerance",
        type=float,
        default=0.005,
        help=(
            "Coverage/static-fitness bucket size used by coverage_first "
            "selection. Transition cost only decides between candidates inside "
            "the same performance bucket."
        ),
    )
    parser.add_argument(
        "--subframes",
        type=int,
        default=5,
        help=(
            "Display-only interpolation frames per planner step. "
            "Higher values look smoother but create larger HTML files."
        ),
    )
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=90,
        help="Milliseconds per interactive animation frame.",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Also export the older Matplotlib GIF fallback.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="GIF frame rate when --gif is enabled.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/visualization",
    )

    return parser.parse_args()


def make_algorithms(
    name: str,
    budget: dict,
    objective_config: ReconfigurationObjectiveConfig,
    selection_mode: str,
    group_mutation_probability: float,
    max_group_hops: int,
    coverage_refine_rounds: int,
    coverage_refine_exact_candidates: int,
    finalist_count: int,
    performance_tolerance: float,
):
    """Build only the algorithms requested for this visualization run."""
    algorithms = []

    if name in {"static", "both"}:
        algorithms.append(
            StaticThenTransition(
                population_size=budget["population_size"],
                generations=budget["generations"],
                reconfiguration_objective_config=objective_config,
            )
        )

    if name in {"aware", "both"}:
        algorithms.append(
            TransitionAwareGA(
                population_size=budget["population_size"],
                generations=budget["generations"],
                reconfiguration_objective_config=objective_config,
                selection_mode=selection_mode,
                group_mutation_probability=group_mutation_probability,
                max_group_hops=max_group_hops,
                coverage_refine_rounds=coverage_refine_rounds,
                coverage_refine_exact_candidates=(
                    coverage_refine_exact_candidates
                ),
                finalist_count=finalist_count,
                performance_tolerance=performance_tolerance,
            )
        )

    return algorithms


def main():
    args = parse_args()

    if args.profile == "quick":
        problems = quick_reconfiguration_profile()
    elif args.profile == "stress":
        problems = stress_reconfiguration_profile()
    else:
        problems = showcase_reconfiguration_profile()

    if args.problem_index is None:
        problem_index = 2 if args.profile == "quick" else 0
    else:
        problem_index = args.problem_index

    if not 0 <= problem_index < len(problems):
        raise ValueError(
            f"--problem-index must be between 0 and {len(problems) - 1}"
        )

    if not 0.0 <= args.group_mutation_probability <= 1.0:
        raise ValueError(
            "--group-mutation-probability must be in [0, 1]"
        )
    if args.max_group_hops < 0:
        raise ValueError("--max-group-hops must be non-negative")
    if args.coverage_refine_rounds < 0:
        raise ValueError("--coverage-refine-rounds must be non-negative")
    if args.coverage_refine_exact_candidates <= 0:
        raise ValueError(
            "--coverage-refine-exact-candidates must be positive"
        )
    if args.finalists <= 0:
        raise ValueError("--finalists must be positive")
    if args.performance_tolerance <= 0:
        raise ValueError("--performance-tolerance must be positive")
    if args.time_weight < 0:
        raise ValueError("--time-weight must be non-negative")
    if args.travel_weight < 0:
        raise ValueError("--travel-weight must be non-negative")
    if args.subframes <= 0:
        raise ValueError("--subframes must be positive")
    if args.frame_ms <= 0:
        raise ValueError("--frame-ms must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    problem = problems[problem_index]
    budget = BUDGETS[args.budget]
    objective_config = ReconfigurationObjectiveConfig(
        time_weight=args.time_weight,
        travel_weight=args.travel_weight,
    )

    algorithms = make_algorithms(
        args.algorithm,
        budget,
        objective_config,
        args.selection_mode,
        args.group_mutation_probability,
        args.max_group_hops,
        args.coverage_refine_rounds,
        args.coverage_refine_exact_candidates,
        args.finalists,
        args.performance_tolerance,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"problem={problem.name} | "
        f"budget={args.budget} | "
        f"population={budget['population_size']} | "
        f"generations={budget['generations']} | "
        f"seed={args.seed} | "
        f"selection={args.selection_mode} | "
        f"group_mutation={args.group_mutation_probability:.2f} | "
        f"group_hops={args.max_group_hops} | "
        f"refine_rounds={args.coverage_refine_rounds} | "
        f"refine_exact={args.coverage_refine_exact_candidates} | "
        f"finalists={args.finalists} | "
        f"performance_tol={args.performance_tolerance:.4f} | "
        f"time_weight={args.time_weight:.3f} | "
        f"travel_weight={args.travel_weight:.3f}"
    )

    for algorithm in algorithms:
        solution, runtime = algorithm.solve(
            problem,
            seed=args.seed,
        )

        metrics = solution.evaluation

        print(
            f"{algorithm.name:26s} | "
            f"coverage={metrics.weighted_coverage_ratio:.3f} | "
            f"potential={metrics.coverage_potential:.3f} | "
            f"time={metrics.formation_time_sec:7.2f} | "
            f"transition_feasible={int(metrics.transition_feasible)} | "
            f"joint={metrics.joint_fitness:.3f} | "
            f"runtime={runtime:.2f}s"
        )

        if solution.transition_solution is None:
            print(
                "  no transition trajectory was produced; "
                "skipping visualization export"
            )
            continue

        transition_problem = problem.transition_problem(
            solution.final_positions
        )

        stem = (
            f"{problem.name}__"
            f"{algorithm.name}__"
            f"seed{args.seed}"
        )

        title = (
            f"{problem.name} | {algorithm.name}<br>"
            f"coverage={metrics.weighted_coverage_ratio:.3f}, "
            f"potential={metrics.coverage_potential:.3f}, "
            f"joint={metrics.joint_fitness:.3f}"
        )

        html_path = output_dir / f"{stem}.html"
        png_path = output_dir / f"{stem}.png"

        save_transition_html(
            transition_problem,
            solution.transition_solution,
            html_path,
            title=title,
            targets=problem.scenario.targets,
            sensing_radius=problem.scenario.sensing_radius,
            target_weights=problem.scenario.target_weights,
            subframes_per_step=args.subframes,
            frame_duration_ms=args.frame_ms,
        )

        # Keep one simple static artifact for reports and quick inspection.
        save_transition_plot(
            transition_problem,
            solution.transition_solution,
            png_path,
            title=title.replace("<br>", "\n"),
            targets=problem.scenario.targets,
            sensing_radius=problem.scenario.sensing_radius,
            target_weights=problem.scenario.target_weights,
        )

        print(f"  saved interactive: {html_path}")
        print(f"  saved static:      {png_path}")

        if args.gif:
            gif_path = output_dir / f"{stem}.gif"

            save_transition_animation(
                transition_problem,
                solution.transition_solution,
                gif_path,
                title=title.replace("<br>", "\n"),
                targets=problem.scenario.targets,
                sensing_radius=problem.scenario.sensing_radius,
                target_weights=problem.scenario.target_weights,
                fps=args.fps,
            )

            print(f"  saved GIF:         {gif_path}")


if __name__ == "__main__":
    main()
