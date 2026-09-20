from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..transition_visualization import save_transition_plot


def _save_figures(df: pd.DataFrame, output_dir: Path) -> None:
    """Save planner-level trade-off figures."""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    grouped = (
        df.groupby("algorithm", as_index=False)
        .agg(
            coverage=("weighted_coverage_ratio", "mean"),
            final_fitness=("final_fitness", "mean"),
            joint_fitness=("joint_fitness", "mean"),
            feasible_rate=("feasible", "mean"),
            transition_feasible_rate=("transition_feasible", "mean"),
            formation_time=("formation_time_sec", "mean"),
            travel=("total_travel_distance", "mean"),
            runtime=("runtime_sec", "mean"),
        )
    )

    for metric in [
        "coverage",
        "final_fitness",
        "joint_fitness",
        "feasible_rate",
        "transition_feasible_rate",
        "formation_time",
        "travel",
        "runtime",
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(grouped["algorithm"], grouped[metric])
        ax.set_ylabel(metric)
        ax.set_title(f"Joint reconfiguration: {metric}")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(
            figure_dir / f"{metric}_by_algorithm.png",
            dpi=150,
        )
        plt.close(fig)

    # A scatter plot is more informative than a single scalar score for the
    # central CP3 trade-off: final coverage versus time needed to reach it.
    feasible = df[
        df["transition_feasible"]
        & df["formation_time_sec"].notna()
    ]

    if len(feasible):
        fig, ax = plt.subplots(figsize=(7.5, 5.0))

        for algorithm, part in feasible.groupby("algorithm"):
            ax.scatter(
                part["formation_time_sec"],
                part["weighted_coverage_ratio"],
                label=algorithm,
                alpha=0.75,
            )

        ax.set_xlabel("formation time (s)")
        ax.set_ylabel("weighted coverage")
        ax.set_title("Coverage vs reconfiguration time")
        ax.grid(True, alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            figure_dir / "coverage_vs_formation_time.png",
            dpi=150,
        )
        plt.close(fig)


def run_reconfiguration_benchmark(
    problems,
    algorithms,
    seeds,
    output_dir: str | Path,
):
    """Benchmark static-then-transition against joint transition-aware search."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    trajectory_dir = output_dir / "trajectory_figures"
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    # Materialize seeds once because callers may pass range/generators.
    seed_values = list(seeds)

    for problem in problems:
        for algorithm in algorithms:
            for seed in seed_values:
                solution, runtime = algorithm.solve(
                    problem,
                    seed=seed,
                )
                metrics = solution.evaluation

                row = {
                    "problem": problem.name,
                    "pattern": problem.scenario.pattern,
                    "algorithm": algorithm.name,
                    "seed": seed,
                    "n_uavs": problem.scenario.n_uavs,
                    "n_targets": len(problem.scenario.targets),
                    "communication_radius": (
                        problem.scenario.communication_radius
                    ),
                    "sensing_radius": problem.scenario.sensing_radius,
                    "max_speed": problem.max_speed,
                    "runtime_sec": runtime,
                    **metrics.to_dict(),
                }
                rows.append(row)

                # Save one representative trajectory per problem/algorithm.
                # Full multi-seed visualization would create hundreds of files
                # while adding little diagnostic value.
                if (
                    seed == seed_values[0]
                    and solution.transition_solution is not None
                ):
                    transition_problem = problem.transition_problem(
                        solution.final_positions
                    )
                    save_transition_plot(
                        transition_problem,
                        solution.transition_solution,
                        trajectory_dir
                        / (
                            f"{problem.name}__"
                            f"{algorithm.name}.png"
                        ),
                        title=(
                            f"{problem.name} | {algorithm.name} | "
                            f"coverage="
                            f"{metrics.weighted_coverage_ratio:.3f}"
                        ),
                    )

                print(
                    f"{problem.name:32s} | "
                    f"{algorithm.name:26s} | "
                    f"seed={seed:2d} | "
                    f"cov={metrics.weighted_coverage_ratio:.3f} | "
                    f"time={metrics.formation_time_sec:7.2f} | "
                    f"tfeas={int(metrics.transition_feasible)} | "
                    f"joint={metrics.joint_fitness:.3f} | "
                    f"{runtime:.2f}s"
                )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "results.csv", index=False)

    summary = (
        df.groupby(
            [
                "problem",
                "pattern",
                "communication_radius",
                "algorithm",
            ],
            as_index=False,
        )
        .agg(
            coverage_mean=("weighted_coverage_ratio", "mean"),
            coverage_std=("weighted_coverage_ratio", "std"),
            final_fitness_mean=("final_fitness", "mean"),
            joint_fitness_mean=("joint_fitness", "mean"),
            joint_fitness_std=("joint_fitness", "std"),
            static_feasible_rate=("static_feasible", "mean"),
            transition_feasible_rate=("transition_feasible", "mean"),
            final_obstacle_free_rate=("final_obstacle_free", "mean"),
            transition_obstacle_free_rate=(
                "transition_obstacle_free",
                "mean",
            ),
            feasible_rate=("feasible", "mean"),
            formation_time_mean_sec=("formation_time_sec", "mean"),
            formation_time_std_sec=("formation_time_sec", "std"),
            travel_mean=("total_travel_distance", "mean"),
            efficiency_mean=("time_efficiency", "mean"),
            runtime_mean_sec=("runtime_sec", "mean"),
        )
    )

    summary.to_csv(output_dir / "summary.csv", index=False)
    _save_figures(df, output_dir)

    return df, summary
