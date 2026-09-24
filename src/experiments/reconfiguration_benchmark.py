from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd

from ..transition_visualization import save_transition_plot


RESULT_KEY = ["problem", "algorithm", "seed"]


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Replace a CSV only after its temporary copy is fully written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(path)


def _load_completed_results(path: Path, *, resume: bool) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if not resume:
        raise FileExistsError(
            f"{path} already exists; use a new output directory or enable resume"
        )

    existing = pd.read_csv(path)
    missing = set(RESULT_KEY) - set(existing.columns)
    if missing:
        raise ValueError(
            f"cannot resume {path}: missing key columns {sorted(missing)}"
        )
    duplicated = existing.duplicated(RESULT_KEY, keep=False)
    if duplicated.any():
        keys = existing.loc[duplicated, RESULT_KEY].to_dict("records")
        raise ValueError(f"cannot resume {path}: duplicated result keys {keys[:5]}")
    return existing


def _record_failure(path: Path, key: dict, error: Exception) -> None:
    records = []
    if path.exists():
        records = json.loads(path.read_text(encoding="utf-8"))
    records.append({
        **key,
        "error_type": type(error).__name__,
        "error": str(error),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    })
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(records, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


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
    *,
    resume: bool = True,
):
    """Benchmark algorithms and persist every completed run for safe resume."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.csv"
    failures_path = output_dir / "failures.json"
    existing = _load_completed_results(results_path, resume=resume)
    rows = existing.to_dict("records")
    completed = {
        (str(row["problem"]), str(row["algorithm"]), int(row["seed"]))
        for row in rows
    }
    trajectory_dir = output_dir / "trajectory_figures"
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    # Materialize inputs once because callers may pass ranges/generators.
    problems = list(problems)
    algorithms = list(algorithms)
    seed_values = list(seeds)
    if not seed_values:
        raise ValueError("seeds must contain at least one value")
    problem_names = [problem.name for problem in problems]
    algorithm_names = [algorithm.name for algorithm in algorithms]
    if len(set(problem_names)) != len(problem_names):
        raise ValueError("problem names must be unique")
    if len(set(algorithm_names)) != len(algorithm_names):
        raise ValueError("algorithm names must be unique")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("seed values must be unique")
    expected = {
        (problem_name, algorithm_name, int(seed))
        for problem_name in problem_names
        for algorithm_name in algorithm_names
        for seed in seed_values
    }
    unexpected = completed - expected
    if unexpected:
        raise ValueError(
            "cannot resume results containing cases outside this experiment: "
            f"{sorted(unexpected)[:5]}"
        )

    for problem in problems:
        for algorithm in algorithms:
            for seed in seed_values:
                result_key = (problem.name, algorithm.name, int(seed))
                if result_key in completed:
                    print(
                        f"{problem.name:32s} | {algorithm.name:26s} | "
                        f"seed={seed:2d} | resumed"
                    )
                    continue

                try:
                    solution, runtime = algorithm.solve(
                        problem,
                        seed=seed,
                    )
                except Exception as error:
                    _record_failure(
                        failures_path,
                        dict(zip(RESULT_KEY, result_key)),
                        error,
                    )
                    raise
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

                rows.append(row)
                completed.add(result_key)
                _atomic_write_csv(pd.DataFrame(rows), results_path)

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

    missing = expected - completed
    if missing:
        raise RuntimeError(
            f"benchmark finished with {len(missing)} missing result rows"
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(RESULT_KEY).reset_index(drop=True)
    _atomic_write_csv(df, results_path)

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

    _atomic_write_csv(summary, output_dir / "summary.csv")
    _save_figures(df, output_dir)

    return df, summary
