from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..transition import evaluate_transition
from ..transition_visualization import save_transition_plot


def _save_figures(df: pd.DataFrame, output_dir: Path) -> None:
    """Save compact planner-level figures for Checkpoint 03."""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    summary = (
        df.groupby("planner", as_index=False)
        .agg(
            feasible_rate=("feasible", "mean"),
            reached_rate=("reached_goal", "mean"),
            mean_time=("formation_time_sec", "mean"),
            mean_travel=("total_travel_distance", "mean"),
            mean_efficiency=("time_efficiency", "mean"),
        )
    )

    for metric in [
        "feasible_rate",
        "reached_rate",
        "mean_time",
        "mean_travel",
        "mean_efficiency",
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(summary["planner"], summary[metric])
        ax.set_ylabel(metric)
        ax.set_title(f"Checkpoint 03: {metric}")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(
            figure_dir / f"{metric}_by_planner.png",
            dpi=150,
        )
        plt.close(fig)


def run_transition_benchmark(
    problems,
    planners,
    output_dir: str | Path,
):
    """Run every transition planner on every formation pair."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    trajectory_dir = output_dir / "trajectory_figures"
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    for problem in problems:
        for planner in planners:
            solution = planner.solve(problem)
            metrics = evaluate_transition(problem, solution)

            row = {
                "problem": problem.name,
                "planner": planner.name,
                "n_uavs": problem.n_uavs,
                "communication_radius": problem.communication_radius,
                "min_separation": problem.min_separation,
                "max_speed": problem.max_speed,
                "dt": problem.dt,
                "allow_reassignment": problem.allow_reassignment,
                **metrics.to_dict(),
            }
            rows.append(row)

            save_transition_plot(
                problem,
                solution,
                trajectory_dir
                / f"{problem.name}__{planner.name}.png",
            )

            print(
                f"{problem.name:28s} | "
                f"{planner.name:28s} | "
                f"reach={int(metrics.reached_goal)} | "
                f"feas={int(metrics.feasible)} | "
                f"time={metrics.formation_time_sec:7.2f} | "
                f"travel={metrics.total_travel_distance:8.1f} | "
                f"eff={metrics.time_efficiency:6.3f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "results.csv", index=False)

    summary = (
        df.groupby("planner", as_index=False)
        .agg(
            feasible_rate=("feasible", "mean"),
            reached_rate=("reached_goal", "mean"),
            formation_time_mean_sec=("formation_time_sec", "mean"),
            formation_time_std_sec=("formation_time_sec", "std"),
            travel_mean=("total_travel_distance", "mean"),
            efficiency_mean=("time_efficiency", "mean"),
            continuous_connectivity_mean=(
                "sampled_continuous_connected_rate",
                "mean",
            ),
            minimum_separation_mean=(
                "min_continuous_pair_distance",
                "mean",
            ),
            backbone_certified_rate=("backbone_certified", "mean"),
            obstacle_free_rate=("obstacle_free", "mean"),
            obstacle_segment_violations_mean=(
                "obstacle_segment_violations",
                "mean",
            ),
        )
    )

    summary.to_csv(output_dir / "summary.csv", index=False)
    _save_figures(df, output_dir)

    return df, summary
