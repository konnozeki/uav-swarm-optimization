from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

from ..metrics import evaluate


def _save_figures(df: pd.DataFrame, output_dir: Path):
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    grouped = (
        df.groupby("algorithm", as_index=False)
        .agg(
            coverage=("weighted_coverage_ratio", "mean"),
            connected_rate=("connected", "mean"),
            feasible_rate=("feasible", "mean"),
            redundancy=("redundancy_excess", "mean"),
            runtime=("runtime_sec", "mean"),
            fitness=("fitness", "mean"),
        )
        .sort_values("fitness", ascending=False)
    )

    for metric in [
        "coverage",
        "connected_rate",
        "feasible_rate",
        "redundancy",
        "runtime",
        "fitness",
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(grouped["algorithm"], grouped[metric])
        ax.set_title(f"Mean {metric} by algorithm")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{metric}_by_algorithm.png", dpi=150)
        plt.close(fig)

    stress = (
        df.groupby(
            ["algorithm", "communication_radius"],
            as_index=False,
        )
        .agg(
            coverage=("weighted_coverage_ratio", "mean"),
            connected_rate=("connected", "mean"),
        )
    )

    if stress["communication_radius"].nunique() > 1:
        for metric in ["coverage", "connected_rate"]:
            fig, ax = plt.subplots(figsize=(8, 4.5))

            for algorithm, part in stress.groupby("algorithm"):
                part = part.sort_values("communication_radius")
                ax.plot(
                    part["communication_radius"],
                    part[metric],
                    marker="o",
                    label=algorithm,
                )

            ax.set_xlabel("communication radius")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} vs communication radius")
            ax.grid(True, alpha=0.2)
            ax.legend()
            fig.tight_layout()
            fig.savefig(
                fig_dir / f"{metric}_vs_communication_radius.png",
                dpi=150,
            )
            plt.close(fig)


def run_benchmark(
    scenarios,
    algorithms,
    seeds,
    output_dir: str | Path,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for scenario in scenarios:
        for algorithm in algorithms:
            for seed in seeds:
                solution, runtime = algorithm.solve(
                    scenario,
                    seed=seed,
                )
                metrics = evaluate(scenario, solution)

                row = {
                    "scenario": scenario.name,
                    "pattern": scenario.pattern,
                    "scenario_seed": scenario.seed,
                    "algorithm": algorithm.name,
                    "seed": seed,
                    "n_uavs": scenario.n_uavs,
                    "n_targets": len(scenario.targets),
                    "sensing_radius": scenario.sensing_radius,
                    "communication_radius": scenario.communication_radius,
                    "min_separation": scenario.min_separation,
                    "runtime_sec": runtime,
                    "positions_json": json.dumps(
                        solution.positions.tolist()
                    ),
                    **metrics.to_dict(),
                }

                rows.append(row)

                print(
                    f"{scenario.name:34s} | "
                    f"{algorithm.name:14s} | "
                    f"seed={seed:2d} | "
                    f"cov={metrics.weighted_coverage_ratio:.3f} | "
                    f"conn={int(metrics.connected)} | "
                    f"feas={int(metrics.feasible)} | "
                    f"red={metrics.redundancy_excess:.3f} | "
                    f"fit={metrics.fitness:.3f} | "
                    f"{runtime:.3f}s"
                )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "results.csv", index=False)

    summary = (
        df.groupby(
            [
                "scenario",
                "pattern",
                "n_uavs",
                "n_targets",
                "communication_radius",
                "algorithm",
            ],
            as_index=False,
        )
        .agg(
            coverage_mean=("weighted_coverage_ratio", "mean"),
            coverage_std=("weighted_coverage_ratio", "std"),
            redundancy_mean=("redundancy_excess", "mean"),
            connected_rate=("connected", "mean"),
            feasible_rate=("feasible", "mean"),
            connectivity_deficit_mean=("connectivity_deficit", "mean"),
            constraint_violation_mean=("constraint_violation", "mean"),
            collision_mean=("collision_violations", "mean"),
            fitness_mean=("fitness", "mean"),
            fitness_std=("fitness", "std"),
            runtime_mean_sec=("runtime_sec", "mean"),
            runtime_std_sec=("runtime_sec", "std"),
        )
    )

    summary.to_csv(output_dir / "summary.csv", index=False)

    _save_figures(df, output_dir)

    return df, summary
