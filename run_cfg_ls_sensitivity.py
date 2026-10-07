"""Sensitivity study for CFG-LS hyperparameters.

The study changes one CFG-LS parameter at a time around the default report
configuration and records coverage, feasibility, redundancy, and runtime.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_static_coverage_study import make_scenario
from run_static_sota_suite import (
    BASE_K,
    BASE_MIN_SEPARATION,
    BASE_N,
    BASE_RC,
    BASE_RS,
)
from src.metrics import evaluate
from src.proposed import ConnectedFrontierConfig, ConnectedFrontierLeafSwap


DEFAULT_PATTERNS = ("uniform", "islands", "corner_clusters")
PROFILE_SEEDS = {
    "smoke": 2,
    "report": 10,
    "extended": 30,
}
SENSITIVITY_VALUES = {
    "grid_size": (12, 15, 18, 21, 24),
    "local_rounds": (0, 4, 8, 12, 20),
    "multi_start_roots": (1, 3, 5, 8),
    "lookahead_depth": (1, 2, 3),
    "lookahead_branch": (2, 4, 6, 8),
    "lookahead_beam": (1, 2, 4),
    "potential_weight": (0.0, 0.06, 0.12, 0.18, 0.24),
    "redundancy_weight": (0.0, 0.05, 0.10, 0.15, 0.20),
    "connectivity_bonus": (0.0, 0.005, 0.01, 0.02, 0.04),
}


def config_with(parameter: str, value: int | float) -> ConnectedFrontierConfig:
    kwargs = ConnectedFrontierConfig().__dict__.copy()
    kwargs[parameter] = value
    return ConnectedFrontierConfig(**kwargs)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    keys = sorted({
        (row["parameter"], row["value"], row["pattern"])
        for row in rows
    }, key=lambda item: (item[0], float(item[1]), item[2]))
    summary: list[dict] = []

    for parameter, value, pattern in keys:
        group = [
            row
            for row in rows
            if (
                row["parameter"] == parameter
                and row["value"] == value
                and row["pattern"] == pattern
            )
        ]
        coverage = np.asarray([row["coverage"] for row in group], dtype=float)
        runtime_ms = np.asarray(
            [row["runtime_sec"] * 1000.0 for row in group],
            dtype=float,
        )
        summary.append(
            dict(
                parameter=parameter,
                value=value,
                pattern=pattern,
                runs=len(group),
                coverage_mean=float(np.mean(coverage)),
                coverage_std=float(np.std(coverage)),
                redundancy_mean=float(np.mean([
                    row["redundancy"]
                    for row in group
                ])),
                feasible_rate=float(np.mean([
                    row["feasible"]
                    for row in group
                ])),
                runtime_mean_ms=float(np.mean(runtime_ms)),
                runtime_median_ms=float(np.median(runtime_ms)),
                runtime_p95_ms=float(np.percentile(runtime_ms, 95)),
            )
        )

    return summary


def summarize_overall(rows: list[dict]) -> list[dict]:
    keys = sorted({
        (row["parameter"], row["value"])
        for row in rows
    }, key=lambda item: (item[0], float(item[1])))
    summary: list[dict] = []

    for parameter, value in keys:
        group = [
            row
            for row in rows
            if row["parameter"] == parameter and row["value"] == value
        ]
        coverage = np.asarray([row["coverage"] for row in group], dtype=float)
        runtime_ms = np.asarray(
            [row["runtime_sec"] * 1000.0 for row in group],
            dtype=float,
        )
        summary.append(
            dict(
                parameter=parameter,
                value=value,
                runs=len(group),
                coverage_mean=float(np.mean(coverage)),
                coverage_std=float(np.std(coverage)),
                redundancy_mean=float(np.mean([
                    row["redundancy"]
                    for row in group
                ])),
                feasible_rate=float(np.mean([
                    row["feasible"]
                    for row in group
                ])),
                runtime_mean_ms=float(np.mean(runtime_ms)),
                runtime_median_ms=float(np.median(runtime_ms)),
                runtime_p95_ms=float(np.percentile(runtime_ms, 95)),
            )
        )

    return summary


def plot_parameter(summary: list[dict], parameter: str, output_dir: Path) -> None:
    rows = [row for row in summary if row["parameter"] == parameter]
    if not rows:
        return

    patterns = sorted({row["pattern"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=170)

    for pattern in patterns:
        group = sorted(
            [row for row in rows if row["pattern"] == pattern],
            key=lambda row: float(row["value"]),
        )
        x = [float(row["value"]) for row in group]
        axes[0].plot(
            x,
            [row["coverage_mean"] for row in group],
            marker="o",
            label=pattern,
        )
        axes[1].plot(
            x,
            [row["runtime_mean_ms"] for row in group],
            marker="o",
            label=pattern,
        )

    axes[0].set_ylabel("Mean weighted coverage")
    axes[1].set_ylabel("Mean runtime (ms)")
    for ax in axes:
        ax.set_xlabel(parameter)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{parameter}.png", dpi=220)
    plt.close(fig)


def run_sensitivity(
    *,
    patterns: tuple[str, ...],
    parameters: tuple[str, ...],
    seeds: int,
    output_dir: Path,
) -> list[dict]:
    rows: list[dict] = []

    for parameter in parameters:
        values = SENSITIVITY_VALUES[parameter]
        for value in values:
            algorithm = ConnectedFrontierLeafSwap(config_with(parameter, value))
            for pattern in patterns:
                for seed in range(seeds):
                    scenario = make_scenario(
                        pattern,
                        seed,
                        n_uavs=BASE_N,
                        n_targets=BASE_K,
                        sensing_radius=BASE_RS,
                        communication_radius=BASE_RC,
                        min_separation=BASE_MIN_SEPARATION,
                    )
                    solution, runtime = algorithm.solve(scenario, seed=seed)
                    metrics = evaluate(scenario, solution)
                    rows.append(
                        dict(
                            parameter=parameter,
                            value=value,
                            pattern=pattern,
                            seed=seed,
                            n_uavs=scenario.n_uavs,
                            n_targets=len(scenario.targets),
                            rs=scenario.sensing_radius,
                            rc=scenario.communication_radius,
                            min_separation=scenario.min_separation,
                            coverage=metrics.weighted_coverage_ratio,
                            redundancy=metrics.redundancy_excess,
                            feasible=int(metrics.feasible),
                            connected=int(metrics.connected),
                            collision_violations=metrics.collision_violations,
                            forbidden_violations=metrics.forbidden_violations,
                            runtime_sec=runtime,
                        )
                    )

            print(
                f"finished parameter={parameter}, value={value}",
                flush=True,
            )

    write_csv(output_dir / "sensitivity_runs.csv", rows)
    per_pattern = summarize(rows)
    overall = summarize_overall(rows)
    write_csv(output_dir / "sensitivity_summary_by_pattern.csv", per_pattern)
    write_csv(output_dir / "sensitivity_summary.csv", overall)

    figure_dir = output_dir / "figures"
    for parameter in parameters:
        plot_parameter(per_pattern, parameter, figure_dir)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CFG-LS one-factor-at-a-time sensitivity tests.",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_SEEDS),
        default="smoke",
        help="smoke is quick; report is suitable for table/figure generation.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        help="Override the number of seeds from --profile.",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=list(DEFAULT_PATTERNS),
        help="Scenario patterns used for every parameter value.",
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        choices=tuple(SENSITIVITY_VALUES),
        default=list(SENSITIVITY_VALUES),
        help="CFG-LS hyperparameters to sweep.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sensitivity"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else PROFILE_SEEDS[args.profile]
    if seeds < 1:
        raise ValueError("--seeds must be at least 1")

    run_sensitivity(
        patterns=tuple(args.patterns),
        parameters=tuple(args.parameters),
        seeds=seeds,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
