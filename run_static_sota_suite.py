"""Report-scale static UAV coverage/connectivity evaluation suite.

The suite keeps every algorithm on the same static target-point problem and
separates the experiments instead of taking a Cartesian product of every
parameter:

A. Main benchmark: clean static target patterns.
B. Hard benchmark: deliberately difficult generated patterns.
C. Obstacle benchmark: explicit no-deploy regions, reported separately.
D. UAV/target/connectivity scalability sweeps.
E. Ablation: CFG versus CFG-LS is present in every experiment.
F. Small exact instances: CFG-LS versus a MILP optimum on the SAME candidate set.

Use --profile smoke while editing and --profile report for final tables.

Examples:
    python run_static_sota_suite.py --profile smoke
    python run_static_sota_suite.py --profile report
    python run_static_sota_suite.py --profile extended --skip-exact
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import time
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from run_static_coverage_study import (
    BASIC_PATTERNS,
    HARD_PATTERNS,
    PATTERNS,
    algorithms,
    make_scenario,
)
from src.baselines import ExactStaticCoverageMILP, ExactStaticMILPConfig
from src.metrics import evaluate
from src.proposed import ConnectedFrontierConfig, ConnectedFrontierLeafSwap
from src.scenario_io import read_scenarios


BASE_N = 6
BASE_K = 120
BASE_RS = 170.0
BASE_RC = 300.0
BASE_MIN_SEPARATION = 30.0

UAV_VALUES = (4, 6, 8, 10, 12)
TARGET_VALUES = (50, 100, 200, 400)
RC_VALUES = (220.0, 260.0, 300.0, 360.0)
EXTENDED_UAV_VALUES = (4, 6, 8, 10, 12, 16)
EXTENDED_TARGET_VALUES = (50, 100, 200, 400, 800)
EXTENDED_RC_VALUES = (180.0, 220.0, 260.0, 300.0, 360.0)
EXACT_N_VALUES = (3, 4)
EXACT_K_VALUES = (20, 30)


PROFILE_CONFIGS = {
    "smoke": dict(
        main_seeds=3,
        hard_seeds=2,
        obstacle_seeds=2,
        sweep_seeds=2,
        exact_seeds=1,
    ),
    "report": dict(
        main_seeds=30,
        hard_seeds=20,
        obstacle_seeds=15,
        sweep_seeds=10,
        exact_seeds=3,
    ),
    "extended": dict(
        main_seeds=200,
        hard_seeds=100,
        obstacle_seeds=100,
        sweep_seeds=30,
        exact_seeds=5,
    ),
}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_algorithms(
    *,
    experiment: str,
    parameter_name: str,
    parameter_value: float | int | str,
    patterns: tuple[str, ...],
    seeds: int,
    n_uavs: int,
    n_targets: int,
    rs: float,
    rc: float,
    min_separation: float,
    iterations: int,
    obstacle_layout: str = "none",
) -> list[dict]:
    methods = algorithms(iterations)
    rows: list[dict] = []

    for pattern in patterns:
        for seed in range(seeds):
            scenario = make_scenario(
                pattern,
                seed,
                n_uavs=n_uavs,
                n_targets=n_targets,
                sensing_radius=rs,
                communication_radius=rc,
                min_separation=min_separation,
                obstacle_layout=obstacle_layout,
            )

            for name, method in methods.items():
                solution, runtime = method.solve(scenario, seed=seed)
                result = evaluate(scenario, solution)
                rows.append(
                    dict(
                        experiment=experiment,
                        parameter=parameter_name,
                        value=parameter_value,
                        pattern=pattern,
                        seed=seed,
                        algorithm=name,
                        n_uavs=n_uavs,
                        n_targets=n_targets,
                        rs=rs,
                        rc=rc,
                        obstacle_layout=obstacle_layout,
                        coverage=result.weighted_coverage_ratio,
                        redundancy=result.redundancy_excess,
                        feasible=int(result.feasible),
                        connected=int(result.connected),
                        collision_violations=result.collision_violations,
                        forbidden_violations=result.forbidden_violations,
                        runtime_sec=runtime,
                    )
                )

    return rows


def run_scenario_list(
    *,
    experiment: str,
    scenarios,
    iterations: int,
) -> list[dict]:
    methods = algorithms(iterations)
    rows: list[dict] = []

    for scenario_id, scenario in enumerate(scenarios):
        for name, method in methods.items():
            solution, runtime = method.solve(
                scenario,
                seed=scenario.seed,
            )
            result = evaluate(scenario, solution)
            rows.append(
                dict(
                    experiment=experiment,
                    parameter="scenario",
                    value=scenario.pattern,
                    pattern=scenario.pattern,
                    seed=scenario.seed,
                    scenario=scenario.name,
                    scenario_id=scenario_id,
                    algorithm=name,
                    n_uavs=scenario.n_uavs,
                    n_targets=len(scenario.targets),
                    rs=scenario.sensing_radius,
                    rc=scenario.communication_radius,
                    obstacle_layout="scenario_json",
                    coverage=result.weighted_coverage_ratio,
                    redundancy=result.redundancy_excess,
                    feasible=int(result.feasible),
                    connected=int(result.connected),
                    collision_violations=result.collision_violations,
                    forbidden_violations=result.forbidden_violations,
                    runtime_sec=runtime,
                )
            )

        print(
            f"finished {experiment}: {scenario.name}",
            flush=True,
        )

    return rows


def summarize(
    rows: list[dict],
    *,
    group_field: str = "value",
) -> list[dict]:
    keys = sorted({
        (row[group_field], row["algorithm"])
        for row in rows
    }, key=lambda item: (str(item[0]), item[1]))
    out = []

    for value, algorithm in keys:
        group = [
            row
            for row in rows
            if row[group_field] == value and row["algorithm"] == algorithm
        ]
        runtimes_ms = np.asarray(
            [row["runtime_sec"] * 1000.0 for row in group],
            dtype=float,
        )
        coverage = np.asarray(
            [row["coverage"] for row in group],
            dtype=float,
        )
        out.append(
            dict(
                value=value,
                algorithm=algorithm,
                runs=len(group),
                coverage_mean=float(np.mean(coverage)),
                coverage_std=float(np.std(coverage)),
                redundancy_mean=float(np.mean([row["redundancy"] for row in group])),
                feasible_rate=float(np.mean([row["feasible"] for row in group])),
                runtime_mean_ms=float(np.mean(runtimes_ms)),
                runtime_median_ms=float(np.median(runtimes_ms)),
                runtime_p95_ms=float(np.percentile(runtimes_ms, 95)),
            )
        )

    return out


def paired_statistics(rows: list[dict]) -> list[dict]:
    proposed = {
        (row["pattern"], row["seed"]): row
        for row in rows
        if row["algorithm"] == "cfg_ls"
    }
    comparators = ("greedy", "jocc_cpgs", "jocc_dpgs", "graph_ga", "cfg")
    out = []

    for comparator in comparators:
        baseline = {
            (row["pattern"], row["seed"]): row
            for row in rows
            if row["algorithm"] == comparator
        }
        keys = sorted(set(proposed) & set(baseline))
        if not keys:
            continue

        diff = np.asarray([
            proposed[key]["coverage"] - baseline[key]["coverage"]
            for key in keys
        ], dtype=float)
        speedup = np.asarray([
            baseline[key]["runtime_sec"]
            / max(proposed[key]["runtime_sec"], 1e-12)
            for key in keys
        ], dtype=float)

        nonzero = diff[np.abs(diff) > 1e-12]
        if len(nonzero) == 0:
            pvalue = 1.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pvalue = float(
                    wilcoxon(
                        diff,
                        alternative="two-sided",
                        zero_method="wilcox",
                    ).pvalue
                )

        out.append(
            dict(
                comparator=comparator,
                pairs=len(keys),
                mean_coverage_diff_pp=float(np.mean(diff) * 100.0),
                median_coverage_diff_pp=float(np.median(diff) * 100.0),
                proposed_win_rate=float(np.mean(diff > 1e-12)),
                wilcoxon_pvalue=pvalue,
                runtime_speedup_median=float(np.median(speedup)),
                runtime_speedup_mean=float(np.mean(speedup)),
            )
        )

    return out


def plot_metric(
    summary: list[dict],
    *,
    output: Path,
    xlabel: str,
    metric: str,
    ylabel: str,
) -> None:
    algorithms_present = sorted({row["algorithm"] for row in summary})
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for algorithm in algorithms_present:
        group = [row for row in summary if row["algorithm"] == algorithm]
        group = sorted(group, key=lambda row: float(row["value"]))
        x = [float(row["value"]) for row in group]
        y = [float(row[metric]) for row in group]
        ax.plot(x, y, marker="o", label=algorithm)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_main(summary: list[dict], output: Path) -> None:
    rows = sorted(summary, key=lambda row: row["coverage_mean"], reverse=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.bar(
        [row["algorithm"] for row in rows],
        [row["coverage_mean"] for row in rows],
    )
    ax.set_ylabel("Mean weighted coverage")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def run_exact_oracle(
    *,
    patterns: tuple[str, ...],
    seeds: int,
    grid_size: int,
    time_limit_sec: float,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []

    for n_uavs in EXACT_N_VALUES:
        for n_targets in EXACT_K_VALUES:
            for pattern in patterns:
                for seed in range(seeds):
                    scenario = make_scenario(
                        pattern,
                        seed,
                        n_uavs=n_uavs,
                        n_targets=n_targets,
                        sensing_radius=BASE_RS,
                        communication_radius=BASE_RC,
                        min_separation=BASE_MIN_SEPARATION,
                    )

                    proposed = ConnectedFrontierLeafSwap(
                        ConnectedFrontierConfig(
                            grid_size=grid_size,
                            local_rounds=12,
                        )
                    )
                    cfg_solution, cfg_runtime = proposed.solve(
                        scenario,
                        seed=seed,
                    )
                    cfg_metrics = evaluate(
                        scenario,
                        cfg_solution,
                    )

                    oracle = ExactStaticCoverageMILP(
                        ExactStaticMILPConfig(
                            grid_size=grid_size,
                            time_limit_sec=time_limit_sec,
                            mip_rel_gap=0.0,
                        )
                    )
                    try:
                        exact_solution, exact_runtime = oracle.solve(
                            scenario,
                            seed=seed,
                        )
                        exact_metrics = evaluate(
                            scenario,
                            exact_solution,
                        )
                        optimum = exact_metrics.weighted_coverage_ratio
                        gap = (
                            optimum - cfg_metrics.weighted_coverage_ratio
                        ) / max(optimum, 1e-12)
                        status = "optimal"
                    except RuntimeError as failure:
                        exact_runtime = time_limit_sec
                        optimum = np.nan
                        gap = np.nan
                        status = str(failure)

                    rows.append(
                        dict(
                            n_uavs=n_uavs,
                            n_targets=n_targets,
                            pattern=pattern,
                            seed=seed,
                            grid_size=grid_size,
                            cfg_ls_coverage=cfg_metrics.weighted_coverage_ratio,
                            exact_coverage=optimum,
                            optimality_gap=gap,
                            cfg_ls_runtime_ms=cfg_runtime * 1000.0,
                            exact_runtime_ms=exact_runtime * 1000.0,
                            status=status,
                        )
                    )

    solved = [row for row in rows if row["status"] == "optimal"]
    summary = []
    for n_uavs in EXACT_N_VALUES:
        for n_targets in EXACT_K_VALUES:
            group = [
                row for row in solved
                if row["n_uavs"] == n_uavs
                and row["n_targets"] == n_targets
            ]
            if not group:
                continue
            summary.append(
                dict(
                    n_uavs=n_uavs,
                    n_targets=n_targets,
                    solved=len(group),
                    mean_cfg_ls_coverage=float(np.mean([
                        row["cfg_ls_coverage"] for row in group
                    ])),
                    mean_exact_coverage=float(np.mean([
                        row["exact_coverage"] for row in group
                    ])),
                    mean_optimality_gap_pct=float(np.mean([
                        row["optimality_gap"] for row in group
                    ]) * 100.0),
                    median_optimality_gap_pct=float(np.median([
                        row["optimality_gap"] for row in group
                    ]) * 100.0),
                    exact_match_rate=float(np.mean([
                        abs(row["optimality_gap"]) <= 1e-12
                        for row in group
                    ])),
                    median_cfg_ls_runtime_ms=float(np.median([
                        row["cfg_ls_runtime_ms"] for row in group
                    ])),
                    median_exact_runtime_ms=float(np.median([
                        row["exact_runtime_ms"] for row in group
                    ])),
                )
            )

    return rows, summary


def print_main(
    summary: list[dict],
    stats: list[dict],
    *,
    title: str = "Main benchmark",
) -> None:
    rows = sorted(summary, key=lambda row: row["coverage_mean"], reverse=True)
    print()
    print(title)
    print("-" * len(title))
    has_multiple_values = len({row["value"] for row in rows}) > 1
    prefix = f"{'case':<22} " if has_multiple_values else ""
    print(
        f"{prefix}{'method':<12} {'coverage':>9} {'std':>8} "
        f"{'redund':>8} {'feas':>7} {'med ms':>10} {'p95 ms':>10}"
    )
    for row in rows:
        prefix_value = f"{str(row['value']):<22} " if has_multiple_values else ""
        print(
            f"{prefix_value}{row['algorithm']:<12} "
            f"{row['coverage_mean']:>9.3f} "
            f"{row['coverage_std']:>8.3f} "
            f"{row['redundancy_mean']:>8.3f} "
            f"{row['feasible_rate']:>7.3f} "
            f"{row['runtime_median_ms']:>10.2f} "
            f"{row['runtime_p95_ms']:>10.2f}"
        )

    print()
    stats_title = "Paired CFG-LS comparisons"
    print(stats_title)
    print("-" * len(stats_title))
    for row in stats:
        print(
            f"vs {row['comparator']:<10} "
            f"coverage={row['mean_coverage_diff_pp']:+6.2f} pp, "
            f"p={row['wilcoxon_pvalue']:.4g}, "
            f"speedup={row['runtime_speedup_median']:.2f}x"
        )


def profile_counts(args) -> dict:
    counts = dict(PROFILE_CONFIGS[args.profile])
    overrides = {
        "main_seeds": args.main_seeds,
        "hard_seeds": args.hard_seeds,
        "obstacle_seeds": args.obstacle_seeds,
        "sweep_seeds": args.sweep_seeds,
        "exact_seeds": args.exact_seeds,
    }
    for key, value in overrides.items():
        if value is not None:
            counts[key] = value
    return counts


def run_pattern_split(
    *,
    experiment: str,
    patterns: tuple[str, ...],
    seeds: int,
    n_uavs: int,
    n_targets: int,
    rs: float,
    rc: float,
    min_separation: float,
    iterations: int,
    obstacle_layout: str = "none",
) -> list[dict]:
    rows: list[dict] = []
    for pattern in patterns:
        rows.extend(
            run_algorithms(
                experiment=experiment,
                parameter_name="pattern",
                parameter_value=pattern,
                patterns=(pattern,),
                seeds=seeds,
                n_uavs=n_uavs,
                n_targets=n_targets,
                rs=rs,
                rc=rc,
                min_separation=min_separation,
                iterations=iterations,
                obstacle_layout=obstacle_layout,
            )
        )
        print(
            f"finished {experiment}: pattern={pattern}, seeds={seeds}",
            flush=True,
        )
    return rows


def append_manifest(
    rows: list[dict],
    *,
    experiment: str,
    seeds: int,
    patterns: tuple[str, ...],
    methods: int,
    n_uavs: int,
    n_targets: int,
    rs: float,
    rc: float,
    obstacle_layout: str = "none",
) -> None:
    rows.append(
        dict(
            experiment=experiment,
            scenarios=len(patterns) * seeds,
            runs=len(patterns) * seeds * methods,
            seeds=seeds,
            patterns=";".join(patterns),
            methods=methods,
            n_uavs=n_uavs,
            n_targets=n_targets,
            rs=rs,
            rc=rc,
            obstacle_layout=obstacle_layout,
        )
    )


def run_suite(args) -> None:
    counts = profile_counts(args)
    main_seeds = counts["main_seeds"]
    hard_seeds = counts["hard_seeds"]
    obstacle_seeds = counts["obstacle_seeds"]
    sweep_seeds = counts["sweep_seeds"]
    exact_seeds = counts["exact_seeds"]

    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    suite_started = time.perf_counter()
    manifest: list[dict] = []
    method_count = len(algorithms(args.iterations))

    if not args.skip_main:
        main_rows = run_algorithms(
            experiment="main",
            parameter_name="base",
            parameter_value=0,
            patterns=tuple(args.patterns),
            seeds=main_seeds,
            n_uavs=BASE_N,
            n_targets=BASE_K,
            rs=BASE_RS,
            rc=BASE_RC,
            min_separation=BASE_MIN_SEPARATION,
            iterations=args.iterations,
        )
        append_manifest(
            manifest,
            experiment="main",
            seeds=main_seeds,
            patterns=tuple(args.patterns),
            methods=method_count,
            n_uavs=BASE_N,
            n_targets=BASE_K,
            rs=BASE_RS,
            rc=BASE_RC,
        )
        main_summary = summarize(main_rows)
        main_stats = paired_statistics(main_rows)
        write_csv(root / "main_runs.csv", main_rows)
        write_csv(root / "main_summary.csv", main_summary)
        write_csv(root / "main_paired_statistics.csv", main_stats)
        print_main(main_summary, main_stats, title="Main clean benchmark")

    if not args.skip_hard:
        hard_rows = run_pattern_split(
            experiment="hard_static",
            patterns=tuple(args.hard_patterns),
            seeds=hard_seeds,
            n_uavs=max(4, BASE_N - 1),
            n_targets=max(BASE_K, 180),
            rs=0.82 * BASE_RS,
            rc=0.82 * BASE_RC,
            min_separation=BASE_MIN_SEPARATION,
            iterations=args.iterations,
        )
        hard_summary = summarize(hard_rows)
        hard_stats = paired_statistics(hard_rows)
        write_csv(root / "hard_static_runs.csv", hard_rows)
        write_csv(root / "hard_static_summary.csv", hard_summary)
        write_csv(root / "hard_static_paired_statistics.csv", hard_stats)
        append_manifest(
            manifest,
            experiment="hard_static",
            seeds=hard_seeds,
            patterns=tuple(args.hard_patterns),
            methods=method_count,
            n_uavs=max(4, BASE_N - 1),
            n_targets=max(BASE_K, 180),
            rs=0.82 * BASE_RS,
            rc=0.82 * BASE_RC,
        )
        print_main(hard_summary, hard_stats, title="Hard static benchmark")

    if not args.skip_obstacles:
        obstacle_rows = run_pattern_split(
            experiment="obstacle_static",
            patterns=tuple(args.obstacle_patterns),
            seeds=obstacle_seeds,
            n_uavs=BASE_N,
            n_targets=BASE_K,
            rs=BASE_RS,
            rc=BASE_RC,
            min_separation=BASE_MIN_SEPARATION,
            iterations=args.iterations,
            obstacle_layout="blocks",
        )
        obstacle_summary = summarize(obstacle_rows)
        obstacle_stats = paired_statistics(obstacle_rows)
        write_csv(root / "obstacle_static_runs.csv", obstacle_rows)
        write_csv(root / "obstacle_static_summary.csv", obstacle_summary)
        write_csv(root / "obstacle_static_paired_statistics.csv", obstacle_stats)
        append_manifest(
            manifest,
            experiment="obstacle_static",
            seeds=obstacle_seeds,
            patterns=tuple(args.obstacle_patterns),
            methods=method_count,
            n_uavs=BASE_N,
            n_targets=BASE_K,
            rs=BASE_RS,
            rc=BASE_RC,
            obstacle_layout="blocks",
        )
        print_main(
            obstacle_summary,
            obstacle_stats,
            title="Obstacle static benchmark",
        )

    if args.external_scenarios:
        external_scenarios = read_scenarios(args.external_scenarios)
        external_rows = run_scenario_list(
            experiment="external_static",
            scenarios=external_scenarios,
            iterations=args.iterations,
        )
        external_summary = summarize(external_rows)
        external_stats = paired_statistics(external_rows)
        write_csv(root / "external_static_runs.csv", external_rows)
        write_csv(root / "external_static_summary.csv", external_summary)
        write_csv(root / "external_static_paired_statistics.csv", external_stats)
        if external_scenarios:
            patterns = tuple(sorted({
                scenario.pattern for scenario in external_scenarios
            }))
            manifest.append(
                dict(
                    experiment="external_static",
                    scenarios=len(external_scenarios),
                    runs=len(external_scenarios) * method_count,
                    seeds=len(external_scenarios),
                    patterns=";".join(patterns),
                    methods=method_count,
                    n_uavs=int(np.median([
                        scenario.n_uavs for scenario in external_scenarios
                    ])),
                    n_targets=int(np.median([
                        len(scenario.targets) for scenario in external_scenarios
                    ])),
                    rs=float(np.median([
                        scenario.sensing_radius for scenario in external_scenarios
                    ])),
                    rc=float(np.median([
                        scenario.communication_radius for scenario in external_scenarios
                    ])),
                    obstacle_layout="scenario_json",
                )
            )
        print_main(
            external_summary,
            external_stats,
            title="External static benchmark",
        )

    if not args.skip_sweeps:
        sweep_specs = (
            (
                "uav_scalability",
                "n_uavs",
                EXTENDED_UAV_VALUES if args.profile == "extended" else UAV_VALUES,
            ),
            (
                "target_scalability",
                "n_targets",
                EXTENDED_TARGET_VALUES if args.profile == "extended" else TARGET_VALUES,
            ),
            (
                "connectivity_stress",
                "rc",
                EXTENDED_RC_VALUES if args.profile == "extended" else RC_VALUES,
            ),
        )

        for experiment, parameter, values in sweep_specs:
            rows = []
            for value in values:
                kwargs = dict(
                    n_uavs=BASE_N,
                    n_targets=BASE_K,
                    rs=BASE_RS,
                    rc=BASE_RC,
                )
                kwargs[parameter] = value
                rows.extend(
                    run_algorithms(
                        experiment=experiment,
                        parameter_name=parameter,
                        parameter_value=value,
                        patterns=tuple(args.patterns),
                        seeds=sweep_seeds,
                        min_separation=BASE_MIN_SEPARATION,
                        iterations=args.iterations,
                        **kwargs,
                    )
                )
                append_manifest(
                    manifest,
                    experiment=f"{experiment}_{parameter}_{value}",
                    seeds=sweep_seeds,
                    patterns=tuple(args.patterns),
                    methods=method_count,
                    n_uavs=kwargs["n_uavs"],
                    n_targets=kwargs["n_targets"],
                    rs=kwargs["rs"],
                    rc=kwargs["rc"],
                )
                print(
                    f"finished {experiment}: {parameter}={value}",
                    flush=True,
                )

            summary = summarize(rows)
            write_csv(root / f"{experiment}_runs.csv", rows)
            write_csv(root / f"{experiment}_summary.csv", summary)

            if not args.no_plots:
                plot_metric(
                    summary,
                    output=root / f"{experiment}_coverage.png",
                    xlabel=parameter,
                    metric="coverage_mean",
                    ylabel="Mean weighted coverage",
                )
                plot_metric(
                    summary,
                    output=root / f"{experiment}_runtime.png",
                    xlabel=parameter,
                    metric="runtime_median_ms",
                    ylabel="Median runtime (ms)",
                )

    if not args.no_plots:
        if not args.skip_main:
            plot_main(
                main_summary,
                root / "main_coverage.png",
            )

    if not args.skip_exact:
        exact_rows, exact_summary = run_exact_oracle(
            patterns=tuple(args.patterns),
            seeds=exact_seeds,
            grid_size=args.exact_grid_size,
            time_limit_sec=args.exact_time_limit,
        )
        write_csv(root / "exact_runs.csv", exact_rows)
        write_csv(root / "exact_summary.csv", exact_summary)

        solved = [row for row in exact_rows if row["status"] == "optimal"]
        print()
        print("Exact-oracle check")
        print("------------------")
        print(
            f"optimal solves: {len(solved)}/{len(exact_rows)}"
        )
        if solved:
            print(
                "mean CFG-LS optimality gap: "
                f"{np.mean([row['optimality_gap'] for row in solved]) * 100.0:.2f}%"
            )
            print(
                "exact-match rate: "
                f"{np.mean([abs(row['optimality_gap']) <= 1e-12 for row in solved]):.2f}"
            )

    write_csv(root / "suite_manifest.csv", manifest)

    print()
    print(
        f"Suite complete in {time.perf_counter() - suite_started:.2f}s"
    )
    print(f"Results: {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_CONFIGS),
        default="smoke",
        help=(
            "smoke is quick; report is paper-sized; extended uses hundreds "
            "of clean, hard, obstacle, and scalability cases"
        ),
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=PATTERNS,
        default=list(BASIC_PATTERNS),
        help="clean main/sweep patterns",
    )
    parser.add_argument(
        "--hard-patterns",
        nargs="+",
        choices=HARD_PATTERNS,
        default=list(HARD_PATTERNS),
    )
    parser.add_argument(
        "--obstacle-patterns",
        nargs="+",
        choices=PATTERNS,
        default=["uniform", "clustered", "chain"],
    )
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--main-seeds", type=int)
    parser.add_argument("--hard-seeds", type=int)
    parser.add_argument("--obstacle-seeds", type=int)
    parser.add_argument("--sweep-seeds", type=int)
    parser.add_argument("--exact-seeds", type=int)
    parser.add_argument("--skip-main", action="store_true")
    parser.add_argument("--skip-hard", action="store_true")
    parser.add_argument("--skip-obstacles", action="store_true")
    parser.add_argument("--skip-sweeps", action="store_true")
    parser.add_argument("--skip-exact", action="store_true")
    parser.add_argument(
        "--external-scenarios",
        nargs="+",
        type=Path,
        help="Scenario JSON files or directories produced by prepare_external_datasets.py",
    )
    parser.add_argument("--exact-grid-size", type=int, default=6)
    parser.add_argument("--exact-time-limit", type=float, default=20.0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/static_sota_suite"),
    )
    args = parser.parse_args()

    if args.iterations < 1:
        parser.error("--iterations must be positive")
    for name in (
        "main_seeds",
        "hard_seeds",
        "obstacle_seeds",
        "sweep_seeds",
        "exact_seeds",
    ):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.exact_grid_size < 2:
        parser.error("--exact-grid-size must be >= 2")
    if args.exact_time_limit <= 0:
        parser.error("--exact-time-limit must be positive")

    run_suite(args)


if __name__ == "__main__":
    main()
