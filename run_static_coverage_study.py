"""Static UAV coverage/connectivity study for the report-sized problem.

By default this benchmark is obstacle-free and static. Pass --obstacles blocks
to run the explicit no-deploy-region variant. It intentionally excludes raster
maps, path planning and online replanning.

Example:
    python run_static_coverage_study.py --seeds 20
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import numpy as np

from src.baselines import (
    ConnectedGreedy,
    JOCCCentralizedProjectedGradient,
    JOCCDistributedProjectedGradient,
    JOCCGradientConfig,
)
from src.metrics import evaluate
from src.problem import ForbiddenRegion, Scenario
from src.proposed import (
    ConnectedFrontierConfig,
    ConnectedFrontierLeafSwap,
    GraphAwareGA,
)


BASIC_PATTERNS = ("uniform", "clustered", "chain")
HARD_PATTERNS = (
    "hard_chain",
    "islands",
    "corner_clusters",
    "few_uavs_many_targets",
)
PATTERNS = BASIC_PATTERNS + HARD_PATTERNS
OBSTACLE_LAYOUTS = ("none", "blocks")


def obstacle_regions(layout: str) -> tuple[ForbiddenRegion, ...]:
    if layout == "none":
        return ()
    if layout == "blocks":
        return (
            ForbiddenRegion.rectangle("central_block", 440.0, 320.0, 560.0, 680.0),
            ForbiddenRegion.rectangle("west_block", 210.0, 420.0, 330.0, 580.0),
            ForbiddenRegion.circle("east_disc", 760.0, 500.0, 85.0),
        )
    raise ValueError(f"unknown obstacle layout: {layout}")


def make_scenario(
    pattern: str,
    seed: int,
    *,
    n_uavs: int,
    n_targets: int,
    sensing_radius: float,
    communication_radius: float,
    min_separation: float,
    obstacle_layout: str = "none",
) -> Scenario:
    rng = np.random.default_rng(seed)
    width = height = 1000.0

    if pattern == "uniform":
        targets = rng.uniform(
            [80.0, 80.0],
            [920.0, 920.0],
            size=(n_targets, 2),
        )
    elif pattern == "clustered":
        centers = np.array(
            [
                [240.0, 240.0],
                [760.0, 240.0],
                [240.0, 760.0],
                [760.0, 760.0],
            ],
            dtype=float,
        )
        labels = rng.integers(0, len(centers), size=n_targets)
        targets = (
            centers[labels]
            + rng.normal(0.0, 65.0, size=(n_targets, 2))
        )
    elif pattern == "chain":
        centers = np.array(
            [
                [140.0, 500.0],
                [320.0, 500.0],
                [500.0, 500.0],
                [680.0, 500.0],
                [860.0, 500.0],
            ],
            dtype=float,
        )
        labels = rng.integers(0, len(centers), size=n_targets)
        targets = (
            centers[labels]
            + rng.normal(0.0, [38.0, 70.0], size=(n_targets, 2))
        )
    elif pattern == "hard_chain":
        centers = np.array(
            [
                [90.0, 500.0],
                [250.0, 500.0],
                [410.0, 500.0],
                [570.0, 500.0],
                [730.0, 500.0],
                [910.0, 500.0],
            ],
            dtype=float,
        )
        labels = rng.integers(0, len(centers), size=n_targets)
        targets = (
            centers[labels]
            + rng.normal(0.0, [24.0, 45.0], size=(n_targets, 2))
        )
    elif pattern == "islands":
        centers = np.array(
            [
                [100.0, 120.0],
                [900.0, 120.0],
                [120.0, 880.0],
                [880.0, 880.0],
                [500.0, 500.0],
                [500.0, 120.0],
            ],
            dtype=float,
        )
        labels = rng.integers(0, len(centers), size=n_targets)
        targets = (
            centers[labels]
            + rng.normal(0.0, [42.0, 42.0], size=(n_targets, 2))
        )
    elif pattern == "corner_clusters":
        centers = np.array(
            [
                [80.0, 80.0],
                [920.0, 80.0],
                [80.0, 920.0],
                [920.0, 920.0],
            ],
            dtype=float,
        )
        probabilities = np.array([0.35, 0.20, 0.25, 0.20])
        labels = rng.choice(
            len(centers),
            size=n_targets,
            p=probabilities,
        )
        targets = (
            centers[labels]
            + rng.normal(0.0, [35.0, 35.0], size=(n_targets, 2))
        )
    elif pattern == "few_uavs_many_targets":
        centers = np.array(
            [
                [180.0, 220.0],
                [360.0, 740.0],
                [610.0, 260.0],
                [790.0, 760.0],
                [500.0, 500.0],
            ],
            dtype=float,
        )
        labels = rng.integers(0, len(centers), size=n_targets)
        targets = (
            centers[labels]
            + rng.normal(0.0, [78.0, 78.0], size=(n_targets, 2))
        )
    else:
        raise ValueError(f"unknown pattern: {pattern}")

    targets[:, 0] = np.clip(targets[:, 0], 0.0, width)
    targets[:, 1] = np.clip(targets[:, 1], 0.0, height)
    weights = np.ones(n_targets, dtype=float)

    return Scenario(
        name=f"{pattern}_{seed}",
        pattern=pattern,
        width=width,
        height=height,
        targets=targets,
        target_weights=weights,
        n_uavs=n_uavs,
        sensing_radius=sensing_radius,
        communication_radius=communication_radius,
        min_separation=min_separation,
        seed=seed,
        forbidden_regions=obstacle_regions(obstacle_layout),
    )


def algorithms(iterations: int):
    return {
        "greedy": ConnectedGreedy(grid_size=12),
        "jocc_cpgs": JOCCCentralizedProjectedGradient(
            JOCCGradientConfig(iterations=iterations)
        ),
        "jocc_dpgs": JOCCDistributedProjectedGradient(
            JOCCGradientConfig(
                iterations=iterations,
                step_fraction=0.05,
                connectivity_weight=0.80,
            )
        ),
        "graph_ga": GraphAwareGA(
            population_size=28,
            generations=max(24, iterations // 3),
            elite_size=4,
        ),
        "cfg": ConnectedFrontierLeafSwap(
            ConnectedFrontierConfig(
                grid_size=18,
                local_rounds=0,
            )
        ),
        "cfg_ls": ConnectedFrontierLeafSwap(
            ConnectedFrontierConfig(
                grid_size=18,
                local_rounds=12,
            )
        ),
    }


def aggregate(rows):
    methods = sorted({row["algorithm"] for row in rows})
    summary = []

    for method in methods:
        group = [row for row in rows if row["algorithm"] == method]
        summary.append(
            dict(
                algorithm=method,
                runs=len(group),
                coverage=float(np.mean([r["coverage"] for r in group])),
                coverage_std=float(np.std([r["coverage"] for r in group])),
                redundancy=float(np.mean([r["redundancy"] for r in group])),
                feasible_rate=float(np.mean([r["feasible"] for r in group])),
                runtime_ms=float(np.mean([r["runtime_sec"] for r in group])) * 1000.0,
                runtime_p95_ms=float(
                    np.percentile([r["runtime_sec"] for r in group], 95)
                ) * 1000.0,
            )
        )

    return summary


def paired_claims(rows):
    proposed = {
        (r["pattern"], r["seed"]): r
        for r in rows
        if r["algorithm"] == "cfg_ls"
    }
    comparators = [
        "greedy",
        "jocc_cpgs",
        "jocc_dpgs",
        "graph_ga",
        "cfg",
    ]
    output = []

    for comparator in comparators:
        baseline = {
            (r["pattern"], r["seed"]): r
            for r in rows
            if r["algorithm"] == comparator
        }
        keys = sorted(set(proposed) & set(baseline))
        if not keys:
            continue

        coverage_diff = np.asarray(
            [
                proposed[key]["coverage"]
                - baseline[key]["coverage"]
                for key in keys
            ],
            dtype=float,
        )
        runtime_ratio = np.asarray(
            [
                baseline[key]["runtime_sec"]
                / max(proposed[key]["runtime_sec"], 1e-12)
                for key in keys
            ],
            dtype=float,
        )

        output.append(
            dict(
                comparator=comparator,
                pairs=len(keys),
                mean_coverage_diff_pp=float(np.mean(coverage_diff) * 100.0),
                median_coverage_diff_pp=float(np.median(coverage_diff) * 100.0),
                proposed_win_rate=float(np.mean(coverage_diff > 1e-12)),
                mean_runtime_speedup=float(np.mean(runtime_ratio)),
                median_runtime_speedup=float(np.median(runtime_ratio)),
            )
        )

    return output


def print_table(summary):
    print()
    print(
        f"{'method':<12} {'coverage':>9} {'std':>8} "
        f"{'redund':>8} {'feas':>7} {'mean ms':>10} {'p95 ms':>10}"
    )
    print("-" * 72)
    for row in summary:
        print(
            f"{row['algorithm']:<12} "
            f"{row['coverage']:>9.3f} "
            f"{row['coverage_std']:>8.3f} "
            f"{row['redundancy']:>8.3f} "
            f"{row['feasible_rate']:>7.3f} "
            f"{row['runtime_ms']:>10.2f} "
            f"{row['runtime_p95_ms']:>10.2f}"
        )


def print_claims(claims):
    print()
    print("CFG-LS paired differences")
    print("-------------------------")
    for row in claims:
        print(
            f"vs {row['comparator']:<10} "
            f"coverage={row['mean_coverage_diff_pp']:+6.2f} pp, "
            f"win={row['proposed_win_rate']:.2f}, "
            f"runtime speedup={row['median_runtime_speedup']:.2f}x"
        )


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--n-uavs", type=int, default=6)
    parser.add_argument("--n-targets", type=int, default=120)
    parser.add_argument("--rs", type=float, default=170.0)
    parser.add_argument("--rc", type=float, default=300.0)
    parser.add_argument("--min-separation", type=float, default=30.0)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument(
        "--obstacles",
        choices=OBSTACLE_LAYOUTS,
        default="none",
        help=(
            "Obstacle layout for an explicit static variant. The default keeps "
            "the report/SOTA problem obstacle-free."
        ),
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=PATTERNS,
        default=list(BASIC_PATTERNS),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/static_coverage_study"),
    )
    args = parser.parse_args()

    if (
        args.seeds < 1
        or args.n_uavs < 2
        or args.n_targets < 1
        or args.rs <= 0
        or args.rc <= 0
        or args.min_separation < 0
        or args.iterations < 1
    ):
        parser.error("invalid positive count/radius/iteration argument")

    methods = algorithms(args.iterations)
    rows = []
    started = time.perf_counter()

    for pattern in args.patterns:
        for seed in range(args.seeds):
            scenario = make_scenario(
                pattern,
                seed,
                n_uavs=args.n_uavs,
                n_targets=args.n_targets,
                sensing_radius=args.rs,
                communication_radius=args.rc,
                min_separation=args.min_separation,
                obstacle_layout=args.obstacles,
            )

            for name, algorithm in methods.items():
                solution, runtime = algorithm.solve(
                    scenario,
                    seed=seed,
                )
                result = evaluate(
                    scenario,
                    solution,
                )
                row = dict(
                    pattern=pattern,
                    seed=seed,
                    algorithm=name,
                    coverage=result.weighted_coverage_ratio,
                    redundancy=result.redundancy_excess,
                    connected=int(result.connected),
                    collision_violations=result.collision_violations,
                    forbidden_violations=result.forbidden_violations,
                    feasible=int(result.feasible),
                    runtime_sec=runtime,
                )
                rows.append(row)

        print(
            f"finished {pattern}: {args.seeds} seeds x {len(methods)} methods",
            flush=True,
        )

    summary = aggregate(rows)
    claims = paired_claims(rows)
    print_table(summary)
    print_claims(claims)

    proposed = next(row for row in summary if row["algorithm"] == "cfg_ls")
    non_greedy = [
        row
        for row in summary
        if row["algorithm"] in {"jocc_cpgs", "jocc_dpgs", "graph_ga"}
    ]
    best_other_coverage = max(row["coverage"] for row in non_greedy)
    fastest_other_ms = min(row["runtime_ms"] for row in non_greedy)

    coverage_gap = proposed["coverage"] - best_other_coverage
    speedup = fastest_other_ms / max(proposed["runtime_ms"], 1e-12)

    print()
    print("Hypothesis check")
    print("----------------")
    print(
        "H1 hard feasibility: "
        f"{'PASS' if proposed['feasible_rate'] == 1.0 else 'FAIL'} "
        f"(rate={proposed['feasible_rate']:.3f})"
    )
    print(
        "H2 materially faster than gradient/GA comparators: "
        f"{'PASS' if speedup >= 2.0 else 'FAIL'} "
        f"(at least {speedup:.2f}x vs fastest comparator)"
    )
    print(
        "H3 competitive coverage (within 5 pp of best comparator): "
        f"{'PASS' if coverage_gap >= -0.05 else 'FAIL'} "
        f"(gap={coverage_gap * 100:+.2f} pp)"
    )

    write_csv(args.output_dir / "runs.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "paired_claims.csv", claims)

    elapsed = time.perf_counter() - started
    print()
    print(f"Total wall time: {elapsed:.2f}s")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
