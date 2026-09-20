from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from .statistics import holm_adjust


# Positive effect always means "proposed is better".
METRIC_DIRECTIONS = {
    "joint_fitness": "higher",
    "weighted_coverage_ratio": "higher",
    "formation_time_sec": "lower",
    "total_travel_distance": "lower",
    "runtime_sec": "lower",
}


def paired_reconfiguration_comparisons(
    df: pd.DataFrame,
    proposed: str = "transition_aware_ga",
    baseline: str = "static_then_transition",
    metrics=(
        "joint_fitness",
        "weighted_coverage_ratio",
        "formation_time_sec",
        "total_travel_distance",
    ),
) -> pd.DataFrame:
    """Paired Wilcoxon comparisons for Checkpoint-03 joint experiments.

    Rows are paired on (problem, seed). For lower-is-better metrics such as time
    and travel, effect is baseline - proposed. Therefore a positive mean/median
    effect consistently means that the proposed method improved the metric.
    """
    rows = []

    left = df[df["algorithm"] == proposed]
    right = df[df["algorithm"] == baseline]

    for metric in metrics:
        if metric not in METRIC_DIRECTIONS:
            raise ValueError(f"unknown metric direction: {metric}")

        proposed_values = left[
            ["problem", "seed", metric]
        ].rename(columns={metric: "proposed_value"})

        baseline_values = right[
            ["problem", "seed", metric]
        ].rename(columns={metric: "baseline_value"})

        paired = proposed_values.merge(
            baseline_values,
            on=["problem", "seed"],
            how="inner",
        )

        # Formation time is NaN for a deadlocked/infeasible transition. Those
        # pairs are intentionally excluded from the time-specific Wilcoxon test;
        # feasibility is analyzed separately from continuous time quality.
        paired = paired.dropna(
            subset=["proposed_value", "baseline_value"]
        )

        if len(paired) == 0:
            continue

        proposed_array = paired["proposed_value"].to_numpy()
        baseline_array = paired["baseline_value"].to_numpy()

        if METRIC_DIRECTIONS[metric] == "higher":
            effect = proposed_array - baseline_array
        else:
            effect = baseline_array - proposed_array

        if np.allclose(effect, 0.0):
            statistic = 0.0
            p_value = 1.0
        else:
            result = wilcoxon(
                effect,
                zero_method="wilcox",
                alternative="two-sided",
            )
            statistic = float(result.statistic)
            p_value = float(result.pvalue)

        rows.append({
            "metric": metric,
            "direction": METRIC_DIRECTIONS[metric],
            "proposed": proposed,
            "baseline": baseline,
            "n_pairs": len(paired),
            "proposed_mean": float(np.mean(proposed_array)),
            "baseline_mean": float(np.mean(baseline_array)),
            "mean_effect": float(np.mean(effect)),
            "median_effect": float(np.median(effect)),
            "wilcoxon_statistic": statistic,
            "p_value": p_value,
        })

    out = pd.DataFrame(rows)

    if len(out):
        out["p_holm"] = holm_adjust(
            out["p_value"].to_numpy()
        )
        out["significant_0_05"] = out["p_holm"] < 0.05

    return out


def feasibility_comparison(
    df: pd.DataFrame,
    proposed: str = "transition_aware_ga",
    baseline: str = "static_then_transition",
) -> pd.DataFrame:
    """Report paired transition-feasibility counts.

    This table is intentionally descriptive rather than forcing a Wilcoxon test
    onto binary outcomes. It also exposes asymmetric pairs: cases where only one
    method reaches a feasible transition.
    """
    columns = [
        "problem",
        "seed",
        "transition_feasible",
    ]

    left = df[df["algorithm"] == proposed][columns].rename(
        columns={
            "transition_feasible": "proposed_feasible",
        }
    )
    right = df[df["algorithm"] == baseline][columns].rename(
        columns={
            "transition_feasible": "baseline_feasible",
        }
    )

    paired = left.merge(
        right,
        on=["problem", "seed"],
        how="inner",
    )

    both = int(
        np.sum(
            paired["proposed_feasible"]
            & paired["baseline_feasible"]
        )
    )
    proposed_only = int(
        np.sum(
            paired["proposed_feasible"]
            & ~paired["baseline_feasible"]
        )
    )
    baseline_only = int(
        np.sum(
            ~paired["proposed_feasible"]
            & paired["baseline_feasible"]
        )
    )
    neither = int(
        np.sum(
            ~paired["proposed_feasible"]
            & ~paired["baseline_feasible"]
        )
    )

    discordant = proposed_only + baseline_only

    # Exact McNemar test is equivalent to a two-sided Binomial(0.5) test on
    # discordant pairs. It is appropriate here because transition feasibility
    # is binary and observations are paired by the same problem/seed.
    if discordant:
        mcnemar_p = float(
            binomtest(
                proposed_only,
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    else:
        mcnemar_p = 1.0

    return pd.DataFrame([{
        "proposed": proposed,
        "baseline": baseline,
        "n_pairs": len(paired),
        "both_feasible": both,
        "proposed_only_feasible": proposed_only,
        "baseline_only_feasible": baseline_only,
        "neither_feasible": neither,
        "discordant_pairs": discordant,
        "mcnemar_exact_p": mcnemar_p,
        "proposed_feasible_rate": (
            float(paired["proposed_feasible"].mean())
            if len(paired)
            else float("nan")
        ),
        "baseline_feasible_rate": (
            float(paired["baseline_feasible"].mean())
            if len(paired)
            else float("nan")
        ),
    }])


def run_reconfiguration_statistics(
    csv_path: str | Path,
    proposed: str = "transition_aware_ga",
    baseline: str = "static_then_transition",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    continuous = paired_reconfiguration_comparisons(
        df,
        proposed=proposed,
        baseline=baseline,
    )
    feasibility = feasibility_comparison(
        df,
        proposed=proposed,
        baseline=baseline,
    )

    continuous.to_csv(
        csv_path.parent / "statistics.csv",
        index=False,
    )
    feasibility.to_csv(
        csv_path.parent / "feasibility_comparison.csv",
        index=False,
    )

    return continuous, feasibility
