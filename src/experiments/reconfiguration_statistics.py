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
    *,
    apply_holm: bool = True,
) -> pd.DataFrame:
    """Paired Wilcoxon comparisons for one CP3 baseline."""
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
        ).dropna(
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

    if apply_holm and len(out):
        out["p_holm"] = holm_adjust(
            out["p_value"].to_numpy()
        )
        out["significant_0_05"] = out["p_holm"] < 0.05

    return out


def blocked_reconfiguration_comparisons(
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
    """Compare algorithms across problem-level effects, not pooled seed rows.

    Seeds are repeated trials inside a problem. The primary statistical unit is
    therefore one problem: first average paired seed effects per problem, then
    apply Wilcoxon across those problem-level effects.
    """
    rows = []
    left = df[df["algorithm"] == proposed]
    right = df[df["algorithm"] == baseline]

    for metric in metrics:
        proposed_values = left[["problem", "seed", metric]].rename(
            columns={metric: "proposed_value"}
        )
        baseline_values = right[["problem", "seed", metric]].rename(
            columns={metric: "baseline_value"}
        )
        paired = proposed_values.merge(
            baseline_values,
            on=["problem", "seed"],
            how="inner",
        ).dropna(subset=["proposed_value", "baseline_value"])
        if paired.empty:
            continue

        if METRIC_DIRECTIONS[metric] == "higher":
            paired["effect"] = (
                paired["proposed_value"] - paired["baseline_value"]
            )
        else:
            paired["effect"] = (
                paired["baseline_value"] - paired["proposed_value"]
            )
        problem_effects = paired.groupby("problem")["effect"].mean()
        effects = problem_effects.to_numpy()
        if np.allclose(effects, 0.0):
            statistic = 0.0
            p_value = 1.0
        else:
            result = wilcoxon(
                effects,
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
            "n_problems": len(effects),
            "n_seed_pairs": len(paired),
            "mean_problem_effect": float(np.mean(effects)),
            "median_problem_effect": float(np.median(effects)),
            "problems_better": int(np.sum(effects > 0)),
            "problems_equal": int(np.sum(np.isclose(effects, 0.0))),
            "problems_worse": int(np.sum(effects < 0)),
            "wilcoxon_statistic": statistic,
            "p_value": p_value,
        })

    return pd.DataFrame(rows)


def feasibility_comparison(
    df: pd.DataFrame,
    proposed: str = "transition_aware_ga",
    baseline: str = "static_then_transition",
) -> pd.DataFrame:
    """Report paired feasibility counts and exact McNemar p-value."""
    columns = [
        "problem",
        "seed",
        "transition_feasible",
    ]

    left = df[df["algorithm"] == proposed][columns].rename(
        columns={"transition_feasible": "proposed_feasible"}
    )
    right = df[df["algorithm"] == baseline][columns].rename(
        columns={"transition_feasible": "baseline_feasible"}
    )

    paired = left.merge(
        right,
        on=["problem", "seed"],
        how="inner",
    )

    proposed_flag = paired["proposed_feasible"].astype(bool)
    baseline_flag = paired["baseline_feasible"].astype(bool)

    both = int(np.sum(proposed_flag & baseline_flag))
    proposed_only = int(np.sum(proposed_flag & ~baseline_flag))
    baseline_only = int(np.sum(~proposed_flag & baseline_flag))
    neither = int(np.sum(~proposed_flag & ~baseline_flag))
    discordant = proposed_only + baseline_only

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
            float(proposed_flag.mean())
            if len(paired)
            else float("nan")
        ),
        "baseline_feasible_rate": (
            float(baseline_flag.mean())
            if len(paired)
            else float("nan")
        ),
    }])


def _resolve_baselines(
    df: pd.DataFrame,
    proposed: str,
    baselines,
) -> list[str]:
    if baselines is None:
        return sorted(
            algorithm
            for algorithm in df["algorithm"].dropna().unique()
            if algorithm != proposed
        )

    if isinstance(baselines, str):
        return [baselines]

    return list(baselines)


def run_reconfiguration_statistics(
    csv_path: str | Path,
    proposed: str = "transition_aware_ga",
    baseline: str | None = None,
    baselines=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare CP3 against one or every available baseline.

    baseline is retained for backward compatibility. When neither baseline nor
    baselines is supplied, every algorithm other than proposed is compared.
    Holm correction is applied globally across continuous tests and separately
    across exact McNemar feasibility tests.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if baseline is not None and baselines is not None:
        raise ValueError("use either baseline or baselines, not both")

    requested = baseline if baseline is not None else baselines
    baseline_names = _resolve_baselines(
        df,
        proposed,
        requested,
    )

    continuous_parts = []
    blocked_parts = []
    feasibility_parts = []

    for baseline_name in baseline_names:
        continuous_parts.append(
            paired_reconfiguration_comparisons(
                df,
                proposed=proposed,
                baseline=baseline_name,
                apply_holm=False,
            )
        )
        blocked_parts.append(
            blocked_reconfiguration_comparisons(
                df,
                proposed=proposed,
                baseline=baseline_name,
            )
        )
        feasibility_parts.append(
            feasibility_comparison(
                df,
                proposed=proposed,
                baseline=baseline_name,
            )
        )

    continuous = (
        pd.concat(continuous_parts, ignore_index=True)
        if continuous_parts
        else pd.DataFrame()
    )
    feasibility = (
        pd.concat(feasibility_parts, ignore_index=True)
        if feasibility_parts
        else pd.DataFrame()
    )
    blocked = (
        pd.concat(blocked_parts, ignore_index=True)
        if blocked_parts
        else pd.DataFrame()
    )

    if len(continuous):
        continuous["p_holm"] = holm_adjust(
            continuous["p_value"].to_numpy()
        )
        continuous["significant_0_05"] = (
            continuous["p_holm"] < 0.05
        )

    if len(feasibility):
        feasibility["mcnemar_p_holm"] = holm_adjust(
            feasibility["mcnemar_exact_p"].to_numpy()
        )

    if len(blocked):
        blocked["p_holm"] = holm_adjust(
            blocked["p_value"].to_numpy()
        )
        blocked["significant_0_05"] = blocked["p_holm"] < 0.05
        feasibility["mcnemar_significant_0_05"] = (
            feasibility["mcnemar_p_holm"] < 0.05
        )

    continuous.to_csv(
        csv_path.parent / "statistics.csv",
        index=False,
    )
    feasibility.to_csv(
        csv_path.parent / "feasibility_comparison.csv",
        index=False,
    )
    blocked.to_csv(
        csv_path.parent / "statistics_blocked_by_problem.csv",
        index=False,
    )

    return continuous, feasibility
