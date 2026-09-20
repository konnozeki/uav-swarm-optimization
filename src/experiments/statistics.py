from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)

    running_max = 0.0
    m = len(p_values)

    for rank, idx in enumerate(order):
        value = (m - rank) * p_values[idx]
        running_max = max(running_max, value)
        adjusted[idx] = min(1.0, running_max)

    return adjusted


def paired_comparisons(
    df: pd.DataFrame,
    proposed: str,
    metrics=("fitness", "weighted_coverage_ratio"),
) -> pd.DataFrame:
    rows = []

    baselines = sorted(
        a for a in df["algorithm"].unique()
        if a != proposed
    )

    for metric in metrics:
        for baseline in baselines:
            left = df[df["algorithm"] == proposed][
                ["scenario", "seed", metric]
            ].rename(columns={metric: "proposed_value"})

            right = df[df["algorithm"] == baseline][
                ["scenario", "seed", metric]
            ].rename(columns={metric: "baseline_value"})

            paired = left.merge(
                right,
                on=["scenario", "seed"],
                how="inner",
            )

            if len(paired) == 0:
                continue

            diff = (
                paired["proposed_value"].to_numpy()
                - paired["baseline_value"].to_numpy()
            )

            if np.allclose(diff, 0.0):
                statistic = 0.0
                p_value = 1.0
            else:
                try:
                    result = wilcoxon(
                        paired["proposed_value"],
                        paired["baseline_value"],
                        zero_method="wilcox",
                        alternative="two-sided",
                    )
                    statistic = float(result.statistic)
                    p_value = float(result.pvalue)
                except ValueError:
                    statistic = 0.0
                    p_value = 1.0

            rows.append({
                "metric": metric,
                "proposed": proposed,
                "baseline": baseline,
                "n_pairs": len(paired),
                "proposed_mean": paired["proposed_value"].mean(),
                "baseline_mean": paired["baseline_value"].mean(),
                "mean_diff": float(np.mean(diff)),
                "median_diff": float(np.median(diff)),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            })

    out = pd.DataFrame(rows)

    if len(out):
        out["p_holm"] = holm_adjust(out["p_value"].to_numpy())
        out["significant_0_05"] = out["p_holm"] < 0.05

    return out


def run_statistics(
    csv_path: str | Path,
    proposed: str = "graph_ga",
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    result = paired_comparisons(df, proposed=proposed)

    output_path = csv_path.parent / "statistics.csv"
    result.to_csv(output_path, index=False)

    return result
