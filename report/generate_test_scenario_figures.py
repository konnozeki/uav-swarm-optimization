from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_static_coverage_study import make_scenario
from run_static_sota_suite import (
    BASE_K,
    BASE_MIN_SEPARATION,
    BASE_N,
    BASE_RC,
    BASE_RS,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "test_scenario_figures"
BASE_PATTERNS = ("uniform", "clustered", "chain")
HARD_PATTERNS = (
    "hard_chain",
    "islands",
    "corner_clusters",
    "few_uavs_many_targets",
)
OBSTACLE_PATTERNS = BASE_PATTERNS


def make_report_scenario(pattern: str, obstacle_layout: str = "none"):
    return make_scenario(
        pattern,
        seed=0,
        n_uavs=BASE_N,
        n_targets=BASE_K,
        sensing_radius=BASE_RS,
        communication_radius=BASE_RC,
        min_separation=BASE_MIN_SEPARATION,
        obstacle_layout=obstacle_layout,
    )


def draw_forbidden_regions(ax, scenario) -> None:
    for region in scenario.forbidden_regions:
        if region.kind == "rectangle":
            xmin, ymin, xmax, ymax = region.bounds
            patch = Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                facecolor="#f97316",
                edgecolor="#9a3412",
                linewidth=1.4,
                alpha=0.28,
                hatch="///",
                zorder=1,
            )
        else:
            patch = Circle(
                region.center,
                region.radius,
                facecolor="#f97316",
                edgecolor="#9a3412",
                linewidth=1.4,
                alpha=0.28,
                hatch="///",
                zorder=1,
            )
        ax.add_patch(patch)


def draw_scenario(ax, scenario, title: str) -> None:
    draw_forbidden_regions(ax, scenario)
    weights = scenario.target_weights
    sizes = 18 + 28 * (weights - weights.min()) / max(np.ptp(weights), 1.0)
    ax.scatter(
        scenario.targets[:, 0],
        scenario.targets[:, 1],
        s=sizes,
        c="#2563eb",
        edgecolors="#0f172a",
        linewidths=0.25,
        alpha=0.82,
        zorder=2,
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlim(0, scenario.width)
    ax.set_ylim(0, scenario.height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_yticks([0, 250, 500, 750, 1000])
    ax.grid(True, color="#cbd5e1", linewidth=0.6, alpha=0.7)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(0.9)


def save_single(pattern: str, obstacle_layout: str = "none") -> Path:
    scenario = make_report_scenario(pattern, obstacle_layout)
    suffix = "_with_obstacles" if obstacle_layout != "none" else ""
    title = pattern.replace("_", " ") + (" + no-deploy regions" if suffix else "")
    path = OUTPUT_DIR / f"{pattern}{suffix}.png"

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=180)
    draw_scenario(ax, scenario, title)
    fig.tight_layout(pad=0.7)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def save_contact_sheet(items: list[tuple[str, str]], filename: str, ncols: int) -> Path:
    nrows = (len(items) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 4.1 * nrows), dpi=170)
    axes = list(axes.flat if hasattr(axes, "flat") else [axes])

    for ax, (pattern, obstacle_layout) in zip(axes, items):
        scenario = make_report_scenario(pattern, obstacle_layout)
        suffix = " + no-deploy" if obstacle_layout != "none" else ""
        draw_scenario(ax, scenario, pattern.replace("_", " ") + suffix)

    for ax in axes[len(items):]:
        ax.axis("off")

    fig.suptitle(
        f"Test scenarios, N={BASE_N}, K={BASE_K}, Rs={BASE_RS:g}, Rc={BASE_RC:g}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97), pad=1.0)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    clean_items = [(pattern, "none") for pattern in BASE_PATTERNS + HARD_PATTERNS]
    obstacle_items = [(pattern, "blocks") for pattern in OBSTACLE_PATTERNS]

    for pattern, obstacle_layout in clean_items + obstacle_items:
        generated.append(save_single(pattern, obstacle_layout))

    generated.append(save_contact_sheet(clean_items, "all_clean_scenarios.png", ncols=4))
    generated.append(save_contact_sheet(obstacle_items, "obstacle_scenarios.png", ncols=3))

    for path in generated:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
