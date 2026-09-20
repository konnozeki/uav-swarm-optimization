from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

from .problem import Scenario, Solution
from .metrics import coverage_counts, evaluate
from .graph_ops import communication_graph


def plot_solution(
    scenario: Scenario,
    solution: Solution,
    output_path: str | Path | None = None,
    show: bool = False,
) -> None:
    counts = coverage_counts(scenario, solution.positions)
    covered = counts >= 1
    graph = communication_graph(scenario, solution.positions)
    metrics = evaluate(scenario, solution)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        scenario.targets[~covered, 0],
        scenario.targets[~covered, 1],
        marker="x",
        label="Uncovered target",
    )
    ax.scatter(
        scenario.targets[covered, 0],
        scenario.targets[covered, 1],
        marker=".",
        label="Covered target",
    )

    for i, j in graph.edges:
        p1 = solution.positions[i]
        p2 = solution.positions[j]
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            linewidth=1,
        )

    ax.scatter(
        solution.positions[:, 0],
        solution.positions[:, 1],
        marker="^",
        s=85,
        label="UAV",
    )

    for idx, (x, y) in enumerate(solution.positions):
        circle = plt.Circle(
            (x, y),
            scenario.sensing_radius,
            fill=False,
            alpha=0.22,
        )
        ax.add_patch(circle)
        ax.text(x + 7, y + 7, f"U{idx}", fontsize=8)

    ax.set_xlim(0, scenario.width)
    ax.set_ylim(0, scenario.height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)

    ax.set_title(
        f"{solution.algorithm} | "
        f"cov={metrics.weighted_coverage_ratio:.3f} | "
        f"conn={metrics.connected} | "
        f"red={metrics.redundancy_excess:.3f}"
    )

    ax.legend(loc="upper right")
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)

    if show:
        plt.show()

    plt.close(fig)
