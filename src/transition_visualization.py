from __future__ import annotations

from pathlib import Path

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .graph_ops import communication_graph_from_positions
from .obstacles import formation_obstacle_free
from .transition import (
    TransitionProblem,
    TransitionSolution,
    formation_min_distance,
)


def _covered_targets(
    positions: np.ndarray,
    targets: np.ndarray | None,
    sensing_radius: float | None,
) -> np.ndarray | None:
    """Return a boolean target-coverage mask for visualization only.

    Coverage shown in the animation is intentionally binary and only used as a
    visual aid. Research metrics still come from the normal objective/metrics
    pipeline rather than from plotting code.
    """
    if targets is None or sensing_radius is None:
        return None

    targets = np.asarray(targets, dtype=float)
    positions = np.asarray(positions, dtype=float)

    if len(targets) == 0:
        return np.zeros(0, dtype=bool)

    delta = targets[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    return np.any(distance <= sensing_radius, axis=1)


def _communication_edges(
    positions: np.ndarray,
    communication_radius: float,
) -> list[tuple[int, int]]:
    """Return communication edges for one animation frame."""
    graph = communication_graph_from_positions(
        positions,
        communication_radius,
    )
    return [
        (int(i), int(j))
        for i, j in graph.edges()
    ]


def _draw_common_frame(
    ax,
    problem: TransitionProblem,
    solution: TransitionSolution,
    frame_idx: int,
    targets: np.ndarray | None = None,
    sensing_radius: float | None = None,
    target_weights: np.ndarray | None = None,
    title: str | None = None,
) -> None:
    """Draw one complete animation frame using basic Matplotlib primitives.

    The function redraws the axes from scratch. This is intentionally simpler
    than a highly optimized artist-update pipeline because CP3 trajectories are
    short and the visualization is primarily for interpretation/debugging.
    """
    trajectory = np.asarray(solution.trajectory, dtype=float)
    positions = trajectory[frame_idx]
    n_uavs = positions.shape[0]

    ax.clear()

    for obstacle in problem.obstacles:
        ax.add_patch(
            Rectangle(
                (obstacle.x_min, obstacle.y_min),
                obstacle.x_max - obstacle.x_min,
                obstacle.y_max - obstacle.y_min,
                facecolor="0.55",
                edgecolor="0.25",
                alpha=0.30,
                linewidth=1.2,
                label="obstacle",
            )
        )

        if problem.obstacle_clearance > 0:
            x_min, y_min, x_max, y_max = obstacle.expanded(
                problem.obstacle_clearance
            )
            ax.add_patch(
                Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    facecolor="none",
                    edgecolor="0.45",
                    alpha=0.55,
                    linewidth=1.0,
                    linestyle="--",
                    label="obstacle clearance",
                )
            )

    # Targets are kept visually quiet so UAV motion and graph structure remain
    # the primary information. Covered targets are emphasized slightly.
    coverage_mask = _covered_targets(
        positions,
        targets,
        sensing_radius,
    )

    if targets is not None:
        targets = np.asarray(targets, dtype=float)

        if target_weights is None:
            sizes = np.full(len(targets), 12.0)
        else:
            target_weights = np.asarray(target_weights, dtype=float)
            normalized = target_weights / max(
                float(np.max(target_weights)),
                1e-12,
            )
            sizes = 10.0 + 16.0 * normalized

        ax.scatter(
            targets[:, 0],
            targets[:, 1],
            s=sizes,
            marker=".",
            alpha=0.25,
            label="targets",
        )

        if coverage_mask is not None and np.any(coverage_mask):
            covered = targets[coverage_mask]
            ax.scatter(
                covered[:, 0],
                covered[:, 1],
                s=sizes[coverage_mask],
                marker=".",
                alpha=0.80,
                label="covered targets",
            )

    # Start and goal formations remain visible for the whole animation.
    ax.scatter(
        trajectory[0, :, 0],
        trajectory[0, :, 1],
        marker="o",
        facecolors="none",
        s=70,
        linewidths=1.2,
        label="formation A",
    )
    ax.scatter(
        solution.assigned_goals[:, 0],
        solution.assigned_goals[:, 1],
        marker="x",
        s=70,
        linewidths=1.5,
        label="formation B",
    )

    # Draw the current communication graph first so UAV markers stay on top.
    communication_edges = _communication_edges(
        positions,
        problem.communication_radius,
    )

    for i, j in communication_edges:
        ax.plot(
            [positions[i, 0], positions[j, 0]],
            [positions[i, 1], positions[j, 1]],
            linewidth=1.0,
            alpha=0.35,
        )

    # If the proposed planner supplied a certified backbone for the current
    # segment, overlay those N-1 protected edges more strongly.
    if (
        frame_idx < len(solution.backbones)
        and solution.backbones[frame_idx]
    ):
        for i, j in solution.backbones[frame_idx]:
            ax.plot(
                [positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                linewidth=2.0,
                alpha=0.75,
            )

    # Reuse Matplotlib's default color cycle automatically. No custom palette is
    # required for a research/debug visualization.
    for uav_idx in range(n_uavs):
        path = trajectory[: frame_idx + 1, uav_idx, :]

        line, = ax.plot(
            path[:, 0],
            path[:, 1],
            linewidth=1.2,
            alpha=0.65,
        )

        ax.scatter(
            positions[uav_idx, 0],
            positions[uav_idx, 1],
            s=62,
            color=line.get_color(),
            zorder=5,
        )

        ax.text(
            positions[uav_idx, 0] + 8.0,
            positions[uav_idx, 1] + 8.0,
            str(uav_idx),
            fontsize=8,
        )

    graph = communication_graph_from_positions(
        positions,
        problem.communication_radius,
    )
    connected = nx.is_connected(graph)
    min_distance = formation_min_distance(positions)
    obstacle_clear = formation_obstacle_free(
        positions,
        problem.obstacles,
        problem.obstacle_clearance,
    )
    elapsed = frame_idx * problem.dt

    if coverage_mask is None:
        coverage_text = "coverage: n/a"
    elif len(coverage_mask) == 0:
        coverage_text = "coverage: n/a"
    else:
        coverage_text = (
            f"coverage: {np.mean(coverage_mask):.1%}"
        )

    status = (
        f"t = {elapsed:.1f}s\n"
        f"connected: {'yes' if connected else 'no'}\n"
        f"obstacle clear: {'yes' if obstacle_clear else 'no'}\n"
        f"min separation: {min_distance:.1f}\n"
        f"{coverage_text}"
    )

    ax.text(
        0.015,
        0.985,
        status,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )

    ax.set_xlim(0.0, problem.width)
    ax.set_ylim(0.0, problem.height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)

    if title is None:
        title = (
            f"{problem.name} | {solution.planner}"
        )

    ax.set_title(
        f"{title} | frame {frame_idx}/{len(trajectory) - 1}"
    )

    # Keep the legend compact. It explains the persistent marks; per-UAV labels
    # are written next to the UAVs instead of creating N legend entries.
    ax.legend(
        fontsize=8,
        loc="lower right",
    )


def save_transition_plot(
    problem: TransitionProblem,
    solution: TransitionSolution,
    output_path: str | Path,
    title: str | None = None,
    targets: np.ndarray | None = None,
    sensing_radius: float | None = None,
    target_weights: np.ndarray | None = None,
) -> None:
    """Save a static overview of a complete A -> B swarm transition."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trajectory = np.asarray(solution.trajectory, dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 7.0))

    # Draw the final frame first, then overlay all full trajectories. This keeps
    # the static plot semantically aligned with the animation.
    _draw_common_frame(
        ax,
        problem,
        solution,
        frame_idx=len(trajectory) - 1,
        targets=targets,
        sensing_radius=sensing_radius,
        target_weights=target_weights,
        title=title,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_transition_animation(
    problem: TransitionProblem,
    solution: TransitionSolution,
    output_path: str | Path,
    title: str | None = None,
    targets: np.ndarray | None = None,
    sensing_radius: float | None = None,
    target_weights: np.ndarray | None = None,
    fps: int = 2,
    hold_final_frames: int = 2,
) -> None:
    """Save a lightweight GIF animation of a swarm formation transition.

    GIF is used deliberately because Matplotlib + Pillow is enough on Windows
    and Linux; no external ffmpeg installation is required.

    Parameters
    ----------
    hold_final_frames:
        Number of extra copies of the final state appended to the animation so
        the destination formation remains visible briefly before the GIF loops.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if hold_final_frames < 0:
        raise ValueError("hold_final_frames must be non-negative")

    output_path = Path(output_path)

    if output_path.suffix.lower() != ".gif":
        raise ValueError(
            "save_transition_animation currently supports .gif output only"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    trajectory = np.asarray(solution.trajectory, dtype=float)
    base_frames = list(range(len(trajectory)))
    frames = base_frames + (
        [len(trajectory) - 1] * hold_final_frames
    )

    fig, ax = plt.subplots(figsize=(7.0, 7.0))

    def update(frame_idx: int):
        _draw_common_frame(
            ax,
            problem,
            solution,
            frame_idx=frame_idx,
            targets=targets,
            sensing_radius=sensing_radius,
            target_weights=target_weights,
            title=title,
        )
        return ()

    animation = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 / fps,
        repeat=True,
        blit=False,
    )

    animation.save(
        output_path,
        writer=PillowWriter(fps=fps),
    )
    plt.close(fig)
