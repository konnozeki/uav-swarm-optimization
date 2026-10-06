"""Realtime-style visualization for moving target nodes.

This script is intentionally separate from the static SOTA runners. It animates
mobile target nodes with different trajectories and replans a static deployment
for each frame so visual behavior can be inspected without mixing dynamic
movement into the report-scale static benchmark.

Example:
    python run_realtime_mobility_demo.py --frames 80 --output outputs/realtime_demo.gif
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle
import numpy as np

from run_static_coverage_study import obstacle_regions
from src.metrics import evaluate
from src.problem import ForbiddenRegion, Scenario
from src.proposed import ConnectedFrontierConfig, ConnectedFrontierLeafSwap


WIDTH = 1000.0
HEIGHT = 1000.0


def moving_nodes(t: float, n_nodes: int) -> np.ndarray:
    """Generate target nodes on intentionally mixed deterministic trajectories."""
    nodes = []
    for i in range(n_nodes):
        mode = i % 4
        phase = 2.0 * np.pi * i / max(n_nodes, 1)

        if mode == 0:
            center = np.array([260.0, 280.0])
            radius = 90.0 + 8.0 * (i % 3)
            point = center + radius * np.array([
                np.cos(t + phase),
                np.sin(t + phase),
            ])
        elif mode == 1:
            x = 160.0 + 680.0 * ((0.5 + 0.5 * np.sin(0.65 * t + phase)))
            y = 720.0 + 55.0 * np.sin(1.7 * t + phase)
            point = np.array([x, y])
        elif mode == 2:
            center = np.array([720.0, 300.0])
            point = center + np.array([
                135.0 * np.sin(t + phase),
                72.0 * np.sin(2.0 * t + phase),
            ])
        else:
            x = 500.0 + 260.0 * np.cos(0.45 * t + phase)
            y = 500.0 + 210.0 * np.sin(0.90 * t + 0.5 * phase)
            point = np.array([x, y])

        nodes.append(point)

    out = np.asarray(nodes, dtype=float)
    out[:, 0] = np.clip(out[:, 0], 40.0, WIDTH - 40.0)
    out[:, 1] = np.clip(out[:, 1], 40.0, HEIGHT - 40.0)
    return out


def make_dynamic_scenario(
    frame: int,
    *,
    frames: int,
    n_uavs: int,
    n_nodes: int,
    obstacles: tuple[ForbiddenRegion, ...],
) -> Scenario:
    t = 2.0 * np.pi * frame / max(frames - 1, 1)
    targets = moving_nodes(t, n_nodes)
    weights = 1.0 + 0.25 * np.sin(t + np.arange(n_nodes))
    return Scenario(
        name=f"dynamic_frame_{frame:03d}",
        pattern="dynamic",
        width=WIDTH,
        height=HEIGHT,
        targets=targets,
        target_weights=weights,
        n_uavs=n_uavs,
        sensing_radius=145.0,
        communication_radius=300.0,
        min_separation=28.0,
        seed=frame,
        forbidden_regions=obstacles,
    )


def draw_obstacles(ax, regions: tuple[ForbiddenRegion, ...]) -> None:
    for region in regions:
        if region.kind == "rectangle":
            assert region.bounds is not None
            xmin, ymin, xmax, ymax = region.bounds
            ax.add_patch(
                Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    facecolor="#3a3a3a",
                    edgecolor="#111111",
                    alpha=0.24,
                    linewidth=1.2,
                )
            )
        else:
            assert region.center is not None and region.radius is not None
            ax.add_patch(
                Circle(
                    region.center,
                    region.radius,
                    facecolor="#3a3a3a",
                    edgecolor="#111111",
                    alpha=0.24,
                    linewidth=1.2,
                )
            )


def precompute_frames(args) -> list[dict]:
    obstacles = obstacle_regions(args.obstacles)
    planner = ConnectedFrontierLeafSwap(
        ConnectedFrontierConfig(
            grid_size=args.grid_size,
            local_rounds=args.local_rounds,
        )
    )
    frames = []

    for frame in range(args.frames):
        scenario = make_dynamic_scenario(
            frame,
            frames=args.frames,
            n_uavs=args.n_uavs,
            n_nodes=args.n_nodes,
            obstacles=obstacles,
        )
        started = time.perf_counter()
        solution, _ = planner.solve(scenario, seed=frame)
        runtime = time.perf_counter() - started
        metrics = evaluate(scenario, solution)
        frames.append(
            dict(
                scenario=scenario,
                positions=solution.positions,
                coverage=metrics.weighted_coverage_ratio,
                feasible=metrics.feasible,
                runtime=runtime,
            )
        )

    return frames


def build_animation(frames: list[dict], output: Path, interval_ms: int) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    obstacles = frames[0]["scenario"].forbidden_regions

    def update(frame_id: int):
        frame = frames[frame_id]
        scenario = frame["scenario"]
        positions = frame["positions"]

        ax.clear()
        ax.set_xlim(0.0, scenario.width)
        ax.set_ylim(0.0, scenario.height)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.18)
        draw_obstacles(ax, obstacles)

        for point in positions:
            ax.add_patch(
                Circle(
                    point,
                    scenario.sensing_radius,
                    facecolor="#2f80ed",
                    edgecolor="#1b4f9c",
                    alpha=0.10,
                    linewidth=0.8,
                )
            )

        ax.scatter(
            scenario.targets[:, 0],
            scenario.targets[:, 1],
            s=36,
            c="#e4572e",
            label="moving nodes",
            zorder=3,
        )
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=78,
            c="#1f6f43",
            marker="^",
            label="UAV deployment",
            zorder=4,
        )

        if len(positions) > 1:
            delta = positions[:, None, :] - positions[None, :, :]
            distance = np.linalg.norm(delta, axis=2)
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    if distance[i, j] <= scenario.communication_radius + 1e-9:
                        ax.plot(
                            [positions[i, 0], positions[j, 0]],
                            [positions[i, 1], positions[j, 1]],
                            color="#1f6f43",
                            linewidth=1.0,
                            alpha=0.45,
                        )

        ax.set_title(
            "Realtime mobility demo | "
            f"frame={frame_id + 1}/{len(frames)} "
            f"coverage={frame['coverage']:.2f} "
            f"feasible={int(frame['feasible'])}"
        )
        ax.legend(loc="upper right", fontsize=8)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval_ms,
        repeat=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(
        output,
        writer=PillowWriter(fps=max(1, round(1000 / interval_ms))),
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--interval-ms", type=int, default=140)
    parser.add_argument("--n-uavs", type=int, default=6)
    parser.add_argument("--n-nodes", type=int, default=18)
    parser.add_argument("--grid-size", type=int, default=14)
    parser.add_argument("--local-rounds", type=int, default=6)
    parser.add_argument(
        "--obstacles",
        choices=("none", "blocks"),
        default="blocks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/realtime_mobility_demo.gif"),
    )
    args = parser.parse_args()

    if (
        args.frames < 2
        or args.interval_ms < 1
        or args.n_uavs < 2
        or args.n_nodes < 1
        or args.grid_size < 2
        or args.local_rounds < 0
    ):
        parser.error("invalid positive count/timing/grid argument")

    frames = precompute_frames(args)
    build_animation(frames, args.output, args.interval_ms)
    print(f"Wrote realtime mobility visualization: {args.output}")


if __name__ == "__main__":
    main()
