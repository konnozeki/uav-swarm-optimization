from __future__ import annotations

import numpy as np

from ..problem import Scenario
from ..obstacles import AxisAlignedRectangle
from ..reconfiguration import ReconfigurationProblem
from ..scenarios import make_scenario
from .transition_cases import ring_formation, line_formation


def quick_reconfiguration_profile() -> list[ReconfigurationProblem]:
    """Small joint A -> optimized-B profile.

    The target maps reuse CP2 scenario generation. The current swarm formation A
    is deliberately different from the target geometry so the optimizer must
    trade final sensing quality against reconfiguration time.
    """
    specs = [
        # pattern, scenario seed, start kind, Rc
        ("uniform", 5100, "ring_left", 300.0),
        ("clustered", 5101, "ring_left", 280.0),
        ("split", 5102, "line_bottom", 260.0),
        ("corridor", 5103, "ring_right", 250.0),
    ]

    problems: list[ReconfigurationProblem] = []

    for index, (pattern, seed, start_kind, rc) in enumerate(specs):
        scenario = make_scenario(
            pattern=pattern,
            seed=seed,
            n_uavs=6,
            n_targets=90,
            communication_radius=rc,
            sensing_radius=175.0,
            min_separation=55.0,
        )

        # Radius 0.48*Rc keeps neighboring vertices of a six-UAV ring well
        # inside communication range while leaving enough spatial extent that A
        # is a genuine formation rather than a co-located "base".
        ring_radius = 0.48 * rc

        if start_kind == "ring_left":
            start = ring_formation(
                (250.0, 500.0),
                scenario.n_uavs,
                ring_radius,
            )
        elif start_kind == "ring_right":
            start = ring_formation(
                (750.0, 500.0),
                scenario.n_uavs,
                ring_radius,
            )
        elif start_kind == "line_bottom":
            start = line_formation(
                (500.0, 250.0),
                scenario.n_uavs,
                spacing=0.42 * rc,
                angle=0.0,
            )
        else:
            raise ValueError(f"unknown start kind: {start_kind}")

        problems.append(
            ReconfigurationProblem(
                name=f"joint_{pattern}_{index}",
                scenario=scenario,
                start_positions=start,
                max_speed=40.0,
                dt=1.0,
                allow_reassignment=True,
            )
        )

    return problems


def stress_reconfiguration_profile() -> list[ReconfigurationProblem]:
    """Connectivity stress profile for joint optimization.

    The same target-map family is evaluated across several communication radii.
    This tests whether transition-aware selection becomes more useful as the
    feasible motion corridor narrows.
    """
    problems: list[ReconfigurationProblem] = []
    index = 0

    for pattern in ["split", "corridor"]:
        for rc in [220.0, 260.0, 320.0]:
            scenario = make_scenario(
                pattern=pattern,
                seed=6200 + index,
                n_uavs=6,
                n_targets=110,
                communication_radius=rc,
                sensing_radius=175.0,
                min_separation=55.0,
            )

            start = ring_formation(
                (250.0, 500.0),
                scenario.n_uavs,
                radius=0.48 * rc,
            )

            problems.append(
                ReconfigurationProblem(
                    name=(
                        f"joint_stress_{pattern}"
                        f"_rc{int(rc)}"
                    ),
                    scenario=scenario,
                    start_positions=start,
                    max_speed=40.0,
                    dt=1.0,
                    allow_reassignment=True,
                )
            )
            index += 1

    return problems


def showcase_reconfiguration_profile() -> list[ReconfigurationProblem]:
    """One presentation-friendly CP3 case with denser geometry.

    The case is intentionally more crowded than the quick benchmark:

    - 10 UAVs instead of 6;
    - 180 sensing targets;
    - two dense target regions plus a bridge/corridor between them;
    - tighter minimum separation, so crossing/packing decisions matter;
    - a compact connected start formation below the sensing regions.

    It is not a statistical benchmark. The purpose is to exercise and visualize
    coverage, communication connectivity, collision avoidance and transition
    cost together in one deterministic scenario.
    """
    rng = np.random.default_rng(7300)

    n_left = 70
    n_right = 70
    n_bridge = 40

    left = np.array([220.0, 650.0]) + rng.normal(
        0.0,
        [85.0, 90.0],
        size=(n_left, 2),
    )
    right = np.array([780.0, 650.0]) + rng.normal(
        0.0,
        [85.0, 90.0],
        size=(n_right, 2),
    )

    bridge_x = rng.uniform(300.0, 700.0, n_bridge)
    bridge_y = (
        450.0
        + 55.0 * np.sin(2.0 * np.pi * bridge_x / 400.0)
        + rng.normal(0.0, 28.0, n_bridge)
    )
    bridge = np.column_stack([
        bridge_x,
        bridge_y,
    ])

    targets = np.vstack([
        left,
        right,
        bridge,
    ])

    targets[:, 0] = np.clip(
        targets[:, 0],
        25.0,
        975.0,
    )
    targets[:, 1] = np.clip(
        targets[:, 1],
        25.0,
        975.0,
    )

    scenario = Scenario(
        name="showcase_dense_bridge",
        pattern="showcase_dense_bridge",
        width=1000.0,
        height=1000.0,
        targets=targets,
        target_weights=np.ones(len(targets), dtype=float),
        n_uavs=10,
        sensing_radius=150.0,
        communication_radius=215.0,
        min_separation=75.0,
        seed=7300,
    )

    # A compact ring keeps the initial graph safely connected while making the
    # eventual move toward the two upper target regions non-trivial. With ten
    # UAVs, adjacent ring spacing is about 96 units, above min_separation=75.
    start = ring_formation(
        (500.0, 220.0),
        scenario.n_uavs,
        radius=155.0,
        phase=np.pi / scenario.n_uavs,
    )

    obstacles = (
        AxisAlignedRectangle(
            420.0,
            405.0,
            580.0,
            675.0,
            name="central_block",
        ),
        AxisAlignedRectangle(
            155.0,
            505.0,
            285.0,
            610.0,
            name="left_block",
        ),
        AxisAlignedRectangle(
            745.0,
            505.0,
            875.0,
            610.0,
            name="right_block",
        ),
    )

    return [
        ReconfigurationProblem(
            name="showcase_dense_bridge",
            scenario=scenario,
            start_positions=start,
            max_speed=40.0,
            dt=1.0,
            allow_reassignment=True,
            obstacles=obstacles,
            obstacle_clearance=18.0,
        )
    ]
