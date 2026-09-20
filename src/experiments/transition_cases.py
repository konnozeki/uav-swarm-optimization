from __future__ import annotations

import numpy as np

from ..transition import TransitionProblem


DEFAULT_WIDTH = 1000.0
DEFAULT_HEIGHT = 1000.0


def ring_formation(
    center: tuple[float, float],
    n_uavs: int,
    radius: float,
    phase: float = 0.0,
) -> np.ndarray:
    """Create a regular ring formation."""
    angles = (
        phase
        + 2.0 * np.pi * np.arange(n_uavs) / n_uavs
    )
    center_array = np.asarray(center, dtype=float)

    offsets = np.column_stack([
        np.cos(angles),
        np.sin(angles),
    ]) * radius

    return center_array + offsets


def line_formation(
    center: tuple[float, float],
    n_uavs: int,
    spacing: float,
    angle: float = 0.0,
) -> np.ndarray:
    """Create an equally spaced line formation centered at center."""
    direction = np.array(
        [np.cos(angle), np.sin(angle)],
        dtype=float,
    )

    scalar = (
        np.arange(n_uavs, dtype=float)
        - 0.5 * (n_uavs - 1)
    ) * spacing

    return np.asarray(center, dtype=float) + scalar[:, None] * direction


def quick_transition_profile() -> list[TransitionProblem]:
    """Small deterministic profile for first CP3 validation.

    All cases are generic formation-A -> formation-B transitions.
    Formation A is not assumed to be a deployment base.
    """
    n = 6

    return [
        TransitionProblem(
            name="ring_translate_horizontal",
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            start_positions=ring_formation((250, 500), n, 130),
            goal_positions=ring_formation((750, 500), n, 130),
            communication_radius=220,
            min_separation=55,
            max_speed=40,
        ),
        TransitionProblem(
            name="ring_translate_diagonal",
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            start_positions=ring_formation((250, 250), n, 125),
            goal_positions=ring_formation((750, 750), n, 125),
            communication_radius=215,
            min_separation=55,
            max_speed=40,
        ),
        TransitionProblem(
            name="ring_expand_and_shift",
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            start_positions=ring_formation((300, 500), n, 80),
            goal_positions=ring_formation((700, 500), n, 170),
            communication_radius=225,
            min_separation=55,
            max_speed=40,
        ),
        TransitionProblem(
            name="ring_to_line",
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            start_positions=ring_formation((300, 500), n, 125),
            goal_positions=line_formation((700, 500), n, 110),
            communication_radius=220,
            min_separation=55,
            max_speed=40,
        ),
        TransitionProblem(
            name="line_rotate_and_shift",
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            start_positions=line_formation((300, 500), n, 105),
            goal_positions=line_formation(
                (700, 500),
                n,
                105,
                angle=np.pi / 2.0,
            ),
            communication_radius=215,
            min_separation=55,
            max_speed=40,
        ),
    ]


def stress_transition_profile() -> list[TransitionProblem]:
    """Sweep communication radius while endpoint formations remain valid.

    Geometry scales with Rc so that the endpoints themselves stay connected.
    The benchmark therefore stresses the transition mechanism rather than
    accidentally feeding it an invalid target formation.
    """
    cases: list[TransitionProblem] = []
    n = 6

    for rc in [170.0, 190.0, 220.0, 260.0]:
        ring_radius = 0.72 * rc

        cases.append(
            TransitionProblem(
                name=f"stress_ring_rc{int(rc)}",
                width=DEFAULT_WIDTH,
                height=DEFAULT_HEIGHT,
                start_positions=ring_formation(
                    (260, 500),
                    n,
                    ring_radius,
                ),
                goal_positions=ring_formation(
                    (740, 500),
                    n,
                    ring_radius,
                    phase=np.pi / n,
                ),
                communication_radius=rc,
                min_separation=55,
                max_speed=40,
            )
        )

        # Keep the line inside the 1000 x 1000 map even at the largest Rc.
        # 0.45 * 260 * 2.5 = 292.5, so the leftmost UAV still stays inside
        # the map when the line is centered at x = 300.
        spacing = 0.45 * rc

        cases.append(
            TransitionProblem(
                name=f"stress_line_rc{int(rc)}",
                width=DEFAULT_WIDTH,
                height=DEFAULT_HEIGHT,
                start_positions=line_formation(
                    (300, 500),
                    n,
                    spacing,
                    angle=0.0,
                ),
                goal_positions=line_formation(
                    (700, 500),
                    n,
                    spacing,
                    angle=np.pi / 2.0,
                ),
                communication_radius=rc,
                min_separation=55,
                max_speed=40,
            )
        )

    return cases
