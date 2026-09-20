from __future__ import annotations

import numpy as np

from .problem import Scenario


DEFAULT_WIDTH = 1000.0
DEFAULT_HEIGHT = 1000.0


def _uniform(rng, n_targets, width, height):
    return np.column_stack([
        rng.uniform(0, width, n_targets),
        rng.uniform(0, height, n_targets),
    ])


def _clustered(rng, n_targets, width, height):
    centers = np.array([
        [0.25 * width, 0.28 * height],
        [0.75 * width, 0.32 * height],
        [0.55 * width, 0.76 * height],
    ])

    ids = rng.integers(0, len(centers), size=n_targets)
    targets = centers[ids] + rng.normal(
        0,
        0.09 * min(width, height),
        size=(n_targets, 2),
    )
    return targets


def _split(rng, n_targets, width, height):
    n_left = n_targets // 2
    n_right = n_targets - n_left

    left = np.array([0.20 * width, 0.50 * height]) + rng.normal(
        0, 0.075 * min(width, height), size=(n_left, 2)
    )
    right = np.array([0.80 * width, 0.50 * height]) + rng.normal(
        0, 0.075 * min(width, height), size=(n_right, 2)
    )
    return np.vstack([left, right])


def _corridor(rng, n_targets, width, height):
    x = rng.uniform(0.08 * width, 0.92 * width, n_targets)
    y_center = 0.5 * height + 0.16 * height * np.sin(2 * np.pi * x / width)
    y = y_center + rng.normal(0, 0.04 * height, n_targets)
    return np.column_stack([x, y])


def make_scenario(
    pattern: str,
    seed: int,
    n_uavs: int,
    n_targets: int,
    communication_radius: float,
    sensing_radius: float = 175.0,
    min_separation: float = 55.0,
    width: float = DEFAULT_WIDTH,
    height: float = DEFAULT_HEIGHT,
) -> Scenario:
    rng = np.random.default_rng(seed)

    if pattern == "uniform":
        targets = _uniform(rng, n_targets, width, height)
    elif pattern == "clustered":
        targets = _clustered(rng, n_targets, width, height)
    elif pattern == "split":
        targets = _split(rng, n_targets, width, height)
    elif pattern == "corridor":
        targets = _corridor(rng, n_targets, width, height)
    elif pattern == "hotspot":
        targets = _uniform(rng, n_targets, width, height)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    targets[:, 0] = np.clip(targets[:, 0], 0, width)
    targets[:, 1] = np.clip(targets[:, 1], 0, height)

    weights = np.ones(n_targets, dtype=float)

    if pattern == "hotspot":
        hotspot = np.array([0.78 * width, 0.78 * height])
        d = np.linalg.norm(targets - hotspot, axis=1)
        weights[d <= 0.25 * min(width, height)] = 3.0

    return Scenario(
        name=(
            f"{pattern}"
            f"_u{n_uavs}"
            f"_m{n_targets}"
            f"_rc{int(communication_radius)}"
            f"_s{seed}"
        ),
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
    )


def quick_profile(seed_base: int = 2000) -> list[Scenario]:
    specs = [
        ("uniform", 5, 60, 320),
        ("uniform", 8, 100, 300),
        ("clustered", 5, 70, 320),
        ("clustered", 8, 110, 280),
        ("hotspot", 6, 90, 300),
        ("split", 7, 100, 260),
        ("corridor", 6, 90, 250),
        ("split", 9, 140, 230),
    ]

    return [
        make_scenario(
            pattern=pattern,
            seed=seed_base + i,
            n_uavs=n_uavs,
            n_targets=n_targets,
            communication_radius=rc,
        )
        for i, (pattern, n_uavs, n_targets, rc) in enumerate(specs)
    ]


def stress_profile(seed_base: int = 3000) -> list[Scenario]:
    scenarios = []
    idx = 0

    for pattern in ["split", "corridor"]:
        for n_uavs in [4, 6, 8]:
            for rc in [200, 250, 300, 350]:
                scenarios.append(
                    make_scenario(
                        pattern=pattern,
                        seed=seed_base + idx,
                        n_uavs=n_uavs,
                        n_targets=120,
                        communication_radius=rc,
                    )
                )
                idx += 1

    return scenarios


def full_profile(seed_base: int = 4000) -> list[Scenario]:
    """
    Full factorial benchmark generator.

    4 patterns * 6 N * 4 M * 5 Rc = 480 scenario templates.
    Mỗi scenario lại chạy nhiều algorithm và nhiều seed, nên rất nặng.
    """
    scenarios = []
    idx = 0

    for pattern in ["uniform", "clustered", "hotspot", "split"]:
        for n_uavs in [4, 6, 8, 10, 12, 16]:
            for n_targets in [50, 100, 200, 500]:
                for rc in [200, 250, 300, 350, 400]:
                    scenarios.append(
                        make_scenario(
                            pattern=pattern,
                            seed=seed_base + idx,
                            n_uavs=n_uavs,
                            n_targets=n_targets,
                            communication_radius=rc,
                        )
                    )
                    idx += 1

    return scenarios
