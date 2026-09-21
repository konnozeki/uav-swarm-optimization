from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path


PACKAGES = [
    "numpy",
    "pandas",
    "matplotlib",
    "networkx",
    "scipy",
    "plotly",
]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def write_experiment_manifest(
    output_dir: str | Path,
    *,
    experiment: str,
    seeds,
    parameters: dict,
) -> Path:
    """Write enough environment/config metadata to identify an experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None

    manifest = {
        "experiment": experiment,
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "seeds": list(seeds),
        "parameters": parameters,
    }

    path = output_dir / "experiment_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path
