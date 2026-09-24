from __future__ import annotations

import hashlib
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


def _git_dirty() -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(output.strip())
    except Exception:
        return None


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(value: dict) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_experiment_manifest(
    output_dir: str | Path,
    *,
    experiment: str,
    seeds,
    parameters: dict,
    require_clean: bool = False,
) -> Path:
    """Create or verify the immutable identity of an experiment directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds)
    git_commit = _git_commit()
    git_dirty = _git_dirty()
    if require_clean and git_dirty is not False:
        state = "unknown" if git_dirty is None else "dirty"
        raise RuntimeError(
            f"final evidence requires a clean git worktree; current state is {state}"
        )

    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None

    identity = {
        "experiment": experiment,
        "git_commit": git_commit,
        "seeds": seeds,
        "parameters": parameters,
    }
    manifest = {
        "schema_version": 2,
        **identity,
        "experiment_fingerprint": _fingerprint(identity),
        "git_dirty": git_dirty,
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "requirements_lock_sha256": _sha256(Path("requirements-lock.txt")),
    }

    path = output_dir / "experiment_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("schema_version") != manifest["schema_version"]:
            raise ValueError(
                f"cannot resume {output_dir}: unsupported or legacy manifest"
            )
        if existing.get("experiment_fingerprint") != manifest["experiment_fingerprint"]:
            raise ValueError(
                f"cannot resume {output_dir}: experiment configuration changed"
            )
        if existing.get("requirements_lock_sha256") != manifest["requirements_lock_sha256"]:
            raise ValueError(
                f"cannot resume {output_dir}: dependency lock changed"
            )
        return path

    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path
