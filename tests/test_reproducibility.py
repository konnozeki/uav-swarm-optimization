import json

import pytest

from src.experiments.reproducibility import write_experiment_manifest


def test_manifest_can_resume_only_the_same_experiment(tmp_path):
    path = write_experiment_manifest(
        tmp_path,
        experiment="test",
        seeds=[0, 1],
        parameters={"population": 4},
    )
    original = json.loads(path.read_text(encoding="utf-8"))

    resumed = write_experiment_manifest(
        tmp_path,
        experiment="test",
        seeds=[0, 1],
        parameters={"population": 4},
    )

    assert resumed == path
    assert json.loads(path.read_text(encoding="utf-8")) == original

    with pytest.raises(ValueError, match="configuration changed"):
        write_experiment_manifest(
            tmp_path,
            experiment="test",
            seeds=[0, 1],
            parameters={"population": 5},
        )


def test_manifest_records_reproducibility_identity(tmp_path):
    path = write_experiment_manifest(
        tmp_path,
        experiment="identity",
        seeds=[3],
        parameters={"mode": "quick"},
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 2
    assert manifest["git_commit"]
    assert len(manifest["experiment_fingerprint"]) == 64
    assert len(manifest["requirements_lock_sha256"]) == 64
    assert manifest["packages"]["numpy"]
