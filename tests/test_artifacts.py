from pathlib import Path

import pytest

from src.artifacts import LocalArtifactRegistry, artifact_id_for_uri, local_artifact_path
from src.capabilities import ImageGenerationRequest
from src.jobs import LocalJobRunner, LocalJobStore


def test_artifact_registry_lists_job_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.providers.ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr("src.artifacts.ARTIFACTS_DIR", tmp_path / "artifacts")
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    job = LocalJobRunner(store).run_mock_image(ImageGenerationRequest(prompt="SMC"))

    artifacts = LocalArtifactRegistry(store).list_artifacts()

    assert len(artifacts) == 1
    assert artifacts[0].job_id == job.job_id
    assert artifacts[0].kind == "image"
    assert artifacts[0].artifact_id == artifact_id_for_uri(job.artifacts[0]["uri"])
    assert local_artifact_path(artifacts[0].uri).exists()


def test_local_artifact_path_rejects_escaped_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.artifacts.ARTIFACTS_DIR", tmp_path / "artifacts")
    outside = tmp_path / "outside.txt"
    outside.write_text("no", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes"):
        local_artifact_path(outside.resolve().as_uri())
