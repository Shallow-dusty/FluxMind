import hashlib
from pathlib import Path

import pytest

from src.artifacts import (
    LocalArtifactRegistry,
    artifact_id_for_uri,
    format_artifact_references,
    local_artifact_path,
)
from src.capabilities import CodeExecutionRequest, ImageGenerationRequest
from src.jobs import JobRecord, LocalJobRunner, LocalJobStore


@pytest.fixture(autouse=True)
def no_runtime_event_disk_writes(monkeypatch):
    monkeypatch.setattr("src.jobs.append_runtime_event", lambda **_kwargs: None)


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
    assert artifacts[0].metadata["prompt"] == "SMC"
    assert artifacts[0].metadata["model"] == "local-mock-svg-v1"
    assert artifacts[0].metadata["checksum_sha256"]
    assert int(artifacts[0].metadata["byte_count"]) > 0
    assert artifacts[0].metadata["cost_estimate_usd"] == "0"
    assert local_artifact_path(artifacts[0].uri).exists()
    assert LocalArtifactRegistry(store).storage_status()["sqlite_rows"] == 1


def test_artifact_registry_carries_job_ownership(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.providers.ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr("src.artifacts.ARTIFACTS_DIR", tmp_path / "artifacts")
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    LocalJobRunner(store).run_mock_image(
        ImageGenerationRequest(prompt="Owned SMC diagram"),
        ownership={"owner_id": "lab-art", "owner_label": "Artifact Lab"},
    )

    registry = LocalArtifactRegistry(store)
    artifacts = registry.list_artifacts(owner_id="lab-art")

    assert len(artifacts) == 1
    assert artifacts[0].owner_id == "lab-art"
    assert artifacts[0].owner_label == "Artifact Lab"
    assert artifacts[0].ownership_source == "request"
    assert registry.list_artifacts(q="Artifact Lab")[0].owner_id == "lab-art"
    assert registry.list_artifacts(owner_id="missing") == []


def test_artifact_registry_filters_local_metadata(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.providers.ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr("src.artifacts.ARTIFACTS_DIR", tmp_path / "artifacts")
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)
    runner.run_mock_image(ImageGenerationRequest(prompt="SMC observer diagram"))
    text_job = runner.run_local_python(
        request=CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "from pathlib import Path\nPath('notes.txt').write_text('artifact notes')\n"},
        )
    )

    registry = LocalArtifactRegistry(store)

    assert [artifact.kind for artifact in registry.list_artifacts(kind="image")] == ["image"]
    assert [artifact.job_kind for artifact in registry.list_artifacts(job_kind="code_execution")] == ["code_execution"]
    assert registry.list_artifacts(q="observer")[0].job_kind == "image_generation"
    assert registry.list_artifacts(q=text_job.job_id)[0].title == "notes.txt"
    assert registry.list_artifacts(kind="missing") == []


def test_local_artifact_path_rejects_escaped_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.artifacts.ARTIFACTS_DIR", tmp_path / "artifacts")
    outside = tmp_path / "outside.txt"
    outside.write_text("no", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes"):
        local_artifact_path(outside.resolve().as_uri())


def test_format_artifact_references_exposes_stable_ids(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.providers.ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr("src.artifacts.ARTIFACTS_DIR", tmp_path / "artifacts")
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    LocalJobRunner(store).run_mock_image(
        ImageGenerationRequest(
            prompt="Sliding mode observer block diagram",
            style="control-diagram",
            diagram_template="sliding-mode-observer",
            reference_uris=["paper://smc#page=3"],
        )
    )

    artifacts = LocalArtifactRegistry(store).list_artifacts()
    context = format_artifact_references(artifacts)

    assert f"[Artifact:{artifacts[0].artifact_id}]" in context
    assert "control-diagram" in context
    assert "template=sliding-mode-observer" in context
    assert "Sliding mode observer block diagram" in context
    assert "paper://smc#page=3" in context


def test_artifact_registry_sqlite_mirror_removes_stale_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.providers.ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr("src.artifacts.ARTIFACTS_DIR", tmp_path / "artifacts")
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)
    first = runner.run_mock_image(ImageGenerationRequest(prompt="first"))
    second = runner.run_mock_image(ImageGenerationRequest(prompt="second"))
    registry = LocalArtifactRegistry(store)

    assert registry.storage_status()["sqlite_rows"] == 2
    first.artifacts = []
    second.artifacts = []
    store.append(first)
    store.append(second)

    assert registry.list_artifacts() == []
    assert registry.storage_status()["sqlite_rows"] == 0


def test_artifact_registry_integrity_status_detects_mismatches(tmp_path: Path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr("src.artifacts.ARTIFACTS_DIR", artifact_root)
    artifact_root.mkdir()
    ok_artifact = artifact_root / "ok.txt"
    ok_artifact.write_text("ok", encoding="utf-8")
    changed_artifact = artifact_root / "changed.txt"
    changed_artifact.write_text("new", encoding="utf-8")
    missing_artifact = artifact_root / "missing.txt"
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    store.append(
        JobRecord(
            job_id="job-integrity",
            kind="code_execution",
            status="succeeded",
            created_at="2026-06-01T00:00:00+00:00",
            updated_at="2026-06-01T00:00:01+00:00",
            request={},
            artifacts=[
                {
                    "kind": "text",
                    "uri": ok_artifact.resolve().as_uri(),
                    "mime_type": "text/plain",
                    "title": "ok.txt",
                    "metadata": {
                        "checksum_sha256": hashlib.sha256(b"ok").hexdigest(),
                        "byte_count": "2",
                    },
                },
                {
                    "kind": "text",
                    "uri": changed_artifact.resolve().as_uri(),
                    "mime_type": "text/plain",
                    "title": "changed.txt",
                    "metadata": {
                        "checksum_sha256": hashlib.sha256(b"old").hexdigest(),
                        "byte_count": "3",
                    },
                },
                {
                    "kind": "text",
                    "uri": missing_artifact.resolve().as_uri(),
                    "mime_type": "text/plain",
                    "title": "missing.txt",
                    "metadata": {
                        "checksum_sha256": hashlib.sha256(b"missing").hexdigest(),
                        "byte_count": "7",
                    },
                },
                {
                    "kind": "file",
                    "uri": ok_artifact.resolve().as_uri(),
                    "mime_type": "application/octet-stream",
                    "title": "unchecked.bin",
                    "metadata": {},
                },
            ],
        )
    )
    registry = LocalArtifactRegistry(store, db_path=artifact_root / "artifacts.sqlite3")

    status = registry.integrity_status()

    assert status["checked"] == 2
    assert status["ok"] == 1
    assert status["missing"] == 1
    assert status["unchecked"] == 1
    assert status["checksum_mismatch"] == 1
    assert status["byte_count_mismatch"] == 0
    assert len(status["issue_artifact_ids"]) == 3
