import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from src.artifacts import (
    ArtifactRecord,
    LocalArtifactRegistry,
    artifact_id_for_uri,
    artifact_public_metadata,
    artifact_to_public_dict,
    format_artifact_references,
    local_artifact_path,
    safe_artifact_download_filename,
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
    assert registry.list_artifacts(q="Artifact Lab") == []
    assert registry.list_artifacts(q="lab-art") == []
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
    assert registry.list_artifacts(q="observer") == []
    assert registry.list_artifacts(q=text_job.job_id) == []
    text_artifact = registry.list_artifacts(job_kind="code_execution")[0]
    assert registry.list_artifacts(q=text_artifact.artifact_id)[0].title == "notes.txt"
    assert registry.list_artifacts(kind="missing") == []


def test_artifact_registry_clamps_negative_limit(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.providers.ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr("src.artifacts.ARTIFACTS_DIR", tmp_path / "artifacts")
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)
    runner.run_mock_image(ImageGenerationRequest(prompt="first"))
    runner.run_mock_image(ImageGenerationRequest(prompt="second"))

    assert LocalArtifactRegistry(store).list_artifacts(limit=-1) == []


def test_artifact_public_projection_omits_sensitive_fields(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.providers.ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr("src.artifacts.ARTIFACTS_DIR", tmp_path / "artifacts")
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    LocalJobRunner(store).run_mock_image(
        ImageGenerationRequest(
            prompt="Private SMC prompt",
            style="private-style",
            diagram_template="sliding-mode-observer",
            reference_uris=["paper://secret#page=7"],
        ),
        ownership={"owner_id": "secret-owner", "owner_label": "Secret Owner"},
    )
    artifact = LocalArtifactRegistry(store).list_artifacts()[0]

    public_artifact = artifact_to_public_dict(artifact)
    serialized = json.dumps(public_artifact, sort_keys=True)

    assert public_artifact["artifact_id"] == artifact.artifact_id
    assert public_artifact["title_present"] is True
    assert public_artifact["metadata"]["checksum_present"] is True
    assert public_artifact["metadata"]["reference_count"] == 1
    assert public_artifact["metadata"]["style_present"] is True
    assert public_artifact["metadata"]["diagram_template_present"] is True
    assert safe_artifact_download_filename(artifact) == f"artifact-{artifact.artifact_id}.svg"
    assert artifact_public_metadata({"cost_estimate_usd": "secret-cost-token"})["cost_estimate_usd"] == "0"
    for sensitive in (
        artifact.uri,
        artifact.title,
        "Private SMC prompt",
        "private-style",
        "sliding-mode-observer",
        "paper://secret#page=7",
        "secret-owner",
        "Secret Owner",
    ):
        assert sensitive not in serialized


def test_artifact_public_metadata_rejects_unbounded_cost_estimates():
    for invalid_cost in ("NaN", "sNaN", "Infinity", "-Infinity", "1e999999", "1e-999999"):
        metadata = artifact_public_metadata({"cost_estimate_usd": invalid_cost})

        assert metadata["cost_estimate_usd"] == "0"
        assert len(metadata["cost_estimate_usd"]) == 1


def test_artifact_download_filename_sanitizes_unexpected_ids():
    artifact = ArtifactRecord(
        artifact_id='abc"\r\nContent-Disposition: x-secret',
        job_id="job-secret",
        job_kind="code_execution",
        kind="text",
        uri="file:///tmp/secret-owner-result.txt",
        mime_type="text/plain",
        title="secret-owner-result.txt",
    )

    filename = safe_artifact_download_filename(artifact)

    assert filename.endswith(".txt")
    assert "abc" not in filename
    assert "Content-Disposition" not in filename
    assert "secret" not in filename


def test_local_artifact_path_rejects_escaped_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.artifacts.ARTIFACTS_DIR", tmp_path / "artifacts")
    outside = tmp_path / "outside.txt"
    outside.write_text("no", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes"):
        local_artifact_path(outside.resolve().as_uri())


def test_local_artifact_path_returns_canonical_path(tmp_path: Path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    nested = artifact_root / "nested"
    nested.mkdir()
    artifact = artifact_root / "result.txt"
    artifact.write_text("artifact-body", encoding="utf-8")
    alias = nested / ".." / "result.txt"
    monkeypatch.setattr("src.artifacts.ARTIFACTS_DIR", artifact_root)

    assert local_artifact_path(f"file://{alias.as_posix()}") == artifact.resolve()


def test_local_artifact_path_rejects_nonlocal_file_uri(tmp_path: Path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifact = artifact_root / "result.txt"
    artifact.write_text("artifact-body", encoding="utf-8")
    monkeypatch.setattr("src.artifacts.ARTIFACTS_DIR", artifact_root)

    with pytest.raises(ValueError, match="Only local file artifacts"):
        local_artifact_path(f"file://remote-host{artifact.as_posix()}")


def test_local_artifact_path_rejects_symlink_artifact(tmp_path: Path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    target = artifact_root / "target.txt"
    linked = artifact_root / "linked.txt"
    target.write_text("target-body", encoding="utf-8")
    linked.symlink_to(target)
    monkeypatch.setattr("src.artifacts.ARTIFACTS_DIR", artifact_root)

    with pytest.raises(ValueError, match="symlinks"):
        local_artifact_path(linked.as_uri())


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
    assert "style_present=true" in context
    assert "diagram_template_present=true" in context
    assert "reference_count=1" in context
    assert "control-diagram" not in context
    assert "sliding-mode-observer" not in context
    assert "Sliding mode observer block diagram" not in context
    assert "paper://smc#page=3" not in context
    assert "owner=" not in context


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


def test_artifact_registry_ignores_bad_sqlite_payload_and_falls_back_to_jobs(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr("src.providers.ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr("src.artifacts.ARTIFACTS_DIR", tmp_path / "artifacts")
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    job = LocalJobRunner(store).run_mock_image(ImageGenerationRequest(prompt="SMC"))
    registry = LocalArtifactRegistry(store)
    artifact_id = artifact_id_for_uri(job.artifacts[0]["uri"])
    registry.list_artifacts()

    with sqlite3.connect(registry.db_path) as conn:
        conn.execute(
            "UPDATE artifacts SET payload = ? WHERE artifact_id = ?",
            ("{bad-json", artifact_id),
        )

    artifact = registry.get_artifact(artifact_id)

    assert artifact is not None
    assert artifact.job_id == job.job_id


def test_artifact_export_finds_older_stable_id_after_many_jobs(tmp_path: Path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr("src.artifacts.ARTIFACTS_DIR", artifact_root)
    artifact_root.mkdir()
    old_artifact = artifact_root / "old.txt"
    old_artifact.write_text("old", encoding="utf-8")
    new_artifact = artifact_root / "new.txt"
    new_artifact.write_text("new", encoding="utf-8")
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    old_uri = old_artifact.resolve().as_uri()
    store.append(
        JobRecord(
            job_id="job-old-artifact",
            kind="code_execution",
            status="succeeded",
            created_at="2026-06-01T00:00:00+00:00",
            updated_at="2026-06-01T00:00:00+00:00",
            request={},
            artifacts=[
                {
                    "kind": "text",
                    "uri": old_uri,
                    "mime_type": "text/plain",
                    "title": "old.txt",
                    "metadata": {},
                }
            ],
        )
    )
    for index in range(1001):
        timestamp = f"2026-06-02T00:{index // 60:02d}:{index % 60:02d}+00:00"
        store.append(
            JobRecord(
                job_id=f"job-new-artifact-{index}",
                kind="code_execution",
                status="succeeded",
                created_at=timestamp,
                updated_at=timestamp,
                request={},
                artifacts=[
                    {
                        "kind": "text",
                        "uri": new_artifact.resolve().as_uri(),
                        "mime_type": "text/plain",
                        "title": "new.txt",
                        "metadata": {},
                    }
                ],
            )
        )

    registry = LocalArtifactRegistry(store, db_path=artifact_root / "artifacts.sqlite3")
    artifact, path = registry.export_path(artifact_id_for_uri(old_uri))

    assert artifact.job_id == "job-old-artifact"
    assert path == old_artifact.resolve()


def test_artifact_export_uses_full_job_history_beyond_recent_window(tmp_path: Path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr("src.artifacts.ARTIFACTS_DIR", artifact_root)
    artifact_root.mkdir()
    old_artifact = artifact_root / "old-window.txt"
    old_artifact.write_text("old", encoding="utf-8")
    new_artifact = artifact_root / "new-window.txt"
    new_artifact.write_text("new", encoding="utf-8")
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    old_uri = old_artifact.resolve().as_uri()
    store.append(
        JobRecord(
            job_id="job-window-old",
            kind="code_execution",
            status="succeeded",
            created_at="2026-06-01T00:00:00+00:00",
            updated_at="2026-06-01T00:00:00+00:00",
            request={},
            artifacts=[
                {
                    "kind": "text",
                    "uri": old_uri,
                    "mime_type": "text/plain",
                    "title": "old-window.txt",
                    "metadata": {},
                }
            ],
        )
    )
    for index in range(3):
        store.append(
            JobRecord(
                job_id=f"job-window-new-{index}",
                kind="code_execution",
                status="succeeded",
                created_at=f"2026-06-02T00:00:0{index}+00:00",
                updated_at=f"2026-06-02T00:00:0{index}+00:00",
                request={},
                artifacts=[
                    {
                        "kind": "text",
                        "uri": new_artifact.resolve().as_uri(),
                        "mime_type": "text/plain",
                        "title": "new-window.txt",
                        "metadata": {},
                    }
                ],
            )
        )

    original_list_latest = store.list_latest
    monkeypatch.setattr(
        store,
        "list_latest",
        lambda **kwargs: original_list_latest(limit=2),
    )
    registry = LocalArtifactRegistry(store, db_path=artifact_root / "artifacts.sqlite3")

    artifact, path = registry.export_path(artifact_id_for_uri(old_uri))

    assert artifact.job_id == "job-window-old"
    assert path == old_artifact.resolve()


def test_artifact_export_keeps_sibling_artifacts_in_sqlite_mirror(tmp_path: Path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr("src.artifacts.ARTIFACTS_DIR", artifact_root)
    artifact_root.mkdir()
    first_artifact = artifact_root / "first.txt"
    first_artifact.write_text("first", encoding="utf-8")
    second_artifact = artifact_root / "second.txt"
    second_artifact.write_text("second", encoding="utf-8")
    first_uri = first_artifact.resolve().as_uri()
    second_uri = second_artifact.resolve().as_uri()
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    store.append(
        JobRecord(
            job_id="job-two-artifacts",
            kind="code_execution",
            status="succeeded",
            created_at="2026-06-01T00:00:00+00:00",
            updated_at="2026-06-01T00:00:00+00:00",
            request={},
            artifacts=[
                {
                    "kind": "text",
                    "uri": first_uri,
                    "mime_type": "text/plain",
                    "title": "first.txt",
                    "metadata": {},
                },
                {
                    "kind": "text",
                    "uri": second_uri,
                    "mime_type": "text/plain",
                    "title": "second.txt",
                    "metadata": {},
                },
            ],
        )
    )

    registry = LocalArtifactRegistry(store, db_path=artifact_root / "artifacts.sqlite3")
    registry.export_path(artifact_id_for_uri(first_uri))

    assert registry.get_artifact(artifact_id_for_uri(second_uri)).title == "second.txt"
    assert registry.storage_status()["sqlite_rows"] == 2


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
