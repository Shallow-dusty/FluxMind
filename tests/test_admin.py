import hashlib
import os

from src.admin import (
    collect_admin_status,
    collect_retention_preview,
    corpus_status_from_state,
    format_admin_status_report,
    format_corpus_profile_status_report,
    storage_readiness_status,
)
from src.jobs import JobRecord, LocalJobStore
from src.metadata import ChunkMetadataStore, PaperRecord
from src.runtime import append_runtime_event
from langchain_core.documents import Document


def test_format_corpus_profile_status_report_is_no_secret_markdown():
    report = format_corpus_profile_status_report(
        {
            "profile": {
                "profile_id": "smc-core",
                "name": "SMC Core",
                "description": "Core papers",
            },
            "paper_count": 1,
            "available_papers": 1,
            "missing_source_paths": [],
            "active_match": True,
            "rebuild_required": False,
            "index": {
                "status": "fresh",
                "fresh": True,
                "faiss_exists": True,
                "profile_source_paths": 1,
                "chunk_source_paths": 1,
                "missing_chunk_sources": [],
                "extra_chunk_sources": [],
            },
            "papers": [
                {
                    "source_path": "papers/library/paper.pdf",
                    "title": "Paper",
                    "indexed_status": "indexed",
                }
            ],
        }
    )

    assert "# FluxMind Corpus Profile Status" in report
    assert "Profile ID: smc-core" in report
    assert "Rebuild required: False" in report
    assert "papers/library/paper.pdf" in report
    assert "api_key" not in report.lower()


def test_collect_admin_status_summarizes_local_runtime(tmp_path, monkeypatch):
    root = tmp_path
    jobs_dir = root / "jobs"
    artifacts_dir = root / "artifacts"
    metadata_dir = root / "metadata"
    index_dir = root / "faiss_index"
    uploads_dir = root / "papers" / "uploads"
    artifact = artifacts_dir / "plot.png"
    artifact.parent.mkdir(parents=True)
    uploads_dir.mkdir(parents=True)
    artifact.write_bytes(b"plot")
    metadata_dir.mkdir()
    index_dir.mkdir()
    (index_dir / "index.faiss").write_bytes(b"index")
    ChunkMetadataStore(metadata_dir / "chunks.sqlite3").replace_for_sources(
        [
            Document(
                page_content="chunk",
                metadata={
                    "source": "paper.pdf",
                    "source_path": "papers/library/paper.pdf",
                    "page": 1,
                },
            )
        ],
        source_paths=["papers/library/paper.pdf"],
    )

    monkeypatch.setattr("src.jobs.JOBS_FILE", jobs_dir / "jobs.jsonl")
    monkeypatch.setattr("src.admin.PROJECT_ROOT", root)
    monkeypatch.setattr("src.admin.JOBS_DIR", jobs_dir)
    monkeypatch.setattr("src.admin.JOBS_FILE", jobs_dir / "jobs.jsonl")
    monkeypatch.setattr("src.admin.JOBS_DB_FILE", jobs_dir / "jobs.sqlite3")
    monkeypatch.setattr("src.admin.ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr("src.artifacts.ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr("src.admin.PAPERS_UPLOADS_DIR", uploads_dir)
    monkeypatch.setattr("src.admin.METADATA_DIR", metadata_dir)
    monkeypatch.setattr("src.admin.FAISS_INDEX_DIR", index_dir)
    monkeypatch.setattr("src.admin.RUNTIME_EVENTS_FILE", metadata_dir / "runtime_events.jsonl")
    monkeypatch.setattr("src.metadata.CORPUS_METADATA_FILE", metadata_dir / "corpus.json")
    monkeypatch.setattr("src.metadata.CORPUS_METADATA_DB_FILE", metadata_dir / "corpus.sqlite3")
    monkeypatch.setattr("src.metadata.CHUNK_METADATA_DB_FILE", metadata_dir / "chunks.sqlite3")
    monkeypatch.setattr("src.runtime.RUNTIME_EVENTS_FILE", metadata_dir / "runtime_events.jsonl")
    monkeypatch.setattr("src.admin.LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setattr(
        "src.admin.refresh_paper_metadata",
        lambda: [
            PaperRecord(
                paper_id="p1",
                source_path="papers/library/paper.pdf",
                filename="paper.pdf",
                source_kind="library",
                checksum_sha256="a" * 64,
                title="Paper",
                active=True,
                indexed_status="indexed",
                updated_at="2026-05-30T00:00:00+00:00",
            )
        ],
    )

    store = LocalJobStore(jobs_dir / "jobs.jsonl")
    store.append(
        JobRecord(
            job_id="job-ok",
            kind="code_execution",
            status="succeeded",
            created_at="2026-05-30T00:00:00+00:00",
            updated_at="2026-05-30T00:00:01+00:00",
            request={},
            artifacts=[
                {
                    "kind": "plot",
                    "uri": artifact.resolve().as_uri(),
                    "mime_type": "image/png",
                    "title": "plot.png",
                    "metadata": {
                        "provider": "local",
                        "checksum_sha256": hashlib.sha256(b"plot").hexdigest(),
                        "byte_count": "4",
                    },
                }
            ],
        )
    )
    store.append(
        JobRecord(
            job_id="job-fail",
            kind="image_generation",
            status="failed",
            created_at="2026-05-30T00:00:00+00:00",
            updated_at="2026-05-30T00:00:02+00:00",
            request={},
            error={"code": "provider_error", "message": "failed"},
        )
    )
    append_runtime_event(
        kind="provider_failure",
        code="provider_timeout",
        message="The model provider timed out. Please retry the request.",
        request_id="req-provider",
        metadata={"endpoint": "/query", "status_code": 504},
    )
    append_runtime_event(
        kind="query_usage",
        code="estimated_usage",
        message="Estimated no-key query usage. This is not provider billing.",
        request_id="req-usage",
        metadata={
            "endpoint": "/query",
            "answer_mode": "explanation",
            "question_chars": 11,
            "answer_chars": 22,
            "estimated_prompt_tokens": 3,
            "estimated_answer_tokens": 6,
            "estimated_total_tokens": 9,
            "usage_source": "provider",
            "provider_prompt_tokens": 4,
            "provider_completion_tokens": 8,
            "provider_total_tokens": 12,
            "estimated_cost_usd": "0",
        },
    )

    status = collect_admin_status().to_dict()

    assert status["jobs"]["total"] == 2
    assert status["jobs"]["by_status"] == {"failed": 1, "succeeded": 1}
    assert status["jobs"]["by_kind"] == {"code_execution": 1, "image_generation": 1}
    assert status["jobs"]["latest_failed"][0]["job_id"] == "job-fail"
    assert status["jobs"]["storage"]["jsonl_exists"] is True
    assert status["jobs"]["storage"]["sqlite_exists"] is True
    assert status["jobs"]["queue_health"] == {
        "queued": 0,
        "due": 0,
        "scheduled": 0,
        "expired": 0,
        "running": 0,
        "leased_queued": 0,
        "lease_expired_queued": 0,
        "running_leased": 0,
        "oldest_queued_at": None,
    }
    assert status["artifacts"]["total"] == 1
    assert status["artifacts"]["bytes"] >= 4
    assert status["artifacts"]["storage"]["sqlite_exists"] is True
    assert status["artifacts"]["integrity"]["checked"] == 1
    assert status["artifacts"]["integrity"]["ok"] == 1
    assert status["artifacts"]["integrity"]["checksum_mismatch"] == 0
    assert status["provider_failures"]["total_recent"] == 1
    assert status["provider_failures"]["by_code"] == {"provider_timeout": 1}
    assert status["provider_failures"]["latest"][0]["request_id"] == "req-provider"
    assert status["query_usage"]["total_recent"] == 1
    assert status["query_usage"]["by_endpoint"] == {"/query": 1}
    assert status["query_usage"]["by_answer_mode"] == {"explanation": 1}
    assert status["query_usage"]["estimated_total_tokens"] == 9
    assert status["query_usage"]["provider_prompt_tokens"] == 4
    assert status["query_usage"]["provider_completion_tokens"] == 8
    assert status["query_usage"]["provider_total_tokens"] == 12
    assert status["query_usage"]["provider_usage_events"] == 1
    assert status["query_usage"]["estimated_cost_usd"] == "0"
    assert status["corpus"]["status"] == "indexed"
    assert status["corpus"]["papers"] == 1
    assert status["corpus"]["active"] == 1
    assert status["corpus"]["available"] == 0
    assert status["corpus"]["indexed"] == 1
    assert status["corpus"]["failed"] == 0
    assert status["corpus"]["storage"]["sqlite_exists"] is True
    assert status["corpus"]["chunks"]["sqlite_exists"] is True
    assert status["corpus"]["index"] == {
        "status": "fresh",
        "fresh": True,
        "faiss_exists": True,
        "active_source_paths": 1,
        "chunk_source_paths": 1,
        "missing_chunk_sources": [],
        "extra_chunk_sources": [],
    }
    assert status["corpus"]["index_jobs"] == {"by_status": {}, "latest": []}
    assert status["config"]["external_providers_enabled"] is False
    assert status["config"]["reranker_model_configured"] is False
    assert status["config"]["reranker_model_available"] is False
    assert status["config"]["code_execution_backend"] == "local"
    assert status["config"]["docker_execution"]["configured"] is False
    assert status["config"]["docker_execution"]["available"] is False
    assert status["config"]["docker_execution"]["reason"] == "not_configured"
    assert status["config"]["storage_readiness"]["metadata"]["backend"] == "local"
    assert status["config"]["storage_readiness"]["metadata"]["available"] is True
    assert status["config"]["storage_readiness"]["metadata"]["database_url_configured"] is False
    assert status["config"]["storage_readiness"]["object_storage"]["backend"] == "local"
    assert status["config"]["storage_readiness"]["object_storage"]["available"] is True
    assert status["config"]["storage_readiness"]["object_storage"]["bucket_configured"] is False
    assert status["config"]["storage_readiness"]["external_storage_configured"] is False
    assert all("path" in item for item in status["runtime_dirs"])

    report = format_admin_status_report(status)
    assert "# FluxMind Admin Status" in report
    assert "No-secret local runtime snapshot" in report
    assert "- Total: 2" in report
    assert "job-fail" in report
    assert "provider_timeout" in report
    assert "Estimated total tokens: 9" in report
    assert "Provider total tokens: 12" in report
    assert "Reranker model configured: false" in report
    assert "Code execution backend: local" in report
    assert "Docker execution available: false" in report
    assert "Metadata storage backend: local" in report
    assert "Object storage backend: local" in report
    assert "req-usage" in report
    assert "postgres://" not in report
    assert "api_key" not in report.lower()


def test_storage_readiness_status_reports_external_config_without_secrets(tmp_path, monkeypatch):
    root = tmp_path
    metadata_dir = root / "metadata"
    jobs_dir = root / "jobs"
    artifacts_dir = root / "artifacts"
    uploads_dir = root / "papers" / "uploads"
    for path in (metadata_dir, jobs_dir, artifacts_dir, uploads_dir):
        path.mkdir(parents=True)

    monkeypatch.setattr("src.admin.PROJECT_ROOT", root)
    monkeypatch.setattr("src.admin.METADATA_DIR", metadata_dir)
    monkeypatch.setattr("src.admin.JOBS_DIR", jobs_dir)
    monkeypatch.setattr("src.admin.ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr("src.admin.PAPERS_UPLOADS_DIR", uploads_dir)

    status = storage_readiness_status(
        metadata_backend="postgres",
        object_backend="s3-compatible",
        database_url="postgres://user:secret@example/db",
        object_bucket="private-bucket",
        object_endpoint="https://objects.example.test",
        object_region="auto",
    )

    assert status["metadata"] == {
        "backend": "postgres",
        "configured": True,
        "available": True,
        "reason": "configured_not_connected",
        "database_url_configured": True,
    }
    assert status["object_storage"] == {
        "backend": "s3-compatible",
        "configured": True,
        "available": True,
        "reason": "configured_not_connected",
        "bucket_configured": True,
        "endpoint_configured": True,
        "region_configured": True,
    }
    assert status["external_storage_configured"] is True
    assert status["external_storage_available"] is True
    assert "secret" not in str(status)
    assert "private-bucket" not in str(status)
    assert "objects.example.test" not in str(status)


def test_storage_readiness_status_rejects_incomplete_external_config(tmp_path, monkeypatch):
    root = tmp_path
    (root / "metadata").mkdir()
    (root / "jobs").mkdir()
    (root / "artifacts").mkdir()
    (root / "papers" / "uploads").mkdir(parents=True)
    monkeypatch.setattr("src.admin.PROJECT_ROOT", root)
    monkeypatch.setattr("src.admin.METADATA_DIR", root / "metadata")
    monkeypatch.setattr("src.admin.JOBS_DIR", root / "jobs")
    monkeypatch.setattr("src.admin.ARTIFACTS_DIR", root / "artifacts")
    monkeypatch.setattr("src.admin.PAPERS_UPLOADS_DIR", root / "papers" / "uploads")

    status = storage_readiness_status(
        metadata_backend="postgres",
        object_backend="s3",
        database_url="",
        object_bucket="bucket",
        object_endpoint="",
        object_region="",
    )

    assert status["metadata"]["configured"] is False
    assert status["metadata"]["available"] is False
    assert status["metadata"]["reason"] == "database_url_missing"
    assert status["object_storage"]["configured"] is False
    assert status["object_storage"]["available"] is False
    assert status["object_storage"]["reason"] == "bucket_or_endpoint_missing"
    assert status["external_storage_available"] is False


def test_collect_retention_preview_lists_old_uploads_and_artifacts(tmp_path, monkeypatch):
    root = tmp_path
    uploads_dir = root / "papers" / "uploads"
    artifacts_dir = root / "artifacts"
    uploads_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    old_upload = uploads_dir / "old.pdf"
    new_upload = uploads_dir / "new.pdf"
    old_artifact = artifacts_dir / "old.txt"
    artifact_db = artifacts_dir / "artifacts.sqlite3"
    old_upload.write_bytes(b"old upload")
    new_upload.write_bytes(b"new upload")
    old_artifact.write_bytes(b"old artifact")
    artifact_db.write_bytes(b"db")
    now = 1_800_000_000.0
    old = now - 10 * 24 * 60 * 60
    recent = now - 1 * 24 * 60 * 60
    os.utime(old_upload, (old, old))
    os.utime(old_artifact, (old, old))
    os.utime(artifact_db, (old, old))
    os.utime(new_upload, (recent, recent))

    monkeypatch.setattr("src.admin.PROJECT_ROOT", root)
    monkeypatch.setattr("src.admin.PAPERS_UPLOADS_DIR", uploads_dir)
    monkeypatch.setattr("src.admin.ARTIFACTS_DIR", artifacts_dir)

    preview = collect_retention_preview(upload_days=7, artifact_days=7, now_ts=now)

    assert preview["mode"] == "preview"
    assert preview["delete_enabled"] is False
    assert preview["uploads"]["total_candidates"] == 1
    assert preview["uploads"]["candidates"][0]["path"] == "papers/uploads/old.pdf"
    assert preview["artifacts"]["total_candidates"] == 1
    assert preview["artifacts"]["candidates"][0]["path"] == "artifacts/old.txt"


def test_corpus_index_status_reports_stale_chunks(tmp_path, monkeypatch):
    from src.admin import corpus_index_status

    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    (index_dir / "index.faiss").write_bytes(b"index")
    monkeypatch.setattr("src.admin.FAISS_INDEX_DIR", index_dir)

    class FakeChunkStore:
        def source_paths(self):
            return ["papers/library/old.pdf"]

    papers = [
        PaperRecord(
            paper_id="p1",
            source_path="papers/library/current.pdf",
            filename="current.pdf",
            source_kind="library",
            checksum_sha256="a" * 64,
            title="Current",
            active=True,
            indexed_status="indexed",
            updated_at="2026-06-01T00:00:00+00:00",
        )
    ]

    status = corpus_index_status(papers, FakeChunkStore())

    assert status["status"] == "stale"
    assert status["fresh"] is False
    assert status["missing_chunk_sources"] == ["papers/library/current.pdf"]
    assert status["extra_chunk_sources"] == ["papers/library/old.pdf"]


def test_corpus_status_reports_index_rebuild_lifecycle(tmp_path, monkeypatch):
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    monkeypatch.setattr("src.admin.FAISS_INDEX_DIR", index_dir)

    class FakeChunkStore:
        def source_paths(self):
            return []

        def storage_status(self):
            return {"sqlite_exists": False, "sqlite_rows": 0}

    class FakeMetadataStore:
        def storage_status(self):
            return {"json_exists": True, "sqlite_exists": True}

    papers = [
        PaperRecord(
            paper_id="p1",
            source_path="papers/library/current.pdf",
            filename="current.pdf",
            source_kind="library",
            checksum_sha256="a" * 64,
            title="Current",
            active=True,
            indexed_status="active",
            updated_at="2026-06-01T00:00:00+00:00",
        )
    ]
    jobs = [
        JobRecord(
            job_id="index-running",
            kind="index_rebuild",
            status="running",
            created_at="2026-06-01T00:00:00+00:00",
            updated_at="2026-06-01T00:00:01+00:00",
            request={},
        )
    ]

    status = corpus_status_from_state(papers, FakeChunkStore(), jobs, FakeMetadataStore())

    assert status["status"] == "parsing"
    assert status["index_jobs"]["by_status"] == {"running": 1}
    assert status["index_jobs"]["latest"][0]["job_id"] == "index-running"


def test_corpus_status_reports_stale_when_no_rebuild_job(tmp_path, monkeypatch):
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    monkeypatch.setattr("src.admin.FAISS_INDEX_DIR", index_dir)

    class FakeChunkStore:
        def source_paths(self):
            return []

        def storage_status(self):
            return {"sqlite_exists": False, "sqlite_rows": 0}

    class FakeMetadataStore:
        def storage_status(self):
            return {"json_exists": True, "sqlite_exists": True}

    papers = [
        PaperRecord(
            paper_id="p1",
            source_path="papers/library/current.pdf",
            filename="current.pdf",
            source_kind="library",
            checksum_sha256="a" * 64,
            title="Current",
            active=True,
            indexed_status="active",
            updated_at="2026-06-01T00:00:00+00:00",
        )
    ]

    status = corpus_status_from_state(papers, FakeChunkStore(), [], FakeMetadataStore())

    assert status["status"] == "stale"
    assert status["index"]["status"] == "missing"
