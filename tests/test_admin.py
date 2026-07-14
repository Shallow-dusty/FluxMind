import hashlib
import json
import os

import src.admin as admin
from src.admin import (
    apply_retention_delete,
    collect_admin_status,
    collect_retention_preview,
    corpus_status_from_state,
    format_admin_metrics,
    format_admin_status_report,
    format_corpus_profile_status_report,
    distributed_job_store_status,
    platform_readiness_status,
    summarize_code_execution_alerts,
    summarize_job_alerts,
    summarize_provider_failure_alerts,
    summarize_query_usage_alerts,
    summarize_retrieval_trace_alerts,
    storage_inventory_status,
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


def test_summarize_code_execution_alerts_uses_thresholds():
    alerts = summarize_code_execution_alerts(
        total_recent=10,
        failed_recent=6,
        failure_rate=0.6,
        max_duration_ms=40000,
        policy_violations=2,
        output_truncations=1,
        artifact_collection_truncations=1,
        min_events=5,
        failure_rate_threshold=0.5,
        duration_ms_threshold=30000,
    )

    codes = {alert["code"] for alert in alerts}
    assert "code_execution_failure_rate_high" in codes
    assert "code_execution_duration_high" in codes
    assert "code_execution_policy_violations_recent" in codes
    assert "code_execution_output_truncated_recent" in codes
    assert "code_execution_artifacts_truncated_recent" in codes
    assert all("source" not in str(alert).casefold() for alert in alerts)


def test_summarize_query_usage_alerts_uses_duration_thresholds():
    alerts = summarize_query_usage_alerts(
        total_recent=5,
        avg_duration_ms=16000,
        max_duration_ms=24000,
        min_events=5,
        duration_ms_threshold=15000,
    )

    assert [alert["code"] for alert in alerts] == ["query_duration_average_high"]
    assert alerts[0]["severity"] == "warning"
    assert alerts[0]["metadata"]["total_recent"] == 5
    assert "question" not in str(alerts).casefold()
    assert "answer" not in str(alerts).casefold()


def test_summarize_query_usage_alerts_reports_single_slow_query():
    alerts = summarize_query_usage_alerts(
        total_recent=1,
        avg_duration_ms=1000,
        max_duration_ms=20000,
        min_events=5,
        duration_ms_threshold=15000,
    )

    assert [alert["code"] for alert in alerts] == ["query_duration_high"]
    assert alerts[0]["severity"] == "info"


def test_summarize_retrieval_trace_alerts_uses_quality_thresholds():
    alerts = summarize_retrieval_trace_alerts(
        total_recent=8,
        empty_recent=3,
        empty_rate=0.375,
        source_page_incomplete_recent=4,
        source_page_incomplete_rate=0.5,
        citation_checked_recent=5,
        citation_failed_recent=2,
        citation_failure_rate=0.4,
        min_events=5,
        empty_rate_threshold=0.25,
        source_page_incomplete_rate_threshold=0.25,
        citation_failure_rate_threshold=0.25,
    )

    codes = {alert["code"] for alert in alerts}
    assert codes == {
        "retrieval_empty_rate_high",
        "retrieval_source_page_incomplete_rate_high",
        "retrieval_citation_failure_rate_high",
    }
    assert all("question" not in str(alert).casefold() for alert in alerts)
    assert all("answer" not in str(alert).casefold() for alert in alerts)
    assert all("source_path" not in str(alert).casefold() for alert in alerts)


def test_summarize_provider_failure_alerts_uses_rate_and_repeat_thresholds():
    alerts = summarize_provider_failure_alerts(
        total_recent=3,
        total_query_outcomes=6,
        failure_rate=0.5,
        by_code={"provider_timeout": 3},
        min_events=3,
        failure_rate_threshold=0.25,
    )

    codes = {alert["code"] for alert in alerts}
    assert codes == {"provider_failure_rate_high", "provider_failure_code_repeated"}
    assert all("api_key" not in str(alert).casefold() for alert in alerts)
    assert all("answer" not in str(alert).casefold() for alert in alerts)


def test_summarize_job_alerts_uses_queue_and_lease_thresholds():
    alerts = summarize_job_alerts(
        failed_recent=4,
        dead_lettered_recent=1,
        queue_health={"expired": 2, "lease_expired_queued": 1},
        worker_leases={"expired_leases": 1},
        failed_min_events=3,
        expired_min_events=1,
    )

    codes = {alert["code"] for alert in alerts}
    assert codes == {
        "job_failures_recent",
        "job_dead_letters_recent",
        "job_queue_deadlines_expired",
        "job_worker_leases_expired",
    }
    assert all("request" not in str(alert).casefold() for alert in alerts)


def test_platform_readiness_status_reports_local_blockers_without_secrets():
    status = platform_readiness_status(
        storage_readiness={
            "metadata": {
                "backend": "local",
                "configured": False,
                "available": True,
            },
            "object_storage": {
                "backend": "local",
                "configured": False,
                "available": True,
            },
        },
        storage_schemas={"ok": True, "problem_count": 0},
        storage={"mode": "local", "groups": [{"name": "metadata"}]},
        jobs={
            "storage": {"sqlite_exists": True},
            "queue_health": {
                "queued": 0,
                "due": 0,
                "scheduled": 0,
                "expired": 0,
                "running": 0,
                "leased_queued": 0,
                "lease_expired_queued": 0,
                "running_leased": 0,
                "oldest_queued_at": None,
            },
            "worker_leases": {
                "total_leased_jobs": 0,
                "worker_ids": [],
                "active_worker_ids": [],
                "expired_worker_ids": [],
                "active_leases": 0,
                "expired_leases": 0,
            },
        },
        distributed_job_store={
            "backend": "local",
            "configured": False,
            "available": True,
        },
    )

    assert status["overall_ready"] is False
    assert status["activation_enabled"] is False
    assert status["storage_migration"]["blockers"] == [
        "production_metadata_database_not_configured",
        "production_object_storage_not_configured",
    ]
    assert status["distributed_workers"]["blockers"] == [
        "distributed_job_store_not_configured",
    ]
    assert status["distributed_workers"]["checks"]["local_worker_bridge_ready"] is True
    assert status["distributed_workers"]["checks"]["distributed_job_store_backend"] == "local"
    assert status["distributed_workers"]["checks"]["distributed_job_store_external_ready"] is False
    assert "secret" not in str(status).casefold()
    assert "source_path" not in str(status).casefold()


def test_platform_readiness_status_accepts_configured_external_targets():
    status = platform_readiness_status(
        storage_readiness={
            "metadata": {
                "backend": "postgres",
                "configured": True,
                "available": True,
            },
            "object_storage": {
                "backend": "s3-compatible",
                "configured": True,
                "available": True,
            },
        },
        storage_schemas={"ok": True, "problem_count": 0},
        storage={"mode": "local", "groups": [{"name": "metadata"}, {"name": "jobs"}]},
        jobs={
            "storage": {"sqlite_exists": True},
            "queue_health": {
                "queued": 0,
                "due": 0,
                "scheduled": 0,
                "expired": 0,
                "running": 0,
                "leased_queued": 0,
                "lease_expired_queued": 0,
                "running_leased": 0,
                "oldest_queued_at": None,
            },
            "worker_leases": {
                "total_leased_jobs": 0,
                "worker_ids": [],
                "active_worker_ids": [],
                "expired_worker_ids": [],
                "active_leases": 0,
                "expired_leases": 0,
            },
        },
        distributed_job_store={
            "backend": "redis",
            "configured": True,
            "available": True,
        },
    )

    assert status["overall_ready"] is True
    assert status["storage_migration"]["ready"] is True
    assert status["distributed_workers"]["ready"] is True
    assert status["storage_migration"]["blockers"] == []
    assert status["distributed_workers"]["blockers"] == []
    assert status["distributed_workers"]["checks"]["distributed_job_store_external_ready"] is True


def test_distributed_job_store_status_reports_external_config_without_secrets():
    status = distributed_job_store_status(
        backend="redis",
        store_url="redis://:secret@example.test:6379/0",
        queue_name="fluxmind-prod",
    )

    assert status == {
        "backend": "redis",
        "configured": True,
        "available": True,
        "reason": "configured_not_connected",
        "store_url_configured": True,
        "queue_name_configured": True,
        "external_job_store_configured": True,
        "external_job_store_available": True,
    }
    assert "secret" not in str(status)
    assert "example.test" not in str(status)
    assert "fluxmind-prod" not in str(status)


def test_distributed_job_store_status_rejects_incomplete_external_config():
    status = distributed_job_store_status(
        backend="postgres",
        store_url="",
        queue_name="fluxmind-jobs",
    )

    assert status["configured"] is False
    assert status["available"] is False
    assert status["reason"] == "store_url_or_queue_name_missing"
    assert status["store_url_configured"] is False
    assert status["queue_name_configured"] is True
    assert status["external_job_store_configured"] is False


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
    monkeypatch.setattr("src.admin.ACTIVE_PAPERS_FILE", index_dir / "active_papers.json")
    monkeypatch.setattr("src.admin.CORPUS_METADATA_FILE", metadata_dir / "corpus.json")
    monkeypatch.setattr("src.config.EXTERNAL_PROVIDERS_ENABLED", False)
    monkeypatch.setattr("src.config.IMAGE_PROVIDER_API_CONFIGURED", False)
    monkeypatch.setattr("src.config.OPENAI_IMAGE_API_KEY", "")
    monkeypatch.setattr("src.config.IMAGE_PROVIDER_BACKEND", "local-mock")
    monkeypatch.setattr("src.admin.CORPUS_PROFILES_FILE", metadata_dir / "corpus_profiles.json")
    monkeypatch.setattr("src.admin.CORPUS_METADATA_DB_FILE", metadata_dir / "corpus.sqlite3")
    monkeypatch.setattr("src.admin.CHUNK_METADATA_DB_FILE", metadata_dir / "chunks.sqlite3")
    monkeypatch.setattr("src.admin.API_KEY_REGISTRY_FILE", metadata_dir / "api_keys.sqlite3")
    monkeypatch.setattr("src.admin.RUNTIME_EVENTS_FILE", metadata_dir / "runtime_events.jsonl")
    monkeypatch.setattr("src.metadata.CORPUS_METADATA_FILE", metadata_dir / "corpus.json")
    monkeypatch.setattr("src.metadata.CORPUS_METADATA_DB_FILE", metadata_dir / "corpus.sqlite3")
    monkeypatch.setattr("src.metadata.CHUNK_METADATA_DB_FILE", metadata_dir / "chunks.sqlite3")
    monkeypatch.setattr("src.runtime.RUNTIME_EVENTS_FILE", metadata_dir / "runtime_events.jsonl")
    monkeypatch.setattr("src.admin.LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setattr("src.admin.QUERY_ALERT_MIN_EVENTS", 1)
    monkeypatch.setattr("src.admin.QUERY_ALERT_DURATION_MS", 40)
    monkeypatch.setattr("src.admin.RETRIEVAL_TRACE_ALERT_MIN_EVENTS", 1)
    monkeypatch.setattr("src.admin.RETRIEVAL_TRACE_ALERT_EMPTY_RATE", 0.5)
    monkeypatch.setattr("src.admin.RETRIEVAL_TRACE_ALERT_SOURCE_PAGE_INCOMPLETE_RATE", 0.5)
    monkeypatch.setattr("src.admin.RETRIEVAL_TRACE_ALERT_CITATION_FAILURE_RATE", 0.5)
    monkeypatch.setattr(
        "src.admin.storage_schema_status",
        lambda: {
            "schema_version": 1,
            "mode": "local_storage_schema_inventory",
            "ok": True,
            "store_count": 2,
            "problem_count": 0,
            "stores": [
                {
                    "name": "corpus_metadata_sqlite",
                    "kind": "sqlite",
                    "required": False,
                    "exists": True,
                    "ok": True,
                    "errors": [],
                },
                {
                    "name": "runtime_events_jsonl",
                    "kind": "jsonl",
                    "required": False,
                    "exists": True,
                    "ok": True,
                    "errors": [],
                },
            ],
        },
    )
    monkeypatch.setattr("src.admin.PROVIDER_FAILURE_ALERT_MIN_EVENTS", 1)
    monkeypatch.setattr("src.admin.PROVIDER_FAILURE_ALERT_RATE", 0.5)
    monkeypatch.setattr("src.admin.JOB_ALERT_FAILED_MIN_EVENTS", 1)
    monkeypatch.setattr("src.admin.JOB_ALERT_EXPIRED_MIN_EVENTS", 1)
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
            owner_id="lab-admin",
            owner_label="Admin Lab",
            ownership_source="request",
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
            "duration_ms": 42,
            "estimated_cost_usd": "0",
        },
    )
    append_runtime_event(
        kind="retrieval_trace",
        code="retrieval_source_page_incomplete",
        message="Metadata-only retrieval trace.",
        request_id=None,
        metadata={
            "endpoint": "/query/inspect",
            "answer_mode": "implementation",
            "context_count": 2,
            "missing_source_page_count": 1,
            "source_page_complete": False,
            "retrieval_ok": False,
            "provider_called": True,
            "citation_ok": False,
            "duration_ms": 21,
        },
    )
    append_runtime_event(
        kind="code_execution",
        code="execution_policy_violation",
        message="Code execution job failed.",
        request_id="req-code",
        metadata={
            "job_id": "job-code",
            "status": "failed",
            "language": "python",
            "backend": "local",
            "duration_ms": 37,
            "artifact_count": 0,
            "owner_id": "lab-admin",
            "policy_violation": "true",
            "execution_policy": "local-safe-v1",
            "execution_policy_violations": "1",
            "output_truncated": "true",
            "artifact_collection_truncated": "true",
            "artifact_exported_bytes": "12",
        },
    )
    append_runtime_event(
        kind="api_access",
        code="auth_invalid",
        message="Metadata-only API access audit event.",
        request_id="req-access",
        metadata={
            "method": "GET",
            "route_present": True,
            "route_fingerprint": "route-admin",
            "status_code": 401,
            "duration_ms": 5,
            "token_status": "invalid",
            "credential_type": "bearer",
            "credential_present": True,
            "auth_configured": True,
        },
    )
    append_runtime_event(
        kind="api_access",
        code="auth_not_configured",
        message="Metadata-only API access audit event.",
        request_id="req-rate-limit",
        metadata={
            "method": "GET",
            "route_present": True,
            "route_fingerprint": "route-health",
            "status_code": 429,
            "duration_ms": 1,
            "token_status": "not_configured",
            "credential_type": "none",
            "credential_present": False,
            "auth_configured": False,
            "rate_limit_enabled": True,
            "rate_limited": True,
            "rate_limit": 2,
            "rate_limit_remaining": 0,
            "rate_limit_window_s": 60,
            "rate_limit_reset_after_s": 59,
        },
    )
    append_runtime_event(
        kind="admin_check",
        code="openapi_contract_ok",
        message="Metadata-only admin readiness check event.",
        request_id="req-admin-check-ok",
        metadata={
            "check": "openapi_contract",
            "ok": True,
            "status_code": 200,
            "route_count": 63,
            "operation_count": 69,
            "blocker_count": 0,
            "operation_fingerprint": "private-fingerprint",
            "snapshot_path": "/private/hunter2-openapi.json",
        },
    )
    append_runtime_event(
        kind="admin_check",
        code="activation_suite_blocked",
        message="Metadata-only admin readiness check event.",
        request_id="req-admin-check-blocked",
        metadata={
            "check": "activation_suite",
            "ok": False,
            "status_code": 200,
            "failed_check_count": 4,
            "blocker_count": 4,
            "raw_report": {"secret_path": "/private/hunter2-live-report.json"},
        },
    )
    append_runtime_event(
        kind="admin_check",
        code="secret-/private/hunter2",
        message="Metadata-only admin readiness check event.",
        request_id="req-admin-check-secret",
        metadata={
            "check": "/private/hunter2",
            "ok": False,
            "status_code": 200,
            "blocker_count": -3,
            "raw_report": {"secret_path": "/private/hunter2-legacy-report.json"},
        },
    )
    append_runtime_event(
        kind="upload_scan",
        code="upload_scan_blocked",
        message="Metadata-only PDF upload scan event.",
        request_id="req-upload-scan",
        metadata={
            "scan_enabled": True,
            "status": "blocked",
            "reason_codes": ["active_content_javascript"],
            "byte_count": 2048,
            "page_count": 0,
            "encrypted": False,
            "active_content_markers": ["javascript"],
            "active_content_marker_count": 1,
            "max_pages": 500,
            "reject_encrypted": True,
            "block_active_content": True,
        },
    )

    status = collect_admin_status().to_dict()

    assert status["jobs"]["total"] == 2
    assert status["jobs"]["by_status"] == {"failed": 1, "succeeded": 1}
    assert status["jobs"]["by_kind"] == {"code_execution": 1, "image_generation": 1}
    assert status["jobs"]["owner_count"] == 2
    assert status["jobs"]["by_ownership_source"] == {"default": 1, "request": 1}
    assert status["jobs"]["dead_lettered"] == 0
    assert status["jobs"]["alert_thresholds"] == {
        "failed_min_events": 1,
        "expired_min_events": 1,
    }
    assert {
        alert["code"]
        for alert in status["jobs"]["alerts"]
    } == {"job_failures_recent"}
    assert status["jobs"]["latest_failed"][0]["job_id"] == "job-fail"
    assert status["jobs"]["latest_failed"][0]["ownership_source"] == "default"
    assert "owner_id" not in status["jobs"]["latest_failed"][0]
    assert "owner_label" not in status["jobs"]["latest_failed"][0]
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
    assert status["jobs"]["worker_leases"] == {
        "total_leased_jobs": 0,
        "worker_ids": [],
        "by_worker": {},
        "active_worker_ids": [],
        "expired_worker_ids": [],
        "active_leases": 0,
        "expired_leases": 0,
        "latest": [],
    }
    assert status["artifacts"]["total"] == 1
    assert status["artifacts"]["owner_count"] == 1
    assert status["artifacts"]["by_ownership_source"] == {"request": 1}
    assert status["artifacts"]["bytes"] >= 4
    assert status["artifacts"]["storage"]["sqlite_exists"] is True
    assert status["artifacts"]["integrity"]["checked"] == 1
    assert status["artifacts"]["integrity"]["ok"] == 1
    assert status["artifacts"]["integrity"]["checksum_mismatch"] == 0
    assert status["storage"]["mode"] == "local"
    assert status["storage"]["content_scanned"] is False
    assert status["storage"]["total_files"] >= 4
    assert status["storage"]["total_bytes"] >= 4
    storage_groups = {group["name"]: group for group in status["storage"]["groups"]}
    assert set(storage_groups) == {"metadata", "jobs", "artifacts", "uploads", "faiss_index"}
    assert storage_groups["metadata"]["files"] >= 2
    assert storage_groups["jobs"]["known_files"][0]["name"] == "jobs_jsonl"
    assert storage_groups["jobs"]["known_files"][0]["exists"] is True
    assert storage_groups["faiss_index"]["known_files"][0]["name"] == "index_faiss"
    assert storage_groups["faiss_index"]["known_files"][0]["exists"] is True
    assert status["provider_failures"]["total_recent"] == 1
    assert status["provider_failures"]["by_code"] == {"provider_timeout": 1}
    assert status["provider_failures"]["failure_rate"] == 0.5
    assert status["provider_failures"]["alert_thresholds"] == {
        "min_events": 1,
        "failure_rate": 0.5,
    }
    assert {
        alert["code"]
        for alert in status["provider_failures"]["alerts"]
    } == {
        "provider_failure_rate_high",
        "provider_failure_code_repeated",
    }
    assert status["provider_failures"]["latest"][0]["request_id_present"] is True
    assert "request_id" not in status["provider_failures"]["latest"][0]
    assert status["query_usage"]["total_recent"] == 1
    assert status["query_usage"]["by_endpoint"] == {"/query": 1}
    assert status["query_usage"]["by_answer_mode"] == {"explanation": 1}
    assert status["query_usage"]["estimated_total_tokens"] == 9
    assert status["query_usage"]["provider_prompt_tokens"] == 4
    assert status["query_usage"]["provider_completion_tokens"] == 8
    assert status["query_usage"]["provider_total_tokens"] == 12
    assert status["query_usage"]["provider_usage_events"] == 1
    assert status["query_usage"]["duration_ms"] == {"avg": 42, "max": 42}
    assert status["query_usage"]["alert_thresholds"] == {
        "min_events": 1,
        "duration_ms": 40,
    }
    assert {
        alert["code"]
        for alert in status["query_usage"]["alerts"]
    } == {"query_duration_average_high"}
    assert status["query_usage"]["estimated_cost_usd"] == "0"
    assert status["query_usage"]["cost_source"] == "not_configured"
    assert status["query_usage"]["pricing"]["configured"] is False
    assert status["query_usage"]["pricing"]["external_billing_enabled"] is False
    assert status["query_usage"]["latest"][0]["request_id_present"] is True
    assert "request_id" not in status["query_usage"]["latest"][0]
    assert status["retrieval_traces"]["total_recent"] == 1
    assert status["retrieval_traces"]["by_code"] == {"retrieval_source_page_incomplete": 1}
    assert status["retrieval_traces"]["by_endpoint"] == {"/query/inspect": 1}
    assert status["retrieval_traces"]["by_answer_mode"] == {"implementation": 1}
    assert status["retrieval_traces"]["empty_recent"] == 0
    assert status["retrieval_traces"]["empty_rate"] == 0.0
    assert status["retrieval_traces"]["source_page_incomplete_recent"] == 1
    assert status["retrieval_traces"]["source_page_incomplete_rate"] == 1.0
    assert status["retrieval_traces"]["citation_checked_recent"] == 1
    assert status["retrieval_traces"]["citation_failed_recent"] == 1
    assert status["retrieval_traces"]["citation_failure_rate"] == 1.0
    assert status["retrieval_traces"]["provider_called_recent"] == 1
    assert status["retrieval_traces"]["alert_thresholds"] == {
        "min_events": 1,
        "empty_rate": 0.5,
        "source_page_incomplete_rate": 0.5,
        "citation_failure_rate": 0.5,
    }
    assert {
        alert["code"]
        for alert in status["retrieval_traces"]["alerts"]
    } == {
        "retrieval_source_page_incomplete_rate_high",
        "retrieval_citation_failure_rate_high",
    }
    assert status["retrieval_traces"]["context_count"] == {"avg": 2, "max": 2}
    assert status["retrieval_traces"]["duration_ms"] == {"avg": 21, "max": 21}
    assert status["retrieval_traces"]["latest"][0]["metadata"]["context_count"] == 2
    assert status["retrieval_traces"]["latest"][0]["request_id_present"] is False
    assert "request_id" not in status["retrieval_traces"]["latest"][0]
    assert status["code_execution"]["total_recent"] == 1
    assert status["code_execution"]["by_code"] == {"execution_policy_violation": 1}
    assert status["code_execution"]["by_status"] == {"failed": 1}
    assert status["code_execution"]["by_backend"] == {"local": 1}
    assert status["code_execution"]["failed_recent"] == 1
    assert status["code_execution"]["failure_rate"] == 1.0
    assert status["code_execution"]["policy_violations"] == 1
    assert status["code_execution"]["output_truncations"] == 1
    assert status["code_execution"]["artifact_collection_truncations"] == 1
    assert status["code_execution"]["artifact_exported_bytes"] == 12
    assert status["code_execution"]["alert_thresholds"] == {
        "min_events": 5,
        "failure_rate": 0.5,
        "duration_ms": 30000,
    }
    assert {
        alert["code"]
        for alert in status["code_execution"]["alerts"]
    } == {
        "code_execution_policy_violations_recent",
        "code_execution_output_truncated_recent",
        "code_execution_artifacts_truncated_recent",
    }
    assert status["code_execution"]["duration_ms"] == {"avg": 37, "max": 37}
    assert status["code_execution"]["latest"][0]["request_id_present"] is True
    assert status["code_execution"]["latest"][0]["metadata_redacted_fields"] == 1
    assert "request_id" not in status["code_execution"]["latest"][0]
    assert "owner_id" not in status["code_execution"]["latest"][0]["metadata"]
    assert status["api_access"]["audit_enabled"] is True
    assert status["api_access"]["total_recent"] == 2
    assert status["api_access"]["by_code"] == {"auth_invalid": 1, "auth_not_configured": 1}
    assert status["api_access"]["by_token_status"] == {"invalid": 1, "not_configured": 1}
    assert status["api_access"]["by_status_code"] == {"401": 1, "429": 1}
    assert status["api_access"]["by_method"] == {"GET": 2}
    assert status["api_access"]["invalid_recent"] == 1
    assert status["api_access"]["missing_recent"] == 0
    assert status["api_access"]["valid_recent"] == 0
    assert status["api_access"]["rate_limited_recent"] == 1
    assert status["api_access"]["rate_limit"] == {
        "enabled": False,
        "max_requests": 300,
        "window_s": 60,
    }
    assert status["api_access"]["latest"][0]["request_id_present"] is True
    assert "request_id" not in status["api_access"]["latest"][0]
    assert "path" not in status["api_access"]["latest"][0]["metadata"]
    assert "secret" not in str(status["api_access"]).casefold()
    assert status["admin_checks"]["audit_enabled"] is True
    assert status["admin_checks"]["total_recent"] == 3
    assert status["admin_checks"]["by_code"] == {
        "activation_suite_blocked": 1,
        "invalid": 1,
        "openapi_contract_ok": 1,
    }
    assert status["admin_checks"]["by_check"] == {
        "activation_suite": 1,
        "invalid": 1,
        "openapi_contract": 1,
    }
    assert status["admin_checks"]["by_status"] == {"blocked": 2, "ok": 1}
    assert status["admin_checks"]["ok_recent"] == 1
    assert status["admin_checks"]["blocked_recent"] == 2
    assert status["admin_checks"]["blocker_count_total"] == 4
    assert status["admin_checks"]["latest"][0]["request_id_present"] is False
    assert status["admin_checks"]["latest"][0]["code"] == "invalid"
    assert status["admin_checks"]["latest"][0]["metadata"]["check"] == "invalid"
    assert "request_id" not in status["admin_checks"]["latest"][0]
    assert "operation_fingerprint" not in str(status["admin_checks"])
    assert "raw_report" not in str(status["admin_checks"])
    assert "hunter2" not in str(status["admin_checks"])
    assert "/private" not in str(status["admin_checks"])
    assert "secret" not in str(status["admin_checks"]).casefold()
    assert status["upload_scans"]["scan_enabled"] is True
    assert status["upload_scans"]["total_recent"] == 1
    assert status["upload_scans"]["by_code"] == {"upload_scan_blocked": 1}
    assert status["upload_scans"]["by_status"] == {"blocked": 1}
    assert status["upload_scans"]["by_reason"] == {"active_content_javascript": 1}
    assert status["upload_scans"]["allowed_recent"] == 0
    assert status["upload_scans"]["blocked_recent"] == 1
    assert status["upload_scans"]["active_content_recent"] == 1
    assert status["upload_scans"]["parse_failed_recent"] == 0
    assert status["upload_scans"]["config"] == {
        "enabled": True,
        "max_pages": 500,
        "reject_encrypted": True,
        "block_active_content": True,
    }
    assert status["upload_scans"]["latest"][0]["request_id_present"] is True
    assert "request_id" not in status["upload_scans"]["latest"][0]
    assert "secret" not in str(status["upload_scans"]).casefold()
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
    assert status["config"]["code_execution_policy"] == "local-safe-v1"
    assert status["config"]["code_execution_max_stdout_bytes"] == 65536
    assert status["config"]["code_execution_max_stderr_bytes"] == 65536
    assert status["config"]["code_execution_max_artifacts"] == 16
    assert status["config"]["code_execution_max_artifact_bytes"] == 2 * 1024 * 1024
    assert status["config"]["code_execution_max_artifact_total_bytes"] == 8 * 1024 * 1024
    assert status["config"]["code_execution_max_artifact_candidates"] == 256
    assert status["config"]["code_execution_alert_min_events"] == 5
    assert status["config"]["code_execution_alert_failure_rate"] == 0.5
    assert status["config"]["code_execution_alert_duration_ms"] == 30000
    assert status["config"]["query_alert_min_events"] == 1
    assert status["config"]["query_alert_duration_ms"] == 40
    assert status["config"]["retrieval_trace_alert_min_events"] == 1
    assert status["config"]["retrieval_trace_alert_empty_rate"] == 0.5
    assert status["config"]["retrieval_trace_alert_source_page_incomplete_rate"] == 0.5
    assert status["config"]["retrieval_trace_alert_citation_failure_rate"] == 0.5
    assert status["config"]["provider_failure_alert_min_events"] == 1
    assert status["config"]["provider_failure_alert_rate"] == 0.5
    assert status["config"]["job_alert_failed_min_events"] == 1
    assert status["config"]["job_alert_expired_min_events"] == 1
    assert status["config"]["api_access_audit_enabled"] is True
    assert status["config"]["api_rate_limit_enabled"] is False
    assert status["config"]["api_rate_limit_max_requests"] == 300
    assert status["config"]["api_rate_limit_window_s"] == 60
    assert status["config"]["upload_scan_enabled"] is True
    assert status["config"]["upload_scan_max_pages"] == 500
    assert status["config"]["upload_scan_reject_encrypted"] is True
    assert status["config"]["upload_scan_block_active_content"] is True
    assert "pathlib" in status["config"]["code_execution_allowed_imports"]
    assert status["config"]["docker_execution"]["configured"] is False
    assert status["config"]["docker_execution"]["available"] is False
    assert status["config"]["docker_execution"]["reason"] == "not_configured"
    product_readiness = status["config"]["product_readiness"]
    assert product_readiness["summary"]["product_quota_guard_enabled"] is False
    assert product_readiness["summary"]["product_rbac_guard_enabled"] is False
    assert "product_rbac_guard_disabled" in product_readiness["advisories"]
    provider_readiness = status["config"]["provider_readiness"]
    assert provider_readiness["local_foundation_ready"] is True
    assert provider_readiness["activation_ready"] is False
    assert provider_readiness["external_providers_enabled"] is False
    assert provider_readiness["summary"]["external_image_provider_configured"] is False
    assert provider_readiness["summary"]["hosted_execution_provider_configured"] is False
    assert provider_readiness["summary"]["matlab_backend_configured"] is False
    assert "external_providers_disabled" in provider_readiness["blockers"]["activation"]
    assert "external_image_provider_not_configured" in provider_readiness["blockers"]["activation"]
    assert "hosted_execution_provider_not_configured" in provider_readiness["blockers"]["activation"]
    assert "matlab_backend_not_configured" in provider_readiness["blockers"]["activation"]
    assert "provider_quota_guard_not_enabled" in provider_readiness["blockers"]["activation"]
    assert status["config"]["storage_readiness"]["metadata"]["backend"] == "local"
    assert status["config"]["storage_readiness"]["metadata"]["available"] is True
    assert status["config"]["storage_readiness"]["metadata"]["database_url_configured"] is False
    assert status["config"]["storage_readiness"]["object_storage"]["backend"] == "local"
    assert status["config"]["storage_readiness"]["object_storage"]["available"] is True
    assert status["config"]["storage_readiness"]["object_storage"]["bucket_configured"] is False
    assert status["config"]["storage_readiness"]["external_storage_configured"] is False
    assert status["config"]["distributed_job_store"]["backend"] == "local"
    assert status["config"]["distributed_job_store"]["available"] is True
    assert status["config"]["distributed_job_store"]["external_job_store_configured"] is False
    assert status["storage_schemas"]["ok"] is True
    assert status["storage_schemas"]["store_count"] == 2
    assert status["storage_schemas"]["problem_count"] == 0
    assert {
        store["name"]
        for store in status["storage_schemas"]["stores"]
    } == {"corpus_metadata_sqlite", "runtime_events_jsonl"}
    assert status["platform_readiness"]["overall_ready"] is False
    assert status["platform_readiness"]["activation_enabled"] is False
    assert status["platform_readiness"]["storage_migration"]["ready"] is False
    assert status["platform_readiness"]["distributed_workers"]["ready"] is False
    assert "production_metadata_database_not_configured" in status["platform_readiness"]["storage_migration"]["blockers"]
    assert "production_object_storage_not_configured" in status["platform_readiness"]["storage_migration"]["blockers"]
    assert status["platform_readiness"]["distributed_workers"]["checks"]["local_worker_bridge_ready"] is True
    assert status["platform_readiness"]["distributed_workers"]["checks"]["distributed_job_store_backend"] == "local"
    assert status["platform_readiness"]["distributed_workers"]["checks"]["distributed_job_store_external_ready"] is False
    assert "distributed_job_store_not_configured" in status["platform_readiness"]["distributed_workers"]["blockers"]
    assert all("path" in item for item in status["runtime_dirs"])
    serialized_status = str(status).casefold()
    for sensitive in (
        "lab-admin",
        "admin lab",
        "owner_id",
        "owner_label",
        "by_owner_id",
        "req-provider",
        "req-usage",
        "req-code",
        "req-access",
        "req-rate-limit",
        "req-upload-scan",
        "/admin/status",
        "/health",
    ):
        assert sensitive not in serialized_status

    report = format_admin_status_report(status)
    assert "# FluxMind Admin Status" in report
    assert "No-secret local runtime snapshot" in report
    assert "- Total: 2" in report
    assert "Owner count: 2" in report
    assert "By ownership source: default=1, request=1" in report
    assert "Job alerts:" in report
    assert "job_failures_recent" in report
    assert "Owner count: 1" in report
    assert "job-fail" in report
    assert "provider_timeout" in report
    assert "Estimated total tokens: 9" in report
    assert "Provider total tokens: 12" in report
    assert "Provider failure alerts:" in report
    assert "provider_failure_rate_high" in report
    assert "Avg duration ms: 42" in report
    assert "Query alerts:" in report
    assert "query_duration_average_high" in report
    assert "## Code Execution Events" in report
    assert "Policy violations: 1" in report
    assert "Output truncations: 1" in report
    assert "Artifact collection truncations: 1" in report
    assert "Alert count: 3" in report
    assert "Code execution alerts:" in report
    assert "code_execution_policy_violations_recent" in report
    assert "Latest code execution events:" in report
    assert "request_id_present=true" in report
    assert "## API Access Audit" in report
    assert "By auth status: invalid=1, not_configured=1" in report
    assert "Rate limited: 1" in report
    assert "Latest API access audit events:" in report
    assert "API access audit enabled: true" in report
    assert "API rate limit enabled: false" in report
    assert "Product quota guard enabled: false" in report
    assert "Product RBAC guard enabled: false" in report
    assert "API rate limit max requests: 300" in report
    assert "## Upload Scans" in report
    assert "By reason: active_content_javascript=1" in report
    assert "Latest upload scan events:" in report
    assert "Upload scan enabled: true" in report
    assert "Upload scan max pages: 500" in report
    assert "Cost source: not_configured" in report
    assert "Pricing configured: false" in report
    assert "Reranker model configured: false" in report
    assert "Worker leases:" in report
    assert "Code execution backend: local" in report
    assert "Code execution policy: local-safe-v1" in report
    assert "Code execution max stdout bytes: 65536" in report
    assert "Code execution max stderr bytes: 65536" in report
    assert "Code execution alert min events: 5" in report
    assert "Query alert min events: 1" in report
    assert "Query alert duration ms: 40" in report
    assert "Provider failure alert min events: 1" in report
    assert "Provider failure alert rate: 0.5" in report
    assert "Job alert failed min events: 1" in report
    assert "Job alert expired min events: 1" in report
    assert "Code execution allowed imports:" in report
    assert "Docker execution available: false" in report
    assert "Provider local foundation ready: true" in report
    assert "Provider activation ready: false" in report
    assert "Provider external image configured: false" in report
    assert "Provider hosted execution configured: false" in report
    assert "Provider MATLAB backend configured: false" in report
    assert "Provider activation blockers:" in report
    assert "external_image_provider_not_configured" in report
    assert "matlab_backend_not_configured" in report
    assert "Metadata storage backend: local" in report
    assert "Object storage backend: local" in report
    assert "## Storage Inventory" in report
    assert "## Storage Schemas" in report
    assert "Problem count: 0" in report
    assert "corpus_metadata_sqlite" in report
    assert "## Platform Readiness" in report
    assert "Storage migration ready: false" in report
    assert "production_metadata_database_not_configured" in report
    assert "Distributed workers ready: false" in report
    assert "distributed_job_store_not_configured" in report
    assert "## Retrieval Traces" in report
    assert "Source/page incomplete: 1" in report
    assert "Source/page incomplete rate: 1.0" in report
    assert "Citation failure rate: 1.0" in report
    assert "Retrieval trace alerts:" in report
    assert "retrieval_source_page_incomplete_rate_high" in report
    assert "Retrieval trace alert min events: 1" in report
    assert "Retrieval trace alert source/page incomplete rate: 0.5" in report
    assert "## Admin Check Events" in report
    assert "By check: activation_suite=1, invalid=1, openapi_contract=1" in report
    assert "By code: activation_suite_blocked=1, invalid=1, openapi_contract_ok=1" in report
    assert "Blocked checks: 2" in report
    assert "Latest admin check events:" in report
    assert "Content scanned: false" in report
    assert "faiss_index" in report
    assert "postgres://" not in report
    assert "api_key" not in report.lower()
    for sensitive in (
        "lab-admin",
        "admin lab",
        "owner_id",
        "owner_label",
        "by_owner_id",
        "req-provider",
        "req-usage",
        "req-code",
        "req-access",
        "req-rate-limit",
        "req-admin-check-ok",
        "req-admin-check-blocked",
        "req-admin-check-secret",
        "req-upload-scan",
        "path=/admin/status",
        "path=/health",
        "private-fingerprint",
        "/private",
        "secret-/private",
        "hunter2",
    ):
        assert sensitive not in report.casefold()

    metrics = format_admin_metrics(status)
    assert "# HELP fluxmind_jobs_total" in metrics
    assert "fluxmind_jobs_total 2" in metrics
    assert 'fluxmind_jobs_by_status{status="failed"} 1' in metrics
    assert 'fluxmind_provider_failures_by_code{code="provider_timeout"} 1' in metrics
    assert 'fluxmind_query_usage_duration_ms{stat="max"} 42' in metrics
    assert "fluxmind_storage_schema_ok 1" in metrics
    assert "fluxmind_storage_schema_problem_total 0" in metrics
    assert "fluxmind_distributed_job_store_external_configured 0" in metrics
    assert 'fluxmind_distributed_job_store_available{backend="local"} 1' in metrics
    assert 'fluxmind_storage_schema_store_ok{store="corpus_metadata_sqlite"} 1' in metrics
    assert "fluxmind_platform_readiness_overall_ready 0" in metrics
    assert "fluxmind_platform_storage_migration_ready 0" in metrics
    assert "fluxmind_platform_distributed_workers_ready 0" in metrics
    assert 'fluxmind_platform_readiness_blocker{area="storage",code="production_metadata_database_not_configured"} 1' in metrics
    assert 'fluxmind_platform_readiness_blocker{area="workers",code="distributed_job_store_not_configured"} 1' in metrics
    assert "fluxmind_retrieval_traces_recent_total 1" in metrics
    assert 'fluxmind_retrieval_traces_by_code{code="retrieval_source_page_incomplete"} 1' in metrics
    assert "fluxmind_retrieval_alerts_total 2" in metrics
    assert "fluxmind_retrieval_source_page_incomplete_rate 1" in metrics
    assert "fluxmind_retrieval_citation_failure_rate 1" in metrics
    assert 'fluxmind_retrieval_context_count{stat="max"} 2' in metrics
    assert 'fluxmind_code_execution_by_code{code="execution_policy_violation"} 1' in metrics
    assert 'fluxmind_api_access_by_token_status{token_status="invalid"} 1' in metrics
    assert 'fluxmind_api_access_by_status_code{status_code="429"} 1' in metrics
    assert "fluxmind_admin_checks_recent_total 3" in metrics
    assert 'fluxmind_admin_checks_by_check{check="openapi_contract"} 1' in metrics
    assert 'fluxmind_admin_checks_by_check{check="activation_suite"} 1' in metrics
    assert 'fluxmind_admin_checks_by_check{check="invalid"} 1' in metrics
    assert 'fluxmind_admin_checks_by_code{code="activation_suite_blocked"} 1' in metrics
    assert 'fluxmind_admin_checks_by_code{code="invalid"} 1' in metrics
    assert "fluxmind_admin_checks_ok_recent 1" in metrics
    assert "fluxmind_admin_checks_blocked_recent 2" in metrics
    assert "fluxmind_admin_checks_blocker_count_total 4" in metrics
    assert 'fluxmind_upload_scans_by_reason{reason="active_content_javascript"} 1' in metrics
    assert "fluxmind_retention_delete_enabled 0" in metrics
    assert "fluxmind_provider_local_foundation_ready 1" in metrics
    assert "fluxmind_provider_activation_ready 0" in metrics
    assert "fluxmind_provider_activation_blockers_total 5" in metrics
    assert "fluxmind_provider_external_image_configured 0" in metrics
    assert "fluxmind_provider_hosted_execution_configured 0" in metrics
    assert "fluxmind_provider_matlab_backend_configured 0" in metrics
    assert "fluxmind_provider_quota_guard_enabled 0" in metrics
    assert "lab-admin" not in metrics
    assert "req-code" not in metrics
    assert "question" not in metrics.casefold()
    assert "answer" not in metrics.casefold()
    assert "api_key" not in metrics.lower()


def test_collect_admin_status_estimates_cost_with_configured_rates(tmp_path, monkeypatch):
    root = tmp_path
    metadata_dir = root / "metadata"
    metadata_dir.mkdir()
    monkeypatch.setattr("src.admin.PROJECT_ROOT", root)
    monkeypatch.setattr("src.admin.JOBS_DIR", root / "jobs")
    monkeypatch.setattr("src.admin.JOBS_FILE", root / "jobs" / "jobs.jsonl")
    monkeypatch.setattr("src.admin.JOBS_DB_FILE", root / "jobs" / "jobs.sqlite3")
    monkeypatch.setattr("src.jobs.JOBS_FILE", root / "jobs" / "jobs.jsonl")
    monkeypatch.setattr("src.admin.ARTIFACTS_DIR", root / "artifacts")
    monkeypatch.setattr("src.artifacts.ARTIFACTS_DIR", root / "artifacts")
    monkeypatch.setattr("src.admin.PAPERS_UPLOADS_DIR", root / "papers" / "uploads")
    monkeypatch.setattr("src.admin.METADATA_DIR", metadata_dir)
    monkeypatch.setattr("src.admin.FAISS_INDEX_DIR", root / "faiss_index")
    monkeypatch.setattr("src.admin.RUNTIME_EVENTS_FILE", metadata_dir / "runtime_events.jsonl")
    monkeypatch.setattr("src.runtime.RUNTIME_EVENTS_FILE", metadata_dir / "runtime_events.jsonl")
    monkeypatch.setattr("src.metadata.CORPUS_METADATA_FILE", metadata_dir / "corpus.json")
    monkeypatch.setattr("src.metadata.CORPUS_METADATA_DB_FILE", metadata_dir / "corpus.sqlite3")
    monkeypatch.setattr("src.metadata.CHUNK_METADATA_DB_FILE", metadata_dir / "chunks.sqlite3")
    monkeypatch.setattr("src.admin.refresh_paper_metadata", lambda: [])
    monkeypatch.setattr("src.admin.QUERY_COST_PROVIDER", "mimo-deepseek")
    monkeypatch.setattr("src.admin.QUERY_COST_PROMPT_USD_PER_1M", "2")
    monkeypatch.setattr("src.admin.QUERY_COST_COMPLETION_USD_PER_1M", "5")

    append_runtime_event(
        kind="query_usage",
        code="estimated_usage",
        message="Estimated no-key query usage. This is not provider billing.",
        request_id="req-usage",
        metadata={
            "endpoint": "/query",
            "answer_mode": "explanation",
            "estimated_prompt_tokens": 3,
            "estimated_answer_tokens": 6,
            "usage_source": "provider",
            "provider_prompt_tokens": 4,
            "provider_completion_tokens": 8,
            "provider_total_tokens": 12,
        },
    )

    status = collect_admin_status().to_dict()

    assert status["query_usage"]["estimated_cost_usd"] == "0.000048"
    assert status["query_usage"]["cost_source"] == "provider_tokens"
    assert status["query_usage"]["cost_prompt_tokens"] == 4
    assert status["query_usage"]["cost_completion_tokens"] == 8
    assert status["query_usage"]["pricing"]["configured"] is True
    assert status["query_usage"]["pricing"]["provider"] == "mimo-deepseek"


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


def test_storage_inventory_status_reports_local_counts_without_content(tmp_path, monkeypatch):
    root = tmp_path
    metadata_dir = root / "metadata"
    jobs_dir = root / "jobs"
    artifacts_dir = root / "artifacts"
    uploads_dir = root / "papers" / "uploads"
    index_dir = root / "faiss_index"
    for path in (metadata_dir, jobs_dir, artifacts_dir, uploads_dir, index_dir):
        path.mkdir(parents=True)
    (metadata_dir / "corpus.json").write_text('{"secret":"not returned"}', encoding="utf-8")
    (jobs_dir / "jobs.jsonl").write_text("job\n", encoding="utf-8")
    (artifacts_dir / "plot.svg").write_text("<svg />", encoding="utf-8")
    (uploads_dir / "paper.pdf").write_bytes(b"pdf")
    (index_dir / "index.faiss").write_bytes(b"index")

    monkeypatch.setattr("src.admin.PROJECT_ROOT", root)
    monkeypatch.setattr("src.admin.METADATA_DIR", metadata_dir)
    monkeypatch.setattr("src.admin.JOBS_DIR", jobs_dir)
    monkeypatch.setattr("src.admin.ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr("src.admin.PAPERS_UPLOADS_DIR", uploads_dir)
    monkeypatch.setattr("src.admin.FAISS_INDEX_DIR", index_dir)
    monkeypatch.setattr("src.admin.CORPUS_METADATA_FILE", metadata_dir / "corpus.json")
    monkeypatch.setattr("src.admin.CORPUS_PROFILES_FILE", metadata_dir / "corpus_profiles.json")
    monkeypatch.setattr("src.admin.CORPUS_METADATA_DB_FILE", metadata_dir / "corpus.sqlite3")
    monkeypatch.setattr("src.admin.CHUNK_METADATA_DB_FILE", metadata_dir / "chunks.sqlite3")
    monkeypatch.setattr("src.admin.RUNTIME_EVENTS_FILE", metadata_dir / "runtime_events.jsonl")
    monkeypatch.setattr("src.admin.PRODUCT_REGISTRY_FILE", metadata_dir / "product_registry.sqlite3")
    monkeypatch.setattr("src.admin.JOBS_FILE", jobs_dir / "jobs.jsonl")
    monkeypatch.setattr("src.admin.JOBS_DB_FILE", jobs_dir / "jobs.sqlite3")
    monkeypatch.setattr("src.admin.ACTIVE_PAPERS_FILE", index_dir / "active_papers.json")

    status = storage_inventory_status()

    assert status["mode"] == "local"
    assert status["content_scanned"] is False
    assert status["total_files"] == 5
    assert "not returned" not in str(status)
    groups = {group["name"]: group for group in status["groups"]}
    assert groups["metadata"]["known_files"][0] == {
        "name": "corpus_json",
        "path": "metadata/corpus.json",
        "exists": True,
        "is_file": True,
        "bytes": 25,
    }
    assert {
        item["name"]
        for item in groups["metadata"]["known_files"]
    } >= {
        "corpus_json",
        "api_key_registry_sqlite",
        "product_registry_sqlite",
        "share_link_registry_sqlite",
        "runtime_events_jsonl",
    }
    assert groups["uploads"]["files"] == 1
    assert groups["faiss_index"]["bytes"] == 5


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


def test_collect_retention_preview_skips_symlinks_without_exposing_targets(tmp_path, monkeypatch):
    root = tmp_path / "project"
    outside = tmp_path / "outside-secret.pdf"
    uploads_dir = root / "papers" / "uploads"
    artifacts_dir = root / "artifacts"
    uploads_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    outside.write_bytes(b"outside")
    (uploads_dir / "linked.pdf").symlink_to(outside)
    now = 1_800_000_000.0
    old = now - 10 * 24 * 60 * 60
    os.utime(outside, (old, old))

    monkeypatch.setattr("src.admin.PROJECT_ROOT", root)
    monkeypatch.setattr("src.admin.PAPERS_UPLOADS_DIR", uploads_dir)
    monkeypatch.setattr("src.admin.ARTIFACTS_DIR", artifacts_dir)

    preview = collect_retention_preview(upload_days=7, artifact_days=7, now_ts=now)
    payload = json.dumps(preview, ensure_ascii=False, sort_keys=True)

    assert preview["uploads"]["total_candidates"] == 0
    assert preview["uploads"]["candidates"] == []
    assert "outside-secret" not in payload
    assert "linked.pdf" not in payload


def test_apply_retention_delete_disabled_keeps_candidates(tmp_path, monkeypatch):
    root = tmp_path
    uploads_dir = root / "papers" / "uploads"
    artifacts_dir = root / "artifacts"
    metadata_dir = root / "metadata"
    uploads_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    metadata_dir.mkdir()
    old_upload = uploads_dir / "old.pdf"
    old_upload.write_bytes(b"old upload")
    now = 1_800_000_000.0
    old = now - 10 * 24 * 60 * 60
    os.utime(old_upload, (old, old))

    monkeypatch.setattr("src.admin.PROJECT_ROOT", root)
    monkeypatch.setattr("src.admin.PAPERS_UPLOADS_DIR", uploads_dir)
    monkeypatch.setattr("src.admin.ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr("src.admin.RETENTION_DELETE_ENABLED", False)
    monkeypatch.setattr("src.runtime.RUNTIME_EVENTS_FILE", metadata_dir / "runtime_events.jsonl")

    result = apply_retention_delete(upload_days=7, artifact_days=7, now_ts=now)

    assert result["mode"] == "delete_disabled"
    assert result["delete_enabled"] is False
    assert result["deleted_files"] == 0
    assert old_upload.exists()
    event = json.loads((metadata_dir / "runtime_events.jsonl").read_text(encoding="utf-8"))
    assert event["kind"] == "retention_delete"
    assert event["code"] == "retention_delete_disabled"
    assert "old.pdf" not in json.dumps(event, ensure_ascii=False)


def test_apply_retention_delete_removes_only_age_matched_candidates(tmp_path, monkeypatch):
    root = tmp_path
    uploads_dir = root / "papers" / "uploads"
    artifacts_dir = root / "artifacts"
    metadata_dir = root / "metadata"
    uploads_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    metadata_dir.mkdir()
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
    for path in (old_upload, old_artifact, artifact_db):
        os.utime(path, (old, old))
    os.utime(new_upload, (recent, recent))

    monkeypatch.setattr("src.admin.PROJECT_ROOT", root)
    monkeypatch.setattr("src.admin.PAPERS_UPLOADS_DIR", uploads_dir)
    monkeypatch.setattr("src.admin.ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr("src.admin.RETENTION_DELETE_ENABLED", True)
    monkeypatch.setattr("src.runtime.RUNTIME_EVENTS_FILE", metadata_dir / "runtime_events.jsonl")

    result = apply_retention_delete(upload_days=7, artifact_days=7, now_ts=now)

    assert result["mode"] == "delete"
    assert result["delete_enabled"] is True
    assert result["deleted_files"] == 2
    assert result["deleted_bytes"] == len(b"old upload") + len(b"old artifact")
    assert result["failed_files"] == 0
    assert not old_upload.exists()
    assert not old_artifact.exists()
    assert new_upload.exists()
    assert artifact_db.exists()
    event = json.loads((metadata_dir / "runtime_events.jsonl").read_text(encoding="utf-8"))
    assert event["code"] == "retention_delete_applied"
    assert event["metadata"]["deleted_files"] == 2
    assert event["metadata"]["upload_deleted_files"] == 1
    assert event["metadata"]["artifact_deleted_files"] == 1
    assert "old.pdf" not in json.dumps(event, ensure_ascii=False)


def test_delete_retention_candidates_rejects_symlink_candidate_without_following(tmp_path, monkeypatch):
    root = tmp_path / "project"
    uploads_dir = root / "papers" / "uploads"
    uploads_dir.mkdir(parents=True)
    protected = uploads_dir / "protected.pdf"
    linked = uploads_dir / "old.pdf"
    protected.write_bytes(b"protected")
    linked.symlink_to(protected)

    def fake_candidates(*_args, **_kwargs):
        return {
            "enabled": True,
            "retention_days": 7,
            "total_candidates": 1,
            "bytes": 9,
            "candidates": [
                {
                    "path": "papers/uploads/old.pdf",
                    "bytes": 9,
                    "age_days": 10,
                }
            ],
        }

    monkeypatch.setattr("src.admin.PROJECT_ROOT", root)
    monkeypatch.setattr(admin, "_retention_candidates", fake_candidates)

    result = admin._delete_retention_candidates(
        uploads_dir,
        retention_days=7,
        limit=100,
        now_ts=1_800_000_000.0,
    )

    assert result["deleted_files"] == 0
    assert result["failed_files"] == 1
    assert protected.read_bytes() == b"protected"
    assert linked.is_symlink()


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
