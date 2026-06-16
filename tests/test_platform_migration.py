import json
import sqlite3

from src.jobs import LocalJobStore
from src.platform_migration import (
    collect_platform_migration_preflight,
    format_platform_migration_preflight_markdown,
)


def _init_empty_job_store(root):
    jobs_dir = root / "jobs"
    jobs_dir.mkdir(parents=True)
    store = LocalJobStore(jobs_dir / "jobs.jsonl")
    store._ensure_sqlite()


def test_platform_migration_preflight_passes_local_evidence_without_activation(tmp_path):
    _init_empty_job_store(tmp_path)

    status = collect_platform_migration_preflight(
        project_root=tmp_path,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    assert status["mode"] == "production_migration_preflight"
    assert status["preflight_ok"] is True
    assert status["activation_ready"] is False
    assert status["content_exported"] is False
    assert status["secrets_exported"] is False
    assert status["connectivity_checked"] is False
    assert status["blockers"]["local_preflight"] == []
    assert "production_metadata_database_not_configured" in status["blockers"]["activation"]
    assert "production_object_storage_not_configured" in status["blockers"]["activation"]
    assert "distributed_job_store_not_configured" in status["blockers"]["activation"]
    assert status["summary"]["local_storage_schema_ok"] is True
    assert status["summary"]["runtime_restore_dry_run_ok"] is True
    assert status["summary"]["local_durable_job_store_ready"] is True


def test_platform_migration_preflight_reports_activation_ready_without_secrets(tmp_path):
    _init_empty_job_store(tmp_path)

    status = collect_platform_migration_preflight(
        project_root=tmp_path,
        generated_at="2026-06-16T00:00:00+00:00",
        metadata_backend="postgres",
        database_url="postgres://user:hunter2@example.test/db",
        object_backend="s3-compatible",
        object_bucket="hidden-bucket",
        object_endpoint="https://objects.example.test",
        object_region="auto",
        distributed_job_store_backend="redis",
        distributed_job_store_url="redis://:hunter2@example.test/0",
        distributed_job_queue_name="hidden-queue",
    )

    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    assert status["preflight_ok"] is True
    assert status["activation_ready"] is True
    assert status["blockers"]["activation"] == []
    assert status["summary"]["external_storage_configured"] is True
    assert status["summary"]["external_job_store_configured"] is True
    assert "hunter2" not in payload
    assert "hidden-bucket" not in payload
    assert "objects.example.test" not in payload
    assert "hidden-queue" not in payload
    assert "redis://:" not in payload


def test_platform_migration_preflight_blocks_missing_local_job_store(tmp_path):
    status = collect_platform_migration_preflight(
        project_root=tmp_path,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    assert status["preflight_ok"] is False
    assert status["activation_ready"] is False
    assert "local_durable_job_store_missing" in status["blockers"]["local_preflight"]
    assert "queue_health_contract_missing" in status["blockers"]["local_preflight"]
    assert "worker_lease_contract_missing" in status["blockers"]["local_preflight"]
    assert "jobs_sqlite_missing" in status["blockers"]["local_preflight"]


def test_platform_migration_preflight_reports_unsupported_external_backends(tmp_path):
    _init_empty_job_store(tmp_path)

    status = collect_platform_migration_preflight(
        project_root=tmp_path,
        generated_at="2026-06-16T00:00:00+00:00",
        metadata_backend="mysql",
        database_url="mysql://hidden.example.test/db",
        object_backend="azure-blob",
        object_bucket="hidden-container",
        object_endpoint="https://objects.example.test",
        distributed_job_store_backend="celery",
        distributed_job_store_url="amqp://hidden.example.test",
        distributed_job_queue_name="hidden-queue",
    )

    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    assert status["preflight_ok"] is True
    assert status["activation_ready"] is False
    assert status["storage_readiness"]["metadata"]["reason"] == "unsupported_metadata_backend"
    assert status["storage_readiness"]["object_storage"]["reason"] == "unsupported_object_storage_backend"
    assert status["distributed_job_store"]["reason"] == "unsupported_job_store_backend"
    assert "hidden.example.test" not in payload
    assert "hidden-container" not in payload
    assert "hidden-queue" not in payload


def test_platform_migration_preflight_blocks_invalid_jobs_sqlite(tmp_path):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    (jobs_dir / "jobs.sqlite3").write_text("not sqlite", encoding="utf-8")

    status = collect_platform_migration_preflight(
        project_root=tmp_path,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    assert status["preflight_ok"] is False
    assert "jobs_sqlite_unreadable" in status["blockers"]["local_preflight"]
    assert "local_storage_schema_drift" in status["blockers"]["local_preflight"]


def test_platform_migration_preflight_blocks_jobs_sqlite_without_jobs_table(tmp_path):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    with sqlite3.connect(jobs_dir / "jobs.sqlite3") as conn:
        conn.execute("CREATE TABLE placeholder (id TEXT)")

    status = collect_platform_migration_preflight(
        project_root=tmp_path,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    assert status["preflight_ok"] is False
    assert "jobs_table_missing" in status["blockers"]["local_preflight"]
    assert "queue_health_contract_missing" in status["blockers"]["local_preflight"]


def test_format_platform_migration_preflight_markdown_is_no_secret(tmp_path):
    _init_empty_job_store(tmp_path)
    status = collect_platform_migration_preflight(
        project_root=tmp_path,
        generated_at="2026-06-16T00:00:00+00:00",
        metadata_backend="postgres",
        database_url="postgres://user:hunter2@example.test/db",
        object_backend="s3",
        object_bucket="hidden-bucket",
        object_endpoint="https://objects.example.test",
        distributed_job_store_backend="redis",
        distributed_job_store_url="redis://:hunter2@example.test/0",
        distributed_job_queue_name="hidden-queue",
    )

    markdown = format_platform_migration_preflight_markdown(status)

    assert "# FluxMind Production Migration Preflight" in markdown
    assert "Preflight OK: true" in markdown
    assert "Activation ready: true" in markdown
    assert "hunter2" not in markdown
    assert "hidden-bucket" not in markdown
    assert "objects.example.test" not in markdown
    assert "hidden-queue" not in markdown
