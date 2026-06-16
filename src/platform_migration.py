"""No-secret production migration preflight.

This module composes the local storage schema, runtime manifest, job-store
contract, and existing admin readiness helpers. It reports migration evidence
and blocker codes only; it never exports runtime contents, job payloads, source
paths, external URLs, buckets, queue names, credentials, or secrets.
"""

from __future__ import annotations

import sqlite3
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import config
from src.admin import (
    distributed_job_store_status,
    platform_readiness_status,
)
from src.storage_manifest import (
    RuntimeFileSpec,
    RuntimeGroupSpec,
    collect_runtime_backup_manifest,
    collect_runtime_restore_check,
)
from src.storage_schema import storage_schema_status_for_root


PLATFORM_MIGRATION_PREFLIGHT_SCHEMA_VERSION = 1
EXTERNAL_STORAGE_BLOCKERS = {
    "production_metadata_database_not_configured",
    "production_object_storage_not_configured",
}
EXTERNAL_WORKER_BLOCKERS = {"distributed_job_store_not_configured"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative_path(path: Path, *, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _runtime_path_readiness(name: str, path: Path, *, project_root: Path) -> dict[str, Any]:
    exists = path.exists()
    writable_target = path if exists else path.parent
    writable = os.access(writable_target, os.W_OK) if writable_target.exists() else False
    return {
        "name": name,
        "path": _relative_path(path, project_root=project_root),
        "exists": exists,
        "writable": writable,
    }


def _storage_readiness_for_root(
    *,
    project_root: Path,
    metadata_backend: str,
    object_backend: str,
    database_url: str,
    object_bucket: str,
    object_endpoint: str,
    object_region: str,
) -> dict[str, Any]:
    metadata_backend = (metadata_backend or "local").strip().lower()
    object_backend = (object_backend or "local").strip().lower()
    local_metadata_paths = [
        _runtime_path_readiness("metadata", project_root / "metadata", project_root=project_root),
        _runtime_path_readiness("jobs", project_root / "jobs", project_root=project_root),
    ]
    local_object_paths = [
        _runtime_path_readiness("artifacts", project_root / "artifacts", project_root=project_root),
        _runtime_path_readiness("uploads", project_root / "papers" / "uploads", project_root=project_root),
    ]

    metadata_local_available = all(item["writable"] for item in local_metadata_paths)
    object_local_available = all(item["writable"] for item in local_object_paths)
    database_configured = bool(database_url.strip())
    object_configured = object_backend != "local"
    bucket_configured = bool(object_bucket.strip())
    endpoint_configured = bool(object_endpoint.strip())
    region_configured = bool(object_region.strip())

    if metadata_backend == "local":
        metadata_status = {
            "backend": "local",
            "configured": False,
            "available": metadata_local_available,
            "reason": "local_runtime_paths_ready"
            if metadata_local_available
            else "local_runtime_paths_not_writable",
            "database_url_configured": False,
        }
    elif metadata_backend in {"postgres", "postgresql"}:
        metadata_status = {
            "backend": metadata_backend,
            "configured": database_configured,
            "available": database_configured,
            "reason": "configured_not_connected"
            if database_configured
            else "database_url_missing",
            "database_url_configured": database_configured,
        }
    else:
        metadata_status = {
            "backend": metadata_backend,
            "configured": True,
            "available": False,
            "reason": "unsupported_metadata_backend",
            "database_url_configured": database_configured,
        }

    if object_backend == "local":
        object_status = {
            "backend": "local",
            "configured": False,
            "available": object_local_available,
            "reason": "local_runtime_paths_ready"
            if object_local_available
            else "local_runtime_paths_not_writable",
            "bucket_configured": False,
            "endpoint_configured": False,
            "region_configured": False,
        }
    elif object_backend in {"s3", "s3-compatible", "r2"}:
        configured = bucket_configured and endpoint_configured
        object_status = {
            "backend": object_backend,
            "configured": configured,
            "available": configured,
            "reason": "configured_not_connected"
            if configured
            else "bucket_or_endpoint_missing",
            "bucket_configured": bucket_configured,
            "endpoint_configured": endpoint_configured,
            "region_configured": region_configured,
        }
    else:
        object_status = {
            "backend": object_backend,
            "configured": object_configured,
            "available": False,
            "reason": "unsupported_object_storage_backend",
            "bucket_configured": bucket_configured,
            "endpoint_configured": endpoint_configured,
            "region_configured": region_configured,
        }

    return {
        "metadata": metadata_status,
        "object_storage": object_status,
        "local_metadata_paths": local_metadata_paths,
        "local_object_paths": local_object_paths,
        "external_storage_configured": metadata_status["configured"]
        or object_status["configured"],
        "external_storage_available": metadata_status["available"]
        and object_status["available"]
        and (metadata_status["configured"] or object_status["configured"]),
    }


def runtime_groups_for_root(project_root: Path) -> tuple[RuntimeGroupSpec, ...]:
    root = project_root.resolve()
    metadata_dir = root / "metadata"
    jobs_dir = root / "jobs"
    artifacts_dir = root / "artifacts"
    faiss_index_dir = root / "faiss_index"
    uploads_dir = root / "papers" / "uploads"
    return (
        RuntimeGroupSpec(
            name="metadata",
            path=metadata_dir,
            restore_priority="required",
            known_files=(
                RuntimeFileSpec("corpus_json", metadata_dir / "corpus.json"),
                RuntimeFileSpec("corpus_profiles_json", metadata_dir / "corpus_profiles.json"),
                RuntimeFileSpec("corpus_sqlite", metadata_dir / "corpus.sqlite3"),
                RuntimeFileSpec("chunks_sqlite", metadata_dir / "chunks.sqlite3"),
                RuntimeFileSpec("runtime_events_jsonl", metadata_dir / "runtime_events.jsonl"),
            ),
        ),
        RuntimeGroupSpec(
            name="jobs",
            path=jobs_dir,
            restore_priority="required",
            known_files=(
                RuntimeFileSpec("jobs_jsonl", jobs_dir / "jobs.jsonl"),
                RuntimeFileSpec("jobs_sqlite", jobs_dir / "jobs.sqlite3"),
            ),
        ),
        RuntimeGroupSpec(
            name="artifacts",
            path=artifacts_dir,
            restore_priority="required",
            known_files=(RuntimeFileSpec("artifacts_sqlite", artifacts_dir / "artifacts.sqlite3"),),
        ),
        RuntimeGroupSpec(
            name="uploads",
            path=uploads_dir,
            restore_priority="required",
        ),
        RuntimeGroupSpec(
            name="faiss_index",
            path=faiss_index_dir,
            restore_priority="required",
            known_files=(
                RuntimeFileSpec("index_faiss", faiss_index_dir / "index.faiss"),
                RuntimeFileSpec("index_pkl", faiss_index_dir / "index.pkl"),
                RuntimeFileSpec("active_papers_json", faiss_index_dir / "active_papers.json"),
            ),
        ),
        RuntimeGroupSpec(
            name="models",
            path=root / "models",
            restore_priority="runtime_dependency",
        ),
    )


def _storage_inventory_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    groups = []
    for group in manifest.get("groups", []):
        groups.append(
            {
                "name": group.get("name", ""),
                "path": group.get("path", ""),
                "exists": bool(group.get("exists")),
                "files": int(group.get("files", 0) or 0),
                "bytes": int(group.get("bytes", 0) or 0),
                "known_files": [
                    {
                        "name": file_info.get("name", ""),
                        "path": file_info.get("path", ""),
                        "exists": bool(file_info.get("exists")),
                        "is_file": bool(file_info.get("is_file")),
                        "bytes": int(file_info.get("bytes", 0) or 0),
                    }
                    for file_info in group.get("known_files", [])
                ],
            }
        )
    return {
        "mode": "local",
        "content_scanned": False,
        "total_files": int(manifest.get("total_files", 0) or 0),
        "total_bytes": int(manifest.get("total_bytes", 0) or 0),
        "groups": groups,
    }


def _redacted_manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest.get("schema_version"),
        "generated_at": manifest.get("generated_at"),
        "mode": manifest.get("mode"),
        "content_exported": bool(manifest.get("content_exported")),
        "secrets_exported": bool(manifest.get("secrets_exported")),
        "delete_enabled": bool(manifest.get("delete_enabled")),
        "hash_algorithm": manifest.get("hash_algorithm"),
        "env_file_present": bool(manifest.get("env_file_present")),
        "env_file_content_exported": bool(manifest.get("env_file_content_exported")),
        "total_files": int(manifest.get("total_files", 0) or 0),
        "total_bytes": int(manifest.get("total_bytes", 0) or 0),
        "group_count": len(manifest.get("groups", [])),
        "groups": [
            {
                "name": group.get("name", ""),
                "exists": bool(group.get("exists")),
                "restore_priority": group.get("restore_priority", ""),
                "files": int(group.get("files", 0) or 0),
                "bytes": int(group.get("bytes", 0) or 0),
                "known_file_count": len(group.get("known_files", [])),
                "known_files_present": sum(
                    1 for file_info in group.get("known_files", []) if file_info.get("exists")
                ),
            }
            for group in manifest.get("groups", [])
        ],
    }


def _redacted_restore_summary(restore_check: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": restore_check.get("schema_version"),
        "generated_at": restore_check.get("generated_at"),
        "mode": restore_check.get("mode"),
        "content_restored": bool(restore_check.get("content_restored")),
        "delete_enabled": bool(restore_check.get("delete_enabled")),
        "ok": bool(restore_check.get("ok")),
        "manifest_errors": list(restore_check.get("manifest_errors", [])),
        "checked_groups": int(restore_check.get("checked_groups", 0) or 0),
        "checked_files": int(restore_check.get("checked_files", 0) or 0),
        "missing_groups": int(restore_check.get("missing_groups", 0) or 0),
        "mismatched_groups": int(restore_check.get("mismatched_groups", 0) or 0),
        "missing_files": int(restore_check.get("missing_files", 0) or 0),
        "mismatched_files": int(restore_check.get("mismatched_files", 0) or 0),
        "groups": [
            {
                "name": group.get("name", ""),
                "expected_exists": bool(group.get("expected_exists")),
                "exists": bool(group.get("exists")),
                "status": group.get("status", ""),
                "ok": bool(group.get("ok")),
                "expected_files": int(group.get("expected_files", 0) or 0),
                "files": int(group.get("files", 0) or 0),
                "known_file_count": len(group.get("known_files", [])),
            }
            for group in restore_check.get("groups", [])
        ],
    }


def _redacted_storage_schema_summary(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": status.get("schema_version"),
        "mode": status.get("mode"),
        "ok": bool(status.get("ok")),
        "store_count": int(status.get("store_count", 0) or 0),
        "problem_count": int(status.get("problem_count", 0) or 0),
        "stores": [
            {
                "name": store.get("name", ""),
                "kind": store.get("kind", ""),
                "required": bool(store.get("required")),
                "exists": bool(store.get("exists")),
                "ok": bool(store.get("ok")),
                "errors": list(store.get("errors", [])),
            }
            for store in status.get("stores", [])
        ],
    }


def _parse_optional_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _connect_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _local_job_contract_for_root(project_root: Path) -> dict[str, Any]:
    jobs_dir = project_root / "jobs"
    jobs_jsonl = jobs_dir / "jobs.jsonl"
    jobs_sqlite = jobs_dir / "jobs.sqlite3"
    contract: dict[str, Any] = {
        "storage": {
            "jsonl_exists": jobs_jsonl.exists(),
            "jsonl_bytes": jobs_jsonl.stat().st_size if jobs_jsonl.exists() else 0,
            "sqlite_exists": jobs_sqlite.exists(),
            "sqlite_bytes": jobs_sqlite.stat().st_size if jobs_sqlite.exists() else 0,
        },
        "queue_health": {},
        "worker_leases": {},
        "errors": [],
        "content_scanned": False,
        "payload_exported": False,
        "worker_ids_exported": False,
    }
    if not jobs_sqlite.exists():
        contract["errors"].append("jobs_sqlite_missing")
        return contract

    try:
        with _connect_readonly(jobs_sqlite) as conn:
            tables = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if "jobs" not in tables:
                contract["errors"].append("jobs_table_missing")
                return contract
            rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT status, created_at, updated_at, not_before, deadline_at,
                           worker_id, lease_expires_at
                    FROM jobs
                    """
                ).fetchall()
            ]
    except sqlite3.Error:
        contract["errors"].append("jobs_sqlite_unreadable")
        return contract

    now = datetime.now(timezone.utc)
    queued = [row for row in rows if row.get("status") == "queued"]
    active_leased_queued = [
        row
        for row in queued
        if row.get("worker_id")
        and (lease_expires_at := _parse_optional_utc(row.get("lease_expires_at")))
        and lease_expires_at > now
    ]
    expired_leased_queued = [
        row
        for row in queued
        if row.get("worker_id")
        and (lease_expires_at := _parse_optional_utc(row.get("lease_expires_at")))
        and lease_expires_at <= now
    ]
    due = [
        row
        for row in queued
        if not (not_before := _parse_optional_utc(row.get("not_before"))) or not_before <= now
        if row not in active_leased_queued
    ]
    scheduled = [
        row
        for row in queued
        if (not_before := _parse_optional_utc(row.get("not_before"))) and not_before > now
    ]
    expired = [
        row
        for row in queued
        if (deadline_at := _parse_optional_utc(row.get("deadline_at"))) and deadline_at <= now
    ]
    running = [row for row in rows if row.get("status") == "running"]
    running_leased = [row for row in running if row.get("worker_id")]
    leased_jobs = [row for row in rows if row.get("worker_id")]
    active_jobs = [
        row
        for row in leased_jobs
        if row.get("status") in {"queued", "running"}
        and (lease_expires_at := _parse_optional_utc(row.get("lease_expires_at")))
        and lease_expires_at > now
    ]
    expired_jobs = [
        row
        for row in leased_jobs
        if row.get("status") in {"queued", "running"}
        and (lease_expires_at := _parse_optional_utc(row.get("lease_expires_at")))
        and lease_expires_at <= now
    ]
    oldest_queued_at = min(
        (str(row.get("created_at")) for row in queued if row.get("created_at")),
        default=None,
    )
    contract["queue_health"] = {
        "queued": len(queued),
        "due": len(due),
        "scheduled": len(scheduled),
        "expired": len(expired),
        "running": len(running),
        "leased_queued": len(active_leased_queued),
        "lease_expired_queued": len(expired_leased_queued),
        "running_leased": len(running_leased),
        "oldest_queued_at": oldest_queued_at,
    }
    contract["worker_leases"] = {
        "total_leased_jobs": len(leased_jobs),
        "worker_ids": [],
        "active_worker_ids": [],
        "expired_worker_ids": [],
        "active_leases": len(active_jobs),
        "expired_leases": len(expired_jobs),
    }
    return contract


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _local_preflight_blockers(
    *,
    storage_schema: dict[str, Any],
    runtime_manifest: dict[str, Any],
    restore_check: dict[str, Any],
    platform_readiness: dict[str, Any],
    jobs: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not storage_schema.get("ok"):
        blockers.append("local_storage_schema_drift")
    if runtime_manifest.get("mode") != "local_runtime_backup_manifest":
        blockers.append("runtime_backup_manifest_unavailable")
    if runtime_manifest.get("content_exported") is not False:
        blockers.append("runtime_backup_manifest_exported_content")
    if runtime_manifest.get("secrets_exported") is not False:
        blockers.append("runtime_backup_manifest_exported_secrets")
    if not runtime_manifest.get("groups"):
        blockers.append("runtime_backup_manifest_empty")
    if not restore_check.get("ok"):
        blockers.append("runtime_restore_dry_run_failed")
    blockers.extend(
        blocker
        for blocker in platform_readiness.get("storage_migration", {}).get("blockers", [])
        if blocker not in EXTERNAL_STORAGE_BLOCKERS
    )
    blockers.extend(
        blocker
        for blocker in platform_readiness.get("distributed_workers", {}).get("blockers", [])
        if blocker not in EXTERNAL_WORKER_BLOCKERS
    )
    blockers.extend(str(error) for error in jobs.get("errors", []))
    return _dedupe(blockers)


def collect_platform_migration_preflight(
    *,
    project_root: Path = config.PROJECT_ROOT,
    generated_at: str | None = None,
    metadata_backend: str = config.METADATA_STORAGE_BACKEND,
    database_url: str = config.DATABASE_URL,
    object_backend: str = config.OBJECT_STORAGE_BACKEND,
    object_bucket: str = config.OBJECT_STORAGE_BUCKET,
    object_endpoint: str = config.OBJECT_STORAGE_ENDPOINT,
    object_region: str = config.OBJECT_STORAGE_REGION,
    distributed_job_store_backend: str = config.DISTRIBUTED_JOB_STORE_BACKEND,
    distributed_job_store_url: str = config.DISTRIBUTED_JOB_STORE_URL,
    distributed_job_queue_name: str = config.DISTRIBUTED_JOB_QUEUE_NAME,
) -> dict[str, Any]:
    """Return no-secret local and activation readiness for production migration."""
    root = project_root.resolve()
    runtime_manifest = collect_runtime_backup_manifest(
        project_root=root,
        groups=runtime_groups_for_root(root),
        generated_at=generated_at,
    )
    restore_check = collect_runtime_restore_check(
        runtime_manifest,
        project_root=root,
        generated_at=generated_at,
    )
    storage_schema = storage_schema_status_for_root(root)
    storage_inventory = _storage_inventory_from_manifest(runtime_manifest)
    storage_readiness = _storage_readiness_for_root(
        project_root=root,
        metadata_backend=metadata_backend,
        object_backend=object_backend,
        database_url=database_url,
        object_bucket=object_bucket,
        object_endpoint=object_endpoint,
        object_region=object_region,
    )
    distributed_store = distributed_job_store_status(
        backend=distributed_job_store_backend,
        store_url=distributed_job_store_url,
        queue_name=distributed_job_queue_name,
    )
    jobs = _local_job_contract_for_root(root)
    platform_readiness = platform_readiness_status(
        storage_readiness=storage_readiness,
        storage_schemas=storage_schema,
        storage=storage_inventory,
        jobs=jobs,
        distributed_job_store=distributed_store,
    )
    local_blockers = _local_preflight_blockers(
        storage_schema=storage_schema,
        runtime_manifest=runtime_manifest,
        restore_check=restore_check,
        platform_readiness=platform_readiness,
        jobs=jobs,
    )
    activation_blockers = _dedupe(
        local_blockers
        + list(platform_readiness.get("storage_migration", {}).get("blockers", []))
        + list(platform_readiness.get("distributed_workers", {}).get("blockers", []))
    )
    preflight_ok = not local_blockers
    activation_ready = preflight_ok and bool(platform_readiness.get("overall_ready"))
    return {
        "schema_version": PLATFORM_MIGRATION_PREFLIGHT_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now(),
        "mode": "production_migration_preflight",
        "content_exported": False,
        "secrets_exported": False,
        "connectivity_checked": False,
        "activation_enabled": False,
        "preflight_ok": preflight_ok,
        "activation_ready": activation_ready,
        "summary": {
            "local_storage_schema_ok": bool(storage_schema.get("ok")),
            "runtime_backup_manifest_ok": bool(runtime_manifest.get("groups"))
            and runtime_manifest.get("content_exported") is False
            and runtime_manifest.get("secrets_exported") is False,
            "runtime_restore_dry_run_ok": bool(restore_check.get("ok")),
            "local_durable_job_store_ready": bool(jobs.get("storage", {}).get("sqlite_exists")),
            "queue_health_contract_ready": bool(
                platform_readiness.get("distributed_workers", {})
                .get("checks", {})
                .get("queue_health_contract_ready")
            ),
            "worker_lease_contract_ready": bool(
                platform_readiness.get("distributed_workers", {})
                .get("checks", {})
                .get("worker_lease_contract_ready")
            ),
            "external_storage_configured": bool(
                storage_readiness.get("external_storage_configured")
            ),
            "external_storage_available": bool(storage_readiness.get("external_storage_available")),
            "external_job_store_configured": bool(
                distributed_store.get("external_job_store_configured")
            ),
            "external_job_store_available": bool(
                distributed_store.get("external_job_store_available")
            ),
            "platform_storage_migration_ready": bool(
                platform_readiness.get("storage_migration", {}).get("ready")
            ),
            "platform_distributed_workers_ready": bool(
                platform_readiness.get("distributed_workers", {}).get("ready")
            ),
        },
        "blockers": {
            "local_preflight": local_blockers,
            "activation": activation_blockers,
        },
        "storage_schema": _redacted_storage_schema_summary(storage_schema),
        "runtime_backup": _redacted_manifest_summary(runtime_manifest),
        "runtime_restore_check": _redacted_restore_summary(restore_check),
        "jobs": jobs,
        "storage_readiness": storage_readiness,
        "distributed_job_store": distributed_store,
        "platform_readiness": platform_readiness,
    }


def _format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def format_platform_migration_preflight_markdown(status: dict[str, Any]) -> str:
    """Render production migration preflight as no-secret Markdown."""
    summary = status.get("summary", {})
    blockers = status.get("blockers", {})
    runtime_backup = status.get("runtime_backup", {})
    runtime_restore = status.get("runtime_restore_check", {})
    storage_schema = status.get("storage_schema", {})
    jobs = status.get("jobs", {})
    job_storage = jobs.get("storage", {})
    lines = [
        "# FluxMind Production Migration Preflight",
        "",
        "No runtime contents, job payloads, external URLs, buckets, queue names, credentials, or secrets are exported.",
        "",
        f"- Generated at: {status.get('generated_at', '')}",
        f"- Mode: {status.get('mode', '')}",
        f"- Preflight OK: {_format_bool(status.get('preflight_ok', False))}",
        f"- Activation ready: {_format_bool(status.get('activation_ready', False))}",
        f"- Activation enabled: {_format_bool(status.get('activation_enabled', False))}",
        f"- Connectivity checked: {_format_bool(status.get('connectivity_checked', False))}",
        f"- Content exported: {_format_bool(status.get('content_exported', False))}",
        f"- Secrets exported: {_format_bool(status.get('secrets_exported', False))}",
        "",
        "## Summary",
        "",
    ]
    for key in sorted(summary):
        lines.append(f"- {key}: {_format_bool(summary[key]) if isinstance(summary[key], bool) else summary[key]}")
    lines.extend(
        [
            "",
            "## Blockers",
            "",
            f"- Local preflight: {', '.join(blockers.get('local_preflight', [])) or 'none'}",
            f"- Activation: {', '.join(blockers.get('activation', [])) or 'none'}",
            "",
            "## Local Evidence",
            "",
            f"- Storage schema OK: {_format_bool(storage_schema.get('ok', False))}",
            f"- Storage stores: {storage_schema.get('store_count', 0)}",
            f"- Storage problems: {storage_schema.get('problem_count', 0)}",
            f"- Runtime manifest groups: {runtime_backup.get('group_count', 0)}",
            f"- Runtime manifest files: {runtime_backup.get('total_files', 0)}",
            f"- Restore dry-run OK: {_format_bool(runtime_restore.get('ok', False))}",
            f"- Restore checked groups: {runtime_restore.get('checked_groups', 0)}",
            f"- Restore checked files: {runtime_restore.get('checked_files', 0)}",
            f"- Jobs JSONL exists: {_format_bool(job_storage.get('jsonl_exists', False))}",
            f"- Jobs SQLite exists: {_format_bool(job_storage.get('sqlite_exists', False))}",
            f"- Job contract errors: {', '.join(jobs.get('errors', [])) or 'none'}",
            "",
            "## Activation Evidence",
            "",
            f"- External storage configured: {_format_bool(summary.get('external_storage_configured', False))}",
            f"- External storage available: {_format_bool(summary.get('external_storage_available', False))}",
            f"- External job store configured: {_format_bool(summary.get('external_job_store_configured', False))}",
            f"- External job store available: {_format_bool(summary.get('external_job_store_available', False))}",
            f"- Platform storage migration ready: {_format_bool(summary.get('platform_storage_migration_ready', False))}",
            f"- Platform distributed workers ready: {_format_bool(summary.get('platform_distributed_workers_ready', False))}",
        ]
    )
    return "\n".join(lines)
