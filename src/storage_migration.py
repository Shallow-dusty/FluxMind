"""Local runtime migration rehearsal helpers.

The rehearsal copies runtime state into a staging root, then verifies that the
staged tree matches a no-secret manifest and still satisfies local storage
schema checks. It is a local backup/restore drill, not external database or
object-storage activation.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import config
from src.platform_migration import (
    collect_platform_migration_preflight,
    runtime_groups_for_root,
)
from src.storage_manifest import (
    collect_runtime_backup_manifest,
    collect_runtime_restore_check,
)
from src.storage_schema import storage_schema_status_for_root


STORAGE_MIGRATION_REHEARSAL_SCHEMA_VERSION = 1
OBJECT_STORAGE_MIGRATION_MANIFEST_SCHEMA_VERSION = 1
OBJECT_STORAGE_MIGRATION_VERIFY_SCHEMA_VERSION = 1
JOB_STORE_MIGRATION_MANIFEST_SCHEMA_VERSION = 1
JOB_STORE_MIGRATION_VERIFY_SCHEMA_VERSION = 1
DEFAULT_OBJECT_KEY_PREFIX = "fluxmind-runtime"
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_GROUP_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_UNSAFE_OBJECT_MANIFEST_FIELDS = {
    "bucket",
    "bucket_name",
    "content",
    "credential",
    "credentials",
    "endpoint",
    "file_name",
    "filename",
    "filenames",
    "path",
    "secret",
    "source_path",
    "source_paths",
}
_UNSAFE_JOB_MANIFEST_FIELDS = {
    "artifact",
    "artifacts",
    "credential",
    "credentials",
    "error",
    "idempotency_key",
    "logs",
    "owner_id",
    "owner_label",
    "payload",
    "request",
    "request_id",
    "result",
    "secret",
    "stderr",
    "stdout",
    "worker_id",
}
JOB_MANIFEST_BUCKET_MAP_FIELDS = {
    "by_kind",
    "by_ownership_source",
    "by_status",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _empty_directory(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True)
        return
    for item in path.iterdir():
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()


def _directory_has_entries(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object_key_prefix_is_safe(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    prefix = value.strip()
    if (
        not prefix
        or prefix != prefix.strip("/")
        or "://" in prefix
        or "\\" in prefix
        or prefix.startswith("~")
        or re.match(r"^[A-Za-z]:", prefix)
    ):
        return False
    segments = prefix.split("/")
    return all(
        segment
        and segment not in {".", ".."}
        and _SAFE_GROUP_RE.fullmatch(segment)
        for segment in segments
    )


def _sanitize_object_key_prefix(value: str) -> str:
    raw_prefix = (value or DEFAULT_OBJECT_KEY_PREFIX).strip()
    if not _object_key_prefix_is_safe(raw_prefix):
        prefix = DEFAULT_OBJECT_KEY_PREFIX
    else:
        prefix = raw_prefix
    prefix = re.sub(r"[^A-Za-z0-9._/-]+", "-", prefix)
    if not _object_key_prefix_is_safe(prefix):
        return DEFAULT_OBJECT_KEY_PREFIX
    return prefix or DEFAULT_OBJECT_KEY_PREFIX


def _path_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _job_token(job_id: str) -> str:
    return hashlib.sha256(f"job:{job_id}".encode("utf-8")).hexdigest()


def _claim_token(kind: str, idempotency_key: str, job_id: str) -> str:
    value = f"claim:{kind}:{idempotency_key}:{job_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _increment_count(counts: dict[str, int], key: Any) -> None:
    clean_key = str(key or "none")
    counts[clean_key] = counts.get(clean_key, 0) + 1


def _manifest_key_forms(key: Any) -> tuple[str, str]:
    expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key).strip())
    normalized = re.sub(r"[^a-z0-9]+", "_", expanded.casefold()).strip("_")
    compact = re.sub(r"[^a-z0-9]+", "", expanded.casefold())
    return normalized, compact


def _manifest_key_is_unsafe(key: Any, unsafe_fields: set[str]) -> bool:
    normalized, compact = _manifest_key_forms(key)
    for field in unsafe_fields:
        unsafe_normalized, unsafe_compact = _manifest_key_forms(field)
        if normalized == unsafe_normalized or compact == unsafe_compact:
            return True
    return False


def _has_unsafe_manifest_field(value: Any, unsafe_fields: set[str]) -> bool:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            if _manifest_key_is_unsafe(key, unsafe_fields):
                return True
            if _has_unsafe_manifest_field(nested_value, unsafe_fields):
                return True
    elif isinstance(value, list):
        return any(_has_unsafe_manifest_field(item, unsafe_fields) for item in value)
    return False


def _copy_runtime_group(
    *,
    name: str,
    source_path: Path,
    target_path: Path,
    restore_priority: str,
    include_runtime_dependencies: bool,
) -> dict[str, Any]:
    if restore_priority == "runtime_dependency" and not include_runtime_dependencies:
        return {
            "name": name,
            "restore_priority": restore_priority,
            "source_exists": source_path.exists(),
            "status": "skipped_runtime_dependency",
            "copied_files": 0,
            "copied_bytes": 0,
            "skipped_symlinks": 0,
            "errors": [],
        }
    if not source_path.exists():
        return {
            "name": name,
            "restore_priority": restore_priority,
            "source_exists": False,
            "status": "source_absent",
            "copied_files": 0,
            "copied_bytes": 0,
            "skipped_symlinks": 0,
            "errors": [],
        }
    if not source_path.is_dir():
        return {
            "name": name,
            "restore_priority": restore_priority,
            "source_exists": True,
            "status": "source_not_directory",
            "copied_files": 0,
            "copied_bytes": 0,
            "skipped_symlinks": 0,
            "errors": ["source_not_directory"],
        }

    copied_files = 0
    copied_bytes = 0
    skipped_symlinks = 0
    errors: list[str] = []
    target_path.mkdir(parents=True, exist_ok=True)
    for item in source_path.rglob("*"):
        relative = item.relative_to(source_path)
        target = target_path / relative
        if item.is_symlink():
            skipped_symlinks += 1
            continue
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not item.is_file():
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            copied_files += 1
            copied_bytes += item.stat().st_size
        except OSError:
            errors.append("copy_failed")
            break

    return {
        "name": name,
        "restore_priority": restore_priority,
        "source_exists": True,
        "status": "copied" if not errors else "copy_failed",
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
        "skipped_symlinks": skipped_symlinks,
        "errors": errors,
    }


def _manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest.get("schema_version"),
        "mode": manifest.get("mode"),
        "content_exported": bool(manifest.get("content_exported")),
        "secrets_exported": bool(manifest.get("secrets_exported")),
        "env_file_present": bool(manifest.get("env_file_present")),
        "env_file_content_exported": bool(manifest.get("env_file_content_exported")),
        "total_files": int(manifest.get("total_files", 0) or 0),
        "total_bytes": int(manifest.get("total_bytes", 0) or 0),
        "group_count": len(manifest.get("groups", [])),
    }


def _restore_summary(check: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": check.get("schema_version"),
        "mode": check.get("mode"),
        "ok": bool(check.get("ok")),
        "content_restored": bool(check.get("content_restored")),
        "delete_enabled": bool(check.get("delete_enabled")),
        "manifest_errors": list(check.get("manifest_errors", [])),
        "checked_groups": int(check.get("checked_groups", 0) or 0),
        "checked_files": int(check.get("checked_files", 0) or 0),
        "missing_groups": int(check.get("missing_groups", 0) or 0),
        "mismatched_groups": int(check.get("mismatched_groups", 0) or 0),
        "missing_files": int(check.get("missing_files", 0) or 0),
        "mismatched_files": int(check.get("mismatched_files", 0) or 0),
    }


def _schema_summary(status: dict[str, Any]) -> dict[str, Any]:
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
                "exists": bool(store.get("exists")),
                "ok": bool(store.get("ok")),
                "errors": list(store.get("errors", [])),
            }
            for store in status.get("stores", [])
        ],
    }


def collect_object_storage_migration_manifest(
    *,
    project_root: Path = config.PROJECT_ROOT,
    groups: tuple[Any, ...] | None = None,
    include_runtime_dependencies: bool = False,
    key_prefix: str = DEFAULT_OBJECT_KEY_PREFIX,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Return an opaque object manifest for copying runtime files to object storage.

    The manifest deliberately omits source paths, filenames, buckets, endpoints,
    and credentials. It gives an operator enough hash/key/count evidence to
    validate a future upload plan without exposing runtime contents in reports.
    """
    root = project_root.resolve()
    runtime_groups = groups if groups is not None else runtime_groups_for_root(root)
    if not include_runtime_dependencies:
        runtime_groups = tuple(
            group for group in runtime_groups if group.restore_priority != "runtime_dependency"
        )
    prefix = _sanitize_object_key_prefix(key_prefix)
    objects: list[dict[str, Any]] = []
    group_summaries: list[dict[str, Any]] = []
    unique_keys: set[str] = set()

    for group in runtime_groups:
        group_path = group.path.resolve()
        group_objects = 0
        group_bytes = 0
        group_exists = group_path.exists()
        group_is_dir = group_path.is_dir() and not group_path.is_symlink()
        if group_is_dir:
            for item in sorted(group_path.rglob("*"), key=lambda path: path.as_posix()):
                if item.is_symlink() or not item.is_file():
                    continue
                relative_path = item.relative_to(group_path).as_posix()
                digest = _sha256_file(item)
                size = item.stat().st_size
                object_key = f"{prefix}/{group.name}/{digest[:2]}/{digest}"
                unique_keys.add(object_key)
                group_objects += 1
                group_bytes += size
                objects.append(
                    {
                        "group": group.name,
                        "restore_priority": group.restore_priority,
                        "object_key": object_key,
                        "bytes": size,
                        "sha256": digest,
                        "source_path_token": _path_token(f"{group.name}/{relative_path}"),
                    }
                )

        group_summaries.append(
            {
                "name": group.name,
                "restore_priority": group.restore_priority,
                "source_exists": group_exists,
                "source_is_directory": group_is_dir,
                "object_count": group_objects,
                "bytes": group_bytes,
            }
        )

    return {
        "schema_version": OBJECT_STORAGE_MIGRATION_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now(),
        "mode": "object_storage_migration_manifest",
        "content_exported": False,
        "secrets_exported": False,
        "source_paths_exported": False,
        "filenames_exported": False,
        "bucket_exported": False,
        "external_connectivity_checked": False,
        "hash_algorithm": "sha256",
        "object_key_strategy": "grouped_by_content_sha256",
        "key_prefix": prefix,
        "group_count": len(group_summaries),
        "object_count": len(objects),
        "unique_object_count": len(unique_keys),
        "duplicate_content_references": len(objects) - len(unique_keys),
        "total_bytes": sum(int(item.get("bytes", 0) or 0) for item in objects),
        "groups": group_summaries,
        "objects": objects,
    }


def _object_manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest.get("schema_version"),
        "mode": manifest.get("mode", ""),
        "content_exported": bool(manifest.get("content_exported")),
        "secrets_exported": bool(manifest.get("secrets_exported")),
        "source_paths_exported": bool(manifest.get("source_paths_exported")),
        "filenames_exported": bool(manifest.get("filenames_exported")),
        "bucket_exported": bool(manifest.get("bucket_exported")),
        "external_connectivity_checked": bool(manifest.get("external_connectivity_checked")),
        "hash_algorithm": manifest.get("hash_algorithm", ""),
        "object_key_strategy": manifest.get("object_key_strategy", ""),
        "group_count": int(manifest.get("group_count", 0) or 0),
        "object_count": int(manifest.get("object_count", 0) or 0),
        "unique_object_count": int(manifest.get("unique_object_count", 0) or 0),
        "duplicate_content_references": int(
            manifest.get("duplicate_content_references", 0) or 0
        ),
        "total_bytes": int(manifest.get("total_bytes", 0) or 0),
    }


def _job_store_manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest.get("schema_version"),
        "mode": manifest.get("mode", ""),
        "ok": bool(manifest.get("ok")),
        "content_exported": bool(manifest.get("content_exported")),
        "secrets_exported": bool(manifest.get("secrets_exported")),
        "payload_exported": bool(manifest.get("payload_exported")),
        "owner_ids_exported": bool(manifest.get("owner_ids_exported")),
        "request_ids_exported": bool(manifest.get("request_ids_exported")),
        "worker_ids_exported": bool(manifest.get("worker_ids_exported")),
        "idempotency_keys_exported": bool(manifest.get("idempotency_keys_exported")),
        "external_connectivity_checked": bool(manifest.get("external_connectivity_checked")),
        "job_count": int(manifest.get("job_count", 0) or 0),
        "claim_count": int(manifest.get("idempotency_claim_count", 0) or 0),
        "manifest_errors": list(manifest.get("manifest_errors", [])),
    }


def _job_store_records_for_root(
    project_root: Path,
    *,
    reference_time: datetime | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[str],
]:
    jobs_dir = project_root / "jobs"
    jobs_jsonl = jobs_dir / "jobs.jsonl"
    jobs_sqlite = jobs_dir / "jobs.sqlite3"
    storage = {
        "jsonl_exists": jobs_jsonl.exists(),
        "jsonl_bytes": jobs_jsonl.stat().st_size if jobs_jsonl.exists() else 0,
        "sqlite_exists": jobs_sqlite.exists(),
        "sqlite_bytes": jobs_sqlite.stat().st_size if jobs_sqlite.exists() else 0,
    }
    errors: list[str] = []
    if not jobs_sqlite.exists():
        return [], [], storage, ["jobs_sqlite_missing"]

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{jobs_sqlite}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if "jobs" not in tables:
            return [], [], storage, ["jobs_table_missing"]
        if "job_idempotency" not in tables:
            errors.append("job_idempotency_table_missing")

        rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT
                    job_id,
                    kind,
                    status,
                    created_at,
                    updated_at,
                    attempts,
                    not_before,
                    deadline_at,
                    CASE
                        WHEN worker_id IS NOT NULL AND worker_id != '' THEN 1
                        ELSE 0
                    END AS has_worker_id,
                    leased_at,
                    lease_expires_at,
                    CASE
                        WHEN idempotency_key IS NOT NULL AND idempotency_key != '' THEN 1
                        ELSE 0
                    END AS has_idempotency_key,
                    max_attempts,
                    retry_backoff_s,
                    dead_lettered_at,
                    ownership_source
                FROM jobs
                """
            ).fetchall()
        ]
        claim_rows = (
            [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT kind, idempotency_key, job_id, created_at
                    FROM job_idempotency
                    """
                ).fetchall()
            ]
            if "job_idempotency" in tables
            else []
        )
    except sqlite3.Error:
        return [], [], storage, ["jobs_sqlite_unreadable"]
    finally:
        if conn is not None:
            conn.close()

    now = reference_time or datetime.now(timezone.utc)
    job_records: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("status") or "")
        not_before = _parse_optional_utc(row.get("not_before"))
        deadline_at = _parse_optional_utc(row.get("deadline_at"))
        lease_expires_at = _parse_optional_utc(row.get("lease_expires_at"))
        has_lease = bool(row.get("has_worker_id") or row.get("leased_at") or row.get("lease_expires_at"))
        job_records.append(
            {
                "job_token": _job_token(str(row.get("job_id") or "")),
                "kind": str(row.get("kind") or ""),
                "status": status,
                "created_at": str(row.get("created_at") or ""),
                "updated_at": str(row.get("updated_at") or ""),
                "attempts": int(row.get("attempts") or 0),
                "max_attempts": int(row.get("max_attempts") or 0),
                "retry_backoff_s": int(row.get("retry_backoff_s") or 0),
                "ownership_source": str(row.get("ownership_source") or "default"),
                "has_not_before": bool(row.get("not_before")),
                "is_due": status == "queued" and (not_before is None or not_before <= now),
                "is_scheduled": status == "queued" and not_before is not None and not_before > now,
                "has_deadline": bool(row.get("deadline_at")),
                "deadline_expired": deadline_at is not None and deadline_at <= now,
                "has_lease": has_lease,
                "lease_expired": has_lease and lease_expires_at is not None and lease_expires_at <= now,
                "has_idempotency_claim": bool(row.get("has_idempotency_key")),
                "dead_lettered": bool(row.get("dead_lettered_at")) or status == "dead_lettered",
            }
        )

    claim_records = [
        {
            "claim_token": _claim_token(
                str(row.get("kind") or ""),
                str(row.get("idempotency_key") or ""),
                str(row.get("job_id") or ""),
            ),
            "kind": str(row.get("kind") or ""),
            "job_token": _job_token(str(row.get("job_id") or "")),
            "created_at": str(row.get("created_at") or ""),
        }
        for row in claim_rows
    ]
    return job_records, claim_records, storage, errors


def collect_job_store_migration_manifest(
    *,
    project_root: Path = config.PROJECT_ROOT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Return a no-secret manifest for the local durable job store.

    The manifest verifies which job rows and idempotency claims would need to
    move to a future distributed job backend. It never exports job payloads,
    owner IDs, request IDs, worker IDs, idempotency keys, stdout/stderr, errors,
    logs, artifacts, or generated contents.
    """
    root = project_root.resolve()
    generated_at_value = generated_at or _utc_now()
    reference_time = _parse_optional_utc(generated_at_value) or datetime.now(timezone.utc)
    job_records, claim_records, storage, errors = _job_store_records_for_root(
        root,
        reference_time=reference_time,
    )
    by_status: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    by_ownership_source: dict[str, int] = {}
    for record in job_records:
        _increment_count(by_status, record.get("status"))
        _increment_count(by_kind, record.get("kind"))
        _increment_count(by_ownership_source, record.get("ownership_source"))

    queue_summary = {
        "queued": sum(1 for record in job_records if record.get("status") == "queued"),
        "running": sum(1 for record in job_records if record.get("status") == "running"),
        "due": sum(1 for record in job_records if record.get("is_due")),
        "scheduled": sum(1 for record in job_records if record.get("is_scheduled")),
        "deadline_expired": sum(1 for record in job_records if record.get("deadline_expired")),
        "leased": sum(1 for record in job_records if record.get("has_lease")),
        "lease_expired": sum(1 for record in job_records if record.get("lease_expired")),
        "dead_lettered": sum(1 for record in job_records if record.get("dead_lettered")),
        "idempotency_claimed_jobs": sum(
            1 for record in job_records if record.get("has_idempotency_claim")
        ),
    }
    timeline = {
        "earliest_created_at": min(
            (record["created_at"] for record in job_records if record.get("created_at")),
            default=None,
        ),
        "latest_updated_at": max(
            (record["updated_at"] for record in job_records if record.get("updated_at")),
            default=None,
        ),
    }
    return {
        "schema_version": JOB_STORE_MIGRATION_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at_value,
        "mode": "job_store_migration_manifest",
        "ok": not errors,
        "content_exported": False,
        "secrets_exported": False,
        "payload_exported": False,
        "owner_ids_exported": False,
        "request_ids_exported": False,
        "worker_ids_exported": False,
        "idempotency_keys_exported": False,
        "external_connectivity_checked": False,
        "hash_algorithm": "sha256",
        "record_identity": "hashed_job_and_idempotency_claim_tokens",
        "storage": storage,
        "job_count": len(job_records),
        "idempotency_claim_count": len(claim_records),
        "by_status": dict(sorted(by_status.items())),
        "by_kind": dict(sorted(by_kind.items())),
        "by_ownership_source": dict(sorted(by_ownership_source.items())),
        "queue_summary": queue_summary,
        "timeline": timeline,
        "manifest_errors": sorted(set(errors)),
        "jobs": sorted(job_records, key=lambda item: item["job_token"]),
        "idempotency_claims": sorted(claim_records, key=lambda item: item["claim_token"]),
    }


def _object_manifest_includes_runtime_dependencies(manifest: dict[str, Any]) -> bool:
    for group in manifest.get("groups", []):
        if isinstance(group, dict) and group.get("restore_priority") == "runtime_dependency":
            return True
    for item in manifest.get("objects", []):
        if isinstance(item, dict) and item.get("restore_priority") == "runtime_dependency":
            return True
    return False


def _object_manifest_schema_errors(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != OBJECT_STORAGE_MIGRATION_MANIFEST_SCHEMA_VERSION:
        errors.append("schema_version_unsupported")
    if manifest.get("mode") != "object_storage_migration_manifest":
        errors.append("mode_invalid")
    if manifest.get("hash_algorithm") != "sha256":
        errors.append("hash_algorithm_invalid")
    if manifest.get("object_key_strategy") != "grouped_by_content_sha256":
        errors.append("object_key_strategy_invalid")
    for field in (
        "content_exported",
        "secrets_exported",
        "source_paths_exported",
        "filenames_exported",
        "bucket_exported",
        "external_connectivity_checked",
    ):
        if manifest.get(field) is not False:
            errors.append(f"{field}_must_be_false")
    key_prefix = manifest.get("key_prefix", DEFAULT_OBJECT_KEY_PREFIX)
    if not _object_key_prefix_is_safe(key_prefix):
        errors.append("key_prefix_invalid")
    if _has_unsafe_manifest_field(manifest, _UNSAFE_OBJECT_MANIFEST_FIELDS):
        errors.append("manifest_contains_unsafe_field")

    groups = manifest.get("groups")
    objects = manifest.get("objects")
    if not isinstance(groups, list):
        errors.append("groups_not_list")
        groups = []
    if not isinstance(objects, list):
        errors.append("objects_not_list")
        objects = []
    for group in groups:
        if not isinstance(group, dict):
            errors.append("group_entry_invalid")
            continue
        if _has_unsafe_manifest_field(group, _UNSAFE_OBJECT_MANIFEST_FIELDS):
            errors.append("group_entry_contains_unsafe_field")
        name = group.get("name")
        if not isinstance(name, str) or not _SAFE_GROUP_RE.fullmatch(name):
            errors.append("group_entry_invalid")

    if isinstance(manifest.get("group_count"), int) and manifest.get("group_count") != len(groups):
        errors.append("group_count_mismatch")
    if isinstance(manifest.get("object_count"), int) and manifest.get("object_count") != len(objects):
        errors.append("object_count_mismatch")
    object_keys = [
        item.get("object_key")
        for item in objects
        if isinstance(item, dict) and isinstance(item.get("object_key"), str)
    ]
    if (
        isinstance(manifest.get("unique_object_count"), int)
        and manifest.get("unique_object_count") != len(set(object_keys))
    ):
        errors.append("unique_object_count_mismatch")
    object_bytes = [
        item.get("bytes")
        for item in objects
        if isinstance(item, dict) and isinstance(item.get("bytes"), int)
    ]
    if (
        isinstance(manifest.get("total_bytes"), int)
        and manifest.get("total_bytes") != sum(object_bytes)
    ):
        errors.append("total_bytes_mismatch")
    return errors


def _job_manifest_schema_errors(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != JOB_STORE_MIGRATION_MANIFEST_SCHEMA_VERSION:
        errors.append("schema_version_unsupported")
    if manifest.get("mode") != "job_store_migration_manifest":
        errors.append("mode_invalid")
    if manifest.get("hash_algorithm") != "sha256":
        errors.append("hash_algorithm_invalid")
    if manifest.get("record_identity") != "hashed_job_and_idempotency_claim_tokens":
        errors.append("record_identity_invalid")
    for field in (
        "content_exported",
        "secrets_exported",
        "payload_exported",
        "owner_ids_exported",
        "request_ids_exported",
        "worker_ids_exported",
        "idempotency_keys_exported",
        "external_connectivity_checked",
    ):
        if manifest.get(field) is not False:
            errors.append(f"{field}_must_be_false")
    manifest_for_unsafe_scan = {
        key: ({} if key in JOB_MANIFEST_BUCKET_MAP_FIELDS and isinstance(value, dict) else value)
        for key, value in manifest.items()
    }
    if _has_unsafe_manifest_field(manifest_for_unsafe_scan, _UNSAFE_JOB_MANIFEST_FIELDS):
        errors.append("manifest_contains_unsafe_field")

    jobs = manifest.get("jobs")
    claims = manifest.get("idempotency_claims")
    if not isinstance(jobs, list):
        errors.append("jobs_not_list")
        jobs = []
    if not isinstance(claims, list):
        errors.append("idempotency_claims_not_list")
        claims = []
    if isinstance(manifest.get("job_count"), int) and manifest.get("job_count") != len(jobs):
        errors.append("job_count_mismatch")
    if (
        isinstance(manifest.get("idempotency_claim_count"), int)
        and manifest.get("idempotency_claim_count") != len(claims)
    ):
        errors.append("idempotency_claim_count_mismatch")
    return errors


def _index_object_manifest_objects(
    manifest: dict[str, Any],
    *,
    key_prefix: str,
    allowed_groups: set[str] | None = None,
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[str]]:
    errors: list[str] = []
    records: dict[tuple[str, str], dict[str, Any]] = {}
    objects = manifest.get("objects")
    if not isinstance(objects, list):
        return records, ["objects_not_list"]

    for item in objects:
        if not isinstance(item, dict):
            errors.append("object_entry_not_object")
            continue
        if _has_unsafe_manifest_field(item, _UNSAFE_OBJECT_MANIFEST_FIELDS):
            errors.append("object_entry_contains_unsafe_field")
            continue
        group = item.get("group")
        token = item.get("source_path_token")
        digest = item.get("sha256")
        object_key = item.get("object_key")
        byte_count = item.get("bytes")
        if (
            not isinstance(group, str)
            or not _SAFE_GROUP_RE.fullmatch(group)
            or not isinstance(token, str)
            or not _HEX64_RE.fullmatch(token)
            or not isinstance(digest, str)
            or not _HEX64_RE.fullmatch(digest)
            or not isinstance(object_key, str)
            or "://" in object_key
            or not object_key.startswith(f"{key_prefix}/{group}/")
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            errors.append("object_entry_invalid")
            continue
        if allowed_groups is not None and group not in allowed_groups:
            errors.append("object_entry_group_unknown")
            continue
        identity = (group, token)
        if identity in records:
            errors.append("duplicate_object_identity")
            continue
        records[identity] = {
            "group": group,
            "source_path_token": token,
            "sha256": digest,
            "bytes": byte_count,
            "object_key": object_key,
            "restore_priority": str(item.get("restore_priority", "")),
        }
    return records, errors


def _index_job_manifest_records(
    manifest: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    errors: list[str] = []
    jobs: dict[str, dict[str, Any]] = {}
    claims: dict[str, dict[str, Any]] = {}
    job_entries = manifest.get("jobs")
    claim_entries = manifest.get("idempotency_claims")
    if not isinstance(job_entries, list):
        return jobs, claims, ["jobs_not_list"]
    if not isinstance(claim_entries, list):
        errors.append("idempotency_claims_not_list")
        claim_entries = []

    for item in job_entries:
        if not isinstance(item, dict):
            errors.append("job_entry_not_object")
            continue
        if _has_unsafe_manifest_field(item, _UNSAFE_JOB_MANIFEST_FIELDS):
            errors.append("job_entry_contains_unsafe_field")
            continue
        token = item.get("job_token")
        if not isinstance(token, str) or not _HEX64_RE.fullmatch(token):
            errors.append("job_entry_invalid")
            continue
        if token in jobs:
            errors.append("duplicate_job_token")
            continue
        jobs[token] = {
            "job_token": token,
            "kind": str(item.get("kind", "")),
            "status": str(item.get("status", "")),
            "attempts": int(item.get("attempts", 0) or 0),
            "max_attempts": int(item.get("max_attempts", 0) or 0),
            "retry_backoff_s": int(item.get("retry_backoff_s", 0) or 0),
            "ownership_source": str(item.get("ownership_source", "")),
            "has_not_before": bool(item.get("has_not_before")),
            "is_due": bool(item.get("is_due")),
            "is_scheduled": bool(item.get("is_scheduled")),
            "has_deadline": bool(item.get("has_deadline")),
            "deadline_expired": bool(item.get("deadline_expired")),
            "has_lease": bool(item.get("has_lease")),
            "lease_expired": bool(item.get("lease_expired")),
            "has_idempotency_claim": bool(item.get("has_idempotency_claim")),
            "dead_lettered": bool(item.get("dead_lettered")),
        }

    for item in claim_entries:
        if not isinstance(item, dict):
            errors.append("claim_entry_not_object")
            continue
        if _has_unsafe_manifest_field(item, _UNSAFE_JOB_MANIFEST_FIELDS):
            errors.append("claim_entry_contains_unsafe_field")
            continue
        token = item.get("claim_token")
        job_token = item.get("job_token")
        if (
            not isinstance(token, str)
            or not _HEX64_RE.fullmatch(token)
            or not isinstance(job_token, str)
            or not _HEX64_RE.fullmatch(job_token)
        ):
            errors.append("claim_entry_invalid")
            continue
        if token in claims:
            errors.append("duplicate_claim_token")
            continue
        claims[token] = {
            "claim_token": token,
            "job_token": job_token,
            "kind": str(item.get("kind", "")),
        }
    return jobs, claims, errors


def verify_object_storage_migration_manifest(
    manifest: dict[str, Any],
    *,
    project_root: Path = config.PROJECT_ROOT,
    include_runtime_dependencies: bool | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Verify a no-secret object manifest against local runtime files.

    The result deliberately never echoes source paths, filenames, buckets,
    endpoints, credentials, or object contents from the supplied manifest.
    """
    manifest_errors: list[str] = []
    if not isinstance(manifest, dict):
        manifest_errors.append("manifest_not_object")
        manifest = {}
    if (
        manifest.get("mode") == "local_runtime_migration_rehearsal"
        and isinstance(manifest.get("object_storage_manifest"), dict)
    ):
        manifest = manifest["object_storage_manifest"]

    key_prefix = _sanitize_object_key_prefix(str(manifest.get("key_prefix", DEFAULT_OBJECT_KEY_PREFIX)))
    manifest_errors.extend(_object_manifest_schema_errors(manifest))

    include_dependencies = (
        _object_manifest_includes_runtime_dependencies(manifest)
        if include_runtime_dependencies is None
        else include_runtime_dependencies
    )
    current_manifest = collect_object_storage_migration_manifest(
        project_root=project_root,
        include_runtime_dependencies=include_dependencies,
        key_prefix=key_prefix,
        generated_at=generated_at,
    )
    allowed_groups = {
        group["name"]
        for group in current_manifest.get("groups", [])
        if isinstance(group, dict) and isinstance(group.get("name"), str)
    }
    expected_records, expected_errors = _index_object_manifest_objects(
        manifest,
        key_prefix=key_prefix,
        allowed_groups=allowed_groups,
    )
    manifest_errors.extend(expected_errors)
    current_records, current_errors = _index_object_manifest_objects(
        current_manifest,
        key_prefix=key_prefix,
        allowed_groups=allowed_groups,
    )
    manifest_errors.extend(f"current_{error}" for error in current_errors)

    expected_keys = set(expected_records)
    current_keys = set(current_records)
    missing_keys = expected_keys - current_keys
    extra_keys = current_keys - expected_keys
    shared_keys = expected_keys & current_keys
    differences: list[dict[str, Any]] = []

    for identity in sorted(missing_keys):
        expected = expected_records[identity]
        differences.append(
            {
                "group": expected["group"],
                "source_path_token": expected["source_path_token"],
                "status": "missing",
                "expected_bytes": expected["bytes"],
                "current_bytes": None,
                "sha256_match": False,
                "bytes_match": False,
                "object_key_match": False,
            }
        )
    for identity in sorted(extra_keys):
        current = current_records[identity]
        differences.append(
            {
                "group": current["group"],
                "source_path_token": current["source_path_token"],
                "status": "extra",
                "expected_bytes": None,
                "current_bytes": current["bytes"],
                "sha256_match": False,
                "bytes_match": False,
                "object_key_match": False,
            }
        )
    for identity in sorted(shared_keys):
        expected = expected_records[identity]
        current = current_records[identity]
        sha256_match = expected["sha256"] == current["sha256"]
        bytes_match = expected["bytes"] == current["bytes"]
        object_key_match = expected["object_key"] == current["object_key"]
        if sha256_match and bytes_match and object_key_match:
            continue
        differences.append(
            {
                "group": expected["group"],
                "source_path_token": expected["source_path_token"],
                "status": "mismatched",
                "expected_bytes": expected["bytes"],
                "current_bytes": current["bytes"],
                "sha256_match": sha256_match,
                "bytes_match": bytes_match,
                "object_key_match": object_key_match,
            }
        )

    group_names = sorted(
        {group for group, _token in expected_keys}
        | {group for group, _token in current_keys}
        | {
            group.get("name")
            for group in manifest.get("groups", [])
            if (
                isinstance(group, dict)
                and isinstance(group.get("name"), str)
                and group.get("name") in allowed_groups
            )
        }
    )
    group_summaries: list[dict[str, Any]] = []
    for group_name in group_names:
        group_missing = [
            item for item in differences
            if item["group"] == group_name and item["status"] == "missing"
        ]
        group_extra = [
            item for item in differences
            if item["group"] == group_name and item["status"] == "extra"
        ]
        group_mismatched = [
            item for item in differences
            if item["group"] == group_name and item["status"] == "mismatched"
        ]
        group_summaries.append(
            {
                "name": group_name,
                "expected_objects": sum(1 for key in expected_keys if key[0] == group_name),
                "current_objects": sum(1 for key in current_keys if key[0] == group_name),
                "missing_objects": len(group_missing),
                "mismatched_objects": len(group_mismatched),
                "extra_objects": len(group_extra),
                "ok": not group_missing and not group_mismatched and not group_extra,
            }
        )

    missing_count = sum(1 for item in differences if item["status"] == "missing")
    mismatched_count = sum(1 for item in differences if item["status"] == "mismatched")
    extra_count = sum(1 for item in differences if item["status"] == "extra")
    ok = not manifest_errors and not missing_count and not mismatched_count and not extra_count
    return {
        "schema_version": OBJECT_STORAGE_MIGRATION_VERIFY_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now(),
        "mode": "object_storage_migration_manifest_verify",
        "ok": ok,
        "content_exported": False,
        "secrets_exported": False,
        "source_paths_exported": False,
        "filenames_exported": False,
        "bucket_exported": False,
        "external_connectivity_checked": False,
        "hash_algorithm": "sha256",
        "key_prefix": key_prefix,
        "include_runtime_dependencies": include_dependencies,
        "checked_objects": len(expected_records),
        "current_objects": len(current_records),
        "missing_objects": missing_count,
        "mismatched_objects": mismatched_count,
        "extra_objects": extra_count,
        "manifest_errors": sorted(set(manifest_errors)),
        "groups": group_summaries,
        "object_differences": differences,
    }


def verify_job_store_migration_manifest(
    manifest: dict[str, Any],
    *,
    project_root: Path = config.PROJECT_ROOT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Verify a no-secret job-store manifest against local durable job state."""
    manifest_errors: list[str] = []
    if not isinstance(manifest, dict):
        manifest_errors.append("manifest_not_object")
        manifest = {}
    if (
        manifest.get("mode") == "local_runtime_migration_rehearsal"
        and isinstance(manifest.get("job_store_manifest"), dict)
    ):
        manifest = manifest["job_store_manifest"]

    manifest_errors.extend(_job_manifest_schema_errors(manifest))
    expected_jobs, expected_claims, expected_errors = _index_job_manifest_records(manifest)
    manifest_errors.extend(expected_errors)
    reference_generated_at = generated_at or str(manifest.get("generated_at") or "")
    current_manifest = collect_job_store_migration_manifest(
        project_root=project_root,
        generated_at=reference_generated_at or None,
    )
    current_jobs, current_claims, current_errors = _index_job_manifest_records(current_manifest)
    manifest_errors.extend(f"current_{error}" for error in current_errors)
    manifest_errors.extend(f"current_{error}" for error in current_manifest.get("manifest_errors", []))

    expected_job_tokens = set(expected_jobs)
    current_job_tokens = set(current_jobs)
    missing_job_tokens = expected_job_tokens - current_job_tokens
    extra_job_tokens = current_job_tokens - expected_job_tokens
    shared_job_tokens = expected_job_tokens & current_job_tokens
    job_differences: list[dict[str, Any]] = []
    for token in sorted(missing_job_tokens):
        job_differences.append(
            {
                "job_token": token,
                "status": "missing",
                "metadata_match": False,
            }
        )
    for token in sorted(extra_job_tokens):
        job_differences.append(
            {
                "job_token": token,
                "status": "extra",
                "metadata_match": False,
            }
        )
    for token in sorted(shared_job_tokens):
        expected = expected_jobs[token]
        current = current_jobs[token]
        comparable_keys = {
            "kind",
            "status",
            "attempts",
            "max_attempts",
            "retry_backoff_s",
            "ownership_source",
            "has_not_before",
            "is_due",
            "is_scheduled",
            "has_deadline",
            "deadline_expired",
            "has_lease",
            "lease_expired",
            "has_idempotency_claim",
            "dead_lettered",
        }
        changed_keys = sorted(
            key for key in comparable_keys if expected.get(key) != current.get(key)
        )
        if not changed_keys:
            continue
        job_differences.append(
            {
                "job_token": token,
                "status": "mismatched",
                "metadata_match": False,
                "changed_fields": changed_keys,
            }
        )

    expected_claim_tokens = set(expected_claims)
    current_claim_tokens = set(current_claims)
    missing_claim_tokens = expected_claim_tokens - current_claim_tokens
    extra_claim_tokens = current_claim_tokens - expected_claim_tokens
    shared_claim_tokens = expected_claim_tokens & current_claim_tokens
    claim_differences = [
        {"claim_token": token, "status": "missing", "job_token": expected_claims[token]["job_token"]}
        for token in sorted(missing_claim_tokens)
    ] + [
        {"claim_token": token, "status": "extra", "job_token": current_claims[token]["job_token"]}
        for token in sorted(extra_claim_tokens)
    ]
    for token in sorted(shared_claim_tokens):
        expected = expected_claims[token]
        current = current_claims[token]
        changed_keys = sorted(
            key for key in ("job_token", "kind") if expected.get(key) != current.get(key)
        )
        if not changed_keys:
            continue
        claim_differences.append(
            {
                "claim_token": token,
                "status": "mismatched",
                "job_token": expected.get("job_token", ""),
                "metadata_match": False,
                "changed_fields": changed_keys,
            }
        )

    missing_jobs = sum(1 for item in job_differences if item["status"] == "missing")
    mismatched_jobs = sum(1 for item in job_differences if item["status"] == "mismatched")
    extra_jobs = sum(1 for item in job_differences if item["status"] == "extra")
    missing_claims = len(missing_claim_tokens)
    extra_claims = len(extra_claim_tokens)
    mismatched_claims = sum(
        1 for item in claim_differences if item["status"] == "mismatched"
    )
    ok = (
        not manifest_errors
        and not missing_jobs
        and not mismatched_jobs
        and not extra_jobs
        and not missing_claims
        and not mismatched_claims
        and not extra_claims
    )
    return {
        "schema_version": JOB_STORE_MIGRATION_VERIFY_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now(),
        "mode": "job_store_migration_manifest_verify",
        "ok": ok,
        "content_exported": False,
        "secrets_exported": False,
        "payload_exported": False,
        "owner_ids_exported": False,
        "request_ids_exported": False,
        "worker_ids_exported": False,
        "idempotency_keys_exported": False,
        "external_connectivity_checked": False,
        "hash_algorithm": "sha256",
        "expected_jobs": len(expected_jobs),
        "current_jobs": len(current_jobs),
        "missing_jobs": missing_jobs,
        "mismatched_jobs": mismatched_jobs,
        "extra_jobs": extra_jobs,
        "expected_idempotency_claims": len(expected_claims),
        "current_idempotency_claims": len(current_claims),
        "missing_idempotency_claims": missing_claims,
        "mismatched_idempotency_claims": mismatched_claims,
        "extra_idempotency_claims": extra_claims,
        "manifest_errors": sorted(set(manifest_errors)),
        "job_differences": job_differences,
        "idempotency_claim_differences": claim_differences,
    }


def run_storage_migration_rehearsal(
    *,
    project_root: Path = config.PROJECT_ROOT,
    staging_root: Path,
    overwrite_staging: bool = False,
    include_runtime_dependencies: bool = False,
    include_object_manifest: bool = False,
    include_job_store_manifest: bool = False,
    object_key_prefix: str = DEFAULT_OBJECT_KEY_PREFIX,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Copy local runtime state into staging and verify it with no-secret checks."""
    source_root = project_root.resolve()
    target_root = staging_root.resolve()
    blockers: list[str] = []
    copy_groups: list[dict[str, Any]] = []
    staging_prepared = False
    staging_created = False

    if source_root == target_root or _is_relative_to(target_root, source_root):
        blockers.append("staging_root_inside_project")
    elif _is_relative_to(source_root, target_root):
        blockers.append("staging_root_contains_project")
    elif _directory_has_entries(target_root) and not overwrite_staging:
        blockers.append("staging_root_not_empty")
    else:
        if target_root.exists() and overwrite_staging:
            _empty_directory(target_root)
        elif not target_root.exists():
            target_root.mkdir(parents=True)
            staging_created = True
        staging_prepared = True

    runtime_groups = runtime_groups_for_root(source_root)
    if not include_runtime_dependencies:
        runtime_groups = tuple(
            group for group in runtime_groups if group.restore_priority != "runtime_dependency"
        )
    source_manifest = collect_runtime_backup_manifest(
        project_root=source_root,
        groups=runtime_groups,
        generated_at=generated_at,
    )
    source_preflight = collect_platform_migration_preflight(
        project_root=source_root,
        generated_at=generated_at,
    )

    if staging_prepared:
        for group in source_manifest.get("groups", []):
            relative_group = Path(str(group.get("path", "")))
            copy_groups.append(
                _copy_runtime_group(
                    name=str(group.get("name", "")),
                    source_path=source_root / relative_group,
                    target_path=target_root / relative_group,
                    restore_priority=str(group.get("restore_priority", "")),
                    include_runtime_dependencies=include_runtime_dependencies,
                )
            )

    copy_errors = [
        error
        for group in copy_groups
        for error in group.get("errors", [])
    ]
    blockers.extend(copy_errors)
    restore_check = collect_runtime_restore_check(
        source_manifest,
        project_root=target_root,
        generated_at=generated_at,
    ) if staging_prepared else {}
    staged_schema = storage_schema_status_for_root(target_root) if staging_prepared else {}
    object_manifest = (
        collect_object_storage_migration_manifest(
            project_root=target_root,
            include_runtime_dependencies=include_runtime_dependencies,
            key_prefix=object_key_prefix,
            generated_at=generated_at,
        )
        if staging_prepared and include_object_manifest
        else {}
    )
    job_store_manifest = (
        collect_job_store_migration_manifest(
            project_root=target_root,
            generated_at=generated_at,
        )
        if staging_prepared and include_job_store_manifest
        else {}
    )
    if source_preflight.get("preflight_ok") is not True:
        blockers.append("source_preflight_failed")
    if staging_prepared and not restore_check.get("ok"):
        blockers.append("staged_restore_check_failed")
    if staging_prepared and not staged_schema.get("ok"):
        blockers.append("staged_storage_schema_drift")
    if (
        staging_prepared
        and include_job_store_manifest
        and not job_store_manifest.get("ok")
    ):
        blockers.append("job_store_manifest_failed")

    copied_files = sum(int(group.get("copied_files", 0) or 0) for group in copy_groups)
    copied_bytes = sum(int(group.get("copied_bytes", 0) or 0) for group in copy_groups)
    skipped_symlinks = sum(int(group.get("skipped_symlinks", 0) or 0) for group in copy_groups)
    rehearsal_ok = not blockers and staging_prepared
    return {
        "schema_version": STORAGE_MIGRATION_REHEARSAL_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now(),
        "mode": "local_runtime_migration_rehearsal",
        "rehearsal_ok": rehearsal_ok,
        "activation_enabled": False,
        "external_connectivity_checked": False,
        "content_copied_to_staging": bool(staging_prepared),
        "content_exported_in_report": False,
        "secrets_copied": False,
        "secrets_exported": False,
        "staging_root_created": staging_created,
        "staging_root_overwritten": bool(overwrite_staging and staging_prepared),
        "staging_root_retained": True,
        "include_runtime_dependencies": include_runtime_dependencies,
        "blockers": sorted(set(blockers)),
        "summary": {
            "source_preflight_ok": bool(source_preflight.get("preflight_ok")),
            "source_activation_ready": bool(source_preflight.get("activation_ready")),
            "copy_group_count": len(copy_groups),
            "copied_files": copied_files,
            "copied_bytes": copied_bytes,
            "skipped_symlinks": skipped_symlinks,
            "restore_check_ok": bool(restore_check.get("ok")),
            "staged_storage_schema_ok": bool(staged_schema.get("ok")),
            "object_manifest_ready": bool(object_manifest.get("mode")),
            "object_manifest_objects": int(object_manifest.get("object_count", 0) or 0),
            "object_manifest_unique_objects": int(
                object_manifest.get("unique_object_count", 0) or 0
            ),
            "job_store_manifest_ready": bool(job_store_manifest.get("mode")),
            "job_store_manifest_jobs": int(job_store_manifest.get("job_count", 0) or 0),
            "job_store_manifest_claims": int(
                job_store_manifest.get("idempotency_claim_count", 0) or 0
            ),
        },
        "source_preflight": {
            "preflight_ok": bool(source_preflight.get("preflight_ok")),
            "activation_ready": bool(source_preflight.get("activation_ready")),
            "local_blockers": list(source_preflight.get("blockers", {}).get("local_preflight", [])),
            "activation_blockers": list(source_preflight.get("blockers", {}).get("activation", [])),
        },
        "source_manifest": _manifest_summary(source_manifest),
        "copy": {
            "groups": copy_groups,
            "copied_files": copied_files,
            "copied_bytes": copied_bytes,
            "skipped_symlinks": skipped_symlinks,
        },
        "staged_restore_check": _restore_summary(restore_check),
        "staged_storage_schema": _schema_summary(staged_schema),
        "object_storage_manifest": object_manifest,
        "object_storage_manifest_summary": _object_manifest_summary(object_manifest)
        if object_manifest
        else {},
        "job_store_manifest": job_store_manifest,
        "job_store_manifest_summary": _job_store_manifest_summary(job_store_manifest)
        if job_store_manifest
        else {},
    }


def storage_migration_rehearsal_public_status(status: dict[str, Any]) -> dict[str, Any]:
    """Return the no-secret public projection for API/UI rehearsal surfaces."""
    return {
        "schema_version": status.get("schema_version"),
        "generated_at": status.get("generated_at", ""),
        "mode": status.get("mode", ""),
        "rehearsal_ok": bool(status.get("rehearsal_ok")),
        "activation_enabled": bool(status.get("activation_enabled")),
        "external_connectivity_checked": bool(
            status.get("external_connectivity_checked")
        ),
        "content_copied_to_staging": bool(status.get("content_copied_to_staging")),
        "content_exported_in_report": bool(status.get("content_exported_in_report")),
        "secrets_copied": bool(status.get("secrets_copied")),
        "secrets_exported": bool(status.get("secrets_exported")),
        "paths_exported": False,
        "raw_manifests_included": False,
        "staging_root_created": bool(status.get("staging_root_created")),
        "staging_root_overwritten": bool(status.get("staging_root_overwritten")),
        "staging_root_retained": bool(status.get("staging_root_retained")),
        "include_runtime_dependencies": bool(status.get("include_runtime_dependencies")),
        "blockers": list(status.get("blockers", [])),
        "summary": dict(status.get("summary", {}) or {}),
        "source_preflight": dict(status.get("source_preflight", {}) or {}),
        "source_manifest": dict(status.get("source_manifest", {}) or {}),
        "copy": dict(status.get("copy", {}) or {}),
        "staged_restore_check": dict(status.get("staged_restore_check", {}) or {}),
        "staged_storage_schema": dict(status.get("staged_storage_schema", {}) or {}),
        "object_storage_manifest_summary": dict(
            status.get("object_storage_manifest_summary", {}) or {}
        ),
        "job_store_manifest_summary": dict(
            status.get("job_store_manifest_summary", {}) or {}
        ),
    }


def collect_platform_migration_rehearsal(
    *,
    project_root: Path = config.PROJECT_ROOT,
    include_runtime_dependencies: bool = False,
    include_object_manifest: bool = True,
    include_job_store_manifest: bool = True,
    object_key_prefix: str = DEFAULT_OBJECT_KEY_PREFIX,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Run a temporary local migration rehearsal and return a public projection."""
    with tempfile.TemporaryDirectory(prefix="fluxmind-migration-rehearsal-") as tmp_dir:
        status = run_storage_migration_rehearsal(
            project_root=project_root,
            staging_root=Path(tmp_dir),
            overwrite_staging=False,
            include_runtime_dependencies=include_runtime_dependencies,
            include_object_manifest=include_object_manifest,
            include_job_store_manifest=include_job_store_manifest,
            object_key_prefix=object_key_prefix,
            generated_at=generated_at,
        )
        status["staging_root_retained"] = False
        return storage_migration_rehearsal_public_status(status)


def _format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def format_storage_migration_rehearsal_markdown(status: dict[str, Any]) -> str:
    """Render a no-secret local migration rehearsal report."""
    summary = status.get("summary", {})
    object_summary = status.get("object_storage_manifest_summary", {})
    job_summary = status.get("job_store_manifest_summary", {})
    lines = [
        "# FluxMind Runtime Migration Rehearsal",
        "",
        "No runtime contents, job payloads, external URLs, bucket names, queue names, credentials, or secrets are exported in this report.",
        "",
        f"- Generated at: {status.get('generated_at', '')}",
        f"- Mode: {status.get('mode', '')}",
        f"- Rehearsal OK: {_format_bool(status.get('rehearsal_ok', False))}",
        f"- Activation enabled: {_format_bool(status.get('activation_enabled', False))}",
        f"- External connectivity checked: {_format_bool(status.get('external_connectivity_checked', False))}",
        f"- Content copied to staging: {_format_bool(status.get('content_copied_to_staging', False))}",
        f"- Content exported in report: {_format_bool(status.get('content_exported_in_report', False))}",
        f"- Secrets copied: {_format_bool(status.get('secrets_copied', False))}",
        f"- Secrets exported: {_format_bool(status.get('secrets_exported', False))}",
        f"- Include runtime dependencies: {_format_bool(status.get('include_runtime_dependencies', False))}",
        "",
        "## Summary",
        "",
        f"- Source preflight OK: {_format_bool(summary.get('source_preflight_ok', False))}",
        f"- Source activation ready: {_format_bool(summary.get('source_activation_ready', False))}",
        f"- Copy groups: {summary.get('copy_group_count', 0)}",
        f"- Copied files: {summary.get('copied_files', 0)}",
        f"- Copied bytes: {summary.get('copied_bytes', 0)}",
        f"- Skipped symlinks: {summary.get('skipped_symlinks', 0)}",
        f"- Restore check OK: {_format_bool(summary.get('restore_check_ok', False))}",
        f"- Staged storage schema OK: {_format_bool(summary.get('staged_storage_schema_ok', False))}",
        f"- Object manifest ready: {_format_bool(summary.get('object_manifest_ready', False))}",
        f"- Object manifest objects: {summary.get('object_manifest_objects', 0)}",
        f"- Object manifest unique objects: {summary.get('object_manifest_unique_objects', 0)}",
        f"- Job-store manifest ready: {_format_bool(summary.get('job_store_manifest_ready', False))}",
        f"- Job-store manifest jobs: {summary.get('job_store_manifest_jobs', 0)}",
        f"- Job-store manifest claims: {summary.get('job_store_manifest_claims', 0)}",
        "",
        "## Blockers",
        "",
        f"- {', '.join(status.get('blockers', [])) or 'none'}",
        "",
        "## Copied Groups",
        "",
    ]
    for group in status.get("copy", {}).get("groups", []):
        lines.append(
            f"- {group.get('name', '')}: status={group.get('status', '')}, "
            f"priority={group.get('restore_priority', '')}, "
            f"source_exists={_format_bool(group.get('source_exists', False))}, "
            f"files={group.get('copied_files', 0)}, "
            f"bytes={group.get('copied_bytes', 0)}, "
            f"skipped_symlinks={group.get('skipped_symlinks', 0)}, "
            f"errors={','.join(group.get('errors', [])) or 'none'}"
        )
    if object_summary:
        lines.extend(
            [
                "",
                "## Object Storage Migration Manifest",
                "",
                "Object keys, hashes, and byte counts are available in JSON output. Source paths, filenames, buckets, endpoints, credentials, and contents are not exported.",
                "",
                f"- Mode: {object_summary.get('mode', '')}",
                f"- Content exported: {_format_bool(object_summary.get('content_exported', False))}",
                f"- Secrets exported: {_format_bool(object_summary.get('secrets_exported', False))}",
                f"- Source paths exported: {_format_bool(object_summary.get('source_paths_exported', False))}",
                f"- Filenames exported: {_format_bool(object_summary.get('filenames_exported', False))}",
                f"- Bucket exported: {_format_bool(object_summary.get('bucket_exported', False))}",
                f"- Object count: {object_summary.get('object_count', 0)}",
                f"- Unique object count: {object_summary.get('unique_object_count', 0)}",
                f"- Duplicate content references: {object_summary.get('duplicate_content_references', 0)}",
                f"- Total bytes: {object_summary.get('total_bytes', 0)}",
            ]
        )
    if job_summary:
        lines.extend(
            [
                "",
                "## Job Store Migration Manifest",
                "",
                "Job and idempotency claim tokens are available in JSON output. Job payloads, owner IDs, request IDs, worker IDs, idempotency keys, logs, artifacts, and execution output are not exported.",
                "",
                f"- Mode: {job_summary.get('mode', '')}",
                f"- OK: {_format_bool(job_summary.get('ok', False))}",
                f"- Content exported: {_format_bool(job_summary.get('content_exported', False))}",
                f"- Secrets exported: {_format_bool(job_summary.get('secrets_exported', False))}",
                f"- Payload exported: {_format_bool(job_summary.get('payload_exported', False))}",
                f"- Owner IDs exported: {_format_bool(job_summary.get('owner_ids_exported', False))}",
                f"- Request IDs exported: {_format_bool(job_summary.get('request_ids_exported', False))}",
                f"- Worker IDs exported: {_format_bool(job_summary.get('worker_ids_exported', False))}",
                f"- Idempotency keys exported: {_format_bool(job_summary.get('idempotency_keys_exported', False))}",
                f"- Job count: {job_summary.get('job_count', 0)}",
                f"- Idempotency claim count: {job_summary.get('claim_count', 0)}",
                f"- Manifest errors: {', '.join(job_summary.get('manifest_errors', [])) or 'none'}",
            ]
        )
    return "\n".join(lines)


def format_object_storage_migration_manifest_markdown(manifest: dict[str, Any]) -> str:
    """Render an object-storage migration manifest summary as no-secret Markdown."""
    lines = [
        "# FluxMind Object Storage Migration Manifest",
        "",
        "No runtime contents, source paths, filenames, buckets, endpoints, credentials, or secrets are exported.",
        "",
        f"- Generated at: {manifest.get('generated_at', '')}",
        f"- Mode: {manifest.get('mode', '')}",
        f"- Content exported: {_format_bool(manifest.get('content_exported', False))}",
        f"- Secrets exported: {_format_bool(manifest.get('secrets_exported', False))}",
        f"- Source paths exported: {_format_bool(manifest.get('source_paths_exported', False))}",
        f"- Filenames exported: {_format_bool(manifest.get('filenames_exported', False))}",
        f"- Bucket exported: {_format_bool(manifest.get('bucket_exported', False))}",
        f"- External connectivity checked: {_format_bool(manifest.get('external_connectivity_checked', False))}",
        f"- Hash algorithm: {manifest.get('hash_algorithm', '')}",
        f"- Object key strategy: {manifest.get('object_key_strategy', '')}",
        f"- Object count: {manifest.get('object_count', 0)}",
        f"- Unique object count: {manifest.get('unique_object_count', 0)}",
        f"- Duplicate content references: {manifest.get('duplicate_content_references', 0)}",
        f"- Total bytes: {manifest.get('total_bytes', 0)}",
        "",
        "## Runtime Groups",
        "",
    ]
    for group in manifest.get("groups", []):
        lines.append(
            f"- {group.get('name', '')}: priority={group.get('restore_priority', '')}, "
            f"source_exists={_format_bool(group.get('source_exists', False))}, "
            f"objects={group.get('object_count', 0)}, bytes={group.get('bytes', 0)}"
        )
    return "\n".join(lines)


def format_object_storage_migration_verify_markdown(status: dict[str, Any]) -> str:
    """Render a no-secret object-manifest verification report."""
    lines = [
        "# FluxMind Object Storage Migration Manifest Verification",
        "",
        "No runtime contents, source paths, filenames, buckets, endpoints, credentials, or secrets are exported.",
        "",
        f"- Generated at: {status.get('generated_at', '')}",
        f"- Mode: {status.get('mode', '')}",
        f"- Verification OK: {_format_bool(status.get('ok', False))}",
        f"- Content exported: {_format_bool(status.get('content_exported', False))}",
        f"- Secrets exported: {_format_bool(status.get('secrets_exported', False))}",
        f"- Source paths exported: {_format_bool(status.get('source_paths_exported', False))}",
        f"- Filenames exported: {_format_bool(status.get('filenames_exported', False))}",
        f"- Bucket exported: {_format_bool(status.get('bucket_exported', False))}",
        f"- External connectivity checked: {_format_bool(status.get('external_connectivity_checked', False))}",
        f"- Include runtime dependencies: {_format_bool(status.get('include_runtime_dependencies', False))}",
        f"- Checked objects: {status.get('checked_objects', 0)}",
        f"- Current objects: {status.get('current_objects', 0)}",
        f"- Missing objects: {status.get('missing_objects', 0)}",
        f"- Mismatched objects: {status.get('mismatched_objects', 0)}",
        f"- Extra objects: {status.get('extra_objects', 0)}",
        "",
        "## Manifest Errors",
        "",
        f"- {', '.join(status.get('manifest_errors', [])) or 'none'}",
        "",
        "## Runtime Groups",
        "",
    ]
    for group in status.get("groups", []):
        lines.append(
            f"- {group.get('name', '')}: ok={_format_bool(group.get('ok', False))}, "
            f"expected={group.get('expected_objects', 0)}, "
            f"current={group.get('current_objects', 0)}, "
            f"missing={group.get('missing_objects', 0)}, "
            f"mismatched={group.get('mismatched_objects', 0)}, "
            f"extra={group.get('extra_objects', 0)}"
        )
    differences = status.get("object_differences", [])
    lines.extend(["", "## Object Differences", ""])
    if not differences:
        lines.append("- none")
    for item in differences:
        lines.append(
            f"- {item.get('group', '')}: status={item.get('status', '')}, "
            f"source_path_token={item.get('source_path_token', '')}, "
            f"sha256_match={_format_bool(item.get('sha256_match', False))}, "
            f"bytes_match={_format_bool(item.get('bytes_match', False))}, "
            f"object_key_match={_format_bool(item.get('object_key_match', False))}"
        )
    return "\n".join(lines)


def format_job_store_migration_manifest_markdown(manifest: dict[str, Any]) -> str:
    """Render a job-store migration manifest summary as no-secret Markdown."""
    lines = [
        "# FluxMind Job Store Migration Manifest",
        "",
        "No job payloads, owner IDs, request IDs, worker IDs, idempotency keys, logs, artifacts, stdout/stderr, credentials, or secrets are exported.",
        "",
        f"- Generated at: {manifest.get('generated_at', '')}",
        f"- Mode: {manifest.get('mode', '')}",
        f"- OK: {_format_bool(manifest.get('ok', False))}",
        f"- Content exported: {_format_bool(manifest.get('content_exported', False))}",
        f"- Secrets exported: {_format_bool(manifest.get('secrets_exported', False))}",
        f"- Payload exported: {_format_bool(manifest.get('payload_exported', False))}",
        f"- Owner IDs exported: {_format_bool(manifest.get('owner_ids_exported', False))}",
        f"- Request IDs exported: {_format_bool(manifest.get('request_ids_exported', False))}",
        f"- Worker IDs exported: {_format_bool(manifest.get('worker_ids_exported', False))}",
        f"- Idempotency keys exported: {_format_bool(manifest.get('idempotency_keys_exported', False))}",
        f"- External connectivity checked: {_format_bool(manifest.get('external_connectivity_checked', False))}",
        f"- Job count: {manifest.get('job_count', 0)}",
        f"- Idempotency claim count: {manifest.get('idempotency_claim_count', 0)}",
        "",
        "## Queue Summary",
        "",
    ]
    for key, value in sorted((manifest.get("queue_summary") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Manifest Errors",
            "",
            f"- {', '.join(manifest.get('manifest_errors', [])) or 'none'}",
        ]
    )
    return "\n".join(lines)


def format_job_store_migration_verify_markdown(status: dict[str, Any]) -> str:
    """Render a no-secret job-store manifest verification report."""
    lines = [
        "# FluxMind Job Store Migration Manifest Verification",
        "",
        "No job payloads, owner IDs, request IDs, worker IDs, idempotency keys, logs, artifacts, stdout/stderr, credentials, or secrets are exported.",
        "",
        f"- Generated at: {status.get('generated_at', '')}",
        f"- Mode: {status.get('mode', '')}",
        f"- Verification OK: {_format_bool(status.get('ok', False))}",
        f"- Content exported: {_format_bool(status.get('content_exported', False))}",
        f"- Secrets exported: {_format_bool(status.get('secrets_exported', False))}",
        f"- Payload exported: {_format_bool(status.get('payload_exported', False))}",
        f"- Owner IDs exported: {_format_bool(status.get('owner_ids_exported', False))}",
        f"- Request IDs exported: {_format_bool(status.get('request_ids_exported', False))}",
        f"- Worker IDs exported: {_format_bool(status.get('worker_ids_exported', False))}",
        f"- Idempotency keys exported: {_format_bool(status.get('idempotency_keys_exported', False))}",
        f"- Expected jobs: {status.get('expected_jobs', 0)}",
        f"- Current jobs: {status.get('current_jobs', 0)}",
        f"- Missing jobs: {status.get('missing_jobs', 0)}",
        f"- Mismatched jobs: {status.get('mismatched_jobs', 0)}",
        f"- Extra jobs: {status.get('extra_jobs', 0)}",
        f"- Expected idempotency claims: {status.get('expected_idempotency_claims', 0)}",
        f"- Current idempotency claims: {status.get('current_idempotency_claims', 0)}",
        f"- Missing idempotency claims: {status.get('missing_idempotency_claims', 0)}",
        f"- Mismatched idempotency claims: {status.get('mismatched_idempotency_claims', 0)}",
        f"- Extra idempotency claims: {status.get('extra_idempotency_claims', 0)}",
        "",
        "## Manifest Errors",
        "",
        f"- {', '.join(status.get('manifest_errors', [])) or 'none'}",
    ]
    return "\n".join(lines)
