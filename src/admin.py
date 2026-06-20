"""Local admin status for no-key platform foundations."""

from __future__ import annotations

import os
import re
import stat
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

from src.config import (
    ACTIVE_PAPERS_FILE,
    API_KEY_REGISTRY_FILE,
    API_ACCESS_AUDIT_ENABLED,
    API_RATE_LIMIT_ENABLED,
    API_RATE_LIMIT_MAX_REQUESTS,
    API_RATE_LIMIT_WINDOW_S,
    ARTIFACTS_DIR,
    CODE_EXECUTION_ALLOWED_IMPORTS,
    CODE_EXECUTION_ALERT_DURATION_MS,
    CODE_EXECUTION_ALERT_FAILURE_RATE,
    CODE_EXECUTION_ALERT_MIN_EVENTS,
    CODE_EXECUTION_BACKEND,
    CODE_EXECUTION_MAX_ARTIFACT_BYTES,
    CODE_EXECUTION_MAX_ARTIFACT_CANDIDATES,
    CODE_EXECUTION_MAX_ARTIFACT_TOTAL_BYTES,
    CODE_EXECUTION_MAX_ARTIFACTS,
    CODE_EXECUTION_MAX_STDERR_BYTES,
    CODE_EXECUTION_MAX_STDOUT_BYTES,
    CODE_EXECUTION_POLICY,
    CHUNK_METADATA_DB_FILE,
    CORPUS_METADATA_DB_FILE,
    CORPUS_METADATA_FILE,
    CORPUS_PROFILES_FILE,
    DATABASE_URL,
    DISTRIBUTED_JOB_QUEUE_NAME,
    DISTRIBUTED_JOB_STORE_BACKEND,
    DISTRIBUTED_JOB_STORE_URL,
    DOCKER_EXECUTION_IMAGE,
    EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
    JOBS_DB_FILE,
    JOBS_DIR,
    JOBS_FILE,
    LLM_BASE_URL,
    LLM_MODEL,
    JOB_ALERT_EXPIRED_MIN_EVENTS,
    JOB_ALERT_FAILED_MIN_EVENTS,
    METADATA_DIR,
    METADATA_STORAGE_BACKEND,
    OBJECT_STORAGE_BACKEND,
    OBJECT_STORAGE_BUCKET,
    OBJECT_STORAGE_ENDPOINT,
    OBJECT_STORAGE_REGION,
    PAPERS_UPLOADS_DIR,
    PRODUCT_REGISTRY_FILE,
    PROJECT_ROOT,
    PROVIDER_FAILURE_ALERT_MIN_EVENTS,
    PROVIDER_FAILURE_ALERT_RATE,
    QUERY_COST_COMPLETION_USD_PER_1M,
    QUERY_COST_PROMPT_USD_PER_1M,
    QUERY_COST_PROVIDER,
    QUERY_ALERT_DURATION_MS,
    QUERY_ALERT_MIN_EVENTS,
    RETRIEVAL_TRACE_ALERT_CITATION_FAILURE_RATE,
    RETRIEVAL_TRACE_ALERT_EMPTY_RATE,
    RETRIEVAL_TRACE_ALERT_MIN_EVENTS,
    RETRIEVAL_TRACE_ALERT_SOURCE_PAGE_INCOMPLETE_RATE,
    RETENTION_DELETE_ENABLED,
    RERANKER_MODEL,
    RUNTIME_EVENTS_FILE,
    SHARE_LINK_TOKEN_STORE_FILE,
    UPLOAD_SCAN_BLOCK_ACTIVE_CONTENT,
    UPLOAD_SCAN_ENABLED,
    UPLOAD_SCAN_MAX_PAGES,
    UPLOAD_SCAN_REJECT_ENCRYPTED,
)
from src.artifacts import LocalArtifactRegistry
from src.costs import summarize_query_cost
from src.jobs import LocalJobStore
from src.metadata import ChunkMetadataStore, CorpusMetadataStore, CorpusProfileStore
from src.product_readiness import collect_product_readiness
from src.provider_readiness import collect_provider_readiness
from src.providers import docker_execution_status
from src.runtime import append_runtime_event, list_runtime_events, runtime_event_to_safe_dict
from src.storage_schema import storage_schema_status

ADMIN_CHECK_LATEST_METADATA_KEYS = {
    "active_key_count",
    "activation_ready",
    "activation_step_count",
    "blocker_count",
    "blocked_recent",
    "check",
    "compared_field_count",
    "copied_files",
    "diff_count",
    "docker_available",
    "external_activation_ready",
    "failed_check_count",
    "full_activation_ready",
    "full_activation_blocker_count",
    "job_store_manifest_ready",
    "local_foundation_ready",
    "object_manifest_ready",
    "ok",
    "ok_recent",
    "operation_count",
    "protected_auth_header_operation_count",
    "protected_operation_count",
    "required_operation_missing_count",
    "response_missing_operation_count",
    "restore_check_ok",
    "route_count",
    "snapshot_raw_schema_included",
    "snapshot_shape_valid",
    "source_preflight_ok",
    "status_code",
    "undocumented_operation_count",
    "workspace_count",
}
ADMIN_CHECK_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_:-]{0,79}$")


def refresh_paper_metadata():
    from src.ingestion import refresh_paper_metadata as _refresh_paper_metadata

    return _refresh_paper_metadata()


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


@dataclass(frozen=True)
class RuntimeDirectoryStatus:
    name: str
    path: str
    exists: bool
    writable: bool
    bytes: int


@dataclass(frozen=True)
class AdminStatus:
    runtime_dirs: list[RuntimeDirectoryStatus]
    jobs: dict[str, Any]
    corpus: dict[str, Any]
    artifacts: dict[str, Any]
    storage: dict[str, Any]
    storage_schemas: dict[str, Any]
    platform_readiness: dict[str, Any]
    provider_failures: dict[str, Any]
    query_usage: dict[str, Any]
    retrieval_traces: dict[str, Any]
    code_execution: dict[str, Any]
    api_access: dict[str, Any]
    admin_checks: dict[str, Any]
    upload_scans: dict[str, Any]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _format_counts(counts: dict[str, Any]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _format_report_codes(codes: list[str]) -> str:
    if not codes:
        return "none"
    return ",".join(str(code).replace("api_key", "access_key") for code in codes)


def _format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _query_cost_token_value(event: Any, provider_key: str, estimated_key: str) -> int:
    provider_value = event.metadata.get(provider_key)
    if event.metadata.get("usage_source") == "provider" and provider_value is not None:
        return int(provider_value or 0)
    return int(event.metadata.get(estimated_key, 0) or 0)


def _event_int_metadata(event: Any, key: str) -> int:
    try:
        return int(event.metadata.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _admin_check_label(value: Any, *, fallback: str = "unknown") -> str:
    text = str(value or "").strip().casefold()
    if not text:
        return fallback
    if ADMIN_CHECK_LABEL_RE.fullmatch(text):
        return text
    return "invalid"


def _runtime_event_admin_dict(event: Any) -> dict[str, Any]:
    """Project runtime events for admin status without correlation identifiers."""
    return runtime_event_to_safe_dict(event, include_request_id=False)


def _admin_check_event_admin_dict(event: Any) -> dict[str, Any]:
    """Project admin check events through a fixed metadata summary."""
    projected = _runtime_event_admin_dict(event)
    projected["code"] = _admin_check_label(projected.get("code"))
    metadata = projected.get("metadata", {}) or {}
    projected["metadata"] = {
        key: value
        for key, value in metadata.items()
        if key in ADMIN_CHECK_LATEST_METADATA_KEYS
    }
    if "check" in projected["metadata"]:
        projected["metadata"]["check"] = _admin_check_label(projected["metadata"]["check"])
    return projected


def _dict_int(data: dict[str, Any], key: str) -> int:
    try:
        return int(data.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _code_execution_alert(
    *,
    code: str,
    severity: str,
    message: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "metadata": metadata,
    }


def summarize_code_execution_alerts(
    *,
    total_recent: int,
    failed_recent: int,
    failure_rate: float,
    max_duration_ms: int,
    policy_violations: int,
    output_truncations: int,
    artifact_collection_truncations: int,
    min_events: int,
    failure_rate_threshold: float,
    duration_ms_threshold: int,
) -> list[dict[str, Any]]:
    """Return no-secret advisory alerts for recent local code execution."""
    alerts: list[dict[str, Any]] = []
    if total_recent >= min_events and failure_rate >= failure_rate_threshold:
        alerts.append(
            _code_execution_alert(
                code="code_execution_failure_rate_high",
                severity="warning",
                message="Recent code execution failure rate is above the configured threshold.",
                metadata={
                    "total_recent": total_recent,
                    "failed_recent": failed_recent,
                    "failure_rate": f"{failure_rate:.2f}",
                    "threshold": f"{failure_rate_threshold:.2f}",
                    "min_events": min_events,
                },
            )
        )
    if max_duration_ms >= duration_ms_threshold:
        alerts.append(
            _code_execution_alert(
                code="code_execution_duration_high",
                severity="warning",
                message="A recent code execution duration exceeded the configured threshold.",
                metadata={
                    "max_duration_ms": max_duration_ms,
                    "threshold_ms": duration_ms_threshold,
                },
            )
        )
    if policy_violations:
        alerts.append(
            _code_execution_alert(
                code="code_execution_policy_violations_recent",
                severity="warning",
                message="Recent code execution requests were blocked by policy.",
                metadata={"policy_violations": policy_violations},
            )
        )
    if output_truncations:
        alerts.append(
            _code_execution_alert(
                code="code_execution_output_truncated_recent",
                severity="info",
                message="Recent code execution output hit stdout/stderr capture limits.",
                metadata={"output_truncations": output_truncations},
            )
        )
    if artifact_collection_truncations:
        alerts.append(
            _code_execution_alert(
                code="code_execution_artifacts_truncated_recent",
                severity="info",
                message="Recent code execution artifact export hit collection limits.",
                metadata={
                    "artifact_collection_truncations": artifact_collection_truncations,
                },
            )
        )
    return alerts


def _admin_alert(
    *,
    code: str,
    severity: str,
    message: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "metadata": metadata,
    }


def summarize_query_usage_alerts(
    *,
    total_recent: int,
    avg_duration_ms: int,
    max_duration_ms: int,
    min_events: int,
    duration_ms_threshold: int,
) -> list[dict[str, Any]]:
    """Return no-secret advisory alerts for recent query latency."""
    alerts: list[dict[str, Any]] = []
    base_metadata = {
        "total_recent": total_recent,
        "avg_duration_ms": avg_duration_ms,
        "max_duration_ms": max_duration_ms,
        "threshold_ms": duration_ms_threshold,
        "min_events": min_events,
    }
    if total_recent >= min_events and avg_duration_ms >= duration_ms_threshold:
        alerts.append(
            _admin_alert(
                code="query_duration_average_high",
                severity="warning",
                message="Recent average query duration exceeded the configured threshold.",
                metadata=base_metadata,
            )
        )
    elif max_duration_ms >= duration_ms_threshold:
        alerts.append(
            _admin_alert(
                code="query_duration_high",
                severity="info",
                message="A recent query duration exceeded the configured threshold.",
                metadata=base_metadata,
            )
        )
    return alerts


def summarize_retrieval_trace_alerts(
    *,
    total_recent: int,
    empty_recent: int,
    empty_rate: float,
    source_page_incomplete_recent: int,
    source_page_incomplete_rate: float,
    citation_checked_recent: int,
    citation_failed_recent: int,
    citation_failure_rate: float,
    min_events: int,
    empty_rate_threshold: float,
    source_page_incomplete_rate_threshold: float,
    citation_failure_rate_threshold: float,
) -> list[dict[str, Any]]:
    """Return no-secret advisory alerts for recent local retrieval traces."""
    alerts: list[dict[str, Any]] = []
    base_metadata = {
        "total_recent": total_recent,
        "min_events": min_events,
    }
    if total_recent >= min_events and empty_rate >= empty_rate_threshold:
        alerts.append(
            _admin_alert(
                code="retrieval_empty_rate_high",
                severity="warning",
                message="Recent retrieval traces were empty above the configured threshold.",
                metadata=base_metadata
                | {
                    "empty_recent": empty_recent,
                    "empty_rate": f"{empty_rate:.2f}",
                    "threshold": f"{empty_rate_threshold:.2f}",
                },
            )
        )
    if (
        total_recent >= min_events
        and source_page_incomplete_rate >= source_page_incomplete_rate_threshold
    ):
        alerts.append(
            _admin_alert(
                code="retrieval_source_page_incomplete_rate_high",
                severity="warning",
                message="Recent retrieval traces missed source/page metadata above the configured threshold.",
                metadata=base_metadata
                | {
                    "source_page_incomplete_recent": source_page_incomplete_recent,
                    "source_page_incomplete_rate": f"{source_page_incomplete_rate:.2f}",
                    "threshold": f"{source_page_incomplete_rate_threshold:.2f}",
                },
            )
        )
    if (
        citation_checked_recent >= min_events
        and citation_failure_rate >= citation_failure_rate_threshold
    ):
        alerts.append(
            _admin_alert(
                code="retrieval_citation_failure_rate_high",
                severity="warning",
                message="Recent generated-query retrieval traces failed citation validation above the configured threshold.",
                metadata={
                    "citation_checked_recent": citation_checked_recent,
                    "citation_failed_recent": citation_failed_recent,
                    "citation_failure_rate": f"{citation_failure_rate:.2f}",
                    "threshold": f"{citation_failure_rate_threshold:.2f}",
                    "min_events": min_events,
                },
            )
        )
    return alerts


def summarize_provider_failure_alerts(
    *,
    total_recent: int,
    total_query_outcomes: int,
    failure_rate: float,
    by_code: dict[str, int],
    min_events: int,
    failure_rate_threshold: float,
) -> list[dict[str, Any]]:
    """Return no-secret advisory alerts for recent provider failures."""
    alerts: list[dict[str, Any]] = []
    if total_recent >= min_events and failure_rate >= failure_rate_threshold:
        alerts.append(
            _admin_alert(
                code="provider_failure_rate_high",
                severity="warning",
                message="Recent provider failure rate is above the configured threshold.",
                metadata={
                    "total_recent_failures": total_recent,
                    "total_query_outcomes": total_query_outcomes,
                    "failure_rate": f"{failure_rate:.2f}",
                    "threshold": f"{failure_rate_threshold:.2f}",
                    "min_events": min_events,
                },
            )
        )
    if by_code:
        repeated_code, repeated_count = max(by_code.items(), key=lambda item: item[1])
        if repeated_count >= min_events:
            alerts.append(
                _admin_alert(
                    code="provider_failure_code_repeated",
                    severity="info",
                    message="A provider failure code repeated in recent events.",
                    metadata={
                        "failure_code": repeated_code,
                        "failure_count": repeated_count,
                        "min_events": min_events,
                    },
                )
            )
    return alerts


def summarize_job_alerts(
    *,
    failed_recent: int,
    dead_lettered_recent: int,
    queue_health: dict[str, Any],
    worker_leases: dict[str, Any],
    failed_min_events: int,
    expired_min_events: int,
) -> list[dict[str, Any]]:
    """Return no-secret advisory alerts for local job and worker health."""
    alerts: list[dict[str, Any]] = []
    if failed_recent >= failed_min_events:
        alerts.append(
            _admin_alert(
                code="job_failures_recent",
                severity="warning",
                message="Recent local jobs failed above the configured threshold.",
                metadata={
                    "failed_recent": failed_recent,
                    "threshold": failed_min_events,
                },
            )
        )
    if dead_lettered_recent:
        alerts.append(
            _admin_alert(
                code="job_dead_letters_recent",
                severity="warning",
                message="Recent local jobs reached dead-letter state.",
                metadata={"dead_lettered_recent": dead_lettered_recent},
            )
        )
    expired_deadlines = _dict_int(queue_health, "expired")
    if expired_deadlines >= expired_min_events:
        alerts.append(
            _admin_alert(
                code="job_queue_deadlines_expired",
                severity="warning",
                message="Queued local jobs have expired deadlines.",
                metadata={
                    "expired": expired_deadlines,
                    "threshold": expired_min_events,
                },
            )
        )
    expired_queued_leases = _dict_int(queue_health, "lease_expired_queued")
    expired_worker_leases = _dict_int(worker_leases, "expired_leases")
    if max(expired_queued_leases, expired_worker_leases) >= expired_min_events:
        alerts.append(
            _admin_alert(
                code="job_worker_leases_expired",
                severity="warning",
                message="Local worker leases have expired.",
                metadata={
                    "lease_expired_queued": expired_queued_leases,
                    "expired_leases": expired_worker_leases,
                    "threshold": expired_min_events,
                },
            )
        )
    return alerts


def _local_model_path_exists(value: str) -> bool:
    if not value:
        return False
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.exists()


def _runtime_path_readiness(name: str, path: Path) -> dict[str, Any]:
    exists = path.exists()
    writable_target = path if exists else path.parent
    writable = os.access(writable_target, os.W_OK) if writable_target.exists() else False
    return {
        "name": name,
        "path": _relative_runtime_path(path),
        "exists": exists,
        "writable": writable,
    }


def _file_inventory(name: str, path: Path) -> dict[str, Any]:
    exists = path.exists()
    is_file = path.is_file()
    return {
        "name": name,
        "path": _relative_runtime_path(path),
        "exists": exists,
        "is_file": is_file,
        "bytes": path.stat().st_size if is_file else 0,
    }


def _directory_inventory(
    name: str,
    path: Path,
    *,
    known_files: dict[str, Path] | None = None,
) -> dict[str, Any]:
    total_files = 0
    total_bytes = 0
    if path.exists():
        for item in path.rglob("*"):
            if not item.is_file():
                continue
            total_files += 1
            total_bytes += item.stat().st_size
    return {
        "name": name,
        "path": _relative_runtime_path(path),
        "exists": path.exists(),
        "files": total_files,
        "bytes": total_bytes,
        "known_files": [
            _file_inventory(file_name, file_path)
            for file_name, file_path in (known_files or {}).items()
        ],
    }


def storage_inventory_status() -> dict[str, Any]:
    """Return no-secret local storage counts for admin dashboards."""
    groups = [
        _directory_inventory(
            "metadata",
            METADATA_DIR,
            known_files={
                "corpus_json": CORPUS_METADATA_FILE,
                "corpus_profiles_json": CORPUS_PROFILES_FILE,
                "corpus_sqlite": CORPUS_METADATA_DB_FILE,
                "chunks_sqlite": CHUNK_METADATA_DB_FILE,
                "api_key_registry_sqlite": API_KEY_REGISTRY_FILE,
                "product_registry_sqlite": PRODUCT_REGISTRY_FILE,
                "share_link_registry_sqlite": SHARE_LINK_TOKEN_STORE_FILE,
                "runtime_events_jsonl": RUNTIME_EVENTS_FILE,
            },
        ),
        _directory_inventory(
            "jobs",
            JOBS_DIR,
            known_files={
                "jobs_jsonl": JOBS_FILE,
                "jobs_sqlite": JOBS_DB_FILE,
            },
        ),
        _directory_inventory(
            "artifacts",
            ARTIFACTS_DIR,
            known_files={
                "artifacts_sqlite": ARTIFACTS_DIR / "artifacts.sqlite3",
            },
        ),
        _directory_inventory("uploads", PAPERS_UPLOADS_DIR),
        _directory_inventory(
            "faiss_index",
            FAISS_INDEX_DIR,
            known_files={
                "index_faiss": FAISS_INDEX_DIR / "index.faiss",
                "index_pkl": FAISS_INDEX_DIR / "index.pkl",
                "active_papers_json": ACTIVE_PAPERS_FILE,
            },
        ),
    ]
    return {
        "mode": "local",
        "content_scanned": False,
        "total_files": sum(group["files"] for group in groups),
        "total_bytes": sum(group["bytes"] for group in groups),
        "groups": groups,
    }


def storage_readiness_status(
    *,
    metadata_backend: str,
    object_backend: str,
    database_url: str,
    object_bucket: str,
    object_endpoint: str,
    object_region: str,
) -> dict[str, Any]:
    """Return no-secret readiness for future durable DB/object storage."""
    metadata_backend = (metadata_backend or "local").strip().lower()
    object_backend = (object_backend or "local").strip().lower()
    local_metadata_paths = [
        _runtime_path_readiness("metadata", METADATA_DIR),
        _runtime_path_readiness("jobs", JOBS_DIR),
    ]
    local_object_paths = [
        _runtime_path_readiness("artifacts", ARTIFACTS_DIR),
        _runtime_path_readiness("uploads", PAPERS_UPLOADS_DIR),
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


def _storage_component_external_ready(component: dict[str, Any]) -> bool:
    return (
        str(component.get("backend", "local")).lower() != "local"
        and bool(component.get("configured"))
        and bool(component.get("available"))
    )


def distributed_job_store_status(
    *,
    backend: str,
    store_url: str,
    queue_name: str,
) -> dict[str, Any]:
    """Return no-secret readiness for future distributed worker job storage."""
    backend = (backend or "local").strip().lower()
    url_configured = bool((store_url or "").strip())
    queue_configured = bool((queue_name or "").strip())

    if backend == "local":
        status = {
            "backend": "local",
            "configured": False,
            "available": True,
            "reason": "local_job_store_active",
            "store_url_configured": False,
            "queue_name_configured": queue_configured,
        }
    elif backend in {"postgres", "postgresql", "redis", "external"}:
        configured = url_configured and queue_configured
        status = {
            "backend": backend,
            "configured": configured,
            "available": configured,
            "reason": "configured_not_connected"
            if configured
            else "store_url_or_queue_name_missing",
            "store_url_configured": url_configured,
            "queue_name_configured": queue_configured,
        }
    else:
        status = {
            "backend": backend,
            "configured": backend != "local",
            "available": False,
            "reason": "unsupported_job_store_backend",
            "store_url_configured": url_configured,
            "queue_name_configured": queue_configured,
        }

    status["external_job_store_configured"] = (
        status["backend"] != "local" and bool(status["configured"])
    )
    status["external_job_store_available"] = (
        status["backend"] != "local" and bool(status["available"])
    )
    return status


def _has_keys(data: dict[str, Any], keys: set[str]) -> bool:
    return keys.issubset(set(data))


def platform_readiness_status(
    *,
    storage_readiness: dict[str, Any],
    storage_schemas: dict[str, Any],
    storage: dict[str, Any],
    jobs: dict[str, Any],
    distributed_job_store: dict[str, Any],
) -> dict[str, Any]:
    """Return no-secret acceptance status for production storage and workers."""
    metadata_storage = storage_readiness.get("metadata", {})
    object_storage = storage_readiness.get("object_storage", {})
    metadata_external_ready = _storage_component_external_ready(metadata_storage)
    object_external_ready = _storage_component_external_ready(object_storage)
    distributed_job_store_external_ready = (
        str(distributed_job_store.get("backend", "local")).lower() != "local"
        and bool(distributed_job_store.get("configured"))
        and bool(distributed_job_store.get("available"))
    )
    schema_ok = bool(storage_schemas.get("ok"))
    schema_problem_count = _dict_int(storage_schemas, "problem_count")
    storage_inventory_ready = storage.get("mode") == "local" and bool(storage.get("groups"))

    storage_blockers: list[str] = []
    if not metadata_external_ready:
        storage_blockers.append("production_metadata_database_not_configured")
    if not object_external_ready:
        storage_blockers.append("production_object_storage_not_configured")
    if not schema_ok:
        storage_blockers.append("local_storage_schema_drift")
    if not storage_inventory_ready:
        storage_blockers.append("local_storage_inventory_unavailable")

    job_storage = jobs.get("storage", {})
    queue_health = jobs.get("queue_health", {})
    worker_leases = jobs.get("worker_leases", {})
    durable_job_store_ready = bool(job_storage.get("sqlite_exists"))
    queue_contract_ready = _has_keys(
        queue_health,
        {
            "queued",
            "due",
            "scheduled",
            "expired",
            "running",
            "leased_queued",
            "lease_expired_queued",
            "running_leased",
            "oldest_queued_at",
        },
    )
    lease_contract_ready = _has_keys(
        worker_leases,
        {
            "total_leased_jobs",
            "worker_ids",
            "active_worker_ids",
            "expired_worker_ids",
            "active_leases",
            "expired_leases",
        },
    )
    queue_health_clean = (
        _dict_int(queue_health, "expired") == 0
        and _dict_int(queue_health, "lease_expired_queued") == 0
        and _dict_int(worker_leases, "expired_leases") == 0
    )

    worker_blockers: list[str] = []
    if not durable_job_store_ready:
        worker_blockers.append("local_durable_job_store_missing")
    if not queue_contract_ready:
        worker_blockers.append("queue_health_contract_missing")
    if not lease_contract_ready:
        worker_blockers.append("worker_lease_contract_missing")
    if not queue_health_clean:
        worker_blockers.append("queue_or_worker_lease_health_not_clean")
    if not distributed_job_store_external_ready:
        worker_blockers.append("distributed_job_store_not_configured")

    storage_ready = not storage_blockers
    workers_ready = not worker_blockers
    return {
        "mode": "local_platform_readiness",
        "scope": [
            "production_storage_migration",
            "distributed_worker_acceptance",
        ],
        "overall_ready": storage_ready and workers_ready,
        "activation_enabled": False,
        "storage_migration": {
            "ready": storage_ready,
            "blockers": storage_blockers,
            "checks": {
                "metadata_database_external_ready": metadata_external_ready,
                "object_storage_external_ready": object_external_ready,
                "storage_schema_ok": schema_ok,
                "storage_schema_problem_count": schema_problem_count,
                "storage_inventory_ready": storage_inventory_ready,
                "storage_group_count": len(storage.get("groups", [])),
            },
        },
        "distributed_workers": {
            "ready": workers_ready,
            "blockers": worker_blockers,
            "checks": {
                "local_worker_bridge_ready": durable_job_store_ready
                and queue_contract_ready
                and lease_contract_ready,
                "local_durable_job_store_ready": durable_job_store_ready,
                "queue_health_contract_ready": queue_contract_ready,
                "worker_lease_contract_ready": lease_contract_ready,
                "queue_health_clean": queue_health_clean,
                "distributed_job_store_backend": distributed_job_store.get("backend", "local"),
                "distributed_job_store_configured": bool(distributed_job_store.get("configured")),
                "distributed_job_store_available": bool(distributed_job_store.get("available")),
                "distributed_job_store_external_ready": distributed_job_store_external_ready,
            },
        },
    }


def format_admin_status_report(status: AdminStatus | dict[str, Any]) -> str:
    """Render the no-secret admin snapshot as a portable Markdown report."""
    data = status.to_dict() if hasattr(status, "to_dict") else status
    jobs = data.get("jobs", {})
    artifacts = data.get("artifacts", {})
    corpus = data.get("corpus", {})
    storage = data.get("storage", {})
    storage_schemas = data.get("storage_schemas", {})
    platform_readiness = data.get("platform_readiness", {})
    provider_failures = data.get("provider_failures", {})
    query_usage = data.get("query_usage", {})
    retrieval_traces = data.get("retrieval_traces", {})
    code_execution = data.get("code_execution", {})
    api_access = data.get("api_access", {})
    admin_checks = data.get("admin_checks", {})
    upload_scans = data.get("upload_scans", {})
    worker_leases = jobs.get("worker_leases", {})
    config = data.get("config", {})
    product_readiness = config.get("product_readiness", {}) or {}
    product_summary = product_readiness.get("summary", {}) or {}
    product_blockers = product_readiness.get("blockers", {}) or {}
    provider_readiness = config.get("provider_readiness", {}) or {}
    provider_summary = provider_readiness.get("summary", {}) or {}
    provider_blockers = provider_readiness.get("blockers", {}) or {}

    lines = [
        "# FluxMind Admin Status",
        "",
        "No-secret local runtime snapshot exported from `GET /admin/status`.",
        "",
        "## Jobs",
        "",
        f"- Total: {jobs.get('total', 0)}",
        f"- By status: {_format_counts(jobs.get('by_status', {}))}",
        f"- By kind: {_format_counts(jobs.get('by_kind', {}))}",
        f"- Owner count: {jobs.get('owner_count', 0)}",
        f"- By ownership source: {_format_counts(jobs.get('by_ownership_source', {}))}",
        f"- Failed: {jobs.get('failed', 0)}",
        f"- Dead lettered: {jobs.get('dead_lettered', 0)}",
        f"- Scheduled: {jobs.get('scheduled', 0)}",
        f"- Alert count: {len(jobs.get('alerts', []))}",
        f"- Queue health: {_format_counts(jobs.get('queue_health', {}))}",
        f"- Worker leases: total={worker_leases.get('total_leased_jobs', 0)}, "
        f"active={worker_leases.get('active_leases', 0)}, "
        f"expired={worker_leases.get('expired_leases', 0)}, "
        f"workers={','.join(worker_leases.get('worker_ids', [])) or 'none'}",
        f"- Storage: {_format_counts(jobs.get('storage', {}))}",
        "",
        "## Corpus",
        "",
        f"- Papers: {corpus.get('papers', 0)}",
        f"- Active: {corpus.get('active', 0)}",
        f"- Indexed: {corpus.get('indexed', 0)}",
        f"- Failed: {corpus.get('failed', 0)}",
        f"- Storage: {_format_counts(corpus.get('storage', {}))}",
        f"- Profiles: {_format_counts(corpus.get('profiles', {}))}",
        f"- Chunks: {_format_counts(corpus.get('chunks', {}))}",
        f"- Index: {_format_counts(corpus.get('index', {}))}",
        "",
        "## Artifacts",
        "",
        f"- Total: {artifacts.get('total', 0)}",
        f"- Owner count: {artifacts.get('owner_count', 0)}",
        f"- By ownership source: {_format_counts(artifacts.get('by_ownership_source', {}))}",
        f"- Bytes: {artifacts.get('bytes', 0)}",
        f"- Storage: {_format_counts(artifacts.get('storage', {}))}",
        f"- Integrity: {_format_counts(artifacts.get('integrity', {}))}",
        "",
        "## Storage Inventory",
        "",
        f"- Mode: {storage.get('mode', 'local')}",
        f"- Content scanned: {_format_bool(storage.get('content_scanned', False))}",
        f"- Total files: {storage.get('total_files', 0)}",
        f"- Total bytes: {storage.get('total_bytes', 0)}",
        "",
    ]
    for group in storage.get("groups", []):
        lines.append(
            f"- {group.get('name', '')}: path={group.get('path', '')}, "
            f"exists={_format_bool(group.get('exists', False))}, "
            f"files={group.get('files', 0)}, bytes={group.get('bytes', 0)}"
        )

    lines.extend(
        [
            "",
            "## Storage Schemas",
            "",
            f"- Schema version: {storage_schemas.get('schema_version', 0)}",
            f"- Mode: {storage_schemas.get('mode', 'local_storage_schema_inventory')}",
            f"- OK: {_format_bool(storage_schemas.get('ok', False))}",
            f"- Store count: {storage_schemas.get('store_count', 0)}",
            f"- Problem count: {storage_schemas.get('problem_count', 0)}",
        ]
    )
    for store in storage_schemas.get("stores", []):
        lines.append(
            f"- {store.get('name', '')}: kind={store.get('kind', '')}, "
            f"exists={_format_bool(store.get('exists', False))}, "
            f"ok={_format_bool(store.get('ok', False))}, "
            f"errors={','.join(store.get('errors', [])) or 'none'}"
        )

    lines.extend(
        [
            "",
            "## Platform Readiness",
            "",
            f"- Mode: {platform_readiness.get('mode', 'local_platform_readiness')}",
            f"- Overall ready: {_format_bool(platform_readiness.get('overall_ready', False))}",
            f"- Activation enabled: {_format_bool(platform_readiness.get('activation_enabled', False))}",
            f"- Storage migration ready: {_format_bool(platform_readiness.get('storage_migration', {}).get('ready', False))}",
            f"- Storage blockers: {', '.join(platform_readiness.get('storage_migration', {}).get('blockers', [])) or 'none'}",
            f"- Distributed workers ready: {_format_bool(platform_readiness.get('distributed_workers', {}).get('ready', False))}",
            f"- Worker blockers: {', '.join(platform_readiness.get('distributed_workers', {}).get('blockers', [])) or 'none'}",
            f"- Distributed job store backend: {platform_readiness.get('distributed_workers', {}).get('checks', {}).get('distributed_job_store_backend', 'local')}",
            f"- Distributed job store external ready: {_format_bool(platform_readiness.get('distributed_workers', {}).get('checks', {}).get('distributed_job_store_external_ready', False))}",
            "",
            "## Provider Failures",
            "",
            f"- Recent total: {provider_failures.get('total_recent', 0)}",
            f"- By code: {_format_counts(provider_failures.get('by_code', {}))}",
            f"- Failure rate: {provider_failures.get('failure_rate', 0)}",
            f"- Alert count: {len(provider_failures.get('alerts', []))}",
            f"- Event log exists: {_format_bool(provider_failures.get('event_log_exists', False))}",
            f"- Event log bytes: {provider_failures.get('event_log_bytes', 0)}",
            "",
            "## Query Usage",
            "",
            f"- Recent total: {query_usage.get('total_recent', 0)}",
            f"- By endpoint: {_format_counts(query_usage.get('by_endpoint', {}))}",
            f"- By answer mode: {_format_counts(query_usage.get('by_answer_mode', {}))}",
            f"- Estimated prompt tokens: {query_usage.get('estimated_prompt_tokens', 0)}",
            f"- Estimated answer tokens: {query_usage.get('estimated_answer_tokens', 0)}",
            f"- Estimated total tokens: {query_usage.get('estimated_total_tokens', 0)}",
            f"- Provider prompt tokens: {query_usage.get('provider_prompt_tokens', 0)}",
            f"- Provider completion tokens: {query_usage.get('provider_completion_tokens', 0)}",
            f"- Provider total tokens: {query_usage.get('provider_total_tokens', 0)}",
            f"- Provider usage events: {query_usage.get('provider_usage_events', 0)}",
            f"- Avg duration ms: {query_usage.get('duration_ms', {}).get('avg', 0)}",
            f"- Max duration ms: {query_usage.get('duration_ms', {}).get('max', 0)}",
            f"- Alert count: {len(query_usage.get('alerts', []))}",
            f"- Estimated cost USD: {query_usage.get('estimated_cost_usd', '0')}",
            f"- Cost source: {query_usage.get('cost_source', 'not_configured')}",
            f"- Pricing configured: {_format_bool(query_usage.get('pricing', {}).get('configured', False))}",
            f"- Pricing provider: {query_usage.get('pricing', {}).get('provider', 'unspecified')}",
            f"- Prompt USD per 1M tokens: {query_usage.get('pricing', {}).get('prompt_usd_per_1m', '0')}",
            f"- Completion USD per 1M tokens: {query_usage.get('pricing', {}).get('completion_usd_per_1m', '0')}",
            "",
            "## Retrieval Traces",
            "",
            f"- Recent total: {retrieval_traces.get('total_recent', 0)}",
            f"- By code: {_format_counts(retrieval_traces.get('by_code', {}))}",
            f"- By endpoint: {_format_counts(retrieval_traces.get('by_endpoint', {}))}",
            f"- By answer mode: {_format_counts(retrieval_traces.get('by_answer_mode', {}))}",
            f"- Empty retrievals: {retrieval_traces.get('empty_recent', 0)}",
            f"- Empty rate: {retrieval_traces.get('empty_rate', 0)}",
            f"- Source/page incomplete: {retrieval_traces.get('source_page_incomplete_recent', 0)}",
            f"- Source/page incomplete rate: {retrieval_traces.get('source_page_incomplete_rate', 0)}",
            f"- Citation checked: {retrieval_traces.get('citation_checked_recent', 0)}",
            f"- Citation failures: {retrieval_traces.get('citation_failed_recent', 0)}",
            f"- Citation failure rate: {retrieval_traces.get('citation_failure_rate', 0)}",
            f"- Provider-called traces: {retrieval_traces.get('provider_called_recent', 0)}",
            f"- Alert count: {len(retrieval_traces.get('alerts', []))}",
            f"- Avg context count: {retrieval_traces.get('context_count', {}).get('avg', 0)}",
            f"- Max context count: {retrieval_traces.get('context_count', {}).get('max', 0)}",
            f"- Avg duration ms: {retrieval_traces.get('duration_ms', {}).get('avg', 0)}",
            f"- Max duration ms: {retrieval_traces.get('duration_ms', {}).get('max', 0)}",
            "",
            "## Code Execution Events",
            "",
            f"- Recent total: {code_execution.get('total_recent', 0)}",
            f"- By code: {_format_counts(code_execution.get('by_code', {}))}",
            f"- By status: {_format_counts(code_execution.get('by_status', {}))}",
            f"- By backend: {_format_counts(code_execution.get('by_backend', {}))}",
            f"- Failed recent: {code_execution.get('failed_recent', 0)}",
            f"- Failure rate: {code_execution.get('failure_rate', 0)}",
            f"- Policy violations: {code_execution.get('policy_violations', 0)}",
            f"- Output truncations: {code_execution.get('output_truncations', 0)}",
            f"- Artifact collection truncations: {code_execution.get('artifact_collection_truncations', 0)}",
            f"- Artifact exported bytes: {code_execution.get('artifact_exported_bytes', 0)}",
            f"- Alert count: {len(code_execution.get('alerts', []))}",
            f"- Avg duration ms: {code_execution.get('duration_ms', {}).get('avg', 0)}",
            f"- Max duration ms: {code_execution.get('duration_ms', {}).get('max', 0)}",
            "",
            "## API Access Audit",
            "",
            f"- Audit enabled: {_format_bool(api_access.get('audit_enabled', False))}",
            f"- Recent total: {api_access.get('total_recent', 0)}",
            f"- By auth status: {_format_counts(api_access.get('by_token_status', {}))}",
            f"- By status code: {_format_counts(api_access.get('by_status_code', {}))}",
            f"- By method: {_format_counts(api_access.get('by_method', {}))}",
            f"- Invalid credentials: {api_access.get('invalid_recent', 0)}",
            f"- Missing credentials: {api_access.get('missing_recent', 0)}",
            f"- Rate limited: {api_access.get('rate_limited_recent', 0)}",
            "",
            "## Admin Check Events",
            "",
            f"- Recent total: {admin_checks.get('total_recent', 0)}",
            f"- By check: {_format_counts(admin_checks.get('by_check', {}))}",
            f"- By code: {_format_counts(admin_checks.get('by_code', {}))}",
            f"- OK checks: {admin_checks.get('ok_recent', 0)}",
            f"- Blocked checks: {admin_checks.get('blocked_recent', 0)}",
            f"- Blocker count total: {admin_checks.get('blocker_count_total', 0)}",
            "",
            "## Upload Scans",
            "",
            f"- Scan enabled: {_format_bool(upload_scans.get('scan_enabled', False))}",
            f"- Recent total: {upload_scans.get('total_recent', 0)}",
            f"- By status: {_format_counts(upload_scans.get('by_status', {}))}",
            f"- By reason: {_format_counts(upload_scans.get('by_reason', {}))}",
            f"- Allowed: {upload_scans.get('allowed_recent', 0)}",
            f"- Blocked: {upload_scans.get('blocked_recent', 0)}",
            f"- Active-content blocks: {upload_scans.get('active_content_recent', 0)}",
            f"- Parse failures: {upload_scans.get('parse_failed_recent', 0)}",
        ]
    )

    code_execution_alerts = code_execution.get("alerts", [])
    if code_execution_alerts:
        lines.extend(["", "Code execution alerts:"])
        for alert in code_execution_alerts[:10]:
            lines.append(
                f"- {alert.get('severity', '')}: {alert.get('code', '')} "
                f"{alert.get('message', '')}"
            )

    query_usage_alerts = query_usage.get("alerts", [])
    if query_usage_alerts:
        lines.extend(["", "Query alerts:"])
        for alert in query_usage_alerts[:10]:
            lines.append(
                f"- {alert.get('severity', '')}: {alert.get('code', '')} "
                f"{alert.get('message', '')}"
            )

    retrieval_trace_alerts = retrieval_traces.get("alerts", [])
    if retrieval_trace_alerts:
        lines.extend(["", "Retrieval trace alerts:"])
        for alert in retrieval_trace_alerts[:10]:
            lines.append(
                f"- {alert.get('severity', '')}: {alert.get('code', '')} "
                f"{alert.get('message', '')}"
            )

    provider_failure_alerts = provider_failures.get("alerts", [])
    if provider_failure_alerts:
        lines.extend(["", "Provider failure alerts:"])
        for alert in provider_failure_alerts[:10]:
            lines.append(
                f"- {alert.get('severity', '')}: {alert.get('code', '')} "
                f"{alert.get('message', '')}"
            )

    job_alerts = jobs.get("alerts", [])
    if job_alerts:
        lines.extend(["", "Job alerts:"])
        for alert in job_alerts[:10]:
            lines.append(
                f"- {alert.get('severity', '')}: {alert.get('code', '')} "
                f"{alert.get('message', '')}"
            )

    latest_failures = provider_failures.get("latest", [])
    if latest_failures:
        lines.extend(["", "Latest provider failures:"])
        for event in latest_failures[:5]:
            metadata = event.get("metadata", {}) or {}
            endpoint = metadata.get("endpoint", "unknown")
            status_code = metadata.get("status_code", "unknown")
            lines.append(
                f"- {event.get('created_at', '')}: {event.get('code', '')} "
                f"request_id_present={_format_bool(event.get('request_id_present', False))} "
                f"endpoint={endpoint} "
                f"status_code={status_code}"
            )

    latest_failed_jobs = jobs.get("latest_failed", [])
    if latest_failed_jobs:
        lines.extend(["", "Latest failed jobs:"])
        for job in latest_failed_jobs[:5]:
            error = job.get("error", {}) or {}
            lines.append(
                f"- {job.get('updated_at', '')}: {job.get('job_id', '')} "
                f"kind={job.get('kind', '')} "
                f"ownership_source={job.get('ownership_source', '')} "
                f"code={error.get('code', '')}"
            )

    latest_usage = query_usage.get("latest", [])
    if latest_usage:
        lines.extend(["", "Latest query usage estimates:"])
        for event in latest_usage[:5]:
            metadata = event.get("metadata", {}) or {}
            lines.append(
                f"- {event.get('created_at', '')}: "
                f"request_id_present={_format_bool(event.get('request_id_present', False))} "
                f"endpoint={metadata.get('endpoint', '')} "
                f"answer_mode={metadata.get('answer_mode', '')} "
                f"estimated_total_tokens={metadata.get('estimated_total_tokens', 0)}"
            )

    latest_retrieval_traces = retrieval_traces.get("latest", [])
    if latest_retrieval_traces:
        lines.extend(["", "Latest retrieval traces:"])
        for event in latest_retrieval_traces[:5]:
            metadata = event.get("metadata", {}) or {}
            lines.append(
                f"- {event.get('created_at', '')}: code={event.get('code', '')} "
                f"endpoint={metadata.get('endpoint', '')} "
                f"answer_mode={metadata.get('answer_mode', '')} "
                f"context_count={metadata.get('context_count', 0)} "
                f"missing_source_page_count={metadata.get('missing_source_page_count', 0)}"
            )

    latest_code_execution = code_execution.get("latest", [])
    if latest_code_execution:
        lines.extend(["", "Latest code execution events:"])
        for event in latest_code_execution[:5]:
            metadata = event.get("metadata", {}) or {}
            lines.append(
                f"- {event.get('created_at', '')}: "
                f"request_id_present={_format_bool(event.get('request_id_present', False))} "
                f"job_id={metadata.get('job_id', '')} "
                f"status={metadata.get('status', '')} "
                f"backend={metadata.get('backend', '')} "
                f"code={event.get('code', '')}"
            )

    latest_api_access = api_access.get("latest", [])
    if latest_api_access:
        lines.extend(["", "Latest API access audit events:"])
        for event in latest_api_access[:5]:
            metadata = event.get("metadata", {}) or {}
            lines.append(
                f"- {event.get('created_at', '')}: "
                f"request_id_present={_format_bool(event.get('request_id_present', False))} "
                f"method={metadata.get('method', '')} "
                f"route_present={_format_bool(metadata.get('route_present', False))} "
                f"status_code={metadata.get('status_code', '')} "
                f"token_status={metadata.get('token_status', '')}"
            )

    latest_admin_checks = admin_checks.get("latest", [])
    if latest_admin_checks:
        lines.extend(["", "Latest admin check events:"])
        for event in latest_admin_checks[:5]:
            metadata = event.get("metadata", {}) or {}
            lines.append(
                f"- {event.get('created_at', '')}: code={event.get('code', '')} "
                f"check={metadata.get('check', '')} "
                f"ok={_format_bool(metadata.get('ok', False))} "
                f"blocker_count={metadata.get('blocker_count', 0)}"
            )

    latest_upload_scans = upload_scans.get("latest", [])
    if latest_upload_scans:
        lines.extend(["", "Latest upload scan events:"])
        for event in latest_upload_scans[:5]:
            metadata = event.get("metadata", {}) or {}
            lines.append(
                f"- {event.get('created_at', '')}: code={event.get('code', '')} "
                f"request_id_present={_format_bool(event.get('request_id_present', False))} "
                f"status={metadata.get('status', '')} "
                f"reasons={','.join(metadata.get('reason_codes', [])) or 'none'} "
                f"pages={metadata.get('page_count', 0)}"
            )

    lines.extend(
        [
            "",
            "## Runtime Directories",
            "",
        ]
    )
    for directory in data.get("runtime_dirs", []):
        lines.append(
            f"- {directory.get('name', '')}: path={directory.get('path', '')}, "
            f"exists={_format_bool(directory.get('exists', False))}, "
            f"writable={_format_bool(directory.get('writable', False))}, "
            f"bytes={directory.get('bytes', 0)}"
        )

    lines.extend(
        [
            "",
            "## Config",
            "",
            f"- LLM model: {config.get('llm_model', '')}",
            f"- Embedding model: {config.get('embedding_model', '')}",
            f"- Reranker model configured: {_format_bool(config.get('reranker_model_configured', False))}",
            f"- Reranker model available: {_format_bool(config.get('reranker_model_available', False))}",
            f"- LLM base URL configured: {_format_bool(config.get('llm_base_url_configured', False))}",
            f"- External providers enabled: {_format_bool(config.get('external_providers_enabled', False))}",
            f"- Metadata storage backend: {config.get('storage_readiness', {}).get('metadata', {}).get('backend', '')}",
            f"- Metadata storage available: {_format_bool(config.get('storage_readiness', {}).get('metadata', {}).get('available', False))}",
            f"- Metadata storage reason: {config.get('storage_readiness', {}).get('metadata', {}).get('reason', '')}",
            f"- Object storage backend: {config.get('storage_readiness', {}).get('object_storage', {}).get('backend', '')}",
            f"- Object storage available: {_format_bool(config.get('storage_readiness', {}).get('object_storage', {}).get('available', False))}",
            f"- Object storage reason: {config.get('storage_readiness', {}).get('object_storage', {}).get('reason', '')}",
            f"- Distributed job store backend: {config.get('distributed_job_store', {}).get('backend', '')}",
            f"- Distributed job store configured: {_format_bool(config.get('distributed_job_store', {}).get('configured', False))}",
            f"- Distributed job store available: {_format_bool(config.get('distributed_job_store', {}).get('available', False))}",
            f"- Distributed job store reason: {config.get('distributed_job_store', {}).get('reason', '')}",
            f"- Code execution backend: {config.get('code_execution_backend', '')}",
            f"- Code execution policy: {config.get('code_execution_policy', '')}",
            f"- Code execution max stdout bytes: {config.get('code_execution_max_stdout_bytes', 0)}",
            f"- Code execution max stderr bytes: {config.get('code_execution_max_stderr_bytes', 0)}",
            f"- Code execution max artifacts: {config.get('code_execution_max_artifacts', 0)}",
            f"- Code execution max artifact bytes: {config.get('code_execution_max_artifact_bytes', 0)}",
            f"- Code execution max artifact total bytes: {config.get('code_execution_max_artifact_total_bytes', 0)}",
            f"- Code execution max artifact candidates: {config.get('code_execution_max_artifact_candidates', 0)}",
            f"- Code execution alert min events: {config.get('code_execution_alert_min_events', 0)}",
            f"- Code execution alert failure rate: {config.get('code_execution_alert_failure_rate', 0)}",
            f"- Code execution alert duration ms: {config.get('code_execution_alert_duration_ms', 0)}",
            f"- Query alert min events: {config.get('query_alert_min_events', 0)}",
            f"- Query alert duration ms: {config.get('query_alert_duration_ms', 0)}",
            f"- Retrieval trace alert min events: {config.get('retrieval_trace_alert_min_events', 0)}",
            f"- Retrieval trace alert empty rate: {config.get('retrieval_trace_alert_empty_rate', 0)}",
            f"- Retrieval trace alert source/page incomplete rate: {config.get('retrieval_trace_alert_source_page_incomplete_rate', 0)}",
            f"- Retrieval trace alert citation failure rate: {config.get('retrieval_trace_alert_citation_failure_rate', 0)}",
            f"- Provider failure alert min events: {config.get('provider_failure_alert_min_events', 0)}",
            f"- Provider failure alert rate: {config.get('provider_failure_alert_rate', 0)}",
            f"- Job alert failed min events: {config.get('job_alert_failed_min_events', 0)}",
            f"- Job alert expired min events: {config.get('job_alert_expired_min_events', 0)}",
            f"- API access audit enabled: {_format_bool(config.get('api_access_audit_enabled', False))}",
            f"- API rate limit enabled: {_format_bool(config.get('api_rate_limit_enabled', False))}",
            f"- API rate limit max requests: {config.get('api_rate_limit_max_requests', 0)}",
            f"- API rate limit window s: {config.get('api_rate_limit_window_s', 0)}",
            f"- Upload scan enabled: {_format_bool(config.get('upload_scan_enabled', False))}",
            f"- Upload scan max pages: {config.get('upload_scan_max_pages', 0)}",
            f"- Upload scan reject encrypted: {_format_bool(config.get('upload_scan_reject_encrypted', False))}",
            f"- Upload scan block active content: {_format_bool(config.get('upload_scan_block_active_content', False))}",
            f"- Retention delete enabled: {_format_bool(config.get('retention_delete_enabled', False))}",
            f"- Code execution allowed imports: {','.join(config.get('code_execution_allowed_imports', []))}",
            f"- Docker execution configured: {_format_bool(config.get('docker_execution', {}).get('configured', False))}",
            f"- Docker execution available: {_format_bool(config.get('docker_execution', {}).get('available', False))}",
            f"- Docker execution reason: {config.get('docker_execution', {}).get('reason', '')}",
            f"- Identity/quotas/billing enabled: {_format_bool(config.get('identity_quotas_billing_enabled', False))}",
            f"- Product local foundation ready: {_format_bool(product_readiness.get('local_foundation_ready', False))}",
            f"- Product activation ready: {_format_bool(product_readiness.get('activation_ready', False))}",
            f"- Product single API token configured: {_format_bool(product_summary.get('single_api_token_configured', False))}",
            f"- Product query cost pricing configured: {_format_bool(product_summary.get('query_cost_pricing_configured', False))}",
            f"- Product quota guard enabled: {_format_bool(product_summary.get('product_quota_guard_enabled', False))}",
            f"- Product RBAC guard enabled: {_format_bool(product_summary.get('product_rbac_guard_enabled', False))}",
            f"- Product local blockers: {_format_report_codes(product_blockers.get('local_foundation', []))}",
            f"- Product activation blockers: {_format_report_codes(product_blockers.get('activation', []))}",
            f"- Provider local foundation ready: {_format_bool(provider_readiness.get('local_foundation_ready', False))}",
            f"- Provider activation ready: {_format_bool(provider_readiness.get('activation_ready', False))}",
            f"- Provider external providers enabled: {_format_bool(provider_readiness.get('external_providers_enabled', False))}",
            f"- Provider external image configured: {_format_bool(provider_summary.get('external_image_provider_configured', False))}",
            f"- Provider hosted execution configured: {_format_bool(provider_summary.get('hosted_execution_provider_configured', False))}",
            f"- Provider MATLAB backend configured: {_format_bool(provider_summary.get('matlab_backend_configured', False))}",
            f"- Provider quota guard enabled: {_format_bool(provider_summary.get('provider_quota_guard_enabled', False))}",
            f"- Provider local blockers: {_format_report_codes(provider_blockers.get('local_foundation', []))}",
            f"- Provider activation blockers: {_format_report_codes(provider_blockers.get('activation', []))}",
            "",
        ]
    )
    return "\n".join(lines)


def _metrics_label_value(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace('"', '\\"')
    )


def _metrics_number(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "0"
    if number != number or number in {float("inf"), float("-inf")}:
        return "0"
    if number.is_integer():
        return str(int(number))
    return f"{number:.6g}"


def format_admin_metrics(status: AdminStatus | dict[str, Any]) -> str:
    """Render no-secret admin status as Prometheus/OpenMetrics-style text."""
    data = status.to_dict() if hasattr(status, "to_dict") else status
    jobs = data.get("jobs", {})
    artifacts = data.get("artifacts", {})
    corpus = data.get("corpus", {})
    storage = data.get("storage", {})
    storage_schemas = data.get("storage_schemas", {})
    platform_readiness = data.get("platform_readiness", {})
    provider_failures = data.get("provider_failures", {})
    query_usage = data.get("query_usage", {})
    retrieval_traces = data.get("retrieval_traces", {})
    code_execution = data.get("code_execution", {})
    api_access = data.get("api_access", {})
    admin_checks = data.get("admin_checks", {})
    upload_scans = data.get("upload_scans", {})
    config = data.get("config", {})
    product_readiness = config.get("product_readiness", {}) or {}
    product_summary = product_readiness.get("summary", {}) or {}
    provider_readiness = config.get("provider_readiness", {}) or {}
    provider_summary = provider_readiness.get("summary", {}) or {}
    storage_readiness = config.get("storage_readiness", {})
    metadata_storage = storage_readiness.get("metadata", {})
    object_storage = storage_readiness.get("object_storage", {})
    distributed_job_store = config.get("distributed_job_store", {})
    docker_execution = config.get("docker_execution", {})
    platform_storage = platform_readiness.get("storage_migration", {})
    platform_workers = platform_readiness.get("distributed_workers", {})

    lines = [
        "# FluxMind no-secret admin metrics.",
        "# Recent event metrics are local-window gauges, not durable counters.",
    ]
    emitted: set[str] = set()

    def emit(
        name: str,
        value: Any,
        help_text: str,
        *,
        labels: dict[str, Any] | None = None,
    ) -> None:
        if name not in emitted:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
            emitted.add(name)
        label_text = ""
        if labels:
            label_pairs = [
                f'{key}="{_metrics_label_value(label_value)}"'
                for key, label_value in sorted(labels.items())
            ]
            label_text = "{" + ",".join(label_pairs) + "}"
        lines.append(f"{name}{label_text} {_metrics_number(value)}")

    def emit_counts(
        name: str,
        help_text: str,
        counts: dict[str, Any],
        label_name: str,
    ) -> None:
        for key, value in sorted((counts or {}).items()):
            emit(name, value, help_text, labels={label_name: key})

    emit("fluxmind_admin_metrics_schema_version", 1, "FluxMind admin metrics schema version.")

    emit("fluxmind_jobs_total", jobs.get("total", 0), "Recent local jobs in the admin status window.")
    emit_counts("fluxmind_jobs_by_status", "Recent local jobs by status.", jobs.get("by_status", {}), "status")
    emit_counts("fluxmind_jobs_by_kind", "Recent local jobs by kind.", jobs.get("by_kind", {}), "kind")
    emit("fluxmind_jobs_failed", jobs.get("failed", 0), "Recent failed local jobs.")
    emit("fluxmind_jobs_cancelled", jobs.get("cancelled", 0), "Recent cancelled local jobs.")
    emit("fluxmind_jobs_scheduled", jobs.get("scheduled", 0), "Recent scheduled local jobs.")
    emit("fluxmind_jobs_dead_lettered", jobs.get("dead_lettered", 0), "Recent dead-lettered local jobs.")
    emit("fluxmind_job_alerts_total", len(jobs.get("alerts", [])), "Current local job advisory alerts.")
    emit_counts(
        "fluxmind_job_queue_state",
        "Current local queue health by state.",
        jobs.get("queue_health", {}),
        "state",
    )
    worker_leases = jobs.get("worker_leases", {})
    emit("fluxmind_worker_leases_total", worker_leases.get("total_leased_jobs", 0), "Local worker leased jobs.")
    emit("fluxmind_worker_leases_active", worker_leases.get("active_leases", 0), "Active local worker leases.")
    emit("fluxmind_worker_leases_expired", worker_leases.get("expired_leases", 0), "Expired local worker leases.")

    emit("fluxmind_corpus_papers_total", corpus.get("papers", 0), "Local corpus papers.")
    emit("fluxmind_corpus_papers_active", corpus.get("active", 0), "Active local corpus papers.")
    emit("fluxmind_corpus_papers_indexed", corpus.get("indexed", 0), "Indexed local corpus papers.")
    emit("fluxmind_corpus_papers_failed", corpus.get("failed", 0), "Failed local corpus papers.")
    corpus_index = corpus.get("index", {})
    emit("fluxmind_corpus_index_fresh", corpus_index.get("fresh", False), "Whether local corpus index is fresh.")
    emit("fluxmind_corpus_faiss_exists", corpus_index.get("faiss_exists", False), "Whether the local FAISS index file exists.")
    emit("fluxmind_corpus_chunk_source_paths", corpus_index.get("chunk_source_paths", 0), "Distinct source paths in chunk metadata.")

    emit("fluxmind_artifacts_total", artifacts.get("total", 0), "Recent local generated artifacts.")
    emit("fluxmind_artifacts_bytes", artifacts.get("bytes", 0), "Local artifact directory bytes.")
    emit_counts(
        "fluxmind_artifact_integrity",
        "Local artifact integrity counts.",
        artifacts.get("integrity", {}),
        "state",
    )

    emit("fluxmind_storage_files_total", storage.get("total_files", 0), "Local runtime storage file count.")
    emit("fluxmind_storage_bytes_total", storage.get("total_bytes", 0), "Local runtime storage bytes.")
    emit("fluxmind_storage_schema_ok", 1 if storage_schemas.get("ok") else 0, "Local storage schema inventory status.")
    emit(
        "fluxmind_storage_schema_problem_total",
        storage_schemas.get("problem_count", 0),
        "Local storage schema inventory problem count.",
    )
    emit_counts(
        "fluxmind_storage_schema_store_ok",
        "Local storage schema status by store.",
        {
            str(store.get("name", "unknown")): 1 if store.get("ok") else 0
            for store in storage_schemas.get("stores", [])
        },
        "store",
    )
    emit(
        "fluxmind_platform_readiness_overall_ready",
        platform_readiness.get("overall_ready", False),
        "Whether local production-storage and distributed-worker acceptance checks are ready.",
    )
    emit(
        "fluxmind_platform_storage_migration_ready",
        platform_storage.get("ready", False),
        "Whether production storage migration acceptance checks are ready.",
    )
    emit(
        "fluxmind_platform_distributed_workers_ready",
        platform_workers.get("ready", False),
        "Whether distributed worker acceptance checks are ready.",
    )
    emit(
        "fluxmind_platform_readiness_blockers_total",
        len(platform_storage.get("blockers", [])) + len(platform_workers.get("blockers", [])),
        "Current production platform readiness blocker count.",
    )
    for code in platform_storage.get("blockers", []):
        emit(
            "fluxmind_platform_readiness_blocker",
            1,
            "Current production platform readiness blockers by area and code.",
            labels={"area": "storage", "code": code},
        )
    for code in platform_workers.get("blockers", []):
        emit(
            "fluxmind_platform_readiness_blocker",
            1,
            "Current production platform readiness blockers by area and code.",
            labels={"area": "workers", "code": code},
        )
    for group in storage.get("groups", []):
        group_label = group.get("name", "unknown")
        emit(
            "fluxmind_storage_group_files",
            group.get("files", 0),
            "Local runtime storage files by group.",
            labels={"group": group_label},
        )
        emit(
            "fluxmind_storage_group_bytes",
            group.get("bytes", 0),
            "Local runtime storage bytes by group.",
            labels={"group": group_label},
        )

    emit("fluxmind_provider_failures_recent_total", provider_failures.get("total_recent", 0), "Recent provider failures.")
    emit_counts(
        "fluxmind_provider_failures_by_code",
        "Recent provider failures by normalized code.",
        provider_failures.get("by_code", {}),
        "code",
    )
    emit("fluxmind_provider_failure_rate", provider_failures.get("failure_rate", 0), "Recent provider failure rate.")
    emit("fluxmind_provider_failure_alerts_total", len(provider_failures.get("alerts", [])), "Current provider-failure advisory alerts.")

    emit("fluxmind_query_usage_recent_total", query_usage.get("total_recent", 0), "Recent query usage events.")
    emit_counts(
        "fluxmind_query_usage_by_endpoint",
        "Recent query usage by internal endpoint.",
        query_usage.get("by_endpoint", {}),
        "endpoint",
    )
    emit("fluxmind_query_estimated_tokens_total", query_usage.get("estimated_total_tokens", 0), "Recent estimated query tokens.")
    emit("fluxmind_query_provider_tokens_total", query_usage.get("provider_total_tokens", 0), "Recent provider-reported query tokens.")
    emit(
        "fluxmind_query_usage_duration_ms",
        query_usage.get("duration_ms", {}).get("avg", 0),
        "Recent query duration in milliseconds.",
        labels={"stat": "avg"},
    )
    emit(
        "fluxmind_query_usage_duration_ms",
        query_usage.get("duration_ms", {}).get("max", 0),
        "Recent query duration in milliseconds.",
        labels={"stat": "max"},
    )
    emit("fluxmind_query_alerts_total", len(query_usage.get("alerts", [])), "Current query advisory alerts.")

    emit("fluxmind_retrieval_traces_recent_total", retrieval_traces.get("total_recent", 0), "Recent retrieval trace events.")
    emit_counts(
        "fluxmind_retrieval_traces_by_code",
        "Recent retrieval trace events by code.",
        retrieval_traces.get("by_code", {}),
        "code",
    )
    emit_counts(
        "fluxmind_retrieval_traces_by_endpoint",
        "Recent retrieval trace events by internal endpoint.",
        retrieval_traces.get("by_endpoint", {}),
        "endpoint",
    )
    emit("fluxmind_retrieval_empty_recent", retrieval_traces.get("empty_recent", 0), "Recent retrieval traces with no context.")
    emit("fluxmind_retrieval_empty_rate", retrieval_traces.get("empty_rate", 0), "Recent retrieval empty rate.")
    emit(
        "fluxmind_retrieval_source_page_incomplete_recent",
        retrieval_traces.get("source_page_incomplete_recent", 0),
        "Recent retrieval traces with missing source/page metadata.",
    )
    emit(
        "fluxmind_retrieval_source_page_incomplete_rate",
        retrieval_traces.get("source_page_incomplete_rate", 0),
        "Recent retrieval source/page incomplete rate.",
    )
    emit(
        "fluxmind_retrieval_citation_checked_recent",
        retrieval_traces.get("citation_checked_recent", 0),
        "Recent retrieval traces with citation validation metadata.",
    )
    emit(
        "fluxmind_retrieval_citation_failed_recent",
        retrieval_traces.get("citation_failed_recent", 0),
        "Recent generated-query retrieval traces with failed citation validation.",
    )
    emit(
        "fluxmind_retrieval_citation_failure_rate",
        retrieval_traces.get("citation_failure_rate", 0),
        "Recent retrieval citation failure rate.",
    )
    emit("fluxmind_retrieval_alerts_total", len(retrieval_traces.get("alerts", [])), "Current retrieval advisory alerts.")
    emit(
        "fluxmind_retrieval_context_count",
        retrieval_traces.get("context_count", {}).get("avg", 0),
        "Recent retrieval context count.",
        labels={"stat": "avg"},
    )
    emit(
        "fluxmind_retrieval_context_count",
        retrieval_traces.get("context_count", {}).get("max", 0),
        "Recent retrieval context count.",
        labels={"stat": "max"},
    )
    emit(
        "fluxmind_retrieval_duration_ms",
        retrieval_traces.get("duration_ms", {}).get("avg", 0),
        "Recent retrieval trace duration in milliseconds.",
        labels={"stat": "avg"},
    )
    emit(
        "fluxmind_retrieval_duration_ms",
        retrieval_traces.get("duration_ms", {}).get("max", 0),
        "Recent retrieval trace duration in milliseconds.",
        labels={"stat": "max"},
    )

    emit("fluxmind_code_execution_recent_total", code_execution.get("total_recent", 0), "Recent code execution events.")
    emit_counts(
        "fluxmind_code_execution_by_code",
        "Recent code execution events by normalized code.",
        code_execution.get("by_code", {}),
        "code",
    )
    emit_counts(
        "fluxmind_code_execution_by_status",
        "Recent code execution events by status.",
        code_execution.get("by_status", {}),
        "status",
    )
    emit_counts(
        "fluxmind_code_execution_by_backend",
        "Recent code execution events by backend.",
        code_execution.get("by_backend", {}),
        "backend",
    )
    emit("fluxmind_code_execution_failed_recent", code_execution.get("failed_recent", 0), "Recent failed code execution events.")
    emit("fluxmind_code_execution_failure_rate", code_execution.get("failure_rate", 0), "Recent code execution failure rate.")
    emit("fluxmind_code_execution_policy_violations", code_execution.get("policy_violations", 0), "Recent code execution policy violations.")
    emit("fluxmind_code_execution_output_truncations", code_execution.get("output_truncations", 0), "Recent code execution output truncations.")
    emit("fluxmind_code_execution_artifact_collection_truncations", code_execution.get("artifact_collection_truncations", 0), "Recent code execution artifact collection truncations.")
    emit("fluxmind_code_execution_artifact_exported_bytes", code_execution.get("artifact_exported_bytes", 0), "Recent code execution exported artifact bytes.")
    emit(
        "fluxmind_code_execution_duration_ms",
        code_execution.get("duration_ms", {}).get("avg", 0),
        "Recent code execution duration in milliseconds.",
        labels={"stat": "avg"},
    )
    emit(
        "fluxmind_code_execution_duration_ms",
        code_execution.get("duration_ms", {}).get("max", 0),
        "Recent code execution duration in milliseconds.",
        labels={"stat": "max"},
    )
    emit("fluxmind_code_execution_alerts_total", len(code_execution.get("alerts", [])), "Current code execution advisory alerts.")

    emit("fluxmind_api_access_audit_enabled", api_access.get("audit_enabled", False), "Whether metadata-only API access audit is enabled.")
    emit("fluxmind_api_access_recent_total", api_access.get("total_recent", 0), "Recent metadata-only API access events.")
    emit_counts(
        "fluxmind_api_access_by_token_status",
        "Recent API access events by token status.",
        api_access.get("by_token_status", {}),
        "token_status",
    )
    emit_counts(
        "fluxmind_api_access_by_status_code",
        "Recent API access events by HTTP status code.",
        api_access.get("by_status_code", {}),
        "status_code",
    )
    emit_counts(
        "fluxmind_api_access_by_method",
        "Recent API access events by HTTP method.",
        api_access.get("by_method", {}),
        "method",
    )
    emit("fluxmind_api_access_invalid_recent", api_access.get("invalid_recent", 0), "Recent invalid API credentials.")
    emit("fluxmind_api_access_missing_recent", api_access.get("missing_recent", 0), "Recent missing API credentials.")
    emit("fluxmind_api_access_valid_recent", api_access.get("valid_recent", 0), "Recent valid API credentials.")
    emit("fluxmind_api_access_rate_limited_recent", api_access.get("rate_limited_recent", 0), "Recent API rate-limited responses.")

    emit("fluxmind_admin_checks_recent_total", admin_checks.get("total_recent", 0), "Recent metadata-only admin readiness check events.")
    emit_counts(
        "fluxmind_admin_checks_by_check",
        "Recent admin readiness check events by check name.",
        admin_checks.get("by_check", {}),
        "check",
    )
    emit_counts(
        "fluxmind_admin_checks_by_code",
        "Recent admin readiness check events by code.",
        admin_checks.get("by_code", {}),
        "code",
    )
    emit("fluxmind_admin_checks_ok_recent", admin_checks.get("ok_recent", 0), "Recent successful admin readiness checks.")
    emit("fluxmind_admin_checks_blocked_recent", admin_checks.get("blocked_recent", 0), "Recent blocked admin readiness checks.")
    emit("fluxmind_admin_checks_blocker_count_total", admin_checks.get("blocker_count_total", 0), "Recent admin readiness blocker counts summed across checks.")

    emit("fluxmind_upload_scan_enabled", upload_scans.get("scan_enabled", False), "Whether local upload scanning is enabled.")
    emit("fluxmind_upload_scans_recent_total", upload_scans.get("total_recent", 0), "Recent upload scan events.")
    emit_counts(
        "fluxmind_upload_scans_by_status",
        "Recent upload scan events by status.",
        upload_scans.get("by_status", {}),
        "status",
    )
    emit_counts(
        "fluxmind_upload_scans_by_reason",
        "Recent upload scan blocked reasons.",
        upload_scans.get("by_reason", {}),
        "reason",
    )
    emit("fluxmind_upload_scans_allowed_recent", upload_scans.get("allowed_recent", 0), "Recent allowed upload scans.")
    emit("fluxmind_upload_scans_blocked_recent", upload_scans.get("blocked_recent", 0), "Recent blocked upload scans.")
    emit("fluxmind_upload_scans_active_content_recent", upload_scans.get("active_content_recent", 0), "Recent active-content upload scan blocks.")
    emit("fluxmind_upload_scans_parse_failed_recent", upload_scans.get("parse_failed_recent", 0), "Recent upload scan parse failures.")

    emit("fluxmind_api_rate_limit_enabled", config.get("api_rate_limit_enabled", False), "Whether local API rate limiting is enabled.")
    emit("fluxmind_api_rate_limit_max_requests", config.get("api_rate_limit_max_requests", 0), "Configured local API rate-limit max requests.")
    emit("fluxmind_api_rate_limit_window_seconds", config.get("api_rate_limit_window_s", 0), "Configured local API rate-limit window seconds.")
    emit("fluxmind_retention_delete_enabled", config.get("retention_delete_enabled", False), "Whether guarded local retention delete is enabled.")
    emit("fluxmind_storage_external_configured", storage_readiness.get("external_storage_configured", False), "Whether external storage is configured.")
    emit("fluxmind_storage_external_available", storage_readiness.get("external_storage_available", False), "Whether configured external storage is available.")
    emit(
        "fluxmind_distributed_job_store_external_configured",
        distributed_job_store.get("external_job_store_configured", False),
        "Whether an external distributed job store is configured.",
    )
    emit(
        "fluxmind_distributed_job_store_external_available",
        distributed_job_store.get("external_job_store_available", False),
        "Whether a configured external distributed job store is available.",
    )
    emit(
        "fluxmind_distributed_job_store_available",
        distributed_job_store.get("available", False),
        "Whether the configured job store readiness target is available.",
        labels={"backend": distributed_job_store.get("backend", "unknown")},
    )
    emit(
        "fluxmind_metadata_storage_available",
        metadata_storage.get("available", False),
        "Whether metadata storage is available.",
        labels={"backend": metadata_storage.get("backend", "unknown")},
    )
    emit(
        "fluxmind_object_storage_available",
        object_storage.get("available", False),
        "Whether object storage is available.",
        labels={"backend": object_storage.get("backend", "unknown")},
    )
    emit("fluxmind_docker_execution_configured", docker_execution.get("configured", False), "Whether Docker execution backend is configured.")
    emit("fluxmind_docker_execution_available", docker_execution.get("available", False), "Whether Docker execution is available to the runtime user.")
    emit("fluxmind_external_providers_enabled", config.get("external_providers_enabled", False), "Whether external providers are enabled.")
    emit("fluxmind_identity_quotas_billing_enabled", config.get("identity_quotas_billing_enabled", False), "Whether identity, quotas, and billing are enabled.")
    emit(
        "fluxmind_product_local_foundation_ready",
        product_readiness.get("local_foundation_ready", False),
        "Whether local no-secret productization foundations are ready.",
    )
    emit(
        "fluxmind_product_activation_ready",
        product_readiness.get("activation_ready", False),
        "Whether configured identity, quota, and billing activation targets are ready.",
    )
    emit(
        "fluxmind_product_activation_blockers_total",
        len((product_readiness.get("blockers", {}) or {}).get("activation", [])),
        "Current product activation blocker count.",
    )
    emit(
        "fluxmind_product_single_api_token_configured",
        product_summary.get("single_api_token_configured", False),
        "Whether the legacy single shared API token is configured.",
    )
    emit(
        "fluxmind_product_quota_guard_enabled",
        product_summary.get("product_quota_guard_enabled", False),
        "Whether local product quota guard is enabled.",
    )
    emit(
        "fluxmind_product_rbac_guard_enabled",
        product_summary.get("product_rbac_guard_enabled", False),
        "Whether local product RBAC guard is enabled.",
    )
    emit(
        "fluxmind_provider_local_foundation_ready",
        provider_readiness.get("local_foundation_ready", False),
        "Whether local no-secret provider foundations are ready.",
    )
    emit(
        "fluxmind_provider_activation_ready",
        provider_readiness.get("activation_ready", False),
        "Whether configured external provider activation targets are ready.",
    )
    emit(
        "fluxmind_provider_activation_blockers_total",
        len((provider_readiness.get("blockers", {}) or {}).get("activation", [])),
        "Current external provider activation blocker count.",
    )
    emit(
        "fluxmind_provider_external_image_configured",
        provider_summary.get("external_image_provider_configured", False),
        "Whether an external image provider target is configured.",
    )
    emit(
        "fluxmind_provider_hosted_execution_configured",
        provider_summary.get("hosted_execution_provider_configured", False),
        "Whether a hosted code execution provider target is configured.",
    )
    emit(
        "fluxmind_provider_matlab_backend_configured",
        provider_summary.get("matlab_backend_configured", False),
        "Whether an external MATLAB backend target is configured.",
    )
    emit(
        "fluxmind_provider_quota_guard_enabled",
        provider_summary.get("provider_quota_guard_enabled", False),
        "Whether external provider quota and cost guards are enabled.",
    )

    lines.append("# EOF")
    return "\n".join(lines) + "\n"


def runtime_directory_status(name: str, path: Path) -> RuntimeDirectoryStatus:
    exists = path.exists()
    writable_target = path if exists else path.parent
    return RuntimeDirectoryStatus(
        name=name,
        path=path.resolve().relative_to(PROJECT_ROOT).as_posix(),
        exists=exists,
        writable=os.access(writable_target, os.W_OK) if writable_target.exists() else False,
        bytes=directory_size_bytes(path),
    )


def _relative_runtime_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _retention_candidates(
    root: Path,
    *,
    retention_days: int,
    limit: int,
    now_ts: float,
    exclude_names: set[str] | None = None,
) -> dict[str, Any]:
    """Return non-destructive retention candidates for one local runtime tree."""
    exclude_names = exclude_names or set()
    retention_days = max(0, int(retention_days))
    if retention_days <= 0:
        return {
            "enabled": False,
            "retention_days": retention_days,
            "total_candidates": 0,
            "bytes": 0,
            "candidates": [],
        }
    if not root.exists():
        return {
            "enabled": True,
            "retention_days": retention_days,
            "total_candidates": 0,
            "bytes": 0,
            "candidates": [],
        }

    cutoff_ts = now_ts - retention_days * 24 * 60 * 60
    candidates: list[dict[str, Any]] = []
    total_bytes = 0
    total_candidates = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.name in exclude_names:
            continue
        try:
            path_stat = path.lstat()
        except OSError:
            continue
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
            continue
        if path_stat.st_mtime > cutoff_ts:
            continue
        total_candidates += 1
        total_bytes += path_stat.st_size
        if len(candidates) < limit:
            candidates.append(
                {
                    "path": _relative_runtime_path(path),
                    "bytes": path_stat.st_size,
                    "age_days": round((now_ts - path_stat.st_mtime) / (24 * 60 * 60), 2),
                    "modified_at": datetime.fromtimestamp(path_stat.st_mtime, timezone.utc).isoformat(),
                }
            )
    return {
        "enabled": True,
        "retention_days": retention_days,
        "total_candidates": total_candidates,
        "bytes": total_bytes,
        "candidates": candidates,
    }


def collect_retention_preview(
    *,
    upload_days: int = 0,
    artifact_days: int = 0,
    limit: int = 100,
    now_ts: float | None = None,
) -> dict[str, Any]:
    """Return a no-delete local retention preview for runtime files."""
    bounded_limit = min(max(limit, 1), 500)
    now = time.time() if now_ts is None else now_ts
    return {
        "mode": "preview",
        "delete_enabled": RETENTION_DELETE_ENABLED,
        "limit": bounded_limit,
        "uploads": _retention_candidates(
            PAPERS_UPLOADS_DIR,
            retention_days=upload_days,
            limit=bounded_limit,
            now_ts=now,
        ),
        "artifacts": _retention_candidates(
            ARTIFACTS_DIR,
            retention_days=artifact_days,
            limit=bounded_limit,
            now_ts=now,
            exclude_names={"artifacts.sqlite3", "artifacts.sqlite3-journal", "artifacts.sqlite3-wal", "artifacts.sqlite3-shm"},
        ),
    }


def _retention_candidate_path(path_value: str, root: Path) -> Path:
    relative_path = Path(path_value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("retention_candidate_path_outside_root")
    path = PROJECT_ROOT / relative_path
    path.parent.resolve().relative_to(root.resolve())
    return path


def _delete_retention_candidates(
    root: Path,
    *,
    retention_days: int,
    limit: int,
    now_ts: float,
    exclude_names: set[str] | None = None,
) -> dict[str, Any]:
    preview = _retention_candidates(
        root,
        retention_days=retention_days,
        limit=limit,
        now_ts=now_ts,
        exclude_names=exclude_names,
    )
    deleted: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    deleted_bytes = 0
    for candidate in preview.get("candidates", []):
        path_text = str(candidate.get("path", ""))
        try:
            path = _retention_candidate_path(path_text, root)
            path_stat = path.lstat()
            if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
                raise ValueError("retention_candidate_not_regular_file")
            bytes_before = path_stat.st_size
            path.unlink()
        except (OSError, ValueError) as exc:
            failures.append(
                {
                    "path": path_text,
                    "error": type(exc).__name__,
                }
            )
            continue
        deleted.append(
            {
                "path": path_text,
                "bytes": bytes_before,
                "age_days": candidate.get("age_days", 0),
            }
        )
        deleted_bytes += bytes_before
    return {
        "enabled": preview["enabled"],
        "retention_days": preview["retention_days"],
        "total_candidates": preview["total_candidates"],
        "candidate_bytes": preview["bytes"],
        "deleted_files": len(deleted),
        "deleted_bytes": deleted_bytes,
        "failed_files": len(failures),
        "deleted": deleted,
        "failures": failures,
    }


def apply_retention_delete(
    *,
    upload_days: int = 0,
    artifact_days: int = 0,
    limit: int = 100,
    now_ts: float | None = None,
) -> dict[str, Any]:
    """Delete age-matched local upload/artifact files only when enabled."""
    bounded_limit = min(max(limit, 1), 500)
    now = time.time() if now_ts is None else now_ts
    if not RETENTION_DELETE_ENABLED:
        preview = collect_retention_preview(
            upload_days=upload_days,
            artifact_days=artifact_days,
            limit=bounded_limit,
            now_ts=now,
        )
        result = {
            **preview,
            "mode": "delete_disabled",
            "deleted_files": 0,
            "deleted_bytes": 0,
            "failed_files": 0,
        }
        _record_retention_delete_event(result)
        return result

    uploads = _delete_retention_candidates(
        PAPERS_UPLOADS_DIR,
        retention_days=upload_days,
        limit=bounded_limit,
        now_ts=now,
    )
    artifacts = _delete_retention_candidates(
        ARTIFACTS_DIR,
        retention_days=artifact_days,
        limit=bounded_limit,
        now_ts=now,
        exclude_names={"artifacts.sqlite3", "artifacts.sqlite3-journal", "artifacts.sqlite3-wal", "artifacts.sqlite3-shm"},
    )
    result = {
        "mode": "delete",
        "delete_enabled": True,
        "limit": bounded_limit,
        "uploads": uploads,
        "artifacts": artifacts,
        "deleted_files": uploads["deleted_files"] + artifacts["deleted_files"],
        "deleted_bytes": uploads["deleted_bytes"] + artifacts["deleted_bytes"],
        "failed_files": uploads["failed_files"] + artifacts["failed_files"],
    }
    _record_retention_delete_event(result)
    return result


def _record_retention_delete_event(result: dict[str, Any]) -> None:
    try:
        append_runtime_event(
            kind="retention_delete",
            code="retention_delete_applied" if result.get("delete_enabled") else "retention_delete_disabled",
            message="Metadata-only local retention delete event.",
            metadata={
                "mode": result.get("mode", ""),
                "delete_enabled": result.get("delete_enabled", False),
                "limit": result.get("limit", 0),
                "deleted_files": result.get("deleted_files", 0),
                "deleted_bytes": result.get("deleted_bytes", 0),
                "failed_files": result.get("failed_files", 0),
                "upload_deleted_files": result.get("uploads", {}).get("deleted_files", 0),
                "artifact_deleted_files": result.get("artifacts", {}).get("deleted_files", 0),
            },
        )
    except OSError:
        pass


def corpus_index_status(papers: list[Any], chunk_metadata_store: ChunkMetadataStore) -> dict[str, Any]:
    """Summarize whether FAISS/chunk metadata matches the active corpus."""
    active_sources = sorted(paper.source_path for paper in papers if paper.active)
    chunk_sources = sorted(chunk_metadata_store.source_paths())
    missing_chunk_sources = sorted(set(active_sources) - set(chunk_sources))
    extra_chunk_sources = sorted(set(chunk_sources) - set(active_sources))
    faiss_exists = (FAISS_INDEX_DIR / "index.faiss").exists()

    if not active_sources:
        status = "empty"
    elif not faiss_exists:
        status = "missing"
    elif missing_chunk_sources or extra_chunk_sources:
        status = "stale"
    else:
        status = "fresh"

    return {
        "status": status,
        "fresh": status == "fresh",
        "faiss_exists": faiss_exists,
        "active_source_paths": len(active_sources),
        "chunk_source_paths": len(chunk_sources),
        "missing_chunk_sources": missing_chunk_sources,
        "extra_chunk_sources": extra_chunk_sources,
    }


def collect_corpus_profile_status(profile_id: str) -> dict[str, Any]:
    """Inspect a saved corpus profile without changing the active selection."""
    profile = CorpusProfileStore().get_profile(profile_id)
    papers = refresh_paper_metadata()
    papers_by_path = {paper.source_path: paper for paper in papers}
    profile_sources = sorted(profile.source_paths)
    available_sources = [source_path for source_path in profile_sources if source_path in papers_by_path]
    missing_sources = [source_path for source_path in profile_sources if source_path not in papers_by_path]
    active_sources = sorted(paper.source_path for paper in papers if paper.active)
    chunk_sources = sorted(ChunkMetadataStore().source_paths())
    missing_chunk_sources = sorted(set(profile_sources) - set(chunk_sources))
    extra_chunk_sources = sorted(set(chunk_sources) - set(profile_sources))
    faiss_exists = (FAISS_INDEX_DIR / "index.faiss").exists()

    if not profile_sources:
        index_status = "empty"
    elif missing_sources:
        index_status = "invalid"
    elif not faiss_exists:
        index_status = "missing"
    elif missing_chunk_sources or extra_chunk_sources:
        index_status = "stale"
    else:
        index_status = "fresh"

    return {
        "profile": asdict(profile),
        "paper_count": len(profile_sources),
        "available_papers": len(available_sources),
        "missing_source_paths": missing_sources,
        "active_match": active_sources == profile_sources,
        "rebuild_required": index_status != "fresh",
        "index": {
            "status": index_status,
            "fresh": index_status == "fresh",
            "faiss_exists": faiss_exists,
            "profile_source_paths": len(profile_sources),
            "chunk_source_paths": len(chunk_sources),
            "missing_chunk_sources": missing_chunk_sources,
            "extra_chunk_sources": extra_chunk_sources,
        },
        "papers": [
            asdict(papers_by_path[source_path])
            for source_path in available_sources
        ],
    }


def format_corpus_profile_status_report(status: dict[str, Any]) -> str:
    """Format a no-secret corpus profile status snapshot as Markdown."""
    profile = status.get("profile", {})
    index = status.get("index", {})
    papers = status.get("papers", [])
    lines = [
        "# FluxMind Corpus Profile Status",
        "",
        "No-secret local corpus profile snapshot.",
        "",
        "## Profile",
        "",
        f"- Profile ID: {profile.get('profile_id', '')}",
        f"- Name: {profile.get('name', '')}",
        f"- Description: {profile.get('description') or ''}",
        f"- Paper count: {status.get('paper_count', 0)}",
        f"- Available papers: {status.get('available_papers', 0)}",
        f"- Active match: {status.get('active_match', False)}",
        f"- Rebuild required: {status.get('rebuild_required', False)}",
        "",
        "## Index",
        "",
        f"- Status: {index.get('status', '')}",
        f"- Fresh: {index.get('fresh', False)}",
        f"- FAISS exists: {index.get('faiss_exists', False)}",
        f"- Profile source paths: {index.get('profile_source_paths', 0)}",
        f"- Chunk source paths: {index.get('chunk_source_paths', 0)}",
        "",
        "## Missing Sources",
        "",
    ]
    missing_sources = status.get("missing_source_paths", [])
    if missing_sources:
        lines.extend(f"- `{source_path}`" for source_path in missing_sources)
    else:
        lines.append("- None")

    lines.extend(["", "## Missing Chunk Sources", ""])
    missing_chunk_sources = index.get("missing_chunk_sources", [])
    if missing_chunk_sources:
        lines.extend(f"- `{source_path}`" for source_path in missing_chunk_sources)
    else:
        lines.append("- None")

    lines.extend(["", "## Extra Chunk Sources", ""])
    extra_chunk_sources = index.get("extra_chunk_sources", [])
    if extra_chunk_sources:
        lines.extend(f"- `{source_path}`" for source_path in extra_chunk_sources)
    else:
        lines.append("- None")

    lines.extend(["", "## Papers", ""])
    if papers:
        for paper in papers:
            title = paper.get("title") or paper.get("filename") or paper.get("source_path")
            lines.append(
                f"- `{paper.get('source_path', '')}` - {title} "
                f"({paper.get('indexed_status', '')})"
            )
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def corpus_status_from_state(
    papers: list[Any],
    chunk_metadata_store: ChunkMetadataStore,
    jobs: list[Any],
    metadata_store: CorpusMetadataStore,
    profile_store: CorpusProfileStore | None = None,
) -> dict[str, Any]:
    """Summarize corpus lifecycle state without exposing document contents."""
    index_status = corpus_index_status(papers, chunk_metadata_store)
    profile_store = profile_store or CorpusProfileStore()
    index_jobs = [job for job in jobs if job.kind == "index_rebuild"]
    index_job_counts = Counter(job.status for job in index_jobs)
    active_count = sum(1 for paper in papers if paper.active)
    failed_count = sum(1 for paper in papers if paper.indexed_status == "failed")

    if any(job.status == "running" for job in index_jobs):
        status = "parsing"
    elif any(job.status == "queued" for job in index_jobs):
        status = "queued"
    elif failed_count:
        status = "failed"
    elif not papers or not active_count:
        status = "empty"
    elif index_status["fresh"]:
        status = "indexed"
    else:
        status = "stale"

    return {
        "status": status,
        "papers": len(papers),
        "active": active_count,
        "available": sum(1 for paper in papers if not paper.active),
        "indexed": sum(1 for paper in papers if paper.indexed_status == "indexed"),
        "failed": failed_count,
        "storage": metadata_store.storage_status(),
        "profiles": profile_store.storage_status(),
        "chunks": chunk_metadata_store.storage_status(),
        "index": index_status,
        "index_jobs": {
            "by_status": dict(sorted(index_job_counts.items())),
            "latest": [
                {
                    "job_id": job.job_id,
                    "status": job.status,
                    "updated_at": job.updated_at,
                    "error": job.error,
                }
                for job in index_jobs[:5]
            ],
        },
    }


def collect_corpus_status(*, job_limit: int = 100) -> dict[str, Any]:
    """Collect corpus-level lifecycle status for API/UI consumers."""
    metadata_store = CorpusMetadataStore()
    profile_store = CorpusProfileStore()
    chunk_metadata_store = ChunkMetadataStore()
    papers = refresh_paper_metadata()
    jobs = LocalJobStore().list_latest(limit=job_limit)
    return corpus_status_from_state(papers, chunk_metadata_store, jobs, metadata_store, profile_store)


def collect_admin_status(*, job_limit: int = 500) -> AdminStatus:
    job_store = LocalJobStore()
    jobs = job_store.list_latest(limit=job_limit)
    job_status_counts = Counter(job.status for job in jobs)
    job_kind_counts = Counter(job.kind for job in jobs)
    job_owner_count = len({job.owner_id for job in jobs if job.owner_id})
    job_ownership_source_counts = Counter(job.ownership_source for job in jobs)
    failed_jobs = [job for job in jobs if job.status == "failed"]
    dead_lettered_jobs = [
        job for job in jobs if job.status == "dead_lettered" or job.dead_lettered_at
    ]
    cancelled_jobs = [job for job in jobs if job.status == "cancelled"]
    scheduled_jobs = [job for job in jobs if job.status == "queued" and job.not_before]
    queue_health = job_store.queue_health()
    worker_leases = job_store.worker_lease_health()
    job_alerts = summarize_job_alerts(
        failed_recent=len(failed_jobs),
        dead_lettered_recent=len(dead_lettered_jobs),
        queue_health=queue_health,
        worker_leases=worker_leases,
        failed_min_events=JOB_ALERT_FAILED_MIN_EVENTS,
        expired_min_events=JOB_ALERT_EXPIRED_MIN_EVENTS,
    )
    provider_failure_events = list_runtime_events(kind="provider_failure", limit=20)
    provider_failure_counts = Counter(event.code for event in provider_failure_events)
    query_usage_events = list_runtime_events(kind="query_usage", limit=100)
    total_query_outcomes = len(provider_failure_events) + len(query_usage_events)
    provider_failure_rate = (
        len(provider_failure_events) / total_query_outcomes
        if total_query_outcomes
        else 0.0
    )
    provider_failure_alerts = summarize_provider_failure_alerts(
        total_recent=len(provider_failure_events),
        total_query_outcomes=total_query_outcomes,
        failure_rate=provider_failure_rate,
        by_code=dict(provider_failure_counts),
        min_events=PROVIDER_FAILURE_ALERT_MIN_EVENTS,
        failure_rate_threshold=PROVIDER_FAILURE_ALERT_RATE,
    )
    query_usage_by_endpoint = Counter(
        str(event.metadata.get("endpoint", "unknown")) for event in query_usage_events
    )
    query_usage_by_answer_mode = Counter(
        str(event.metadata.get("answer_mode", "unknown")) for event in query_usage_events
    )
    estimated_prompt_tokens = sum(
        int(event.metadata.get("estimated_prompt_tokens", 0) or 0)
        for event in query_usage_events
    )
    estimated_answer_tokens = sum(
        int(event.metadata.get("estimated_answer_tokens", 0) or 0)
        for event in query_usage_events
    )
    provider_prompt_tokens = sum(
        int(event.metadata.get("provider_prompt_tokens", 0) or 0)
        for event in query_usage_events
    )
    provider_completion_tokens = sum(
        int(event.metadata.get("provider_completion_tokens", 0) or 0)
        for event in query_usage_events
    )
    provider_total_tokens = sum(
        int(event.metadata.get("provider_total_tokens", 0) or 0)
        for event in query_usage_events
    )
    provider_usage_events = sum(
        1
        for event in query_usage_events
        if event.metadata.get("usage_source") == "provider"
    )
    query_usage_duration_ms = [
        _event_int_metadata(event, "duration_ms")
        for event in query_usage_events
        if event.metadata.get("duration_ms") is not None
    ]
    query_usage_avg_duration_ms = (
        sum(query_usage_duration_ms) // len(query_usage_duration_ms)
        if query_usage_duration_ms
        else 0
    )
    query_usage_max_duration_ms = max(query_usage_duration_ms) if query_usage_duration_ms else 0
    query_usage_alerts = summarize_query_usage_alerts(
        total_recent=len(query_usage_events),
        avg_duration_ms=query_usage_avg_duration_ms,
        max_duration_ms=query_usage_max_duration_ms,
        min_events=QUERY_ALERT_MIN_EVENTS,
        duration_ms_threshold=QUERY_ALERT_DURATION_MS,
    )
    cost_prompt_tokens = sum(
        _query_cost_token_value(event, "provider_prompt_tokens", "estimated_prompt_tokens")
        for event in query_usage_events
    )
    cost_completion_tokens = sum(
        _query_cost_token_value(event, "provider_completion_tokens", "estimated_answer_tokens")
        for event in query_usage_events
    )
    query_cost = summarize_query_cost(
        estimated_prompt_tokens=estimated_prompt_tokens,
        estimated_completion_tokens=estimated_answer_tokens,
        provider_prompt_tokens=provider_prompt_tokens,
        provider_completion_tokens=provider_completion_tokens,
        provider_usage_events=provider_usage_events,
        total_events=len(query_usage_events),
        cost_prompt_tokens=cost_prompt_tokens,
        cost_completion_tokens=cost_completion_tokens,
        provider=QUERY_COST_PROVIDER or LLM_MODEL,
        prompt_usd_per_1m=QUERY_COST_PROMPT_USD_PER_1M,
        completion_usd_per_1m=QUERY_COST_COMPLETION_USD_PER_1M,
    )
    retrieval_trace_events = list_runtime_events(kind="retrieval_trace", limit=100)
    retrieval_trace_by_code = Counter(event.code for event in retrieval_trace_events)
    retrieval_trace_by_endpoint = Counter(
        str(event.metadata.get("endpoint", "unknown")) for event in retrieval_trace_events
    )
    retrieval_trace_by_answer_mode = Counter(
        str(event.metadata.get("answer_mode", "unknown")) for event in retrieval_trace_events
    )
    retrieval_context_counts = [
        _event_int_metadata(event, "context_count") for event in retrieval_trace_events
    ]
    retrieval_duration_ms = [
        _event_int_metadata(event, "duration_ms")
        for event in retrieval_trace_events
        if event.metadata.get("duration_ms") is not None
    ]
    retrieval_source_page_incomplete_recent = sum(
        1
        for event in retrieval_trace_events
        if event.code == "retrieval_source_page_incomplete"
        or event.metadata.get("source_page_complete") is False
        or _event_int_metadata(event, "missing_source_page_count") > 0
    )
    retrieval_empty_recent = sum(
        1
        for event in retrieval_trace_events
        if event.code == "retrieval_empty"
        or _event_int_metadata(event, "context_count") <= 0
    )
    retrieval_empty_rate = (
        retrieval_empty_recent / len(retrieval_trace_events)
        if retrieval_trace_events
        else 0.0
    )
    retrieval_source_page_incomplete_rate = (
        retrieval_source_page_incomplete_recent / len(retrieval_trace_events)
        if retrieval_trace_events
        else 0.0
    )
    retrieval_citation_checked_recent = sum(
        1 for event in retrieval_trace_events if event.metadata.get("citation_ok") is not None
    )
    retrieval_citation_failed_recent = sum(
        1 for event in retrieval_trace_events if event.metadata.get("citation_ok") is False
    )
    retrieval_citation_failure_rate = (
        retrieval_citation_failed_recent / retrieval_citation_checked_recent
        if retrieval_citation_checked_recent
        else 0.0
    )
    retrieval_provider_called_recent = sum(
        1 for event in retrieval_trace_events if event.metadata.get("provider_called") is True
    )
    retrieval_trace_alerts = summarize_retrieval_trace_alerts(
        total_recent=len(retrieval_trace_events),
        empty_recent=retrieval_empty_recent,
        empty_rate=retrieval_empty_rate,
        source_page_incomplete_recent=retrieval_source_page_incomplete_recent,
        source_page_incomplete_rate=retrieval_source_page_incomplete_rate,
        citation_checked_recent=retrieval_citation_checked_recent,
        citation_failed_recent=retrieval_citation_failed_recent,
        citation_failure_rate=retrieval_citation_failure_rate,
        min_events=RETRIEVAL_TRACE_ALERT_MIN_EVENTS,
        empty_rate_threshold=RETRIEVAL_TRACE_ALERT_EMPTY_RATE,
        source_page_incomplete_rate_threshold=RETRIEVAL_TRACE_ALERT_SOURCE_PAGE_INCOMPLETE_RATE,
        citation_failure_rate_threshold=RETRIEVAL_TRACE_ALERT_CITATION_FAILURE_RATE,
    )
    code_execution_events = list_runtime_events(kind="code_execution", limit=100)
    code_execution_by_code = Counter(event.code for event in code_execution_events)
    code_execution_by_status = Counter(
        str(event.metadata.get("status", "unknown")) for event in code_execution_events
    )
    code_execution_by_backend = Counter(
        str(event.metadata.get("backend", "unknown")) for event in code_execution_events
    )
    code_execution_failed_recent = sum(
        1
        for event in code_execution_events
        if event.metadata.get("status") in {"failed", "cancelled", "dead_lettered"}
        or event.code not in {"execution_succeeded"}
    )
    code_execution_failure_rate = (
        code_execution_failed_recent / len(code_execution_events)
        if code_execution_events
        else 0.0
    )
    code_execution_policy_violations = sum(
        1
        for event in code_execution_events
        if event.metadata.get("policy_violation") == "true"
        or event.code == "execution_policy_violation"
    )
    code_execution_duration_ms = [
        _event_int_metadata(event, "duration_ms") for event in code_execution_events
    ]
    code_execution_output_truncations = sum(
        1
        for event in code_execution_events
        if event.metadata.get("output_truncated") == "true"
    )
    code_execution_artifact_collection_truncations = sum(
        1
        for event in code_execution_events
        if event.metadata.get("artifact_collection_truncated") == "true"
    )
    code_execution_artifact_exported_bytes = sum(
        _event_int_metadata(event, "artifact_exported_bytes")
        for event in code_execution_events
    )
    code_execution_max_duration_ms = max(code_execution_duration_ms) if code_execution_duration_ms else 0
    code_execution_alerts = summarize_code_execution_alerts(
        total_recent=len(code_execution_events),
        failed_recent=code_execution_failed_recent,
        failure_rate=code_execution_failure_rate,
        max_duration_ms=code_execution_max_duration_ms,
        policy_violations=code_execution_policy_violations,
        output_truncations=code_execution_output_truncations,
        artifact_collection_truncations=code_execution_artifact_collection_truncations,
        min_events=CODE_EXECUTION_ALERT_MIN_EVENTS,
        failure_rate_threshold=CODE_EXECUTION_ALERT_FAILURE_RATE,
        duration_ms_threshold=CODE_EXECUTION_ALERT_DURATION_MS,
    )
    api_access_events = list_runtime_events(kind="api_access", limit=100)
    api_access_by_code = Counter(event.code for event in api_access_events)
    api_access_by_token_status = Counter(
        str(event.metadata.get("token_status", "unknown")) for event in api_access_events
    )
    api_access_by_status_code = Counter(
        str(event.metadata.get("status_code", "unknown")) for event in api_access_events
    )
    api_access_by_method = Counter(
        str(event.metadata.get("method", "unknown")) for event in api_access_events
    )
    api_access_invalid_recent = sum(
        1 for event in api_access_events if event.metadata.get("token_status") == "invalid"
    )
    api_access_missing_recent = sum(
        1 for event in api_access_events if event.metadata.get("token_status") == "missing"
    )
    api_access_valid_recent = sum(
        1 for event in api_access_events if event.metadata.get("token_status") == "valid"
    )
    api_access_rate_limited_recent = sum(
        1
        for event in api_access_events
        if event.metadata.get("rate_limited") is True
        or event.metadata.get("status_code") == 429
    )
    admin_check_events = list_runtime_events(kind="admin_check", limit=100)
    admin_check_by_code = Counter(_admin_check_label(event.code) for event in admin_check_events)
    admin_check_by_check = Counter(
        _admin_check_label(event.metadata.get("check")) for event in admin_check_events
    )
    admin_check_by_status = Counter(
        "ok" if event.metadata.get("ok") is True else "blocked"
        for event in admin_check_events
    )
    admin_check_blocker_count_total = sum(
        max(0, _event_int_metadata(event, "blocker_count"))
        for event in admin_check_events
    )
    upload_scan_events = list_runtime_events(kind="upload_scan", limit=100)
    upload_scan_by_code = Counter(event.code for event in upload_scan_events)
    upload_scan_by_status = Counter(
        str(event.metadata.get("status", "unknown")) for event in upload_scan_events
    )
    upload_scan_reason_counts: Counter[str] = Counter()
    for event in upload_scan_events:
        for reason in event.metadata.get("reason_codes", []) or []:
            upload_scan_reason_counts[str(reason)] += 1
    upload_scan_blocked_recent = sum(
        1
        for event in upload_scan_events
        if event.code == "upload_scan_blocked" or event.metadata.get("status") == "blocked"
    )
    upload_scan_allowed_recent = sum(
        1 for event in upload_scan_events if event.metadata.get("status") == "allowed"
    )
    upload_scan_active_content_recent = sum(
        1
        for event in upload_scan_events
        if any(str(reason).startswith("active_content_") for reason in event.metadata.get("reason_codes", []) or [])
    )
    upload_scan_parse_failed_recent = upload_scan_reason_counts.get("pdf_parse_failed", 0)

    metadata_store = CorpusMetadataStore()
    profile_store = CorpusProfileStore()
    chunk_metadata_store = ChunkMetadataStore()
    papers = refresh_paper_metadata()
    artifact_registry = LocalArtifactRegistry(job_store, db_path=ARTIFACTS_DIR / "artifacts.sqlite3")
    artifacts = artifact_registry.list_artifacts(limit=job_limit)
    artifact_owner_count = len({artifact.owner_id for artifact in artifacts if artifact.owner_id})
    artifact_ownership_source_counts = Counter(artifact.ownership_source for artifact in artifacts)
    corpus_status = corpus_status_from_state(papers, chunk_metadata_store, jobs, metadata_store, profile_store)
    storage_status = storage_inventory_status()
    storage_schemas_status = storage_schema_status()
    storage_readiness = storage_readiness_status(
        metadata_backend=METADATA_STORAGE_BACKEND,
        object_backend=OBJECT_STORAGE_BACKEND,
        database_url=DATABASE_URL,
        object_bucket=OBJECT_STORAGE_BUCKET,
        object_endpoint=OBJECT_STORAGE_ENDPOINT,
        object_region=OBJECT_STORAGE_REGION,
    )
    distributed_job_store = distributed_job_store_status(
        backend=DISTRIBUTED_JOB_STORE_BACKEND,
        store_url=DISTRIBUTED_JOB_STORE_URL,
        queue_name=DISTRIBUTED_JOB_QUEUE_NAME,
    )
    jobs_for_platform_readiness = {
        "storage": {
            "jsonl_exists": JOBS_FILE.exists(),
            "jsonl_bytes": JOBS_FILE.stat().st_size if JOBS_FILE.exists() else 0,
            "sqlite_exists": JOBS_DB_FILE.exists(),
            "sqlite_bytes": JOBS_DB_FILE.stat().st_size if JOBS_DB_FILE.exists() else 0,
        },
        "queue_health": queue_health,
        "worker_leases": worker_leases,
    }
    platform_readiness = platform_readiness_status(
        storage_readiness=storage_readiness,
        storage_schemas=storage_schemas_status,
        storage=storage_status,
        jobs=jobs_for_platform_readiness,
        distributed_job_store=distributed_job_store,
    )
    product_readiness = collect_product_readiness()
    docker_execution = docker_execution_status(
        configured_backend=CODE_EXECUTION_BACKEND,
        image=DOCKER_EXECUTION_IMAGE,
    )
    provider_readiness = collect_provider_readiness(
        code_execution_backend=CODE_EXECUTION_BACKEND,
        docker_status=docker_execution,
    )

    return AdminStatus(
        runtime_dirs=[
            runtime_directory_status("metadata", METADATA_DIR),
            runtime_directory_status("jobs", JOBS_DIR),
            runtime_directory_status("artifacts", ARTIFACTS_DIR),
            runtime_directory_status("faiss_index", FAISS_INDEX_DIR),
        ],
        jobs={
            "total": len(jobs),
            "by_status": dict(sorted(job_status_counts.items())),
            "by_kind": dict(sorted(job_kind_counts.items())),
            "owner_count": job_owner_count,
            "by_ownership_source": dict(sorted(job_ownership_source_counts.items())),
            "failed": len(failed_jobs),
            "cancelled": len(cancelled_jobs),
            "scheduled": len(scheduled_jobs),
            "dead_lettered": len(dead_lettered_jobs),
            "alerts": job_alerts,
            "alert_thresholds": {
                "failed_min_events": JOB_ALERT_FAILED_MIN_EVENTS,
                "expired_min_events": JOB_ALERT_EXPIRED_MIN_EVENTS,
            },
            "storage": {
                "jsonl_exists": JOBS_FILE.exists(),
                "jsonl_bytes": JOBS_FILE.stat().st_size if JOBS_FILE.exists() else 0,
                "sqlite_exists": JOBS_DB_FILE.exists(),
                "sqlite_bytes": JOBS_DB_FILE.stat().st_size if JOBS_DB_FILE.exists() else 0,
            },
            "queue_health": queue_health,
            "worker_leases": worker_leases,
            "latest_failed": [
                {
                    "job_id": job.job_id,
                    "kind": job.kind,
                    "ownership_source": job.ownership_source,
                    "updated_at": job.updated_at,
                    "error": job.error,
                }
                for job in failed_jobs[:5]
            ],
        },
        corpus=corpus_status,
        artifacts={
            "total": len(artifacts),
            "owner_count": artifact_owner_count,
            "by_ownership_source": dict(sorted(artifact_ownership_source_counts.items())),
            "bytes": directory_size_bytes(ARTIFACTS_DIR),
            "storage": artifact_registry.storage_status(),
            "integrity": artifact_registry.integrity_status(limit=job_limit),
        },
        storage=storage_status,
        storage_schemas=storage_schemas_status,
        platform_readiness=platform_readiness,
        provider_failures={
            "total_recent": len(provider_failure_events),
            "by_code": dict(sorted(provider_failure_counts.items())),
            "failure_rate": round(provider_failure_rate, 3),
            "alerts": provider_failure_alerts,
            "alert_thresholds": {
                "min_events": PROVIDER_FAILURE_ALERT_MIN_EVENTS,
                "failure_rate": PROVIDER_FAILURE_ALERT_RATE,
            },
            "event_log_exists": RUNTIME_EVENTS_FILE.exists(),
            "event_log_bytes": RUNTIME_EVENTS_FILE.stat().st_size if RUNTIME_EVENTS_FILE.exists() else 0,
            "latest": [
                _runtime_event_admin_dict(event)
                for event in provider_failure_events[:5]
            ],
        },
        query_usage={
            "total_recent": len(query_usage_events),
            "by_endpoint": dict(sorted(query_usage_by_endpoint.items())),
            "by_answer_mode": dict(sorted(query_usage_by_answer_mode.items())),
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "estimated_answer_tokens": estimated_answer_tokens,
            "estimated_total_tokens": estimated_prompt_tokens + estimated_answer_tokens,
            "provider_prompt_tokens": provider_prompt_tokens,
            "provider_completion_tokens": provider_completion_tokens,
            "provider_total_tokens": provider_total_tokens,
            "provider_usage_events": provider_usage_events,
            "duration_ms": {
                "avg": query_usage_avg_duration_ms,
                "max": query_usage_max_duration_ms,
            },
            "alerts": query_usage_alerts,
            "alert_thresholds": {
                "min_events": QUERY_ALERT_MIN_EVENTS,
                "duration_ms": QUERY_ALERT_DURATION_MS,
            },
            "estimated_cost_usd": query_cost["estimated_cost_usd"],
            "cost_source": query_cost["cost_source"],
            "cost_prompt_tokens": query_cost["cost_prompt_tokens"],
            "cost_completion_tokens": query_cost["cost_completion_tokens"],
            "pricing": query_cost["pricing"],
            "latest": [
                _runtime_event_admin_dict(event)
                for event in query_usage_events[:5]
            ],
        },
        retrieval_traces={
            "total_recent": len(retrieval_trace_events),
            "by_code": dict(sorted(retrieval_trace_by_code.items())),
            "by_endpoint": dict(sorted(retrieval_trace_by_endpoint.items())),
            "by_answer_mode": dict(sorted(retrieval_trace_by_answer_mode.items())),
            "empty_recent": retrieval_empty_recent,
            "empty_rate": round(retrieval_empty_rate, 3),
            "source_page_incomplete_recent": retrieval_source_page_incomplete_recent,
            "source_page_incomplete_rate": round(retrieval_source_page_incomplete_rate, 3),
            "citation_checked_recent": retrieval_citation_checked_recent,
            "citation_failed_recent": retrieval_citation_failed_recent,
            "citation_failure_rate": round(retrieval_citation_failure_rate, 3),
            "provider_called_recent": retrieval_provider_called_recent,
            "alerts": retrieval_trace_alerts,
            "alert_thresholds": {
                "min_events": RETRIEVAL_TRACE_ALERT_MIN_EVENTS,
                "empty_rate": RETRIEVAL_TRACE_ALERT_EMPTY_RATE,
                "source_page_incomplete_rate": RETRIEVAL_TRACE_ALERT_SOURCE_PAGE_INCOMPLETE_RATE,
                "citation_failure_rate": RETRIEVAL_TRACE_ALERT_CITATION_FAILURE_RATE,
            },
            "context_count": {
                "avg": (
                    sum(retrieval_context_counts) // len(retrieval_context_counts)
                    if retrieval_context_counts
                    else 0
                ),
                "max": max(retrieval_context_counts) if retrieval_context_counts else 0,
            },
            "duration_ms": {
                "avg": (
                    sum(retrieval_duration_ms) // len(retrieval_duration_ms)
                    if retrieval_duration_ms
                    else 0
                ),
                "max": max(retrieval_duration_ms) if retrieval_duration_ms else 0,
            },
            "latest": [
                _runtime_event_admin_dict(event)
                for event in retrieval_trace_events[:5]
            ],
        },
        code_execution={
            "total_recent": len(code_execution_events),
            "by_code": dict(sorted(code_execution_by_code.items())),
            "by_status": dict(sorted(code_execution_by_status.items())),
            "by_backend": dict(sorted(code_execution_by_backend.items())),
            "failed_recent": code_execution_failed_recent,
            "failure_rate": round(code_execution_failure_rate, 3),
            "policy_violations": code_execution_policy_violations,
            "output_truncations": code_execution_output_truncations,
            "artifact_collection_truncations": code_execution_artifact_collection_truncations,
            "artifact_exported_bytes": code_execution_artifact_exported_bytes,
            "alerts": code_execution_alerts,
            "alert_thresholds": {
                "min_events": CODE_EXECUTION_ALERT_MIN_EVENTS,
                "failure_rate": CODE_EXECUTION_ALERT_FAILURE_RATE,
                "duration_ms": CODE_EXECUTION_ALERT_DURATION_MS,
            },
            "duration_ms": {
                "avg": (
                    sum(code_execution_duration_ms) // len(code_execution_duration_ms)
                    if code_execution_duration_ms
                    else 0
                ),
                "max": code_execution_max_duration_ms,
            },
            "latest": [
                _runtime_event_admin_dict(event)
                for event in code_execution_events[:5]
            ],
        },
        api_access={
            "audit_enabled": API_ACCESS_AUDIT_ENABLED,
            "total_recent": len(api_access_events),
            "by_code": dict(sorted(api_access_by_code.items())),
            "by_token_status": dict(sorted(api_access_by_token_status.items())),
            "by_status_code": dict(sorted(api_access_by_status_code.items())),
            "by_method": dict(sorted(api_access_by_method.items())),
            "invalid_recent": api_access_invalid_recent,
            "missing_recent": api_access_missing_recent,
            "valid_recent": api_access_valid_recent,
            "rate_limited_recent": api_access_rate_limited_recent,
            "rate_limit": {
                "enabled": API_RATE_LIMIT_ENABLED,
                "max_requests": API_RATE_LIMIT_MAX_REQUESTS,
                "window_s": API_RATE_LIMIT_WINDOW_S,
            },
            "latest": [
                _runtime_event_admin_dict(event)
                for event in api_access_events[:5]
            ],
        },
        admin_checks={
            "audit_enabled": API_ACCESS_AUDIT_ENABLED,
            "total_recent": len(admin_check_events),
            "by_code": dict(sorted(admin_check_by_code.items())),
            "by_check": dict(sorted(admin_check_by_check.items())),
            "by_status": dict(sorted(admin_check_by_status.items())),
            "ok_recent": admin_check_by_status.get("ok", 0),
            "blocked_recent": admin_check_by_status.get("blocked", 0),
            "blocker_count_total": admin_check_blocker_count_total,
            "latest": [
                _admin_check_event_admin_dict(event)
                for event in admin_check_events[:5]
            ],
        },
        upload_scans={
            "scan_enabled": UPLOAD_SCAN_ENABLED,
            "total_recent": len(upload_scan_events),
            "by_code": dict(sorted(upload_scan_by_code.items())),
            "by_status": dict(sorted(upload_scan_by_status.items())),
            "by_reason": dict(sorted(upload_scan_reason_counts.items())),
            "allowed_recent": upload_scan_allowed_recent,
            "blocked_recent": upload_scan_blocked_recent,
            "active_content_recent": upload_scan_active_content_recent,
            "parse_failed_recent": upload_scan_parse_failed_recent,
            "config": {
                "enabled": UPLOAD_SCAN_ENABLED,
                "max_pages": UPLOAD_SCAN_MAX_PAGES,
                "reject_encrypted": UPLOAD_SCAN_REJECT_ENCRYPTED,
                "block_active_content": UPLOAD_SCAN_BLOCK_ACTIVE_CONTENT,
            },
            "latest": [
                _runtime_event_admin_dict(event)
                for event in upload_scan_events[:5]
            ],
        },
        config={
            "llm_base_url_configured": bool(LLM_BASE_URL and "example.com" not in LLM_BASE_URL),
            "llm_model": LLM_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model_configured": bool(RERANKER_MODEL),
            "reranker_model_available": _local_model_path_exists(RERANKER_MODEL),
            "code_execution_backend": CODE_EXECUTION_BACKEND,
            "code_execution_policy": CODE_EXECUTION_POLICY,
            "code_execution_max_stdout_bytes": CODE_EXECUTION_MAX_STDOUT_BYTES,
            "code_execution_max_stderr_bytes": CODE_EXECUTION_MAX_STDERR_BYTES,
            "code_execution_max_artifacts": CODE_EXECUTION_MAX_ARTIFACTS,
            "code_execution_max_artifact_bytes": CODE_EXECUTION_MAX_ARTIFACT_BYTES,
            "code_execution_max_artifact_total_bytes": CODE_EXECUTION_MAX_ARTIFACT_TOTAL_BYTES,
            "code_execution_max_artifact_candidates": CODE_EXECUTION_MAX_ARTIFACT_CANDIDATES,
            "code_execution_alert_min_events": CODE_EXECUTION_ALERT_MIN_EVENTS,
            "code_execution_alert_failure_rate": CODE_EXECUTION_ALERT_FAILURE_RATE,
            "code_execution_alert_duration_ms": CODE_EXECUTION_ALERT_DURATION_MS,
            "query_alert_min_events": QUERY_ALERT_MIN_EVENTS,
            "query_alert_duration_ms": QUERY_ALERT_DURATION_MS,
            "retrieval_trace_alert_min_events": RETRIEVAL_TRACE_ALERT_MIN_EVENTS,
            "retrieval_trace_alert_empty_rate": RETRIEVAL_TRACE_ALERT_EMPTY_RATE,
            "retrieval_trace_alert_source_page_incomplete_rate": RETRIEVAL_TRACE_ALERT_SOURCE_PAGE_INCOMPLETE_RATE,
            "retrieval_trace_alert_citation_failure_rate": RETRIEVAL_TRACE_ALERT_CITATION_FAILURE_RATE,
            "provider_failure_alert_min_events": PROVIDER_FAILURE_ALERT_MIN_EVENTS,
            "provider_failure_alert_rate": PROVIDER_FAILURE_ALERT_RATE,
            "job_alert_failed_min_events": JOB_ALERT_FAILED_MIN_EVENTS,
            "job_alert_expired_min_events": JOB_ALERT_EXPIRED_MIN_EVENTS,
            "api_access_audit_enabled": API_ACCESS_AUDIT_ENABLED,
            "api_rate_limit_enabled": API_RATE_LIMIT_ENABLED,
            "api_rate_limit_max_requests": API_RATE_LIMIT_MAX_REQUESTS,
            "api_rate_limit_window_s": API_RATE_LIMIT_WINDOW_S,
            "upload_scan_enabled": UPLOAD_SCAN_ENABLED,
            "upload_scan_max_pages": UPLOAD_SCAN_MAX_PAGES,
            "upload_scan_reject_encrypted": UPLOAD_SCAN_REJECT_ENCRYPTED,
            "upload_scan_block_active_content": UPLOAD_SCAN_BLOCK_ACTIVE_CONTENT,
            "retention_delete_enabled": RETENTION_DELETE_ENABLED,
            "code_execution_allowed_imports": [
                item.strip()
                for item in CODE_EXECUTION_ALLOWED_IMPORTS.split(",")
                if item.strip()
            ],
            "storage_readiness": storage_readiness,
            "distributed_job_store": distributed_job_store,
            "docker_execution": docker_execution,
            "product_readiness": product_readiness,
            "provider_readiness": provider_readiness,
            "external_providers_enabled": provider_readiness.get(
                "external_providers_enabled",
                False,
            ),
            "identity_quotas_billing_enabled": product_readiness.get(
                "identity_quotas_billing_enabled",
                False,
            ),
        },
    )
