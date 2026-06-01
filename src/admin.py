"""Local admin status for no-key platform foundations."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

from src.config import (
    ACTIVE_PAPERS_FILE,
    ARTIFACTS_DIR,
    CODE_EXECUTION_BACKEND,
    CHUNK_METADATA_DB_FILE,
    CORPUS_METADATA_DB_FILE,
    CORPUS_METADATA_FILE,
    CORPUS_PROFILES_FILE,
    DATABASE_URL,
    DOCKER_EXECUTION_IMAGE,
    EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
    JOBS_DB_FILE,
    JOBS_DIR,
    JOBS_FILE,
    LLM_BASE_URL,
    LLM_MODEL,
    METADATA_DIR,
    METADATA_STORAGE_BACKEND,
    OBJECT_STORAGE_BACKEND,
    OBJECT_STORAGE_BUCKET,
    OBJECT_STORAGE_ENDPOINT,
    OBJECT_STORAGE_REGION,
    PAPERS_UPLOADS_DIR,
    PROJECT_ROOT,
    QUERY_COST_COMPLETION_USD_PER_1M,
    QUERY_COST_PROMPT_USD_PER_1M,
    QUERY_COST_PROVIDER,
    RERANKER_MODEL,
    RUNTIME_EVENTS_FILE,
)
from src.artifacts import LocalArtifactRegistry
from src.costs import summarize_query_cost
from src.ingestion import refresh_paper_metadata
from src.jobs import LocalJobStore
from src.metadata import ChunkMetadataStore, CorpusMetadataStore, CorpusProfileStore
from src.providers import docker_execution_status
from src.runtime import list_runtime_events


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
    provider_failures: dict[str, Any]
    query_usage: dict[str, Any]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _format_counts(counts: dict[str, Any]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _query_cost_token_value(event: Any, provider_key: str, estimated_key: str) -> int:
    provider_value = event.metadata.get(provider_key)
    if event.metadata.get("usage_source") == "provider" and provider_value is not None:
        return int(provider_value or 0)
    return int(event.metadata.get(estimated_key, 0) or 0)


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


def format_admin_status_report(status: AdminStatus | dict[str, Any]) -> str:
    """Render the no-secret admin snapshot as a portable Markdown report."""
    data = status.to_dict() if hasattr(status, "to_dict") else status
    jobs = data.get("jobs", {})
    artifacts = data.get("artifacts", {})
    corpus = data.get("corpus", {})
    storage = data.get("storage", {})
    provider_failures = data.get("provider_failures", {})
    query_usage = data.get("query_usage", {})
    worker_leases = jobs.get("worker_leases", {})
    config = data.get("config", {})

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
        f"- Failed: {jobs.get('failed', 0)}",
        f"- Scheduled: {jobs.get('scheduled', 0)}",
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
            "## Provider Failures",
            "",
            f"- Recent total: {provider_failures.get('total_recent', 0)}",
            f"- By code: {_format_counts(provider_failures.get('by_code', {}))}",
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
            f"- Estimated cost USD: {query_usage.get('estimated_cost_usd', '0')}",
            f"- Cost source: {query_usage.get('cost_source', 'not_configured')}",
            f"- Pricing configured: {_format_bool(query_usage.get('pricing', {}).get('configured', False))}",
            f"- Pricing provider: {query_usage.get('pricing', {}).get('provider', 'unspecified')}",
            f"- Prompt USD per 1M tokens: {query_usage.get('pricing', {}).get('prompt_usd_per_1m', '0')}",
            f"- Completion USD per 1M tokens: {query_usage.get('pricing', {}).get('completion_usd_per_1m', '0')}",
        ]
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
                f"request_id={event.get('request_id', '')} endpoint={endpoint} "
                f"status_code={status_code}"
            )

    latest_failed_jobs = jobs.get("latest_failed", [])
    if latest_failed_jobs:
        lines.extend(["", "Latest failed jobs:"])
        for job in latest_failed_jobs[:5]:
            error = job.get("error", {}) or {}
            lines.append(
                f"- {job.get('updated_at', '')}: {job.get('job_id', '')} "
                f"kind={job.get('kind', '')} code={error.get('code', '')}"
            )

    latest_usage = query_usage.get("latest", [])
    if latest_usage:
        lines.extend(["", "Latest query usage estimates:"])
        for event in latest_usage[:5]:
            metadata = event.get("metadata", {}) or {}
            lines.append(
                f"- {event.get('created_at', '')}: request_id={event.get('request_id', '')} "
                f"endpoint={metadata.get('endpoint', '')} "
                f"answer_mode={metadata.get('answer_mode', '')} "
                f"estimated_total_tokens={metadata.get('estimated_total_tokens', 0)}"
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
            f"- Code execution backend: {config.get('code_execution_backend', '')}",
            f"- Docker execution configured: {_format_bool(config.get('docker_execution', {}).get('configured', False))}",
            f"- Docker execution available: {_format_bool(config.get('docker_execution', {}).get('available', False))}",
            f"- Docker execution reason: {config.get('docker_execution', {}).get('reason', '')}",
            f"- Identity/quotas/billing enabled: {_format_bool(config.get('identity_quotas_billing_enabled', False))}",
            "",
        ]
    )
    return "\n".join(lines)


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
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name in exclude_names:
            continue
        stat = path.stat()
        if stat.st_mtime > cutoff_ts:
            continue
        total_bytes += stat.st_size
        if len(candidates) < limit:
            candidates.append(
                {
                    "path": _relative_runtime_path(path),
                    "bytes": stat.st_size,
                    "age_days": round((now_ts - stat.st_mtime) / (24 * 60 * 60), 2),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                }
            )
    return {
        "enabled": True,
        "retention_days": retention_days,
        "total_candidates": len(candidates)
        if len(candidates) < limit
        else sum(
            1
            for path in root.rglob("*")
            if path.is_file()
            and path.name not in exclude_names
            and path.stat().st_mtime <= cutoff_ts
        ),
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
        "delete_enabled": False,
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
    failed_jobs = [job for job in jobs if job.status == "failed"]
    cancelled_jobs = [job for job in jobs if job.status == "cancelled"]
    scheduled_jobs = [job for job in jobs if job.status == "queued" and job.not_before]
    provider_failure_events = list_runtime_events(kind="provider_failure", limit=20)
    provider_failure_counts = Counter(event.code for event in provider_failure_events)
    query_usage_events = list_runtime_events(kind="query_usage", limit=100)
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

    metadata_store = CorpusMetadataStore()
    profile_store = CorpusProfileStore()
    chunk_metadata_store = ChunkMetadataStore()
    papers = refresh_paper_metadata()
    artifact_registry = LocalArtifactRegistry(job_store, db_path=ARTIFACTS_DIR / "artifacts.sqlite3")
    artifacts = artifact_registry.list_artifacts(limit=job_limit)
    corpus_status = corpus_status_from_state(papers, chunk_metadata_store, jobs, metadata_store, profile_store)

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
            "failed": len(failed_jobs),
            "cancelled": len(cancelled_jobs),
            "scheduled": len(scheduled_jobs),
            "storage": {
                "jsonl_exists": JOBS_FILE.exists(),
                "jsonl_bytes": JOBS_FILE.stat().st_size if JOBS_FILE.exists() else 0,
                "sqlite_exists": JOBS_DB_FILE.exists(),
                "sqlite_bytes": JOBS_DB_FILE.stat().st_size if JOBS_DB_FILE.exists() else 0,
            },
            "queue_health": job_store.queue_health(),
            "worker_leases": job_store.worker_lease_health(),
            "latest_failed": [
                {
                    "job_id": job.job_id,
                    "kind": job.kind,
                    "updated_at": job.updated_at,
                    "error": job.error,
                }
                for job in failed_jobs[:5]
            ],
        },
        corpus=corpus_status,
        artifacts={
            "total": len(artifacts),
            "bytes": directory_size_bytes(ARTIFACTS_DIR),
            "storage": artifact_registry.storage_status(),
            "integrity": artifact_registry.integrity_status(limit=job_limit),
        },
        storage=storage_inventory_status(),
        provider_failures={
            "total_recent": len(provider_failure_events),
            "by_code": dict(sorted(provider_failure_counts.items())),
            "event_log_exists": RUNTIME_EVENTS_FILE.exists(),
            "event_log_bytes": RUNTIME_EVENTS_FILE.stat().st_size if RUNTIME_EVENTS_FILE.exists() else 0,
            "latest": [
                {
                    "event_id": event.event_id,
                    "code": event.code,
                    "message": event.message,
                    "created_at": event.created_at,
                    "request_id": event.request_id,
                    "metadata": event.metadata,
                }
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
            "estimated_cost_usd": query_cost["estimated_cost_usd"],
            "cost_source": query_cost["cost_source"],
            "cost_prompt_tokens": query_cost["cost_prompt_tokens"],
            "cost_completion_tokens": query_cost["cost_completion_tokens"],
            "pricing": query_cost["pricing"],
            "latest": [
                {
                    "event_id": event.event_id,
                    "code": event.code,
                    "message": event.message,
                    "created_at": event.created_at,
                    "request_id": event.request_id,
                    "metadata": event.metadata,
                }
                for event in query_usage_events[:5]
            ],
        },
        config={
            "llm_base_url_configured": bool(LLM_BASE_URL and "example.com" not in LLM_BASE_URL),
            "llm_model": LLM_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model_configured": bool(RERANKER_MODEL),
            "reranker_model_available": _local_model_path_exists(RERANKER_MODEL),
            "code_execution_backend": CODE_EXECUTION_BACKEND,
            "storage_readiness": storage_readiness_status(
                metadata_backend=METADATA_STORAGE_BACKEND,
                object_backend=OBJECT_STORAGE_BACKEND,
                database_url=DATABASE_URL,
                object_bucket=OBJECT_STORAGE_BUCKET,
                object_endpoint=OBJECT_STORAGE_ENDPOINT,
                object_region=OBJECT_STORAGE_REGION,
            ),
            "docker_execution": docker_execution_status(
                configured_backend=CODE_EXECUTION_BACKEND,
                image=DOCKER_EXECUTION_IMAGE,
            ),
            "external_providers_enabled": False,
            "identity_quotas_billing_enabled": False,
        },
    )
