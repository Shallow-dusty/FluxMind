"""Operational status and retention helpers for the local FluxMind runtime."""

from __future__ import annotations

import os
import sqlite3
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.artifacts import LocalArtifactRegistry
from src.config import (
    ARTIFACTS_DIR,
    CODE_EXECUTION_BACKEND,
    FAISS_INDEX_DIR,
    IMAGE_PROVIDER_BACKEND,
    JOBS_DB_FILE,
    JOBS_DIR,
    JOBS_FILE,
    LLM_MODEL,
    METADATA_DIR,
    PAPERS_UPLOADS_DIR,
    PROJECT_ROOT,
    RETENTION_DELETE_ENABLED,
    USER_STORE_FILE,
)
from src.ingestion import load_active_paper_paths, load_library_manifest
from src.jobs import LocalJobStore
from src.metadata import ChunkMetadataStore, CorpusProfileStore
from src.runtime import list_runtime_events
from src.users import LocalUserStore


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
    users: dict[str, Any]
    providers: dict[str, Any]
    activity: dict[str, Any]
    storage: dict[str, Any]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _normalized_source(value: str | Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return _project_relative(path)


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_symlink() or not item.is_file():
            continue
        try:
            total += item.stat().st_size
        except OSError:
            continue
    return total


def runtime_directory_status(name: str, path: Path) -> RuntimeDirectoryStatus:
    exists = path.exists()
    parent = path if exists else path.parent
    return RuntimeDirectoryStatus(
        name=name,
        path=_project_relative(path),
        exists=exists,
        writable=os.access(parent, os.W_OK),
        bytes=directory_size_bytes(path),
    )


def _event_summary(kind: str, *, limit: int = 100) -> dict[str, Any]:
    events = list_runtime_events(kind=kind, limit=limit)
    return {
        "total_recent": len(events),
        "by_code": dict(sorted(Counter(event.code for event in events).items())),
    }


def _query_activity() -> dict[str, Any]:
    events = list_runtime_events(kind="query_usage", limit=100)
    durations = [
        int(event.metadata.get("duration_ms", 0) or 0)
        for event in events
        if event.metadata.get("duration_ms") is not None
    ]
    return {
        "total_recent": len(events),
        "by_endpoint": dict(
            sorted(
                Counter(
                    str(event.metadata.get("endpoint", "unknown"))
                    for event in events
                ).items()
            )
        ),
        "by_answer_mode": dict(
            sorted(
                Counter(
                    str(event.metadata.get("answer_mode", "unknown"))
                    for event in events
                ).items()
            )
        ),
        "duration_ms": {
            "avg": sum(durations) // len(durations) if durations else 0,
            "max": max(durations) if durations else 0,
        },
    }


def _retrieval_activity() -> dict[str, Any]:
    events = list_runtime_events(kind="retrieval_trace", limit=100)
    citation_checked = sum(
        1 for event in events if event.metadata.get("citation_checked")
    )
    citation_failed = sum(
        1
        for event in events
        if event.metadata.get("citation_checked")
        and not event.metadata.get("citation_ok", True)
    )
    empty = sum(
        1 for event in events if int(event.metadata.get("context_count", 0) or 0) == 0
    )
    return {
        "total_recent": len(events),
        "empty_recent": empty,
        "citation_checked_recent": citation_checked,
        "citation_failed_recent": citation_failed,
        "citation_failure_rate": (
            round(citation_failed / citation_checked, 3)
            if citation_checked
            else 0.0
        ),
    }


def _user_status() -> dict[str, Any]:
    if not USER_STORE_FILE.exists():
        return {
            "configured": False,
            "total": 0,
            "active": 0,
            "admins": 0,
            "students": 0,
        }
    try:
        users = LocalUserStore().list_users(include_inactive=True)
    except (OSError, sqlite3.Error):
        return {
            "configured": True,
            "available": False,
            "total": 0,
            "active": 0,
            "admins": 0,
            "students": 0,
        }
    return {
        "configured": True,
        "available": True,
        "total": len(users),
        "active": sum(user.active for user in users),
        "admins": sum(user.role == "admin" for user in users),
        "students": sum(user.role == "student" for user in users),
    }


def collect_corpus_status() -> dict[str, Any]:
    manifest = load_library_manifest()
    active_sources = sorted(
        {_normalized_source(path) for path in load_active_paper_paths()}
    )
    chunk_store = ChunkMetadataStore()
    indexed_sources = sorted(
        {_normalized_source(path) for path in chunk_store.source_paths()}
    )
    profiles = CorpusProfileStore().list_profiles()
    index_files = (
        sum(1 for path in FAISS_INDEX_DIR.rglob("*") if path.is_file())
        if FAISS_INDEX_DIR.exists()
        else 0
    )
    return {
        "library_papers": len(manifest),
        "active_papers": len(active_sources),
        "indexed_sources": len(indexed_sources),
        "profiles": len(profiles),
        "index_exists": index_files > 0,
        "index_files": index_files,
        "rebuild_required": set(active_sources) != set(indexed_sources),
        "active_source_paths": active_sources,
        "indexed_source_paths": indexed_sources,
        "chunk_store": chunk_store.storage_status(),
    }


def collect_corpus_profile_status(profile_id: str) -> dict[str, Any]:
    profile = CorpusProfileStore().get_profile(profile_id)
    profile_sources = sorted({_normalized_source(path) for path in profile.source_paths})
    active_sources = sorted(
        {_normalized_source(path) for path in load_active_paper_paths()}
    )
    indexed_sources = sorted(
        {_normalized_source(path) for path in ChunkMetadataStore().source_paths()}
    )
    missing_sources = [
        source
        for source in profile_sources
        if not (PROJECT_ROOT / source).is_file()
    ]
    return {
        "profile": asdict(profile),
        "source_paths": profile_sources,
        "paper_count": len(profile_sources),
        "missing_source_paths": missing_sources,
        "active": set(profile_sources) == set(active_sources),
        "indexed": set(profile_sources) == set(indexed_sources),
        "rebuild_required": (
            bool(missing_sources) or set(profile_sources) != set(indexed_sources)
        ),
    }


def format_corpus_profile_status_report(status: dict[str, Any]) -> str:
    profile = status.get("profile", {})
    lines = [
        "# FluxMind Corpus Profile",
        "",
        f"- ID: {profile.get('profile_id', '')}",
        f"- Name: {profile.get('name', '')}",
        f"- Papers: {status.get('paper_count', 0)}",
        f"- Active: {str(bool(status.get('active'))).lower()}",
        f"- Indexed: {str(bool(status.get('indexed'))).lower()}",
        f"- Rebuild required: {str(bool(status.get('rebuild_required'))).lower()}",
        "",
        "## Sources",
        "",
    ]
    lines.extend(f"- `{source}`" for source in status.get("source_paths", []))
    if not status.get("source_paths"):
        lines.append("- none")
    if status.get("missing_source_paths"):
        lines.extend(
            [
                "",
                "## Missing sources",
                "",
                *(f"- `{source}`" for source in status["missing_source_paths"]),
            ]
        )
    return "\n".join(lines) + "\n"


def collect_admin_status(*, job_limit: int = 500) -> AdminStatus:
    job_store = LocalJobStore()
    jobs = job_store.list_latest(limit=job_limit)
    job_status_counts = Counter(job.status for job in jobs)
    job_kind_counts = Counter(job.kind for job in jobs)
    artifact_registry = LocalArtifactRegistry(job_store=job_store)
    artifacts = artifact_registry.list_artifacts(limit=job_limit)
    provider_failures = _event_summary("provider_failure", limit=50)
    code_execution = _event_summary("code_execution", limit=100)

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
            "failed": job_status_counts.get("failed", 0),
            "dead_lettered": job_status_counts.get("dead_lettered", 0),
            "queue_health": job_store.queue_health(),
            "worker_leases": job_store.worker_lease_health(),
        },
        corpus=collect_corpus_status(),
        artifacts={
            "total": len(artifacts),
            "bytes": directory_size_bytes(ARTIFACTS_DIR),
            "integrity": artifact_registry.integrity_status(limit=job_limit),
        },
        users=_user_status(),
        providers={
            "llm_model": LLM_MODEL,
            "image_backend": IMAGE_PROVIDER_BACKEND,
            "execution_backend": CODE_EXECUTION_BACKEND,
            "recent_failures": provider_failures,
        },
        activity={
            "queries": _query_activity(),
            "retrieval": _retrieval_activity(),
            "code_execution": code_execution,
        },
        storage={
            "jobs_jsonl": {
                "exists": JOBS_FILE.exists(),
                "bytes": JOBS_FILE.stat().st_size if JOBS_FILE.exists() else 0,
            },
            "jobs_sqlite": {
                "exists": JOBS_DB_FILE.exists(),
                "bytes": JOBS_DB_FILE.stat().st_size if JOBS_DB_FILE.exists() else 0,
            },
            "total_runtime_bytes": sum(
                directory_size_bytes(path)
                for path in (METADATA_DIR, JOBS_DIR, ARTIFACTS_DIR, FAISS_INDEX_DIR)
            ),
        },
        config={
            "llm_model": LLM_MODEL,
            "image_provider_backend": IMAGE_PROVIDER_BACKEND,
            "code_execution_backend": CODE_EXECUTION_BACKEND,
        },
    )


def _status_dict(status: AdminStatus | dict[str, Any]) -> dict[str, Any]:
    if isinstance(status, dict):
        return status
    return status.to_dict()


def format_admin_status_report(status: AdminStatus | dict[str, Any]) -> str:
    payload = _status_dict(status)
    jobs = payload.get("jobs", {})
    corpus = payload.get("corpus", {})
    artifacts = payload.get("artifacts", {})
    users = payload.get("users", {})
    providers = payload.get("providers", payload.get("config", {}))
    activity = payload.get("activity", {})
    query_activity = activity.get("queries", {})
    retrieval = activity.get("retrieval", {})
    return "\n".join(
        [
            "# FluxMind Runtime Status",
            "",
            "## Research runtime",
            "",
            f"- Library papers: {corpus.get('library_papers', 0)}",
            f"- Active papers: {corpus.get('active_papers', 0)}",
            f"- Indexed sources: {corpus.get('indexed_sources', 0)}",
            f"- Index rebuild required: {str(bool(corpus.get('rebuild_required'))).lower()}",
            f"- Users: {users.get('total', 0)} ({users.get('active', 0)} active)",
            f"- Jobs: {jobs.get('total', 0)}",
            f"- Failed jobs: {jobs.get('failed', 0)}",
            f"- Artifacts: {artifacts.get('total', 0)}",
            f"- Recent queries: {query_activity.get('total_recent', 0)}",
            f"- Citation failures: {retrieval.get('citation_failed_recent', 0)}",
            "",
            "## Providers",
            "",
            f"- LLM model: {providers.get('llm_model', '')}",
            f"- Image backend: {providers.get('image_backend', providers.get('image_provider_backend', ''))}",
            f"- Execution backend: {providers.get('execution_backend', providers.get('code_execution_backend', ''))}",
            f"- Recent provider failures: {providers.get('recent_failures', {}).get('total_recent', 0)}",
            "",
        ]
    )


def format_admin_metrics(status: AdminStatus | dict[str, Any]) -> str:
    payload = _status_dict(status)
    jobs = payload.get("jobs", {})
    corpus = payload.get("corpus", {})
    artifacts = payload.get("artifacts", {})
    users = payload.get("users", {})
    activity = payload.get("activity", {})
    metrics = {
        "fluxmind_jobs_total": jobs.get("total", 0),
        "fluxmind_jobs_failed": jobs.get("failed", 0),
        "fluxmind_corpus_library_papers": corpus.get("library_papers", 0),
        "fluxmind_corpus_active_papers": corpus.get("active_papers", 0),
        "fluxmind_corpus_indexed_sources": corpus.get("indexed_sources", 0),
        "fluxmind_corpus_rebuild_required": int(
            bool(corpus.get("rebuild_required"))
        ),
        "fluxmind_artifacts_total": artifacts.get("total", 0),
        "fluxmind_users_total": users.get("total", 0),
        "fluxmind_users_active": users.get("active", 0),
        "fluxmind_queries_recent": activity.get("queries", {}).get(
            "total_recent", 0
        ),
        "fluxmind_citation_failures_recent": activity.get("retrieval", {}).get(
            "citation_failed_recent", 0
        ),
    }
    return "\n".join(f"{name} {value}" for name, value in metrics.items()) + "\n"


def _retention_files(root: Path, *, kind: str) -> list[tuple[str, Path]]:
    if not root.exists():
        return []
    candidates: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        if kind == "artifact" and path.name.startswith("artifacts.sqlite3"):
            continue
        candidates.append((kind, path))
    return candidates


def _retention_candidates(
    *,
    upload_days: int,
    artifact_days: int,
    limit: int,
    now_ts: float,
) -> list[tuple[dict[str, Any], Path]]:
    output: list[tuple[dict[str, Any], Path]] = []
    specs = [
        ("upload", PAPERS_UPLOADS_DIR, max(upload_days, 0)),
        ("artifact", ARTIFACTS_DIR, max(artifact_days, 0)),
    ]
    for kind, root, age_days in specs:
        cutoff = now_ts - age_days * 86400
        for _, path in _retention_files(root, kind=kind):
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_mtime > cutoff:
                continue
            output.append(
                (
                    {
                        "kind": kind,
                        "path": _project_relative(path),
                        "bytes": stat.st_size,
                        "age_days": max(0, int((now_ts - stat.st_mtime) // 86400)),
                    },
                    path,
                )
            )
    output.sort(key=lambda item: (-item[0]["age_days"], item[0]["path"]))
    return output[: max(1, min(limit, 1000))]


def collect_retention_preview(
    *,
    upload_days: int = 30,
    artifact_days: int = 30,
    limit: int = 100,
    now_ts: float | None = None,
) -> dict[str, Any]:
    candidates = _retention_candidates(
        upload_days=upload_days,
        artifact_days=artifact_days,
        limit=limit,
        now_ts=time.time() if now_ts is None else now_ts,
    )
    return {
        "delete_enabled": RETENTION_DELETE_ENABLED,
        "candidate_count": len(candidates),
        "candidate_bytes": sum(item["bytes"] for item, _ in candidates),
        "candidates": [item for item, _ in candidates],
        "deleted_count": 0,
        "deleted_bytes": 0,
        "errors": [],
    }


def apply_retention_delete(
    *,
    upload_days: int = 30,
    artifact_days: int = 30,
    limit: int = 100,
    now_ts: float | None = None,
) -> dict[str, Any]:
    current_ts = time.time() if now_ts is None else now_ts
    candidates = _retention_candidates(
        upload_days=upload_days,
        artifact_days=artifact_days,
        limit=limit,
        now_ts=current_ts,
    )
    result = {
        "delete_enabled": RETENTION_DELETE_ENABLED,
        "candidate_count": len(candidates),
        "candidate_bytes": sum(item["bytes"] for item, _ in candidates),
        "candidates": [item for item, _ in candidates],
        "deleted_count": 0,
        "deleted_bytes": 0,
        "errors": [],
    }
    if not RETENTION_DELETE_ENABLED:
        return result
    for item, path in candidates:
        try:
            path.unlink()
        except OSError as exc:
            result["errors"].append(
                {"path": item["path"], "error": exc.__class__.__name__}
            )
            continue
        result["deleted_count"] += 1
        result["deleted_bytes"] += item["bytes"]
    return result
