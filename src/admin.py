"""Local admin status for no-key platform foundations."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config import (
    ARTIFACTS_DIR,
    EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
    JOBS_DB_FILE,
    JOBS_DIR,
    JOBS_FILE,
    LLM_BASE_URL,
    LLM_MODEL,
    METADATA_DIR,
    PROJECT_ROOT,
)
from src.ingestion import refresh_paper_metadata
from src.jobs import LocalJobStore


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
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def collect_admin_status(*, job_limit: int = 500) -> AdminStatus:
    jobs = LocalJobStore().list_latest(limit=job_limit)
    job_status_counts = Counter(job.status for job in jobs)
    job_kind_counts = Counter(job.kind for job in jobs)
    failed_jobs = [job for job in jobs if job.status == "failed"]
    cancelled_jobs = [job for job in jobs if job.status == "cancelled"]

    papers = refresh_paper_metadata()
    artifact_count = sum(len(job.artifacts) for job in jobs)

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
            "storage": {
                "jsonl_exists": JOBS_FILE.exists(),
                "jsonl_bytes": JOBS_FILE.stat().st_size if JOBS_FILE.exists() else 0,
                "sqlite_exists": JOBS_DB_FILE.exists(),
                "sqlite_bytes": JOBS_DB_FILE.stat().st_size if JOBS_DB_FILE.exists() else 0,
            },
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
        corpus={
            "papers": len(papers),
            "active": sum(1 for paper in papers if paper.active),
            "indexed": sum(1 for paper in papers if paper.indexed_status == "indexed"),
            "failed": sum(1 for paper in papers if paper.indexed_status == "failed"),
        },
        artifacts={
            "total": artifact_count,
            "bytes": directory_size_bytes(ARTIFACTS_DIR),
        },
        config={
            "llm_base_url_configured": bool(LLM_BASE_URL and "example.com" not in LLM_BASE_URL),
            "llm_model": LLM_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "external_providers_enabled": False,
            "identity_quotas_billing_enabled": False,
        },
    )
