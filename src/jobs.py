"""Local job records for no-key artifact and execution workflows."""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from src.capabilities import (
    CodeExecutionRequest,
    CodeExecutionResult,
    GeneratedArtifact,
    ImageGenerationRequest,
)
from src.config import CODE_EXECUTION_BACKEND, DOCKER_EXECUTION_IMAGE, JOBS_FILE, PROJECT_ROOT
from src import ingestion
from src.providers import (
    DockerExecutionProvider,
    LocalArtifactStore,
    LocalOctaveExecutionProvider,
    LocalPythonExecutionProvider,
    MockImageGenerationProvider,
)
from src.runtime import append_runtime_event, new_request_id, normalize_exception


JobKind = Literal["image_generation", "code_execution", "index_rebuild"]
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled", "dead_lettered"]
MAX_IDEMPOTENCY_KEY_LENGTH = 128
MAX_OWNER_FIELD_LENGTH = 128
DEFAULT_OWNER_ID = "local-user"
DEFAULT_OWNER_LABEL = "Local user"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def future_utc(seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=max(seconds, 0))).isoformat()


def has_deadline_expired(record: "JobRecord") -> bool:
    return bool(record.deadline_at and parse_utc(record.deadline_at) <= datetime.now(timezone.utc))


def normalize_idempotency_key(value: str | None) -> str | None:
    """Normalize a user-provided idempotency key for durable local job lookup."""
    if value is None:
        return None
    key = value.strip()
    if not key:
        return None
    if len(key) > MAX_IDEMPOTENCY_KEY_LENGTH:
        raise ValueError(
            f"idempotency_key must be {MAX_IDEMPOTENCY_KEY_LENGTH} characters or fewer"
        )
    return key


def normalize_ownership(
    *,
    owner_id: str | None = None,
    owner_label: str | None = None,
    ownership_source: str | None = None,
) -> dict[str, str]:
    """Normalize local no-key ownership metadata without treating it as auth."""
    clean_owner_id = (owner_id or "").strip() or DEFAULT_OWNER_ID
    clean_owner_label = (owner_label or "").strip()
    if len(clean_owner_id) > MAX_OWNER_FIELD_LENGTH:
        raise ValueError(f"owner_id must be {MAX_OWNER_FIELD_LENGTH} characters or fewer")
    if len(clean_owner_label) > MAX_OWNER_FIELD_LENGTH:
        raise ValueError(f"owner_label must be {MAX_OWNER_FIELD_LENGTH} characters or fewer")
    if not clean_owner_label:
        clean_owner_label = DEFAULT_OWNER_LABEL if clean_owner_id == DEFAULT_OWNER_ID else clean_owner_id
    clean_source = (ownership_source or "").strip()
    if clean_source not in {"default", "request", "inherited"}:
        clean_source = (
            "request"
            if (owner_id and owner_id.strip()) or (owner_label and owner_label.strip())
            else "default"
        )
    return {
        "owner_id": clean_owner_id,
        "owner_label": clean_owner_label,
        "ownership_source": clean_source,
    }


def ownership_from_record(record: "JobRecord") -> dict[str, str]:
    return normalize_ownership(
        owner_id=record.owner_id,
        owner_label=record.owner_label,
        ownership_source=record.ownership_source,
    )


def apply_job_ownership(record: "JobRecord") -> None:
    ownership = ownership_from_record(record)
    record.owner_id = ownership["owner_id"]
    record.owner_label = ownership["owner_label"]
    record.ownership_source = ownership["ownership_source"]


@dataclass
class JobRecord:
    """Serializable local job state."""

    job_id: str
    kind: JobKind
    status: JobStatus
    created_at: str
    updated_at: str
    request: dict[str, Any]
    result: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None
    attempts: int = 0
    request_id: str | None = None
    parent_job_id: str | None = None
    not_before: str | None = None
    deadline_at: str | None = None
    worker_id: str | None = None
    leased_at: str | None = None
    lease_expires_at: str | None = None
    idempotency_key: str | None = None
    max_attempts: int = 1
    retry_backoff_s: int = 0
    dead_lettered_at: str | None = None
    owner_id: str = DEFAULT_OWNER_ID
    owner_label: str = DEFAULT_OWNER_LABEL
    ownership_source: str = "default"
    logs: list[dict[str, Any]] = field(default_factory=list)


def append_job_log(record: JobRecord, status: str, message: str, **metadata: Any) -> None:
    """Append a no-secret transition log entry to a local job record."""
    entry: dict[str, Any] = {
        "created_at": utc_now(),
        "status": status,
        "message": message,
    }
    clean_metadata = {
        key: value
        for key, value in metadata.items()
        if value not in (None, "", [])
    }
    if clean_metadata:
        entry["metadata"] = clean_metadata
    record.logs.append(entry)


class LocalJobStore:
    """JSONL history plus SQLite current-state index for local jobs."""

    def __init__(self, path: Path | None = None, db_path: Path | None = None):
        self.path = path or JOBS_FILE
        self.db_path = db_path or self.path.with_suffix(".sqlite3")

    def append(self, record: JobRecord) -> None:
        apply_job_ownership(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        self._upsert_sqlite(record)

    def append_new(self, record: JobRecord) -> tuple[JobRecord, bool]:
        """Persist a new job, returning an existing idempotent match when claimed."""
        apply_job_ownership(record)
        clean_key = normalize_idempotency_key(record.idempotency_key)
        record.idempotency_key = clean_key
        if clean_key is None:
            self.append(record)
            return record, True

        self._ensure_sqlite()
        existing: JobRecord | None = None
        created = False
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._create_jobs_table(conn)
            self._ensure_jobs_columns(conn)
            self._create_idempotency_table(conn)
            claim = conn.execute(
                """
                SELECT job_id
                FROM job_idempotency
                WHERE kind = ? AND idempotency_key = ?
                LIMIT 1
                """,
                (record.kind, clean_key),
            ).fetchone()
            if claim is not None:
                row = conn.execute(
                    "SELECT payload FROM jobs WHERE job_id = ?",
                    (claim["job_id"],),
                ).fetchone()
                if row is not None:
                    existing = JobRecord(**json.loads(row["payload"]))
                else:
                    conn.execute(
                        "DELETE FROM job_idempotency WHERE kind = ? AND idempotency_key = ?",
                        (record.kind, clean_key),
                    )
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO job_idempotency (kind, idempotency_key, job_id, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (record.kind, clean_key, record.job_id, record.created_at),
                )
                self._upsert_sqlite_conn(conn, record)
                created = True

        if created:
            self._append_jsonl(record)
            return record, True
        return existing, False

    def get(self, job_id: str) -> JobRecord | None:
        self._ensure_sqlite()
        record = self._get_sqlite(job_id)
        if record is not None:
            apply_job_ownership(record)
            return record
        record = self._get_jsonl(job_id)
        if record is not None:
            apply_job_ownership(record)
        return record

    def _get_jsonl(self, job_id: str) -> JobRecord | None:
        latest: dict[str, Any] | None = None
        if not self.path.exists():
            return None
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("job_id") == job_id:
                    latest = item
        if not latest:
            return None
        record = JobRecord(**latest)
        apply_job_ownership(record)
        return record

    def list_latest(
        self,
        *,
        limit: int = 50,
        status: str | None = None,
        kind: str | None = None,
        owner_id: str | None = None,
        q: str | None = None,
    ) -> list[JobRecord]:
        self._ensure_sqlite()
        if self.db_path.exists():
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT payload
                    FROM jobs
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT ?
                    """,
                    (max(limit, 1000),),
                ).fetchall()
            records = [JobRecord(**json.loads(row["payload"])) for row in rows]
            return self._filter_records(records, status=status, kind=kind, owner_id=owner_id, q=q)[:limit]
        records = self._list_latest_jsonl(limit=max(limit, 1000))
        return self._filter_records(records, status=status, kind=kind, owner_id=owner_id, q=q)[:limit]

    def find_by_idempotency_key(self, *, kind: JobKind, key: str | None) -> JobRecord | None:
        """Return the durable job claimed for a kind/idempotency key pair."""
        clean_key = normalize_idempotency_key(key)
        if clean_key is None:
            return None
        self._ensure_sqlite()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT jobs.payload
                FROM job_idempotency
                JOIN jobs ON jobs.job_id = job_idempotency.job_id
                WHERE job_idempotency.kind = ? AND job_idempotency.idempotency_key = ?
                LIMIT 1
                """,
                (kind, clean_key),
            ).fetchone()
            if row is None:
                row = conn.execute(
                    """
                    SELECT payload
                    FROM jobs
                    WHERE kind = ? AND idempotency_key = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (kind, clean_key),
                ).fetchone()
        if row is None:
            return None
        record = JobRecord(**json.loads(row["payload"]))
        apply_job_ownership(record)
        return record

    @staticmethod
    def _filter_records(
        records: list[JobRecord],
        *,
        status: str | None = None,
        kind: str | None = None,
        owner_id: str | None = None,
        q: str | None = None,
    ) -> list[JobRecord]:
        status = status.strip() if status else None
        kind = kind.strip() if kind else None
        owner_id = owner_id.strip() if owner_id else None
        query = (q or "").strip().casefold()
        filtered: list[JobRecord] = []
        for record in records:
            apply_job_ownership(record)
            if status and record.status != status:
                continue
            if kind and record.kind != kind:
                continue
            if owner_id and record.owner_id != owner_id:
                continue
            if query:
                searchable = " ".join(
                    str(value or "")
                    for value in (
                        record.job_id,
                        record.kind,
                        record.status,
                        record.request_id,
                        record.parent_job_id,
                        record.owner_id,
                        record.owner_label,
                        record.ownership_source,
                        record.idempotency_key,
                        json.dumps(record.request or {}, ensure_ascii=False, sort_keys=True),
                        json.dumps(record.result or {}, ensure_ascii=False, sort_keys=True),
                        json.dumps(record.error or {}, ensure_ascii=False, sort_keys=True),
                        json.dumps(record.artifacts or [], ensure_ascii=False, sort_keys=True),
                        json.dumps(record.logs or [], ensure_ascii=False, sort_keys=True),
                    )
                ).casefold()
                if query not in searchable:
                    continue
            filtered.append(record)
        return filtered

    def _list_latest_jsonl(self, *, limit: int = 50) -> list[JobRecord]:
        latest: dict[str, dict[str, Any]] = {}
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                latest[item["job_id"]] = item
        records = [JobRecord(**item) for item in latest.values()]
        for record in records:
            apply_job_ownership(record)
        records.sort(key=lambda record: record.updated_at, reverse=True)
        return records[:limit]

    def cancel(self, job_id: str) -> JobRecord | None:
        record = self.get(job_id)
        if record is None:
            return None
        if record.status in {"succeeded", "failed", "cancelled", "dead_lettered"}:
            return record
        record.status = "cancelled"
        record.updated_at = utc_now()
        record.error = {"code": "cancelled", "message": "Job was cancelled."}
        append_job_log(record, "cancelled", "Job was cancelled.", code="cancelled")
        self.append(record)
        return record

    def scheduled_count(self) -> int:
        now = datetime.now(timezone.utc)
        return sum(
            1
            for job in self.list_latest(limit=10000)
            if job.status == "queued"
            and job.not_before
            and parse_utc(job.not_before) > now
        )

    def list_queued(self, *, limit: int = 1000) -> list[JobRecord]:
        """Return queued jobs from durable state, ordered by creation time."""
        jobs = [job for job in self.list_latest(limit=limit) if job.status == "queued"]
        jobs.sort(key=lambda job: job.created_at)
        return jobs

    def queue_health(self) -> dict[str, Any]:
        """Summarize durable local queue state for admin/status surfaces."""
        now = datetime.now(timezone.utc)
        queued = self.list_queued(limit=10000)
        active_leased_queued = [
            job
            for job in queued
            if job.worker_id
            and job.lease_expires_at
            and parse_utc(job.lease_expires_at) > now
        ]
        expired_leased_queued = [
            job
            for job in queued
            if job.worker_id
            and job.lease_expires_at
            and parse_utc(job.lease_expires_at) <= now
        ]
        due = [
            job for job in queued
            if not job.not_before or parse_utc(job.not_before) <= now
            if job not in active_leased_queued
        ]
        scheduled = [
            job for job in queued
            if job.not_before and parse_utc(job.not_before) > now
        ]
        expired = [job for job in queued if has_deadline_expired(job)]
        running = [job for job in self.list_latest(limit=10000) if job.status == "running"]
        running_leased = [job for job in running if job.worker_id]
        return {
            "queued": len(queued),
            "due": len(due),
            "scheduled": len(scheduled),
            "expired": len(expired),
            "running": len(running),
            "leased_queued": len(active_leased_queued),
            "lease_expired_queued": len(expired_leased_queued),
            "running_leased": len(running_leased),
            "oldest_queued_at": queued[0].created_at if queued else None,
        }

    def worker_lease_health(self, *, limit: int = 5) -> dict[str, Any]:
        """Summarize no-secret worker lease activity for admin/status surfaces."""
        now = datetime.now(timezone.utc)
        leased_jobs = [
            job for job in self.list_latest(limit=10000)
            if job.worker_id
        ]
        by_worker = Counter(str(job.worker_id) for job in leased_jobs if job.worker_id)
        active_jobs = [
            job for job in leased_jobs
            if job.status in {"queued", "running"}
            and job.lease_expires_at
            and parse_utc(job.lease_expires_at) > now
        ]
        expired_jobs = [
            job for job in leased_jobs
            if job.status in {"queued", "running"}
            and job.lease_expires_at
            and parse_utc(job.lease_expires_at) <= now
        ]
        latest = sorted(leased_jobs, key=lambda job: job.updated_at, reverse=True)[:limit]
        return {
            "total_leased_jobs": len(leased_jobs),
            "worker_ids": sorted(by_worker),
            "by_worker": dict(sorted(by_worker.items())),
            "active_worker_ids": sorted({str(job.worker_id) for job in active_jobs if job.worker_id}),
            "expired_worker_ids": sorted({str(job.worker_id) for job in expired_jobs if job.worker_id}),
            "active_leases": len(active_jobs),
            "expired_leases": len(expired_jobs),
            "latest": [
                {
                    "job_id": job.job_id,
                    "kind": job.kind,
                    "status": job.status,
                    "worker_id": job.worker_id,
                    "updated_at": job.updated_at,
                    "leased_at": job.leased_at,
                    "lease_expires_at": job.lease_expires_at,
                    "lease_expired": bool(
                        job.lease_expires_at
                        and parse_utc(job.lease_expires_at) <= now
                    ),
                }
                for job in latest
            ],
        }

    def claim_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_seconds: int = 300,
    ) -> JobRecord | None:
        """Claim a queued job for a worker without starting provider execution."""
        return self._claim_candidate(
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            job_id=job_id,
        )

    def claim_next_due_job(
        self,
        *,
        worker_id: str,
        lease_seconds: int = 300,
    ) -> JobRecord | None:
        """Claim the oldest due queued job for future durable worker loops."""
        return self._claim_candidate(worker_id=worker_id, lease_seconds=lease_seconds)

    def release_job_lease(self, job_id: str, *, worker_id: str | None = None) -> JobRecord | None:
        """Clear a queued job lease so another worker can claim it."""
        record = self.get(job_id)
        if record is None:
            return None
        if worker_id is not None and record.worker_id != worker_id:
            return record
        record.worker_id = None
        record.leased_at = None
        record.lease_expires_at = None
        record.updated_at = utc_now()
        self.append(record)
        return record

    def _claim_candidate(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        job_id: str | None = None,
    ) -> JobRecord | None:
        self._ensure_sqlite()
        now = datetime.now(timezone.utc)
        claimed: JobRecord | None = None
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if job_id:
                rows = conn.execute(
                    """
                    SELECT payload
                    FROM jobs
                    WHERE job_id = ? AND status = 'queued'
                    LIMIT 1
                    """,
                    (job_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT payload
                    FROM jobs
                    WHERE status = 'queued'
                    ORDER BY created_at ASC
                    LIMIT 1000
                    """
                ).fetchall()
            for row in rows:
                record = JobRecord(**json.loads(row["payload"]))
                if not self._is_claimable(record, worker_id=worker_id, now=now):
                    continue
                record.worker_id = worker_id
                record.leased_at = now.isoformat()
                record.lease_expires_at = (
                    now + timedelta(seconds=max(lease_seconds, 1))
                ).isoformat()
                record.updated_at = utc_now()
                conn.execute(
                    """
                    UPDATE jobs
                    SET updated_at = ?,
                        worker_id = ?,
                        leased_at = ?,
                        lease_expires_at = ?,
                        payload = ?
                    WHERE job_id = ?
                    """,
                    (
                        record.updated_at,
                        record.worker_id,
                        record.leased_at,
                        record.lease_expires_at,
                        json.dumps(asdict(record), ensure_ascii=False),
                        record.job_id,
                    ),
                )
                claimed = record
                break
        if claimed is not None:
            self._append_jsonl(claimed)
        return claimed

    @staticmethod
    def _is_claimable(
        record: JobRecord,
        *,
        worker_id: str,
        now: datetime,
    ) -> bool:
        if record.status != "queued":
            return False
        if record.not_before and parse_utc(record.not_before) > now:
            return False
        if (
            record.worker_id
            and record.worker_id != worker_id
            and record.lease_expires_at
            and parse_utc(record.lease_expires_at) > now
        ):
            return False
        return True

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_sqlite(self) -> None:
        with self._connect() as conn:
            self._create_jobs_table(conn)
            self._ensure_jobs_columns(conn)
            self._create_idempotency_table(conn)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_kind ON jobs(kind)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_not_before ON jobs(not_before)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_lease_expires_at ON jobs(lease_expires_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_idempotency_key ON jobs(kind, idempotency_key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_owner_id ON jobs(owner_id)")
        if self.path.exists():
            for record in self._list_latest_jsonl(limit=10000):
                self._upsert_sqlite(record)

    def _upsert_sqlite(self, record: JobRecord) -> None:
        with self._connect() as conn:
            self._create_jobs_table(conn)
            self._ensure_jobs_columns(conn)
            self._create_idempotency_table(conn)
            self._upsert_sqlite_conn(conn, record)

    @staticmethod
    def _upsert_sqlite_conn(conn: sqlite3.Connection, record: JobRecord) -> None:
        apply_job_ownership(record)
        conn.execute(
            """
            INSERT INTO jobs (
                    job_id, kind, status, created_at, updated_at, request_id,
                    attempts, not_before, deadline_at, worker_id, leased_at,
                    lease_expires_at, idempotency_key, max_attempts,
                    retry_backoff_s, dead_lettered_at, owner_id, owner_label,
                    ownership_source, payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    kind=excluded.kind,
                    status=excluded.status,
                updated_at=excluded.updated_at,
                request_id=excluded.request_id,
                attempts=excluded.attempts,
                not_before=excluded.not_before,
                deadline_at=excluded.deadline_at,
                worker_id=excluded.worker_id,
                    leased_at=excluded.leased_at,
                    lease_expires_at=excluded.lease_expires_at,
                    idempotency_key=excluded.idempotency_key,
                    max_attempts=excluded.max_attempts,
                    retry_backoff_s=excluded.retry_backoff_s,
                    dead_lettered_at=excluded.dead_lettered_at,
                    owner_id=excluded.owner_id,
                    owner_label=excluded.owner_label,
                    ownership_source=excluded.ownership_source,
                    payload=excluded.payload
                """,
                (
                record.job_id,
                record.kind,
                record.status,
                record.created_at,
                record.updated_at,
                record.request_id,
                record.attempts,
                record.not_before,
                record.deadline_at,
                record.worker_id,
                    record.leased_at,
                    record.lease_expires_at,
                    record.idempotency_key,
                    record.max_attempts,
                    record.retry_backoff_s,
                    record.dead_lettered_at,
                    record.owner_id,
                    record.owner_label,
                    record.ownership_source,
                    json.dumps(asdict(record), ensure_ascii=False),
                ),
            )
        if record.idempotency_key:
            conn.execute(
                """
                INSERT OR IGNORE INTO job_idempotency (
                    kind, idempotency_key, job_id, created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    record.kind,
                    record.idempotency_key,
                    record.job_id,
                    record.created_at,
                ),
            )

    @staticmethod
    def _create_jobs_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                request_id TEXT,
                attempts INTEGER NOT NULL,
                not_before TEXT,
                deadline_at TEXT,
                worker_id TEXT,
                leased_at TEXT,
                lease_expires_at TEXT,
                idempotency_key TEXT,
                max_attempts INTEGER NOT NULL DEFAULT 1,
                retry_backoff_s INTEGER NOT NULL DEFAULT 0,
                dead_lettered_at TEXT,
                owner_id TEXT NOT NULL DEFAULT 'local-user',
                owner_label TEXT NOT NULL DEFAULT 'Local user',
                ownership_source TEXT NOT NULL DEFAULT 'default',
                payload TEXT NOT NULL
            )
            """
        )

    @staticmethod
    def _ensure_jobs_columns(conn: sqlite3.Connection) -> None:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        for column, definition in {
            "not_before": "TEXT",
            "deadline_at": "TEXT",
            "worker_id": "TEXT",
            "leased_at": "TEXT",
            "lease_expires_at": "TEXT",
            "idempotency_key": "TEXT",
            "max_attempts": "INTEGER NOT NULL DEFAULT 1",
            "retry_backoff_s": "INTEGER NOT NULL DEFAULT 0",
            "dead_lettered_at": "TEXT",
            "owner_id": "TEXT NOT NULL DEFAULT 'local-user'",
            "owner_label": "TEXT NOT NULL DEFAULT 'Local user'",
            "ownership_source": "TEXT NOT NULL DEFAULT 'default'",
        }.items():
            if column not in columns:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {definition}")

    @staticmethod
    def _create_idempotency_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_idempotency (
                kind TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                job_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (kind, idempotency_key)
            )
            """
        )

    def _append_jsonl(self, record: JobRecord) -> None:
        apply_job_ownership(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def _get_sqlite(self, job_id: str) -> JobRecord | None:
        if not self.db_path.exists():
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        record = JobRecord(**json.loads(row["payload"]))
        apply_job_ownership(record)
        return record


class LocalJobRunner:
    """Run no-key providers and persist job transitions."""

    def __init__(
        self,
        store: LocalJobStore | None = None,
        *,
        artifact_root: Path | None = None,
        record_runtime_events: bool = True,
    ):
        self.store = store or LocalJobStore()
        self.artifact_root = artifact_root
        self.record_runtime_events = record_runtime_events

    def _artifact_store(self) -> LocalArtifactStore:
        return LocalArtifactStore(self.artifact_root) if self.artifact_root else LocalArtifactStore()

    def _code_execution_provider(self, language: str):
        store = self._artifact_store()
        if CODE_EXECUTION_BACKEND == "docker":
            return DockerExecutionProvider(store, image=DOCKER_EXECUTION_IMAGE)
        if language == "octave":
            return LocalOctaveExecutionProvider(store)
        return LocalPythonExecutionProvider(store)

    @staticmethod
    def _code_execution_event_code(
        *,
        status: JobStatus,
        error: dict[str, Any] | None,
    ) -> str:
        if error and error.get("code"):
            return str(error["code"])
        if status == "succeeded":
            return "execution_succeeded"
        if status == "cancelled":
            return "cancelled"
        return "execution_failed"

    def _record_code_execution_event(
        self,
        record: JobRecord,
        request: CodeExecutionRequest,
        *,
        status: JobStatus,
        result: CodeExecutionResult | None,
        error: dict[str, Any] | None,
        duration_ms: int,
    ) -> None:
        """Append a no-secret execution outcome event without affecting the job."""
        if not self.record_runtime_events:
            return
        runtime_metadata = result.runtime_metadata if result else {}
        metadata: dict[str, Any] = {
            "job_id": record.job_id,
            "status": status,
            "language": request.language,
            "backend": CODE_EXECUTION_BACKEND,
            "attempt": record.attempts,
            "max_attempts": record.max_attempts,
            "duration_ms": max(duration_ms, 0),
            "artifact_count": len(result.artifacts) if result else 0,
            "owner_id": record.owner_id,
            "owner_label": record.owner_label,
            "ownership_source": record.ownership_source,
        }
        if result is not None:
            metadata["exit_code"] = result.exit_code
        if error and error.get("code"):
            metadata["error_code"] = error["code"]
        for key in (
            "provider_runtime",
            "runtime_available",
            "network_policy_enforced",
            "filesystem_isolation",
            "timeout_s",
            "memory_mb",
            "memory_limit_enforced",
            "cpu_limit_enforced",
            "execution_policy",
            "execution_policy_enforced",
            "execution_policy_checked_files",
            "execution_policy_violations",
            "policy_violation",
            "max_stdout_bytes",
            "max_stderr_bytes",
            "max_artifacts",
            "max_artifact_bytes",
            "max_artifact_total_bytes",
            "max_artifact_candidates",
            "stdout_bytes",
            "stderr_bytes",
            "stdout_truncated",
            "stderr_truncated",
            "output_truncated",
            "artifact_scanned_entries",
            "artifact_scanned_files",
            "artifact_candidate_count",
            "artifact_exported_count",
            "artifact_exported_bytes",
            "artifact_skipped_count",
            "artifact_skipped_too_large_count",
            "artifact_skipped_count_limit",
            "artifact_skipped_total_bytes_limit",
            "artifact_skipped_unreadable_count",
            "artifact_skipped_unreadable_dirs",
            "artifact_scan_truncated",
            "artifact_collection_truncated",
            "docker_image",
            "docker_returncode",
        ):
            if key in runtime_metadata:
                metadata[key] = runtime_metadata[key]

        code = self._code_execution_event_code(status=status, error=error)
        try:
            append_runtime_event(
                kind="code_execution",
                code=code,
                message=f"Code execution job {status}.",
                request_id=record.request_id,
                metadata=metadata,
            )
        except OSError:
            append_job_log(
                record,
                "runtime_event_warning",
                "Code execution runtime event could not be written.",
                error_code="runtime_event_log_failed",
            )

    def _start_with_create_result(
        self,
        kind: JobKind,
        request: dict[str, Any],
        request_id: str | None,
        *,
        idempotency_key: str | None = None,
        max_attempts: int = 1,
        retry_backoff_s: int = 0,
        ownership: dict[str, str] | None = None,
    ) -> tuple[JobRecord, bool]:
        now = utc_now()
        owner = normalize_ownership(**(ownership or {}))
        record = JobRecord(
            job_id=new_request_id(),
            kind=kind,
            status="running",
            created_at=now,
            updated_at=now,
            request=request,
            attempts=1,
            request_id=request_id,
            idempotency_key=normalize_idempotency_key(idempotency_key),
            max_attempts=max(max_attempts, 1),
            retry_backoff_s=max(retry_backoff_s, 0),
            owner_id=owner["owner_id"],
            owner_label=owner["owner_label"],
            ownership_source=owner["ownership_source"],
        )
        append_job_log(record, "running", f"{kind} job started.", **owner)
        return self.store.append_new(record)

    def _start(
        self,
        kind: JobKind,
        request: dict[str, Any],
        request_id: str | None,
        *,
        idempotency_key: str | None = None,
        max_attempts: int = 1,
        retry_backoff_s: int = 0,
        ownership: dict[str, str] | None = None,
    ) -> JobRecord:
        record, _created = self._start_with_create_result(
            kind,
            request,
            request_id,
            idempotency_key=idempotency_key,
            max_attempts=max_attempts,
            retry_backoff_s=retry_backoff_s,
            ownership=ownership,
        )
        return record

    def _enqueue_with_create_result(
        self,
        kind: JobKind,
        request: dict[str, Any],
        request_id: str | None,
        *,
        parent_job_id: str | None = None,
        not_before: str | None = None,
        deadline_at: str | None = None,
        idempotency_key: str | None = None,
        max_attempts: int = 1,
        retry_backoff_s: int = 0,
        ownership: dict[str, str] | None = None,
    ) -> tuple[JobRecord, bool]:
        now = utc_now()
        clean_idempotency_key = normalize_idempotency_key(idempotency_key)
        owner = normalize_ownership(**(ownership or {}))
        record = JobRecord(
            job_id=new_request_id(),
            kind=kind,
            status="queued",
            created_at=now,
            updated_at=now,
            request=request,
            attempts=0,
            request_id=request_id,
            parent_job_id=parent_job_id,
            not_before=not_before,
            deadline_at=deadline_at,
            idempotency_key=clean_idempotency_key,
            max_attempts=max(max_attempts, 1),
            retry_backoff_s=max(retry_backoff_s, 0),
            owner_id=owner["owner_id"],
            owner_label=owner["owner_label"],
            ownership_source=owner["ownership_source"],
        )
        append_job_log(record, "queued", f"{kind} job queued.", **owner)
        return self.store.append_new(record)

    def _enqueue(
        self,
        kind: JobKind,
        request: dict[str, Any],
        request_id: str | None,
        *,
        parent_job_id: str | None = None,
        not_before: str | None = None,
        deadline_at: str | None = None,
        idempotency_key: str | None = None,
        max_attempts: int = 1,
        retry_backoff_s: int = 0,
        ownership: dict[str, str] | None = None,
    ) -> JobRecord:
        record, _created = self._enqueue_with_create_result(
            kind,
            request,
            request_id,
            parent_job_id=parent_job_id,
            not_before=not_before,
            deadline_at=deadline_at,
            idempotency_key=idempotency_key,
            max_attempts=max_attempts,
            retry_backoff_s=retry_backoff_s,
            ownership=ownership,
        )
        return record

    def _mark_running(self, record: JobRecord) -> JobRecord:
        apply_job_ownership(record)
        record.status = "running"
        record.updated_at = utc_now()
        record.attempts += 1
        append_job_log(record, "running", f"{record.kind} job started.", **ownership_from_record(record))
        self.store.append(record)
        return record

    def _finish(
        self,
        record: JobRecord,
        *,
        status: JobStatus,
        result: dict[str, Any] | None = None,
        artifacts: list[GeneratedArtifact] | None = None,
        error: dict[str, Any] | None = None,
    ) -> JobRecord:
        apply_job_ownership(record)
        record.status = status
        record.updated_at = utc_now()
        record.result = result
        record.artifacts = [asdict(artifact) for artifact in artifacts or []]
        record.error = error
        metadata: dict[str, Any] = {"artifact_count": len(record.artifacts)}
        if result and "exit_code" in result:
            metadata["exit_code"] = result["exit_code"]
        if error:
            metadata["error_code"] = error.get("code")
        append_job_log(record, status, f"{record.kind} job {status}.", **metadata)
        if status == "failed" and self._should_auto_retry(record, error=error):
            record.status = "queued"
            record.not_before = future_utc(record.retry_backoff_s)
            record.worker_id = None
            record.leased_at = None
            record.lease_expires_at = None
            record.updated_at = utc_now()
            append_job_log(
                record,
                "queued",
                f"{record.kind} job scheduled for automatic retry.",
                attempt=record.attempts,
                max_attempts=record.max_attempts,
                retry_backoff_s=record.retry_backoff_s,
            )
        elif status == "failed" and record.max_attempts > 1 and record.attempts >= record.max_attempts:
            record.status = "dead_lettered"
            record.dead_lettered_at = utc_now()
            record.updated_at = record.dead_lettered_at
            append_job_log(
                record,
                "dead_lettered",
                f"{record.kind} job moved to dead letter after retry policy was exhausted.",
                attempts=record.attempts,
                max_attempts=record.max_attempts,
                error_code=error.get("code") if error else None,
            )
        self.store.append(record)
        return record

    @staticmethod
    def _should_auto_retry(record: JobRecord, *, error: dict[str, Any] | None = None) -> bool:
        if record.max_attempts <= 1:
            return False
        if record.attempts >= record.max_attempts:
            return False
        if error and error.get("code") == "job_deadline_exceeded":
            return False
        return True

    def run_mock_image(
        self,
        request: ImageGenerationRequest,
        *,
        request_id: str | None = None,
        idempotency_key: str | None = None,
        ownership: dict[str, str] | None = None,
        record: JobRecord | None = None,
        cancel_event: threading.Event | None = None,
    ) -> JobRecord:
        if record:
            record = self._mark_running(record)
        else:
            record, created = self._start_with_create_result(
                "image_generation",
                asdict(request),
                request_id,
                idempotency_key=idempotency_key,
                ownership=ownership,
            )
            if not created:
                return record
        try:
            if cancel_event and cancel_event.is_set():
                return self._finish(
                    record,
                    status="cancelled",
                    error={"code": "cancelled", "message": "Job was cancelled."},
                )
            artifact = MockImageGenerationProvider(self._artifact_store()).generate(request)
        except Exception as exc:
            error = normalize_exception(exc)
            return self._finish(record, status="failed", error=asdict(error))
        return self._finish(record, status="succeeded", artifacts=[artifact])

    def run_local_python(
        self,
        request: CodeExecutionRequest,
        *,
        request_id: str | None = None,
        idempotency_key: str | None = None,
        ownership: dict[str, str] | None = None,
        record: JobRecord | None = None,
        cancel_event: threading.Event | None = None,
    ) -> JobRecord:
        if record:
            record = self._mark_running(record)
        else:
            record, created = self._start_with_create_result(
                "code_execution",
                asdict(request),
                request_id,
                idempotency_key=idempotency_key,
                ownership=ownership,
            )
            if not created:
                return record
        try:
            started = time.monotonic()
            result = self._code_execution_provider("python").run(
                request,
                cancel_event=cancel_event,
            )
        except Exception as exc:
            error = normalize_exception(exc)
            self._record_code_execution_event(
                record,
                request,
                status="failed",
                result=None,
                error=asdict(error),
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            return self._finish(record, status="failed", error=asdict(error))

        if cancel_event and cancel_event.is_set():
            status: JobStatus = "cancelled"
            error = {"code": "cancelled", "message": result.stderr or "Job was cancelled."}
        else:
            status = "succeeded" if result.success else "failed"
            if result.success:
                error = None
            elif result.runtime_metadata.get("policy_violation") == "true":
                error = {"code": "execution_policy_violation", "message": result.stderr}
            elif result.exit_code == 127:
                error = {"code": "runtime_unavailable", "message": result.stderr}
            elif result.exit_code == 124:
                error = {"code": "execution_timeout", "message": result.stderr}
            else:
                error = {"code": "execution_failed", "message": result.stderr}
        payload = asdict(result)
        self._record_code_execution_event(
            record,
            request,
            status=status,
            result=result,
            error=error,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        return self._finish(
            record,
            status=status,
            result=payload,
            artifacts=result.artifacts,
            error=error,
        )

    def run_local_octave(
        self,
        request: CodeExecutionRequest,
        *,
        request_id: str | None = None,
        idempotency_key: str | None = None,
        ownership: dict[str, str] | None = None,
        record: JobRecord | None = None,
        cancel_event: threading.Event | None = None,
    ) -> JobRecord:
        if record:
            record = self._mark_running(record)
        else:
            record, created = self._start_with_create_result(
                "code_execution",
                asdict(request),
                request_id,
                idempotency_key=idempotency_key,
                ownership=ownership,
            )
            if not created:
                return record
        try:
            started = time.monotonic()
            result = self._code_execution_provider("octave").run(
                request,
                cancel_event=cancel_event,
            )
        except Exception as exc:
            error = normalize_exception(exc)
            self._record_code_execution_event(
                record,
                request,
                status="failed",
                result=None,
                error=asdict(error),
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            return self._finish(record, status="failed", error=asdict(error))

        if cancel_event and cancel_event.is_set():
            status: JobStatus = "cancelled"
            error = {"code": "cancelled", "message": result.stderr or "Job was cancelled."}
        else:
            status = "succeeded" if result.success else "failed"
            if result.success:
                error = None
            elif result.runtime_metadata.get("policy_violation") == "true":
                error = {"code": "execution_policy_violation", "message": result.stderr}
            elif result.exit_code == 127:
                error = {"code": "runtime_unavailable", "message": result.stderr}
            elif result.exit_code == 124:
                error = {"code": "execution_timeout", "message": result.stderr}
            else:
                error = {"code": "execution_failed", "message": result.stderr}
        payload = asdict(result)
        self._record_code_execution_event(
            record,
            request,
            status=status,
            result=result,
            error=error,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        return self._finish(
            record,
            status=status,
            result=payload,
            artifacts=result.artifacts,
            error=error,
        )

    def run_local_code(
        self,
        request: CodeExecutionRequest,
        *,
        request_id: str | None = None,
        idempotency_key: str | None = None,
        ownership: dict[str, str] | None = None,
        record: JobRecord | None = None,
        cancel_event: threading.Event | None = None,
    ) -> JobRecord:
        if request.language == "octave":
            return self.run_local_octave(
                request,
                request_id=request_id,
                idempotency_key=idempotency_key,
                ownership=ownership,
                record=record,
                cancel_event=cancel_event,
            )
        return self.run_local_python(
            request,
            request_id=request_id,
            idempotency_key=idempotency_key,
            ownership=ownership,
            record=record,
            cancel_event=cancel_event,
        )

    def run_index_rebuild(
        self,
        source_paths: list[str],
        *,
        request_id: str | None = None,
        idempotency_key: str | None = None,
        ownership: dict[str, str] | None = None,
        record: JobRecord | None = None,
        cancel_event: threading.Event | None = None,
    ) -> JobRecord:
        request = {"source_paths": source_paths}
        if record:
            record = self._mark_running(record)
        else:
            record, created = self._start_with_create_result(
                "index_rebuild",
                request,
                request_id,
                idempotency_key=idempotency_key,
                ownership=ownership,
            )
            if not created:
                return record
        try:
            if cancel_event and cancel_event.is_set():
                return self._finish(
                    record,
                    status="cancelled",
                    error={"code": "cancelled", "message": "Job was cancelled."},
                )
            paths = [self._resolve_pdf_path(source_path) for source_path in source_paths]
            _store, chunks = ingestion.rebuild_vector_store_from_pdfs(paths, cancel_event=cancel_event)
        except ingestion.IngestionCancelled as exc:
            return self._finish(
                record,
                status="cancelled",
                error={"code": "cancelled", "message": str(exc)},
            )
        except Exception as exc:
            error = normalize_exception(exc)
            return self._finish(record, status="failed", error=asdict(error))

        return self._finish(
            record,
            status="succeeded",
            result={
                "source_paths": [path.resolve().relative_to(PROJECT_ROOT).as_posix() for path in paths],
                "paper_count": len(paths),
                "chunk_count": chunks,
            },
        )

    def retry(self, job_id: str, *, request_id: str | None = None) -> JobRecord | None:
        job = self.store.get(job_id)
        if job is None:
            return None
        if job.status not in {"failed", "cancelled", "dead_lettered"}:
            return job
        if job.kind == "image_generation":
            return self.run_mock_image(
                ImageGenerationRequest(**job.request),
                request_id=request_id,
                ownership={**ownership_from_record(job), "ownership_source": "inherited"},
            )
        if job.kind == "code_execution":
            return self.run_local_code(
                CodeExecutionRequest(**job.request),
                request_id=request_id,
                ownership={**ownership_from_record(job), "ownership_source": "inherited"},
            )
        if job.kind == "index_rebuild":
            return self.run_index_rebuild(
                job.request.get("source_paths", []),
                request_id=request_id,
                ownership={**ownership_from_record(job), "ownership_source": "inherited"},
            )
        return job

    def schedule_retry(
        self,
        job_id: str,
        *,
        delay_s: int = 30,
        queue_timeout_s: int | None = None,
        request_id: str | None = None,
    ) -> JobRecord | None:
        job = self.store.get(job_id)
        if job is None:
            return None
        if job.status not in {"failed", "cancelled", "dead_lettered"}:
            return job
        return self._enqueue(
            job.kind,
            job.request,
            request_id,
            parent_job_id=job.job_id,
            not_before=future_utc(delay_s),
            deadline_at=future_utc(queue_timeout_s) if queue_timeout_s is not None else None,
            ownership={**ownership_from_record(job), "ownership_source": "inherited"},
        )

    @staticmethod
    def _resolve_pdf_path(source_path: str) -> Path:
        path = (PROJECT_ROOT / source_path).resolve()
        try:
            path.relative_to(PROJECT_ROOT)
        except ValueError as exc:
            raise ValueError(f"PDF path escapes project root: {source_path}") from exc
        if path.suffix.lower() != ".pdf" or not path.exists():
            raise ValueError(f"PDF path is not selectable: {source_path}")
        selectable = {item.resolve() for item in ingestion.discover_pdfs()}
        if path not in selectable:
            raise ValueError(f"PDF path is not in the selectable corpus: {source_path}")
        return path


class AsyncJobManager:
    """In-process background queue for local no-key jobs."""

    def __init__(
        self,
        store: LocalJobStore | None = None,
        *,
        recover_existing: bool = True,
        worker_id: str | None = None,
        lease_seconds: int = 3600,
        artifact_root: Path | None = None,
    ):
        self.store = store or LocalJobStore()
        self.runner = LocalJobRunner(self.store, artifact_root=artifact_root)
        self.worker_id = worker_id or f"in-process-{new_request_id()}"
        self.lease_seconds = lease_seconds
        self._queue: queue.Queue[str] = queue.Queue()
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        if recover_existing:
            self.recover_queued_jobs()

    def recover_queued_jobs(self) -> int:
        """Rehydrate queued/scheduled jobs from SQLite/JSONL after restart."""
        recovered = 0
        for record in self.store.list_queued(limit=10000):
            with self._lock:
                if record.job_id in self._events:
                    continue
            self._enqueue(record)
            recovered += 1
        return recovered

    def enqueue_mock_image(
        self,
        request: ImageGenerationRequest,
        *,
        queue_timeout_s: int | None = None,
        request_id: str | None = None,
        idempotency_key: str | None = None,
        max_attempts: int = 1,
        retry_backoff_s: int = 0,
        ownership: dict[str, str] | None = None,
    ) -> JobRecord:
        existing = self.store.find_by_idempotency_key(
            kind="image_generation",
            key=idempotency_key,
        )
        if existing is not None:
            return existing
        record, created = self.runner._enqueue_with_create_result(
            "image_generation",
            asdict(request),
            request_id,
            deadline_at=future_utc(queue_timeout_s) if queue_timeout_s is not None else None,
            idempotency_key=idempotency_key,
            max_attempts=max_attempts,
            retry_backoff_s=retry_backoff_s,
            ownership=ownership,
        )
        if created:
            self._enqueue(record)
        return record

    def enqueue_local_python(
        self,
        request: CodeExecutionRequest,
        *,
        queue_timeout_s: int | None = None,
        request_id: str | None = None,
        idempotency_key: str | None = None,
        max_attempts: int = 1,
        retry_backoff_s: int = 0,
        ownership: dict[str, str] | None = None,
    ) -> JobRecord:
        existing = self.store.find_by_idempotency_key(
            kind="code_execution",
            key=idempotency_key,
        )
        if existing is not None:
            return existing
        record, created = self.runner._enqueue_with_create_result(
            "code_execution",
            asdict(request),
            request_id,
            deadline_at=future_utc(queue_timeout_s) if queue_timeout_s is not None else None,
            idempotency_key=idempotency_key,
            max_attempts=max_attempts,
            retry_backoff_s=retry_backoff_s,
            ownership=ownership,
        )
        if created:
            self._enqueue(record)
        return record

    def enqueue_local_octave(
        self,
        request: CodeExecutionRequest,
        *,
        queue_timeout_s: int | None = None,
        request_id: str | None = None,
        idempotency_key: str | None = None,
        max_attempts: int = 1,
        retry_backoff_s: int = 0,
        ownership: dict[str, str] | None = None,
    ) -> JobRecord:
        existing = self.store.find_by_idempotency_key(
            kind="code_execution",
            key=idempotency_key,
        )
        if existing is not None:
            return existing
        record, created = self.runner._enqueue_with_create_result(
            "code_execution",
            asdict(request),
            request_id,
            deadline_at=future_utc(queue_timeout_s) if queue_timeout_s is not None else None,
            idempotency_key=idempotency_key,
            max_attempts=max_attempts,
            retry_backoff_s=retry_backoff_s,
            ownership=ownership,
        )
        if created:
            self._enqueue(record)
        return record

    def enqueue_index_rebuild(
        self,
        source_paths: list[str],
        *,
        queue_timeout_s: int | None = None,
        request_id: str | None = None,
        idempotency_key: str | None = None,
        max_attempts: int = 1,
        retry_backoff_s: int = 0,
        ownership: dict[str, str] | None = None,
    ) -> JobRecord:
        existing = self.store.find_by_idempotency_key(
            kind="index_rebuild",
            key=idempotency_key,
        )
        if existing is not None:
            return existing
        record, created = self.runner._enqueue_with_create_result(
            "index_rebuild",
            {"source_paths": source_paths},
            request_id,
            deadline_at=future_utc(queue_timeout_s) if queue_timeout_s is not None else None,
            idempotency_key=idempotency_key,
            max_attempts=max_attempts,
            retry_backoff_s=retry_backoff_s,
            ownership=ownership,
        )
        if created:
            self._enqueue(record)
        return record

    def schedule_retry(
        self,
        job_id: str,
        *,
        delay_s: int = 30,
        queue_timeout_s: int | None = None,
        request_id: str | None = None,
    ) -> JobRecord | None:
        record = self.runner.schedule_retry(
            job_id,
            delay_s=delay_s,
            queue_timeout_s=queue_timeout_s,
            request_id=request_id,
        )
        if record is None or record.status != "queued":
            return record
        self._enqueue(record)
        return record

    def cancel(self, job_id: str) -> JobRecord | None:
        with self._lock:
            event = self._events.get(job_id)
            if event:
                event.set()
        return self.store.cancel(job_id)

    def _enqueue(self, record: JobRecord) -> None:
        with self._lock:
            self._events[record.job_id] = threading.Event()
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work_loop, daemon=True)
                self._worker.start()
        delay_s = self._delay_seconds(record)
        if delay_s > 0:
            timer = threading.Timer(delay_s, self._put_job, args=(record.job_id,))
            timer.daemon = True
            timer.start()
        else:
            self._put_job(record.job_id)

    def _put_job(self, job_id: str) -> None:
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work_loop, daemon=True)
                self._worker.start()
        self._queue.put(job_id)

    @staticmethod
    def _delay_seconds(record: JobRecord) -> float:
        if not record.not_before:
            return 0
        return max(0.0, (parse_utc(record.not_before) - datetime.now(timezone.utc)).total_seconds())

    def _work_loop(self) -> None:
        while True:
            try:
                job_id = self._queue.get(timeout=1)
            except queue.Empty:
                return
            try:
                self._run_job(job_id)
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        record = self.store.claim_job(
            job_id,
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
        )
        if record is None:
            return
        event = self._events.get(job_id)
        if record.status == "cancelled" or (event and event.is_set()):
            self.store.cancel(job_id)
            return
        if has_deadline_expired(record):
            result = self.runner._finish(
                record,
                status="failed",
                error={
                    "code": "job_deadline_exceeded",
                    "message": "Job deadline passed before execution.",
                },
            )
            self._requeue_if_needed(result)
            return
        if record.kind == "image_generation":
            result = self.runner.run_mock_image(
                ImageGenerationRequest(**record.request),
                request_id=record.request_id,
                record=record,
                cancel_event=event,
            )
        elif record.kind == "code_execution":
            result = self.runner.run_local_code(
                CodeExecutionRequest(**record.request),
                request_id=record.request_id,
                record=record,
                cancel_event=event,
            )
        elif record.kind == "index_rebuild":
            result = self.runner.run_index_rebuild(
                record.request.get("source_paths", []),
                request_id=record.request_id,
                record=record,
                cancel_event=event,
            )
        else:
            result = None
        self._requeue_if_needed(result)

    def _requeue_if_needed(self, record: JobRecord | None) -> None:
        if record is not None and record.status == "queued":
            self._enqueue(record)


class LocalDurableJobWorker:
    """Explicit durable worker loop for future out-of-process local jobs."""

    def __init__(
        self,
        store: LocalJobStore | None = None,
        *,
        worker_id: str | None = None,
        lease_seconds: int = 3600,
        cancel_poll_interval_s: float = 0.25,
        artifact_root: Path | None = None,
    ):
        self.store = store or LocalJobStore()
        self.runner = LocalJobRunner(self.store, artifact_root=artifact_root)
        self.worker_id = worker_id or f"durable-{new_request_id()}"
        self.lease_seconds = lease_seconds
        self.cancel_poll_interval_s = cancel_poll_interval_s

    def run_once(self) -> JobRecord | None:
        """Claim and execute one due queued job from durable state."""
        record = self.store.claim_next_due_job(
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
        )
        if record is None:
            return None
        return self._run_claimed_job(record)

    def run_until_empty(self, *, max_jobs: int | None = None) -> list[JobRecord]:
        """Run due jobs until no claimable job remains or max_jobs is reached."""
        results: list[JobRecord] = []
        while max_jobs is None or len(results) < max_jobs:
            result = self.run_once()
            if result is None:
                break
            results.append(result)
        return results

    def run_polling(
        self,
        *,
        poll_interval_s: float = 2.0,
        max_jobs: int | None = None,
    ) -> list[JobRecord]:
        """Poll durable state for due jobs; intended for manual/future service use."""
        results: list[JobRecord] = []
        while max_jobs is None or len(results) < max_jobs:
            result = self.run_once()
            if result is None:
                time.sleep(max(poll_interval_s, 0.1))
                continue
            results.append(result)
        return results

    def _run_claimed_job(self, record: JobRecord) -> JobRecord:
        if has_deadline_expired(record):
            return self.runner._finish(
                record,
                status="failed",
                error={
                    "code": "job_deadline_exceeded",
                    "message": "Job deadline passed before execution.",
                },
            )
        cancel_event = threading.Event()
        stop_monitor = threading.Event()
        monitor = threading.Thread(
            target=self._monitor_cancellation,
            args=(record.job_id, cancel_event, stop_monitor),
            daemon=True,
        )
        monitor.start()
        try:
            if record.kind == "image_generation":
                return self.runner.run_mock_image(
                    ImageGenerationRequest(**record.request),
                    request_id=record.request_id,
                    record=record,
                    cancel_event=cancel_event,
                )
            if record.kind == "code_execution":
                return self.runner.run_local_code(
                    CodeExecutionRequest(**record.request),
                    request_id=record.request_id,
                    record=record,
                    cancel_event=cancel_event,
                )
            if record.kind == "index_rebuild":
                return self.runner.run_index_rebuild(
                    record.request.get("source_paths", []),
                    request_id=record.request_id,
                    record=record,
                    cancel_event=cancel_event,
                )
            return record
        finally:
            stop_monitor.set()
            monitor.join(timeout=1)

    def _monitor_cancellation(
        self,
        job_id: str,
        cancel_event: threading.Event,
        stop_monitor: threading.Event,
    ) -> None:
        while not stop_monitor.is_set() and not cancel_event.is_set():
            record = self.store.get(job_id)
            if record and record.status == "cancelled":
                cancel_event.set()
                return
            time.sleep(max(self.cancel_poll_interval_s, 0.05))


_ASYNC_JOB_MANAGER: AsyncJobManager | None = None


def get_async_job_manager() -> AsyncJobManager:
    global _ASYNC_JOB_MANAGER
    if _ASYNC_JOB_MANAGER is None:
        _ASYNC_JOB_MANAGER = AsyncJobManager()
    return _ASYNC_JOB_MANAGER
