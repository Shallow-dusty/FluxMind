"""Local job records for no-key artifact and execution workflows."""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
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
from src.config import JOBS_FILE, PROJECT_ROOT
from src import ingestion
from src.providers import (
    LocalOctaveExecutionProvider,
    LocalPythonExecutionProvider,
    MockImageGenerationProvider,
)
from src.runtime import new_request_id, normalize_exception


JobKind = Literal["image_generation", "code_execution", "index_rebuild"]
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def future_utc(seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=max(seconds, 0))).isoformat()


def has_deadline_expired(record: "JobRecord") -> bool:
    return bool(record.deadline_at and parse_utc(record.deadline_at) <= datetime.now(timezone.utc))


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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        self._upsert_sqlite(record)

    def get(self, job_id: str) -> JobRecord | None:
        self._ensure_sqlite()
        record = self._get_sqlite(job_id)
        if record is not None:
            return record
        return self._get_jsonl(job_id)

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
        return JobRecord(**latest) if latest else None

    def list_latest(
        self,
        *,
        limit: int = 50,
        status: str | None = None,
        kind: str | None = None,
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
            return self._filter_records(records, status=status, kind=kind, q=q)[:limit]
        records = self._list_latest_jsonl(limit=max(limit, 1000))
        return self._filter_records(records, status=status, kind=kind, q=q)[:limit]

    @staticmethod
    def _filter_records(
        records: list[JobRecord],
        *,
        status: str | None = None,
        kind: str | None = None,
        q: str | None = None,
    ) -> list[JobRecord]:
        status = status.strip() if status else None
        kind = kind.strip() if kind else None
        query = (q or "").strip().casefold()
        filtered: list[JobRecord] = []
        for record in records:
            if status and record.status != status:
                continue
            if kind and record.kind != kind:
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
        records.sort(key=lambda record: record.updated_at, reverse=True)
        return records[:limit]

    def cancel(self, job_id: str) -> JobRecord | None:
        record = self.get(job_id)
        if record is None:
            return None
        if record.status in {"succeeded", "failed", "cancelled"}:
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_kind ON jobs(kind)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_not_before ON jobs(not_before)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_lease_expires_at ON jobs(lease_expires_at)")
        if self.path.exists():
            for record in self._list_latest_jsonl(limit=10000):
                self._upsert_sqlite(record)

    def _upsert_sqlite(self, record: JobRecord) -> None:
        with self._connect() as conn:
            self._create_jobs_table(conn)
            self._ensure_jobs_columns(conn)
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, kind, status, created_at, updated_at, request_id,
                    attempts, not_before, deadline_at, worker_id, leased_at,
                    lease_expires_at, payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    json.dumps(asdict(record), ensure_ascii=False),
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
        }.items():
            if column not in columns:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {definition}")

    def _append_jsonl(self, record: JobRecord) -> None:
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
        return JobRecord(**json.loads(row["payload"]))


class LocalJobRunner:
    """Run no-key providers and persist job transitions."""

    def __init__(self, store: LocalJobStore | None = None):
        self.store = store or LocalJobStore()

    def _start(self, kind: JobKind, request: dict[str, Any], request_id: str | None) -> JobRecord:
        now = utc_now()
        record = JobRecord(
            job_id=new_request_id(),
            kind=kind,
            status="running",
            created_at=now,
            updated_at=now,
            request=request,
            attempts=1,
            request_id=request_id,
        )
        append_job_log(record, "running", f"{kind} job started.")
        self.store.append(record)
        return record

    def _enqueue(
        self,
        kind: JobKind,
        request: dict[str, Any],
        request_id: str | None,
        *,
        parent_job_id: str | None = None,
        not_before: str | None = None,
        deadline_at: str | None = None,
    ) -> JobRecord:
        now = utc_now()
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
        )
        append_job_log(record, "queued", f"{kind} job queued.")
        self.store.append(record)
        return record

    def _mark_running(self, record: JobRecord) -> JobRecord:
        record.status = "running"
        record.updated_at = utc_now()
        record.attempts += 1
        append_job_log(record, "running", f"{record.kind} job started.")
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
        self.store.append(record)
        return record

    def run_mock_image(
        self,
        request: ImageGenerationRequest,
        *,
        request_id: str | None = None,
        record: JobRecord | None = None,
        cancel_event: threading.Event | None = None,
    ) -> JobRecord:
        record = self._mark_running(record) if record else self._start("image_generation", asdict(request), request_id)
        try:
            if cancel_event and cancel_event.is_set():
                return self._finish(
                    record,
                    status="cancelled",
                    error={"code": "cancelled", "message": "Job was cancelled."},
                )
            artifact = MockImageGenerationProvider().generate(request)
        except Exception as exc:
            error = normalize_exception(exc)
            return self._finish(record, status="failed", error=asdict(error))
        return self._finish(record, status="succeeded", artifacts=[artifact])

    def run_local_python(
        self,
        request: CodeExecutionRequest,
        *,
        request_id: str | None = None,
        record: JobRecord | None = None,
        cancel_event: threading.Event | None = None,
    ) -> JobRecord:
        record = self._mark_running(record) if record else self._start("code_execution", asdict(request), request_id)
        try:
            result = LocalPythonExecutionProvider().run(request, cancel_event=cancel_event)
        except Exception as exc:
            error = normalize_exception(exc)
            return self._finish(record, status="failed", error=asdict(error))

        if cancel_event and cancel_event.is_set():
            status: JobStatus = "cancelled"
            error = {"code": "cancelled", "message": result.stderr or "Job was cancelled."}
        else:
            status = "succeeded" if result.success else "failed"
            if result.success:
                error = None
            elif result.exit_code == 127:
                error = {"code": "runtime_unavailable", "message": result.stderr}
            elif result.exit_code == 124:
                error = {"code": "execution_timeout", "message": result.stderr}
            else:
                error = {"code": "execution_failed", "message": result.stderr}
        payload = asdict(result)
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
        record: JobRecord | None = None,
        cancel_event: threading.Event | None = None,
    ) -> JobRecord:
        record = self._mark_running(record) if record else self._start("code_execution", asdict(request), request_id)
        try:
            result = LocalOctaveExecutionProvider().run(request, cancel_event=cancel_event)
        except Exception as exc:
            error = normalize_exception(exc)
            return self._finish(record, status="failed", error=asdict(error))

        if cancel_event and cancel_event.is_set():
            status: JobStatus = "cancelled"
            error = {"code": "cancelled", "message": result.stderr or "Job was cancelled."}
        else:
            status = "succeeded" if result.success else "failed"
            if result.success:
                error = None
            elif result.exit_code == 127:
                error = {"code": "runtime_unavailable", "message": result.stderr}
            elif result.exit_code == 124:
                error = {"code": "execution_timeout", "message": result.stderr}
            else:
                error = {"code": "execution_failed", "message": result.stderr}
        payload = asdict(result)
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
        record: JobRecord | None = None,
        cancel_event: threading.Event | None = None,
    ) -> JobRecord:
        if request.language == "octave":
            return self.run_local_octave(
                request,
                request_id=request_id,
                record=record,
                cancel_event=cancel_event,
            )
        return self.run_local_python(
            request,
            request_id=request_id,
            record=record,
            cancel_event=cancel_event,
        )

    def run_index_rebuild(
        self,
        source_paths: list[str],
        *,
        request_id: str | None = None,
        record: JobRecord | None = None,
        cancel_event: threading.Event | None = None,
    ) -> JobRecord:
        request = {"source_paths": source_paths}
        record = self._mark_running(record) if record else self._start("index_rebuild", request, request_id)
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
        if job.status not in {"failed", "cancelled"}:
            return job
        if job.kind == "image_generation":
            return self.run_mock_image(ImageGenerationRequest(**job.request), request_id=request_id)
        if job.kind == "code_execution":
            return self.run_local_code(CodeExecutionRequest(**job.request), request_id=request_id)
        if job.kind == "index_rebuild":
            return self.run_index_rebuild(job.request.get("source_paths", []), request_id=request_id)
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
        if job.status not in {"failed", "cancelled"}:
            return job
        return self._enqueue(
            job.kind,
            job.request,
            request_id,
            parent_job_id=job.job_id,
            not_before=future_utc(delay_s),
            deadline_at=future_utc(queue_timeout_s) if queue_timeout_s is not None else None,
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
    ):
        self.store = store or LocalJobStore()
        self.runner = LocalJobRunner(self.store)
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
    ) -> JobRecord:
        record = self.runner._enqueue(
            "image_generation",
            asdict(request),
            request_id,
            deadline_at=future_utc(queue_timeout_s) if queue_timeout_s is not None else None,
        )
        self._enqueue(record)
        return record

    def enqueue_local_python(
        self,
        request: CodeExecutionRequest,
        *,
        queue_timeout_s: int | None = None,
        request_id: str | None = None,
    ) -> JobRecord:
        record = self.runner._enqueue(
            "code_execution",
            asdict(request),
            request_id,
            deadline_at=future_utc(queue_timeout_s) if queue_timeout_s is not None else None,
        )
        self._enqueue(record)
        return record

    def enqueue_local_octave(
        self,
        request: CodeExecutionRequest,
        *,
        queue_timeout_s: int | None = None,
        request_id: str | None = None,
    ) -> JobRecord:
        record = self.runner._enqueue(
            "code_execution",
            asdict(request),
            request_id,
            deadline_at=future_utc(queue_timeout_s) if queue_timeout_s is not None else None,
        )
        self._enqueue(record)
        return record

    def enqueue_index_rebuild(
        self,
        source_paths: list[str],
        *,
        queue_timeout_s: int | None = None,
        request_id: str | None = None,
    ) -> JobRecord:
        record = self.runner._enqueue(
            "index_rebuild",
            {"source_paths": source_paths},
            request_id,
            deadline_at=future_utc(queue_timeout_s) if queue_timeout_s is not None else None,
        )
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
            self.runner._finish(
                record,
                status="failed",
                error={
                    "code": "job_deadline_exceeded",
                    "message": "Job deadline passed before execution.",
                },
            )
            return
        if record.kind == "image_generation":
            self.runner.run_mock_image(
                ImageGenerationRequest(**record.request),
                request_id=record.request_id,
                record=record,
                cancel_event=event,
            )
        elif record.kind == "code_execution":
            self.runner.run_local_code(
                CodeExecutionRequest(**record.request),
                request_id=record.request_id,
                record=record,
                cancel_event=event,
            )
        elif record.kind == "index_rebuild":
            self.runner.run_index_rebuild(
                record.request.get("source_paths", []),
                request_id=record.request_id,
                record=record,
                cancel_event=event,
            )


class LocalDurableJobWorker:
    """Explicit durable worker loop for future out-of-process local jobs."""

    def __init__(
        self,
        store: LocalJobStore | None = None,
        *,
        worker_id: str | None = None,
        lease_seconds: int = 3600,
        cancel_poll_interval_s: float = 0.25,
    ):
        self.store = store or LocalJobStore()
        self.runner = LocalJobRunner(self.store)
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
