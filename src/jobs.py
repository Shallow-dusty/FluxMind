"""Local job records for no-key artifact and execution workflows."""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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
from src.providers import LocalPythonExecutionProvider, MockImageGenerationProvider
from src.runtime import new_request_id, normalize_exception


JobKind = Literal["image_generation", "code_execution", "index_rebuild"]
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    def list_latest(self, *, limit: int = 50) -> list[JobRecord]:
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
                    (limit,),
                ).fetchall()
            return [JobRecord(**json.loads(row["payload"])) for row in rows]
        return self._list_latest_jsonl(limit=limit)

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
        self.append(record)
        return record

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_sqlite(self) -> None:
        with self._connect() as conn:
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
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_kind ON jobs(kind)")
        if self.path.exists():
            for record in self._list_latest_jsonl(limit=10000):
                self._upsert_sqlite(record)

    def _upsert_sqlite(self, record: JobRecord) -> None:
        with self._connect() as conn:
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
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, kind, status, created_at, updated_at, request_id,
                    attempts, payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    kind=excluded.kind,
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    request_id=excluded.request_id,
                    attempts=excluded.attempts,
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
                    json.dumps(asdict(record), ensure_ascii=False),
                ),
            )

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
        self.store.append(record)
        return record

    def _enqueue(self, kind: JobKind, request: dict[str, Any], request_id: str | None) -> JobRecord:
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
        )
        self.store.append(record)
        return record

    def _mark_running(self, record: JobRecord) -> JobRecord:
        record.status = "running"
        record.updated_at = utc_now()
        record.attempts += 1
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
            error = None if result.success else {"code": "execution_failed", "message": result.stderr}
        payload = asdict(result)
        return self._finish(
            record,
            status=status,
            result=payload,
            artifacts=result.artifacts,
            error=error,
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
            _store, chunks = ingestion.rebuild_vector_store_from_pdfs(paths)
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
            return self.run_local_python(CodeExecutionRequest(**job.request), request_id=request_id)
        if job.kind == "index_rebuild":
            return self.run_index_rebuild(job.request.get("source_paths", []), request_id=request_id)
        return job

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

    def __init__(self, store: LocalJobStore | None = None):
        self.store = store or LocalJobStore()
        self.runner = LocalJobRunner(self.store)
        self._queue: queue.Queue[str] = queue.Queue()
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def enqueue_mock_image(
        self,
        request: ImageGenerationRequest,
        *,
        request_id: str | None = None,
    ) -> JobRecord:
        record = self.runner._enqueue("image_generation", asdict(request), request_id)
        self._enqueue(record)
        return record

    def enqueue_local_python(
        self,
        request: CodeExecutionRequest,
        *,
        request_id: str | None = None,
    ) -> JobRecord:
        record = self.runner._enqueue("code_execution", asdict(request), request_id)
        self._enqueue(record)
        return record

    def enqueue_index_rebuild(
        self,
        source_paths: list[str],
        *,
        request_id: str | None = None,
    ) -> JobRecord:
        record = self.runner._enqueue("index_rebuild", {"source_paths": source_paths}, request_id)
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
        self._queue.put(record.job_id)

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
        record = self.store.get(job_id)
        if record is None:
            return
        event = self._events.get(job_id)
        if record.status == "cancelled" or (event and event.is_set()):
            self.store.cancel(job_id)
            return
        if record.kind == "image_generation":
            self.runner.run_mock_image(
                ImageGenerationRequest(**record.request),
                request_id=record.request_id,
                record=record,
                cancel_event=event,
            )
        elif record.kind == "code_execution":
            self.runner.run_local_python(
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


_ASYNC_JOB_MANAGER: AsyncJobManager | None = None


def get_async_job_manager() -> AsyncJobManager:
    global _ASYNC_JOB_MANAGER
    if _ASYNC_JOB_MANAGER is None:
        _ASYNC_JOB_MANAGER = AsyncJobManager()
    return _ASYNC_JOB_MANAGER
