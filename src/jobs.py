"""Local job records for no-key artifact and execution workflows."""

from __future__ import annotations

import json
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
    """Append-only JSONL job store with latest-record reads."""

    def __init__(self, path: Path | None = None):
        self.path = path or JOBS_FILE

    def append(self, record: JobRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def get(self, job_id: str) -> JobRecord | None:
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
    ) -> JobRecord:
        record = self._start("image_generation", asdict(request), request_id)
        try:
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
    ) -> JobRecord:
        record = self._start("code_execution", asdict(request), request_id)
        try:
            result = LocalPythonExecutionProvider().run(request)
        except Exception as exc:
            error = normalize_exception(exc)
            return self._finish(record, status="failed", error=asdict(error))

        status: JobStatus = "succeeded" if result.success else "failed"
        payload = asdict(result)
        return self._finish(
            record,
            status=status,
            result=payload,
            artifacts=result.artifacts,
            error=None if result.success else {"code": "execution_failed", "message": result.stderr},
        )

    def run_index_rebuild(
        self,
        source_paths: list[str],
        *,
        request_id: str | None = None,
    ) -> JobRecord:
        request = {"source_paths": source_paths}
        record = self._start("index_rebuild", request, request_id)
        try:
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
