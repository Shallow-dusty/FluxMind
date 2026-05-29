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
from src.config import JOBS_FILE
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

    def __init__(self, path: Path = JOBS_FILE):
        self.path = path

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
