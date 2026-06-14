"""FluxMind API — FastAPI wrapper for RAG pipeline, compatible with Coze custom plugin."""

import os
import logging
import hashlib
import re
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from src.admin import (
    apply_retention_delete,
    collect_admin_status,
    collect_corpus_profile_status,
    collect_corpus_status,
    collect_retention_preview,
    format_admin_status_report,
    format_admin_metrics,
    format_corpus_profile_status_report,
)
from src.artifacts import LocalArtifactRegistry
from src.capabilities import CodeExecutionRequest, ImageGenerationRequest
from src.chain import get_vector_store, query_with_metadata, retrieve_with_metadata
from src.config import (
    API_ACCESS_AUDIT_ENABLED,
    API_RATE_LIMIT_ENABLED,
    API_RATE_LIMIT_MAX_REQUESTS,
    API_RATE_LIMIT_WINDOW_S,
    FAISS_INDEX_DIR,
    LLM_MODEL,
    QUERY_COST_COMPLETION_USD_PER_1M,
    QUERY_COST_PROMPT_USD_PER_1M,
    QUERY_COST_PROVIDER,
)
from src.costs import summarize_query_cost
from src.ingestion import (
    discover_pdfs,
    extract_pdf_structure_markers,
    refresh_paper_metadata,
    resolve_selectable_source_paths,
    set_active_paper_source_paths,
)
from src.jobs import JobRecord, LocalJobRunner, LocalJobStore, get_async_job_manager, normalize_ownership, ownership_from_record
from src.metadata import ChunkMetadataStore, CorpusProfileStore
from src.runtime import append_runtime_event, estimate_text_tokens, list_runtime_events, logger, new_request_id, normalize_exception
from src.storage_manifest import (
    collect_runtime_backup_manifest,
    collect_runtime_restore_check,
    format_runtime_backup_manifest_markdown,
    format_runtime_restore_check_markdown,
)

API_TOKEN = os.getenv("FLUXMIND_API_TOKEN", "")
logging.basicConfig(level=os.getenv("FLUXMIND_LOG_LEVEL", "INFO"))
logging.getLogger("faiss.loader").setLevel(logging.ERROR)

_CODE_BLOCK_RE = re.compile(r"```(?P<language>[\w.+-]*)\n(?P<body>.*?)```", re.DOTALL)
_ARTIFACT_REF_RE = re.compile(r"\[Artifact:(?P<artifact_id>[A-Za-z0-9_.:-]+)\]")
_PAPER_TO_CODE_TERMS = (
    "paper-to-code",
    "code",
    "matlab",
    "simulink",
    "octave",
    "python",
    "simulation",
    "plot",
)
_API_RATE_LIMIT_BUCKETS: dict[str, deque[float]] = defaultdict(deque)
_STARTUP_WARMUP_LOCK = threading.Lock()
_STARTUP_WARMUP_STATE: dict[str, Any] = {
    "status": "not_started",
    "ready": False,
    "error": "",
}


def _set_startup_warmup_state(status: str, *, ready: bool, error: str = "") -> None:
    with _STARTUP_WARMUP_LOCK:
        _STARTUP_WARMUP_STATE.update({"status": status, "ready": ready, "error": error})


def startup_warmup_status() -> dict[str, Any]:
    with _STARTUP_WARMUP_LOCK:
        return dict(_STARTUP_WARMUP_STATE)


def warm_existing_vector_store() -> bool:
    """Best-effort startup warmup without rebuilding a missing index."""
    if not (FAISS_INDEX_DIR / "index.faiss").exists():
        logger.warning("startup.index_missing path=%s", FAISS_INDEX_DIR)
        _set_startup_warmup_state("missing_index", ready=False)
        return False
    try:
        get_vector_store()
    except Exception:
        logger.exception("startup.index_warmup_failed path=%s", FAISS_INDEX_DIR)
        _set_startup_warmup_state("failed", ready=False, error="index_warmup_failed")
        return False
    _set_startup_warmup_state("ready", ready=True)
    return True


def start_background_vector_store_warmup() -> None:
    """Warm retrieval state without blocking the API socket bind."""
    _set_startup_warmup_state("warming", ready=False)
    thread = threading.Thread(
        target=warm_existing_vector_store,
        name="fluxmind-vector-store-warmup",
        daemon=True,
    )
    thread.start()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Warm existing retrieval state and recover durable queued local jobs."""
    start_background_vector_store_warmup()
    get_async_job_manager().recover_queued_jobs()
    yield

app = FastAPI(
    title="FluxMind API",
    description="RAG-based Copilot for Sliding Mode Control & Flux Linkage Estimation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def api_token_status(authorization: str | None, x_api_key: str | None) -> dict[str, Any]:
    """Classify API token headers without returning token values."""
    bearer_token = ""
    if authorization and authorization.lower().startswith("bearer "):
        bearer_token = authorization[7:].strip()
    has_bearer = bool(bearer_token)
    has_x_api_key = bool(x_api_key)
    if has_bearer and has_x_api_key:
        credential_type = "multiple"
    elif has_bearer:
        credential_type = "bearer"
    elif has_x_api_key:
        credential_type = "x_api_key"
    else:
        credential_type = "none"

    if not API_TOKEN:
        token_status = "not_configured"
    elif x_api_key == API_TOKEN or bearer_token == API_TOKEN:
        token_status = "valid"
    elif has_bearer or has_x_api_key:
        token_status = "invalid"
    else:
        token_status = "missing"

    return {
        "token_status": token_status,
        "credential_type": credential_type,
        "credential_present": credential_type != "none",
        "auth_configured": bool(API_TOKEN),
    }


def record_api_access_event(
    *,
    request: Request,
    response: Response | None,
    status_code: int,
    duration_ms: int,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Append a no-secret API access audit event."""
    if not API_ACCESS_AUDIT_ENABLED:
        return
    token_status = api_token_status(
        request.headers.get("authorization"),
        request.headers.get("x-api-key"),
    )
    request_id = None
    if response is not None:
        request_id = response.headers.get("X-Request-ID")
    request_id = request_id or request.headers.get("X-Request-ID")
    try:
        append_runtime_event(
            kind="api_access",
            code=f"auth_{token_status['token_status']}",
            message="Metadata-only API access audit event.",
            request_id=request_id,
            metadata={
                "method": request.method,
                "path": request.url.path[:256],
                "status_code": status_code,
                "duration_ms": duration_ms,
                **token_status,
                **(extra_metadata or {}),
            },
        )
    except OSError:
        logger.warning("api_access.event_log_failed path=%s", request.url.path)


def api_rate_limit_decision(request: Request, *, now: float | None = None) -> dict[str, Any]:
    """Return a local in-memory API rate-limit decision without persisted identity."""
    if not API_RATE_LIMIT_ENABLED:
        return {
            "enabled": False,
            "allowed": True,
            "limited": False,
            "limit": API_RATE_LIMIT_MAX_REQUESTS,
            "remaining": API_RATE_LIMIT_MAX_REQUESTS,
            "window_s": API_RATE_LIMIT_WINDOW_S,
            "reset_after_s": 0,
        }

    window_s = max(1, int(API_RATE_LIMIT_WINDOW_S))
    max_requests = max(1, int(API_RATE_LIMIT_MAX_REQUESTS))
    now_ts = time.monotonic() if now is None else now
    token_status = api_token_status(
        request.headers.get("authorization"),
        request.headers.get("x-api-key"),
    )
    client_host = request.client.host if request.client else "unknown"
    client_hash = hashlib.sha256(client_host.encode("utf-8")).hexdigest()[:16]
    key = "|".join(
        [
            request.method.upper(),
            request.url.path[:128],
            str(token_status["token_status"]),
            client_hash,
        ]
    )
    bucket = _API_RATE_LIMIT_BUCKETS[key]
    while bucket and now_ts - bucket[0] >= window_s:
        bucket.popleft()
    allowed = len(bucket) < max_requests
    if allowed:
        bucket.append(now_ts)
    oldest = bucket[0] if bucket else now_ts
    reset_after_s = max(0, int(window_s - (now_ts - oldest)))
    return {
        "enabled": True,
        "allowed": allowed,
        "limited": not allowed,
        "limit": max_requests,
        "remaining": max(0, max_requests - len(bucket)),
        "window_s": window_s,
        "reset_after_s": reset_after_s,
    }


def rate_limit_event_metadata(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "rate_limit_enabled": decision.get("enabled", False),
        "rate_limited": decision.get("limited", False),
        "rate_limit": decision.get("limit", 0),
        "rate_limit_remaining": decision.get("remaining", 0),
        "rate_limit_window_s": decision.get("window_s", 0),
        "rate_limit_reset_after_s": decision.get("reset_after_s", 0),
    }


def apply_rate_limit_headers(response: Response, decision: dict[str, Any]) -> None:
    if not decision.get("enabled"):
        return
    response.headers["X-RateLimit-Limit"] = str(decision.get("limit", 0))
    response.headers["X-RateLimit-Remaining"] = str(decision.get("remaining", 0))
    response.headers["X-RateLimit-Reset"] = str(decision.get("reset_after_s", 0))


@app.middleware("http")
async def api_access_audit_middleware(request: Request, call_next):
    started = time.monotonic()
    response: Response | None = None
    rate_limit = api_rate_limit_decision(request)
    if rate_limit.get("limited"):
        response = JSONResponse(
            status_code=429,
            content={"detail": "API rate limit exceeded"},
        )
        apply_rate_limit_headers(response, rate_limit)
        record_api_access_event(
            request=request,
            response=response,
            status_code=response.status_code,
            duration_ms=int((time.monotonic() - started) * 1000),
            extra_metadata=rate_limit_event_metadata(rate_limit),
        )
        return response
    try:
        response = await call_next(request)
    except Exception:
        record_api_access_event(
            request=request,
            response=None,
            status_code=500,
            duration_ms=int((time.monotonic() - started) * 1000),
            extra_metadata=rate_limit_event_metadata(rate_limit),
        )
        raise
    apply_rate_limit_headers(response, rate_limit)
    record_api_access_event(
        request=request,
        response=response,
        status_code=response.status_code,
        duration_ms=int((time.monotonic() - started) * 1000),
        extra_metadata=rate_limit_event_metadata(rate_limit),
    )
    return response


class OwnershipRequest(BaseModel):
    owner_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Optional local owner metadata; not used for authentication",
    )
    owner_label: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Optional human-readable local owner label",
    )


class QueryRequest(OwnershipRequest):
    question: str = Field(..., description="User question about SMC or flux estimation", examples=["What is sliding mode control?"])
    answer_mode: str = Field(
        default="explanation",
        description="Answer mode: explanation, derivation, implementation, literature_review, or code_generation",
    )


class QueryResponse(BaseModel):
    answer: str = Field(..., description="RAG-generated answer with citations")
    request_id: str = Field(..., description="Correlation ID for logs and support")


class QueryInspectResponse(BaseModel):
    result: dict = Field(..., description="RAG answer plus retrieved-context citation validation")
    request_id: str = Field(..., description="Correlation ID for logs and support")


class QueryRetrieveResponse(BaseModel):
    retrieval: dict = Field(..., description="Retrieved context refs and source/page diagnostics without LLM generation")
    request_id: str = Field(..., description="Correlation ID for logs and support")


class MockImageJobRequest(OwnershipRequest):
    prompt: str = Field(..., description="Diagram prompt")
    style: str = Field(default="engineering-diagram")
    size: str = Field(default="1024x1024")
    diagram_template: str = Field(default="generic", description="Local SVG template")
    reference_uris: list[str] = Field(default_factory=list)
    queue_timeout_s: int | None = Field(default=None, ge=1, le=86400, description="Optional async queue deadline in seconds")
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128, description="Optional local job idempotency key")
    max_attempts: int = Field(default=1, ge=1, le=10, description="Maximum attempts for async local jobs")
    retry_backoff_s: int = Field(default=0, ge=0, le=3600, description="Delay before automatic async retry")


class LocalPythonJobRequest(OwnershipRequest):
    entrypoint: str = Field(..., description="Python entrypoint filename")
    files: dict[str, str] = Field(..., description="Files to materialize for execution")
    timeout_s: int = Field(default=30, ge=1, le=120)
    memory_mb: int = Field(default=512, ge=64, le=4096)
    queue_timeout_s: int | None = Field(default=None, ge=1, le=86400, description="Optional async queue deadline in seconds")
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128, description="Optional local job idempotency key")
    max_attempts: int = Field(default=1, ge=1, le=10, description="Maximum attempts for async local jobs")
    retry_backoff_s: int = Field(default=0, ge=0, le=3600, description="Delay before automatic async retry")


class LocalOctaveJobRequest(OwnershipRequest):
    entrypoint: str = Field(..., description="Octave-compatible entrypoint filename")
    files: dict[str, str] = Field(..., description="Files to materialize for execution")
    timeout_s: int = Field(default=30, ge=1, le=120)
    memory_mb: int = Field(default=512, ge=64, le=4096)
    queue_timeout_s: int | None = Field(default=None, ge=1, le=86400, description="Optional async queue deadline in seconds")
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128, description="Optional local job idempotency key")
    max_attempts: int = Field(default=1, ge=1, le=10, description="Maximum attempts for async local jobs")
    retry_backoff_s: int = Field(default=0, ge=0, le=3600, description="Delay before automatic async retry")


class IndexRebuildJobRequest(OwnershipRequest):
    source_paths: list[str] = Field(..., description="Project-relative selectable PDF paths")
    queue_timeout_s: int | None = Field(default=None, ge=1, le=86400, description="Optional async queue deadline in seconds")
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128, description="Optional local job idempotency key")
    max_attempts: int = Field(default=1, ge=1, le=10, description="Maximum attempts for async local jobs")
    retry_backoff_s: int = Field(default=0, ge=0, le=3600, description="Delay before automatic async retry")


class ActiveCorpusRequest(BaseModel):
    source_paths: list[str] = Field(..., description="Project-relative selectable PDF paths to keep active")


class CorpusProfileRequest(BaseModel):
    name: str = Field(..., description="Human-readable local corpus profile name")
    source_paths: list[str] = Field(..., description="Project-relative selectable PDF paths in this profile")
    profile_id: str | None = Field(default=None, description="Optional stable local profile ID")
    description: str | None = Field(default=None, description="Optional no-secret profile description")


class CorpusProfileRebuildRequest(OwnershipRequest):
    queue_timeout_s: int | None = Field(default=None, ge=1, le=86400, description="Optional async queue deadline in seconds")
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128, description="Optional local job idempotency key")
    max_attempts: int = Field(default=1, ge=1, le=10, description="Maximum attempts for async local jobs")
    retry_backoff_s: int = Field(default=0, ge=0, le=3600, description="Delay before automatic async retry")


class RetryScheduleRequest(BaseModel):
    delay_s: int = Field(default=30, ge=0, le=3600, description="Delay before retry execution")
    queue_timeout_s: int | None = Field(default=None, ge=1, le=86400, description="Optional queued retry deadline in seconds")


class JobResponse(BaseModel):
    job: dict = Field(..., description="Persisted local job record")


class JobListResponse(BaseModel):
    jobs: list[dict] = Field(..., description="Latest local job records")


class CorpusPapersResponse(BaseModel):
    papers: list[dict] = Field(..., description="Current local corpus paper metadata")


class CorpusChunksResponse(BaseModel):
    chunks: list[dict] = Field(..., description="Current local indexed chunk metadata")


class CorpusStructureResponse(BaseModel):
    markers: list[dict] = Field(..., description="Best-effort PDF layout markers from selectable PDFs")


class CorpusStatusResponse(BaseModel):
    status: dict = Field(..., description="Corpus-level local lifecycle status")


class ActiveCorpusResponse(BaseModel):
    papers: list[dict] = Field(..., description="Updated local corpus paper metadata")
    active_source_paths: list[str] = Field(..., description="Persisted active paper source paths")
    rebuild_required: bool = Field(..., description="Whether the FAISS index should be rebuilt to apply selection")


class CorpusProfilesResponse(BaseModel):
    profiles: list[dict] = Field(..., description="Reusable local corpus selection profiles")


class CorpusProfileResponse(BaseModel):
    profile: dict = Field(..., description="Reusable local corpus selection profile")


class CorpusProfileStatusResponse(BaseModel):
    status: dict = Field(..., description="No-secret status for one saved corpus profile")


class CorpusProfileRebuildResponse(BaseModel):
    profile: dict = Field(..., description="Applied reusable local corpus selection profile")
    job: dict = Field(..., description="Queued local index rebuild job")
    active_source_paths: list[str] = Field(..., description="Persisted active paper source paths")
    rebuild_required: bool = Field(..., description="Whether a rebuild is required after queueing")
    queued_rebuild: bool = Field(..., description="Whether a rebuild job was queued")


class ArtifactListResponse(BaseModel):
    artifacts: list[dict] = Field(..., description="Generated local artifacts")


class AdminStatusResponse(BaseModel):
    status: dict = Field(..., description="Local admin/runtime status")


class RetentionPreviewResponse(BaseModel):
    retention: dict = Field(..., description="No-delete local retention preview")


class RetentionDeleteResponse(BaseModel):
    retention: dict = Field(..., description="Guarded local retention delete result")


class RuntimeEventsResponse(BaseModel):
    events: list[dict] = Field(..., description="Latest no-secret local runtime events")


class RuntimeManifestResponse(BaseModel):
    manifest: dict = Field(..., description="No-secret local runtime backup manifest")


class RuntimeRestoreCheckRequest(BaseModel):
    manifest: dict = Field(..., description="No-secret runtime backup manifest to verify")


class RuntimeRestoreCheckResponse(BaseModel):
    restore_check: dict = Field(..., description="No-secret runtime restore dry-run result")


def verify_api_token(authorization: str | None, x_api_key: str | None) -> None:
    """Protect public Coze/plugin calls when FLUXMIND_API_TOKEN is configured."""
    status = api_token_status(authorization, x_api_key)
    if status["token_status"] in {"not_configured", "valid"}:
        return
    else:
        raise HTTPException(status_code=401, detail="Invalid API token")


def request_id_header(response: Response, x_request_id: str | None) -> str:
    request_id = (x_request_id or new_request_id()).strip()[:64]
    response.headers["X-Request-ID"] = request_id
    return request_id


def request_ownership(req: Any) -> dict[str, str]:
    return normalize_ownership(
        owner_id=getattr(req, "owner_id", None),
        owner_label=getattr(req, "owner_label", None),
    )


def job_to_dict(record: JobRecord) -> dict:
    ownership = ownership_from_record(record)
    return {
        "job_id": record.job_id,
        "kind": record.kind,
        "status": record.status,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "request": record.request,
        "result": record.result,
        "artifacts": record.artifacts,
        "error": record.error,
        "attempts": record.attempts,
        "request_id": record.request_id,
        "parent_job_id": record.parent_job_id,
        "not_before": record.not_before,
        "deadline_at": record.deadline_at,
        "idempotency_key": record.idempotency_key,
        "max_attempts": record.max_attempts,
        "retry_backoff_s": record.retry_backoff_s,
        "dead_lettered_at": record.dead_lettered_at,
        "owner_id": ownership["owner_id"],
        "owner_label": ownership["owner_label"],
        "ownership_source": ownership["ownership_source"],
        "logs": record.logs,
    }


def existing_idempotent_job(kind: str, idempotency_key: str | None) -> JobRecord | None:
    if not idempotency_key:
        return None
    return LocalJobStore().find_by_idempotency_key(kind=kind, key=idempotency_key)


def record_to_dict(record) -> dict:
    return asdict(record)


def runtime_event_to_dict(event) -> dict:
    return asdict(event)


def filter_paper_records(
    records: list[Any],
    *,
    q: str | None = None,
    active: bool | None = None,
    source_kind: str | None = None,
    indexed_status: str | None = None,
) -> list[Any]:
    """Filter local paper metadata for lightweight corpus search."""
    query = (q or "").strip().casefold()
    source_kind = source_kind.strip() if source_kind else None
    indexed_status = indexed_status.strip() if indexed_status else None
    filtered: list[Any] = []
    for record in records:
        if active is not None and bool(record.active) is not active:
            continue
        if source_kind and record.source_kind != source_kind:
            continue
        if indexed_status and record.indexed_status != indexed_status:
            continue
        if query:
            searchable = " ".join(
                str(value or "")
                for value in (
                    record.paper_id,
                    record.source_path,
                    record.filename,
                    record.source_kind,
                    record.checksum_sha256,
                    record.title,
                    record.authors,
                    record.year,
                    record.topic,
                    record.doi,
                    record.arxiv_id,
                    record.venue,
                    " ".join(record.topic_tags or []),
                )
            ).casefold()
            if query not in searchable:
                continue
        filtered.append(record)
    return filtered


def validate_corpus_profile_source_paths(source_paths: list[str]) -> list[str]:
    """Validate local profile paths against the selectable corpus metadata."""
    selectable = {record.source_path for record in refresh_paper_metadata()}
    clean_paths: list[str] = []
    seen: set[str] = set()
    for source_path in source_paths:
        clean_path = source_path.strip()
        if not clean_path or clean_path in seen:
            continue
        if clean_path not in selectable:
            raise ValueError(f"PDF path is not in the selectable corpus: {source_path}")
        seen.add(clean_path)
        clean_paths.append(clean_path)
    if not clean_paths:
        raise ValueError("At least one source path is required")
    return clean_paths


def collect_corpus_structure_markers(
    *,
    source_path: str | None = None,
    kind: str | None = None,
    page: int | None = None,
    q: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Collect no-key PDF layout markers from selectable local PDFs."""
    bounded_limit = min(max(limit, 1), 500)
    query = (q or "").strip().casefold()
    if source_path:
        try:
            paths = resolve_selectable_source_paths([source_path])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        paths = discover_pdfs()
    clean_kind = kind.strip().casefold() if kind else ""
    kinds = {clean_kind} if clean_kind else None
    markers: list[dict[str, Any]] = []
    for path in paths:
        remaining = bounded_limit - len(markers)
        if remaining <= 0:
            break
        scan_limit = 500 if query else remaining
        for marker in extract_pdf_structure_markers(
            path,
            kinds=kinds,
            page=page,
            max_markers=scan_limit,
        ):
            marker_dict = asdict(marker)
            if query:
                searchable = " ".join(
                    str(marker_dict.get(key, ""))
                    for key in ("kind", "source", "source_path", "page", "text", "rule")
                ).casefold()
                if query not in searchable:
                    continue
            markers.append(marker_dict)
            if len(markers) >= bounded_limit:
                break
    return markers


def format_corpus_structure_report(
    *,
    markers: list[dict[str, Any]],
    source_path: str | None = None,
    kind: str | None = None,
    page: int | None = None,
    q: str | None = None,
    limit: int = 100,
) -> str:
    """Render PDF layout markers as a no-secret Markdown report."""
    kind_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for marker in markers:
        marker_kind = str(marker.get("kind", "unknown") or "unknown")
        marker_source = str(marker.get("source_path", "unknown") or "unknown")
        kind_counts[marker_kind] = kind_counts.get(marker_kind, 0) + 1
        source_counts[marker_source] = source_counts.get(marker_source, 0) + 1

    lines = [
        "# FluxMind Corpus Structure Report",
        "",
        f"- Source filter: `{source_path or 'all selectable PDFs'}`",
        f"- Kind filter: `{kind or 'all'}`",
        f"- Page filter: `{page if page is not None else 'all'}`",
        f"- Text filter: `{q or 'none'}`",
        f"- Limit: `{min(max(limit, 1), 500)}`",
        f"- Marker count: `{len(markers)}`",
        "",
        "## Summary",
        "",
    ]
    if kind_counts:
        for marker_kind, count in sorted(kind_counts.items()):
            lines.append(f"- `{marker_kind}`: {count}")
    else:
        lines.append("- No markers matched the filters.")

    lines.extend(["", "## Sources", ""])
    if source_counts:
        for marker_source, count in sorted(source_counts.items()):
            lines.append(f"- `{marker_source}`: {count}")
    else:
        lines.append("- No source markers matched the filters.")

    lines.extend(["", "## Markers", ""])
    for index, marker in enumerate(markers, start=1):
        lines.append(
            f"{index}. `{marker.get('kind', 'unknown')}` "
            f"`{marker.get('source_path', marker.get('source', 'unknown'))}` "
            f"page `{marker.get('page', '?')}` rule `{marker.get('rule', 'unknown')}`"
        )
        lines.append(f"   - {str(marker.get('text', '')).strip()}")
    if not markers:
        lines.append("No markers matched the filters.")
    return "\n".join(lines) + "\n"


def corpus_profile_status(profile_id: str) -> dict[str, Any]:
    """Inspect a saved corpus profile without changing the active selection."""
    try:
        return collect_corpus_profile_status(profile_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Corpus profile not found") from exc


def record_query_usage(
    *,
    endpoint: str,
    request_id: str,
    answer_mode: str,
    question: str,
    answer: str,
    citation_ok: bool | None = None,
    provider_usage: dict[str, int] | None = None,
    ownership: dict[str, str] | None = None,
    duration_ms: int | None = None,
) -> None:
    """Append no-secret estimated query usage for local admin/cost-shape checks."""
    estimated_prompt_tokens = estimate_text_tokens(question)
    estimated_answer_tokens = estimate_text_tokens(answer)
    provider_prompt_tokens = 0
    provider_completion_tokens = 0
    provider_usage_events = 0
    if provider_usage:
        provider_prompt_tokens = int(provider_usage.get("prompt_tokens", 0) or 0)
        provider_completion_tokens = int(provider_usage.get("completion_tokens", 0) or 0)
        provider_usage_events = 1
    cost_summary = summarize_query_cost(
        estimated_prompt_tokens=estimated_prompt_tokens,
        estimated_completion_tokens=estimated_answer_tokens,
        provider_prompt_tokens=provider_prompt_tokens,
        provider_completion_tokens=provider_completion_tokens,
        provider_usage_events=provider_usage_events,
        provider=QUERY_COST_PROVIDER or LLM_MODEL,
        prompt_usd_per_1m=QUERY_COST_PROMPT_USD_PER_1M,
        completion_usd_per_1m=QUERY_COST_COMPLETION_USD_PER_1M,
    )
    metadata = {
        "endpoint": endpoint,
        "answer_mode": answer_mode,
        "question_chars": len(question),
        "answer_chars": len(answer),
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "estimated_answer_tokens": estimated_answer_tokens,
        "estimated_total_tokens": estimated_prompt_tokens + estimated_answer_tokens,
        "estimated_cost_usd": cost_summary["estimated_cost_usd"],
        "cost_source": cost_summary["cost_source"],
        "usage_source": "provider" if provider_usage else "estimated",
    }
    if duration_ms is not None:
        metadata["duration_ms"] = max(duration_ms, 0)
    metadata.update(normalize_ownership(**(ownership or {})))
    if provider_usage:
        metadata.update(
            {
                "provider_prompt_tokens": provider_prompt_tokens,
                "provider_completion_tokens": provider_completion_tokens,
                "provider_total_tokens": int(provider_usage.get("total_tokens", 0) or 0),
            }
        )
    if citation_ok is not None:
        metadata["citation_ok"] = citation_ok
    try:
        append_runtime_event(
            kind="query_usage",
            code="estimated_usage",
            message="Estimated no-key query usage. This is not provider billing.",
            request_id=request_id,
            metadata=metadata,
        )
    except OSError:
        logger.warning("query.usage_event_log_failed request_id=%s endpoint=%s", request_id, endpoint)


def context_refs_missing_source_page_count(context_refs: list[dict[str, Any]]) -> int:
    """Count retrieved refs that lack source/page metadata without exposing refs."""
    missing = 0
    for ref in context_refs:
        has_source = bool(ref.get("source") or ref.get("source_path"))
        if not has_source or ref.get("page") in {None, "", "?"}:
            missing += 1
    return missing


def record_retrieval_trace(
    *,
    endpoint: str,
    answer_mode: str,
    context_count: int,
    missing_source_page_count: int,
    provider_called: bool,
    duration_ms: int,
    citation_ok: bool | None = None,
    retrieval_ok: bool | None = None,
) -> None:
    """Append metadata-only retrieval trace data for local admin observability."""
    bounded_context_count = max(int(context_count or 0), 0)
    bounded_missing_count = max(int(missing_source_page_count or 0), 0)
    source_page_complete = bounded_missing_count == 0
    ok = (
        bool(retrieval_ok)
        if retrieval_ok is not None
        else bounded_context_count > 0 and source_page_complete
    )
    if bounded_context_count <= 0:
        code = "retrieval_empty"
    elif not source_page_complete:
        code = "retrieval_source_page_incomplete"
    else:
        code = "retrieval_ok"

    metadata: dict[str, Any] = {
        "endpoint": endpoint,
        "answer_mode": answer_mode,
        "context_count": bounded_context_count,
        "missing_source_page_count": bounded_missing_count,
        "source_page_complete": source_page_complete,
        "retrieval_ok": ok,
        "provider_called": provider_called,
        "duration_ms": max(duration_ms, 0),
    }
    if citation_ok is not None:
        metadata["citation_ok"] = bool(citation_ok)
    try:
        append_runtime_event(
            kind="retrieval_trace",
            code=code,
            message="Metadata-only retrieval trace. No prompt, answer, retrieved text, source path, owner, or request ID is stored.",
            request_id=None,
            metadata=metadata,
        )
    except OSError:
        logger.warning("query.retrieval_trace_event_log_failed endpoint=%s", endpoint)


def record_result_retrieval_trace(
    *,
    endpoint: str,
    answer_mode: str,
    result: Any,
    duration_ms: int,
    provider_called: bool,
) -> None:
    """Record retrieval trace metadata from a generated query result."""
    context_refs = getattr(result, "context_refs", None) or []
    if not context_refs and hasattr(result, "to_dict"):
        try:
            context_refs = result.to_dict().get("context_refs", []) or []
        except (AttributeError, TypeError, ValueError):
            context_refs = []
    citation_validation = getattr(result, "citation_validation", None)
    citation_ok = getattr(citation_validation, "ok", None)
    record_retrieval_trace(
        endpoint=endpoint,
        answer_mode=str(getattr(result, "answer_mode", answer_mode) or answer_mode),
        context_count=len(context_refs),
        missing_source_page_count=context_refs_missing_source_page_count(context_refs),
        provider_called=provider_called,
        citation_ok=citation_ok if citation_ok is None else bool(citation_ok),
        duration_ms=duration_ms,
    )


def extract_markdown_code_blocks(markdown: str) -> list[dict[str, str]]:
    """Return fenced code blocks from a generated Markdown answer."""
    blocks: list[dict[str, str]] = []
    for match in _CODE_BLOCK_RE.finditer(markdown):
        body = match.group("body").strip()
        if not body:
            continue
        blocks.append(
            {
                "language": (match.group("language") or "text").strip() or "text",
                "body": body,
            }
        )
    return blocks


def extract_artifact_refs(markdown: str) -> list[str]:
    """Return stable generated artifact IDs cited in a Markdown answer."""
    seen: set[str] = set()
    artifact_ids: list[str] = []
    for match in _ARTIFACT_REF_RE.finditer(markdown):
        artifact_id = match.group("artifact_id")
        if artifact_id in seen:
            continue
        seen.add(artifact_id)
        artifact_ids.append(artifact_id)
    return artifact_ids


def include_paper_to_code_handoff(*, question: str, result: Any) -> bool:
    """Decide whether a query report needs implementation handoff sections."""
    answer_mode = str(getattr(result, "answer_mode", "") or "").lower()
    if answer_mode in {"implementation", "code_generation"}:
        return True
    search_text = f"{question}\n{getattr(result, 'answer', '')}".casefold()
    return any(term in search_text for term in _PAPER_TO_CODE_TERMS)


def context_ref_line(ref: dict[str, Any]) -> str:
    """Format one retrieved context ref for Markdown reports."""
    return (
        f"- [{ref.get('ref')}] {ref.get('source_path') or ref.get('source') or 'unknown'} "
        f"page={ref.get('page', '?')} preview={ref.get('preview', '')}"
    )


def append_paper_to_code_handoff(lines: list[str], *, question: str, result: Any) -> None:
    """Append a structured paper-to-code report section without external providers."""
    code_blocks = extract_markdown_code_blocks(result.answer)
    artifact_ids = extract_artifact_refs(result.answer)
    validation = result.citation_validation.to_dict()

    lines.extend(
        [
            "",
            "## Paper-to-Code Handoff",
            "",
            "### Source Trace",
            "",
        ]
    )
    if not result.context_refs:
        lines.append("- No retrieved source refs available for this handoff.")
    for ref in result.context_refs:
        lines.append(context_ref_line(ref))

    lines.extend(
        [
            "",
            "### Assumptions and Parameters",
            "",
            "- Preserve source/page citations for every plant model, observer equation, parameter, and tuning claim.",
            "- Mark missing motor parameters, sample time, solver settings, units, or operating envelope as unresolved.",
            "- Do not treat generated code as validated until it is run through a local job and its artifacts are attached.",
            "",
            "### Generated Code Blocks",
            "",
        ]
    )
    if not code_blocks:
        lines.append("- No fenced code blocks were detected in the generated answer.")
    for index, block in enumerate(code_blocks, start=1):
        lines.extend(
            [
                f"- Code block {index}: language=`{block['language']}`",
                "",
                f"```{block['language']}",
                block["body"],
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "### Execution Outputs and Plot Artifacts",
            "",
        ]
    )
    if artifact_ids:
        for artifact_id in artifact_ids:
            lines.append(f"- Cited artifact: `[Artifact:{artifact_id}]`")
    else:
        lines.append("- No generated artifact refs were cited in the answer.")
    lines.append("- Execution output is not attached unless a local job produced an exported artifact.")

    lines.extend(
        [
            "",
            "### Validation Checklist",
            "",
            f"- Citation validation: {'ok' if validation.get('ok') else 'needs_review'}",
            f"- Source refs attached: {len(result.context_refs)}",
            f"- Code blocks attached: {len(code_blocks)}",
            f"- Artifact refs attached: {len(artifact_ids)}",
            f"- Original question: {question.strip()}",
        ]
    )


def format_query_report(*, question: str, result: Any, request_id: str) -> str:
    """Render a query result as a Markdown research report."""
    validation = result.citation_validation.to_dict()
    lines = [
        "# FluxMind Query Report",
        "",
        f"- Request ID: `{request_id}`",
        f"- Answer mode: `{result.answer_mode}`",
        f"- Citation check: {'ok' if validation.get('ok') else 'needs_review'}",
        "",
        "## Question",
        "",
        question.strip(),
        "",
        "## Answer",
        "",
        result.answer.strip(),
        "",
        "## Citation Validation",
        "",
        f"- Cited refs: {validation.get('cited_refs', [])}",
        f"- Invalid refs: {validation.get('invalid_refs', [])}",
        f"- Missing required refs: {validation.get('missing_required_refs', [])}",
        f"- Missing source/page refs: {validation.get('missing_source_page_refs', [])}",
        "",
        "## Retrieved Context",
        "",
    ]
    if not result.context_refs:
        lines.append("- No retrieved context refs.")
    for ref in result.context_refs:
        lines.append(context_ref_line(ref))
    if include_paper_to_code_handoff(question=question, result=result):
        append_paper_to_code_handoff(lines, question=question, result=result)
    lines.append("")
    return "\n".join(lines)


@app.get("/artifacts", response_model=ArtifactListResponse, summary="List local artifacts")
def list_artifacts(
    q: str | None = None,
    kind: str | None = None,
    job_kind: str | None = None,
    owner_id: str | None = None,
    limit: int = 100,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """List artifacts produced by local jobs."""
    verify_api_token(authorization, x_api_key)
    bounded_limit = min(max(limit, 1), 500)
    artifacts = [
        asdict(artifact)
        for artifact in LocalArtifactRegistry().list_artifacts(
            limit=bounded_limit,
            kind=kind,
            job_kind=job_kind,
            owner_id=owner_id,
            q=q,
        )
    ]
    return ArtifactListResponse(artifacts=artifacts)


@app.get("/artifacts/{artifact_id}", summary="Download local artifact")
def download_artifact(
    artifact_id: str,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Download a local generated artifact by stable artifact ID."""
    verify_api_token(authorization, x_api_key)
    try:
        artifact, path = LocalArtifactRegistry().export_path(artifact_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(
        path,
        media_type=artifact.mime_type,
        filename=artifact.title or path.name,
    )


@app.get("/corpus/papers", response_model=CorpusPapersResponse, summary="List corpus papers")
def list_corpus_papers(
    q: str | None = None,
    active: bool | None = None,
    source_kind: str | None = None,
    indexed_status: str | None = None,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """List selectable papers with local metadata, active state, and index status."""
    verify_api_token(authorization, x_api_key)
    papers = [
        record_to_dict(record)
        for record in filter_paper_records(
            refresh_paper_metadata(),
            q=q,
            active=active,
            source_kind=source_kind,
            indexed_status=indexed_status,
        )
    ]
    return CorpusPapersResponse(papers=papers)


@app.get("/corpus/chunks", response_model=CorpusChunksResponse, summary="List corpus chunks")
def list_corpus_chunks(
    source_path: str | None = None,
    page: int | None = None,
    q: str | None = None,
    limit: int = 100,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """List local indexed chunk metadata for source/page/citation inspection."""
    verify_api_token(authorization, x_api_key)
    bounded_limit = min(max(limit, 1), 1000)
    chunks = [
        asdict(record)
        for record in ChunkMetadataStore().list_chunks(
            source_path=source_path,
            page=page,
            q=q,
            limit=bounded_limit,
        )
    ]
    return CorpusChunksResponse(chunks=chunks)


@app.get("/corpus/structure", response_model=CorpusStructureResponse, summary="List corpus PDF structure markers")
def list_corpus_structure(
    source_path: str | None = None,
    kind: str | None = None,
    page: int | None = None,
    q: str | None = None,
    limit: int = 100,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """List no-key PDF layout markers for source-grounded engineering checks."""
    verify_api_token(authorization, x_api_key)
    markers = collect_corpus_structure_markers(
        source_path=source_path,
        kind=kind,
        page=page,
        q=q,
        limit=limit,
    )
    return CorpusStructureResponse(markers=markers)


@app.get("/corpus/structure/report", response_class=PlainTextResponse, summary="Export corpus PDF structure report")
def corpus_structure_report(
    source_path: str | None = None,
    kind: str | None = None,
    page: int | None = None,
    q: str | None = None,
    limit: int = 100,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Export no-key PDF layout markers as a Markdown handoff report."""
    verify_api_token(authorization, x_api_key)
    markers = collect_corpus_structure_markers(
        source_path=source_path,
        kind=kind,
        page=page,
        q=q,
        limit=limit,
    )
    report = format_corpus_structure_report(
        markers=markers,
        source_path=source_path,
        kind=kind,
        page=page,
        q=q,
        limit=limit,
    )
    return PlainTextResponse(report, media_type="text/markdown")


@app.get("/corpus/status", response_model=CorpusStatusResponse, summary="Inspect corpus lifecycle status")
def corpus_status(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return no-secret corpus status such as queued, parsing, indexed, failed, or stale."""
    verify_api_token(authorization, x_api_key)
    return CorpusStatusResponse(status=collect_corpus_status())


@app.put("/corpus/active", response_model=ActiveCorpusResponse, summary="Update active corpus selection")
def update_active_corpus(
    req: ActiveCorpusRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Persist active/deactivated papers without requiring filesystem edits."""
    verify_api_token(authorization, x_api_key)
    if not req.source_paths:
        raise HTTPException(status_code=400, detail="At least one source path is required")
    try:
        papers = [record_to_dict(record) for record in set_active_paper_source_paths(req.source_paths)]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    active_source_paths = [paper["source_path"] for paper in papers if paper["active"]]
    return ActiveCorpusResponse(
        papers=papers,
        active_source_paths=active_source_paths,
        rebuild_required=True,
    )


@app.get("/corpus/profiles", response_model=CorpusProfilesResponse, summary="List local corpus profiles")
def list_corpus_profiles(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """List reusable no-key local corpus selection profiles."""
    verify_api_token(authorization, x_api_key)
    profiles = [record_to_dict(profile) for profile in CorpusProfileStore().list_profiles()]
    return CorpusProfilesResponse(profiles=profiles)


@app.post("/corpus/profiles", response_model=CorpusProfileResponse, summary="Create or update local corpus profile")
def upsert_corpus_profile(
    req: CorpusProfileRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Persist a named local corpus selection without changing the active FAISS index."""
    verify_api_token(authorization, x_api_key)
    try:
        source_paths = validate_corpus_profile_source_paths(req.source_paths)
        profile = CorpusProfileStore().upsert_profile(
            name=req.name,
            profile_id=req.profile_id,
            description=req.description,
            source_paths=source_paths,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CorpusProfileResponse(profile=record_to_dict(profile))


@app.get(
    "/corpus/profiles/{profile_id}/status",
    response_model=CorpusProfileStatusResponse,
    summary="Inspect local corpus profile status",
)
def inspect_corpus_profile(
    profile_id: str,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Inspect profile paper availability and index freshness without activation."""
    verify_api_token(authorization, x_api_key)
    return CorpusProfileStatusResponse(status=corpus_profile_status(profile_id))


@app.get(
    "/corpus/profiles/{profile_id}/report",
    summary="Download local corpus profile status report",
)
def corpus_profile_report(
    profile_id: str,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return one no-secret corpus profile status snapshot as Markdown."""
    verify_api_token(authorization, x_api_key)
    report = format_corpus_profile_status_report(corpus_profile_status(profile_id))
    return PlainTextResponse(
        report,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="fluxmind-corpus-profile-{profile_id}.md"'},
    )


@app.post(
    "/corpus/profiles/{profile_id}/activate",
    response_model=ActiveCorpusResponse,
    summary="Activate local corpus profile",
)
def activate_corpus_profile(
    profile_id: str,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Apply a saved corpus profile to the active local selection."""
    verify_api_token(authorization, x_api_key)
    try:
        profile = CorpusProfileStore().get_profile(profile_id)
        papers = [record_to_dict(record) for record in set_active_paper_source_paths(profile.source_paths)]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Corpus profile not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    active_source_paths = [paper["source_path"] for paper in papers if paper["active"]]
    return ActiveCorpusResponse(
        papers=papers,
        active_source_paths=active_source_paths,
        rebuild_required=True,
    )


@app.post(
    "/corpus/profiles/{profile_id}/rebuild",
    response_model=CorpusProfileRebuildResponse,
    summary="Activate local corpus profile and queue index rebuild",
)
def rebuild_corpus_profile(
    profile_id: str,
    req: CorpusProfileRebuildRequest,
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Apply a saved corpus profile and queue FAISS rebuild through the local job system."""
    verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req)
    try:
        profile = CorpusProfileStore().get_profile(profile_id)
        papers = [record_to_dict(record) for record in set_active_paper_source_paths(profile.source_paths)]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Corpus profile not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    active_source_paths = [paper["source_path"] for paper in papers if paper["active"]]
    job = get_async_job_manager().enqueue_index_rebuild(
        profile.source_paths,
        request_id=request_id,
        queue_timeout_s=req.queue_timeout_s,
        idempotency_key=req.idempotency_key,
        max_attempts=req.max_attempts,
        retry_backoff_s=req.retry_backoff_s,
        ownership=ownership,
    )
    return CorpusProfileRebuildResponse(
        profile=record_to_dict(profile),
        job=job_to_dict(job),
        active_source_paths=active_source_paths,
        rebuild_required=True,
        queued_rebuild=True,
    )


@app.get("/admin/status", response_model=AdminStatusResponse, summary="Inspect local runtime status")
def admin_status(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return no-secret local runtime status for operations and admin UI."""
    verify_api_token(authorization, x_api_key)
    return AdminStatusResponse(status=collect_admin_status().to_dict())


@app.get("/admin/status/report", summary="Download local runtime status report")
def admin_status_report(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return the no-secret local runtime status as a portable Markdown report."""
    verify_api_token(authorization, x_api_key)
    report = format_admin_status_report(collect_admin_status())
    return PlainTextResponse(
        report,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="fluxmind-admin-status.md"'},
    )


@app.get("/admin/metrics", summary="Export no-secret local metrics")
def admin_metrics(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return local admin summaries as Prometheus/OpenMetrics-style text."""
    verify_api_token(authorization, x_api_key)
    metrics = format_admin_metrics(collect_admin_status())
    return PlainTextResponse(
        metrics,
        media_type="text/plain; version=0.0.4; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="fluxmind-admin-metrics.prom"'},
    )


@app.get("/admin/runtime-manifest", response_model=RuntimeManifestResponse, summary="Inspect runtime backup manifest")
def admin_runtime_manifest(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return a no-secret backup manifest for excluded local runtime state."""
    verify_api_token(authorization, x_api_key)
    return RuntimeManifestResponse(manifest=collect_runtime_backup_manifest())


@app.get("/admin/runtime-manifest/report", summary="Download runtime backup manifest report")
def admin_runtime_manifest_report(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return the no-secret runtime backup manifest as portable Markdown."""
    verify_api_token(authorization, x_api_key)
    report = format_runtime_backup_manifest_markdown(collect_runtime_backup_manifest())
    return PlainTextResponse(
        report,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="fluxmind-runtime-manifest.md"'},
    )


@app.post(
    "/admin/runtime-manifest/restore-check",
    response_model=RuntimeRestoreCheckResponse,
    summary="Dry-run verify runtime manifest restore state",
)
def admin_runtime_manifest_restore_check(
    request: RuntimeRestoreCheckRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Check whether local runtime state matches a no-secret manifest."""
    verify_api_token(authorization, x_api_key)
    return RuntimeRestoreCheckResponse(
        restore_check=collect_runtime_restore_check(request.manifest)
    )


@app.post(
    "/admin/runtime-manifest/restore-check/report",
    summary="Download runtime restore dry-run report",
)
def admin_runtime_manifest_restore_check_report(
    request: RuntimeRestoreCheckRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return a no-secret Markdown report for the runtime restore dry-run."""
    verify_api_token(authorization, x_api_key)
    report = format_runtime_restore_check_markdown(
        collect_runtime_restore_check(request.manifest)
    )
    return PlainTextResponse(
        report,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="fluxmind-runtime-restore-dry-run.md"'
        },
    )


@app.get("/admin/retention", response_model=RetentionPreviewResponse, summary="Preview local retention candidates")
def admin_retention_preview(
    upload_days: int = 0,
    artifact_days: int = 0,
    limit: int = 100,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Preview local upload/artifact files that match retention age thresholds."""
    verify_api_token(authorization, x_api_key)
    return RetentionPreviewResponse(
        retention=collect_retention_preview(
            upload_days=upload_days,
            artifact_days=artifact_days,
            limit=limit,
        )
    )


@app.post("/admin/retention/delete", response_model=RetentionDeleteResponse, summary="Delete local retention candidates")
def admin_retention_delete(
    upload_days: int = 0,
    artifact_days: int = 0,
    limit: int = 100,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Delete local upload/artifact retention candidates when explicitly enabled."""
    verify_api_token(authorization, x_api_key)
    retention = apply_retention_delete(
        upload_days=upload_days,
        artifact_days=artifact_days,
        limit=limit,
    )
    if not retention.get("delete_enabled", False):
        raise HTTPException(
            status_code=403,
            detail={
                "code": "retention_delete_disabled",
                "retention": retention,
            },
        )
    return RetentionDeleteResponse(retention=retention)


@app.get("/admin/events", response_model=RuntimeEventsResponse, summary="List local runtime events")
def admin_runtime_events(
    kind: str | None = None,
    code: str | None = None,
    q: str | None = None,
    limit: int = 50,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """List no-secret provider failure and query usage events without reading JSONL by hand."""
    verify_api_token(authorization, x_api_key)
    bounded_limit = min(max(limit, 1), 200)
    events = [
        runtime_event_to_dict(event)
        for event in list_runtime_events(
            kind=kind,
            code=code,
            q=q,
            limit=bounded_limit,
        )
    ]
    return RuntimeEventsResponse(events=events)


@app.post("/query", response_model=QueryResponse, summary="Ask FluxMind")
def ask(
    req: QueryRequest,
    response: Response,
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Query the FluxMind knowledge base. Retrieves relevant paper chunks and generates an answer."""
    verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req)
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    logger.info(
        "query.start request_id=%s client=%s chars=%s",
        request_id,
        request.client.host if request.client else "unknown",
        len(req.question),
    )
    started = time.monotonic()
    try:
        result = query_with_metadata(req.question, answer_mode=req.answer_mode)
        answer = result.answer
    except Exception as exc:
        error = normalize_exception(exc)
        duration_ms = int((time.monotonic() - started) * 1000)
        try:
            append_runtime_event(
                kind="provider_failure",
                code=error.code,
                message=error.message,
                request_id=request_id,
                metadata={
                    "endpoint": "/query",
                    "answer_mode": req.answer_mode,
                    "status_code": error.status_code,
                    "duration_ms": duration_ms,
                    **ownership,
                },
            )
        except OSError:
            logger.warning("query.event_log_failed request_id=%s code=%s", request_id, error.code)
        logger.exception("query.error request_id=%s code=%s", request_id, error.code)
        raise HTTPException(
            status_code=error.status_code,
            detail={
                "code": error.code,
                "message": error.message,
                "request_id": request_id,
            },
        ) from exc
    duration_ms = int((time.monotonic() - started) * 1000)
    record_query_usage(
        endpoint="/query",
        request_id=request_id,
        answer_mode=req.answer_mode,
        question=req.question,
        answer=answer,
        provider_usage=result.provider_usage,
        ownership=ownership,
        duration_ms=duration_ms,
    )
    record_result_retrieval_trace(
        endpoint="/query",
        answer_mode=req.answer_mode,
        result=result,
        provider_called=True,
        duration_ms=duration_ms,
    )
    logger.info("query.ok request_id=%s chars=%s", request_id, len(answer))
    return QueryResponse(answer=answer, request_id=request_id)


@app.post("/query/inspect", response_model=QueryInspectResponse, summary="Ask FluxMind with citation inspection")
def ask_with_inspection(
    req: QueryRequest,
    response: Response,
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Return an answer plus numbered citation validation against retrieved chunks."""
    verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req)
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    logger.info(
        "query.inspect_start request_id=%s client=%s chars=%s",
        request_id,
        request.client.host if request.client else "unknown",
        len(req.question),
    )
    started = time.monotonic()
    try:
        result = query_with_metadata(req.question, answer_mode=req.answer_mode)
    except Exception as exc:
        error = normalize_exception(exc)
        duration_ms = int((time.monotonic() - started) * 1000)
        try:
            append_runtime_event(
                kind="provider_failure",
                code=error.code,
                message=error.message,
                request_id=request_id,
                metadata={
                    "endpoint": "/query/inspect",
                    "answer_mode": req.answer_mode,
                    "status_code": error.status_code,
                    "duration_ms": duration_ms,
                    **ownership,
                },
            )
        except OSError:
            logger.warning("query.inspect_event_log_failed request_id=%s code=%s", request_id, error.code)
        logger.exception("query.inspect_error request_id=%s code=%s", request_id, error.code)
        raise HTTPException(
            status_code=error.status_code,
            detail={
                "code": error.code,
                "message": error.message,
                "request_id": request_id,
            },
        ) from exc
    duration_ms = int((time.monotonic() - started) * 1000)
    logger.info(
        "query.inspect_ok request_id=%s chars=%s citation_ok=%s",
        request_id,
        len(result.answer),
        result.citation_validation.ok,
    )
    record_query_usage(
        endpoint="/query/inspect",
        request_id=request_id,
        answer_mode=req.answer_mode,
        question=req.question,
        answer=result.answer,
        citation_ok=result.citation_validation.ok,
        provider_usage=getattr(result, "provider_usage", None),
        ownership=ownership,
        duration_ms=duration_ms,
    )
    record_result_retrieval_trace(
        endpoint="/query/inspect",
        answer_mode=req.answer_mode,
        result=result,
        provider_called=True,
        duration_ms=duration_ms,
    )
    return QueryInspectResponse(result=result.to_dict(), request_id=request_id)


@app.post("/query/retrieve", response_model=QueryRetrieveResponse, summary="Inspect FluxMind retrieval without LLM generation")
def inspect_retrieval(
    req: QueryRequest,
    response: Response,
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Return retrieved source/page context refs without calling the model provider."""
    verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req)
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    logger.info(
        "query.retrieve_start request_id=%s client=%s chars=%s",
        request_id,
        request.client.host if request.client else "unknown",
        len(req.question),
    )
    started = time.monotonic()
    try:
        retrieval = retrieve_with_metadata(req.question, answer_mode=req.answer_mode)
    except Exception as exc:
        error = normalize_exception(exc)
        logger.exception("query.retrieve_error request_id=%s code=%s", request_id, error.code)
        raise HTTPException(
            status_code=error.status_code,
            detail={
                "code": error.code,
                "message": error.message,
                "request_id": request_id,
            },
        ) from exc
    logger.info(
        "query.retrieve_ok request_id=%s context_count=%s ok=%s",
        request_id,
        retrieval.context_count,
        retrieval.ok,
    )
    duration_ms = int((time.monotonic() - started) * 1000)
    missing_source_page_refs = getattr(retrieval, "missing_source_page_refs", None)
    if missing_source_page_refs is None and hasattr(retrieval, "to_dict"):
        try:
            missing_source_page_refs = retrieval.to_dict().get("missing_source_page_refs", [])
        except (AttributeError, TypeError, ValueError):
            missing_source_page_refs = []
    record_retrieval_trace(
        endpoint="/query/retrieve",
        answer_mode=req.answer_mode,
        context_count=retrieval.context_count,
        missing_source_page_count=len(missing_source_page_refs or []),
        provider_called=False,
        retrieval_ok=retrieval.ok,
        duration_ms=duration_ms,
    )
    return QueryRetrieveResponse(retrieval=retrieval.to_dict(), request_id=request_id)


@app.post("/query/report", summary="Ask FluxMind and download a Markdown report")
def ask_with_report(
    req: QueryRequest,
    response: Response,
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Return a Markdown report with answer, citation validation, and context refs."""
    verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req)
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    logger.info(
        "query.report_start request_id=%s client=%s chars=%s",
        request_id,
        request.client.host if request.client else "unknown",
        len(req.question),
    )
    started = time.monotonic()
    try:
        result = query_with_metadata(req.question, answer_mode=req.answer_mode)
    except Exception as exc:
        error = normalize_exception(exc)
        duration_ms = int((time.monotonic() - started) * 1000)
        try:
            append_runtime_event(
                kind="provider_failure",
                code=error.code,
                message=error.message,
                request_id=request_id,
                metadata={
                    "endpoint": "/query/report",
                    "answer_mode": req.answer_mode,
                    "status_code": error.status_code,
                    "duration_ms": duration_ms,
                    **ownership,
                },
            )
        except OSError:
            logger.warning("query.report_event_log_failed request_id=%s code=%s", request_id, error.code)
        logger.exception("query.report_error request_id=%s code=%s", request_id, error.code)
        raise HTTPException(
            status_code=error.status_code,
            detail={
                "code": error.code,
                "message": error.message,
                "request_id": request_id,
            },
        ) from exc
    duration_ms = int((time.monotonic() - started) * 1000)
    record_query_usage(
        endpoint="/query/report",
        request_id=request_id,
        answer_mode=req.answer_mode,
        question=req.question,
        answer=result.answer,
        citation_ok=result.citation_validation.ok,
        provider_usage=getattr(result, "provider_usage", None),
        ownership=ownership,
        duration_ms=duration_ms,
    )
    record_result_retrieval_trace(
        endpoint="/query/report",
        answer_mode=req.answer_mode,
        result=result,
        provider_called=True,
        duration_ms=duration_ms,
    )
    report = format_query_report(question=req.question, result=result, request_id=request_id)
    logger.info(
        "query.report_ok request_id=%s chars=%s citation_ok=%s",
        request_id,
        len(result.answer),
        result.citation_validation.ok,
    )
    return PlainTextResponse(
        report,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="fluxmind-query-report.md"'},
    )


@app.post("/jobs/image/mock", response_model=JobResponse, summary="Run no-key mock image job")
def create_mock_image_job(
    req: MockImageJobRequest,
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Generate a deterministic local SVG artifact without an external provider key."""
    verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req)
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    existing = existing_idempotent_job("image_generation", req.idempotency_key)
    if existing is not None:
        return JobResponse(job=job_to_dict(existing))
    job = LocalJobRunner().run_mock_image(
        ImageGenerationRequest(
            prompt=req.prompt,
            style=req.style,
            size=req.size,
            diagram_template=req.diagram_template,
            reference_uris=req.reference_uris,
        ),
        request_id=request_id,
        idempotency_key=req.idempotency_key,
        ownership=ownership,
    )
    return JobResponse(job=job_to_dict(job))


@app.post("/jobs/async/image/mock", response_model=JobResponse, summary="Queue no-key mock image job")
def enqueue_mock_image_job(
    req: MockImageJobRequest,
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Queue deterministic local SVG artifact generation without an external provider key."""
    verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req)
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    job = get_async_job_manager().enqueue_mock_image(
        ImageGenerationRequest(
            prompt=req.prompt,
            style=req.style,
            size=req.size,
            diagram_template=req.diagram_template,
            reference_uris=req.reference_uris,
        ),
        request_id=request_id,
        queue_timeout_s=req.queue_timeout_s,
        idempotency_key=req.idempotency_key,
        max_attempts=req.max_attempts,
        retry_backoff_s=req.retry_backoff_s,
        ownership=ownership,
    )
    return JobResponse(job=job_to_dict(job))


@app.post("/jobs/code/python-local", response_model=JobResponse, summary="Run local Python job")
def create_local_python_job(
    req: LocalPythonJobRequest,
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Run a development-only local Python job without hosted sandbox keys."""
    verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req)
    if not req.entrypoint.strip():
        raise HTTPException(status_code=400, detail="Entrypoint cannot be empty")
    existing = existing_idempotent_job("code_execution", req.idempotency_key)
    if existing is not None:
        return JobResponse(job=job_to_dict(existing))
    job = LocalJobRunner().run_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint=req.entrypoint,
            files=req.files,
            timeout_s=req.timeout_s,
            memory_mb=req.memory_mb,
        ),
        request_id=request_id,
        idempotency_key=req.idempotency_key,
        ownership=ownership,
    )
    return JobResponse(job=job_to_dict(job))


@app.post("/jobs/async/code/python-local", response_model=JobResponse, summary="Queue local Python job")
def enqueue_local_python_job(
    req: LocalPythonJobRequest,
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Queue a development-only local Python job without hosted sandbox keys."""
    verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req)
    if not req.entrypoint.strip():
        raise HTTPException(status_code=400, detail="Entrypoint cannot be empty")
    job = get_async_job_manager().enqueue_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint=req.entrypoint,
            files=req.files,
            timeout_s=req.timeout_s,
            memory_mb=req.memory_mb,
        ),
        request_id=request_id,
        queue_timeout_s=req.queue_timeout_s,
        idempotency_key=req.idempotency_key,
        max_attempts=req.max_attempts,
        retry_backoff_s=req.retry_backoff_s,
        ownership=ownership,
    )
    return JobResponse(job=job_to_dict(job))


@app.post("/jobs/code/octave-local", response_model=JobResponse, summary="Run local Octave-compatible job")
def create_local_octave_job(
    req: LocalOctaveJobRequest,
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Run a no-key local GNU Octave-compatible job when octave is installed."""
    verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req)
    if not req.entrypoint.strip():
        raise HTTPException(status_code=400, detail="Entrypoint cannot be empty")
    existing = existing_idempotent_job("code_execution", req.idempotency_key)
    if existing is not None:
        return JobResponse(job=job_to_dict(existing))
    job = LocalJobRunner().run_local_octave(
        CodeExecutionRequest(
            language="octave",
            entrypoint=req.entrypoint,
            files=req.files,
            timeout_s=req.timeout_s,
            memory_mb=req.memory_mb,
        ),
        request_id=request_id,
        idempotency_key=req.idempotency_key,
        ownership=ownership,
    )
    return JobResponse(job=job_to_dict(job))


@app.post("/jobs/async/code/octave-local", response_model=JobResponse, summary="Queue local Octave-compatible job")
def enqueue_local_octave_job(
    req: LocalOctaveJobRequest,
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Queue a no-key local GNU Octave-compatible job when octave is installed."""
    verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req)
    if not req.entrypoint.strip():
        raise HTTPException(status_code=400, detail="Entrypoint cannot be empty")
    job = get_async_job_manager().enqueue_local_octave(
        CodeExecutionRequest(
            language="octave",
            entrypoint=req.entrypoint,
            files=req.files,
            timeout_s=req.timeout_s,
            memory_mb=req.memory_mb,
        ),
        request_id=request_id,
        queue_timeout_s=req.queue_timeout_s,
        idempotency_key=req.idempotency_key,
        max_attempts=req.max_attempts,
        retry_backoff_s=req.retry_backoff_s,
        ownership=ownership,
    )
    return JobResponse(job=job_to_dict(job))


@app.post("/jobs/index/rebuild", response_model=JobResponse, summary="Run local index rebuild job")
def create_index_rebuild_job(
    req: IndexRebuildJobRequest,
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Rebuild the local FAISS index from selected project PDFs as a persisted job."""
    verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req)
    if not req.source_paths:
        raise HTTPException(status_code=400, detail="At least one source path is required")
    existing = existing_idempotent_job("index_rebuild", req.idempotency_key)
    if existing is not None:
        return JobResponse(job=job_to_dict(existing))
    job = LocalJobRunner().run_index_rebuild(
        req.source_paths,
        request_id=request_id,
        idempotency_key=req.idempotency_key,
        ownership=ownership,
    )
    return JobResponse(job=job_to_dict(job))


@app.post("/jobs/async/index/rebuild", response_model=JobResponse, summary="Queue local index rebuild job")
def enqueue_index_rebuild_job(
    req: IndexRebuildJobRequest,
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Queue local FAISS rebuild from selected project PDFs."""
    verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req)
    if not req.source_paths:
        raise HTTPException(status_code=400, detail="At least one source path is required")
    job = get_async_job_manager().enqueue_index_rebuild(
        req.source_paths,
        request_id=request_id,
        queue_timeout_s=req.queue_timeout_s,
        idempotency_key=req.idempotency_key,
        max_attempts=req.max_attempts,
        retry_backoff_s=req.retry_backoff_s,
        ownership=ownership,
    )
    return JobResponse(job=job_to_dict(job))


@app.get("/jobs", response_model=JobListResponse, summary="List local jobs")
def list_jobs(
    q: str | None = None,
    status: str | None = None,
    kind: str | None = None,
    owner_id: str | None = None,
    limit: int = 50,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """List latest local job records."""
    verify_api_token(authorization, x_api_key)
    bounded_limit = min(max(limit, 1), 200)
    jobs = [
        job_to_dict(job)
        for job in LocalJobStore().list_latest(
            limit=bounded_limit,
            status=status,
            kind=kind,
            owner_id=owner_id,
            q=q,
        )
    ]
    return JobListResponse(jobs=jobs)


@app.get("/jobs/{job_id}", response_model=JobResponse, summary="Get local job status")
def get_job(
    job_id: str,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Fetch the latest persisted state for a local job."""
    verify_api_token(authorization, x_api_key)
    job = LocalJobStore().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(job=job_to_dict(job))


@app.post("/jobs/{job_id}/cancel", response_model=JobResponse, summary="Cancel local job")
def cancel_job(
    job_id: str,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Mark a queued/running local job as cancelled."""
    verify_api_token(authorization, x_api_key)
    job = get_async_job_manager().cancel(job_id) or LocalJobStore().cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(job=job_to_dict(job))


@app.post("/jobs/{job_id}/retry", response_model=JobResponse, summary="Retry local job")
def retry_job(
    job_id: str,
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Retry a failed/cancelled local job with a new job ID."""
    verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    job = LocalJobRunner().retry(job_id, request_id=request_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(job=job_to_dict(job))


@app.post("/jobs/{job_id}/retry-scheduled", response_model=JobResponse, summary="Schedule local job retry")
def schedule_retry_job(
    job_id: str,
    req: RetryScheduleRequest,
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Queue a failed/cancelled local job retry after a bounded backoff delay."""
    verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    job = get_async_job_manager().schedule_retry(
        job_id,
        delay_s=req.delay_s,
        queue_timeout_s=req.queue_timeout_s,
        request_id=request_id,
    )
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(job=job_to_dict(job))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    status = startup_warmup_status()
    if status.get("ready"):
        return {"status": "ready", "warmup": status}
    raise HTTPException(status_code=503, detail={"status": status.get("status", "not_ready"), "warmup": status})
