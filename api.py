"""FluxMind API — FastAPI wrapper for RAG pipeline, compatible with Coze custom plugin."""

import os
import logging
import hashlib
import hmac
import json
import re
import sqlite3
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from src.activation_suite import collect_activation_suite, format_activation_suite_markdown
from src.collaboration_readiness import (
    collect_collaboration_readiness,
    format_collaboration_readiness_markdown,
)
from src.openapi_contract import (
    collect_openapi_contract,
    format_openapi_contract_markdown,
    format_openapi_contract_snapshot_verify_markdown,
    verify_openapi_contract_snapshot,
)
from src.quality_readiness import collect_quality_readiness, format_quality_readiness_markdown
from src.product_activation_rehearsal import (
    collect_product_activation_rehearsal,
    format_product_activation_rehearsal_markdown,
)
from src.provider_runtime_rehearsal import (
    collect_provider_runtime_rehearsal,
    format_provider_runtime_rehearsal_markdown,
)
from src.storage_migration import (
    collect_platform_migration_rehearsal,
    format_storage_migration_rehearsal_markdown,
)
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
from src.api_keys import api_key_registry_backend_status, verify_configured_api_key_token
from src.artifacts import (
    LocalArtifactRegistry,
    artifact_to_public_dict,
    job_artifact_to_public_dict,
    safe_artifact_download_filename,
)
from src.capabilities import CodeExecutionRequest, ImageGenerationRequest
from src.chain import get_vector_store, query_with_metadata, retrieve_with_metadata
from src.config import (
    API_ACCESS_AUDIT_ENABLED,
    API_RATE_LIMIT_ENABLED,
    API_RATE_LIMIT_MAX_REQUESTS,
    API_RATE_LIMIT_WINDOW_S,
    FAISS_INDEX_DIR,
    IDENTITY_QUOTAS_BILLING_ENABLED,
    LLM_MODEL,
    PRODUCT_QUOTA_GUARD_ENABLED,
    PRODUCT_QUOTA_METRIC,
    PRODUCT_RBAC_GUARD_ENABLED,
    PRODUCT_REGISTRY_BACKEND,
    QUERY_COST_COMPLETION_USD_PER_1M,
    QUERY_COST_PROMPT_USD_PER_1M,
    QUERY_COST_PROVIDER,
    QUOTA_STORE_BACKEND,
    SHARE_LINK_TOKEN_STORE_BACKEND,
)
from src.costs import summarize_query_cost
from src.jobs import JobRecord, LocalJobRunner, LocalJobStore, get_async_job_manager, normalize_ownership, ownership_from_record
from src.metadata import ChunkMetadataStore, CorpusProfileStore, safe_corpus_profile_report_filename
from src.product_registry import LocalProductRegistry, product_registry_backend_status
from src.runtime import (
    ProviderQuotaGuardError,
    append_runtime_event,
    estimate_text_tokens,
    list_runtime_events,
    logger,
    new_request_id,
    normalize_exception,
    runtime_event_to_safe_dict,
    runtime_ownership_metadata,
)
from src.share_links import LocalShareLinkRegistry, share_link_registry_backend_status
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
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_SENSITIVE_REQUEST_ID_RE = re.compile(
    r"(authorization|bearer|api[-_\s]?key|token|secret|sk-[A-Za-z0-9])",
    re.IGNORECASE,
)
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


def discover_pdfs():
    from src.ingestion import discover_pdfs as _discover_pdfs

    return _discover_pdfs()


def extract_pdf_structure_markers(*args, **kwargs):
    from src.ingestion import extract_pdf_structure_markers as _extract_pdf_structure_markers

    return _extract_pdf_structure_markers(*args, **kwargs)


def refresh_paper_metadata():
    from src.ingestion import refresh_paper_metadata as _refresh_paper_metadata

    return _refresh_paper_metadata()


def resolve_selectable_source_paths(*args, **kwargs):
    from src.ingestion import resolve_selectable_source_paths as _resolve_selectable_source_paths

    return _resolve_selectable_source_paths(*args, **kwargs)


def set_active_paper_source_paths(*args, **kwargs):
    from src.ingestion import set_active_paper_source_paths as _set_active_paper_source_paths

    return _set_active_paper_source_paths(*args, **kwargs)


def public_error_detail(code: str) -> dict[str, str]:
    """Return a stable API error body without exception text or local paths."""
    return {"code": code}


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


def public_request_validation_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project request validation errors without echoing submitted values."""
    public_errors: list[dict[str, Any]] = []
    for error in errors:
        loc = error.get("loc") or []
        if not isinstance(loc, (list, tuple)):
            loc = [loc]
        public_errors.append(
            {
                "type": str(error.get("type") or "validation_error"),
                "loc": [str(part) for part in loc],
                "msg": "Invalid request field.",
            }
        )
    return public_errors


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(_request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": public_request_validation_errors(exc.errors())},
    )


def api_auth_context(
    authorization: str | None,
    x_api_key: str | None,
    *,
    update_registry_usage: bool = False,
) -> dict[str, Any]:
    """Classify API token headers and keep registry owner data internal."""
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

    static_token_valid = any(
        hmac.compare_digest(candidate, API_TOKEN)
        for candidate in (x_api_key or "", bearer_token)
        if candidate
    )
    registry_status = api_key_registry_backend_status()
    registry_configured = bool(registry_status.get("configured") and registry_status.get("supported"))
    registry_record = None
    if registry_configured:
        for candidate in (x_api_key or "", bearer_token):
            if not candidate:
                continue
            registry_record = verify_configured_api_key_token(
                candidate,
                update_usage=update_registry_usage,
            )
            if registry_record is not None:
                break

    token_valid = static_token_valid or registry_record is not None
    auth_configured = bool(API_TOKEN) or registry_configured
    if not auth_configured:
        token_status = "not_configured"
    elif token_valid:
        token_status = "valid"
    elif has_bearer or has_x_api_key:
        token_status = "invalid"
    else:
        token_status = "missing"
    if static_token_valid:
        auth_source = "static_token"
    elif registry_record is not None:
        auth_source = "api_key_registry"
    else:
        auth_source = "none"

    return {
        "token_status": token_status,
        "credential_type": credential_type,
        "credential_present": credential_type != "none",
        "auth_configured": auth_configured,
        "auth_source": auth_source,
        "api_key_registry_configured": registry_configured,
        "auth_key_id": registry_record.key_id if registry_record is not None else "",
        "auth_owner_id": registry_record.owner_id if registry_record is not None else "",
        "auth_owner_label": registry_record.owner_label if registry_record is not None else "",
        "auth_owner_source": "api_key" if registry_record is not None else "none",
    }


def api_token_status(
    authorization: str | None,
    x_api_key: str | None,
    *,
    update_registry_usage: bool = False,
) -> dict[str, Any]:
    """Classify API token headers without returning token values or owners."""
    status = api_auth_context(
        authorization,
        x_api_key,
        update_registry_usage=update_registry_usage,
    )
    return {
        "token_status": status["token_status"],
        "credential_type": status["credential_type"],
        "credential_present": status["credential_present"],
        "auth_configured": status["auth_configured"],
        "auth_source": status["auth_source"],
        "api_key_registry_configured": status["api_key_registry_configured"],
    }


def _clean_request_id(value: str | None) -> str | None:
    request_id = (value or "").strip()
    if not request_id:
        return None
    request_id = request_id[:64]
    if _SENSITIVE_REQUEST_ID_RE.search(request_id):
        return None
    if not _REQUEST_ID_RE.fullmatch(request_id):
        return None
    return request_id


def _api_access_route_metadata(request: Request) -> dict[str, Any]:
    route = request.scope.get("route")
    route_path = str(getattr(route, "path", "") or "").strip()
    if not route_path:
        return {"route_present": False, "route_fingerprint": ""}
    route_fingerprint = hashlib.sha256(route_path.encode("utf-8")).hexdigest()[:12]
    return {"route_present": True, "route_fingerprint": route_fingerprint}


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
    response_request_id = (
        _clean_request_id(response.headers.get("X-Request-ID"))
        if response is not None
        else None
    )
    request_id = response_request_id or _clean_request_id(request.headers.get("X-Request-ID"))
    route_metadata = _api_access_route_metadata(request)
    try:
        append_runtime_event(
            kind="api_access",
            code=f"auth_{token_status['token_status']}",
            message="Metadata-only API access audit event.",
            request_id=request_id,
            metadata={
                "method": request.method,
                **route_metadata,
                "status_code": status_code,
                "duration_ms": duration_ms,
                **token_status,
                **(extra_metadata or {}),
            },
        )
    except OSError:
        logger.warning(
            "api_access.event_log_failed route_present=%s",
            route_metadata["route_present"],
        )


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
    workspace_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Optional local workspace metadata for product quota attribution",
    )
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


class ActiveCorpusRequest(OwnershipRequest):
    source_paths: list[str] = Field(..., description="Project-relative selectable PDF paths to keep active")


class CorpusProfileRequest(OwnershipRequest):
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
    job: dict = Field(..., description="Public no-secret local job record projection")


class JobListResponse(BaseModel):
    jobs: list[dict] = Field(..., description="Latest local job summaries")


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


class ActivationSuiteResponse(BaseModel):
    activation_suite: dict = Field(..., description="No-secret local activation suite status")


class OpenAPIContractResponse(BaseModel):
    openapi_contract: dict = Field(..., description="No-secret OpenAPI contract readiness status")


class OpenAPIContractSnapshotVerifyResponse(BaseModel):
    openapi_contract_snapshot_verify: dict = Field(
        ...,
        description="No-secret OpenAPI contract snapshot verification status",
    )


class QualityReadinessResponse(BaseModel):
    quality_readiness: dict = Field(..., description="No-secret staged quality readiness status")


class ProductActivationRehearsalResponse(BaseModel):
    product_activation_rehearsal: dict = Field(
        ...,
        description="No-secret local product activation rehearsal status",
    )


class CollaborationReadinessResponse(BaseModel):
    collaboration_readiness: dict = Field(
        ...,
        description="No-secret private-corpus and share-link readiness status",
    )


class ProviderRuntimeRehearsalResponse(BaseModel):
    provider_runtime_rehearsal: dict = Field(
        ...,
        description="No-secret local provider runtime rehearsal status",
    )


class PlatformMigrationRehearsalResponse(BaseModel):
    platform_migration_rehearsal: dict = Field(
        ...,
        description="No-secret local platform migration rehearsal status",
    )


class ActivationSuiteRequest(BaseModel):
    live_report: dict | None = Field(
        default=None,
        description="Optional no-secret evaluate_rag JSON report.",
    )
    live_reports: list[dict] = Field(
        default_factory=list,
        max_length=4,
        description="Optional no-secret evaluate_rag JSON reports.",
    )


class QualityReadinessRequest(BaseModel):
    live_report: dict | None = Field(
        default=None,
        description="Optional no-secret evaluate_rag JSON report.",
    )
    live_reports: list[dict] = Field(
        default_factory=list,
        max_length=4,
        description="Optional no-secret evaluate_rag JSON reports.",
    )


class OpenAPIContractSnapshotVerifyRequest(BaseModel):
    snapshot: dict = Field(
        ...,
        description="Prior no-secret OpenAPI contract JSON report.",
    )


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


class ProductRegistryStatusResponse(BaseModel):
    status: dict = Field(..., description="No-secret local product registry status")


class ProductRegistryWorkspaceListResponse(BaseModel):
    status: dict = Field(..., description="No-secret local product registry status")
    workspaces: list[dict] = Field(..., description="Local workspace summaries")


class ProductRegistryWorkspaceResponse(BaseModel):
    workspace: dict = Field(..., description="Local workspace detail")


class ProductRegistryWorkspaceRequest(BaseModel):
    workspace_id: str | None = Field(default=None, min_length=1, max_length=128)
    label: str | None = Field(default=None, min_length=1, max_length=128)
    owner_user_id: str | None = Field(default=None, min_length=1, max_length=128)
    owner_label: str | None = Field(default=None, min_length=1, max_length=128)


class ProductRegistryMemberRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)
    label: str | None = Field(default=None, min_length=1, max_length=128)
    role: str = Field(default="member", description="Local role: owner, admin, member, or viewer")


class ProductRegistryQuotaRequest(BaseModel):
    metric: str = Field(default="requests", min_length=1, max_length=128)
    limit_value: int = Field(..., ge=0, le=1_000_000_000)
    window_s: int = Field(..., ge=0, le=31_536_000)


class ProductRegistryBillingRequest(BaseModel):
    billing_mode: str = Field(default="local-ledger", min_length=1, max_length=128)
    status: str = Field(default="active", description="Local billing status: active or disabled")
    attribution_enabled: bool = True


class ProductRegistryPermissionCheckRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)
    action: str = Field(..., min_length=1, max_length=128)
    workspace_id: str | None = Field(default=None, min_length=1, max_length=128)


class ProductRegistryPermissionResponse(BaseModel):
    permission: dict = Field(..., description="No-secret local product RBAC decision")


class ShareLinkRegistryStatusResponse(BaseModel):
    status: dict = Field(..., description="No-secret local share-link registry status")


class ShareLinkListResponse(BaseModel):
    status: dict = Field(..., description="No-secret local share-link registry status")
    share_links: list[dict] = Field(..., description="Local share-link summaries")


class ShareLinkResponse(BaseModel):
    share_link: dict = Field(..., description="Local share-link summary")


class ShareLinkCreateResponse(BaseModel):
    token: str = Field(..., description="One-time local share token")
    share_link: dict = Field(..., description="Created local share-link summary")


class ShareLinkResolveResponse(BaseModel):
    resolution: dict = Field(..., description="No-secret local share-link token resolution")


class ShareLinkCreateRequest(BaseModel):
    workspace_id: str = Field(..., min_length=1, max_length=128)
    created_by_user_id: str | None = Field(default=None, min_length=1, max_length=128)
    resource_kind: str = Field(default="corpus_profile", min_length=1, max_length=64)
    resource_ref: str = Field(..., min_length=1, max_length=512)
    description: str | None = Field(default=None, max_length=256)
    expires_in_s: int | None = Field(default=None, ge=60, le=31_536_000)
    max_redemptions: int = Field(default=0, ge=0, le=1_000_000)


class ShareLinkResolveRequest(BaseModel):
    token: str = Field(..., min_length=1, max_length=512)
    record_redeem: bool = False


def verify_api_token(authorization: str | None, x_api_key: str | None) -> dict[str, Any]:
    """Protect public Coze/plugin calls when FLUXMIND_API_TOKEN is configured."""
    status = api_auth_context(authorization, x_api_key, update_registry_usage=True)
    if status["token_status"] in {"not_configured", "valid"}:
        return status
    else:
        raise HTTPException(status_code=401, detail="Invalid API token")


def request_id_header(response: Response, x_request_id: str | None) -> str:
    request_id = _clean_request_id(x_request_id) or new_request_id()
    response.headers["X-Request-ID"] = request_id
    return request_id


def request_ownership(req: Any, auth_context: dict[str, Any] | None = None) -> dict[str, str]:
    if not getattr(req, "owner_id", None) and auth_context and auth_context.get("auth_owner_id"):
        return normalize_ownership(
            owner_id=auth_context.get("auth_owner_id"),
            owner_label=auth_context.get("auth_owner_label"),
            ownership_source="api_key",
        )
    return normalize_ownership(
        owner_id=getattr(req, "owner_id", None),
        owner_label=getattr(req, "owner_label", None),
    )


def product_quota_guard_decision(
    *,
    req: Any,
    ownership: dict[str, str],
    auth_context: dict[str, Any],
    endpoint: str,
    amount: int = 1,
) -> dict[str, Any]:
    """Check optional local product quota before expensive query work."""
    if not (IDENTITY_QUOTAS_BILLING_ENABLED and PRODUCT_QUOTA_GUARD_ENABLED):
        return {
            "enabled": False,
            "allowed": True,
            "limited": False,
            "reason": "product_quota_guard_disabled",
        }
    if PRODUCT_REGISTRY_BACKEND.strip().lower() != "sqlite" or QUOTA_STORE_BACKEND.strip().lower() != "sqlite":
        return {
            "enabled": True,
            "allowed": False,
            "limited": False,
            "reason": "product_quota_guard_backend_not_configured",
            "status_code": 503,
        }
    status = product_registry_backend_status(backend="sqlite")
    if not status.get("available"):
        return {
            "enabled": True,
            "allowed": False,
            "limited": False,
            "reason": "product_registry_unavailable",
            "status_code": 503,
        }

    registry = LocalProductRegistry()
    user_id = auth_context.get("auth_owner_id") or ownership.get("owner_id", "local-user")
    workspace_hint = getattr(req, "workspace_id", None)
    membership = registry.workspace_for_user(user_id=user_id, workspace_id=workspace_hint)
    if membership is None:
        return {
            "enabled": True,
            "allowed": False,
            "limited": False,
            "reason": "product_workspace_not_found",
            "status_code": 403,
            "user_id": user_id,
            "workspace_id": workspace_hint or "",
        }
    decision = registry.quota_decision(
        workspace_id=membership["workspace_id"],
        user_id=user_id,
        metric=PRODUCT_QUOTA_METRIC,
        amount=amount,
        source=f"api:{endpoint}",
        record=True,
    )
    decision["role"] = membership.get("role", "")
    return decision


def product_quota_event_metadata(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "product_quota_guard_enabled": bool(decision.get("enabled", False)),
        "product_quota_limited": bool(decision.get("limited", False)),
        "product_quota_reason": decision.get("reason", ""),
        "product_quota_metric": decision.get("metric", ""),
        "product_quota_limit": int(decision.get("limit_value", 0) or 0),
        "product_quota_remaining": int(decision.get("remaining", 0) or 0),
        "product_quota_window_s": int(decision.get("window_s", 0) or 0),
        "product_workspace_present": bool(str(decision.get("workspace_id", "") or "").strip()),
    }


def product_quota_headers(decision: dict[str, Any]) -> dict[str, str]:
    if not decision.get("enabled"):
        return {}
    headers = {"X-Product-Quota-Reason": str(decision.get("reason", ""))}
    if decision.get("quota_configured"):
        headers["X-Product-Quota-Limit"] = str(decision.get("limit_value", 0))
        headers["X-Product-Quota-Remaining"] = str(decision.get("remaining", 0))
        headers["X-Product-Quota-Reset"] = str(decision.get("reset_after_s", 0))
    return headers


def apply_product_quota_headers(response: Response, decision: dict[str, Any]) -> None:
    for key, value in product_quota_headers(decision).items():
        response.headers[key] = value


def enforce_product_quota(
    *,
    req: Any,
    response: Response,
    request_id: str,
    ownership: dict[str, str],
    auth_context: dict[str, Any],
    endpoint: str,
) -> dict[str, Any]:
    decision = product_quota_guard_decision(
        req=req,
        ownership=ownership,
        auth_context=auth_context,
        endpoint=endpoint,
    )
    apply_product_quota_headers(response, decision)
    if decision.get("allowed", True):
        return decision
    status_code = int(
        decision.get(
            "status_code",
            429 if decision.get("limited", False) else 403,
        )
    )
    try:
        append_runtime_event(
            kind="product_quota",
            code=str(decision.get("reason", "product_quota_denied")),
            message="Metadata-only product quota guard event.",
            request_id=request_id,
            metadata={
                "endpoint": endpoint,
                "status_code": status_code,
                **product_quota_event_metadata(decision),
                **runtime_ownership_metadata(ownership),
            },
        )
    except OSError:
        logger.warning("product_quota.event_log_failed request_id=%s endpoint=%s", request_id, endpoint)
    raise HTTPException(
        status_code=status_code,
        detail={
            "code": decision.get("reason", "product_quota_denied"),
            "message": "Product quota guard denied this request.",
            "request_id": request_id,
        },
        headers=product_quota_headers(decision),
    )


def product_rbac_guard_decision(
    *,
    req: Any | None,
    ownership: dict[str, str],
    auth_context: dict[str, Any],
    endpoint: str,
    action: str,
) -> dict[str, Any]:
    """Check optional local product RBAC before product-scoped work."""
    if not (IDENTITY_QUOTAS_BILLING_ENABLED and PRODUCT_RBAC_GUARD_ENABLED):
        return {
            "enabled": False,
            "allowed": True,
            "reason": "product_rbac_guard_disabled",
            "action": action,
        }
    if PRODUCT_REGISTRY_BACKEND.strip().lower() != "sqlite":
        return {
            "enabled": True,
            "allowed": False,
            "reason": "product_rbac_guard_backend_not_configured",
            "action": action,
            "status_code": 503,
        }
    status = product_registry_backend_status(backend="sqlite")
    if not status.get("available"):
        return {
            "enabled": True,
            "allowed": False,
            "reason": "product_registry_unavailable",
            "action": action,
            "status_code": 503,
        }

    user_id = auth_context.get("auth_owner_id")
    if not user_id:
        return {
            "enabled": True,
            "allowed": False,
            "reason": "product_identity_not_authenticated",
            "action": action,
            "status_code": 403,
        }
    workspace_hint = getattr(req, "workspace_id", None) if req is not None else None
    decision = LocalProductRegistry().permission_decision(
        user_id=user_id,
        workspace_id=workspace_hint,
        action=action,
    )
    decision["enabled"] = True
    decision["endpoint"] = endpoint
    decision.setdefault("status_code", 403)
    return decision


def product_rbac_event_metadata(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "product_rbac_guard_enabled": bool(decision.get("enabled", False)),
        "product_rbac_reason": decision.get("reason", ""),
        "product_rbac_action": decision.get("action", ""),
        "product_rbac_role": decision.get("role", ""),
        "product_rbac_required_roles": ",".join(decision.get("required_roles", []) or []),
        "product_workspace_present": bool(str(decision.get("workspace_id", "") or "").strip()),
    }


def product_rbac_headers(decision: dict[str, Any]) -> dict[str, str]:
    if not decision.get("enabled"):
        return {}
    headers = {"X-Product-RBAC-Reason": str(decision.get("reason", ""))}
    if decision.get("role"):
        headers["X-Product-RBAC-Role"] = str(decision.get("role", ""))
    if decision.get("action"):
        headers["X-Product-RBAC-Action"] = str(decision.get("action", ""))
    return headers


def apply_product_rbac_headers(response: Response, decision: dict[str, Any]) -> None:
    for key, value in product_rbac_headers(decision).items():
        response.headers[key] = value


def enforce_product_rbac(
    *,
    req: Any | None,
    response: Response,
    request_id: str,
    ownership: dict[str, str],
    auth_context: dict[str, Any],
    endpoint: str,
    action: str,
) -> dict[str, Any]:
    decision = product_rbac_guard_decision(
        req=req,
        ownership=ownership,
        auth_context=auth_context,
        endpoint=endpoint,
        action=action,
    )
    apply_product_rbac_headers(response, decision)
    if decision.get("allowed", True):
        return decision
    status_code = int(decision.get("status_code", 403))
    try:
        append_runtime_event(
            kind="product_rbac",
            code=str(decision.get("reason", "product_rbac_denied")),
            message="Metadata-only product RBAC guard event.",
            request_id=request_id,
            metadata={
                "endpoint": endpoint,
                "status_code": status_code,
                **product_rbac_event_metadata(decision),
                **runtime_ownership_metadata(ownership),
            },
        )
    except OSError:
        logger.warning("product_rbac.event_log_failed request_id=%s endpoint=%s", request_id, endpoint)
    raise HTTPException(
        status_code=status_code,
        detail={
            "code": decision.get("reason", "product_rbac_denied"),
            "message": "Product RBAC guard denied this request.",
            "request_id": request_id,
        },
        headers=product_rbac_headers(decision),
    )


def enforce_product_registry_admin_read(
    *,
    response: Response,
    request_id: str,
    auth_context: dict[str, Any],
    endpoint: str,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Apply local product-admin RBAC to registry read/inspection routes."""
    ownership = request_ownership(None, auth_context)
    req = SimpleNamespace(workspace_id=workspace_id) if workspace_id else None
    return enforce_product_rbac(
        req=req,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint=endpoint,
        action="admin_write",
    )


def product_registry_admin_status() -> dict[str, Any]:
    return product_registry_backend_status(backend=PRODUCT_REGISTRY_BACKEND)


def require_local_product_registry() -> LocalProductRegistry:
    status = product_registry_admin_status()
    if not status.get("available"):
        raise HTTPException(
            status_code=503,
            detail={
                "code": status.get("reason", "product_registry_unavailable"),
                "message": "Local product registry is not available.",
                "status": status,
            },
        )
    return LocalProductRegistry()


def record_product_registry_admin_event(
    *,
    action: str,
    status_code: int,
    request_id: str | None = None,
    workspace_id: str = "",
    reason: str = "ok",
) -> None:
    try:
        append_runtime_event(
            kind="product_registry_admin",
            code=reason,
            message="Metadata-only product registry admin event.",
            request_id=request_id,
            metadata={
                "action": action,
                "status_code": status_code,
                "product_workspace_present": bool(str(workspace_id or "").strip()),
                "product_registry_backend": PRODUCT_REGISTRY_BACKEND,
                "content_exported": False,
                "secrets_exported": False,
            },
        )
    except OSError:
        logger.warning("product_registry_admin.event_log_failed action=%s", action)


def share_link_registry_admin_status() -> dict[str, Any]:
    return share_link_registry_backend_status(backend=SHARE_LINK_TOKEN_STORE_BACKEND)


def require_local_share_link_registry() -> LocalShareLinkRegistry:
    status = share_link_registry_admin_status()
    if not status.get("available"):
        raise HTTPException(
            status_code=503,
            detail={
                "code": status.get("reason", "share_link_token_store_unavailable"),
                "message": "Local share-link registry is not available.",
                "status": status,
            },
        )
    return LocalShareLinkRegistry()


def record_share_link_admin_event(
    *,
    action: str,
    status_code: int,
    request_id: str | None = None,
    workspace_id: str = "",
    workspace_present: bool | None = None,
    link_id: str = "",
    reason: str = "ok",
    share_link_valid: bool | None = None,
) -> None:
    metadata = {
        "action": action,
        "status_code": status_code,
        "product_workspace_present": (
            bool(workspace_present)
            if workspace_present is not None
            else bool(str(workspace_id or "").strip())
        ),
        "share_link_present": bool(str(link_id or "").strip()),
        "share_link_backend": SHARE_LINK_TOKEN_STORE_BACKEND,
        "content_exported": False,
        "secrets_exported": False,
        "share_tokens_exported": False,
        "share_urls_exported": False,
    }
    if share_link_valid is not None:
        metadata["share_link_valid"] = bool(share_link_valid)
    try:
        append_runtime_event(
            kind="share_link_admin",
            code=reason,
            message="Metadata-only share-link admin event.",
            request_id=request_id,
            metadata=metadata,
        )
    except OSError:
        logger.warning("share_link_admin.event_log_failed action=%s", action)


def _safe_event_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _safe_event_list_count(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def record_admin_check_event(
    *,
    check: str,
    ok: bool,
    metadata: dict[str, Any] | None = None,
    status_code: int = 200,
) -> None:
    """Append a metadata-only audit event for admin readiness checks."""
    if not API_ACCESS_AUDIT_ENABLED:
        return
    safe_check = re.sub(r"[^a-z0-9_]+", "_", str(check).casefold()).strip("_") or "admin"
    try:
        append_runtime_event(
            kind="admin_check",
            code=f"{safe_check}_{'ok' if ok else 'blocked'}",
            message="Metadata-only admin readiness check event.",
            metadata={
                "check": safe_check,
                "ok": bool(ok),
                "status_code": status_code,
                **(metadata or {}),
                "content_exported": False,
                "secrets_exported": False,
                "paths_exported": False,
            },
        )
    except OSError:
        logger.warning("admin_check.event_log_failed check=%s", safe_check)


def _record_openapi_contract_check(status: dict[str, Any]) -> None:
    record_admin_check_event(
        check="openapi_contract",
        ok=bool(status.get("local_contract_ready")),
        metadata={
            "route_count": _safe_event_int(status.get("route_count")),
            "operation_count": _safe_event_int(status.get("operation_count")),
            "required_operation_missing_count": _safe_event_int(
                status.get("required_operation_missing_count")
            ),
            "undocumented_operation_count": _safe_event_int(
                status.get("undocumented_operation_count")
            ),
            "response_missing_operation_count": _safe_event_int(
                status.get("response_missing_operation_count")
            ),
            "protected_operation_count": _safe_event_int(
                status.get("protected_operation_count")
            ),
            "protected_auth_header_operation_count": _safe_event_int(
                status.get("protected_auth_header_operation_count")
            ),
            "blocker_count": _safe_event_list_count(status.get("blockers")),
        },
    )


def _record_openapi_snapshot_check(status: dict[str, Any]) -> None:
    record_admin_check_event(
        check="openapi_contract_snapshot_verify",
        ok=bool(status.get("ok")),
        metadata={
            "diff_count": _safe_event_int(status.get("diff_count")),
            "compared_field_count": _safe_event_int(status.get("compared_field_count")),
            "snapshot_shape_valid": bool(status.get("snapshot_shape_valid")),
            "snapshot_raw_schema_included": bool(status.get("snapshot_raw_schema_included")),
            "blocker_count": _safe_event_list_count(status.get("blockers")),
        },
    )


def _record_quality_readiness_check(status: dict[str, Any]) -> None:
    record_admin_check_event(
        check="quality_readiness",
        ok=bool(status.get("local_foundation_ready")),
        metadata={
            "local_foundation_ready": bool(status.get("local_foundation_ready")),
            "small_group_ready": bool(status.get("small_group_ready")),
            "community_ready": bool(status.get("community_ready")),
            "live_evidence_included": bool(status.get("live_evidence_included")),
            "evidence_request_count": _safe_event_list_count(status.get("evidence_requests")),
        },
    )


def _record_product_activation_check(status: dict[str, Any]) -> None:
    readiness = status.get("readiness", {}) or {}
    lifecycle = status.get("api_key_lifecycle", {}) or {}
    registry = status.get("product_registry", {}) or {}
    record_admin_check_event(
        check="product_activation_rehearsal",
        ok=bool(status.get("ok")),
        metadata={
            "local_foundation_ready": bool(readiness.get("local_foundation_ready")),
            "activation_ready": bool(readiness.get("activation_ready")),
            "active_key_count": _safe_event_int(lifecycle.get("active_key_count")),
            "workspace_count": _safe_event_int(registry.get("workspace_count")),
        },
    )


def _record_collaboration_readiness_check(status: dict[str, Any]) -> None:
    summary = status.get("summary", {}) or {}
    blockers = status.get("blockers", {}) or {}
    record_admin_check_event(
        check="collaboration_readiness",
        ok=bool(status.get("ok")),
        metadata={
            "local_foundation_ready": bool(status.get("local_foundation_ready")),
            "safe_default_ready": bool(status.get("safe_default_ready")),
            "activation_ready": bool(status.get("activation_ready")),
            "private_corpora_enabled": bool(summary.get("private_corpora_enabled")),
            "share_links_enabled": bool(summary.get("share_links_enabled")),
            "policy_scenario_count": _safe_event_int(summary.get("policy_scenario_count")),
            "activation_blocker_count": _safe_event_list_count(
                blockers.get("activation")
            ),
        },
    )


def _record_provider_runtime_check(status: dict[str, Any]) -> None:
    readiness = status.get("readiness", {}) or {}
    docker = status.get("docker_execution", {}) or {}
    record_admin_check_event(
        check="provider_runtime_rehearsal",
        ok=bool(status.get("ok")),
        metadata={
            "local_foundation_ready": bool(readiness.get("local_foundation_ready")),
            "external_activation_ready": bool(status.get("external_activation_ready")),
            "docker_available": bool(docker.get("available")),
        },
    )


def _record_platform_migration_check(status: dict[str, Any]) -> None:
    summary = status.get("summary", {}) or {}
    record_admin_check_event(
        check="platform_migration_rehearsal",
        ok=bool(status.get("rehearsal_ok")),
        metadata={
            "source_preflight_ok": bool(summary.get("source_preflight_ok")),
            "restore_check_ok": bool(summary.get("restore_check_ok")),
            "object_manifest_ready": bool(summary.get("object_manifest_ready")),
            "job_store_manifest_ready": bool(summary.get("job_store_manifest_ready")),
            "copied_files": _safe_event_int(summary.get("copied_files")),
            "blocker_count": _safe_event_list_count(status.get("blockers")),
        },
    )


def _record_activation_suite_check(status: dict[str, Any]) -> None:
    action_plan = status.get("activation_action_plan", {}) or {}
    blockers = status.get("blockers", {}) or {}
    local_foundation_blockers = blockers.get("local_foundation")
    full_activation_blockers = blockers.get("full_activation")
    record_admin_check_event(
        check="activation_suite",
        ok=bool(status.get("local_foundation_ready")),
        metadata={
            "local_foundation_ready": bool(status.get("local_foundation_ready")),
            "full_activation_ready": bool(status.get("full_activation_ready")),
            "failed_check_count": _safe_event_int(
                status.get("failed_check_count")
                if "failed_check_count" in status
                else _safe_event_list_count(local_foundation_blockers)
            ),
            "full_activation_blocker_count": _safe_event_int(
                status.get("full_activation_blocker_count")
                if "full_activation_blocker_count" in status
                else _safe_event_list_count(full_activation_blockers)
            ),
            "activation_step_count": _safe_event_int(action_plan.get("step_count")),
        },
    )


def record_query_exception_event(
    *,
    exc: Exception,
    endpoint: str,
    request_id: str,
    answer_mode: str,
    duration_ms: int,
    ownership: dict[str, str],
) -> Any:
    """Record query failures without misclassifying local guard denials."""
    error = normalize_exception(exc)
    event_kind = "provider_quota_guard" if isinstance(exc, ProviderQuotaGuardError) else "provider_failure"
    message = (
        "Metadata-only provider quota/cost guard denial."
        if event_kind == "provider_quota_guard"
        else error.message
    )
    metadata: dict[str, Any] = {
        "endpoint": endpoint,
        "answer_mode": answer_mode,
        "status_code": error.status_code,
        "duration_ms": duration_ms,
        **runtime_ownership_metadata(ownership),
    }
    if isinstance(exc, ProviderQuotaGuardError):
        decision = dict(getattr(exc, "decision", {}) or {})
        metadata.update(
            {
                "provider_quota_guard_enabled": bool(decision.get("enabled", False)),
                "provider_quota_limited": bool(decision.get("limited", False)),
                "provider_quota_reason": decision.get("reason", error.code),
                "provider_operation": decision.get("operation", ""),
                "provider_label": decision.get("provider", ""),
                "estimated_prompt_tokens": int(decision.get("estimated_prompt_tokens", 0) or 0),
                "requested_completion_tokens": int(decision.get("requested_completion_tokens", 0) or 0),
                "estimated_total_tokens": int(decision.get("estimated_total_tokens", 0) or 0),
                "max_prompt_tokens_per_request": int(
                    decision.get("max_prompt_tokens_per_request", 0) or 0
                ),
                "max_completion_tokens_per_request": int(
                    decision.get("max_completion_tokens_per_request", 0) or 0
                ),
                "cost_limit_configured": bool(decision.get("cost_limit_configured", False)),
                "pricing_configured": bool(decision.get("pricing_configured", False)),
            }
        )
    append_runtime_event(
        kind=event_kind,
        code=error.code,
        message=message,
        request_id=request_id,
        metadata=metadata,
    )
    return error


def job_to_dict(record: JobRecord) -> dict:
    ownership = ownership_from_record(record)
    idempotency_key = record.idempotency_key or ""
    return {
        "job_id": record.job_id,
        "kind": record.kind,
        "status": record.status,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "request": public_job_request(record),
        "result": public_job_result(record),
        "artifacts": [
            job_artifact_to_public_dict(record, artifact)
            for artifact in record.artifacts
        ],
        "error": public_job_error(record),
        "attempts": record.attempts,
        "request_id": record.request_id,
        "parent_job_id": record.parent_job_id,
        "not_before": record.not_before,
        "deadline_at": record.deadline_at,
        "idempotency_key_present": bool(idempotency_key),
        "idempotency_key_fingerprint": (
            hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:16]
            if idempotency_key
            else ""
        ),
        "idempotency_key_exported": False,
        "max_attempts": record.max_attempts,
        "retry_backoff_s": record.retry_backoff_s,
        "dead_lettered_at": record.dead_lettered_at,
        "owner_id": ownership["owner_id"],
        "owner_label": ownership["owner_label"],
        "ownership_source": ownership["ownership_source"],
        "logs": public_job_logs(record),
    }


def _source_path_count(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _string_byte_count(value: Any) -> int:
    return len(str(value).encode("utf-8")) if isinstance(value, str) else 0


def _request_files_summary(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {"input_file_count": 0, "input_total_bytes": 0}
    return {
        "input_file_count": len(value),
        "input_total_bytes": sum(_string_byte_count(content) for content in value.values()),
    }


def _public_scalar(value: Any) -> Any:
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return str(value)


PUBLIC_CODE_RUNTIME_METADATA_KEYS = {
    "language",
    "input_file_count",
    "input_total_bytes",
    "provider_runtime",
    "runtime_available",
    "filesystem_isolation",
    "network_policy_enforced",
    "timeout_s",
    "cpu_time_s",
    "memory_mb",
    "memory_limit_enforced",
    "cpu_limit_enforced",
    "max_files",
    "max_file_bytes",
    "max_total_file_bytes",
    "max_stdout_bytes",
    "max_stderr_bytes",
    "max_artifacts",
    "max_artifact_bytes",
    "max_artifact_total_bytes",
    "max_artifact_candidates",
    "execution_policy",
    "execution_policy_enforced",
    "execution_policy_checked_files",
    "execution_policy_violations",
    "policy_violation",
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
    "docker_network",
    "runtime",
    "cost_estimate_usd",
}


def public_code_runtime_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    return {
        str(key): _public_scalar(value)
        for key, value in metadata.items()
        if key in PUBLIC_CODE_RUNTIME_METADATA_KEYS
    }


def public_job_request(record: JobRecord) -> dict[str, Any]:
    request = record.request if isinstance(record.request, dict) else {}
    if record.kind == "image_generation":
        prompt = request.get("prompt")
        references = request.get("reference_uris")
        return {
            "prompt_present": bool(prompt),
            "prompt_length": len(prompt) if isinstance(prompt, str) else 0,
            "style_present": bool(request.get("style")),
            "size_present": bool(request.get("size")),
            "diagram_template_present": bool(request.get("diagram_template")),
            "reference_count": len(references) if isinstance(references, list) else 0,
        }
    if record.kind == "code_execution":
        summary = _request_files_summary(request.get("files"))
        return {
            "language": str(request.get("language") or ""),
            "entrypoint_present": bool(request.get("entrypoint")),
            **summary,
            "timeout_s": _public_scalar(request.get("timeout_s")),
            "memory_mb": _public_scalar(request.get("memory_mb")),
        }
    if record.kind != "index_rebuild":
        return {}
    source_paths = (
        request.get("source_paths")
        if isinstance(request, dict)
        else None
    )
    return {"source_path_count": _source_path_count(source_paths)}


def public_job_result(record: JobRecord) -> dict[str, Any] | None:
    if record.result is None:
        return None
    if record.kind == "code_execution":
        result = record.result if isinstance(record.result, dict) else {}
        stdout = result.get("stdout")
        stderr = result.get("stderr")
        metadata = public_code_runtime_metadata(result.get("runtime_metadata"))
        return {
            "exit_code": _public_scalar(result.get("exit_code")),
            "stdout_present": bool(stdout),
            "stderr_present": bool(stderr),
            "stdout_bytes": _public_scalar(
                metadata.get("stdout_bytes", _string_byte_count(stdout))
            ),
            "stderr_bytes": _public_scalar(
                metadata.get("stderr_bytes", _string_byte_count(stderr))
            ),
            "stdout_truncated": _public_scalar(metadata.get("stdout_truncated", "false")),
            "stderr_truncated": _public_scalar(metadata.get("stderr_truncated", "false")),
            "output_truncated": _public_scalar(metadata.get("output_truncated", "false")),
            "artifact_count": len(record.artifacts),
            "runtime_metadata": metadata,
        }
    if record.kind != "index_rebuild":
        return record.result
    result = dict(record.result)
    source_paths = result.pop("source_paths", None)
    result["source_path_count"] = _source_path_count(source_paths)
    return result


PUBLIC_JOB_ERROR_MESSAGES = {
    "cancelled": "Job was cancelled.",
    "execution_failed": "Execution failed.",
    "execution_policy_violation": "Execution policy rejected the request.",
    "execution_timeout": "Execution timed out.",
    "job_deadline_exceeded": "Job deadline exceeded before execution.",
    "runtime_unavailable": "Execution runtime unavailable.",
}


def public_job_error(record: JobRecord) -> dict[str, Any] | None:
    if not isinstance(record.error, dict):
        return None
    code = str(record.error.get("code") or "job_failed")
    message = PUBLIC_JOB_ERROR_MESSAGES.get(code, "Job failed.")
    raw_message = record.error.get("message")
    return {
        "code": code,
        "message": message,
        "message_redacted": bool(raw_message and raw_message != message),
    }


PUBLIC_JOB_LOG_METADATA_KEYS = {
    "artifact_count",
    "attempt",
    "error_code",
    "exit_code",
    "max_attempts",
    "retry_backoff_s",
}


def public_job_logs(record: JobRecord) -> list[dict[str, Any]]:
    public_logs: list[dict[str, Any]] = []
    for entry in record.logs:
        if not isinstance(entry, dict):
            continue
        public_entry = {
            "created_at": entry.get("created_at"),
            "status": entry.get("status"),
            "message": entry.get("message"),
        }
        metadata = entry.get("metadata")
        if isinstance(metadata, dict):
            public_metadata = {
                str(key): _public_scalar(value)
                for key, value in metadata.items()
                if key in PUBLIC_JOB_LOG_METADATA_KEYS
            }
            if public_metadata:
                public_entry["metadata"] = public_metadata
        public_logs.append(public_entry)
    return public_logs


def job_summary_to_dict(record: JobRecord) -> dict:
    ownership = ownership_from_record(record)
    error_code = None
    if isinstance(record.error, dict):
        error_code = record.error.get("code")
    return {
        "job_id": record.job_id,
        "kind": record.kind,
        "status": record.status,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "artifacts": [
            job_artifact_to_public_dict(record, artifact)
            for artifact in record.artifacts
        ],
        "error": {"code": error_code} if error_code else None,
        "attempts": record.attempts,
        "request_id_present": bool(record.request_id),
        "parent_job_id": record.parent_job_id,
        "not_before": record.not_before,
        "deadline_at": record.deadline_at,
        "idempotency_key_present": bool(record.idempotency_key),
        "max_attempts": record.max_attempts,
        "retry_backoff_s": record.retry_backoff_s,
        "dead_lettered_at": record.dead_lettered_at,
        "owner_id_present": bool(ownership["owner_id"]),
        "owner_label_present": bool(ownership["owner_label"]),
        "ownership_source": ownership["ownership_source"],
        "log_statuses": [
            str(entry.get("status"))
            for entry in record.logs
            if isinstance(entry, dict) and entry.get("status")
        ],
    }


def existing_idempotent_job(kind: str, idempotency_key: str | None) -> JobRecord | None:
    if not idempotency_key:
        return None
    return LocalJobStore().find_by_idempotency_key(kind=kind, key=idempotency_key)


def record_to_dict(record) -> dict:
    return asdict(record)


def runtime_event_to_dict(event) -> dict:
    return runtime_event_to_safe_dict(event, include_request_id=True)


def runtime_event_matches_safe_query(event: dict[str, Any], query: str) -> bool:
    searchable = json.dumps(event, ensure_ascii=False, sort_keys=True).casefold()
    return query.casefold() in searchable


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
            raise HTTPException(
                status_code=400,
                detail=public_error_detail("invalid_corpus_source_path"),
            ) from exc
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
    metadata.update(runtime_ownership_metadata(normalize_ownership(**(ownership or {}))))
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
        artifact_to_public_dict(artifact)
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
        raise HTTPException(
            status_code=400,
            detail=public_error_detail("artifact_export_denied"),
        ) from exc
    return FileResponse(
        path,
        media_type=artifact.mime_type,
        filename=safe_artifact_download_filename(artifact, path),
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
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Persist active/deactivated papers without requiring filesystem edits."""
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req, auth_context)
    enforce_product_rbac(
        req=req,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/corpus/active",
        action="corpus_write",
    )
    if not req.source_paths:
        raise HTTPException(status_code=400, detail="At least one source path is required")
    try:
        papers = [record_to_dict(record) for record in set_active_paper_source_paths(req.source_paths)]
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=public_error_detail("invalid_corpus_source_path"),
        ) from exc
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
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Persist a named local corpus selection without changing the active FAISS index."""
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req, auth_context)
    enforce_product_rbac(
        req=req,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/corpus/profiles",
        action="corpus_write",
    )
    try:
        source_paths = validate_corpus_profile_source_paths(req.source_paths)
        profile = CorpusProfileStore().upsert_profile(
            name=req.name,
            profile_id=req.profile_id,
            description=req.description,
            source_paths=source_paths,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=public_error_detail("invalid_corpus_source_path"),
        ) from exc
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
    status = corpus_profile_status(profile_id)
    report = format_corpus_profile_status_report(status)
    report_profile_id = str(status.get("profile", {}).get("profile_id") or profile_id)
    return PlainTextResponse(
        report,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{safe_corpus_profile_report_filename(report_profile_id)}"'
            )
        },
    )


@app.post(
    "/corpus/profiles/{profile_id}/activate",
    response_model=ActiveCorpusResponse,
    summary="Activate local corpus profile",
)
def activate_corpus_profile(
    profile_id: str,
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Apply a saved corpus profile to the active local selection."""
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(None, auth_context)
    enforce_product_rbac(
        req=None,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint=f"/corpus/profiles/{profile_id}/activate",
        action="corpus_write",
    )
    try:
        profile = CorpusProfileStore().get_profile(profile_id)
        papers = [record_to_dict(record) for record in set_active_paper_source_paths(profile.source_paths)]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Corpus profile not found") from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=public_error_detail("invalid_corpus_source_path"),
        ) from exc
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
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req, auth_context)
    enforce_product_rbac(
        req=req,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint=f"/corpus/profiles/{profile_id}/rebuild",
        action="corpus_write",
    )
    try:
        profile = CorpusProfileStore().get_profile(profile_id)
        papers = [record_to_dict(record) for record in set_active_paper_source_paths(profile.source_paths)]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Corpus profile not found") from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=public_error_detail("invalid_corpus_source_path"),
        ) from exc
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


@app.get(
    "/admin/openapi-contract",
    response_model=OpenAPIContractResponse,
    summary="Collect local OpenAPI contract readiness",
)
def admin_openapi_contract(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return no-secret OpenAPI contract readiness for API integration work."""
    verify_api_token(authorization, x_api_key)
    status = collect_openapi_contract(app.openapi())
    _record_openapi_contract_check(status)
    return OpenAPIContractResponse(openapi_contract=status)


@app.get("/admin/openapi-contract/report", summary="Download local OpenAPI contract report")
def admin_openapi_contract_report(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return OpenAPI contract readiness as no-secret Markdown."""
    verify_api_token(authorization, x_api_key)
    status = collect_openapi_contract(app.openapi())
    _record_openapi_contract_check(status)
    report = format_openapi_contract_markdown(status)
    return PlainTextResponse(
        report,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="fluxmind-openapi-contract.md"'},
    )


@app.post(
    "/admin/openapi-contract/verify",
    response_model=OpenAPIContractSnapshotVerifyResponse,
    summary="Verify local OpenAPI contract snapshot",
)
def admin_openapi_contract_snapshot_verify(
    req: OpenAPIContractSnapshotVerifyRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Compare the current OpenAPI contract with a prior no-secret report."""
    verify_api_token(authorization, x_api_key)
    current = collect_openapi_contract(app.openapi())
    status = verify_openapi_contract_snapshot(current, req.snapshot)
    _record_openapi_snapshot_check(status)
    return OpenAPIContractSnapshotVerifyResponse(
        openapi_contract_snapshot_verify=status
    )


@app.post(
    "/admin/openapi-contract/verify/report",
    summary="Download local OpenAPI contract snapshot verification report",
)
def admin_openapi_contract_snapshot_verify_report(
    req: OpenAPIContractSnapshotVerifyRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return OpenAPI contract snapshot verification as no-secret Markdown."""
    verify_api_token(authorization, x_api_key)
    current = collect_openapi_contract(app.openapi())
    status = verify_openapi_contract_snapshot(current, req.snapshot)
    _record_openapi_snapshot_check(status)
    report = format_openapi_contract_snapshot_verify_markdown(status)
    return PlainTextResponse(
        report,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="fluxmind-openapi-contract-verify.md"'
        },
    )


@app.get(
    "/admin/quality-readiness",
    response_model=QualityReadinessResponse,
    summary="Collect local quality readiness",
)
def admin_quality_readiness(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return no-secret staged quality readiness on demand."""
    verify_api_token(authorization, x_api_key)
    status = collect_quality_readiness()
    _record_quality_readiness_check(status)
    return QualityReadinessResponse(quality_readiness=status)


@app.post(
    "/admin/quality-readiness",
    response_model=QualityReadinessResponse,
    summary="Collect local quality readiness with live evidence",
)
def admin_quality_readiness_with_report(
    req: QualityReadinessRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return quality readiness with supplied no-secret eval report evidence."""
    verify_api_token(authorization, x_api_key)
    live_reports = list(req.live_reports)
    if req.live_report is not None:
        live_reports.insert(0, req.live_report)
    status = collect_quality_readiness(live_reports=live_reports)
    _record_quality_readiness_check(status)
    return QualityReadinessResponse(quality_readiness=status)


@app.get("/admin/quality-readiness/report", summary="Download local quality readiness report")
def admin_quality_readiness_report(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return staged quality readiness as no-secret Markdown."""
    verify_api_token(authorization, x_api_key)
    status = collect_quality_readiness()
    _record_quality_readiness_check(status)
    report = format_quality_readiness_markdown(status)
    return PlainTextResponse(
        report,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="fluxmind-quality-readiness.md"'},
    )


@app.post("/admin/quality-readiness/report", summary="Download local quality readiness report with live evidence")
def admin_quality_readiness_report_with_report(
    req: QualityReadinessRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return a no-secret quality-readiness report with supplied eval evidence."""
    verify_api_token(authorization, x_api_key)
    live_reports = list(req.live_reports)
    if req.live_report is not None:
        live_reports.insert(0, req.live_report)
    status = collect_quality_readiness(live_reports=live_reports)
    _record_quality_readiness_check(status)
    report = format_quality_readiness_markdown(status)
    return PlainTextResponse(
        report,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="fluxmind-quality-readiness.md"'},
    )


@app.get(
    "/admin/product-activation-rehearsal",
    response_model=ProductActivationRehearsalResponse,
    summary="Run local product activation rehearsal",
)
def admin_product_activation_rehearsal(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Run the disposable no-secret product activation rehearsal on demand."""
    verify_api_token(authorization, x_api_key)
    status = collect_product_activation_rehearsal()
    _record_product_activation_check(status)
    return ProductActivationRehearsalResponse(product_activation_rehearsal=status)


@app.get(
    "/admin/product-activation-rehearsal/report",
    summary="Download local product activation rehearsal report",
)
def admin_product_activation_rehearsal_report(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return product activation rehearsal as no-secret Markdown."""
    verify_api_token(authorization, x_api_key)
    status = collect_product_activation_rehearsal()
    _record_product_activation_check(status)
    report = format_product_activation_rehearsal_markdown(status)
    return PlainTextResponse(
        report,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="fluxmind-product-activation-rehearsal.md"'
        },
    )


@app.get(
    "/admin/collaboration-readiness",
    response_model=CollaborationReadinessResponse,
    summary="Collect local collaboration readiness",
)
def admin_collaboration_readiness(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return no-secret readiness for private corpora and share links."""
    verify_api_token(authorization, x_api_key)
    status = collect_collaboration_readiness()
    _record_collaboration_readiness_check(status)
    return CollaborationReadinessResponse(collaboration_readiness=status)


@app.get(
    "/admin/collaboration-readiness/report",
    summary="Download local collaboration readiness report",
)
def admin_collaboration_readiness_report(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return collaboration readiness as no-secret Markdown."""
    verify_api_token(authorization, x_api_key)
    status = collect_collaboration_readiness()
    _record_collaboration_readiness_check(status)
    report = format_collaboration_readiness_markdown(status)
    return PlainTextResponse(
        report,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="fluxmind-collaboration-readiness.md"'
        },
    )


@app.get(
    "/admin/provider-runtime-rehearsal",
    response_model=ProviderRuntimeRehearsalResponse,
    summary="Run local provider runtime rehearsal",
)
def admin_provider_runtime_rehearsal(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Run the disposable no-secret provider runtime rehearsal on demand."""
    verify_api_token(authorization, x_api_key)
    status = collect_provider_runtime_rehearsal()
    _record_provider_runtime_check(status)
    return ProviderRuntimeRehearsalResponse(provider_runtime_rehearsal=status)


@app.get(
    "/admin/provider-runtime-rehearsal/report",
    summary="Download local provider runtime rehearsal report",
)
def admin_provider_runtime_rehearsal_report(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return provider runtime rehearsal as no-secret Markdown."""
    verify_api_token(authorization, x_api_key)
    status = collect_provider_runtime_rehearsal()
    _record_provider_runtime_check(status)
    report = format_provider_runtime_rehearsal_markdown(status)
    return PlainTextResponse(
        report,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="fluxmind-provider-runtime-rehearsal.md"'
        },
    )


@app.get(
    "/admin/platform-migration-rehearsal",
    response_model=PlatformMigrationRehearsalResponse,
    summary="Run local platform migration rehearsal",
)
def admin_platform_migration_rehearsal(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Run the disposable no-secret platform migration rehearsal on demand."""
    verify_api_token(authorization, x_api_key)
    status = collect_platform_migration_rehearsal()
    _record_platform_migration_check(status)
    return PlatformMigrationRehearsalResponse(platform_migration_rehearsal=status)


@app.get(
    "/admin/platform-migration-rehearsal/report",
    summary="Download local platform migration rehearsal report",
)
def admin_platform_migration_rehearsal_report(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return platform migration rehearsal as no-secret Markdown."""
    verify_api_token(authorization, x_api_key)
    status = collect_platform_migration_rehearsal()
    _record_platform_migration_check(status)
    report = format_storage_migration_rehearsal_markdown(status)
    return PlainTextResponse(
        report,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="fluxmind-platform-migration-rehearsal.md"'
        },
    )


@app.get(
    "/admin/activation-suite",
    response_model=ActivationSuiteResponse,
    summary="Run local activation suite",
)
def admin_activation_suite(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Run the explicit no-secret local activation suite on demand."""
    verify_api_token(authorization, x_api_key)
    status = collect_activation_suite(openapi_schema=app.openapi())
    _record_activation_suite_check(status)
    return ActivationSuiteResponse(activation_suite=status)


@app.post(
    "/admin/activation-suite",
    response_model=ActivationSuiteResponse,
    summary="Run local activation suite with live evidence",
)
def admin_activation_suite_with_report(
    req: ActivationSuiteRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Run the activation suite with supplied no-secret eval report evidence."""
    verify_api_token(authorization, x_api_key)
    live_reports = list(req.live_reports)
    if req.live_report is not None:
        live_reports.insert(0, req.live_report)
    status = collect_activation_suite(
        live_reports=live_reports,
        openapi_schema=app.openapi(),
    )
    _record_activation_suite_check(status)
    return ActivationSuiteResponse(activation_suite=status)


@app.get("/admin/activation-suite/report", summary="Download local activation suite report")
def admin_activation_suite_report(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return the explicit no-secret local activation suite as Markdown."""
    verify_api_token(authorization, x_api_key)
    status = collect_activation_suite(openapi_schema=app.openapi())
    _record_activation_suite_check(status)
    report = format_activation_suite_markdown(status)
    return PlainTextResponse(
        report,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="fluxmind-activation-suite.md"'},
    )


@app.post("/admin/activation-suite/report", summary="Download local activation suite report with live evidence")
def admin_activation_suite_report_with_report(
    req: ActivationSuiteRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return a no-secret activation-suite report with supplied eval evidence."""
    verify_api_token(authorization, x_api_key)
    live_reports = list(req.live_reports)
    if req.live_report is not None:
        live_reports.insert(0, req.live_report)
    status = collect_activation_suite(
        live_reports=live_reports,
        openapi_schema=app.openapi(),
    )
    _record_activation_suite_check(status)
    report = format_activation_suite_markdown(status)
    return PlainTextResponse(
        report,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="fluxmind-activation-suite.md"'},
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
    response: Response,
    upload_days: int = 0,
    artifact_days: int = 0,
    limit: int = 100,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Delete local upload/artifact retention candidates when explicitly enabled."""
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(None, auth_context)
    enforce_product_rbac(
        req=None,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/admin/retention/delete",
        action="admin_write",
    )
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
    safe_query = (q or "").strip()
    events = [
        runtime_event_to_dict(event)
        for event in list_runtime_events(
            kind=kind,
            code=code,
            q=None,
            limit=1000 if safe_query else bounded_limit,
        )
    ]
    if safe_query:
        events = [
            event for event in events if runtime_event_matches_safe_query(event, safe_query)
        ][:bounded_limit]
    return RuntimeEventsResponse(events=events)


@app.get(
    "/admin/product-registry/status",
    response_model=ProductRegistryStatusResponse,
    summary="Inspect local product registry status",
)
def admin_product_registry_status(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return no-secret local product registry availability and counts."""
    verify_api_token(authorization, x_api_key)
    return ProductRegistryStatusResponse(status=product_registry_admin_status())


@app.get(
    "/admin/product-registry/workspaces",
    response_model=ProductRegistryWorkspaceListResponse,
    summary="List local product workspaces",
)
def admin_product_registry_workspaces(
    response: Response,
    limit: int = 50,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """List local product workspace summaries when the SQLite registry is enabled."""
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    enforce_product_registry_admin_read(
        response=response,
        request_id=request_id,
        auth_context=auth_context,
        endpoint="/admin/product-registry/workspaces",
    )
    registry = require_local_product_registry()
    bounded_limit = min(max(limit, 1), 200)
    return ProductRegistryWorkspaceListResponse(
        status=product_registry_admin_status(),
        workspaces=registry.list_workspace_summaries(limit=bounded_limit),
    )


@app.post(
    "/admin/product-registry/workspaces",
    response_model=ProductRegistryWorkspaceResponse,
    summary="Create or update a local product workspace",
)
def admin_product_registry_create_workspace(
    req: ProductRegistryWorkspaceRequest,
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Create or update a local workspace and owner membership."""
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(None, auth_context)
    enforce_product_rbac(
        req=req,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/admin/product-registry/workspaces",
        action="admin_write",
    )
    registry = require_local_product_registry()
    try:
        workspace = registry.create_workspace(
            workspace_id=req.workspace_id,
            label=req.label,
            owner_user_id=req.owner_user_id,
            owner_label=req.owner_label,
        )
        detail = registry.workspace_detail(workspace_id=workspace.workspace_id)
    except (OSError, sqlite3.Error, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail={"code": "product_registry_write_failed"},
        ) from exc
    record_product_registry_admin_event(
        action="workspace_upsert",
        status_code=200,
        request_id=request_id,
        workspace_id=workspace.workspace_id,
    )
    return ProductRegistryWorkspaceResponse(workspace=detail or {})


@app.get(
    "/admin/product-registry/workspaces/{workspace_id}",
    response_model=ProductRegistryWorkspaceResponse,
    summary="Inspect one local product workspace",
)
def admin_product_registry_workspace_detail(
    workspace_id: str,
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Return one local workspace with members, quota limits, and billing state."""
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    enforce_product_registry_admin_read(
        response=response,
        request_id=request_id,
        auth_context=auth_context,
        endpoint="/admin/product-registry/workspaces/{workspace_id}",
        workspace_id=workspace_id,
    )
    detail = require_local_product_registry().workspace_detail(workspace_id=workspace_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Product workspace not found")
    return ProductRegistryWorkspaceResponse(workspace=detail)


@app.post(
    "/admin/product-registry/workspaces/{workspace_id}/members",
    response_model=ProductRegistryWorkspaceResponse,
    summary="Add or update a local product workspace member",
)
def admin_product_registry_add_member(
    workspace_id: str,
    req: ProductRegistryMemberRequest,
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Add or update a local workspace member role."""
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(None, auth_context)
    enforce_product_rbac(
        req=SimpleNamespace(workspace_id=workspace_id),
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/admin/product-registry/workspaces/{workspace_id}/members",
        action="admin_write",
    )
    registry = require_local_product_registry()
    if registry.workspace_detail(workspace_id=workspace_id) is None:
        raise HTTPException(status_code=404, detail="Product workspace not found")
    try:
        registry.add_member(
            workspace_id=workspace_id,
            user_id=req.user_id,
            label=req.label,
            role=req.role,
        )
        detail = registry.workspace_detail(workspace_id=workspace_id)
    except (OSError, sqlite3.Error, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail={"code": "product_registry_write_failed"},
        ) from exc
    record_product_registry_admin_event(
        action="member_upsert",
        status_code=200,
        request_id=request_id,
        workspace_id=workspace_id,
    )
    return ProductRegistryWorkspaceResponse(workspace=detail or {})


@app.put(
    "/admin/product-registry/workspaces/{workspace_id}/quota",
    response_model=ProductRegistryWorkspaceResponse,
    summary="Set a local product workspace quota",
)
def admin_product_registry_set_quota(
    workspace_id: str,
    req: ProductRegistryQuotaRequest,
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Set one local quota limit for a workspace."""
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(None, auth_context)
    enforce_product_rbac(
        req=SimpleNamespace(workspace_id=workspace_id),
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/admin/product-registry/workspaces/{workspace_id}/quota",
        action="admin_write",
    )
    registry = require_local_product_registry()
    if registry.workspace_detail(workspace_id=workspace_id) is None:
        raise HTTPException(status_code=404, detail="Product workspace not found")
    try:
        registry.set_quota(
            workspace_id=workspace_id,
            metric=req.metric,
            limit_value=req.limit_value,
            window_s=req.window_s,
        )
        detail = registry.workspace_detail(workspace_id=workspace_id)
    except (OSError, sqlite3.Error, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail={"code": "product_registry_write_failed"},
        ) from exc
    record_product_registry_admin_event(
        action="quota_set",
        status_code=200,
        request_id=request_id,
        workspace_id=workspace_id,
    )
    return ProductRegistryWorkspaceResponse(workspace=detail or {})


@app.put(
    "/admin/product-registry/workspaces/{workspace_id}/billing",
    response_model=ProductRegistryWorkspaceResponse,
    summary="Set local product billing attribution state",
)
def admin_product_registry_set_billing(
    workspace_id: str,
    req: ProductRegistryBillingRequest,
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Set local billing-attribution metadata without external payment credentials."""
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(None, auth_context)
    enforce_product_rbac(
        req=SimpleNamespace(workspace_id=workspace_id),
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/admin/product-registry/workspaces/{workspace_id}/billing",
        action="admin_write",
    )
    registry = require_local_product_registry()
    if registry.workspace_detail(workspace_id=workspace_id) is None:
        raise HTTPException(status_code=404, detail="Product workspace not found")
    try:
        registry.set_billing_account(
            workspace_id=workspace_id,
            billing_mode=req.billing_mode,
            status=req.status,
            attribution_enabled=req.attribution_enabled,
        )
        detail = registry.workspace_detail(workspace_id=workspace_id)
    except (OSError, sqlite3.Error, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail={"code": "product_registry_write_failed"},
        ) from exc
    record_product_registry_admin_event(
        action="billing_set",
        status_code=200,
        request_id=request_id,
        workspace_id=workspace_id,
    )
    return ProductRegistryWorkspaceResponse(workspace=detail or {})


@app.post(
    "/admin/product-registry/permissions/check",
    response_model=ProductRegistryPermissionResponse,
    summary="Check local product RBAC permission",
)
def admin_product_registry_check_permission(
    req: ProductRegistryPermissionCheckRequest,
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Check a local product RBAC decision without performing the target action."""
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    enforce_product_registry_admin_read(
        response=response,
        request_id=request_id,
        auth_context=auth_context,
        endpoint="/admin/product-registry/permissions/check",
        workspace_id=req.workspace_id,
    )
    registry = require_local_product_registry()
    decision = registry.permission_decision(
        user_id=req.user_id,
        action=req.action,
        workspace_id=req.workspace_id,
    )
    record_product_registry_admin_event(
        action="permission_check",
        status_code=200,
        request_id=request_id,
        workspace_id=str(decision.get("workspace_id", req.workspace_id or "")),
        reason=str(decision.get("reason", "ok")),
    )
    return ProductRegistryPermissionResponse(permission=decision)


@app.get(
    "/admin/share-links/status",
    response_model=ShareLinkRegistryStatusResponse,
    summary="Inspect local share-link registry status",
)
def admin_share_link_registry_status(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return no-secret local share-link registry availability and counts."""
    verify_api_token(authorization, x_api_key)
    return ShareLinkRegistryStatusResponse(status=share_link_registry_admin_status())


@app.get(
    "/admin/share-links",
    response_model=ShareLinkListResponse,
    summary="List local share links",
)
def admin_share_link_list(
    response: Response,
    workspace_id: str | None = None,
    include_revoked: bool = False,
    limit: int = 50,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """List local share-link summaries without share tokens, URLs, or resource refs."""
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    enforce_product_registry_admin_read(
        response=response,
        request_id=request_id,
        auth_context=auth_context,
        endpoint="/admin/share-links",
        workspace_id=workspace_id,
    )
    registry = require_local_share_link_registry()
    bounded_limit = min(max(limit, 1), 200)
    links = [
        record.to_public_dict()
        for record in registry.list_links(
            workspace_id=workspace_id,
            include_revoked=include_revoked,
            limit=bounded_limit,
        )
    ]
    record_share_link_admin_event(
        action="list",
        status_code=200,
        request_id=request_id,
        workspace_id=workspace_id or "",
    )
    return ShareLinkListResponse(
        status=share_link_registry_admin_status(),
        share_links=links,
    )


@app.post(
    "/admin/share-links",
    response_model=ShareLinkCreateResponse,
    summary="Create a local share link",
)
def admin_share_link_create(
    req: ShareLinkCreateRequest,
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Create a local hash-only share token and return it once."""
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(None, auth_context)
    enforce_product_rbac(
        req=req,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/admin/share-links",
        action="admin_write",
    )
    creator_id = req.created_by_user_id or ownership["owner_id"]
    registry = require_local_share_link_registry()
    try:
        payload = registry.create_link(
            workspace_id=req.workspace_id,
            created_by_user_id=creator_id,
            resource_kind=req.resource_kind,
            resource_ref=req.resource_ref,
            description=req.description,
            expires_in_s=req.expires_in_s,
            max_redemptions=req.max_redemptions,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"code": str(exc)}) from exc
    except (OSError, sqlite3.Error) as exc:
        raise HTTPException(status_code=500, detail={"code": "share_link_write_failed"}) from exc
    share_link = payload.get("share_link", {}) or {}
    record_share_link_admin_event(
        action="create",
        status_code=200,
        request_id=request_id,
        workspace_id=str(share_link.get("workspace_id", req.workspace_id)),
        link_id=str(share_link.get("link_id", "")),
    )
    return ShareLinkCreateResponse(**payload)


@app.post(
    "/admin/share-links/{link_id}/revoke",
    response_model=ShareLinkResponse,
    summary="Revoke a local share link",
)
def admin_share_link_revoke(
    link_id: str,
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Revoke a local share link without returning its token or resource ref."""
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    registry = require_local_share_link_registry()
    existing = registry.get_link(link_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Share link not found")
    ownership = request_ownership(None, auth_context)
    enforce_product_rbac(
        req=SimpleNamespace(workspace_id=existing.workspace_id),
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/admin/share-links/{link_id}/revoke",
        action="admin_write",
    )
    try:
        record = registry.revoke_link(link_id)
    except (OSError, sqlite3.Error) as exc:
        raise HTTPException(status_code=500, detail={"code": "share_link_write_failed"}) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Share link not found")
    record_share_link_admin_event(
        action="revoke",
        status_code=200,
        request_id=request_id,
        workspace_id=record.workspace_id,
        link_id=record.link_id,
    )
    return ShareLinkResponse(share_link=record.to_public_dict())


@app.post(
    "/admin/share-links/resolve",
    response_model=ShareLinkResolveResponse,
    summary="Resolve a local share token",
)
def admin_share_link_resolve(
    req: ShareLinkResolveRequest,
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Resolve a local share token without echoing the token, URL, or content."""
    verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    registry = require_local_share_link_registry()
    try:
        resolution = registry.resolve_token(req.token, record_redeem=req.record_redeem)
    except (OSError, sqlite3.Error) as exc:
        raise HTTPException(status_code=500, detail={"code": "share_link_read_failed"}) from exc
    share_link = resolution.get("share_link", {}) or {}
    record_share_link_admin_event(
        action="resolve",
        status_code=200,
        request_id=request_id,
        workspace_present=bool(share_link.get("workspace_present")),
        link_id=str(share_link.get("link_id", "")),
        reason=str(resolution.get("reason", "unknown")),
        share_link_valid=bool(resolution.get("valid")),
    )
    return ShareLinkResolveResponse(resolution=resolution)


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
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req, auth_context)
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    enforce_product_rbac(
        req=req,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/query",
        action="query",
    )
    quota_decision = enforce_product_quota(
        req=req,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/query",
    )
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
        duration_ms = int((time.monotonic() - started) * 1000)
        error = normalize_exception(exc)
        try:
            error = record_query_exception_event(
                exc=exc,
                endpoint="/query",
                request_id=request_id,
                answer_mode=req.answer_mode,
                duration_ms=duration_ms,
                ownership=ownership,
            )
        except OSError:
            logged_error = normalize_exception(exc)
            logger.warning("query.event_log_failed request_id=%s code=%s", request_id, logged_error.code)
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
    apply_product_quota_headers(response, quota_decision)
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
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req, auth_context)
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    enforce_product_rbac(
        req=req,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/query/inspect",
        action="query",
    )
    quota_decision = enforce_product_quota(
        req=req,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/query/inspect",
    )
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
        duration_ms = int((time.monotonic() - started) * 1000)
        error = normalize_exception(exc)
        try:
            error = record_query_exception_event(
                exc=exc,
                endpoint="/query/inspect",
                request_id=request_id,
                answer_mode=req.answer_mode,
                duration_ms=duration_ms,
                ownership=ownership,
            )
        except OSError:
            logged_error = normalize_exception(exc)
            logger.warning("query.inspect_event_log_failed request_id=%s code=%s", request_id, logged_error.code)
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
    apply_product_quota_headers(response, quota_decision)
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
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req, auth_context)
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    enforce_product_rbac(
        req=req,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/query/retrieve",
        action="query",
    )
    quota_decision = enforce_product_quota(
        req=req,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/query/retrieve",
    )
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
    apply_product_quota_headers(response, quota_decision)
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
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req, auth_context)
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    enforce_product_rbac(
        req=req,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/query/report",
        action="query",
    )
    quota_decision = enforce_product_quota(
        req=req,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/query/report",
    )
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
        duration_ms = int((time.monotonic() - started) * 1000)
        error = normalize_exception(exc)
        try:
            error = record_query_exception_event(
                exc=exc,
                endpoint="/query/report",
                request_id=request_id,
                answer_mode=req.answer_mode,
                duration_ms=duration_ms,
                ownership=ownership,
            )
        except OSError:
            logged_error = normalize_exception(exc)
            logger.warning("query.report_event_log_failed request_id=%s code=%s", request_id, logged_error.code)
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
    apply_product_quota_headers(response, quota_decision)
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
    headers = dict(response.headers)
    headers["Content-Disposition"] = 'attachment; filename="fluxmind-query-report.md"'
    return PlainTextResponse(
        report,
        media_type="text/markdown; charset=utf-8",
        headers=headers,
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
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req, auth_context)
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    enforce_product_rbac(
        req=req,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/jobs/image/mock",
        action="job_submit",
    )
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
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req, auth_context)
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    enforce_product_rbac(
        req=req,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/jobs/async/image/mock",
        action="job_submit",
    )
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
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req, auth_context)
    if not req.entrypoint.strip():
        raise HTTPException(status_code=400, detail="Entrypoint cannot be empty")
    enforce_product_rbac(
        req=req,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/jobs/code/python-local",
        action="job_submit",
    )
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
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req, auth_context)
    if not req.entrypoint.strip():
        raise HTTPException(status_code=400, detail="Entrypoint cannot be empty")
    enforce_product_rbac(
        req=req,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/jobs/async/code/python-local",
        action="job_submit",
    )
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
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req, auth_context)
    if not req.entrypoint.strip():
        raise HTTPException(status_code=400, detail="Entrypoint cannot be empty")
    enforce_product_rbac(
        req=req,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/jobs/code/octave-local",
        action="job_submit",
    )
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
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req, auth_context)
    if not req.entrypoint.strip():
        raise HTTPException(status_code=400, detail="Entrypoint cannot be empty")
    enforce_product_rbac(
        req=req,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/jobs/async/code/octave-local",
        action="job_submit",
    )
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
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req, auth_context)
    if not req.source_paths:
        raise HTTPException(status_code=400, detail="At least one source path is required")
    enforce_product_rbac(
        req=req,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/jobs/index/rebuild",
        action="corpus_write",
    )
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
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(req, auth_context)
    if not req.source_paths:
        raise HTTPException(status_code=400, detail="At least one source path is required")
    enforce_product_rbac(
        req=req,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint="/jobs/async/index/rebuild",
        action="corpus_write",
    )
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
    """List latest local job summaries."""
    verify_api_token(authorization, x_api_key)
    bounded_limit = min(max(limit, 1), 200)
    jobs = [
        job_summary_to_dict(job)
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
    response: Response,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """Mark a queued/running local job as cancelled."""
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(None, auth_context)
    enforce_product_rbac(
        req=None,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint=f"/jobs/{job_id}/cancel",
        action="job_submit",
    )
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
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(None, auth_context)
    enforce_product_rbac(
        req=None,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint=f"/jobs/{job_id}/retry",
        action="job_submit",
    )
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
    auth_context = verify_api_token(authorization, x_api_key)
    request_id = request_id_header(response, x_request_id)
    ownership = request_ownership(None, auth_context)
    enforce_product_rbac(
        req=req,
        response=response,
        request_id=request_id,
        ownership=ownership,
        auth_context=auth_context,
        endpoint=f"/jobs/{job_id}/retry-scheduled",
        action="job_submit",
    )
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
