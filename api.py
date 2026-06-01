"""FluxMind API — FastAPI wrapper for RAG pipeline, compatible with Coze custom plugin."""

import os
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from src.admin import (
    collect_admin_status,
    collect_corpus_profile_status,
    collect_corpus_status,
    collect_retention_preview,
    format_admin_status_report,
    format_corpus_profile_status_report,
)
from src.artifacts import LocalArtifactRegistry
from src.capabilities import CodeExecutionRequest, ImageGenerationRequest
from src.chain import get_vector_store, query_with_metadata, retrieve_with_metadata
from src.config import FAISS_INDEX_DIR
from src.ingestion import refresh_paper_metadata, set_active_paper_source_paths
from src.jobs import JobRecord, LocalJobRunner, LocalJobStore, get_async_job_manager
from src.metadata import ChunkMetadataStore, CorpusProfileStore
from src.runtime import append_runtime_event, estimate_text_tokens, list_runtime_events, logger, new_request_id, normalize_exception

API_TOKEN = os.getenv("FLUXMIND_API_TOKEN", "")
logging.basicConfig(level=os.getenv("FLUXMIND_LOG_LEVEL", "INFO"))

def warm_existing_vector_store() -> bool:
    """Best-effort startup warmup without rebuilding a missing index."""
    if not (FAISS_INDEX_DIR / "index.faiss").exists():
        logger.warning("startup.index_missing path=%s", FAISS_INDEX_DIR)
        return False
    try:
        get_vector_store()
    except Exception:
        logger.exception("startup.index_warmup_failed path=%s", FAISS_INDEX_DIR)
        return False
    return True


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Warm existing retrieval state and recover durable queued local jobs."""
    warm_existing_vector_store()
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


class QueryRequest(BaseModel):
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


class MockImageJobRequest(BaseModel):
    prompt: str = Field(..., description="Diagram prompt")
    style: str = Field(default="engineering-diagram")
    size: str = Field(default="1024x1024")
    reference_uris: list[str] = Field(default_factory=list)
    queue_timeout_s: int | None = Field(default=None, ge=1, le=86400, description="Optional async queue deadline in seconds")


class LocalPythonJobRequest(BaseModel):
    entrypoint: str = Field(..., description="Python entrypoint filename")
    files: dict[str, str] = Field(..., description="Files to materialize for execution")
    timeout_s: int = Field(default=30, ge=1, le=120)
    memory_mb: int = Field(default=512, ge=64, le=4096)
    queue_timeout_s: int | None = Field(default=None, ge=1, le=86400, description="Optional async queue deadline in seconds")


class LocalOctaveJobRequest(BaseModel):
    entrypoint: str = Field(..., description="Octave-compatible entrypoint filename")
    files: dict[str, str] = Field(..., description="Files to materialize for execution")
    timeout_s: int = Field(default=30, ge=1, le=120)
    memory_mb: int = Field(default=512, ge=64, le=4096)
    queue_timeout_s: int | None = Field(default=None, ge=1, le=86400, description="Optional async queue deadline in seconds")


class IndexRebuildJobRequest(BaseModel):
    source_paths: list[str] = Field(..., description="Project-relative selectable PDF paths")
    queue_timeout_s: int | None = Field(default=None, ge=1, le=86400, description="Optional async queue deadline in seconds")


class ActiveCorpusRequest(BaseModel):
    source_paths: list[str] = Field(..., description="Project-relative selectable PDF paths to keep active")


class CorpusProfileRequest(BaseModel):
    name: str = Field(..., description="Human-readable local corpus profile name")
    source_paths: list[str] = Field(..., description="Project-relative selectable PDF paths in this profile")
    profile_id: str | None = Field(default=None, description="Optional stable local profile ID")
    description: str | None = Field(default=None, description="Optional no-secret profile description")


class CorpusProfileRebuildRequest(BaseModel):
    queue_timeout_s: int | None = Field(default=None, ge=1, le=86400, description="Optional async queue deadline in seconds")


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


class RuntimeEventsResponse(BaseModel):
    events: list[dict] = Field(..., description="Latest no-secret local runtime events")


def verify_api_token(authorization: str | None, x_api_key: str | None) -> None:
    """Protect public Coze/plugin calls when FLUXMIND_API_TOKEN is configured."""
    if not API_TOKEN:
        return

    bearer_token = ""
    if authorization and authorization.lower().startswith("bearer "):
        bearer_token = authorization[7:].strip()

    if x_api_key != API_TOKEN and bearer_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API token")


def request_id_header(response: Response, x_request_id: str | None) -> str:
    request_id = (x_request_id or new_request_id()).strip()[:64]
    response.headers["X-Request-ID"] = request_id
    return request_id


def job_to_dict(record: JobRecord) -> dict:
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
        "logs": record.logs,
    }


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
) -> None:
    """Append no-secret estimated query usage for local admin/cost-shape checks."""
    estimated_prompt_tokens = estimate_text_tokens(question)
    estimated_answer_tokens = estimate_text_tokens(answer)
    metadata = {
        "endpoint": endpoint,
        "answer_mode": answer_mode,
        "question_chars": len(question),
        "answer_chars": len(answer),
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "estimated_answer_tokens": estimated_answer_tokens,
        "estimated_total_tokens": estimated_prompt_tokens + estimated_answer_tokens,
        "estimated_cost_usd": "0",
        "usage_source": "provider" if provider_usage else "estimated",
    }
    if provider_usage:
        metadata.update(
            {
                "provider_prompt_tokens": int(provider_usage.get("prompt_tokens", 0) or 0),
                "provider_completion_tokens": int(provider_usage.get("completion_tokens", 0) or 0),
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
        lines.append(
            f"- [{ref.get('ref')}] {ref.get('source_path') or ref.get('source') or 'unknown'} "
            f"page={ref.get('page', '?')} preview={ref.get('preview', '')}"
        )
    lines.append("")
    return "\n".join(lines)


@app.get("/artifacts", response_model=ArtifactListResponse, summary="List local artifacts")
def list_artifacts(
    q: str | None = None,
    kind: str | None = None,
    job_kind: str | None = None,
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
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    logger.info(
        "query.start request_id=%s client=%s chars=%s",
        request_id,
        request.client.host if request.client else "unknown",
        len(req.question),
    )
    try:
        result = query_with_metadata(req.question, answer_mode=req.answer_mode)
        answer = result.answer
    except Exception as exc:
        error = normalize_exception(exc)
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
    record_query_usage(
        endpoint="/query",
        request_id=request_id,
        answer_mode=req.answer_mode,
        question=req.question,
        answer=answer,
        provider_usage=result.provider_usage,
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
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    logger.info(
        "query.inspect_start request_id=%s client=%s chars=%s",
        request_id,
        request.client.host if request.client else "unknown",
        len(req.question),
    )
    try:
        result = query_with_metadata(req.question, answer_mode=req.answer_mode)
    except Exception as exc:
        error = normalize_exception(exc)
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
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    logger.info(
        "query.retrieve_start request_id=%s client=%s chars=%s",
        request_id,
        request.client.host if request.client else "unknown",
        len(req.question),
    )
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
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    logger.info(
        "query.report_start request_id=%s client=%s chars=%s",
        request_id,
        request.client.host if request.client else "unknown",
        len(req.question),
    )
    try:
        result = query_with_metadata(req.question, answer_mode=req.answer_mode)
    except Exception as exc:
        error = normalize_exception(exc)
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
    record_query_usage(
        endpoint="/query/report",
        request_id=request_id,
        answer_mode=req.answer_mode,
        question=req.question,
        answer=result.answer,
        citation_ok=result.citation_validation.ok,
        provider_usage=getattr(result, "provider_usage", None),
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
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    job = LocalJobRunner().run_mock_image(
        ImageGenerationRequest(
            prompt=req.prompt,
            style=req.style,
            size=req.size,
            reference_uris=req.reference_uris,
        ),
        request_id=request_id,
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
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    job = get_async_job_manager().enqueue_mock_image(
        ImageGenerationRequest(
            prompt=req.prompt,
            style=req.style,
            size=req.size,
            reference_uris=req.reference_uris,
        ),
        request_id=request_id,
        queue_timeout_s=req.queue_timeout_s,
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
    if not req.entrypoint.strip():
        raise HTTPException(status_code=400, detail="Entrypoint cannot be empty")
    job = LocalJobRunner().run_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint=req.entrypoint,
            files=req.files,
            timeout_s=req.timeout_s,
            memory_mb=req.memory_mb,
        ),
        request_id=request_id,
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
    if not req.entrypoint.strip():
        raise HTTPException(status_code=400, detail="Entrypoint cannot be empty")
    job = LocalJobRunner().run_local_octave(
        CodeExecutionRequest(
            language="octave",
            entrypoint=req.entrypoint,
            files=req.files,
            timeout_s=req.timeout_s,
            memory_mb=req.memory_mb,
        ),
        request_id=request_id,
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
    if not req.source_paths:
        raise HTTPException(status_code=400, detail="At least one source path is required")
    job = LocalJobRunner().run_index_rebuild(req.source_paths, request_id=request_id)
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
    if not req.source_paths:
        raise HTTPException(status_code=400, detail="At least one source path is required")
    job = get_async_job_manager().enqueue_index_rebuild(
        req.source_paths,
        request_id=request_id,
        queue_timeout_s=req.queue_timeout_s,
    )
    return JobResponse(job=job_to_dict(job))


@app.get("/jobs", response_model=JobListResponse, summary="List local jobs")
def list_jobs(
    q: str | None = None,
    status: str | None = None,
    kind: str | None = None,
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
