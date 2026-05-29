"""FluxMind API — FastAPI wrapper for RAG pipeline, compatible with Coze custom plugin."""

import os
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.admin import collect_admin_status
from src.artifacts import LocalArtifactRegistry
from src.capabilities import CodeExecutionRequest, ImageGenerationRequest
from src.chain import query
from src.ingestion import build_vector_store, refresh_paper_metadata, set_active_paper_source_paths
from src.jobs import JobRecord, LocalJobRunner, LocalJobStore, get_async_job_manager
from src.runtime import logger, new_request_id, normalize_exception

API_TOKEN = os.getenv("FLUXMIND_API_TOKEN", "")
logging.basicConfig(level=os.getenv("FLUXMIND_LOG_LEVEL", "INFO"))

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize the vector store when the API process starts."""
    build_vector_store()
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


class MockImageJobRequest(BaseModel):
    prompt: str = Field(..., description="Diagram prompt")
    style: str = Field(default="engineering-diagram")
    size: str = Field(default="1024x1024")
    reference_uris: list[str] = Field(default_factory=list)


class LocalPythonJobRequest(BaseModel):
    entrypoint: str = Field(..., description="Python entrypoint filename")
    files: dict[str, str] = Field(..., description="Files to materialize for execution")
    timeout_s: int = Field(default=30, ge=1, le=120)
    memory_mb: int = Field(default=512, ge=64, le=4096)


class IndexRebuildJobRequest(BaseModel):
    source_paths: list[str] = Field(..., description="Project-relative selectable PDF paths")


class ActiveCorpusRequest(BaseModel):
    source_paths: list[str] = Field(..., description="Project-relative selectable PDF paths to keep active")


class RetryScheduleRequest(BaseModel):
    delay_s: int = Field(default=30, ge=0, le=3600, description="Delay before retry execution")


class JobResponse(BaseModel):
    job: dict = Field(..., description="Persisted local job record")


class JobListResponse(BaseModel):
    jobs: list[dict] = Field(..., description="Latest local job records")


class CorpusPapersResponse(BaseModel):
    papers: list[dict] = Field(..., description="Current local corpus paper metadata")


class ActiveCorpusResponse(BaseModel):
    papers: list[dict] = Field(..., description="Updated local corpus paper metadata")
    active_source_paths: list[str] = Field(..., description="Persisted active paper source paths")
    rebuild_required: bool = Field(..., description="Whether the FAISS index should be rebuilt to apply selection")


class ArtifactListResponse(BaseModel):
    artifacts: list[dict] = Field(..., description="Generated local artifacts")


class AdminStatusResponse(BaseModel):
    status: dict = Field(..., description="Local admin/runtime status")


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
    }


def record_to_dict(record) -> dict:
    return asdict(record)


@app.get("/artifacts", response_model=ArtifactListResponse, summary="List local artifacts")
def list_artifacts(
    limit: int = 100,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """List artifacts produced by local jobs."""
    verify_api_token(authorization, x_api_key)
    bounded_limit = min(max(limit, 1), 500)
    artifacts = [
        asdict(artifact)
        for artifact in LocalArtifactRegistry().list_artifacts(limit=bounded_limit)
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
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """List selectable papers with local metadata, active state, and index status."""
    verify_api_token(authorization, x_api_key)
    papers = [record_to_dict(record) for record in refresh_paper_metadata()]
    return CorpusPapersResponse(papers=papers)


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


@app.get("/admin/status", response_model=AdminStatusResponse, summary="Inspect local runtime status")
def admin_status(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Return no-secret local runtime status for operations and admin UI."""
    verify_api_token(authorization, x_api_key)
    return AdminStatusResponse(status=collect_admin_status().to_dict())


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
        answer = query(req.question, answer_mode=req.answer_mode)
    except Exception as exc:
        error = normalize_exception(exc)
        logger.exception("query.error request_id=%s code=%s", request_id, error.code)
        raise HTTPException(
            status_code=error.status_code,
            detail={
                "code": error.code,
                "message": error.message,
                "request_id": request_id,
            },
        ) from exc
    logger.info("query.ok request_id=%s chars=%s", request_id, len(answer))
    return QueryResponse(answer=answer, request_id=request_id)


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
    job = get_async_job_manager().enqueue_index_rebuild(req.source_paths, request_id=request_id)
    return JobResponse(job=job_to_dict(job))


@app.get("/jobs", response_model=JobListResponse, summary="List local jobs")
def list_jobs(
    limit: int = 50,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """List latest local job records."""
    verify_api_token(authorization, x_api_key)
    bounded_limit = min(max(limit, 1), 200)
    jobs = [job_to_dict(job) for job in LocalJobStore().list_latest(limit=bounded_limit)]
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
    job = get_async_job_manager().schedule_retry(job_id, delay_s=req.delay_s, request_id=request_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(job=job_to_dict(job))


@app.get("/health")
def health():
    return {"status": "ok"}
