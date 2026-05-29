"""FluxMind API — FastAPI wrapper for RAG pipeline, compatible with Coze custom plugin."""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.capabilities import CodeExecutionRequest, ImageGenerationRequest
from src.chain import query
from src.ingestion import build_vector_store
from src.jobs import JobRecord, LocalJobRunner, LocalJobStore
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


class JobResponse(BaseModel):
    job: dict = Field(..., description="Persisted local job record")


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
    }


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
        answer = query(req.question)
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


@app.get("/health")
def health():
    return {"status": "ok"}
