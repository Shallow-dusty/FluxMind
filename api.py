"""FluxMind API — FastAPI wrapper for RAG pipeline, compatible with Coze custom plugin."""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.chain import query
from src.ingestion import build_vector_store
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


def verify_api_token(authorization: str | None, x_api_key: str | None) -> None:
    """Protect public Coze/plugin calls when FLUXMIND_API_TOKEN is configured."""
    if not API_TOKEN:
        return

    bearer_token = ""
    if authorization and authorization.lower().startswith("bearer "):
        bearer_token = authorization[7:].strip()

    if x_api_key != API_TOKEN and bearer_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API token")


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
    request_id = (x_request_id or new_request_id()).strip()[:64]
    response.headers["X-Request-ID"] = request_id
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


@app.get("/health")
def health():
    return {"status": "ok"}
