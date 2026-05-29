"""FluxMind API — FastAPI wrapper for RAG pipeline, compatible with Coze custom plugin."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.chain import query
from src.ingestion import build_vector_store

API_TOKEN = os.getenv("FLUXMIND_API_TOKEN", "")

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
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Query the FluxMind knowledge base. Retrieves relevant paper chunks and generates an answer."""
    verify_api_token(authorization, x_api_key)
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    answer = query(req.question)
    return QueryResponse(answer=answer)


@app.get("/health")
def health():
    return {"status": "ok"}
