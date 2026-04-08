"""FluxMind API — FastAPI wrapper for RAG pipeline, compatible with Coze custom plugin."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.chain import query
from src.ingestion import build_vector_store

# Init vector store on startup
build_vector_store()

app = FastAPI(
    title="FluxMind API",
    description="RAG-based Copilot for Sliding Mode Control & Flux Linkage Estimation",
    version="1.0.0",
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


@app.post("/query", response_model=QueryResponse, summary="Ask FluxMind")
def ask(req: QueryRequest):
    """Query the FluxMind knowledge base. Retrieves relevant paper chunks and generates an answer."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    answer = query(req.question)
    return QueryResponse(answer=answer)


@app.get("/health")
def health():
    return {"status": "ok"}
