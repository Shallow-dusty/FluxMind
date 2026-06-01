"""FluxMind configuration."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# LLM
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.example.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "DeepSeek-V3.2")

# Embedding
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "")

# Local execution backend. `local` keeps the current child-process provider.
# `docker` is a no-key future sandbox backend and remains unavailable unless the
# runtime user can access Docker safely.
CODE_EXECUTION_BACKEND = os.getenv("CODE_EXECUTION_BACKEND", "local").strip().lower()
DOCKER_EXECUTION_IMAGE = os.getenv("DOCKER_EXECUTION_IMAGE", "python:3.12-slim")

# Storage backend readiness. Local JSON/SQLite/filesystem storage remains the
# active no-key backend; external database/object storage requires explicit
# configuration and is only reported as readiness here.
METADATA_STORAGE_BACKEND = os.getenv("METADATA_STORAGE_BACKEND", "local").strip().lower()
DATABASE_URL = os.getenv("DATABASE_URL", "")
OBJECT_STORAGE_BACKEND = os.getenv("OBJECT_STORAGE_BACKEND", "local").strip().lower()
OBJECT_STORAGE_BUCKET = os.getenv("OBJECT_STORAGE_BUCKET", "")
OBJECT_STORAGE_ENDPOINT = os.getenv("OBJECT_STORAGE_ENDPOINT", "")
OBJECT_STORAGE_REGION = os.getenv("OBJECT_STORAGE_REGION", "")

# Optional local cost estimation. These rates are no-secret configuration used
# only for admin estimates; FluxMind does not connect to external billing.
QUERY_COST_PROVIDER = os.getenv("QUERY_COST_PROVIDER", "").strip()
QUERY_COST_PROMPT_USD_PER_1M = os.getenv("QUERY_COST_PROMPT_USD_PER_1M", "0").strip()
QUERY_COST_COMPLETION_USD_PER_1M = os.getenv("QUERY_COST_COMPLETION_USD_PER_1M", "0").strip()

# Paths
PAPERS_DIR = PROJECT_ROOT / "papers"
PAPERS_LIBRARY_DIR = PAPERS_DIR / "library"
PAPERS_UPLOADS_DIR = PAPERS_DIR / "uploads"
PAPER_LIBRARY_MANIFEST = PAPERS_LIBRARY_DIR / "manifest.json"
FAISS_INDEX_DIR = PROJECT_ROOT / "faiss_index"
ACTIVE_PAPERS_FILE = FAISS_INDEX_DIR / "active_papers.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
JOBS_DIR = PROJECT_ROOT / "jobs"
JOBS_FILE = JOBS_DIR / "jobs.jsonl"
JOBS_DB_FILE = JOBS_DIR / "jobs.sqlite3"
METADATA_DIR = PROJECT_ROOT / "metadata"
CORPUS_METADATA_FILE = METADATA_DIR / "corpus.json"
CORPUS_PROFILES_FILE = METADATA_DIR / "corpus_profiles.json"
CORPUS_METADATA_DB_FILE = METADATA_DIR / "corpus.sqlite3"
CHUNK_METADATA_DB_FILE = METADATA_DIR / "chunks.sqlite3"
RUNTIME_EVENTS_FILE = METADATA_DIR / "runtime_events.jsonl"

# RAG parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
