"""FluxMind configuration."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
try:
    load_dotenv(PROJECT_ROOT / ".env")
except PermissionError:
    # Production systemd units load /opt/fluxmind/.env through EnvironmentFile
    # while keeping the file root-only. In that case the process environment is
    # already populated and direct dotenv reads should not prevent startup.
    pass

# LLM
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.example.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "DeepSeek-V3.2")
# Auxiliary/fallback model: small tasks, assist, and final fallback when primary fails.
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "").strip()


def _env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _project_path_from_env(name: str, default: Path) -> Path:
    path = Path(os.getenv(name, str(default))).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


# Embedding
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "")

# Local execution backend. `local` keeps the current child-process provider.
# `docker` runs jobs through language-specific local Docker images when the
# runtime user can access Docker.
CODE_EXECUTION_BACKEND = os.getenv("CODE_EXECUTION_BACKEND", "local").strip().lower()
DOCKER_EXECUTION_IMAGE = os.getenv("DOCKER_EXECUTION_IMAGE", "python:3.11-slim").strip() or "python:3.11-slim"
DOCKER_PYTHON_EXECUTION_IMAGE = os.getenv(
    "DOCKER_PYTHON_EXECUTION_IMAGE",
    DOCKER_EXECUTION_IMAGE or "python:3.11-slim",
).strip() or DOCKER_EXECUTION_IMAGE
DOCKER_OCTAVE_EXECUTION_IMAGE = os.getenv(
    "DOCKER_OCTAVE_EXECUTION_IMAGE",
    "gnuoctave/octave:latest",
).strip() or "gnuoctave/octave:latest"
CODE_EXECUTION_POLICY = os.getenv("CODE_EXECUTION_POLICY", "local-safe-v1").strip().lower()
CODE_EXECUTION_ALLOWED_IMPORTS = os.getenv(
    "CODE_EXECUTION_ALLOWED_IMPORTS",
    "collections,csv,dataclasses,decimal,fractions,itertools,json,math,matplotlib,numpy,pathlib,random,statistics,time,typing",
).strip()
CODE_EXECUTION_MAX_STDOUT_BYTES = int(os.getenv("CODE_EXECUTION_MAX_STDOUT_BYTES", "65536"))
CODE_EXECUTION_MAX_STDERR_BYTES = int(os.getenv("CODE_EXECUTION_MAX_STDERR_BYTES", "65536"))
CODE_EXECUTION_MAX_ARTIFACTS = int(os.getenv("CODE_EXECUTION_MAX_ARTIFACTS", "16"))
CODE_EXECUTION_MAX_ARTIFACT_BYTES = int(os.getenv("CODE_EXECUTION_MAX_ARTIFACT_BYTES", str(2 * 1024 * 1024)))
CODE_EXECUTION_MAX_ARTIFACT_TOTAL_BYTES = int(
    os.getenv("CODE_EXECUTION_MAX_ARTIFACT_TOTAL_BYTES", str(8 * 1024 * 1024))
)
CODE_EXECUTION_MAX_ARTIFACT_CANDIDATES = int(
    os.getenv("CODE_EXECUTION_MAX_ARTIFACT_CANDIDATES", "256")
)
IMAGE_PROVIDER_BACKEND = os.getenv("IMAGE_PROVIDER_BACKEND", "local-mock").strip()
OPENAI_IMAGE_API_KEY = os.getenv(
    "OPENAI_IMAGE_API_KEY",
    os.getenv("OPENAI_API_KEY", ""),
).strip()
OPENAI_IMAGE_BASE_URL = os.getenv("OPENAI_IMAGE_BASE_URL", "").strip()
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2").strip() or "gpt-image-2"
OPENAI_IMAGE_QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "low").strip() or "low"
OPENAI_IMAGE_OUTPUT_FORMAT = os.getenv("OPENAI_IMAGE_OUTPUT_FORMAT", "png").strip() or "png"
# Some OpenAI-compatible relays hang on the `output_format` query param. Keep it
# off by default; enable only when the upstream confirms it accepts the param.
OPENAI_IMAGE_SEND_OUTPUT_FORMAT = _env_flag("OPENAI_IMAGE_SEND_OUTPUT_FORMAT", "false")
OPENAI_IMAGE_TIMEOUT_S = int(os.getenv("OPENAI_IMAGE_TIMEOUT_S", "180"))
PROVIDER_QUOTA_GUARD_ENABLED = _env_flag("PROVIDER_QUOTA_GUARD_ENABLED", "false")
PROVIDER_QUOTA_MAX_PROMPT_TOKENS_PER_REQUEST = int(
    os.getenv("PROVIDER_QUOTA_MAX_PROMPT_TOKENS_PER_REQUEST", "128000")
)
PROVIDER_QUOTA_MAX_COMPLETION_TOKENS_PER_REQUEST = int(
    os.getenv("PROVIDER_QUOTA_MAX_COMPLETION_TOKENS_PER_REQUEST", "4096")
)
PROVIDER_QUOTA_MAX_COST_USD_PER_REQUEST = os.getenv(
    "PROVIDER_QUOTA_MAX_COST_USD_PER_REQUEST",
    "0",
).strip()

# Optional local cost estimation.
QUERY_COST_PROVIDER = os.getenv("QUERY_COST_PROVIDER", "").strip()
QUERY_COST_PROMPT_USD_PER_1M = os.getenv("QUERY_COST_PROMPT_USD_PER_1M", "0").strip()
QUERY_COST_COMPLETION_USD_PER_1M = os.getenv("QUERY_COST_COMPLETION_USD_PER_1M", "0").strip()
API_RATE_LIMIT_ENABLED = _env_flag("API_RATE_LIMIT_ENABLED", "false")
API_RATE_LIMIT_MAX_REQUESTS = int(os.getenv("API_RATE_LIMIT_MAX_REQUESTS", "300"))
API_RATE_LIMIT_WINDOW_S = int(os.getenv("API_RATE_LIMIT_WINDOW_S", "60"))

UPLOAD_SCAN_ENABLED = _env_flag("UPLOAD_SCAN_ENABLED", "true")
UPLOAD_SCAN_REJECT_ENCRYPTED = _env_flag("UPLOAD_SCAN_REJECT_ENCRYPTED", "true")
UPLOAD_SCAN_BLOCK_ACTIVE_CONTENT = _env_flag("UPLOAD_SCAN_BLOCK_ACTIVE_CONTENT", "true")
UPLOAD_SCAN_MAX_PAGES = int(os.getenv("UPLOAD_SCAN_MAX_PAGES", "500"))
RETENTION_DELETE_ENABLED = _env_flag("RETENTION_DELETE_ENABLED", "false")

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
USER_STORE_FILE = _project_path_from_env(
    "FLUXMIND_USER_STORE_FILE",
    METADATA_DIR / "users.sqlite3",
)
# RAG parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
