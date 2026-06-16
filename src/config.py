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
# `docker` enables the no-key container backend when the runtime user can access
# Docker safely.
CODE_EXECUTION_BACKEND = os.getenv("CODE_EXECUTION_BACKEND", "local").strip().lower()
DOCKER_EXECUTION_IMAGE = os.getenv("DOCKER_EXECUTION_IMAGE", "python:3.12-slim")
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
CODE_EXECUTION_ALERT_MIN_EVENTS = int(os.getenv("CODE_EXECUTION_ALERT_MIN_EVENTS", "5"))
CODE_EXECUTION_ALERT_FAILURE_RATE = float(os.getenv("CODE_EXECUTION_ALERT_FAILURE_RATE", "0.5"))
CODE_EXECUTION_ALERT_DURATION_MS = int(os.getenv("CODE_EXECUTION_ALERT_DURATION_MS", "30000"))

# External provider activation readiness. Local mock/image and local execution
# remain active; these settings only report future real provider activation
# targets and never expose credential values.
EXTERNAL_PROVIDERS_ENABLED = _env_flag("EXTERNAL_PROVIDERS_ENABLED", "false")
IMAGE_PROVIDER_BACKEND = os.getenv("IMAGE_PROVIDER_BACKEND", "local-mock").strip()
IMAGE_PROVIDER_API_CONFIGURED = _env_flag("IMAGE_PROVIDER_API_CONFIGURED", "false")
HOSTED_EXECUTION_BACKEND = os.getenv("HOSTED_EXECUTION_BACKEND", "none").strip()
HOSTED_EXECUTION_CONFIGURED = _env_flag("HOSTED_EXECUTION_CONFIGURED", "false")
MATLAB_BACKEND = os.getenv("MATLAB_BACKEND", "none").strip()
MATLAB_LICENSE_CONFIGURED = _env_flag("MATLAB_LICENSE_CONFIGURED", "false")
PROVIDER_QUOTA_GUARD_ENABLED = _env_flag("PROVIDER_QUOTA_GUARD_ENABLED", "false")

# Storage backend readiness. Local JSON/SQLite/filesystem storage remains the
# active no-key backend; external database/object storage requires explicit
# configuration and is only reported as readiness here.
METADATA_STORAGE_BACKEND = os.getenv("METADATA_STORAGE_BACKEND", "local").strip().lower()
DATABASE_URL = os.getenv("DATABASE_URL", "")
OBJECT_STORAGE_BACKEND = os.getenv("OBJECT_STORAGE_BACKEND", "local").strip().lower()
OBJECT_STORAGE_BUCKET = os.getenv("OBJECT_STORAGE_BUCKET", "")
OBJECT_STORAGE_ENDPOINT = os.getenv("OBJECT_STORAGE_ENDPOINT", "")
OBJECT_STORAGE_REGION = os.getenv("OBJECT_STORAGE_REGION", "")

# Distributed worker readiness. Local SQLite/JSONL remains the active job store;
# external job-store configuration is reported as a no-secret readiness contract
# until a deliberate migration activates a distributed backend.
DISTRIBUTED_JOB_STORE_BACKEND = os.getenv("DISTRIBUTED_JOB_STORE_BACKEND", "local").strip().lower()
DISTRIBUTED_JOB_STORE_URL = os.getenv("DISTRIBUTED_JOB_STORE_URL", "")
DISTRIBUTED_JOB_QUEUE_NAME = os.getenv("DISTRIBUTED_JOB_QUEUE_NAME", "fluxmind-jobs").strip()

# Optional local cost estimation. These rates are no-secret configuration used
# only for admin estimates; FluxMind does not connect to external billing.
QUERY_COST_PROVIDER = os.getenv("QUERY_COST_PROVIDER", "").strip()
QUERY_COST_PROMPT_USD_PER_1M = os.getenv("QUERY_COST_PROMPT_USD_PER_1M", "0").strip()
QUERY_COST_COMPLETION_USD_PER_1M = os.getenv("QUERY_COST_COMPLETION_USD_PER_1M", "0").strip()
QUERY_ALERT_MIN_EVENTS = int(os.getenv("QUERY_ALERT_MIN_EVENTS", "5"))
QUERY_ALERT_DURATION_MS = int(os.getenv("QUERY_ALERT_DURATION_MS", "15000"))
RETRIEVAL_TRACE_ALERT_MIN_EVENTS = int(os.getenv("RETRIEVAL_TRACE_ALERT_MIN_EVENTS", "5"))
RETRIEVAL_TRACE_ALERT_EMPTY_RATE = float(os.getenv("RETRIEVAL_TRACE_ALERT_EMPTY_RATE", "0.25"))
RETRIEVAL_TRACE_ALERT_SOURCE_PAGE_INCOMPLETE_RATE = float(
    os.getenv("RETRIEVAL_TRACE_ALERT_SOURCE_PAGE_INCOMPLETE_RATE", "0.25")
)
RETRIEVAL_TRACE_ALERT_CITATION_FAILURE_RATE = float(
    os.getenv("RETRIEVAL_TRACE_ALERT_CITATION_FAILURE_RATE", "0.25")
)
PROVIDER_FAILURE_ALERT_MIN_EVENTS = int(os.getenv("PROVIDER_FAILURE_ALERT_MIN_EVENTS", "3"))
PROVIDER_FAILURE_ALERT_RATE = float(os.getenv("PROVIDER_FAILURE_ALERT_RATE", "0.25"))
JOB_ALERT_FAILED_MIN_EVENTS = int(os.getenv("JOB_ALERT_FAILED_MIN_EVENTS", "3"))
JOB_ALERT_EXPIRED_MIN_EVENTS = int(os.getenv("JOB_ALERT_EXPIRED_MIN_EVENTS", "1"))
API_ACCESS_AUDIT_ENABLED = _env_flag("API_ACCESS_AUDIT_ENABLED", "true")
API_RATE_LIMIT_ENABLED = _env_flag("API_RATE_LIMIT_ENABLED", "false")
API_RATE_LIMIT_MAX_REQUESTS = int(os.getenv("API_RATE_LIMIT_MAX_REQUESTS", "300"))
API_RATE_LIMIT_WINDOW_S = int(os.getenv("API_RATE_LIMIT_WINDOW_S", "60"))

# Productization readiness. These values are no-secret configuration signals
# for future identity, quota, and billing activation; the current local runtime
# remains no-key/no-account unless explicitly changed by later implementation.
FLUXMIND_API_TOKEN_CONFIGURED = bool(os.getenv("FLUXMIND_API_TOKEN", "").strip())
IDENTITY_PROVIDER = os.getenv("FLUXMIND_IDENTITY_PROVIDER", os.getenv("IDENTITY_PROVIDER", "none")).strip()
API_KEY_REGISTRY_BACKEND = os.getenv(
    "FLUXMIND_API_KEY_REGISTRY_BACKEND",
    os.getenv("API_KEY_REGISTRY_BACKEND", "none"),
).strip()
QUOTA_STORE_BACKEND = os.getenv(
    "FLUXMIND_QUOTA_STORE_BACKEND",
    os.getenv("QUOTA_STORE_BACKEND", "none"),
).strip()
BILLING_PROVIDER = os.getenv(
    "FLUXMIND_BILLING_PROVIDER",
    os.getenv("BILLING_PROVIDER", "none"),
).strip()
BILLING_ATTRIBUTION_ENABLED = _env_flag("FLUXMIND_BILLING_ATTRIBUTION_ENABLED", "false")
IDENTITY_QUOTAS_BILLING_ENABLED = _env_flag("IDENTITY_QUOTAS_BILLING_ENABLED", "false")
PRODUCT_REGISTRY_BACKEND = os.getenv(
    "FLUXMIND_PRODUCT_REGISTRY_BACKEND",
    os.getenv("PRODUCT_REGISTRY_BACKEND", "none"),
).strip()
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
API_KEY_REGISTRY_FILE = _project_path_from_env(
    "FLUXMIND_API_KEY_REGISTRY_FILE",
    METADATA_DIR / "api_keys.sqlite3",
)
PRODUCT_REGISTRY_FILE = _project_path_from_env(
    "FLUXMIND_PRODUCT_REGISTRY_FILE",
    METADATA_DIR / "product_registry.sqlite3",
)

# RAG parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
