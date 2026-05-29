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

# RAG parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
