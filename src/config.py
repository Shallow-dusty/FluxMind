"""FluxMind configuration."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# LLM
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.268526.eu.cc/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "DeepInfra/DeepSeek-V3.2")

# Embedding
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Paths
PAPERS_DIR = PROJECT_ROOT / "papers"
FAISS_INDEX_DIR = PROJECT_ROOT / "faiss_index"

# RAG parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
