# ⚡ FluxMind

**A RAG-based Research Copilot for Sliding Mode Control & Flux Linkage Estimation**

FluxMind is an intelligent research assistant built on Retrieval-Augmented Generation (RAG) architecture, designed to help control engineering researchers and students with theoretical analysis, MATLAB code generation, and literature navigation in the domains of Sliding Mode Control (SMC) and Flux Linkage Estimation.

## Workspace Index

- AI-Prism formal project number: `11`
- Active workspace directory: `11.FluxMind/`
- Previous temporary index `80` has been retired; the pre-formal snapshot is kept under `90.Archive/11-FluxMind-PreFormal/`.

## ✨ Features

- **📖 Literature-Grounded Q&A** — Ask theoretical questions and get answers with citations from your uploaded papers
- **💻 MATLAB Code Generation** — Generate control system code (observers, controllers, Simulink blocks) on demand
- **🧲 Domain Expertise** — Specialized in SMC reaching law design, chattering reduction, MRAS observers, EKF-based estimation
- **📊 Mathematical Support** — LaTeX-formatted equations and derivations
- **🌐 Bilingual** — Supports both English and Chinese queries
- **🗂️ Curated Seed Library** — Bundled open-access papers can be selected manually before rebuilding the RAG index

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                     │
│              (Chat UI / PDF Upload / History)             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Intent & Query Layer                     │
│                                                           │
│  User Query ──► Embedding ──► Vector Similarity Search    │
│                  (local)        (FAISS, top-k=5)          │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐  ┌──────────────────────────────────┐
│  Vector Database  │  │        LLM Generation Layer       │
│  (FAISS Index)    │  │                                    │
│                   │  │  Retrieved Context + System Prompt │
│  PDF ► Chunks ►   │  │         ▼                          │
│  Embeddings ►     │──│  DeepSeek-V3.2 (via OpenAI API)   │
│  Index            │  │         ▼                          │
│                   │  │  Cited Answer / MATLAB Code        │
└──────────────────┘  └──────────────────────────────────┘
```

### Data Pipeline

```
PDF Papers ──► PyMuPDF (text extraction)
           ──► RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
           ──► SentenceTransformer (all-MiniLM-L6-v2, local embedding)
           ──► FAISS Index (L2 similarity, persistent storage)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Conda (recommended) or venv

### Installation

```bash
# Clone the repository
git clone https://github.com/Shallow-dusty/FluxMind.git
cd FluxMind

# Create environment
conda create -n fluxmind python=3.11 -y
conda activate fluxmind

# Install dependencies
pip install -r requirements.txt

# Configure API
cp .env.example .env
# Edit .env with your LLM API credentials
```

### Run

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Verify

```bash
pip install -r requirements-dev.txt
python -m pytest
python scripts/evaluate_rag.py
python scripts/health_check.py
python scripts/health_check.py \
  --url https://smy.hyper-dusty.cloud/ \
  --url https://api-smy.hyper-dusty.cloud/health
python scripts/health_check.py --ssh-host root@100.100.233.26
python scripts/update_local_references.py
```

### Docker

```bash
docker build -t fluxmind .
docker run -p 8501:8501 --env-file .env fluxmind
```

## Deployment Status

Deployment details and live-check commands are recorded in
[`docs/DEPLOYMENT_STATUS.md`](docs/DEPLOYMENT_STATUS.md).
Current public URL: `https://smy.hyper-dusty.cloud/`.

## Project Documents

- [`docs/DEPLOYMENT_STATUS.md`](docs/DEPLOYMENT_STATUS.md) — live deployment snapshot and refresh commands
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — current runtime boundaries and next architecture step
- [`docs/PLATFORM_AUDIT_AND_ROADMAP.md`](docs/PLATFORM_AUDIT_AND_ROADMAP.md) — current audit, bug notes, and platform roadmap
- [`docs/BACKLOG.md`](docs/BACKLOG.md) — implementation work packages and acceptance criteria
- [`docs/demo-script.md`](docs/demo-script.md) — five-minute demo script and defense Q&A
- [`docs/handover.html`](docs/handover.html) — single-file delivery handover

## No-Key Capability Development

`src/capabilities.py` defines provider contracts and `src/providers.py` contains
local no-key implementations for artifact storage, mock SVG diagram generation,
and development-only Python execution. Real external image providers, hosted
sandboxes, MATLAB integration, multi-user identity, quotas, and billing stay
disabled until keys, accounts, licenses, and runtime boundaries are configured.
The current production workflow remains RAG Q&A, corpus selection, PDF
upload/indexing, Streamlit UI, and the token-protected FastAPI `/query`
endpoint.

The Streamlit sidebar includes a local job panel for development workflows. It
submits jobs to an in-process background queue and displays persisted JSONL
status. The FastAPI service exposes both immediate local endpoints and queued
local endpoints:

- `POST /jobs/image/mock`
- `POST /jobs/code/python-local`
- `POST /jobs/index/rebuild`
- `POST /jobs/async/image/mock`
- `POST /jobs/async/code/python-local`
- `POST /jobs/async/index/rebuild`
- `GET /jobs`
- `GET /jobs/{job_id}`
- `POST /jobs/{job_id}/cancel`
- `POST /jobs/{job_id}/retry`

These endpoints persist JSONL job records under `jobs/` and artifacts under
`artifacts/`; both directories are git-ignored runtime state. This queue is a
no-key development bridge, not a durable multi-worker job system.

## RAG Quality Baseline

`eval/rag_baseline.json` records offline control-engineering evaluation cases,
expected source/page references, fixture answers, and provider-error fixtures.
`python scripts/evaluate_rag.py` validates that fixture answers cite retrieved
context refs such as `[1]` and that provider failures normalize to stable
user-facing codes. The `/query` API and Streamlit UI accept answer modes:
`explanation`, `derivation`, `implementation`, `literature_review`, and
`code_generation`.

## 📚 Building the Knowledge Base

1. Use the sidebar paper selector to choose bundled papers from `papers/library/`
2. Click "Apply Selection and Rebuild Index" to rebuild FAISS from the selected papers
3. Upload additional PDFs through the web interface; uploads are stored under `papers/uploads/`

### Recommended Paper Topics

- Sliding Mode Control (SMC) fundamentals and reaching law design
- Higher-order SMC and super-twisting algorithms
- Chattering reduction techniques
- Flux linkage observers (MRAS, Luenberger, EKF)
- PMSM/IM sensorless control
- Adaptive and robust observer design

## 🔧 Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| RAG Framework | LangChain | Orchestration of retrieval and generation pipeline |
| Vector Store | FAISS | Local vector similarity search (no external DB needed) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local text embedding (384-dim) |
| LLM | Any OpenAI-compatible model (default `DeepSeek-V3.2`, configurable via `LLM_MODEL`) | Response generation |
| PDF Parser | PyMuPDF (fitz) | Fast and accurate PDF text extraction |
| Frontend | Streamlit | Interactive chat interface with file upload |

## 📁 Project Structure

```
FluxMind/
├── app.py                 # Streamlit application entry point
├── src/
│   ├── config.py          # Configuration and environment variables
│   ├── embeddings.py      # Local embedding model setup
│   ├── ingestion.py       # PDF loading, chunking, and indexing
│   ├── capabilities.py    # Image/code execution provider contracts
│   ├── providers.py       # Local no-key provider implementations
│   ├── jobs.py            # Local JSONL job records, runner, and async queue
│   ├── evaluation.py      # Offline RAG evaluation and citation checks
│   └── chain.py           # RAG chain (retrieval + LLM generation)
├── papers/                # Research paper PDFs (git-ignored)
├── faiss_index/           # Persistent FAISS index (git-ignored)
├── artifacts/             # Generated local artifacts (git-ignored)
├── jobs/                  # Local job records (git-ignored)
├── assets/                # Architecture diagrams and images
├── docs/                  # Additional documentation
├── eval/                  # Offline RAG evaluation fixtures
├── scripts/               # Local and deployment health checks
├── tests/                 # Regression tests
├── Dockerfile             # Container deployment
├── requirements.txt       # Python dependencies
└── .env.example           # Environment variable template
```

## 📄 License

MIT
