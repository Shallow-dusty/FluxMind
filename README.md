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

# Create environment with conda
conda create -n fluxmind python=3.11 -y
conda activate fluxmind

# Install dependencies
pip install -r requirements.txt

# Configure API
cp .env.example .env
# Edit .env with your LLM API credentials
```

For the current local checkout, an existing `.venv` may already contain the
development dependencies. Use it directly when present:

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### Run

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Verify

```bash
python -m pytest
python scripts/evaluate_rag.py
python scripts/evaluate_rag.py --json-report artifacts/eval/latest.json
python scripts/evaluate_rag.py --retrieval-url http://127.0.0.1:18502
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
development-only Python execution, and GNU Octave-compatible local execution
when an `octave` binary is installed. Generated file/plot artifacts are
captured for both execution providers, with input paths constrained to the
per-run workdir, symlink outputs excluded from artifact export, and file
count/byte limits enforced before local materialization. Unix child processes
receive address-space and CPU-time limits where supported. Execution results
persist no-secret reproducibility metadata for the language, entrypoint, input
file counts/bytes, provider runtime, local runtime version, temporary workdir
isolation, and the current fact that network policy is not enforced by these
development providers. `CODE_EXECUTION_BACKEND` and `DOCKER_EXECUTION_IMAGE`
define a future no-key Docker sandbox backend, and admin status reports whether
that backend is configured and accessible by the runtime user. Real external
image providers, hosted sandboxes, MATLAB
integration, multi-user identity, quotas, and billing stay disabled until keys,
accounts, licenses, and runtime boundaries are configured.
The current production workflow remains RAG Q&A, corpus selection, PDF
upload/indexing, Streamlit UI, and the token-protected FastAPI `/query`
endpoint.

The Streamlit sidebar includes a local job panel for development workflows. It
submits jobs to an in-process background queue and displays persisted JSONL
status with cancel/retry controls plus a local artifact gallery with download
buttons, stable artifact IDs, and provider-neutral metadata including byte
counts and SHA-256 checksums. Admin status verifies local artifact integrity
against those checksums without exposing artifact contents. The FastAPI
service exposes both immediate local endpoints and queued local endpoints:

- `POST /jobs/image/mock`
- `POST /jobs/code/python-local`
- `POST /jobs/code/octave-local`
- `POST /jobs/index/rebuild`
- `POST /jobs/async/image/mock`
- `POST /jobs/async/code/python-local`
- `POST /jobs/async/code/octave-local`
- `POST /jobs/async/index/rebuild`
- `GET /jobs`
- `GET /jobs/{job_id}`
- `POST /jobs/{job_id}/cancel`
- `POST /jobs/{job_id}/retry`
- `POST /jobs/{job_id}/retry-scheduled`
- `GET /artifacts`
- `GET /artifacts/{artifact_id}`
- `GET /corpus/papers`
- `GET /corpus/chunks`
- `GET /corpus/status`
- `PUT /corpus/active`
- `GET /corpus/profiles`
- `POST /corpus/profiles`
- `GET /corpus/profiles/{profile_id}/status`
- `GET /corpus/profiles/{profile_id}/report`
- `POST /corpus/profiles/{profile_id}/activate`
- `POST /corpus/profiles/{profile_id}/rebuild`
- `POST /query/inspect`
- `POST /query/report`
- `GET /admin/status`
- `GET /admin/status/report`

`GET /jobs` supports local metadata filters with `q`, `status`, and `kind`, and
the Streamlit recent-job panel exposes matching local controls.
`GET /artifacts` supports local metadata filters with `q`, `kind`, and
`job_kind`, and the Streamlit artifact gallery exposes matching local controls
before download.

These endpoints persist JSONL job records under `jobs/` and artifacts under
`artifacts/`; both directories are git-ignored runtime state. Corpus paper
metadata is persisted under `metadata/`, also git-ignored runtime state, and is
mirrored to `metadata/corpus.sqlite3` as a local current-state index. Paper
records carry bibliographic fields such as authors, year, DOI, arXiv ID, venue,
and topic tags when available; uploaded PDFs get best-effort no-key extraction
from embedded metadata and first-page title, author, DOI/arXiv, year, and
keyword/index-term text. Uploaded PDFs are
deduplicated by SHA-256 against the current selectable corpus before writing a
new local file or adding duplicate chunks. Indexed chunk metadata is mirrored to
`metadata/chunks.sqlite3` with source/page/hash fields for local inspection and
future citation/storage migration. This queue and metadata storage are no-key
development bridges, not durable multi-worker
infrastructure. Job writes are also mirrored to `jobs/jobs.sqlite3` as a local
current-state index for faster status queries and future worker migration. On
API startup, queued/scheduled local jobs are rehydrated from durable state and
returned to the in-process worker queue; this preserves no-key delayed retries
across service restarts, but it is still not a multi-worker distributed queue.
Before an in-process worker starts a queued job, it claims the record with
`worker_id`, `leased_at`, and `lease_expires_at` metadata in the same local
SQLite/JSONL state. Expired queued leases can be reclaimed, giving future
distributed workers a concrete local lease contract without enabling external
worker infrastructure yet.
`scripts/run_job_worker.py` provides an explicit local durable worker entrypoint
that claims due queued jobs from the same store and executes them outside the
API/Streamlit request path. `deploy/systemd/fluxmind-worker.service` is the
matching no-key local systemd unit for running that worker continuously. This is
a local worker-service bridge, not a distributed database-backed queue. While
running local providers, the explicit durable worker polls job state and
forwards `cancelled` status through the provider cancellation event so local
child processes can terminate cleanly.
Failed/cancelled jobs can also be scheduled for delayed local retry with
`not_before` metadata and parent-job lineage. Queued local jobs and scheduled
retries can set `queue_timeout_s`; expired queued records fail before execution
with `job_deadline_exceeded`.
Job records include no-secret transition logs for queued, running, terminal, and
cancelled states, so job history can be inspected without scraping process logs.
Running Python jobs observe cancellation, and selected-PDF index rebuild jobs
check cancellation during loading/splitting and before committing rebuilt FAISS
or chunk metadata state.
`GET /corpus/status` summarizes the local corpus lifecycle as `queued`,
`parsing`, `indexed`, `failed`, `stale`, or `empty`, including recent index
rebuild job state and index freshness.
`GET /corpus/papers` supports lightweight local metadata filters with `q`,
`active`, `source_kind`, and `indexed_status` query parameters.
`GET /corpus/profiles`, `POST /corpus/profiles`, and
`GET /corpus/profiles/{profile_id}/status` persist and inspect reusable no-key
local corpus selections under `metadata/corpus_profiles.json`, so different
paper sets can coexist before real users/workspaces are introduced. Profile
status reports paper availability, active-selection match, index freshness, and
whether a rebuild is required before activation. The Streamlit corpus profile
panel exposes the same no-secret Markdown report as a download.
`POST /corpus/profiles/{profile_id}/rebuild` activates a saved profile and
queues the selected-PDF FAISS rebuild through the same local async job manager
used by `POST /jobs/async/index/rebuild`.
`POST /query/report` returns the generated answer, citation validation, and
retrieved context refs as a Markdown research report for local export.
`GET /admin/status` exposes no-secret runtime counts for jobs, corpus papers,
queue health, artifacts, runtime directories, and disabled
external-provider/productization switches. It also reports no-secret durable
storage readiness for future metadata database and object-storage backends:
local JSON/SQLite/filesystem storage remains active, while external database
URLs, buckets, and endpoints are surfaced only as configured booleans and are
not connected or exposed. Successful `/query` and
`/query/inspect` calls append no-secret estimated usage events with character
counts and rough token estimates. When the provider response exposes token
usage, FluxMind also stores no-secret provider prompt/completion/total token
counts in the same local runtime event; this is usage visibility, not billing.
`GET /admin/status/report` exports the same no-secret snapshot
as a Markdown operations report for handoff or offline review. `GET
/admin/retention` previews upload and artifact files that would match local
age-based retention thresholds; it is no-delete by design. The Streamlit admin
panel exposes the same retention preview with local day/limit controls. `GET
/admin/events` lists no-secret runtime events with local `kind`, `code`, and
`q` filters, and the Streamlit admin panel exposes the same event inspection
controls.
Generated mock diagrams and execution outputs include no-key metadata such as
prompt, style, size, source references, model/provider, and zero-cost estimates.
Artifact metadata is mirrored into `artifacts/artifacts.sqlite3` as a local
current-state index while the job history remains the append-only source of
execution transitions.
Recent artifacts are also injected into the RAG prompt as stable
`[Artifact:<id>]` references so answers can point to generated local diagrams,
plots, or files without activating a real image provider.
`PUT /corpus/active` lets API clients persist paper activation/deactivation
state without editing `active_papers.json` directly; a rebuild is still required
to make the FAISS index exactly match the changed selection.
`GET /corpus/chunks` exposes local indexed chunk metadata for source/page/hash
inspection with `source_path`, `page`, and `q` filters, without reading FAISS
internals.
`GET /corpus/profiles/{profile_id}/report` exports a no-secret Markdown status
snapshot for one saved local corpus profile.
`POST /query/retrieve` returns retrieved context refs, the citation guard, and
source/page completeness diagnostics without calling the LLM provider.
`POST /query/inspect` returns generated-answer citation validation and retrieved
context refs for local or deployed quality checks without changing the
compatibility-oriented `/query` response.

## RAG Quality Baseline

`eval/rag_baseline.json` records offline control-engineering evaluation cases,
expected source/page references, fixture answers, recorded answers, key answer
terms, and provider-error fixtures. `python scripts/evaluate_rag.py` validates
that expected source PDFs/pages contain their configured snippets, that fixture
and recorded answers cite retrieved context refs such as `[1]`, that recorded
answers meet deterministic key-term coverage thresholds, and that provider
failures normalize to stable user-facing codes. The same command also evaluates
aggregate `quality_gates` so the eval set fails if it loses required answer-mode
coverage, minimum case/source-ref breadth, recorded-answer pass rate, or average
term coverage. The `/query` API and Streamlit UI accept answer modes:
`explanation`, `derivation`, `implementation`, `literature_review`, and
`code_generation`. Retrieval uses FAISS vector search plus local BM25-lite
keyword supplementation from the indexed docstore, deterministic BM25-lite
reranking, optional local CrossEncoder reranking when `RERANKER_MODEL` points to
an existing local model path, dedupe, and a `TOP_K` context cap. FluxMind does
not download reranker models at runtime; an empty or missing reranker path
falls back to BM25-lite.
`POST /query/retrieve` exposes that retrieval layer directly for no-key
deployment checks before running generation.
For deployed checks, add `--retrieval-url <api-base>` to score
`/query/retrieve` responses without provider generation, or `--live-url
<api-base>` to score `/query/inspect` responses for citation validity,
expected-source retrieval coverage, answer-term coverage, and configured live
aggregate pass-rate thresholds. The API token is read from
`FLUXMIND_API_TOKEN` by default; do not commit tokens. Add `--json-report
<path>` to write a no-secret machine-readable summary for CI or deployment
evidence.

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
| Optional Reranker | sentence-transformers CrossEncoder via local `RERANKER_MODEL` path | No-key learned reranking when a local model is installed |
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
│   ├── jobs.py            # Local JSONL job records, runners, async queue, worker loop
│   ├── admin.py           # No-secret local runtime/admin status
│   ├── metadata.py        # Local JSON corpus metadata registry
│   ├── evaluation.py      # Offline RAG evaluation and citation checks
│   └── chain.py           # RAG chain (retrieval + LLM generation)
├── papers/                # Research paper PDFs (git-ignored)
├── faiss_index/           # Persistent FAISS index (git-ignored)
├── artifacts/             # Generated local artifacts (git-ignored)
├── jobs/                  # Local job records (git-ignored)
├── metadata/              # Local corpus metadata records (git-ignored)
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
