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
python scripts/storage_schema.py --format markdown
python scripts/health_check.py \
  --url https://smy.hyper-dusty.cloud/ \
  --url https://api-smy.hyper-dusty.cloud/health
python scripts/health_check.py --ssh-host root@100.100.233.26
python scripts/update_local_references.py
```

### Deploy Sync

Use the guarded sync script instead of invoking `rsync --delete` directly. It
is a dry run unless `--apply` is passed, and it always excludes server runtime
state such as `.env`, `venv/`, `models/`, metadata, jobs, artifacts, uploaded
papers, and the FAISS index.

```bash
python scripts/deploy_sync.py
python scripts/deploy_sync.py --apply --restart
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

- [`docs/README.md`](docs/README.md) — documentation index, reading order, and source-of-truth map
- [`docs/REPO_STATUS.md`](docs/REPO_STATUS.md) — current git/worktree snapshot and verification record
- [`docs/DEPLOYMENT_STATUS.md`](docs/DEPLOYMENT_STATUS.md) — live deployment snapshot and refresh commands
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — current runtime boundaries and next architecture step
- [`docs/PLATFORM_AUDIT_AND_ROADMAP.md`](docs/PLATFORM_AUDIT_AND_ROADMAP.md) — current audit, bug notes, and platform roadmap
- [`docs/PRODUCTION_GAP_AND_MARKET_RESEARCH.md`](docs/PRODUCTION_GAP_AND_MARKET_RESEARCH.md) — production gap, competitor scan, and community demand research
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
isolation, and the current fact that network policy is not enforced by the
child-process development providers. `CODE_EXECUTION_MAX_STDOUT_BYTES` and
`CODE_EXECUTION_MAX_STDERR_BYTES` cap captured stdout/stderr bytes for local and
Docker execution; result metadata records total observed bytes plus
`stdout_truncated`, `stderr_truncated`, and `output_truncated` flags.
`CODE_EXECUTION_MAX_ARTIFACTS`, `CODE_EXECUTION_MAX_ARTIFACT_BYTES`,
`CODE_EXECUTION_MAX_ARTIFACT_TOTAL_BYTES`, and
`CODE_EXECUTION_MAX_ARTIFACT_CANDIDATES` bound generated-artifact export for
local and Docker execution; result metadata records exported/skipped counts,
bytes, and `artifact_collection_truncated` when output files exceed those
limits. Streamlit includes local Python and Octave-compatible
control-engineering templates for SMC/PMSM examples so artifact-producing jobs
can be started without writing a blank script from scratch.
`CODE_EXECUTION_BACKEND=docker` switches Python/Octave-compatible code
jobs to a no-key local Docker backend that runs `docker run` with network
disabled, a bind-mounted per-run workdir, read-only root filesystem, memory,
CPU, and PID limits, dropped capabilities, and `no-new-privileges`.
`DOCKER_EXECUTION_IMAGE` selects the container image, and admin status reports
whether Docker is configured and accessible by the runtime user.
`CODE_EXECUTION_POLICY` defaults to `local-safe-v1`, which rejects disallowed
Python imports, shell/package manager commands, absolute-path literals in common
file constructors, and Octave/MATLAB-compatible shell or network calls before
any local process or container starts. `CODE_EXECUTION_ALLOWED_IMPORTS` controls
the Python import allowlist. Real external image providers, hosted sandboxes,
MATLAB integration, multi-user identity, quotas, and billing stay disabled until
keys, accounts, licenses, and runtime boundaries are configured.
Optional `QUERY_COST_*` rates only drive local no-secret USD estimates in admin
views; they do not activate billing or provider account integration.
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
- `GET /corpus/structure`
- `GET /corpus/structure/report`
- `GET /corpus/status`
- `PUT /corpus/active`
- `GET /corpus/profiles`
- `POST /corpus/profiles`
- `GET /corpus/profiles/{profile_id}/status`
- `GET /corpus/profiles/{profile_id}/report`
- `POST /corpus/profiles/{profile_id}/activate`
- `POST /corpus/profiles/{profile_id}/rebuild`
- `POST /query/inspect`
- `POST /query/retrieve`
- `POST /query/report`
- `GET /admin/status`
- `GET /admin/status/report`
- `GET /admin/metrics`
- `GET /admin/runtime-manifest`
- `GET /admin/runtime-manifest/report`
- `POST /admin/runtime-manifest/restore-check`
- `POST /admin/runtime-manifest/restore-check/report`
- `GET /admin/retention`
- `POST /admin/retention/delete`
- `GET /admin/events`

`GET /jobs` supports local metadata filters with `q`, `status`, `kind`, and
`owner_id`, and the Streamlit recent-job panel exposes matching local controls.
Immediate and async job-creation requests can include an optional
`idempotency_key` of up to 128 characters. For the same job kind and key,
duplicate local submissions return the existing persisted job through the
durable SQLite idempotency claim table; omitted keys preserve the previous
"create a new job" behavior. Corpus profile rebuild requests use the same
idempotency path as async index rebuild jobs.
Query and job-creation requests can also include optional local `owner_id` and
`owner_label` metadata. When omitted, FluxMind records the no-key default
`local-user` / `Local user`; when supplied, the values are persisted on durable
job records, job transition logs, query runtime events, generated artifact
records, and admin status owner summaries. These fields are for local
inspection and future migration shape only; they do not authenticate users,
isolate tenants, enforce quotas, or enable billing.
Async job requests can also set `max_attempts` and `retry_backoff_s`. Failed
queued attempts are requeued until the attempt cap is exhausted, then the same
job is marked `dead_lettered` with `dead_lettered_at` and transition logs. The
default remains one attempt, so existing jobs keep the previous single-run
behavior unless a retry policy is requested. Manual retry endpoints can create a
fresh retry from failed, cancelled, or dead-lettered jobs.
Code execution attempts also append no-secret `code_execution` runtime events
with job id, owner metadata, language, backend, status/error code, duration,
artifact count, output/artifact limit metadata, and policy metadata; submitted
source files, stdout, and stderr are not copied into those events. Admin status
and the Streamlit admin panel summarize recent code-execution failure rate,
duration, policy violations, output truncation, artifact truncation, and
configurable advisory alert thresholds without exposing submitted source.
`CODE_EXECUTION_ALERT_MIN_EVENTS`, `CODE_EXECUTION_ALERT_FAILURE_RATE`, and
`CODE_EXECUTION_ALERT_DURATION_MS` control those local alert thresholds.
FastAPI middleware also appends metadata-only `api_access` runtime events when
`API_ACCESS_AUDIT_ENABLED` is enabled. These events classify token checks as
`not_configured`, `valid`, `missing`, or `invalid` and record only method, path,
status code, duration, credential type, and request ID when present. They do not
copy token values, headers, request bodies, prompts, answers, client IPs, or
uploaded/runtime file contents. Admin status/report and the Streamlit runtime
status panel summarize recent API access by token status, HTTP status code, and
method.
`API_RATE_LIMIT_ENABLED` can turn on a local in-memory request-rate guard with
`API_RATE_LIMIT_MAX_REQUESTS` over `API_RATE_LIMIT_WINDOW_S`. When enabled, the
middleware returns HTTP 429 before route handling once the local bucket is
exhausted, emits only metadata-only `api_access` rate-limit fields, and exposes
standard `X-RateLimit-*` response headers. The limiter is a local guardrail for
the current deployment shape, not identity-backed quotas or billing.
PDF uploads now pass through a local pre-write scan controlled by
`UPLOAD_SCAN_ENABLED`, `UPLOAD_SCAN_MAX_PAGES`,
`UPLOAD_SCAN_REJECT_ENCRYPTED`, and `UPLOAD_SCAN_BLOCK_ACTIVE_CONTENT`.
The scan validates PDF magic and PyMuPDF parseability, rejects encrypted PDFs
by default, blocks common active-content markers such as JavaScript, launch
actions, embedded files, rich media, and XFA, and records only metadata-only
`upload_scan` runtime events with reason codes, byte counts, page counts, and
threshold config. It does not store filenames, uploaded contents, checksums, or
request bodies in scan events, and it is a local abuse guardrail rather than a
production antivirus or sandbox-scanning service.
`GET /artifacts` supports local metadata filters with `q`, `kind`, `job_kind`,
and `owner_id`, and the Streamlit artifact gallery exposes matching local
controls before download.

These endpoints persist JSONL job records under `jobs/` and artifacts under
`artifacts/`; both directories are git-ignored runtime state. Corpus paper
metadata is persisted under `metadata/`, also git-ignored runtime state, and is
mirrored to `metadata/corpus.sqlite3` as a local current-state index. Paper
records carry bibliographic fields such as authors, year, DOI, arXiv ID, venue,
and topic tags when available; uploaded PDFs get best-effort no-key extraction
from embedded metadata and first-page title, author, DOI/arXiv, year, and
keyword/index-term text. Uploaded PDFs are scanned before local write, then
deduplicated by SHA-256 against the current selectable corpus before writing a
new local file or adding duplicate chunks. Indexed chunk metadata is mirrored to
`metadata/chunks.sqlite3` with source/page/hash fields for local inspection and
future citation/storage migration. This queue and metadata storage are no-key
development bridges, not durable multi-worker
infrastructure. Job writes are also mirrored to `jobs/jobs.sqlite3` as a local
current-state index for faster status queries, durable idempotency claims,
bounded retry/dead-letter metadata, and future worker migration. On API startup,
queued/scheduled local jobs are rehydrated from durable state and returned to
the in-process worker queue; this preserves no-key delayed retries and retry
policy state across service restarts, but it is still not a multi-worker
distributed queue.
Before an in-process worker starts a queued job, it claims the record with
`worker_id`, `leased_at`, and `lease_expires_at` metadata in the same local
SQLite/JSONL state. Expired queued leases can be reclaimed, giving future
distributed workers a concrete local lease contract without enabling external
worker infrastructure yet.
Admin status/report also expose metadata-only local job-health advisory alerts
for recent failed jobs, dead-lettered jobs, expired queued deadlines, and
expired worker leases. `JOB_ALERT_FAILED_MIN_EVENTS` and
`JOB_ALERT_EXPIRED_MIN_EVENTS` tune those local thresholds.
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
`GET /corpus/structure` and `GET /corpus/structure/report` expose no-key PDF
layout anchors with `source_path`, `kind`, `page`, `q`, and `limit` filters for
JSON inspection or Markdown handoff export.
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
retrieved context refs as a Markdown research report for local export. For
implementation and code-generation requests it also adds a paper-to-code
handoff section with source refs, assumption/parameter guardrails, fenced code
blocks, cited artifact IDs, and validation checklist fields.
`GET /admin/status` exposes no-secret runtime counts for jobs, job/artifact
owner metadata, corpus papers, queue health, worker lease activity, job-health
alerts, API access audit and rate-limit summaries, upload-scan summaries,
retrieval-trace summaries, artifacts, runtime
directories, and disabled external-provider/productization switches. It also reports no-secret durable
storage readiness for future metadata database and object-storage backends:
local JSON/SQLite/filesystem storage remains active, while external database
URLs, buckets, and endpoints are surfaced only as configured booleans and are
not connected or exposed. The same status includes a local storage inventory
with per-runtime-tree file counts and byte totals for metadata, jobs, artifacts,
uploads, and FAISS index files without reading or returning file contents. The
Streamlit runtime status panel displays the same storage readiness, inventory,
and local metadata/object storage paths. Successful `/query` and `/query/inspect`
calls append no-secret estimated usage events with duration, character counts,
and rough token estimates. When the provider response exposes token usage,
FluxMind also stores no-secret provider prompt/completion/total token counts in
the same local runtime event; this is usage visibility, not billing. Admin
status/report summarize recent query duration with average/max milliseconds and
metadata-only local advisory alerts controlled by `QUERY_ALERT_MIN_EVENTS` and
`QUERY_ALERT_DURATION_MS`.
Successful `/query`, `/query/inspect`, `/query/report`, and `/query/retrieve`
calls also append metadata-only `retrieval_trace` events with endpoint, answer
mode, context count, source/page completeness counts, citation status when
available, duration, and whether an LLM provider was called. These events do
not store prompts, answers, retrieved text, source paths, owner IDs, or request
IDs, and admin status/report plus metrics summarize them as local-window
retrieval observability. The same admin surfaces derive metadata-only advisory
alerts for empty retrievals, missing source/page metadata, and citation
validation failures using `RETRIEVAL_TRACE_ALERT_*` thresholds.
When optional `QUERY_COST_PROVIDER`,
`QUERY_COST_PROMPT_USD_PER_1M`, and
`QUERY_COST_COMPLETION_USD_PER_1M` are configured, admin status estimates local
USD query cost from provider token counts when available, or from the rough
token estimates otherwise. The pricing status is exposed as no-secret
configuration and explicitly keeps external billing disabled.
Recent `/query` provider failures are summarized with metadata-only local
advisory alerts controlled by `PROVIDER_FAILURE_ALERT_MIN_EVENTS` and
`PROVIDER_FAILURE_ALERT_RATE`; alerts include only counts, codes, and threshold
metadata, not prompt or response content.
`GET /admin/status/report` exports the same no-secret snapshot as a Markdown
operations report for handoff or offline review. `GET /admin/metrics` exports
the same local admin summaries as Prometheus/OpenMetrics-style text for local
scraping. It reports local-window gauges and intentionally omits owner IDs,
request IDs, paths, prompts, answers, uploaded content, filenames, and artifact
contents. Example:

```bash
curl -H "X-API-Key: $FLUXMIND_API_TOKEN" \
  http://127.0.0.1:18502/admin/metrics
```

`GET /admin/retention` previews
upload and artifact files that would match local age-based retention thresholds;
preview remains the default. `POST /admin/retention/delete` can delete those
age-matched local upload/artifact candidates only when
`RETENTION_DELETE_ENABLED` is explicitly true; otherwise it returns a guarded
disabled result. The delete path is authenticated, limit-bounded, excludes the
artifact SQLite metadata files, and records only aggregate `retention_delete`
runtime-event counts. The Streamlit admin panel exposes the same retention
preview with local day/limit controls and only shows the delete button when the
config flag is enabled. `GET /admin/events` lists no-secret
runtime events with local `kind`, `code`, and
`q` filters, and the Streamlit admin panel exposes the same event inspection
controls for provider failures, query usage, code execution outcomes, API
access audit events, upload scan events, and retention delete events.
`python scripts/runtime_manifest.py`, `GET /admin/runtime-manifest`, and the
Streamlit runtime status panel can export a no-secret backup manifest for
runtime state that source deploys intentionally exclude, including file counts,
byte totals, and SHA-256 hashes for known metadata/job/index files without
exporting file contents or `.env` values. Use `--format markdown` or
`GET /admin/runtime-manifest/report` for a handoff-friendly report before
manual backup or storage migration work. A saved manifest can be checked against
a target runtime root with `python scripts/runtime_manifest.py --restore-check <manifest.json> --target-root /opt/fluxmind`,
or remotely through authenticated
`POST /admin/runtime-manifest/restore-check` and
`POST /admin/runtime-manifest/restore-check/report`. The restore check reports
missing or mismatched groups, files, byte counts, and SHA-256 hashes, exits
nonzero on CLI mismatch, and never copies, overwrites, deletes, or restores
files. The Streamlit runtime status panel can upload the saved manifest JSON and
download the same restore dry-run report.
`GET /admin/status`, `GET /admin/status/report`, `GET /admin/metrics`, and the
Streamlit runtime status panel also expose a no-secret local storage-schema
inventory from `src.storage_schema`. It checks schema version, JSON/JSONL shape,
and expected SQLite tables/columns for corpus, chunk, job, artifact, and runtime
event stores without returning row contents, prompts, answers, filenames, owner
IDs, request IDs, source paths, or runtime file contents.
The same check is available as `python scripts/storage_schema.py`, which supports
JSON or Markdown output, `--target-root`, and a nonzero exit code when schema
drift is found.
The same admin status/report/metrics/UI surfaces now include a no-secret
`platform_readiness` summary for production storage migration and distributed
worker acceptance. It reports only booleans, counts, and blocker codes: the
current local runtime has clean schema/inventory and a clean local worker bridge,
but remains blocked on external metadata database, object storage, and
distributed job-store configuration.
Generated mock diagrams can use local SVG templates for generic engineering
diagrams, sliding-mode observers, PMSM control loops, and paper-figure redrafts.
Diagram and execution outputs include no-key metadata such as prompt, style,
template, size, source references, model/provider, and zero-cost estimates.
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

`eval/rag_baseline.json` records 32 offline control-engineering answer cases,
54 retrieval-only source/page cases, eight local Python code-output cases
(four job-backed), 16 PDF equation/table/figure structure cases, expected refs, fixture answers,
recorded answers, key answer terms, topic/lane metadata, domain ontology gates,
and provider-error fixtures. The `python scripts/evaluate_rag.py` command
validates that expected source PDFs/pages contain their
configured snippets, that retrieval-only cases have concrete expected refs, that
fixture and recorded answers cite retrieved context refs such as `[1]`, that
recorded answers meet deterministic key-term coverage thresholds, that local
code-output cases run in a temporary artifact store and produce expected stdout,
plot/text artifacts, runtime metadata, provider-mode coverage, local job-backed
execution coverage, and the reusable `smc_reaching_law` and `pmsm_current_step`
execution templates, and
that representative PDF pages expose equation/table/figure markers for
paper-to-code work, and that provider failures normalize to stable user-facing
codes. The same command evaluates aggregate
`quality_gates` so the eval set fails if it loses answer-mode coverage,
retrieval-question breadth, source-ref breadth, topic/lane coverage,
ontology-group coverage, recorded-answer pass rate, code-output pass rate, or
PDF structure pass rate, or average term coverage. The `/query` API and
Streamlit UI accept answer modes:
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
evidence, including code-output artifact checks.

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
