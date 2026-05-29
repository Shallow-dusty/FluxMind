# FluxMind Platform Audit and Roadmap

Last updated: 2026-05-30

Workspace index: `11.FluxMind`

## Current State

FluxMind is currently a focused RAG research copilot for sliding mode control
and flux linkage estimation. It has:

- Streamlit Web UI with Chinese/English labels.
- FastAPI `/query` endpoint for external agent/plugin use.
- Local FAISS vector store and sentence-transformers embeddings.
- Curated seed PDF library plus upload/index flow.
- Trace-Twin deployment through systemd services and Cloudflare Tunnel.

This is usable as a demo and personal research assistant, but it is not yet a
durable multi-user platform. The main gap is not the RAG prototype; it is the
missing platform layer around identity, jobs, storage, observability, safety,
and isolated execution.

Current implementation boundary: real external provider activation is disabled
until keys, accounts, licenses, or sandbox infrastructure are configured.
Feature development should still proceed behind provider-neutral interfaces,
local mocks, fixtures, and explicit runtime flags. Production remains limited to
RAG Q&A, corpus selection, PDF upload/indexing, Streamlit UI, and the
token-protected FastAPI `/query` endpoint until those activation decisions are
made.

Current no-key feature work now includes local provider implementations:
`MockImageGenerationProvider` writes deterministic SVG artifacts, and
`LocalPythonExecutionProvider` exercises the code-execution contract for
development. These are wired through immediate local job endpoints, async
in-process job endpoints, and the Streamlit local job panel, but not into any
real external provider.

## Workspace Reference Migration

FluxMind has been moved from the temporary AI-Prism index `80` to the formal
active project path `11.FluxMind`.

Updated surfaces:

- Active project directory: `/home/shallow/04.AI-Prism/11.FluxMind`
- Pre-formal archive snapshot:
  `/home/shallow/04.AI-Prism/90.Archive/11-FluxMind-PreFormal`
- AI-Prism workspace index: `/home/shallow/04.AI-Prism/CLAUDE.md`
- Project docs and handover docs in this repository.
- Memory update note:
  `/home/shallow/.codex/memories/extensions/ad_hoc/notes/20260529T024105Z-fluxmind-formal-index.md`

Known remaining external references are in read-only home config during this
session and must be refreshed from a writable shell. A dry-run/update helper is
available at `scripts/update_local_references.py`.

- `/home/shallow/.codex/config.toml`
- `/home/shallow/.claude.json`

Historical rollout summaries and memory registry entries still contain the old
`80` path as evidence of prior work; those are append-only history, not current
state.

## Fix Applied: Browser Translation and Streaming

Observed risk: the UI streamed `reasoning_content` and final answers through
`st.write_stream(query_stream(...))`. Streamlit's documented behavior is to
iterate the provided stream and write chunks into the app with a typewriter
effect. That is convenient, but it means the frontend repeatedly updates text
nodes while the answer is still being rendered.

Root cause: Chrome/Google Translate and similar browser translation layers can
mutate text nodes under React/Streamlit while the app framework is reconciling
updates. The known failure mode is a DOM exception such as `removeChild` /
`insertBefore` on a node that has been moved or wrapped by translation.

Current mitigation:

- `app.py` installs a translation guard with `translate="no"` and
  `notranslate` on the app and chat nodes.
- Streaming now uses a stable `st.empty()` markdown placeholder via
  `render_streaming_response()` instead of the `st.write_stream` black box.
- The RAG stream contract in `src/chain.py::query_stream` remains unchanged.
- `src/capabilities.py` defines provider contracts and `src/providers.py`
  contains local no-key providers without coupling those features to the UI.
- Regression coverage now includes stream formatting, translation-guard static
  checks, ingestion filename safety, and capability dataclass contracts.
- `api.py` initializes the vector store in FastAPI lifespan instead of at
  module import, so tests and future OpenAPI/worker preload paths remain
  side-effect free.

Reference context:

- Streamlit `st.write_stream`: https://docs.streamlit.io/develop/api-reference/write-magic/st.write_stream
- React/Google Translate DOM mutation issue: https://github.com/facebook/react/issues/11538

## Current Architecture Review

### Strengths

- Small codebase with clear boundaries: UI (`app.py`), API (`api.py`), RAG
  chain (`src/chain.py`), ingestion (`src/ingestion.py`), config
  (`src/config.py`), embeddings (`src/embeddings.py`).
- Local embeddings reduce runtime dependency on remote embedding APIs.
- The API token boundary is simple and appropriate for the current public
  endpoint.
- Deployment isolation is explicit: independent systemd services and ports,
  no Docker restart, no Trace-Twin bot-stack coupling.
- Seed paper selection gives the user control over the active corpus instead
  of blindly indexing every PDF.

### Risks

- Streamlit is fast for demos, but it is weak as a long-term platform shell:
  authentication, durable user sessions, background jobs, and fine-grained UI
  state will get awkward.
- FAISS local storage is acceptable for one machine but not enough for
  multi-user corpus management, metadata search, or horizontal scaling.
- `/query` is synchronous and can block on retrieval plus LLM latency.
- Uploaded PDFs are stored locally without per-user ownership, quotas, malware
  scanning, deduplication, or retention policy.
- The initial pytest suite, GitHub Actions CI gate, and local/remote health
  checker now exist. They still do not cover query-quality evaluation or
  citation correctness.
- Citation quality depends on raw chunk metadata. There is no source
  normalization, DOI/arXiv metadata enrichment, reranking, or citation verifier.
- LLM/provider errors are normalized for the UI and API, but provider-specific
  retry policy and richer error taxonomy are still basic.
- Request IDs are logged for UI/API requests, but latency, retrieval hits,
  token usage, and provider failures are not persisted.
- No execution sandbox. Any future Python/MATLAB compiler feature must not run
  arbitrary code in the main web/API process. The current local Python provider
  is a development contract exercise only.

## Target Product Direction

The credible platform identity is:

> A control-engineering research workspace that combines trusted paper-grounded
> answers, executable modeling snippets, reproducible notebooks, and generated
> visual artifacts.

That is stronger than "chat with PDFs" because the domain expects equations,
simulation code, plots, and traceable engineering decisions.

## Platform Architecture Target

Recommended medium-term split:

```text
Web App
  - project/corpus management
  - chat and notebook UI
  - artifact gallery
  - job status

API Gateway
  - auth, quotas, request IDs
  - OpenAPI contracts
  - sync health and async job submission

RAG Service
  - document parsing
  - embeddings/vector search
  - reranking and citation verification
  - LLM answer synthesis

Artifact Service
  - image generation jobs
  - plot/image storage
  - prompt/version metadata

Execution Service
  - isolated Python/Octave/MATLAB-compatible runs
  - resource limits
  - stdout/stderr/files/plots capture

Storage
  - object storage for PDFs and generated artifacts
  - relational metadata store
  - vector index
  - logs/traces
```

## Roadmap

The roadmap below is directional. For phases that eventually require external
provider keys, paid accounts, MATLAB licensing, or sandbox infrastructure, build
the local contracts, storage, request/response flow, tests, and disabled
provider switches first. Only real external activation is deferred.

### Phase 0: Stabilize the Current App

- Expand smoke tests for `api.py`, active paper selection, and deployment
  refresh behavior.
- Extend `scripts/health_check.py` further so it can report active paper count,
  index size, and recent journal errors in addition to its current HTTP,
  systemd, port, model, and disk checks.
- Normalize provider errors into user-readable messages.
- Add request IDs and structured logs.
- Keep Streamlit for now, but stop adding platform-only complexity to it.

### Phase 1: Real Platform Baseline

- Introduce a durable metadata database for users, corpora, papers, chunks,
  jobs, artifacts, and API keys.
- Move file storage to object storage or an explicit volume layout with
  checksums and ownership metadata.
- Add background jobs for PDF parsing, indexing, image generation, and code
  execution.
- Add corpus-level status: queued, parsing, indexed, failed, stale.
- Add an evaluation set: standard SMC/flux questions with expected citation
  behavior and regression scoring.

Current progress: a local JSONL job store, immediate no-key job endpoints, and
in-process async no-key job endpoints exist for mock image generation,
development-only Python execution, and selected-PDF index rebuilds.
List/status/retry/cancel endpoints exist. The Streamlit sidebar can trigger
queued no-key jobs and display latest job state. This proves the
UI/API/status/artifact shape but is not yet a durable multi-worker queue or
database.

### Phase 2: Better RAG

- Enrich PDFs with title, authors, DOI/arXiv ID, year, venue, and topic tags.
- Add hybrid retrieval: vector search plus keyword/BM25.
- Add reranking before final context assembly.
- Add citation verification: every cited source should map back to a retrieved
  chunk and page.
- Add answer modes: explanation, derivation, implementation, literature review,
  and code generation.

### Phase 3: Image and Diagram Generation Interface

Status: planned. Build provider-neutral plumbing and a no-key mock/local
provider first. Real image-provider activation remains disabled until a
key/account and artifact storage policy are configured.

Use an internal provider interface before binding the app to one vendor:

```python
class ImageGenerationProvider:
    def generate(self, prompt, *, style, size, references=None) -> ImageArtifact:
        ...
```

Initial use cases should be engineering-specific:

- Control block diagrams.
- Observer/controller architecture diagrams.
- Paper figure redrafts.
- Simulation result plots generated from code outputs.

OpenAI's current image docs separate simple Image API generation/edits from
Responses API image tools for conversational, iterative image workflows. That
maps well to FluxMind: use a simple image endpoint for one-shot diagrams, and
reserve conversational editing for later.

Reference: https://platform.openai.com/docs/guides/image-generation

### Phase 4: Code Execution

Status: planned. Build request/result plumbing, job state, and the sandbox
boundary first. Real hosted execution remains disabled until infrastructure is
configured. Real MATLAB support additionally requires license/account decisions;
GNU Octave or Python-only execution should be considered before MATLAB.

Do not run user code in the Streamlit/API process. Add an execution provider:

```python
class CodeExecutionProvider:
    def run(self, language, files, entrypoint, *, timeout_s, memory_mb) -> ExecutionResult:
        ...
```

Recommended path:

- Start with Python execution for numerical examples and plotting.
- Add GNU Octave as a MATLAB-compatible stepping stone.
- Add real MATLAB only if licensing, server resources, and isolation are
  deliberately solved.
- Capture stdout, stderr, exit code, generated files, and plots as artifacts.
- Enforce quotas, timeouts, network policy, and per-session workspaces.

Cloudflare Sandbox SDK is relevant for hosted isolated execution because it
provides container-backed sandbox instances, command execution, files, and
stdout/stderr/exit-code capture from Workers. The current Trace-Twin server can
also host containerized execution, but that should be a separate service with
hard resource limits, not a subprocess from the web app.

Reference: https://developers.cloudflare.com/sandbox/

### Phase 5: Productization

Status: planned. Build local/admin foundations first. Public identity,
API-key lifecycle, quotas, and billing remain disabled until those product and
operational decisions are made.

- Replace or wrap Streamlit with a real frontend once user/workspace concepts
  outgrow the demo UI.
- Add accounts, private corpora, exportable reports, artifact galleries, and
  share links.
- Add billing/cost accounting only after usage patterns are visible.
- Add admin views for queue health, provider failures, token spend, and corpus
  storage.

## Near-Term Implementation Plan

1. Land the current numbering/docs/streaming fix.
2. Expand tests and the local smoke command into a deploy gate.
3. Use `docs/ARCHITECTURE.md` and `docs/BACKLOG.md` as the implementation
   source of truth for platformization work.
4. Extend `src/capabilities.py` into concrete no-key providers, fixtures, and
   disabled provider switches.
5. Replace the local in-process job queue with durable worker/storage
   foundations, then add true running cancellation, retry backoff, and richer
   UI retry/cancel controls before enabling real external image generation or
   code execution providers.

## Open Decisions

- Keep Streamlit as the production UI for the next iteration, or treat it as a
  demo shell and begin a frontend/API split.
- Choose storage: local volume first, or object storage plus relational
  metadata now.
- Choose execution backend: local Docker service, Cloudflare Sandbox, or a
  dedicated VM with container isolation.
- Decide whether "MATLAB compiler" means true MATLAB, Octave-compatible code,
  or Python equivalents for control systems.
