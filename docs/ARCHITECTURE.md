# FluxMind Architecture

Last updated: 2026-05-30

## Current Runtime

```text
Browser
  -> Streamlit UI (:18501)
       -> src.chain.query_stream()
            -> FAISS local index
            -> OpenAI-compatible LLM endpoint

External agent / plugin
  -> FastAPI (:18502)
       -> src.chain.query()
            -> FAISS local index
            -> OpenAI-compatible LLM endpoint
```

The production instance runs on Trace-Twin under `/opt/fluxmind` with separate
systemd services for UI and API. Cloudflare Tunnel exposes:

- `https://smy.hyper-dusty.cloud/` -> Streamlit UI
- `https://api-smy.hyper-dusty.cloud/health` and `/query` -> FastAPI

## Module Boundaries

- `app.py`: Streamlit UI, bilingual labels, PDF selection/upload controls, chat
  rendering, browser-translation guard, and local no-key job panel.
- `api.py`: FastAPI request contract, token verification, lifecycle startup,
  and best-effort warming of an already-present FAISS index.
- `src/chain.py`: RAG prompt, retrieval, non-streaming answer generation, and
  reasoning-aware streaming, answer modes, generated-answer inspection
  metadata, and numbered citation validation.
- `src/ingestion.py`: PDF discovery, upload name safety, PyMuPDF extraction,
  best-effort uploaded-PDF bibliographic extraction, chunking, FAISS
  persistence, active paper selection, and paper metadata refresh.
- `src/metadata.py`: local JSON corpus metadata registry for selectable papers,
  bibliographic enrichment, checksums, active/indexed state, chunk counts,
  parse/index error fields, and SQLite current-state mirrors for paper and chunk
  metadata.
- `src/embeddings.py`: local sentence-transformers embedding model factory.
- `src/capabilities.py`: provider-neutral future contracts for image
  generation and isolated Python/Octave/MATLAB-compatible execution.
- `src/providers.py`: no-key local providers for artifact storage, mock SVG
  diagram generation, development-only Python execution, and GNU
  Octave-compatible local execution with generated file/plot capture.
- `src/artifacts.py`: local artifact registry, SQLite current-state artifact
  metadata mirror, and safe file export helpers for artifacts referenced by
  persisted jobs and RAG answer context.
- `src/admin.py`: no-secret local admin/runtime status for queue, corpus,
  artifact, provider-failure history, directory, and disabled-provider/
  productization switches.
- `src/jobs.py`: local JSONL job records, immediate runner, and in-process
  background queue for mock image generation, development-only Python
  execution, and selected-PDF index rebuilds.
- `src/evaluation.py`: offline RAG fixture evaluation, recorded answer
  key-term coverage, provider-error fixture checks, and citation validation
  helpers.
- `eval/rag_baseline.json`: no-network baseline cases with expected source/page
  references and fixture answers.
- `scripts/evaluate_rag.py`: CLI gate for the offline RAG baseline.
- `scripts/health_check.py`: local, HTTP, and SSH runtime checks.
- `scripts/update_local_references.py`: local config path migration helper for
  the retired temporary `80` index.
- `.github/workflows/ci.yml`: CI gate for tests and local health checks.

## Current Constraints

- Streamlit is acceptable for the present demo and personal assistant phase,
  but should not become the long-term platform shell for accounts, teams, jobs,
  artifact management, or complex workspace state.
- FAISS local storage is simple and fast, but it is not yet a multi-user vector
  platform. Metadata, ownership, and indexing jobs should move into explicit
  storage layers before public platform use.
- Code execution and image generation must remain provider-backed services.
  They should not run inside the UI process or the synchronous `/query` path.
- Real external provider activation is intentionally disabled until keys,
  accounts, licenses, or sandbox infrastructure are configured. Feature
  development should still proceed behind provider-neutral interfaces, local
  mocks, fixtures, and explicit runtime flags. This includes image generation,
  hosted code execution, real MATLAB integration, multi-user accounts, quotas,
  and billing.
- `LocalPythonExecutionProvider` is a development provider only. It proves the
  execution request/result contract, but production execution still needs a
  dedicated isolated service.

## Next Architecture Step

The next implementation step should split long-running work into explicit jobs:

```text
API request
  -> create job row
  -> worker executes parse/index/generate/run
  -> artifacts stored by URI
  -> UI polls or subscribes to job status
```

The first job boundary now exists as a local runner plus an in-process
background queue:

```text
API request
  -> create JSONL job history record and SQLite current-state row
  -> enqueue or run local no-key provider
  -> persist result/artifact/error
  -> expose status through GET /jobs and GET /jobs/{job_id}
```

The local runner also supports retrying failed/cancelled jobs, scheduling a
failed/cancelled job retry after a bounded local backoff delay, and marking
queued/running records as cancelled. Scheduled retries preserve
`parent_job_id` and `not_before` metadata. Async jobs and scheduled retries can
also carry `deadline_at`, derived from `queue_timeout_s`; the worker fails jobs
that expire before execution with `job_deadline_exceeded` instead of starting
the provider. `GET /jobs` and the Streamlit recent-job panel support local `q`,
`status`, and `kind` filters, so operators can find job records without reading
JSONL/SQLite state directly. Job records include no-secret transition logs for queued, running,
terminal, and cancelled states. Job transitions are retained in append-only JSONL
and mirrored into `jobs/jobs.sqlite3` as a local current-state index. On API startup,
`AsyncJobManager.recover_queued_jobs()` rehydrates
queued/scheduled jobs from durable state and returns them to the process-local
worker queue. This makes no-key delayed retries survive service restarts, while
remaining short of a distributed multi-worker platform queue. Before execution,
the in-process worker claims a queued job through the same durable store using
`worker_id`, `leased_at`, and `lease_expires_at`; expired queued leases are
claimable by another worker. This creates a local lease contract for the future
worker/storage migration without changing the current process-local execution
model. Admin status summarizes the same lease activity as `worker_leases`,
including no-secret worker IDs, active/expired leases, and the latest leased job
summaries. `LocalDurableJobWorker`, `scripts/run_job_worker.py`, and
`deploy/systemd/fluxmind-worker.service` add an explicit local worker-service
foundation that claims due jobs from the durable store and runs the existing
local providers outside the API or UI services. While a durable worker runs a
local provider, it polls the
job store for `cancelled` state and forwards cancellation through the provider
`cancel_event`, so local Python/Octave child processes can be terminated even
outside the API process. Running local Python jobs observe cancellation, and
local execution timeout failures persist as `execution_timeout` instead of
generic execution failures.
FastAPI startup intentionally does not synchronously rebuild a missing FAISS
index. If the index is missing or cannot be warmed, `/health` and job/admin
routes should still bind so operators can inspect state and trigger an explicit
index rebuild job.
Index rebuild jobs now check cancellation during PDF loading, chunk splitting,
and before committing rebuilt index state, so a cancelled local rebuild should
not replace the live FAISS index or publish updated chunk metadata after the
cancel signal is observed. The Streamlit sidebar can trigger selected-PDF index
jobs, mock SVG image jobs, local Python jobs, display recent job status, cancel
queued/running jobs, and retry failed/cancelled jobs immediately or after a local
backoff delay. Real external providers can be attached later without changing the
UI/API workflow.

Implementation work packages are tracked in `docs/BACKLOG.md`.

## RAG Quality Gate

The first RAG quality gate is intentionally offline. `eval/rag_baseline.json`
stores domain questions, answer modes, expected source/page references, fixture
answers, recorded answers, required answer terms, and provider failure fixtures.
`scripts/evaluate_rag.py` validates that expected PDF source files exist, that
configured pages can be parsed, that source snippets appear on those pages, that
fixture and recorded answers only cite retrieved context refs, that recorded
answers meet deterministic key-term coverage thresholds, and that provider
errors normalize to stable user-facing codes. It also evaluates aggregate
`quality_gates` for eval-set breadth, answer-mode coverage, recorded-answer
count/pass rate, average term coverage, and optional live pass-rate thresholds.
By default this avoids live provider calls. When `--retrieval-url` is supplied,
the same script calls the deployed `/query/retrieve` endpoint and scores
retrieval coverage plus source/page completeness without model generation. When
`--live-url` is supplied, it calls `/query/inspect` and scores generated answers
for citation validity, expected-source retrieval coverage, key-term coverage,
and configured live aggregate gates without storing API tokens in the
repository. `--json-report` writes the same
offline/provider/recorded/live-retrieval/live-answer/gate result summary as
no-secret JSON for CI or deployment evidence.

For retrieval-only checks, `src.chain.retrieve_with_metadata()` returns
retrieved context refs, source/page completeness, and the citation guard without
calling the LLM provider. The authenticated `POST /query/retrieve` API exposes
that no-key diagnostic path for deployment and regression checks. For freshly
generated answers, `src.chain.query_with_metadata()` returns the answer,
retrieved context refs, and numbered citation validation. The authenticated
`POST /query/inspect` API exposes that metadata without changing the
compatibility-oriented `/query` response body. This lets operators verify whether
generated citations map to retrieved chunks with source/page metadata before any
hosted evaluation service is introduced. The prompt also tells the model the valid
numbered context-ref range for the current answer, reducing invented numbered
citations that do not map to retrieved chunks. If the provider still emits
out-of-range bracket numbers, FluxMind neutralizes them before validation so
bibliography-style refs from source papers cannot be mistaken for retrieved
context refs.

Retrieval now uses `src.chain.hybrid_retrieve()` for both non-streaming and
streaming answers. It starts with an expanded FAISS vector candidate pool,
supplements it with local keyword matches from the indexed docstore when
available, dedupes chunks, and then applies `src.chain.rerank_documents()`, a
deterministic no-key BM25-lite lexical reranker over chunk text and metadata.
The reranker first preserves source diversity among positive-scoring chunks,
then fills the remaining context by score, before keeping the final context
bounded by `TOP_K`. If `RERANKER_MODEL` points to an existing local model path,
`src.chain.learned_rerank_documents()` lazy-loads a sentence-transformers
CrossEncoder and reranks the merged candidate pool before final context
assembly. Empty or missing reranker paths do not download models at runtime and
fall back to BM25-lite. This is a local retrieval-quality baseline, not a
replacement for broader live answer scoring.

Recent generated artifacts are formatted by
`src.artifacts.format_artifact_references()` and injected into the RAG prompt as
`[Artifact:<id>]` references. The model can cite those IDs when a generated
diagram, plot, or file is relevant, but it is explicitly instructed not to
invent artifact IDs.

## Corpus Metadata

The first corpus storage boundary is `metadata/corpus.json`, managed through
`src.metadata.CorpusMetadataStore`, with a local SQLite current-state mirror in
`metadata/corpus.sqlite3`. Indexed chunk metadata is mirrored into
`metadata/chunks.sqlite3` through `src.metadata.ChunkMetadataStore`. Paper
records include source path, checksum, manifest title fields, source kind, active
flag, indexed status, chunk count, and parse/index error slots.
The local JSON stores write through same-directory temporary files and atomic
replace so concurrent local readers do not observe empty or partial JSON during
metadata refreshes.
`metadata/corpus_profiles.json` stores reusable no-key local corpus profiles:
named active-paper selections that can be listed, updated, and reactivated
without copying PDFs or editing runtime files by hand. Chunk records include
source path, page, chunk sequence, content hash, character count, and preview
text. Uploads are deduplicated by SHA-256 against selectable local PDFs before
writing a new file or adding duplicate chunks. Upload and selected-PDF rebuild
flows update this state. FastAPI exposes paper records through
`GET /corpus/papers`, with optional local filters for query text, active state,
source kind, and indexed status. It exposes chunk records through
`GET /corpus/chunks`, with local `source_path`, `page`, and `q` filters, corpus
lifecycle state through `GET /corpus/status`, and
reusable local selections through `GET /corpus/profiles`,
`POST /corpus/profiles`, `GET /corpus/profiles/{profile_id}/status`, and
`POST /corpus/profiles/{profile_id}/activate`. `POST
/corpus/profiles/{profile_id}/rebuild` activates a saved profile and queues the
same selected-PDF FAISS rebuild job used by the general async index route.
Profile status is read-only: it checks paper availability, active-selection
match, profile-vs-chunk index freshness, and rebuild requirement without
changing the active FAISS selection. `GET
/corpus/profiles/{profile_id}/report` exports that same no-secret profile
status as Markdown for handoff or offline review.
`GET /admin/status` reports JSON/SQLite corpus and chunk storage state. Admin
status also reports local index freshness by comparing the active paper source
set with the distinct source paths represented in chunk metadata. Corpus status
folds index rebuild jobs and freshness into `queued`, `parsing`, `indexed`,
`failed`, `stale`, or `empty`, making stale FAISS/chunk state visible without SSH
or manual SQLite inspection. The same admin status includes no-secret durable
storage readiness for future metadata database and object-storage backends.
Local JSON/SQLite/filesystem storage remains active; external database URLs,
buckets, and endpoints are represented only by configured/available booleans and
reason codes, without opening a connection or exposing secret values.
`PUT /corpus/active` validates project-relative source paths against the
selectable corpus, persists the active/deactivated selection to
`faiss_index/active_papers.json`, refreshes local metadata, and returns
`rebuild_required=true` so clients can decide whether to trigger an index job.

This is still a local development store. It makes corpus state inspectable
without reading the filesystem manually and creates a migration path toward a
database-backed store, but it is not the future multi-user metadata database or
object storage layer.

## Execution Artifacts

`LocalPythonExecutionProvider` runs development-only Python snippets in a
temporary workdir. `LocalOctaveExecutionProvider` runs GNU Octave-compatible
scripts through a local `octave` executable when one is installed, and returns a
structured failure when the binary is absent. Both providers capture
stdout/stderr/exit code, persist timeout/memory metadata, apply Unix child
address-space and CPU-time limits where supported, and persist generated files
under `artifacts/code-runs/` or `artifacts/octave-runs/`. Execution results also
include no-secret reproducibility metadata for language, entrypoint, input file
counts/bytes, provider runtime, local Python or Octave runtime availability/
details, temporary workdir isolation, and the explicit fact that these
development providers do not enforce a network policy. Image files are returned
as `plot` artifacts; text files are returned as `text` artifacts; other small
outputs are returned as `file` artifacts. Request files and entrypoints must stay
inside the per-run workdir, and symlink or out-of-workdir outputs are not exported
as artifacts. This gives the UI/API/job model a concrete artifact shape for
generated plots and files before any hosted sandbox or real MATLAB backend is
activated. Streamlit layers editable no-key execution templates on top of the
same provider path: a Python SMC reaching-law example writes CSV/SVG artifacts,
and an Octave-compatible PMSM current-response example writes CSV output when a
local Octave runtime exists.

FastAPI exposes `GET /artifacts` and `GET /artifacts/{artifact_id}` so generated
mock diagrams, plots, and execution files can be listed and exported without
exposing raw filesystem paths. Export only supports local `file://` artifacts
that resolve under `ARTIFACTS_DIR`. The local artifact registry mirrors current
artifact metadata into `artifacts/artifacts.sqlite3` while still deriving records
from persisted jobs, giving later durable artifact storage a concrete migration
shape. The Streamlit sidebar also reads the local artifact registry and renders
recent artifacts with stable IDs, metadata, local filters, and download buttons.
`GET /artifacts` and the sidebar gallery support local `q`, `kind`, and
`job_kind` filters for narrowing generated diagrams, plots, and files. Local
artifact records include byte counts and SHA-256 checksums, and mock diagram
artifacts layer prompt, style, local SVG template, size, source references,
provider/model, and zero-cost metadata on top. The local templates cover generic
engineering diagrams, sliding-mode observers, PMSM control loops, and
paper-figure redraft scaffolds before any real image provider is activated.
Admin status verifies current local files against those
checksums and byte counts without reading artifact contents into user-facing
responses, giving later real image providers a concrete metadata and integrity
shape without using external keys.

## Admin Status

`GET /admin/status` exposes a token-protected, no-secret operations snapshot for
local development and production checks. It reports job counts by status/kind,
durable queue health, latest failed local jobs, corpus paper counts, artifact
counts/bytes, recent `/query` provider failures, estimated no-secret query
usage, provider token usage when the upstream response exposes it, runtime
directory existence/writability/bytes, public model names, durable storage
readiness, code-execution backend readiness, Docker sandbox accessibility, and
optional no-secret query-cost estimates from configured per-1M-token rates, plus
explicit disabled switches for external providers and identity/quotas/billing.
The Streamlit sidebar renders the same status, including durable storage
readiness, query-cost pricing status, and local metadata/object storage paths,
so common operational questions do not require SSH or raw filesystem inspection.
`GET /admin/status/report` renders the
same snapshot as a Markdown operations report, and the Streamlit runtime panel
exposes the report as a download for handoff or offline review. `GET
/admin/retention` provides a no-delete preview of upload and artifact files that
match local age-based retention thresholds, and the Streamlit admin panel exposes
the same preview with local day/limit controls. This establishes retention
policy shape before any destructive cleanup command exists.
`GET /admin/events` lists no-secret runtime events with local `kind`, `code`, and
`q` filters, and the Streamlit admin panel exposes the same event viewer for
provider-failure and query-usage inspection without reading raw JSONL.

`POST /query/report` reuses `query_with_metadata()` to return a Markdown research
report containing the generated answer, citation validation, and retrieved
context refs. It is an export surface for the current single-user/local workflow,
not a share-link or multi-user report store.
