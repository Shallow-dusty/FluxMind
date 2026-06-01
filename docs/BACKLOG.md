# FluxMind Implementation Backlog

Last updated: 2026-05-30

This backlog turns the platform roadmap into concrete implementation packages.
It is intentionally ordered by dependency, not by excitement.

## WP0: Stabilize Current Production

Status: in progress

- Keep CI green: `python -m pytest` and `python scripts/health_check.py`.
- Keep Trace-Twin health green with:
  `python scripts/health_check.py --ssh-host root@100.100.233.26`.
- Provider-error normalization exists for UI and API responses.
- Request IDs exist for `/query`, Streamlit responses, and logs.
- FastAPI startup only warms an already-present FAISS index and recovers queued
  jobs; it does not synchronously rebuild a missing index before binding the
  API port.
- `scripts/health_check.py` reports local/remote FAISS index size and active
  paper count when available.
- `scripts/health_check.py` verifies that API startup avoids synchronous index
  rebuilds, that remote chunk metadata rows exist when an active corpus is
  present, and that deployed `/corpus/chunks` filters can return a sampled
  chunk while rejecting an impossible query.
- `scripts/health_check.py --url` retries transient HTTP warmup failures such
  as 502/503/504/429 before reporting endpoint failure.
- `scripts/health_check.py --ssh-host ...` includes recent journal error lines
  for the UI/API services.
- `scripts/deploy_sync.py` wraps the production source sync with a dry-run
  default and required excludes for secrets, virtual environments, models, and
  mutable runtime state before allowing `rsync --delete`.

Acceptance:

- Local tests pass.
- Public UI/API return 200.
- Remote systemd services, ports, model config, and disk checks pass.
- Production source sync is reproducible without copying over runtime state.
- Browser translation guard remains covered by tests.

## WP1: RAG Quality Baseline

Status: offline baseline, source/page fixture verification, recorded-answer
scoring, optional live `/query/inspect` regression scoring, no-LLM
`/query/retrieve` retrieval diagnostics, local hybrid retrieval, deterministic
BM25-lite lexical reranking, optional local CrossEncoder reranking,
generated-answer citation-inspection metadata, numbered-citation prompt guards,
optional JSON eval report export, and aggregate eval-set regression gates
implemented; external/service reranking remains planned

- `eval/rag_baseline.json` contains a small control-engineering evaluation set.
- Each case records expected source papers/pages, fixture snippets, recorded
  answers, required answer terms, and minimum answer-term coverage.
- `src.chain.validate_numbered_citations()` validates answer citations like
  `[1]` against retrieved document refs.
- `scripts/evaluate_rag.py` runs the offline baseline without network calls and
  fails recorded answers that miss required refs or key-term coverage thresholds.
- Offline expected refs now verify that the referenced PDF exists, the page is
  parseable, and the configured snippet appears on that page.
- Provider failure fixtures cover timeout, 429/rate-limit, empty output, and
  malformed streaming chunks.
- Answer modes exist in the prompt/API/UI: explanation, derivation,
  implementation, literature review, and code generation.
- `src.chain.hybrid_retrieve()` merges FAISS vector hits with local keyword
  matches from an expanded candidate pool in the indexed docstore, dedupes
  chunks, and keeps the final context bounded by `TOP_K`.
- `src.chain.rerank_documents()` applies a deterministic no-key BM25-lite
  lexical reranker before context formatting, with a first pass that preserves
  source diversity so one high-scoring paper cannot fill the entire context for
  comparison questions.
- `src.chain.learned_rerank_documents()` can apply a no-key local CrossEncoder
  reranker before final context assembly when `RERANKER_MODEL` points to an
  existing local model path. Empty or missing paths fall back to BM25-lite and
  do not trigger runtime model downloads.
- `src.chain.query_with_metadata()` returns the generated answer, retrieved
  context refs, and numbered citation validation; `POST /query/inspect` exposes
  the same metadata for authenticated inspection without changing the
  compatibility shape of `/query`.
- `src.chain.retrieve_with_metadata()` returns retrieved context refs,
  source/page completeness, and the citation guard without calling the LLM;
  `POST /query/retrieve` exposes the same diagnostics for authenticated no-key
  retrieval checks.
- `scripts/evaluate_rag.py --retrieval-url ...` can call a deployed
  `/query/retrieve` endpoint and score expected-source retrieval coverage plus
  source/page completeness without model generation.
- `scripts/evaluate_rag.py --live-url ...` can call a deployed `/query/inspect`
  endpoint, score citation validity, expected-source retrieval coverage, and
  answer-term coverage, while reading the API token from an environment variable
  instead of the repository.
- `scripts/evaluate_rag.py --json-report ...` writes a no-secret
  machine-readable summary of offline/provider/recorded/live retrieval/live
  answer eval results for deployment records.
- `eval/rag_baseline.json` includes aggregate `quality_gates` for minimum case
  count, expected source-ref count, provider fixture count, recorded-answer
  count/pass rate/average term coverage, answer-mode coverage, and optional
  live answer/retrieval pass-rate thresholds.
- The baseline now covers all answer modes: explanation, derivation,
  implementation, literature review, and code generation.
- The generation prompt now tells the model the valid numbered context-ref range
  for each answer so live answers are less likely to invent citations such as
  bibliography numbers.
- Generated answers neutralize out-of-range bracket numbers before validation so
  source-paper bibliography refs cannot masquerade as FluxMind context refs.
- Still planned: external/service reranking that would require a hosted model or
  new account.

Acceptance:

- Evaluation command runs without network where fixtures are available.
- Citation regressions fail locally before deployment.
- Source/page fixture regressions fail locally before deployment.
- Recorded answer coverage regressions fail locally before deployment.
- Generated answer citation refs can be inspected against retrieved source/page
  context without reading logs or raw prompts.
- Retrieval source/page quality can be inspected without calling a model
  provider.
- Live model answers can be scored through the deployed inspect endpoint without
  committing provider tokens.
- Eval breadth and aggregate answer-quality regressions fail through configured
  quality gates, not only per-case checks.
- Eval results can be exported as JSON for CI/deployment evidence without
  copying provider tokens.
- Provider errors surface as structured user-facing messages.

## WP2: Corpus and Storage Layer

Status: local JSON corpus metadata baseline, reusable local corpus profiles,
bibliographic paper enrichment, uploaded-PDF metadata extraction with
first-page author/keyword fallback, SQLite current-state paper/chunk metadata
mirrors, checksum-based uploaded-PDF deduplication, active/deactivated
selection workflow, corpus lifecycle status, local paper metadata filtering, and
admin index freshness plus durable storage readiness checks implemented; durable
local storage inventory and no-secret runtime backup manifest implemented;
durable multi-user database/object storage migration remains planned

- `src/metadata.py` stores local paper metadata in git-ignored
  `metadata/corpus.json`.
- Local corpus/profile JSON writes use same-directory temporary files followed
  by atomic replace, so concurrent local reads should not observe partial JSON.
- Paper metadata is mirrored into `metadata/corpus.sqlite3` as a local
  current-state index for faster inspection and future database migration.
- Indexed chunk metadata is mirrored into `metadata/chunks.sqlite3` with source
  path, page, chunk index, content hash, character count, and preview text.
- Paper records include checksum, title, authors, year, DOI, arXiv ID, venue,
  topic tags, source path, source kind, active flag, indexed status, chunk
  count, and parse/index error fields.
- Uploaded/unmanifested PDFs get best-effort no-key metadata extraction from
  embedded PDF metadata and first-page title, author, DOI/arXiv, year, and
  keyword/index-term text, while curated manifest values still take precedence.
- Uploaded PDFs are deduplicated by SHA-256 against the selectable local corpus
  before writing a new local file or adding duplicate chunks to FAISS.
- `GET /corpus/papers` lists the current local paper metadata without requiring
  manual filesystem inspection, with local filters for query text, active state,
  source kind, and indexed status.
- `GET /corpus/chunks` lists local indexed chunk metadata with optional
  `source_path`, `page`, and `q` filters.
- `GET /corpus/status` exposes corpus lifecycle state as `queued`, `parsing`,
  `indexed`, `failed`, `stale`, or `empty` from index rebuild jobs, paper status,
  and index freshness.
- `GET /admin/status` reports corpus JSON/SQLite storage state without SSH.
- `GET /admin/status` reports whether the current FAISS/chunk metadata source
  set is fresh, stale, missing, or empty relative to the active corpus.
- `GET /admin/status` reports no-secret metadata database and object-storage
  readiness for future production storage backends. Local JSON/SQLite/filesystem
  storage remains active; external database URLs, buckets, and endpoints are
  reported only as configured booleans and are not connected or exposed.
- `GET /admin/status` reports local storage inventory for metadata, jobs,
  artifacts, uploads, and FAISS index files as paths, file counts, byte totals,
  and known-file existence flags without returning file contents.
- `scripts/runtime_manifest.py` exports a no-secret runtime backup manifest for
  the local state trees that source deploys exclude, with file counts, byte
  totals, and SHA-256 hashes for known metadata/job/index files without
  exporting file contents or `.env` values.
- `GET /admin/runtime-manifest`, `GET /admin/runtime-manifest/report`, and the
  Streamlit runtime status panel expose the same no-secret backup manifest
  without requiring SSH access to run the CLI by hand.
- `PUT /corpus/active` persists activation/deactivation choices after validating
  project-relative source paths against the selectable corpus.
- `GET /corpus/profiles`, `POST /corpus/profiles`,
  `GET /corpus/profiles/{profile_id}/status`, and
  `POST /corpus/profiles/{profile_id}/activate` persist and inspect reusable
  no-key local active-paper selections under `metadata/corpus_profiles.json`,
  allowing multiple named corpus selections to coexist before real
  workspace/user storage exists.
- `GET /corpus/profiles/{profile_id}/report` exports the same no-secret profile
  status as Markdown for handoff or offline review.
- `POST /corpus/profiles/{profile_id}/rebuild` activates a saved profile and
  queues the selected-PDF FAISS rebuild through the local async job manager.
- Upload and selected-PDF index rebuild flows update paper metadata.
- Decide storage path: local volume first or object storage plus relational DB.
- Still planned: durable metadata for chunks, corpora, jobs, artifacts, users,
  ownership, and retention in a production database; object storage; richer
  durable deactivation/reactivation workflows across multiple users/corpora.

Acceptance:

- Rebuilding an index is a job with status and logs.
- A paper can be indexed and traced to source path/checksum/chunk count.
- Duplicate uploads reuse an existing selectable/indexed PDF instead of creating
  a second local file or duplicate vector chunks.
- Indexed chunks can be traced to source path, page, sequence, hash, and preview.
- Corpus lifecycle status can be inspected without inferring from job and index
  records by hand.
- Index freshness can be inspected without comparing active paper files and
  chunk SQLite rows by hand.
- Active/deactivated state can be managed without editing runtime files by hand.
- Named local corpus selections can be saved and reactivated without copying or
  editing runtime files by hand.
- Named local corpus selections can be inspected for paper availability,
  active-selection match, index freshness, and rebuild requirement before
  activation.
- Named local corpus selection status can be exported without reading raw
  profile JSON or activating the selection.
- Storage state can be listed without reading raw filesystem layout manually.
- Runtime backup scope can be inspected before manual backup or storage
  migration without copying runtime contents into source control.
- Future database/object-storage readiness can be inspected without committing
  credentials, opening external connections, or migrating runtime data.
- Paper metadata can be searched or filtered locally without fetching and
  scanning the full corpus list by hand.

## WP3: Job System

Status: local JSONL history plus SQLite current-state index, in-process async
queue, scheduled retry/backoff, restart recovery for queued jobs, queue health,
queue-level deadlines, durable worker lease metadata, enabled local durable
worker service foundation, stable execution timeout diagnostics, running Python
cancellation for in-process and explicit durable local workers, and cancellable
index-rebuild checkpoints plus admin worker-lease visibility implemented;
distributed multi-worker queue and full running cancellation for every future
worker type remain planned

- Local JSONL job records exist in `src/jobs.py`.
- Job writes are mirrored into `jobs/jobs.sqlite3` for current-state lookups and
  migration toward durable worker storage.
- `POST /jobs/image/mock` creates a mock image-generation job.
- `POST /jobs/code/python-local` creates a development-only Python execution
  job.
- `POST /jobs/index/rebuild` rebuilds the FAISS index from selected PDFs as a
  persisted job.
- `POST /jobs/async/image/mock`, `POST /jobs/async/code/python-local`,
  `POST /jobs/async/code/octave-local`, and `POST /jobs/async/index/rebuild`
  enqueue those local jobs through an in-process background worker.
- `GET /jobs` lists latest jobs with local `q`, `status`, and `kind` filters.
- `GET /jobs/{job_id}` returns persisted job status.
- `POST /jobs/{job_id}/retry` retries failed/cancelled local jobs with a new
  job ID.
- `POST /jobs/{job_id}/retry-scheduled` queues failed/cancelled local jobs for
  delayed retry with `parent_job_id` and `not_before` metadata.
- Async job and scheduled-retry requests can set `queue_timeout_s`; expired
  queued jobs persist `job_deadline_exceeded` before execution.
- `AsyncJobManager.recover_queued_jobs()` rehydrates queued/scheduled jobs from
  SQLite/JSONL after service restart and returns them to the local worker queue.
- Workers claim queued jobs with `worker_id`, `leased_at`, and
  `lease_expires_at` metadata before provider execution. Expired queued leases
  can be reclaimed, and the current in-process worker uses this same durable
  claim path.
- `LocalDurableJobWorker`, `scripts/run_job_worker.py`, and
  `deploy/systemd/fluxmind-worker.service` provide an enabled no-key local
  worker-service foundation that can claim and run due queued jobs from durable
  state outside the API/Streamlit process.
- The explicit durable worker polls durable job state while a provider is
  running; if another process marks that job `cancelled`, local Python/Octave
  providers receive a cancellation event and terminate the child process.
- `GET /admin/status` exposes `queue_health` with queued, due, scheduled,
  expired, running, leased queued, expired lease, running lease, and oldest
  queued timestamps.
- `GET /admin/status` exposes `worker_leases` with no-secret worker IDs,
  active/expired lease counts, and latest leased job summaries.
- `POST /jobs/{job_id}/cancel` records cancellation for queued/running job
  states. Running local Python jobs observe cancellation; index rebuild jobs
  check cancellation during PDF loading, splitting, and before committing rebuilt
  index state.
- Streamlit sidebar can trigger selected-PDF index rebuild jobs, mock image
  jobs, local Python jobs, and display filtered latest job status.
- Streamlit recent-job panel can cancel queued/running jobs and retry
  failed/cancelled jobs immediately or after a local backoff delay.
- Local execution timeouts persist as `execution_timeout` instead of generic
  execution failures.
- Job records preserve no-secret transition logs for queued, running, terminal,
  and cancelled states.
- Still planned: distributed database-backed worker beyond the local SQLite
  recovery/lease/service bridge and cancellation for future external workers.

Acceptance:

- Job records preserve request, result, artifacts, errors, attempts, request
  IDs, and transition logs.
- Failed code-execution jobs preserve stderr/error details.
- API and Streamlit can show running/succeeded/failed states through job status.
- Queued local job endpoints return without blocking request handlers.
- Queued delayed retries can recover after API service restart.
- Remaining: production-grade long-running work still needs a distributed
  worker/storage backend beyond the local SQLite worker service, plus
  cancellation propagation for future non-local providers.
- Worker/lease activity can be inspected without SSH or raw SQLite reads.

## WP4: Image and Diagram Generation

Status: provider-neutral plumbing, no-key mock provider, local SVG engineering
diagram templates, local artifact export, artifact metadata with SQLite
current-state mirror, and RAG artifact references implemented; real
image-provider activation remains disabled until a key/account is configured

- `MockImageGenerationProvider` implements the `ImageGenerationProvider`
  contract with deterministic local SVG output.
- Local SVG templates cover generic engineering diagrams,
  sliding-mode-observer blocks, PMSM control loops, and paper-figure redraft
  scaffolds without external image providers.
- `GET /artifacts` lists generated local artifacts from persisted jobs.
- `GET /artifacts/{artifact_id}` exports local file artifacts by stable ID.
- `GET /artifacts` supports local `q`, `kind`, and `job_kind` filters for
  artifact metadata inspection.
- Artifact metadata is mirrored into `artifacts/artifacts.sqlite3` as a local
  current-state index for inspection and future durable storage migration.
- `GET /admin/status` reports local artifact integrity counts by checking
  persisted byte-count and SHA-256 metadata against current files.
- Streamlit sidebar includes a local artifact gallery with stable IDs,
  provider-neutral metadata, local filters, and download buttons.
- Mock diagrams and execution artifacts store byte counts and SHA-256 checksums
  in provider-neutral metadata, with prompt/style/template/source-reference/
  model/cost fields layered on diagram artifacts without external keys.
- RAG prompts include recent generated artifacts as stable `[Artifact:<id>]`
  references so answers can point to local diagrams, plots, or files.
- Keep generated images as artifacts rather than inline chat-only blobs.

Acceptance:

- A request can generate an artifact record with a stable URI.
- Generated mock diagrams produce persisted artifact URIs.
- Generated mock diagrams can use engineering-specific templates instead of only
  a generic placeholder.
- Generated diagrams and execution artifacts can be listed and exported.
- Generated local artifacts can be downloaded from Streamlit.
- Generated local artifact metadata can be inspected without scanning raw job
  history manually.
- Generated local artifact metadata includes stable byte counts and SHA-256
  checksums for later durable storage migration.
- Local artifact integrity can be inspected without reading or exporting raw
  artifact contents by hand.
- Provider can be swapped without changing the UI flow.

## WP5: Code Execution

Status: local request/result plumbing, Python execution, Octave-compatible
execution interface, file/plot artifact capture, workdir path containment,
input file size/count limits, no-secret execution environment/policy metadata,
Unix child-process memory/CPU limit metadata/enforcement, and Docker sandbox
readiness reporting plus Streamlit control-engineering execution templates
implemented; real hosted execution, enabled Docker sandbox execution, and MATLAB
activation remain disabled until infrastructure and license/account decisions
are made

- `LocalPythonExecutionProvider` implements the `CodeExecutionProvider`
  contract for development-only Python snippets.
- `LocalOctaveExecutionProvider` implements the same contract for GNU
  Octave-compatible scripts when a local `octave` binary is installed; when it
  is absent, jobs fail with a structured runtime-unavailable diagnostic.
- Python and Octave-compatible snippets can capture stdout, stderr, exit code,
  generated text/file artifacts, and generated image files as plot artifacts.
- Local execution providers reject absolute or escaping input paths/entrypoints
  and skip symlink or out-of-workdir artifact collection.
- Local execution providers reject excessive input file count, per-file bytes,
  and total input bytes before materializing files into the temporary workdir.
- Local execution results persist no-secret reproducibility metadata for the
  language, entrypoint, input file counts/bytes, provider runtime, local Python
  or Octave runtime availability/details, temporary workdir isolation, and
  current network-policy enforcement state.
- Local execution results persist timeout, CPU-time, and memory metadata, and
  Unix child processes receive address-space and CPU-time limits where supported.
- `CODE_EXECUTION_BACKEND` and `DOCKER_EXECUTION_IMAGE` define a future no-key
  Docker sandbox backend; admin status reports whether that backend is
  configured and whether Docker is accessible by the runtime user without
  running a container.
- Timed-out local executions return a stable `execution_timeout` job error code.
- `POST /jobs/code/octave-local` and `POST /jobs/async/code/octave-local`
  expose immediate and queued no-key Octave-compatible job flows.
- Streamlit includes a local Octave-compatible job panel.
- Streamlit includes editable no-key Python and Octave-compatible templates for
  SMC reaching-law and PMSM current-response examples that produce local
  artifacts through the existing providers.
- Job records persist execution artifacts alongside the execution result.
- Still planned: actually run code in an isolated service/container with
  filesystem and network policy plus stronger production-grade resource limits.

Acceptance:

- Current development provider runs code in a child Python process and temporary
  workdir with path-containment checks, but it is not a production sandbox.
- Octave-compatible requests have stable API/UI/job behavior without enabling
  real MATLAB licensing.
- Local control-engineering examples can be launched without writing a blank
  script while still running through the same development providers.
- Execution results are reproducible from stored input files and environment
  metadata.
- Docker/container execution readiness can be inspected without granting Docker
  access or running user code.
- A failed local execution returns structured diagnostics without breaking the
  app.

## WP6: Product Shell

Status: local/admin status foundation, reusable local corpus profiles,
no-secret Markdown status-report export,
Markdown query-report export, provider-failure event history, estimated
query-usage history, local storage-readiness dashboard, and local storage
inventory dashboard plus no-secret runtime backup manifest implemented; keep
public identity, API-key lifecycle, quotas, and billing disabled until decisions
are made

- Decide when to replace Streamlit with a real frontend.
- Add users, private corpora, API keys, quotas, and share/export flows.
- Local corpus profiles let multiple named paper selections coexist without
  introducing accounts, permissions, or a public share model.
- `GET /admin/status` exposes no-secret local runtime status for job counts,
  failed jobs, corpus counts, artifact counts/bytes, runtime directory
  existence/writability/bytes, public model names, and disabled external
  provider/productization switches.
- Streamlit includes a local runtime status panel for common operational checks.
- The Streamlit runtime status panel displays durable storage readiness and
  local metadata/object storage paths from `/admin/status` without exposing
  external storage credentials.
- Admin status and the Streamlit runtime status panel display a no-secret local
  storage inventory with file counts and byte totals for metadata, jobs,
  artifacts, uploads, and FAISS index files.
- `GET /admin/runtime-manifest`, `GET /admin/runtime-manifest/report`, and the
  Streamlit runtime status panel expose a no-secret runtime backup manifest for
  the state trees that source deploys exclude.
- `GET /admin/status/report` and the Streamlit status panel can export the same
  no-secret status snapshot as a Markdown operations report.
- `GET /admin/retention` returns a no-delete preview of upload/artifact files
  matching local age-based retention thresholds.
- The Streamlit status panel exposes the same no-delete retention preview with
  local upload/artifact day thresholds and a candidate limit.
- `GET /admin/events` lists no-secret runtime events with local `kind`, `code`,
  and `q` filters; the Streamlit status panel exposes the same event viewer.
- `POST /query/report` exports an answer, citation validation, and retrieved
  context refs as a Markdown research report.
- `/query` provider failures are appended to a no-secret local JSONL event log
  under `metadata/` and summarized by `GET /admin/status`.
- Successful `/query`, `/query/inspect`, and `/query/report` calls append
  no-secret estimated usage events with character counts and rough token
  estimates. When provider responses include token usage, the same local events
  also store provider prompt/completion/total token counts and admin status
  aggregates them separately from estimates.
- Admin status and the Streamlit status panel expose optional no-secret
  provider/model pricing configuration through `QUERY_COST_PROVIDER`,
  `QUERY_COST_PROMPT_USD_PER_1M`, and
  `QUERY_COST_COMPLETION_USD_PER_1M`. When rates are configured, FluxMind
  estimates USD query cost from provider token counts when available and rough
  estimated tokens otherwise; external billing remains disabled.
- Still planned: production durable storage dashboards beyond the local
  inventory/readiness view, billing attribution, and user/workspace admin once
  identity exists.

Acceptance:

- Multiple corpora can coexist without leaking documents or generated artifacts.
- User-facing workflows are not tied to local server filesystem assumptions.
- Operational state is inspectable without SSH for common local runtime
  questions.
- Durable storage readiness is visible in the UI without activating external
  storage accounts.
- Local storage inventory is visible in the UI without reading file contents or
  activating external storage accounts.
- Local retention candidates can be previewed without deleting files or reading
  raw runtime directories by hand.
