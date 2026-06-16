# FluxMind Architecture

Last updated: 2026-06-16

For reading order and document ownership, see `docs/README.md`. Current git and
verification state is tracked in `docs/REPO_STATUS.md`.

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
- `api.py`: FastAPI request contract, constant-time token verification,
  metadata-only API access audit and local rate-limit middleware, non-blocking
  retrieval warmup, `/health` process liveness, and `/ready` retrieval readiness.
- `src/chain.py`: RAG prompt, retrieval, non-streaming answer generation, and
  reasoning-aware streaming, answer modes, generated-answer inspection
  metadata, and numbered citation validation.
- `src/ingestion.py`: PDF discovery, upload name safety, PyMuPDF extraction,
  best-effort uploaded-PDF bibliographic extraction, PDF layout structure
  marker extraction, chunking, FAISS persistence, active paper selection, and
  hardened manifest/active-selection runtime-state handling.
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
  artifact, provider-failure history, API access audit/rate-limit summaries,
  retrieval trace summaries, platform-readiness blockers, local metrics export,
  provider-readiness/product-readiness blockers, runtime directory, and
  disabled-provider/productization switches.
- `src/jobs.py`: local JSONL job records, SQLite current-state and
  idempotency-claim mirrors, immediate runner, in-process background queue, and
  explicit durable worker loop for mock image generation, development-only
  Python/Octave execution, selected-PDF index rebuilds, and tolerant JSONL
  fallback recovery.
- `src/evaluation.py`: offline RAG fixture evaluation, recorded answer
  key-term coverage, provider-error fixture checks, and citation validation
  helpers.
- `eval/rag_baseline.json`: no-network baseline cases with expected source/page
  references and fixture answers.
- `scripts/evaluate_rag.py`: CLI gate for the offline RAG baseline.
- `scripts/health_check.py`: local, HTTP, and SSH runtime checks.
- `scripts/storage_schema.py`: no-secret local storage-schema preflight with JSON
  or Markdown output and nonzero exit on drift.
- `scripts/platform_migration_preflight.py`: no-secret production migration
  preflight that separates local migration evidence from external activation
  readiness.
- `scripts/platform_migration_rehearsal.py`: local runtime migration rehearsal
  that stages required runtime state, then verifies restore and schema checks
  without exposing contents.
- `src/api_keys.py` and `scripts/api_key_registry.py`: optional local SQLite API
  key lifecycle registry. It persists token hashes only, returns raw tokens once
  on create, supports list/verify/revoke, and can back FastAPI auth when
  `FLUXMIND_API_KEY_REGISTRY_BACKEND=sqlite`.
- `src/product_registry.py` and `scripts/product_registry.py`: optional local
  SQLite user/workspace/RBAC/quota/usage/billing-attribution ledger. It gives
  product-readiness a no-secret local contract for product identity state, role
  permissions, and usage attribution without connecting to an external identity
  provider or payment processor.
- `src/product_readiness.py` and `scripts/product_readiness.py`: no-secret
  productization readiness check for identity, API-key lifecycle, RBAC, quotas,
  and billing activation. It can verify the local SQLite key registry, product
  registry, local quota guard, and local RBAC guard when enabled and reports
  local foundation checks and blocker codes without exposing token values, owner
  IDs, billing credentials, or provider secrets.
- `src/provider_readiness.py` and `scripts/provider_readiness.py`: no-secret
  external provider activation readiness check for real image providers, hosted
  execution sandboxes, MATLAB backend/licensing, and provider quota/cost guards.
  It reports safe backend names, local foundation checks, and activation blocker
  codes without exporting prompts, content, URLs, credentials, or license data.
- `src/quality_readiness.py` and `scripts/quality_readiness.py`: no-secret
  quality maturity readiness check for the staged self-use, small-group, and
  community targets. It reuses `eval/rag_baseline.json`, optionally merges
  explicit no-secret live eval reports, and reports only counts, booleans, and
  blocker codes.
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
- Code execution and image generation must remain provider-neutral job-backed
  services. They should not run inside the UI process or the synchronous
  `/query` path.
- Real external provider activation is intentionally disabled until keys,
  accounts, licenses, or sandbox infrastructure are configured. Feature
  development should still proceed behind provider-neutral interfaces, local
  mocks, fixtures, and explicit runtime flags. This includes image generation,
  hosted code execution, real MATLAB integration, multi-user accounts, quotas,
  and billing. Local API key lifecycle and the local product ledger can be enabled
  through SQLite registries, but they are not external identity, payment, or
  distributed quota systems. The provider-readiness and product-readiness checks
  make the disabled state explicit as blocker codes; they do not activate
  providers.
- `LocalPythonExecutionProvider` is a development provider only. It proves the
  execution request/result contract; `DockerExecutionProvider` is the local
  isolated backend, while hosted/distributed production execution still needs a
  deliberate runtime decision and live verification.

## Next Architecture Step

The long-running work boundary now exists locally. The production foundation now
has explicit no-secret readiness targets for metadata storage, object storage,
and the distributed job store, but activation still requires choosing and
testing the real external backends. The next architecture step is not another
process-local queue; it is moving the proven job/storage contracts behind
production-grade metadata, object storage, and distributed worker state:

```text
API request
  -> durable metadata database row
  -> distributed worker lease/claim
  -> provider or sandbox executes parse/index/generate/run
  -> artifacts stored by URI in object storage
  -> UI/API polls or subscribes to job status
```

The first job boundary now exists as a local runner plus an in-process
background queue and explicit local durable worker:

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
`status`, `kind`, and `owner_id` filters, so operators can find job records
without reading JSONL/SQLite state directly. Job creation requests can include
`idempotency_key`; for the same job kind and key, immediate and async routes
return the existing persisted job instead of creating or executing a duplicate.
The durable SQLite `job_idempotency` claim table backs this lookup, and omitted
keys still create new jobs. Query and job-creation API requests can carry
optional `owner_id` and `owner_label` metadata; omitted values normalize to the
local no-key owner `local-user` / `Local user`. The owner fields are persisted
on durable job records, job transition logs, query runtime events, artifact
records, and admin summaries, but they are not authentication, tenant
isolation, quota, or billing controls. Job records include no-secret transition
logs for queued, running, terminal, and cancelled states. Job transitions are
retained in append-only JSONL and mirrored into `jobs/jobs.sqlite3` as a local
current-state index. Queued jobs can also carry `max_attempts` and
`retry_backoff_s`; failed
attempts are returned to `queued` until the cap is reached, then the same job is
marked `dead_lettered` with `dead_lettered_at`. This is a local bounded retry
and dead-letter contract, not a distributed dead-letter queue. On API startup,
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
index or block the API socket while warming an existing index. `/health`
reports process liveness after bind, while `/ready` reports whether retrieval
warmup has loaded the existing FAISS index. If the index is missing or cannot be
warmed, `/health` and job/admin routes should still bind so operators can
inspect state and trigger an explicit index rebuild job. The in-process
embedding model and FAISS store are cached, and the FAISS cache reloads when the
persisted index files change.
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
answers, recorded answers, required answer terms, local code-output cases, PDF
equation/table/figure/algorithm structure cases, and provider failure fixtures.
`scripts/evaluate_rag.py` validates that expected PDF source files exist, that
configured pages can be parsed, that source snippets appear on those pages, that
fixture and recorded answers only cite retrieved context refs, that recorded
answers meet deterministic key-term coverage thresholds, that local code-output
cases execute through direct no-key provider mode and local job-backed mode, and
produce expected stdout/artifacts/runtime metadata, including reusable
execution-template coverage, that representative PDF pages expose equation/table/figure/algorithm
markers, and that provider errors normalize to stable user-facing codes. It also
evaluates aggregate
`quality_gates` for eval-set breadth, answer-mode coverage, recorded-answer
count/pass rate, code-output count/language/template/execution-mode/pass rate,
PDF structure count/kind/pass rate, average term coverage, and optional live
pass-rate thresholds.
By default this avoids live provider calls. When `--retrieval-url` is supplied,
the same script calls the deployed `/query/retrieve` endpoint and scores
retrieval coverage plus source/page completeness without model generation. When
`--live-url` is supplied, it calls `/query/inspect` and scores generated answers
for citation validity, expected-source retrieval coverage, key-term coverage,
and configured live aggregate gates without storing API tokens in the
repository. `--json-report` writes the same
offline/retrieval-only/code-output/provider/recorded/live-retrieval/live-answer/gate
result summary as no-secret JSON for CI or deployment evidence.
`scripts/quality_readiness.py` is the small wrapper for reading that eval
baseline as staged readiness. By default it proves the local source-quality
foundation; when supplied with `--live-report`, it merges the no-secret live
retrieval/live answer counts from an eval JSON report before deciding whether
the small-group or community targets are met. `--require-target community`
stays nonzero until the community bar has enough curated papers, answer cases,
retrieval questions, PDF structure cases, and live evidence.

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

Successful `/query`, `/query/inspect`, `/query/report`, and `/query/retrieve`
calls also emit metadata-only `retrieval_trace` runtime events. The events
record endpoint, answer mode, context count, source/page completeness counts,
citation status when available, duration, and whether the provider was called;
they intentionally omit prompts, answers, retrieved text, source paths, owner
IDs, and request IDs. Admin status/report, Streamlit, and the local metrics
export summarize these events as a shallow retrieval observability layer, not a
production tracing pipeline.

The same admin surfaces derive metadata-only advisory alerts from recent
retrieval traces for empty retrievals, missing source/page metadata, and failed
citation validation. Alert thresholds are local configuration only
(`RETRIEVAL_TRACE_ALERT_*`); alert payloads carry counts, rates, threshold
values, severity, and codes rather than prompts, answers, retrieved text, source
paths, owner IDs, request IDs, or alert-routing state.

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
The curated library manifest and active-paper selection loaders treat malformed
runtime JSON as recoverable local state, constrain selections to project-local
selectable PDFs, drop stale/duplicate entries, and write active-paper selection
through atomic replace. The main corpus registry remains an operator-visible
state file; it is not silently overwritten when corrupted.
`metadata/corpus_profiles.json` stores reusable no-key local corpus profiles:
named active-paper selections that can be listed, updated, and reactivated
without copying PDFs or editing runtime files by hand. Chunk records include
source path, page, chunk sequence, content hash, character count, and preview
text. Uploads are deduplicated by SHA-256 against selectable local PDFs before
writing a new file or adding duplicate chunks. Uploads also pass through a
local pre-write scan before persistence: the scan validates PDF magic and
PyMuPDF parseability, rejects encrypted PDFs by default, blocks common
active-content markers, caps page count, and records only metadata-only
`upload_scan` events. Upload and selected-PDF rebuild flows update this state.
FastAPI exposes paper records through
`GET /corpus/papers`, with optional local filters for query text, active state,
source kind, and indexed status. It exposes chunk records through
`GET /corpus/chunks`, with local `source_path`, `page`, and `q` filters, corpus
lifecycle state through `GET /corpus/status`, no-key PDF layout marker
inspection through `GET /corpus/structure`, Markdown layout-marker export
through `GET /corpus/structure/report`, both with source/kind/page/text
filters, and
reusable local selections through `GET /corpus/profiles`,
`POST /corpus/profiles`, `GET /corpus/profiles/{profile_id}/status`, and
`POST /corpus/profiles/{profile_id}/activate`. The
`POST /corpus/profiles/{profile_id}/rebuild` route activates a saved profile and
queues the same selected-PDF FAISS rebuild job used by the general async index
route.
Profile status is read-only: it checks paper availability, active-selection
match, profile-vs-chunk index freshness, and rebuild requirement without
changing the active FAISS selection. The
`GET /corpus/profiles/{profile_id}/report` route exports that same no-secret
profile status as Markdown for handoff or offline review.
`GET /admin/status` reports JSON/SQLite corpus and chunk storage state. Admin
status also reports local index freshness by comparing the active paper source
set with the distinct source paths represented in chunk metadata. Corpus status
folds index rebuild jobs and freshness into `queued`, `parsing`, `indexed`,
`failed`, `stale`, or `empty`, making stale FAISS/chunk state visible without SSH
or manual SQLite inspection. The same admin status includes no-secret durable
storage readiness for future metadata database and object-storage backends.
Local JSON/SQLite/filesystem storage remains active; external database URLs,
buckets, and endpoints are represented only by configured/available booleans and
reason codes, without opening a connection or exposing secret values. Admin
status also reports a local storage inventory for metadata, jobs, artifacts,
uploads, and FAISS index files using only paths, file counts, byte totals, and
known-file existence flags; it does not read or return runtime file contents.
`src.storage_schema` provides the matching local storage-schema inventory for
future migration work. Admin status/report, Streamlit, and metrics expose schema
version, JSON/JSONL shape, and expected SQLite table/column presence for corpus,
chunk, job, artifact, API-key registry, product registry, and runtime-event stores without
returning row contents, prompts, answers, filenames, owner IDs, request IDs,
source paths, token values, or runtime file contents. `scripts/storage_schema.py`
exposes the same check for local or target-root preflight use and exits nonzero
when drift is found.
Admin status/report, Streamlit, and metrics also expose a derived
`platform_readiness` summary for production storage migration and distributed
worker acceptance. It deliberately reports only blocker codes, booleans, and
counts. In the current local runtime the schema/inventory and local worker
bridge checks pass, while production activation remains blocked until external
metadata database, object storage, and distributed job-store targets are
configured. The distributed worker target is reported separately from metadata
storage through `DISTRIBUTED_JOB_STORE_BACKEND`,
`DISTRIBUTED_JOB_STORE_URL`, and `DISTRIBUTED_JOB_QUEUE_NAME`; admin status only
returns backend, configured/available booleans, and reason codes, never the
store URL or queue name.
`src.platform_migration` composes the schema inventory, runtime backup manifest,
restore dry-run, read-only local job-store contract, storage readiness, and
distributed job-store readiness into a production migration preflight. The CLI
`scripts/platform_migration_preflight.py` returns success when local migration
evidence is complete, and `--require-activation` makes the command fail until
external metadata database, object storage, and distributed job-store targets are
configured. The output keeps the same no-secret boundary: no runtime contents,
job payloads, external URLs, bucket names, queue names, or credentials.
`src.storage_migration` adds a local migration rehearsal that copies required
runtime state into a staging root, then verifies that the staged tree matches the
source manifest and passes local storage-schema checks. The CLI
`scripts/platform_migration_rehearsal.py` uses a temporary staging directory by
default and deletes it after reporting; retained staging requires an explicit
`--staging-root`, and clearing an existing staging root requires
`--overwrite-staging`. Reports expose only group counts, byte totals, status
codes, and blocker codes; `.env` is never copied, runtime dependencies such as
models are skipped unless `--include-runtime-dependencies` is supplied, and
staging roots inside the source project are rejected.
`src.product_readiness` adds the equivalent no-secret productization check for
identity, API-key lifecycle, quota-store, and billing-provider readiness. It
separates `local_foundation_ready` from `activation_ready`: the current local
foundation can pass through API access audit, owner metadata, local rate-limit
configuration, query-cost estimation surfaces, and the optional local hashed-key
registry. When the optional local product registry is enabled, local-registry
identity, SQLite quota store, local RBAC, and local-ledger billing attribution
can also be verified without external accounts. When
`FLUXMIND_PRODUCT_QUOTA_GUARD_ENABLED`
is explicitly enabled together with the local product registry and SQLite quota
store, the FastAPI `/query`, `/query/inspect`, `/query/retrieve`, and
`/query/report` paths resolve the local user/workspace, check the configured
request quota, record a metadata-only usage event, and reject over-limit
requests before calling the model provider. When
`FLUXMIND_PRODUCT_RBAC_GUARD_ENABLED` is explicitly enabled together with the
local product registry and local API-key ownership, query routes require an
active workspace membership, local job submit/manage routes require
member/admin/owner roles, and corpus/index/admin destructive writes require
admin/owner roles. Denials return structured 403 responses and metadata-only
`product_rbac` runtime events. Activation remains blocked until the chosen
identity, quota-store, billing-provider, billing-attribution, runtime quota
guard, and RBAC guard targets are configured. The CLI
`scripts/product_readiness.py` exits successfully for local foundation readiness
and exits nonzero under `--require-activation` until those product activation
blockers are cleared. Reports expose only booleans, safe backend names, counts,
and blocker codes.
`src.provider_readiness` adds the same no-secret split for external provider
activation. The current local foundation passes with local mock image
generation, local Python execution, artifact registry, provider-failure
observability, and optional Docker/Octave readiness checks. Activation remains
blocked until `EXTERNAL_PROVIDERS_ENABLED` is set deliberately and external
image, hosted execution, MATLAB backend/license, and provider quota/cost guard
targets are configured. The CLI `scripts/provider_readiness.py` succeeds for
local foundation readiness and exits nonzero under `--require-activation` until
those provider activation blockers are cleared.
`src.quality_readiness` applies the same no-secret reporting discipline to RAG
quality maturity. It separates local source/eval readiness from live-evidence
readiness, masks report paths down to filenames, and keeps prompts, answers,
source paths, API keys, and runtime contents out of the output.
`src.storage_manifest` also owns the no-secret runtime backup manifest and
restore dry-run verifier. The manifest records group totals and SHA-256 hashes
for known metadata/job/API-key registry/index files without exporting file contents or `.env`
values; the verifier checks a supplied manifest against a target runtime root or
absolute manifest paths and reports missing or mismatched groups, files, byte
counts, and hashes without copying, overwriting, deleting, or restoring files.
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
under `artifacts/code-runs/` or `artifacts/octave-runs/`.

`DockerExecutionProvider` is the opt-in no-key container backend selected by
`CODE_EXECUTION_BACKEND=docker`. It preserves the same job/API contract while
running `docker run --rm` with network disabled, a bind-mounted per-run workdir,
read-only root filesystem, memory, CPU, and PID limits, dropped capabilities,
and `no-new-privileges`. `DOCKER_EXECUTION_IMAGE` selects the image; Python
requests run `python <entrypoint>`, and Octave/MATLAB-compatible requests run
`octave --quiet --no-gui <entrypoint>` inside that image.

`src/execution_policy.py` enforces a request-level `local-safe-v1` policy before
either child-process or Docker execution starts. It checks Python files with
`ast`, applies a configurable Python import allowlist, rejects obvious shell or
package-manager commands, blocks absolute-path literals in common file
constructors, and blocks Octave/MATLAB-compatible shell, network, and package
install calls. Policy failures return the stable job error code
`execution_policy_violation`; policy metadata is persisted with execution
results without exporting source code.

Execution results include no-secret reproducibility metadata for language,
entrypoint, input file counts/bytes, provider runtime, runtime availability/
details, filesystem isolation, network policy, timeout, memory, and CPU policy.
Local and Docker execution capture stdout/stderr through bounded stream readers;
`CODE_EXECUTION_MAX_STDOUT_BYTES` and `CODE_EXECUTION_MAX_STDERR_BYTES` cap
stored output while metadata records observed bytes and truncation flags.
Generated-artifact export is also bounded by `CODE_EXECUTION_MAX_ARTIFACTS`,
`CODE_EXECUTION_MAX_ARTIFACT_BYTES`, `CODE_EXECUTION_MAX_ARTIFACT_TOTAL_BYTES`,
and `CODE_EXECUTION_MAX_ARTIFACT_CANDIDATES`; metadata records scanned,
exported, skipped, and truncated collection counts without copying skipped
files.
Each code-execution attempt also appends a no-secret `code_execution` runtime
event with job id, owner metadata, language, selected backend, status/error
code, duration, artifact count, exit code, output/artifact limit metadata, and
policy metadata. Submitted source files, stdout, and stderr are intentionally
not copied into the runtime event. Admin status/report and the Streamlit admin
panel derive local advisory alerts from those events for failure-rate,
slow-duration, policy-violation, stdout/stderr truncation, and artifact
collection truncation signals.
Image files are returned as `plot` artifacts; text files are returned as `text`
artifacts; other small outputs are returned as `file` artifacts. Request files
and entrypoints must stay inside the per-run workdir, and symlink or
out-of-workdir outputs are not exported as artifacts. This gives the UI/API/job
model a concrete artifact shape for generated plots and files before any hosted
sandbox or real MATLAB backend is activated. Streamlit layers editable no-key
execution templates on top of the same provider-neutral job flow.
Python templates `smc_reaching_law` and `pmsm_current_step` write CSV/SVG
artifacts, and Octave-compatible templates `pmsm_current_decay` and
`smc_sign_switching` write CSV output when a local Octave runtime exists.

FastAPI exposes `GET /artifacts` and `GET /artifacts/{artifact_id}` so generated
mock diagrams, plots, and execution files can be listed and exported without
exposing raw filesystem paths. Export only supports local `file://` artifacts
that resolve under `ARTIFACTS_DIR`. The local artifact registry mirrors current
artifact metadata into `artifacts/artifacts.sqlite3` while still deriving records
from persisted jobs, giving later durable artifact storage a concrete migration
shape. The Streamlit sidebar also reads the local artifact registry and renders
recent artifacts with stable IDs, metadata, local filters, and download buttons.
`GET /artifacts` and the sidebar gallery support local `q`, `kind`, `job_kind`,
and `owner_id` filters for narrowing generated diagrams, plots, and files. Local
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
local development and production checks. It reports job counts by status/kind/
owner, artifact counts by owner, durable queue health, latest failed local jobs,
corpus paper counts, artifact counts/bytes, recent `/query` provider failures,
estimated no-secret query usage, metadata-only API access audit counts,
local API rate-limit status, upload-scan counts/reason-code summaries, provider token usage when the upstream response exposes it, runtime
directory existence/writability/bytes, local storage inventory, public model
names, durable storage readiness, code-execution backend readiness, Docker
sandbox accessibility, platform-readiness blocker codes, and optional no-secret query-cost estimates from
configured per-1M-token rates, recent query duration average/max from
`query_usage` events, metadata-only retrieval trace summaries,
retrieval-quality advisory alerts, query latency advisory alerts,
provider-failure advisory alerts, job-health advisory alerts
for failed/dead-lettered jobs and expired queue/lease state, product-readiness
local foundation and activation blockers, provider-readiness local foundation
and activation blockers, local API key registry readiness, plus explicit disabled switches for external providers
and identity/quotas/billing. The
Streamlit sidebar renders the same status,
including durable storage readiness, storage inventory, query-cost pricing
status, platform-readiness, product-readiness, and provider-readiness status,
query usage and latency status, API access audit status, upload-scan
status, local rate-limit status, and local metadata/object storage paths, so common
operational questions do not require SSH or raw filesystem inspection.
`GET /admin/status/report` renders the same snapshot as a Markdown operations
report, and the Streamlit runtime panel exposes the report as a download for
handoff or offline review. `GET /admin/metrics` exports the same local admin
summaries as Prometheus/OpenMetrics-style text for local scraping. The metrics
are metadata-only local-window gauges and omit owner IDs, request IDs, paths,
prompts, answers, uploaded contents, filenames, and artifact contents. The
Streamlit admin panel exposes the same text as a download. `GET
/admin/retention` provides the default preview of upload and artifact files that
match local age-based retention thresholds.
`POST /admin/retention/delete` can delete the same bounded candidate set only
when `RETENTION_DELETE_ENABLED` is explicitly true; otherwise it returns a
guarded disabled result. The delete path is authenticated, excludes artifact
SQLite metadata files, reports paths/bytes for the deleted candidates in the
response, and records only aggregate `retention_delete` runtime-event counts.
The Streamlit admin panel exposes the preview with local day/limit controls and
only shows the delete button when the same config flag is enabled.
`GET /admin/events` lists no-secret runtime events with local `kind`, `code`, and
`q` filters, and the Streamlit admin panel exposes the same event viewer for
provider-failure, query-usage, retrieval-trace, code-execution, API-access, and
upload-scan inspection plus retention-delete events without reading raw JSONL.
Malformed runtime-event JSONL lines are skipped with warnings so one bad history
line does not break the admin/event viewer path.
Admin status also summarizes recent code execution events by code,
status, backend, policy violations, output/artifact truncations, exported
artifact bytes, failure rate, duration, and advisory alert codes.
`GET /admin/runtime-manifest` and `GET /admin/runtime-manifest/report` export the
same no-secret runtime backup manifest as the CLI. The authenticated JSON route
`POST /admin/runtime-manifest/restore-check` and Markdown route
`POST /admin/runtime-manifest/restore-check/report` run the non-destructive
manifest verifier against the local runtime root. The Streamlit runtime panel
can upload a saved manifest JSON, render the same no-secret summary, and download
the Markdown dry-run report without copying, overwriting, deleting, or restoring
files.

`POST /query/report` reuses `query_with_metadata()` to return a Markdown research
report containing the generated answer, citation validation, and retrieved
context refs. For implementation and code-generation requests it also appends a
paper-to-code handoff section with source refs, assumption/parameter guardrails,
fenced code blocks, cited artifact IDs, and validation checklist fields. It is
an export surface for the current single-user/local workflow, not a share-link
or multi-user report store.
