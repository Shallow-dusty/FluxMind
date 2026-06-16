# FluxMind Platform Audit and Roadmap

Last updated: 2026-06-16

For reading order and document ownership, see `docs/README.md`. Current git and
verification state is tracked in `docs/REPO_STATUS.md`.

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

## Project Evaluation: 2026-06-15

Current verified state: the no-key/local platform baseline is healthy and
deployed, but the project is still a single-machine product foundation rather
than a full multi-user SaaS platform. The strongest current capabilities are the
paper-grounded RAG workflow, local corpus/profile management, no-secret admin
status surfaces, durable local job records, local worker-service bridge, artifact
metadata/export, retrieval diagnostics, and deterministic offline RAG regression
gates, plus the opt-in no-key Docker execution backend.

Current verification run:

```text
Gate                                      Result
----------------------------------------  -------------------------------------
.venv/bin/python -m pytest                pass, 334 tests, 2 known warnings
.venv/bin/python -m coverage run -m pytest
coverage report --fail-under=88           pass, 88% total branch coverage
.venv/bin/python scripts/evaluate_rag.py  pass, 42 answer cases, 65 retrieval-only
                                             cases, 12 code-output cases,
                                             20 PDF structure cases,
                                             42 recorded answers
health_check.py local/docs anchors         pass, including query-latency/query-alert/provider-alert/job-alert/API-access-audit/API-rate-limit/upload-scan/retention-delete/metrics-export/retrieval-trace/retrieval-alerts/storage-schema and repo/roadmap drift checks
storage_schema.py local preflight          pass, ok=true, 7 stores, 0 problems
runtime manifest restore dry-run          pass, ok=true, 6 groups, 5 checked files, manifest_errors=0 against exported local manifest
health_check.py HTTPS endpoints            14:17 snapshot, UI/API 200
health_check.py SSH runtime                14:17 snapshot, services active, active_papers=30, chunks=1934, index_fresh=True, retrieval/admin metrics smokes OK
```

Current assessment:

```text
Dimension                 Assessment
------------------------  -----------------------------------------------------
Research demo readiness   strong: deployed UI/API, seed corpus, citation checks
No-key local platform     strong: WP0-WP6 baseline implemented and tested
Operational visibility    strong for local runtime; no-secret reports available
RAG quality coverage      small-group retrieval bar met; broad live answer QA remains open
Execution/artifacts       good contract baseline; not production sandboxed
Storage/queue durability  local SQLite/JSONL bridge plus explicit external
                          job-store readiness target; not distributed yet
Product maturity          pre-platform: no accounts, quotas, billing, teams
Frontend maturity         demo/personal workflow; Streamlit remains limiting
```

Priority implication: the next work should improve platform foundations before
activating real paid providers. The most defensible next lane is production
storage plus distributed job/worker state, because identity, quotas, billing,
frontend replacement, and provider activation all depend on durable ownership and
job/artifact boundaries.

Current no-key feature work now includes local provider implementations:
`MockImageGenerationProvider` writes deterministic SVG artifacts, and
`LocalPythonExecutionProvider` and `LocalOctaveExecutionProvider` exercise the
code-execution contract for development, including generated file and plot
artifact capture. Octave-compatible jobs use a local `octave` executable when
present and otherwise persist a structured runtime-unavailable failure. These
are wired through immediate local job endpoints, async in-process job endpoints,
artifact list/export endpoints, and the Streamlit local job panel. Generated
artifact metadata is mirrored into local SQLite for current-state inspection.
Generated artifacts now carry provider-neutral byte counts, SHA-256 checksums,
prompt/style/source-reference/cost metadata, and admin status can verify local
artifact integrity without exposing artifact contents. Recent artifacts are
available to RAG answers as stable
`[Artifact:<id>]` references, but not through any real external provider.

Current RAG quality work includes an offline baseline in
`eval/rag_baseline.json`, a no-network evaluator in `scripts/evaluate_rag.py`,
numbered citation validation, source/page fixture verification, provider-error
fixtures, generated-answer inspection metadata, and selectable answer modes. The
baseline now has 42 small-group answer cases, 42 recorded answers, and 65
retrieval-only cases for 107 total no-LLM retrieval questions, plus 12 local
Python code-output cases (four job-backed) and 20 seeded PDF
equation/table/figure/algorithm structure cases. The baseline gates 145 expected source/page refs, 111 topic tags,
ontology-group coverage, eval-lane coverage spanning retrieval, answer quality, equation
fidelity, code generation, forum-style debugging, failure modes, and
paper-to-code reports, code-output case/language/template/execution-mode/pass
rate coverage, and PDF structure kind/pass-rate coverage. The evaluator checks
that expected PDF sources/pages contain their configured snippets, checks
retrieval-only source/page anchors without answer fixtures, checks recorded
answers for citation validity plus deterministic key-term coverage thresholds,
verifies that local Python code-output fixtures produce expected stdout plus
plot/text artifacts, including reusable execution templates, paper-specific
paper-to-code examples, and job-backed local execution paths, and verifies that
representative PDF pages still expose equation/table/figure/algorithm markers for
paper-to-code work.
Retrieval now uses a local hybrid path: FAISS
vector hits plus BM25-lite keyword matches from the indexed docstore, with
dedupe, deterministic BM25-lite lexical reranking, optional no-key local
CrossEncoder reranking when `RERANKER_MODEL` points to an existing local model
path, and a `TOP_K` context cap. Missing or empty reranker paths fall back to
BM25-lite without runtime downloads.
`src.chain.retrieve_with_metadata()` and `POST /query/retrieve` expose
retrieval-only source/page diagnostics without calling an LLM provider.
`src.chain.query_with_metadata()` and `POST /query/inspect` expose generated
answer citation validation against retrieved source/page refs. This is a
regression harness and retrieval baseline, not a claim that fresh live model
output has been fully scored.

Current corpus storage work includes a 30-paper curated seed library plus a
local JSON metadata registry in
`metadata/corpus.json` with checksums, source paths, titles, authors, year,
DOI, arXiv ID, venue, topic tags, active/indexed state, chunk counts, and
parse/index error fields. That state is now mirrored into
`metadata/corpus.sqlite3` as a local current-state index. Indexed chunk metadata
is mirrored into `metadata/chunks.sqlite3` with source path, page, chunk
sequence, content hash, character count, and preview text. Uploaded PDFs are
deduplicated by SHA-256 against the selectable local corpus before creating a
new local upload file or adding duplicate chunks to FAISS. `GET /corpus/papers`
exposes paper records through the API and supports local filters for query text,
active state, source kind, and indexed status. `GET /corpus/chunks` exposes
chunk records, and `GET /admin/status` reports JSON/SQLite corpus and chunk
storage state. Corpus/profile JSON writes now use atomic replace to avoid
transient empty reads during concurrent metadata refreshes. `PUT
/corpus/active` persists active/deactivated paper selection
without direct runtime-file edits; an index rebuild is still required to make
FAISS exactly match a changed selection. Manifest and active-selection runtime
JSON reads are tolerant of malformed local state, active selections are
constrained to selectable project-local PDFs, and active-selection writes use
atomic replace. Admin status now reports index freshness
by comparing active paper source paths with chunk metadata source paths, so stale
or missing index state is visible through the API. The same admin status reports
future metadata database/object-storage readiness without SSH or exposing
external storage credentials. The same local admin surfaces now expose
storage-schema readiness for JSON, JSONL, and SQLite stores without returning row
contents, prompts, answers, filenames, owner IDs, request IDs, source paths, or
runtime file contents, and `scripts/storage_schema.py` exposes the same check as
a CLI preflight with JSON/Markdown output. Admin status/report, Streamlit, and
metrics now also expose a no-secret `platform_readiness` summary for production
storage migration and distributed worker acceptance: local schema/inventory and
the local worker bridge are clean, while external metadata database, object
storage, and the separately configured distributed job-store target remain
explicit blockers. `scripts/platform_migration_preflight.py` now combines that
schema readiness, the runtime manifest, restore dry-run, local durable job-store
contract, and external backend blocker split into one no-secret CLI gate. The
default command validates local migration evidence; `--require-activation`
fails until real external metadata database, object storage, and distributed
job-store targets are configured. The no-secret runtime manifest can also be checked
against a target runtime root through the CLI or authenticated API without
copying, deleting, or restoring files. This is still a local
baseline, not the final multi-user database. Reusable local corpus profiles now
persist named active-paper selections under `metadata/corpus_profiles.json`, with
API routes to list, upsert, inspect status, and activate them. Profile status
reports paper availability, active-selection match, profile-vs-chunk index
freshness, and rebuild requirement without changing the current active corpus.
Profile status can be exported as a no-secret Markdown handoff report from both
the API and the Streamlit corpus profile panel. Profiles can also be activated
with a queued selected-PDF FAISS rebuild through the local async job manager.
This gives multiple local corpus selections a no-key coexistence path without
introducing real user/workspace permissions yet.

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
  endpoint; metadata-only API access audit events now make token-status outcomes
  inspectable without storing credentials or request bodies, the local
  rate-limit guard can return HTTP 429 before route handling when configured,
  and `/admin/metrics` can export the same no-secret admin summaries as local
  scrapeable metrics text.
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
- Uploaded PDFs are stored locally without per-user ownership or quotas. A local
  pre-write upload scan now validates PDF magic and parseability, blocks
  encrypted PDFs and common active-content markers by default, and records
  metadata-only reason-code events; this is still not a production antivirus,
  sandbox-scanning, or identity-backed abuse-control service. Local checksum
  deduplication avoids writing duplicate upload files or duplicate vector chunks
  for already indexed PDFs, and admin retention preview shows age-based
  upload/artifact cleanup candidates. A guarded local retention-delete route can
  remove those candidates only when explicitly enabled, but production
  identity-backed deletion/audit controls remain planned.
- The pytest suite, GitHub Actions CI gate, local/remote health checker, and
  offline RAG fixture evaluator now exist. They still do not cover live
  retrieval-quality scoring or model-answer citation correctness.
- Citation quality depends on raw chunk metadata. There is no source
  normalization, DOI/arXiv metadata enrichment, reranking, or citation verifier.
- LLM/provider errors are normalized for the UI and API, but provider-specific
  retry policy and richer error taxonomy are still basic.
- Request IDs are logged for UI/API requests, and successful query duration,
  token usage, and provider failures are now persisted as no-secret runtime
  events; API access token-status outcomes are also summarized locally, while
  local API rate-limit outcomes are visible through the same admin surfaces.
  Retrieval-hit traces now have a metadata-only local event/summary baseline,
  and the metrics export is still a local baseline rather than a production
  scrape, tracing, or alert-routing stack.
- No production execution sandbox is enabled by default. The local Python/Octave
  child-process providers remain development contract exercises, while the
  opt-in Docker backend and request-level policy preflight prove the no-key
  container/policy path before any hosted or public execution surface is enabled.

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
- Extend the offline evaluation set into live or recorded-model regression
  scoring for standard SMC/flux questions.

Current progress: a local JSONL job history, SQLite current-state job index,
durable SQLite idempotency claims, immediate no-key job endpoints, in-process
async no-key job endpoints, and local scheduled retry/backoff exist for mock
image generation, development-only Python execution, Octave-compatible
execution, and selected-PDF index rebuilds.
List/status/retry/scheduled-retry/cancel endpoints exist. The Streamlit sidebar
can trigger queued no-key jobs, display filtered latest job state, cancel
queued/running jobs, and retry failed/cancelled jobs immediately or after a local
backoff delay. `GET /jobs` and the sidebar recent-job panel support local `q`,
`status`, `kind`, and `owner_id` filters for job inspection without raw
JSONL/SQLite reads.
Immediate and async job submissions can include `idempotency_key`; duplicate
submissions for the same job kind and key return the existing persisted job, and
missing keys still create fresh jobs.
Query and job submissions can also include optional local `owner_id` and
`owner_label` metadata. These fields default to `local-user` / `Local user` and
persist through durable jobs, job logs, query runtime events, artifacts, and
admin summaries, but they are not identity, quota, tenancy, or billing controls.
Async submissions can set `max_attempts` and `retry_backoff_s`; failed attempts
are requeued as the same durable job until exhausted, then marked
`dead_lettered` with no-secret transition logs.
Queued/scheduled jobs are rehydrated from SQLite/JSONL on API startup, async
jobs and scheduled retries can set `queue_timeout_s`, and expired queued jobs
fail before execution with `job_deadline_exceeded`. Admin status exposes queue
health including queued, due, scheduled, expired, running, leased queued,
expired lease, running lease, and oldest queued timestamps. In-process workers
now claim queued jobs through the durable store with `worker_id`, `leased_at`,
and `lease_expires_at` before provider execution, and expired queued leases can
be reclaimed. Admin status also exposes no-secret worker lease health with
worker IDs, active/expired lease counts, and latest leased job summaries.
`LocalDurableJobWorker`, `scripts/run_job_worker.py`, and
`deploy/systemd/fluxmind-worker.service` can now claim and execute due queued
jobs outside the API/Streamlit process as an enabled no-key local worker-service
foundation. The explicit durable worker also polls durable job state while running
local providers and forwards `cancelled` state through the existing
`cancel_event` path, so local Python/Octave child processes can terminate
outside the API process. Job records now include no-secret transition logs for
queued, running, terminal, and cancelled states. This proves the
UI/API/status/artifact shape and a local restart-recovery/lease/retry/
dead-letter/worker-service bridge, but it is not yet a distributed multi-worker
queue or database-backed worker.
Local execution timeouts now persist as `execution_timeout` so UI/API/admin
surfaces can distinguish timeout failures from ordinary non-zero exits.
API startup now warms only an already-present FAISS index and does not
synchronously rebuild a missing index before binding the port; missing/corrupt
index recovery should happen through explicit index rebuild jobs.
Index rebuild jobs now observe cancellation during loading/splitting and before
committing rebuilt index state, preventing a cancelled local rebuild from
publishing new FAISS/chunk metadata after the cancellation signal is seen.

Corpus metadata progress: local paper metadata exists and can be listed through
the API. Active/deactivated paper selection can be updated through the API and
Streamlit without direct filesystem edits. Upload and selected-PDF rebuild flows
update the local metadata file.
Corpus and chunk metadata are also mirrored into local SQLite for current-state
inspection and future database migration.
Admin status exposes local index freshness (`fresh`, `stale`, `missing`, or
`empty`) so active-corpus changes that still need an index rebuild are visible
without reading runtime files by hand.
`GET /corpus/status` now exposes corpus lifecycle state directly as `queued`,
`parsing`, `indexed`, `failed`, `stale`, or `empty`, using index rebuild jobs,
paper status, and index freshness instead of requiring callers to infer status
from multiple endpoints.
Uploaded/unmanifested PDFs now get best-effort no-key bibliographic extraction
from embedded PDF metadata and first-page title, author, DOI/arXiv, year, and
keyword/index-term text, with curated seed-manifest values taking precedence.
Durable user/corpus/chunk/artifact/job metadata is still planned.

### Phase 2: Better RAG

- Current progress: answer modes, generated-answer citation inspection, numbered
  citation prompt guards, an offline fixture/recorded-answer
  citation/provider/source-page regression gate, and an optional live
  `/query/inspect` regression gate exist. No-LLM `/query/retrieve` diagnostics
  can inspect deployed retrieval context refs and source/page completeness
  before generation, and `scripts/evaluate_rag.py --retrieval-url` can score
  those deployed retrieval diagnostics as part of the no-key quality gate. Local
  hybrid vector+BM25-lite keyword retrieval, deterministic BM25-lite reranking,
  and optional no-key local CrossEncoder reranking exist. The evaluator can
  export no-secret JSON summaries for deployment evidence, and aggregate
  `quality_gates` now fail eval-set breadth, answer-mode coverage,
  recorded-answer, and optional live pass-rate regressions. Hosted/service
  model-answer scoring remains planned only if external evaluation
  infrastructure is deliberately activated.
- Continue broadening bibliographic enrichment where uploaded PDFs need stronger
  multi-line author/affiliation parsing or external resolver-backed metadata.
- Add external/service reranking only if the local BM25-lite and optional local
  CrossEncoder baseline are insufficient and account/model operations are
  deliberately activated.
- Add citation verification: every cited source should map back to a retrieved
  chunk and page. Current inspect/live-eval plumbing verifies this for numbered
  retrieved-context refs.
- Add answer modes: explanation, derivation, implementation, literature review,
  and code generation.

### Phase 3: Image and Diagram Generation Interface

Status: provider-neutral plumbing and a no-key mock/local provider exist. Real
image-provider activation remains disabled until a key/account and artifact
storage policy are configured.

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

Current progress: mock diagram and execution artifacts store byte counts and
SHA-256 checksums, while diagram artifacts also store prompt, style, local SVG
template, size, source references, provider/model, and zero-cost metadata. The
no-key local SVG templates now cover generic engineering diagrams,
sliding-mode-observer blocks, PMSM control loops, and paper-figure redraft
scaffolds. Artifact metadata is mirrored into local SQLite as a current-state
index, and admin status reports artifact integrity counts for
ok/missing/unchecked/mismatched local files. The Streamlit artifact gallery
exposes stable artifact IDs, metadata, local filters, and downloads, and
`GET /artifacts` supports local `q`, `kind`, `job_kind`, and `owner_id` filters
for generated diagrams, plots, and files. Artifact records inherit owner
metadata from their source jobs. RAG prompts can include recent generated
diagrams, plots, and files as
`[Artifact:<id>]` references.

OpenAI's current image docs separate simple Image API generation/edits from
Responses API image tools for conversational, iterative image workflows. That
maps well to FluxMind: use a simple image endpoint for one-shot diagrams, and
reserve conversational editing for later.

Reference: https://platform.openai.com/docs/guides/image-generation

### Phase 4: Code Execution

Status: local Python, Octave-compatible, and opt-in Docker execution interfaces
exist. Real hosted execution remains disabled until infrastructure is
configured. Real MATLAB support additionally requires license/account
decisions.

Current progress: the local Python provider and the local GNU Octave-compatible
provider capture stdout, stderr, exit code, generated files, and generated image
plots as persisted artifacts. The Octave-compatible provider returns structured
runtime-unavailable diagnostics when the `octave` binary is absent. Local
execution results now persist timeout/memory limit metadata plus no-secret
environment/policy metadata for language, entrypoint, input file counts/bytes,
provider runtime, local runtime availability/details, temporary workdir
isolation, and the current lack of network-policy enforcement. Unix child
processes receive address-space and CPU-time limits where supported. Input files
and entrypoints are constrained to the per-run workdir, and symlink or
out-of-workdir outputs are not exported as artifacts. File count, per-file bytes,
and total input bytes are capped before materialization. Generated-artifact
export is also bounded by configurable artifact count, per-artifact bytes, total
artifact bytes, and candidate scan limits, with exported/skipped/truncated
metadata on execution results. The child-process providers remain development
providers. `CODE_EXECUTION_BACKEND=docker` now
switches code jobs to `DockerExecutionProvider`, which runs `docker run --rm`
with network disabled, a bind-mounted per-run workdir, read-only root
filesystem, memory, CPU, and PID limits, dropped capabilities, and
`no-new-privileges`. The Streamlit job panel offers editable local Python and
Octave-compatible control-engineering templates, so SMC/PMSM example jobs can
produce artifacts without requiring a blank script. `DOCKER_EXECUTION_IMAGE`
selects the container image, and admin status reports whether the Docker backend
is configured and accessible to the runtime user without running user code.
`CODE_EXECUTION_POLICY=local-safe-v1` now applies a request-level preflight
before any child process, Octave lookup, or Docker container starts. It uses
Python AST checks plus Octave text checks to reject disallowed imports,
shell/package-manager commands, absolute-path literals in common file
constructors, and Octave shell/network/package-install calls. Policy failures
persist as `execution_policy_violation` job errors with no-secret policy
metadata. Each code execution attempt also emits a no-secret `code_execution`
runtime event with job id, owner metadata, language, backend, status/error code,
duration, artifact count, exit code, policy metadata, and output/artifact limit
metadata; submitted source, stdout, and stderr are not copied into the event.
Admin status/report summarize recent execution outcomes by code, status,
backend, policy violations, output/artifact truncations, exported artifact
bytes, failure rate, duration, and advisory alert codes. Local and Docker
execution now capture stdout/stderr through bounded readers controlled by
`CODE_EXECUTION_MAX_STDOUT_BYTES` and `CODE_EXECUTION_MAX_STDERR_BYTES`, with
observed byte counts and truncation flags persisted in execution metadata.
Generated-artifact export is bounded by configurable count, per-artifact byte,
total-byte, and candidate-scan limits. Local advisory alerts cover high failure
rate, slow duration, policy violations, stdout/stderr truncation, and artifact
collection truncation.

Artifact progress: generated local artifacts can be listed and downloaded
through `GET /artifacts` and `GET /artifacts/{artifact_id}`. This gives image
and execution outputs an export path before real provider storage is configured.
The Streamlit sidebar also includes a local artifact gallery for recent job
outputs.

Remaining work is no longer "add an execution provider" or "add first package
policy"; it is hardening the chosen execution backend for production operations:
production metrics/tracing/alerts beyond the local advisory baseline, broader
abuse controls, distributed execution if needed, and a clear MATLAB-vs-Octave
product decision.

Recommended path:

- Start with Python execution for numerical examples and plotting.
- Keep GNU Octave as the MATLAB-compatible stepping stone.
- Add real MATLAB only if licensing, server resources, and isolation are
  deliberately solved.
- Capture stdout, stderr, exit code, generated files, and plots as artifacts.
- Enforce quotas, network policy, package/filesystem controls, and per-session
  workspaces.

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
- Add real billing attribution only after identity/workspace decisions are made.
- Add admin views for provider failures, token spend, and corpus storage.

Current progress: the first no-key admin foundation exists through
`GET /admin/status` and a Streamlit sidebar status panel. It reports local job,
corpus, artifact, recent `/query` provider-failure events, estimated no-secret
query usage with duration summaries, provider token usage when returned by the
upstream response, optional configured query-cost estimates, runtime-directory,
durable storage readiness, public model, job/artifact owner summaries,
metadata-only API access audit summaries, local API rate-limit status,
metadata-only retrieval trace summaries, no-secret storage-schema readiness,
no-secret platform-readiness blockers, no-secret metrics text export, and
disabled-provider/product switch state
without exposing API keys, storage
credentials, or requiring real identity/billing systems. The Streamlit status
panel now renders the same storage readiness, local metadata/object storage
paths, platform-readiness status, API access audit/rate-limit status, metrics download, and no-secret
pricing status directly for dashboard use. The runtime
manifest and restore-check API/CLI/UI provide no-secret backup and dry-run
verification evidence for local runtime state. `GET /admin/status/report` and
the Streamlit status panel can export
that same no-secret snapshot as a Markdown operations report for handoff or
offline review. Successful `/query`, `/query/inspect`, `/query/report`, and
`/query/retrieve` calls emit metadata-only `retrieval_trace` events with
endpoint, answer mode, context count, source/page completeness counts, citation
status when available, duration, and whether an LLM provider was called; admin
status/report, Streamlit, and metrics summarize them without prompts, answers,
retrieved text, source paths, owner IDs, or request IDs. The same local status
surfaces derive metadata-only advisory alerts for empty retrievals, missing
source/page metadata, and citation validation failures; this remains local
observability, not production alert routing. `POST /query/report`
exports an answer, citation validation, and
retrieved context refs as a Markdown research report; implementation and
code-generation reports now include a paper-to-code handoff with source refs,
assumption/parameter guardrails, fenced code blocks, cited artifact IDs, and
validation checklist fields. The `GET /corpus/profiles/{profile_id}/report`
route exports one saved local corpus profile's read-only status as a Markdown
handoff report. `GET /corpus/structure/report` exports filtered
equation/table/figure/algorithm layout anchors as a Markdown handoff report with
kind/source summaries. `GET /admin/retention`
previews upload/artifact files matching age-based retention thresholds.
`POST /admin/retention/delete` can delete the same bounded candidate set only
when explicitly enabled, and the Streamlit admin panel exposes the delete action
only under that config flag. `GET /admin/events` lists no-secret runtime events with
local `kind`, `code`, and `q` filters, and the Streamlit admin panel exposes the
same event viewer, including provider-failure, query-usage, retrieval-trace,
code-execution, and API-access audit event kinds. Estimated query usage remains
the fallback when provider
usage data is absent; provider-specific pricing is available as local
configuration only, while billing attribution and user cost dashboards remain
blocked on product decisions.

## Near-Term Implementation Plan

1. Use `docs/QUALITY_ROADMAP.md` and the `quality_maturity` section of the eval
   JSON report as the product steering layer: the small-group quality bar is
   met, so the next decision is community-quality evidence versus production
   storage/distributed-worker hardening.
2. Treat `9b1cbc5` as the current source/eval quality baseline and `18200f6`
   as the last deployed application baseline until the next guarded deploy.
   Keep `docs/REPO_STATUS.md` plus `docs/DEPLOYMENT_STATUS.md` refreshed after
   any future release or live-state verification pass. Repo documentation commits
   may be newer than the deployed application-code baseline.
3. Keep the local health/eval/deploy-smoke checks as the deploy gate, and extend
   them when a new status or drift class is found.
4. Use `docs/ARCHITECTURE.md` and `docs/BACKLOG.md` as the implementation
   source of truth for platformization work.
5. Use `scripts/platform_migration_preflight.py` plus the local
   `platform_readiness` blockers to choose and test the production metadata
   database, object storage, and distributed job-store backend. The readiness
   surface already has separate metadata, object, and job-store targets; the
   remaining work is backend selection, real migration execution, and live
   activation. Then extend the local restart-recovery/lease/worker-service bridge
   into a distributed worker/storage runtime and extend the local metrics export
   into production scrape/tracing/alert routing plus deeper abuse controls
   before enabling real
   external image generation or hosted code execution providers.
6. Choose the next platformization lane explicitly: frontend/API split,
   production storage, isolated execution, or identity/quota/billing. Do not
   activate real provider keys, hosted sandboxes, or MATLAB licensing until the
   matching runtime boundary is implemented and tested.

## Open Decisions

- Keep Streamlit as the production UI for the next iteration, or treat it as a
  demo shell and begin a frontend/API split.
- Choose storage: local volume first, or object storage plus relational
  metadata now.
- Decide whether the implemented local Docker backend is sufficient for the next
  release, or whether production use needs Cloudflare Sandbox or a dedicated VM
  with stronger container isolation.
- Decide whether "MATLAB compiler" means true MATLAB, Octave-compatible code,
  or Python equivalents for control systems.
