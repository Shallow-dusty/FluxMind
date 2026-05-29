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
- `scripts/health_check.py` reports local/remote FAISS index size and active
  paper count when available.
- `scripts/health_check.py --ssh-host ...` includes recent journal error lines
  for the UI/API services.

Acceptance:

- Local tests pass.
- Public UI/API return 200.
- Remote systemd services, ports, model config, and disk checks pass.
- Browser translation guard remains covered by tests.

## WP1: RAG Quality Baseline

Status: offline baseline, recorded-answer scoring, local hybrid retrieval, and
deterministic lexical reranking implemented; live provider scoring and full
citation correctness remain planned

- `eval/rag_baseline.json` contains a small control-engineering evaluation set.
- Each case records expected source papers/pages, fixture snippets, recorded
  answers, required answer terms, and minimum answer-term coverage.
- `src.chain.validate_numbered_citations()` validates answer citations like
  `[1]` against retrieved document refs.
- `scripts/evaluate_rag.py` runs the offline baseline without network calls and
  fails recorded answers that miss required refs or key-term coverage thresholds.
- Provider failure fixtures cover timeout, 429/rate-limit, empty output, and
  malformed streaming chunks.
- Answer modes exist in the prompt/API/UI: explanation, derivation,
  implementation, literature review, and code generation.
- `src.chain.hybrid_retrieve()` merges FAISS vector hits with local keyword
  matches from the indexed docstore, dedupes chunks, and keeps the final context
  bounded by `TOP_K`.
- `src.chain.rerank_documents()` applies a deterministic no-key lexical
  relevance reranker before context formatting.
- Still planned: live provider retrieval-quality scoring, source/page
  verification against newly generated model answers, stronger learned/service
  reranking, and regression thresholds on real eval answers.

Acceptance:

- Evaluation command runs without network where fixtures are available.
- Citation regressions fail locally before deployment.
- Recorded answer coverage regressions fail locally before deployment.
- Provider errors surface as structured user-facing messages.

## WP2: Corpus and Storage Layer

Status: local JSON corpus metadata baseline and active/deactivated selection
workflow implemented; durable database/object storage remains planned

- `src/metadata.py` stores local paper metadata in git-ignored
  `metadata/corpus.json`.
- Paper records include checksum, title, source path, source kind, active flag,
  indexed status, chunk count, and parse/index error fields.
- `GET /corpus/papers` lists the current local paper metadata without requiring
  manual filesystem inspection.
- `PUT /corpus/active` persists activation/deactivation choices after validating
  project-relative source paths against the selectable corpus.
- Upload and selected-PDF index rebuild flows update paper metadata.
- Decide storage path: local volume first or object storage plus relational DB.
- Still planned: durable metadata for chunks, corpora, jobs, artifacts, users,
  ownership, and retention; object storage or relational DB; richer
  durable deactivation/reactivation workflows across multiple users/corpora.

Acceptance:

- Rebuilding an index is a job with status and logs.
- A paper can be indexed and traced to source path/checksum/chunk count.
- Active/deactivated state can be managed without editing runtime files by hand.
- Storage state can be listed without reading raw filesystem layout manually.

## WP3: Job System

Status: local JSONL history plus SQLite current-state index, in-process async
queue, scheduled retry/backoff, restart recovery for queued jobs, and queue
health implemented; distributed multi-worker queue and full running
cancellation remain planned

- Local JSONL job records exist in `src/jobs.py`.
- Job writes are mirrored into `jobs/jobs.sqlite3` for current-state lookups and
  migration toward durable worker storage.
- `POST /jobs/image/mock` creates a mock image-generation job.
- `POST /jobs/code/python-local` creates a development-only Python execution
  job.
- `POST /jobs/index/rebuild` rebuilds the FAISS index from selected PDFs as a
  persisted job.
- `POST /jobs/async/image/mock`, `POST /jobs/async/code/python-local`, and
  `POST /jobs/async/index/rebuild` enqueue those local jobs through an
  in-process background worker.
- `GET /jobs` lists latest jobs.
- `GET /jobs/{job_id}` returns persisted job status.
- `POST /jobs/{job_id}/retry` retries failed/cancelled local jobs with a new
  job ID.
- `POST /jobs/{job_id}/retry-scheduled` queues failed/cancelled local jobs for
  delayed retry with `parent_job_id` and `not_before` metadata.
- `AsyncJobManager.recover_queued_jobs()` rehydrates queued/scheduled jobs from
  SQLite/JSONL after service restart and returns them to the local worker queue.
- `GET /admin/status` exposes `queue_health` with queued, due, scheduled,
  running, and oldest queued timestamps.
- `POST /jobs/{job_id}/cancel` records cancellation for queued/running job
  states. Running local Python jobs observe cancellation; index rebuild
  cancellation is only checked before execution starts.
- Streamlit sidebar can trigger selected-PDF index rebuild jobs, mock image
  jobs, local Python jobs, and display latest job status.
- Streamlit recent-job panel can cancel queued/running jobs and retry
  failed/cancelled jobs immediately or after a local backoff delay.
- Still planned: distributed database-backed worker beyond the local SQLite
  recovery bridge, true cancellation of all running work, and richer timeout
  policy.

Acceptance:

- Job records preserve request, result, artifacts, errors, attempts, and
  request IDs.
- Failed code-execution jobs preserve stderr/error details.
- API and Streamlit can show running/succeeded/failed states through job status.
- Queued local job endpoints return without blocking request handlers.
- Queued delayed retries can recover after API service restart.
- Remaining: production-grade long-running work still needs a distributed worker
  outside the Streamlit/API processes.

## WP4: Image and Diagram Generation

Status: provider-neutral plumbing, no-key mock provider, local artifact export,
artifact metadata, and RAG artifact references implemented; real image-provider
activation remains disabled until a key/account is configured

- `MockImageGenerationProvider` implements the `ImageGenerationProvider`
  contract with deterministic local SVG output.
- `GET /artifacts` lists generated local artifacts from persisted jobs.
- `GET /artifacts/{artifact_id}` exports local file artifacts by stable ID.
- Streamlit sidebar includes a local artifact gallery with stable IDs,
  provider-neutral metadata, and download buttons.
- Mock diagrams store prompt, style, size, source references, provider/model,
  and zero-cost metadata without external keys.
- RAG prompts include recent generated artifacts as stable `[Artifact:<id>]`
  references so answers can point to local diagrams, plots, or files.
- Start with engineering diagrams and paper-figure redrafts.
- Keep generated images as artifacts rather than inline chat-only blobs.

Acceptance:

- A request can generate an artifact record with a stable URI.
- Generated mock diagrams produce persisted artifact URIs.
- Generated diagrams and execution artifacts can be listed and exported.
- Generated local artifacts can be downloaded from Streamlit.
- Provider can be swapped without changing the UI flow.

## WP5: Code Execution

Status: local request/result plumbing, Python execution, Octave-compatible
execution interface, and file/plot artifact capture implemented; real hosted
execution and MATLAB activation remain disabled until infrastructure and
license/account decisions are made

- `LocalPythonExecutionProvider` implements the `CodeExecutionProvider`
  contract for development-only Python snippets.
- `LocalOctaveExecutionProvider` implements the same contract for GNU
  Octave-compatible scripts when a local `octave` binary is installed; when it
  is absent, jobs fail with a structured runtime-unavailable diagnostic.
- Python and Octave-compatible snippets can capture stdout, stderr, exit code,
  generated text/file artifacts, and generated image files as plot artifacts.
- `POST /jobs/code/octave-local` and `POST /jobs/async/code/octave-local`
  expose immediate and queued no-key Octave-compatible job flows.
- Streamlit includes a local Octave-compatible job panel.
- Job records persist execution artifacts alongside the execution result.
- Still planned: run code in an isolated service with CPU, memory, timeout,
  filesystem, and network limits.

Acceptance:

- Current development provider runs code in a child Python process and temporary
  workdir, but it is not a production sandbox.
- Octave-compatible requests have stable API/UI/job behavior without enabling
  real MATLAB licensing.
- Execution results are reproducible from stored input files and environment
  metadata.
- A failed local execution returns structured diagnostics without breaking the
  app.

## WP6: Product Shell

Status: local/admin status foundation implemented; keep public identity,
API-key lifecycle, quotas, and billing disabled until decisions are made

- Decide when to replace Streamlit with a real frontend.
- Add users, private corpora, API keys, quotas, and share/export flows.
- `GET /admin/status` exposes no-secret local runtime status for job counts,
  failed jobs, corpus counts, artifact counts/bytes, runtime directory
  existence/writability/bytes, public model names, and disabled external
  provider/productization switches.
- Streamlit includes a local runtime status panel for common operational checks.
- Still planned: provider failure history beyond failed jobs, real token spend,
  durable storage dashboards, and user/workspace admin once identity exists.

Acceptance:

- Multiple corpora can coexist without leaking documents or generated artifacts.
- User-facing workflows are not tied to local server filesystem assumptions.
- Operational state is inspectable without SSH for common local runtime
  questions.
