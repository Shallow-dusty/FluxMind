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

Status: planned

- Create a small evaluation set of control-engineering questions.
- Record expected source papers/pages for citation-sensitive questions.
- Add citation validation: every cited source must map to retrieved metadata.
- Add provider failure fixtures for timeout, 429, empty output, and malformed
  streaming chunks.
- Add answer modes: explanation, derivation, implementation, literature review,
  and code generation.

Acceptance:

- Evaluation command runs without network where fixtures are available.
- Citation regressions fail locally before deployment.
- Provider errors surface as structured user-facing messages.

## WP2: Corpus and Storage Layer

Status: planned

- Add durable metadata for papers, chunks, corpora, jobs, and artifacts.
- Store upload ownership, checksum, title, source path, indexed status, and
  parse/index errors.
- Decide storage path: local volume first or object storage plus relational DB.
- Separate uploaded PDFs from active indexed corpus with explicit activation.

Acceptance:

- Rebuilding an index is a job with status and logs.
- A paper can be indexed, deactivated, reactivated, and traced to source chunks.
- Storage state can be listed without reading raw filesystem layout manually.

## WP3: Job System

Status: local synchronous job model implemented; async queue and true running
cancellation remain planned

- Local JSONL job records exist in `src/jobs.py`.
- `POST /jobs/image/mock` creates a mock image-generation job.
- `POST /jobs/code/python-local` creates a development-only Python execution
  job.
- `POST /jobs/index/rebuild` rebuilds the FAISS index from selected PDFs as a
  persisted job.
- `GET /jobs` lists latest jobs.
- `GET /jobs/{job_id}` returns persisted job status.
- `POST /jobs/{job_id}/retry` retries failed/cancelled local jobs with a new
  job ID.
- `POST /jobs/{job_id}/cancel` records cancellation for queued/running job
  states.
- Streamlit sidebar can trigger selected-PDF index rebuild jobs, mock image
  jobs, local Python jobs, and display latest job status.
- Still planned: async queue, true cancellation of running work, retry
  scheduler/backoff, richer timeout policy, and richer UI controls for
  cancellation/retry.

Acceptance:

- Job records preserve request, result, artifacts, errors, attempts, and
  request IDs.
- Failed code-execution jobs preserve stderr/error details.
- API and Streamlit can show running/succeeded/failed states through job status.
- Remaining: long-running work still needs a real async runner before it stops
  blocking request handlers.

## WP4: Image and Diagram Generation

Status: planned; implement provider-neutral plumbing and a no-key mock/local
provider first, keep real image-provider activation disabled until a key/account
is configured

- `MockImageGenerationProvider` implements the `ImageGenerationProvider`
  contract with deterministic local SVG output.
- Start with engineering diagrams and paper-figure redrafts.
- Store prompt, provider, model, size, source references, output URI, and cost
  metadata.
- Keep generated images as artifacts rather than inline chat-only blobs.

Acceptance:

- A request can generate an artifact record with a stable URI.
- Generated mock diagrams produce persisted artifact URIs.
- Remaining: generated diagrams can be referenced from answers and exported.
- Provider can be swapped without changing the UI flow.

## WP5: Code Execution

Status: planned; implement request/result plumbing and sandbox boundary first,
keep real hosted execution and MATLAB activation disabled until infrastructure
and license/account decisions are made

- `LocalPythonExecutionProvider` implements the `CodeExecutionProvider`
  contract for development-only Python snippets.
- Start with Python numerical snippets and generated plots.
- Add GNU Octave before considering real MATLAB.
- Run code in an isolated service with CPU, memory, timeout, filesystem, and
  network limits.
- Capture stdout, stderr, exit code, plots, and files as artifacts.

Acceptance:

- User code never runs inside the Streamlit or FastAPI process.
- Execution results are reproducible from stored input files and environment
  metadata.
- A failed local execution returns structured diagnostics without breaking the
  app.

## WP6: Product Shell

Status: planned; implement local/admin foundations first, keep public identity,
API-key lifecycle, quotas, and billing disabled until decisions are made

- Decide when to replace Streamlit with a real frontend.
- Add users, private corpora, API keys, quotas, and share/export flows.
- Add admin views for queue health, provider errors, token spend, and storage.

Acceptance:

- Multiple corpora can coexist without leaking documents or generated artifacts.
- User-facing workflows are not tied to local server filesystem assumptions.
- Operational state is inspectable without SSH for common questions.
