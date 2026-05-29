# FluxMind Implementation Backlog

Last updated: 2026-05-30

This backlog turns the platform roadmap into concrete implementation packages.
It is intentionally ordered by dependency, not by excitement.

## WP0: Stabilize Current Production

Status: in progress

- Keep CI green: `python -m pytest` and `python scripts/health_check.py`.
- Keep Trace-Twin health green with:
  `python scripts/health_check.py --ssh-host root@100.100.233.26`.
- Add provider-error normalization for UI and API responses.
- Add request IDs to `/query` and Streamlit logs.
- Add recent journal error checks to `scripts/health_check.py`.
- Add active paper count and FAISS index size checks to `scripts/health_check.py`.

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

Status: deferred until a provider/sandbox implementation is selected

- Introduce job records for PDF parsing, indexing, image generation, and code
  execution.
- Add retry, cancellation, timeout, and artifact URI fields.
- Expose job status to API and UI.

Acceptance:

- Long-running work no longer blocks the request handler.
- Failed jobs preserve useful stderr/error details.
- The UI can show queued/running/succeeded/failed states.

## WP4: Image and Diagram Generation

Status: deferred; requires a configured image provider key/account

- Implement the `ImageGenerationProvider` contract.
- Start with engineering diagrams and paper-figure redrafts.
- Store prompt, provider, model, size, source references, output URI, and cost
  metadata.
- Keep generated images as artifacts rather than inline chat-only blobs.

Acceptance:

- A request can generate an artifact record with a stable URI.
- Generated diagrams can be referenced from answers and exported.
- Provider can be swapped without changing the UI flow.

## WP5: Code Execution

Status: deferred; requires isolated execution infrastructure and, for MATLAB,
license/account decisions

- Implement the `CodeExecutionProvider` contract.
- Start with Python numerical snippets and generated plots.
- Add GNU Octave before considering real MATLAB.
- Run code in an isolated service with CPU, memory, timeout, filesystem, and
  network limits.
- Capture stdout, stderr, exit code, plots, and files as artifacts.

Acceptance:

- User code never runs inside the Streamlit or FastAPI process.
- Execution results are reproducible from stored input files and environment
  metadata.
- A failed execution returns structured diagnostics without breaking the app.

## WP6: Product Shell

Status: deferred; requires identity/API-key/quota decisions

- Decide when to replace Streamlit with a real frontend.
- Add users, private corpora, API keys, quotas, and share/export flows.
- Add admin views for queue health, provider errors, token spend, and storage.

Acceptance:

- Multiple corpora can coexist without leaking documents or generated artifacts.
- User-facing workflows are not tied to local server filesystem assumptions.
- Operational state is inspectable without SSH for common questions.
