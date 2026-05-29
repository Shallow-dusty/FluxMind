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
- `api.py`: FastAPI request contract, token verification, lifecycle startup.
- `src/chain.py`: RAG prompt, retrieval, non-streaming answer generation, and
  reasoning-aware streaming.
- `src/ingestion.py`: PDF discovery, upload name safety, PyMuPDF extraction,
  chunking, FAISS persistence, active paper selection.
- `src/embeddings.py`: local sentence-transformers embedding model factory.
- `src/capabilities.py`: provider-neutral future contracts for image
  generation and isolated Python/Octave/MATLAB-compatible execution.
- `src/providers.py`: no-key local providers for artifact storage, mock SVG
  diagram generation, and development-only Python execution.
- `src/jobs.py`: local JSONL job records and synchronous no-key job runner for
  mock image generation, development-only Python execution, and selected-PDF
  index rebuilds.
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

The first job boundary now exists as a local synchronous runner:

```text
API request
  -> create JSONL job record
  -> run local no-key provider
  -> persist result/artifact/error
  -> expose status through GET /jobs and GET /jobs/{job_id}
```

The local runner also supports retrying failed/cancelled jobs and marking
queued/running records as cancelled. The next step is to make this runner
asynchronous, add actual cancellation of running work, and improve observability.
The Streamlit sidebar can already trigger selected-PDF index jobs, mock SVG
image jobs, local Python jobs, and display recent job status. Real external
providers can be attached later without changing the UI/API workflow.

Implementation work packages are tracked in `docs/BACKLOG.md`.
