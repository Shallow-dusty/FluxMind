# FluxMind Architecture

Last updated: 2026-05-29

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
  rendering, browser-translation guard.
- `api.py`: FastAPI request contract, token verification, lifecycle startup.
- `src/chain.py`: RAG prompt, retrieval, non-streaming answer generation, and
  reasoning-aware streaming.
- `src/ingestion.py`: PDF discovery, upload name safety, PyMuPDF extraction,
  chunking, FAISS persistence, active paper selection.
- `src/embeddings.py`: local sentence-transformers embedding model factory.
- `src/capabilities.py`: provider-neutral future contracts for image
  generation and isolated Python/Octave/MATLAB-compatible execution.
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

## Next Architecture Step

The next implementation step should split long-running work into explicit jobs:

```text
API request
  -> create job row
  -> worker executes parse/index/generate/run
  -> artifacts stored by URI
  -> UI polls or subscribes to job status
```

That job boundary is the prerequisite for reliable PDF indexing, image
generation, Python/Octave execution, quotas, cancellation, retries, and
observability.

Implementation work packages are tracked in `docs/BACKLOG.md`.
