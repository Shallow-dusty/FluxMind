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
  reasoning-aware streaming, answer modes, and numbered citation validation.
- `src/ingestion.py`: PDF discovery, upload name safety, PyMuPDF extraction,
  chunking, FAISS persistence, active paper selection, and paper metadata
  refresh.
- `src/metadata.py`: local JSON corpus metadata registry for selectable papers,
  checksums, active/indexed state, chunk counts, and parse/index error fields.
- `src/embeddings.py`: local sentence-transformers embedding model factory.
- `src/capabilities.py`: provider-neutral future contracts for image
  generation and isolated Python/Octave/MATLAB-compatible execution.
- `src/providers.py`: no-key local providers for artifact storage, mock SVG
  diagram generation, and development-only Python execution.
- `src/jobs.py`: local JSONL job records, immediate runner, and in-process
  background queue for mock image generation, development-only Python
  execution, and selected-PDF index rebuilds.
- `src/evaluation.py`: offline RAG fixture evaluation, provider-error fixture
  checks, and citation validation helpers.
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
  -> create JSONL job record
  -> enqueue or run local no-key provider
  -> persist result/artifact/error
  -> expose status through GET /jobs and GET /jobs/{job_id}
```

The local runner also supports retrying failed/cancelled jobs and marking
queued/running records as cancelled. The async queue is process-local and
JSONL-backed; it is useful for no-key development, but it is not a durable
multi-worker platform queue. Running local Python jobs observe cancellation.
Index rebuild jobs currently check cancellation before execution starts, so
mid-rebuild cancellation remains a later worker/storage concern. The Streamlit
sidebar can trigger selected-PDF index jobs, mock SVG image jobs, local Python
jobs, and display recent job status. Real external providers can be attached
later without changing the UI/API workflow.

Implementation work packages are tracked in `docs/BACKLOG.md`.

## RAG Quality Gate

The first RAG quality gate is intentionally offline. `eval/rag_baseline.json`
stores domain questions, answer modes, expected source/page references, fixture
answers, and provider failure fixtures. `scripts/evaluate_rag.py` validates
that fixture answers only cite retrieved context refs and that provider errors
normalize to stable user-facing codes. This does not prove live answer quality
yet; it establishes the regression harness that later live or recorded model
answers can plug into without changing the deployment gate.

## Corpus Metadata

The first corpus storage boundary is `metadata/corpus.json`, managed through
`src.metadata.CorpusMetadataStore`. It records selectable papers with source
path, checksum, manifest title fields, source kind, active flag, indexed status,
chunk count, and parse/index error slots. Upload and selected-PDF rebuild flows
update this file, and FastAPI exposes it through `GET /corpus/papers`.

This is still a local development store. It makes corpus state inspectable
without reading the filesystem manually, but it is not the future multi-user
metadata database or object storage layer.
