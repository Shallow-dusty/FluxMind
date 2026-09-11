# FluxMind Architecture

> Last updated: 2026-07-27. Product scope and roadmap are owned by
> [DEVELOPMENT.md](../../DEVELOPMENT.md).

## System shape

FluxMind has two entry points and one RAG core:

```text
Streamlit UI (app.py, :18501)       FastAPI (api.py, :18502)
             │                                  │
             └──────────────┬───────────────────┘
                            ▼
                      src/chain.py
              retrieval → rerank → generation
                            │
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
       metadata           jobs              artifacts
     JSON + SQLite    JSONL + SQLite      files + SQLite
```

The production worker is a third process:

```text
scripts/run_job_worker.py
        │
        ├─ claims queued jobs with a lease
        ├─ runs index rebuild / image / Python / Octave work
        └─ persists results and artifacts
```

The deployment is deliberately local and small-group oriented. There is no
tenant layer, external identity service, billing ledger, or distributed queue.

## RAG pipeline

`src/chain.py` owns all retrieval and answer generation.

1. `hybrid_retrieve()` combines:
   - FAISS vector candidates;
   - keyword candidates from the FAISS docstore;
   - source-level metadata from the curated manifest.
2. `rerank_documents()` applies deterministic BM25-lite scoring. A local
   CrossEncoder is used only when `RERANKER_MODEL` points to an existing local
   model directory.
3. The prompt requires inline numbered citations such as `[1]`.
4. Citation validation checks cited numbers against returned context. An answer
   with retrieved context but no valid numbered citation fails validation.

Entry-point mapping:

```text
UI streaming          query_stream()
POST /query           query_with_metadata()
POST /query/inspect   query_with_metadata()
POST /query/retrieve  retrieve_with_metadata()
POST /query/report    query_with_metadata()
```

## Corpus and indexing

The bundled library lives under `papers/library/` and is reconstructed with
`scripts/import_seed_papers.py`. PDFs and runtime index files are gitignored.

Important state:

```text
papers/library/manifest.json     curated source metadata
faiss_index/active_papers.json   current source selection
faiss_index/                     LangChain FAISS index
metadata/corpus.json             paper lifecycle metadata
metadata/corpus.sqlite3          current paper metadata mirror
metadata/chunks.sqlite3          searchable chunk metadata
metadata/corpus_profiles.json    reusable paper selections
```

Index rebuild is job-backed. Changing the active source selection marks the
runtime stale; applying a corpus profile with rebuild queues an `index_rebuild`
job instead of doing heavy work inside the UI or synchronous query route.

`src/embeddings.py` loads the configured HuggingFace semantic model for both
index build and query. FluxMind does not synthesize hash vectors when model
loading fails because those vectors are incompatible with the persisted FAISS
index. A new index build with no PDF documents also fails explicitly instead of
inserting a placeholder document.

## Users and query history

`src/users.py` provides the small-group account model:

- local `admin` and `student` roles;
- PBKDF2 password hashes in `metadata/users.sqlite3`;
- first-run administrator creation;
- administrator account create/update/disable/password reset;
- per-user query history.

The Streamlit UI requires login. Students can query, inspect their own history,
and use research tools; their jobs and artifacts are tagged with their account
and filtered to that owner. Administrators can see all jobs/artifacts and use
corpus mutation and operational controls.

The API keeps a separate deployment-level shared token:

```text
FLUXMIND_API_TOKEN
Authorization: Bearer <token>
X-API-Key: <token>
```

If no API token is configured, the API remains open for local development. A
`user_id` supplied to `/query`, `/query/inspect`, or `/query/report` must
identify an active local account before the generated answer is recorded in
history.

## Jobs and artifacts

`src/jobs.py` is a local durable job system:

- append-only `jobs/jobs.jsonl`;
- SQLite current-state mirror `jobs/jobs.sqlite3`;
- idempotency keys;
- queue timeout and deadline;
- worker lease and recovery;
- retry, scheduled backoff, cancellation, and dead-letter state.

It is not a distributed queue. The explicit worker process allows production to
move image generation, code execution, and index rebuild work outside Streamlit
and synchronous query requests.

`src/artifacts.py` derives artifact records from persisted jobs, mirrors them in
`artifacts/artifacts.sqlite3`, verifies byte count/checksum metadata when
available, and exports local files beneath `ARTIFACTS_DIR`.

Job detail responses and the Streamlit recent-job panel expose the submitted
request, execution result, stdout/stderr, errors, logs, and artifact metadata so
researchers can debug and reuse a run. Idempotency keys and account ownership
remain visible for local debugging. Artifact file URIs remain internal;
downloads resolve by artifact ID and use the artifact title as the filename.

## Provider boundaries

Provider-neutral request contracts live in `src/capabilities.py`; concrete
implementations live in `src/providers.py`.

- Image: deterministic mock or OpenAI-compatible image generation.
- Code: local subprocess or Docker-backed Python/Octave.
- LLM: OpenAI-compatible chat model through LangChain.

`src/execution_policy.py` applies the local Python import/syntax policy before
execution. `src/provider_guard.py` is the optional practical cost boundary: it
can reject a request that exceeds configured provider token or cost ceilings
before a provider client is called.

Generated files become job artifacts. Provider work never runs implicitly
during status collection.

## Runtime events and operations

`metadata/runtime_events.jsonl` records events that help debug the running
system. Query and job events retain local account ownership:

- `retrieval_trace`;
- `query_usage`;
- `provider_failure`;
- `code_execution`;
- `upload_scan`;
- retention actions.

`src/admin.py` aggregates only operational questions:

- corpus/index drift;
- queue and worker state;
- artifact integrity;
- local account counts;
- recent query, citation, execution, and provider failures;
- runtime directory sizes.

FastAPI exposes `/admin/status`, `/admin/status/report`, `/admin/metrics`,
`/admin/events`, retention preview/delete, and runtime backup/restore checks.
The Streamlit admin expander shows the same runtime state.

`src/storage_manifest.py` inventories paths, sizes, and hashes for checking
whether gitignored local state has been backed up. Restore-check commands
compare state but do not copy it.

## Configuration

`src/config.py` loads `.env` from the project root. Paths are anchored to
`PROJECT_ROOT`; secrets belong only in `.env`.

The most important switches are:

```text
LLM_* / OPENAI_*                 LLM endpoint and model
IMAGE_PROVIDER_BACKEND           mock or openai
CODE_EXECUTION_BACKEND           local or docker
FLUXMIND_API_TOKEN               optional shared API token
FLUXMIND_USER_STORE_FILE         optional account DB override
PROVIDER_QUOTA_GUARD_ENABLED     optional provider cost/token ceiling
RETENTION_DELETE_ENABLED         enables explicit retention deletion
```

## Deployment

Trace-Twin runs:

```text
fluxmind-ui.service       streamlit, :18501
fluxmind-api.service      uvicorn, :18502
fluxmind-worker.service   durable local worker
```

Cloudflare Tunnel exposes the UI and API. `scripts/deploy_sync.py` is dry-run by
default; `--apply --restart` is required to synchronize and restart production.

## Verification

The supported gates are intentionally short:

```bash
python -m pytest
python scripts/evaluate_rag.py
python scripts/health_check.py
```

Provider smokes are explicit because they spend quota or require Docker:

```bash
IMAGE_PROVIDER_BACKEND=openai python scripts/openai_image_smoke.py
CODE_EXECUTION_BACKEND=docker python scripts/docker_execution_smoke.py --language all
```
