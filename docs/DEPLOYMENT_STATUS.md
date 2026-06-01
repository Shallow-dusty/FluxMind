# FluxMind Deployment Status

Last live check: 2026-06-02 01:55 CST

This document records the current deployment snapshot. Treat it as a
pointer for re-checking the live host, not as proof that the service is still
healthy at a later time.

## Current Deployment

Workspace directory: `11.FluxMind/`
Last synced source baseline: local working tree based on `55d16a8` with
uncommitted no-key platform changes. `/opt/fluxmind` is not a git checkout, so
the live deployment should be treated as a synchronized source tree rather than
a deployed commit hash.

```
Host          Trace-Twin
Tailscale     root@100.100.233.26
Public IP     223.6.253.9
Deploy root   /opt/fluxmind
Runtime user  fluxmind
UI service    fluxmind-ui.service
API service   fluxmind-api.service
Worker service fluxmind-worker.service
UI port       18501
API port      18502
```

Public endpoints:

- Preferred Web UI: `https://smy.hyper-dusty.cloud/`
- Preferred API:    `https://api-smy.hyper-dusty.cloud/`
- Web UI (raw):     `http://223.6.253.9:18501/`
- API health (raw): `http://223.6.253.9:18502/health`

Both HTTPS hostnames terminate at Cloudflare and tunnel back to the same origin
ports through `fluxmind-smy` (one tunnel, two ingress rules). The raw HTTP IP
endpoints stay reachable for diagnostics, but Coze / third-party agent
integration should use the HTTPS hostname so the OpenAPI schema is fetched over
TLS.

API calls to `/query` require `X-API-Key: <token>` (or the equivalent
`Authorization: Bearer <token>`). The token is stored only on the server in
`/opt/fluxmind/.env`; do not copy it into this repository.

Local runtime state directories are owned by the `fluxmind` runtime user:
`/opt/fluxmind/metadata`, `/opt/fluxmind/jobs`, and
`/opt/fluxmind/artifacts`.

## Isolation Boundary

FluxMind is deployed separately from the Trace-Twin bot stack:

- no Docker deployment for FluxMind
- no Docker restart during deployment
- no changes to `/opt/trace-twin`
- independent systemd services
- independent ports `18501` and `18502`
- independent Cloudflare Tunnel service `cloudflared-fluxmind-smy.service`

The existing bot containers checked healthy at the last verification:

- `bot-resume`
- `bot-lingju`

## Runtime Configuration

The deployed `.env` currently uses:

```
LLM_BASE_URL=https://token-plan-sgp.xiaomimimo.com/v1
LLM_MODEL=mimo-v2.5-pro
EMBEDDING_MODEL=/opt/fluxmind/models/all-MiniLM-L6-v2
```

`mimo-v2.5-pro` is a reasoning model: it emits `reasoning_content` first
and the final answer second. `src/chain.py::query_stream` exposes the
reasoning as a `> 💭` blockquote, then a horizontal rule, then the answer,
so the Streamlit UI no longer looks frozen during the thinking phase. The
Streamlit layer uses a stable placeholder plus browser-translation guards,
because Chrome/Google Translate can mutate streamed text DOM nodes while the
frontend is still updating them. The non-streaming `/query` endpoint returns
the final answer only.

Previous pool `api.268526.eu.cc` with `deepseek-v4-flash` was retired on
2026-05-12 after it began returning `upstream_empty_output` (HTTP 429)
on every call. The Xiaomi MiMo pool (`token-plan-sgp.xiaomimimo.com`)
replaced it.

The sentence-transformers embedding model was copied to the server under
`/opt/fluxmind/models/all-MiniLM-L6-v2`, so normal service startup should not
depend on downloading from Hugging Face.

## Last Verification

Live checks refreshed on 2026-06-02 01:55 CST after syncing the local source
tree based on `55d16a8` plus uncommitted no-key platform changes to
`/opt/fluxmind`, restarting `fluxmind-api.service` and `fluxmind-ui.service`,
installing/enabling `fluxmind-worker.service`,
and verifying that the service still exposes the existing FAISS index, chunk
metadata, authenticated generated-answer citation inspection route, admin corpus
index freshness state, corpus lifecycle status, no-secret query usage estimates,
Markdown query report export, no-secret Markdown admin status report, local
artifact byte-count/SHA-256 checksum metadata plus admin artifact integrity
status, no-secret JSON eval report export, checksum-based uploaded-PDF
deduplication, corpus paper metadata filtering, reusable local corpus profiles,
no-delete retention preview, job transition logs, and no-secret execution
environment/policy metadata. The latest corpus-chunk refresh added local
`source_path`, `page`, and `q` filters for `GET /corpus/chunks`;
authenticated remote smoke matched chunk `111d3970eb3d0d1d86115837` from
`papers/library/arxiv-2510-18420-smc-pmsm-review.pdf` with `page=1`,
`q=Sliding-Mode`, `filtered_count=2`, and `missing_filter_count=0`. The latest runtime-event refresh added filtered
`GET /admin/events` inspection plus a Streamlit runtime event viewer for
provider-failure and query-usage history; authenticated remote smoke created
event `070664f3ced8`, matched it through `kind/code/q`, and verified an
unmatched query-usage filter returned zero results. The latest job-inspection refresh added local job
filters by query, status, and kind in both `GET /jobs` and the Streamlit recent
job panel; authenticated remote smoke created mock image job `f219605a4f28` and
verified `q/status/kind` filters returned that job while an unmatched queued
filter returned zero results. The latest artifact-gallery refresh added local
artifact filters by query, artifact kind, and job kind in both `GET /artifacts`
and the Streamlit gallery; authenticated remote smoke created mock image job
`d8abf4a5c36e`, filtered artifact `dbf8747af5f926d3`, and verified an
unmatched plot filter returned zero results. The latest product-shell refresh exposed the
no-delete upload/artifact retention preview in the Streamlit admin panel with
local day/limit controls. The latest corpus-profile refresh added
`metadata/corpus_profiles.json`, API routes to list/upsert/activate profiles,
read-only profile status inspection, queued profile rebuild, and a Streamlit
sidebar profile panel; the follow-up profile-report refresh added authenticated
`GET /corpus/profiles/{profile_id}/report` for no-secret Markdown status export,
shared the profile status/report formatter through `src.admin`, and exposed the
same no-secret Markdown report as a Streamlit profile-panel download. Remote
source grep found `corpus_profile_report_download` in `/opt/fluxmind/app.py`,
and authenticated API smoke exported `smoke-active-core` with stable report content. A
concurrent profile-report smoke issued 8 report requests successfully after
installing atomic JSON writes for corpus/profile metadata. The old direct JSON
write path exposed one transient `JSONDecodeError` during concurrent metadata
refresh; `src.metadata.atomic_write_json()` now writes through a same-directory
temp file and atomically replaces the JSON file. Post-fix journal inspection
since 2026-06-01 19:38 CST found no new JSONDecodeError/500 traceback.
Authenticated remote smoke created `smoke-active-core`, activated it, then
restored the 6-paper active corpus with `index.status=fresh`. A follow-up
read-only status smoke returned `profile_id=smoke-active-core`,
`available_papers=2`, `active_match=false`, `index_status=stale`, and
`rebuild_required=true` while leaving the active corpus at 6 papers and
`index.status=fresh`. The latest profile rebuild smoke created
`smoke-full-active`, queued job `9be792ffe78d`, finished with
`status=succeeded`, rebuilt 512 chunks, and returned profile/index status
`fresh` without changing the 6-paper active corpus. The latest
execution metadata refresh added language, entrypoint, input-file counts/bytes,
provider runtime, local runtime details, temporary workdir isolation, and
network-policy status to local Python/Octave execution results; authenticated
remote Python smoke returned `provider_runtime=python-local`,
`python_version=3.12.3`, `filesystem_isolation=temporary_workdir`, and
`network_policy_enforced=false`. The refresh also
installed queued job deadlines for async jobs and
scheduled retries: `queue_timeout_s` produces `deadline_at`, and expired queued
jobs fail before execution with `job_deadline_exceeded`. The latest job-system
refresh added durable worker lease metadata for queued jobs:
`worker_id`, `leased_at`, and `lease_expires_at` are persisted through the local
SQLite/JSONL job store before provider execution, expired queued leases can be
reclaimed, and the current in-process async worker uses the same claim path.
Remote isolated smoke returned `remote_lease_smoke=ok` against a temporary job
store under `/tmp`, without touching production `jobs/` state. The latest
durable-worker refresh added `LocalDurableJobWorker` plus
`scripts/run_job_worker.py` as an explicit no-key local worker entrypoint that
can claim and execute due queued jobs outside the API/Streamlit process. Remote
isolated smoke returned `remote_durable_worker_smoke=ok job_id=5e643fd84102
status=succeeded` from a temporary job store under `/tmp`; the script is present
on the server but is not auto-started as a production worker service. A
follow-up durable-worker cancellation refresh added polling of durable job state
while local providers run, so marking the job `cancelled` sets the provider
cancel event outside the API process. Remote isolated smoke returned
`remote_durable_cancel_smoke=ok job_id=9d4271330d36 status=cancelled` from a
temporary job store under `/tmp`. The latest worker-service refresh added
`deploy/systemd/fluxmind-worker.service`, installed it to
`/etc/systemd/system/fluxmind-worker.service`, enabled it, and verified it as
active. A production durable-store smoke enqueued job `296d2d4aab16` directly
into `/opt/fluxmind/jobs`; `fluxmind-worker.service` claimed it as
`worker_id=fluxmind-worker-1` and completed it with
`remote_worker_service_smoke=ok`. The latest execution-sandbox readiness
refresh added `CODE_EXECUTION_BACKEND` and `DOCKER_EXECUTION_IMAGE` config
switches plus no-secret Docker readiness reporting in `/admin/status`; deployed
admin status returned `backend=local`, `configured=false`, `available=false`,
`docker_executable=docker`, `image=python:3.12-slim`, and
`reason=not_configured`, so Docker is visible on the host but container
execution is not silently enabled for the `fluxmind` runtime user. The latest
retrieval-diagnostics refresh added `src.chain.retrieve_with_metadata()` and
authenticated `POST /query/retrieve` so deployed source/page context refs can be
inspected without calling the LLM provider; authenticated remote smoke returned
`context_count=5`, `ok=true`, `missing_source_page_refs=[]`, and first source
`papers/library/arxiv-2510-18420-smc-pmsm-review.pdf`. A follow-up eval refresh
added `scripts/evaluate_rag.py --retrieval-url` and JSON report coverage for
live retrieval scoring; remote eval returned live_retrieval 3/3 ok with
`context_coverage=1.00` for all configured cases and wrote a no-secret report
at `/tmp/fluxmind-retrieval-eval.json`. The latest RAG refresh
upgraded the local keyword side of hybrid retrieval and final context reranking
to deterministic BM25-lite scoring over chunk text and metadata. A follow-up
RAG refresh added optional no-key local CrossEncoder reranking when
`RERANKER_MODEL` points to an existing local model path; empty or missing paths
fall back to BM25-lite without runtime downloads. Remote admin status currently
reports `reranker_model_configured=false` and `reranker_model_available=false`,
so the deployed service is still using the BM25-lite fallback. The latest
corpus refresh added bibliographic enrichment fields for DOI, arXiv ID, venue,
and topic tags, with SQLite migration for existing `metadata/corpus.sqlite3`
tables. It also added `/corpus/status` for no-secret corpus lifecycle reporting.
The latest product-shell refresh added no-secret query usage events for
successful `/query`, `/query/inspect`, and `/query/report` calls; these store
character counts and rough token estimates, not prompt/answer text or provider
billing data. A follow-up query-usage refresh extracts provider token usage from
LangChain response metadata when available, stores no-secret provider
prompt/completion/total token counts beside the estimates, and aggregates those
fields in admin status. Remote smoke verified `provider_usage_from_response()`
on the server venv and confirmed `/admin/status` exposes `provider_total_tokens`
and `provider_usage_events` fields. It also added `/query/report` for Markdown
exports containing the answer, citation validation, and retrieved context refs.
The latest upload-metadata refresh added best-effort no-key extraction
from embedded PDF metadata plus first-page title, author, DOI/arXiv, year, and
keyword/index-term text for uploaded or otherwise unmanifested PDFs. Remote
temp-PDF smoke extracted title `Remote Metadata Smoke for Flux Observer`,
authors `Dana Smith, Eli Zhang and Finn Rao`, year `2026`, and topic tags
`flux observer`, `sliding mode`, and `PMSM`. The latest local-execution refresh
added workdir path containment for Python/Octave input files and entrypoints,
skips symlink or out-of-workdir outputs during artifact export, and caps input
file count, per-file bytes, and total input bytes before materialization. It also
installs Unix child-process CPU-time limits where supported. Earlier refreshes
installed no-secret `/admin/status/report` Markdown export, optional live
`/query/inspect` regression scoring, expanded
source-diverse retrieval, numbered-citation prompt guards, out-of-range citation
neutralization, and cancellable index-rebuild checkpoints for PDF
loading/splitting and pre-commit state updates. Earlier in the same refresh
window,
one rebuild job exposed a deployment permission bug:
the old rebuild path deleted the entire `faiss_index` directory, which required
write access to `/opt/fluxmind`. `src/ingestion.py` now rebuilds through a
temporary directory under `faiss_index` and only replaces the live index after a
successful save. `api.py` also warms only an already-present FAISS index on
startup; it does not synchronously rebuild a missing/corrupt index before
binding the API port. Later documentation-only changes may be synced without
another service restart; use this document plus live checks rather than a remote
git commit to reason about deployment state.

```
fluxmind-ui.service     active
fluxmind-api.service    active
fluxmind-worker.service active
cloudflared service     active
docker.service          active
UI listener             0.0.0.0:18501
API listener            0.0.0.0:18502
local API health        {"status":"ok"}
Cloudflare UI HTTP      200 at https://smy.hyper-dusty.cloud/
Cloudflare API HTTP     200 at https://api-smy.hyper-dusty.cloud/health
public UI HTTP          200 at http://223.6.253.9:18501/
public API health HTTP  200 at http://223.6.253.9:18502/health
deployed source tree     /opt/fluxmind is not a git checkout; source was rsynced
deployed stream guard   present in /opt/fluxmind/app.py
deployed job panel      present in /opt/fluxmind/app.py
deployed capabilities   present in /opt/fluxmind/src/capabilities.py
deployed runtime layer   present in /opt/fluxmind/src/runtime.py
deployed no-key providers present in /opt/fluxmind/src/providers.py
local Octave provider    present in /opt/fluxmind/src/providers.py; gnu-octave-local metadata installed
execution metadata       present in /opt/fluxmind/src/providers.py; provider runtime limits/reporting installed
execution env metadata   present in /opt/fluxmind/src/providers.py; remote Python smoke returned provider_runtime=python-local, python_version=3.12.3, filesystem_isolation=temporary_workdir, network_policy_enforced=false
execution CPU limits     present in /opt/fluxmind/src/providers.py; remote busy-loop smoke returned cpu_limit_enforced=true
execution path containment present in /opt/fluxmind/src/providers.py; remote smoke rejected ../ escape with exit_code=2
execution input limits   present in /opt/fluxmind/src/providers.py; remote smoke rejected >256KiB input with exit_code=2
timeout diagnostics      present in /opt/fluxmind/src/jobs.py; exit 124 is classified as execution_timeout
safe FAISS rebuild       present in /opt/fluxmind/src/ingestion.py; temp save replaces live index only after success
cancellable index rebuild present in /opt/fluxmind/src/ingestion.py and /opt/fluxmind/src/jobs.py; cancelled rebuilds raise before publishing post-cancel state
uploaded metadata extract present in /opt/fluxmind/src/ingestion.py; remote temp-PDF smoke extracted title/authors/year/topic_tags from first-page text, and existing DOI/arXiv extraction remains covered by local tests
uploaded PDF dedup      present in /opt/fluxmind/src/ingestion.py; isolated remote smoke reused existing.pdf with chunk_count=7 and wrote no duplicate file
startup warmup only      present in /opt/fluxmind/api.py; API startup does not call build_vector_store
deployed job layer       present in /opt/fluxmind/src/jobs.py
SQLite job mirror        present in /opt/fluxmind/src/jobs.py
scheduled retry/backoff  present in /opt/fluxmind/src/jobs.py
queued job recovery      present in /opt/fluxmind/src/jobs.py; API startup calls recover_queued_jobs
queued job deadlines     present in /opt/fluxmind/src/jobs.py; expired queued jobs fail with job_deadline_exceeded
worker lease bridge      present in /opt/fluxmind/src/jobs.py; remote isolated temp-store smoke returned remote_lease_smoke=ok
durable worker loop      present in /opt/fluxmind/src/jobs.py and /opt/fluxmind/scripts/run_job_worker.py; remote isolated temp-store smoke returned remote_durable_worker_smoke=ok
durable worker cancellation present in /opt/fluxmind/src/jobs.py; remote isolated temp-store smoke returned remote_durable_cancel_smoke=ok
worker systemd service   enabled and active; ExecStart uses scripts/run_job_worker.py --forever --worker-id fluxmind-worker-1; production store smoke returned remote_worker_service_smoke=ok job_id=296d2d4aab16
execution sandbox readiness present in /opt/fluxmind/src/providers.py and /opt/fluxmind/src/admin.py; admin status returned backend=local configured=false available=false reason=not_configured docker_executable=docker image=python:3.12-slim
admin queue health       present in /opt/fluxmind/src/admin.py; authenticated API returned queue_health
deployed artifact layer  present in /opt/fluxmind/src/artifacts.py
deployed admin layer     present in /opt/fluxmind/src/admin.py
admin status report      present in /opt/fluxmind/src/admin.py and /opt/fluxmind/api.py; authenticated smoke returned text/markdown
admin query usage        present in /opt/fluxmind/src/admin.py, /opt/fluxmind/src/chain.py, and /opt/fluxmind/api.py; query events keep estimated token counts and can include provider prompt/completion/total token counts when the upstream response exposes them; remote smoke returned provider_total_tokens=0 provider_usage_events=0 before any provider-usage-bearing live event
deployed metadata layer  present in /opt/fluxmind/src/metadata.py
corpus SQLite mirror     present in /opt/fluxmind/src/metadata.py; paper metadata mirrors into metadata/corpus.sqlite3
paper enrichment fields  present in /opt/fluxmind/src/metadata.py; DOI/arXiv/venue/topic_tags columns migrated into metadata/corpus.sqlite3
atomic metadata writes   present in /opt/fluxmind/src/metadata.py; corpus/profile JSON writes use temp-file atomic replace, and post-fix concurrent report smoke completed 8/8 requests
chunk SQLite mirror      present in /opt/fluxmind/src/metadata.py; indexed chunk metadata can mirror into metadata/chunks.sqlite3
chunk source listing     present in /opt/fluxmind/src/metadata.py; chunk source paths can be compared with active corpus
deployed eval layer      present in /opt/fluxmind/src/evaluation.py
source/page eval check   present in /opt/fluxmind/src/evaluation.py; expected refs are verified against actual PDFs
live RAG eval gate       present in /opt/fluxmind/src/evaluation.py and scripts/evaluate_rag.py --live-url
live retrieval eval gate present in /opt/fluxmind/src/evaluation.py and scripts/evaluate_rag.py --retrieval-url; remote retrieval eval returned live_retrieval=3/3 context_coverage=1.00
eval JSON report         present in /opt/fluxmind/src/evaluation.py and scripts/evaluate_rag.py --json-report; remote report summary offline=3/3 provider=4/4 recorded=3/3 live_retrieval=3/3
recorded eval gate       present in /opt/fluxmind/eval/rag_baseline.json; recorded answers scored at coverage=1.00
hybrid retrieval         present in /opt/fluxmind/src/chain.py; query uses hybrid_retrieve
local BM25 reranker      present in /opt/fluxmind/src/chain.py; hybrid_retrieve uses bm25_relevance_scores
optional local reranker  present in /opt/fluxmind/src/chain.py; learned_rerank_documents uses local RERANKER_MODEL path only, and remote admin status returned reranker_model_configured=false reranker_model_available=false
source-diverse reranking present in /opt/fluxmind/src/chain.py; expanded candidates preserve multiple sources before TOP_K
generated citation check present in /opt/fluxmind/src/chain.py; QueryResult validates numbered refs against source/page context
retrieval diagnostics present in /opt/fluxmind/src/chain.py; retrieve_with_metadata returns no-LLM context refs and source/page completeness
numbered citation guard  present in /opt/fluxmind/src/chain.py; prompt lists valid context-ref range per answer
citation neutralizer     present in /opt/fluxmind/src/chain.py; out-of-range bracket refs are neutralized before validation
artifact references      present in /opt/fluxmind/src/chain.py; RAG prompt includes Generated Artifact References
artifact formatter       present in /opt/fluxmind/src/artifacts.py; stable artifact IDs can be cited
artifact checksum metadata present in /opt/fluxmind/src/providers.py; remote smoke generated checksum_sha256 and byte_count=2
artifact integrity status present in /opt/fluxmind/src/artifacts.py and /opt/fluxmind/src/admin.py; authenticated admin smoke returned checked=1, ok=1, unchecked=1, checksum_mismatch=0
mock image metadata      present in /opt/fluxmind/src/providers.py; local-mock-svg-v1 metadata installed
execution artifacts      local code job captured result.txt artifact
artifact export route    present; authenticated local API listed result.txt
artifact gallery         present in /opt/fluxmind/app.py; stable IDs, metadata, local filters, and downloads rendered; authenticated artifact filter smoke created job=d8abf4a5c36e artifact=dbf8747af5f926d3 filtered_count=1 missing_filter_count=0
admin status panel       present in /opt/fluxmind/app.py with no-delete retention preview UI
admin status route       present; authenticated local API returned runtime state and provider_failures
admin retention preview  present in /opt/fluxmind/api.py, /opt/fluxmind/src/admin.py, and /opt/fluxmind/app.py; authenticated smoke returned mode=preview delete_enabled=false limit=25 uploads=0 artifacts=0
admin query usage panel  present in /opt/fluxmind/app.py; status_query_usage rendered in Streamlit source
admin events route       present in /opt/fluxmind/api.py and /opt/fluxmind/app.py; runtime event filters available by kind/code/q; authenticated event filter smoke created event=070664f3ced8 filtered_count=1 missing_filter_count=0
admin index freshness    present in /opt/fluxmind/src/admin.py; authenticated local API returned corpus.index.status=fresh
provider failure history present in /opt/fluxmind/src/runtime.py; /query failures append no-secret runtime events
job SQLite state         /opt/fluxmind/jobs/jobs.sqlite3 exists; 17 current rows
scheduled retry smoke    queued retry executed; parent_job_id/not_before present
job transition logs      present in /opt/fluxmind/src/jobs.py and /opt/fluxmind/api.py; authenticated mock image smoke returned logs=[running,succeeded] with artifact_count=1
job retry/cancel UI      present in /opt/fluxmind/app.py with local q/status/kind filters; authenticated job filter smoke created job=f219605a4f28 filtered_count=1 missing_filter_count=0
offline RAG eval         passed in /opt/fluxmind
live retrieval eval      passed in /opt/fluxmind against local API; 3/3 cases context_coverage=1.00 without model generation
live RAG eval            passed in /opt/fluxmind against local API; 3/3 cases context_coverage=1.00 and answer_coverage=1.00
local cancellation tests  passed locally; remote venv has no pytest module, so server-side unit execution was not available
corpus metadata route    present in /opt/fluxmind/api.py
corpus metadata filters  present in /opt/fluxmind/api.py; authenticated smoke with q=sliding active=true source_kind=library indexed_status=indexed returned 4 matching papers
corpus profile routes    present in /opt/fluxmind/api.py and /opt/fluxmind/app.py; authenticated smoke created/activated smoke-active-core, restored 6 active papers with index.status=fresh, read-only status returned active_match=false index_status=stale rebuild_required=true, and profile rebuild smoke created smoke-full-active job=9be792ffe78d status=succeeded chunks=512 profile_index_status=fresh
corpus profile report    present in /opt/fluxmind/api.py, /opt/fluxmind/app.py, and /opt/fluxmind/src/admin.py; authenticated smoke exported smoke-active-core as text/markdown, Streamlit source grep found corpus_profile_report_download, and concurrent smoke returned 8 reports with identical length
active corpus route      present in /opt/fluxmind/api.py
corpus chunks route      present in /opt/fluxmind/api.py; source_path/page/q filters smoke matched chunk=111d3970eb3d0d1d86115837 filtered_count=2 missing_filter_count=0
remote health_check      includes API-level /corpus/chunks, /query/retrieve, and corpus profile report smoke; returned chunk_filter_smoke=111d3970eb3d0d1d86115837 filtered_count=2 missing_filter_count=0, retrieval_smoke=context_count=5 ok=True, and corpus_profile_report_smoke=smoke-active-core
corpus status route      present in /opt/fluxmind/api.py; authenticated API returned status=indexed, papers=6, index.status=fresh
corpus metadata papers   6 indexed papers via authenticated local API check
corpus metadata enrich   6 papers with DOI or arXiv metadata via authenticated local API check
active corpus smoke      PUT /corpus/active preserved 6 active papers; rebuild_required=true
job API routes           present in /opt/fluxmind/api.py
index rebuild job route  present in /opt/fluxmind/api.py
async index job route    present in /opt/fluxmind/api.py
Octave job routes        present in /opt/fluxmind/api.py; immediate and async routes installed
query answer mode        present in /opt/fluxmind/api.py
query retrieve route     present in /opt/fluxmind/api.py; authenticated smoke returned context_count=5 ok=true missing_source_page_refs=[] first_source_path=papers/library/arxiv-2510-18420-smc-pmsm-review.pdf
query inspect route      present in /opt/fluxmind/api.py; authenticated smoke returned citation_ok=True, cited_refs=[5], context_refs=5
query report route       present in /opt/fluxmind/api.py; authenticated smoke returned text/markdown and fluxmind-query-report.md
job retry route          present in /opt/fluxmind/api.py
scheduled retry route    present in /opt/fluxmind/api.py
admin status jobs        17 total, 8 failed historical/smoke/old-rebuild jobs, 9 succeeded jobs
admin status job storage jobs.jsonl 34404 bytes; jobs.sqlite3 61440 bytes
admin status queue       queued 0, due 0, scheduled 0, expired 0, running 0, leased_queued 0, lease_expired_queued 0, running_leased 0
admin status corpus      6 papers, 6 active, 6 indexed
admin status chunks      512 rows across 6 source paths
admin status index       fresh; active_source_paths=6, chunk_source_paths=6
admin status artifacts   1 artifact, 20482 bytes
active paper count      6
FAISS index size        786477 bytes
bot-resume              healthy
bot-lingju              healthy
available memory        about 2.2 GiB
root disk free          26G
```

During restart windows, Cloudflare can briefly fail while the tunnel reaches an
origin that is still restarting. On 2026-06-01, one HTTPS UI probe timed out
after 20 seconds immediately after service restart; follow-up checks returned
200 for the HTTPS UI/API endpoints, and local origin checks returned 200.

## Cloudflare Tunnel

Cloudflare routes both `smy.hyper-dusty.cloud` (UI) and
`api-smy.hyper-dusty.cloud` (FastAPI) through a single named tunnel:

```
Zone       hyper-dusty.cloud
Tunnel     fluxmind-smy
Tunnel ID  692b5ddf-2684-4a2f-84d4-30c87bf32dba
Service    cloudflared-fluxmind-smy.service
Ingress    smy.hyper-dusty.cloud      -> http://127.0.0.1:18501   (Streamlit UI)
           api-smy.hyper-dusty.cloud  -> http://127.0.0.1:18502   (FastAPI)
           *                          -> http_status:404
```

Ingress is managed remotely (this is a token-mode tunnel, no local YAML
config). Updates go through the Cloudflare API:
`PUT /accounts/{acct}/cfd_tunnel/{tunnel_id}/configurations`. DNS CNAMEs for
both hostnames point to `<tunnel_id>.cfargotunnel.com` with `proxied: true`.

The tunnel token is stored only on the server in
`/etc/default/cloudflared-fluxmind-smy`; do not copy it into this repository.

## Refresh Commands

Use live state before making deployment decisions:

```bash
python scripts/health_check.py \
  --url https://smy.hyper-dusty.cloud/ \
  --url https://api-smy.hyper-dusty.cloud/health

python scripts/health_check.py --ssh-host root@100.100.233.26

ssh -o BatchMode=yes root@100.100.233.26 \
  'systemctl is-active cloudflared-fluxmind-smy.service fluxmind-ui.service fluxmind-api.service fluxmind-worker.service docker.service;
   ss -ltnp | egrep "18501|18502" || true;
   curl -sS --max-time 10 http://127.0.0.1:18502/health;
   docker ps --format "{{.Names}} {{.Status}}" | egrep "bot-resume|bot-lingju" || true;
   grep -E "^(LLM_MODEL|EMBEDDING_MODEL)=" /opt/fluxmind/.env;
   free -h | sed -n "2p";
   df -h / | sed -n "2p"'

curl -sS --max-time 10 -o /dev/null -w 'public_ui=%{http_code}\n' \
  http://223.6.253.9:18501/

curl -sS --max-time 10 -o /dev/null -w 'smy_https=%{http_code}\n' \
  https://smy.hyper-dusty.cloud/

curl -sS --max-time 10 -o /dev/null -w 'public_api_health=%{http_code}\n' \
  http://223.6.253.9:18502/health
```
