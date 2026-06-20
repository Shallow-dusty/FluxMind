# FluxMind Deployment Status

Last live check: 2026-06-17 03:14 CST

This document records the current deployment snapshot. Treat it as a
pointer for re-checking the live host, not as proof that the service is still
healthy at a later time.

## Current Deployment

Workspace directory: `11.FluxMind/`
Last verified source/eval update before this deployment record: `bb9cb76`
(`test: expand PDF structure eval gate`), building on the `95f1760`
Octave-aware code-output eval update and the `9b1cbc5` community-quality eval
baseline. `/opt/fluxmind` is not a git checkout, so the live deployment should
be treated as a synchronized source tree rather than a deployed commit hash. Repo
documentation commits may be newer than this application-code baseline.
Last deployed implementation/eval update: `bb9cb76`
(`test: expand PDF structure eval gate`).
Last deployed source/docs/health sync: `0aa1919`
(`docs: document PDF structure gate expansion`).

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

PDF-structure eval status was refreshed on 2026-06-17 03:14 CST after pushing
`bb9cb76` and `0aa1919`. `.venv/bin/python scripts/deploy_sync.py --apply`
synced README, docs, `eval/rag_baseline.json`, and `scripts/health_check.py` to
`/opt/fluxmind` without restarting services, because the active UI/API/worker
runtime path did not change.

The follow-up health gates passed with public HTTPS UI 200 at
`https://smy.hyper-dusty.cloud/`, public API health 200 at
`https://api-smy.hyper-dusty.cloud/health`, services active, ports `18501` and
`18502` listening, local API health `{"status":"ok"}`, `active_papers=30`,
`chunk_metadata_rows=1934`, `chunk_metadata_sources=30`, `index_fresh=True`,
and `/dev/vda3` at 36% used. Server-local `venv/bin/python
scripts/health_check.py` passed, and local
`.venv/bin/python scripts/health_check.py --url https://smy.hyper-dusty.cloud/
--ssh-host root@100.100.233.26` also passed.

Server-local `venv/bin/python scripts/evaluate_rag.py --retrieval-url
http://127.0.0.1:18502 --json-report
/tmp/fluxmind-live-pdf30-eval-report.json` passed 107/107 live retrieval
results and all regression gates. The report metrics include
`answer_case_count=42`, `retrieval_only_case_count=65`,
`retrieval_eval_question_count=107`, `recorded_answer_count=42`,
`code_output_case_count=13`, `pdf_structure_case_count=30`,
`live_retrieval_result_count=107`, `live_retrieval_pass_rate=1.0`, and
`seed_paper_count=30`. The Octave-compatible
`octave-pmsm-current-decay-template` case still passed through the expected
`runtime_unavailable` path on the current host.

Server-local `venv/bin/python scripts/quality_readiness.py --live-report
/tmp/fluxmind-live-pdf30-eval-report.json --format markdown` returned
`local_foundation_ready=true`, `small_group_ready=true`, and
`community_ready=false`. Community remains blocked on corpus/eval breadth and
live-answer evidence, not on the 13-case code-output gate or the 30-case
PDF-structure gate.

Server-local `scripts/product_readiness.py --format markdown` returned
`local_foundation_ready=true`, `activation_ready=false`,
`identity_quotas_billing_enabled=false`, `product_registry_available=false`,
`product_quota_guard_enabled=false`, `product_rbac_guard_enabled=false`, and
expected activation blockers for multi-user identity, API-key lifecycle, quota
store, billing provider, and billing attribution. `product_quota_guard_disabled`
and `product_rbac_guard_disabled` are advisories in the default production
config. Server-local code anchors confirmed `enforce_product_quota` and
`enforce_product_rbac` in `api.py`, `quota_decision` and `permission_decision`
in `src/product_registry.py`, and both `fluxmind_product_quota_guard_enabled`
and `fluxmind_product_rbac_guard_enabled` in `src/admin.py`.

Server-local `scripts/api_key_registry.py status --format markdown` returned
`backend=none`, `available=false`, `active_keys=0`, and
`secrets_exported=false`. Server-local `scripts/product_registry.py status
--format markdown` returned `backend=none`, `available=false`, `users=0`,
`workspaces=0`, `rbac_available=false`, `quota_limits=0`, `usage_events=0`,
`billing_accounts=0`, and `secrets_exported=false`. Production does not activate
either SQLite registry or the query quota/RBAC guards until
`FLUXMIND_API_KEY_REGISTRY_BACKEND=sqlite`,
`FLUXMIND_PRODUCT_REGISTRY_BACKEND=sqlite`, `FLUXMIND_QUOTA_STORE_BACKEND=sqlite`,
`FLUXMIND_PRODUCT_QUOTA_GUARD_ENABLED=true`, and
`FLUXMIND_PRODUCT_RBAC_GUARD_ENABLED=true` are deliberately set.

Server-local product registry management anchors confirmed
`admin_product_registry_create_workspace` and the
`/admin/product-registry/*` route family in `api.py`,
`render_product_registry_management` in `app.py`, and
`list_workspace_summaries` in `src/product_registry.py`. An authenticated local
API smoke of `/admin/product-registry/status` returned `backend=none`,
`available=false`, and `reason=product_registry_not_configured`, which is the
expected default production state. The operator management API/UI is installed
for self-hosted SQLite-registry use, but it is not an external identity,
payment, or production team-admin system.

Server-local object-manifest anchors confirmed
`collect_object_storage_migration_manifest` and
`verify_object_storage_migration_manifest` in `src/storage_migration.py`, plus
`--include-object-manifest`, `--object-key-prefix`, and
`--verify-object-manifest` in
`scripts/platform_migration_rehearsal.py`. A server-local smoke of
`venv/bin/python scripts/platform_migration_rehearsal.py
--include-object-manifest --format json` returned `rehearsal_ok=true`,
`object_manifest_ready=true`, `object_count=19`, `unique_object_count=18`,
`source_paths_exported=false`, `filenames_exported=false`,
`bucket_exported=false`, and `secrets_exported=false`. A follow-up
`--verify-object-manifest /tmp/fluxmind-object-manifest-verify-live.json` smoke
returned `ok=true`, `checked_objects=19`, `missing_objects=0`,
`mismatched_objects=0`, `extra_objects=0`, `manifest_errors=[]`,
`source_paths_exported=false`, `filenames_exported=false`,
`bucket_exported=false`, and `secrets_exported=false`. This is local/staged
object-manifest verification, not live external object storage activation.

Server-local quality-readiness anchors confirmed `live_answer_pass_rate`,
`average_live_answer_term_coverage`, and `minimum_live_retrieval_pass_rate` in
`src/quality_readiness.py`. `venv/bin/python scripts/evaluate_rag.py
--retrieval-url http://127.0.0.1:18502 --json-report
/tmp/fluxmind-live-quality-readiness-report.json` passed 107/107 live retrieval
cases. Re-running `scripts/quality_readiness.py --live-report
/tmp/fluxmind-live-quality-readiness-report.json --require-target small_group`
returned `local_foundation_ready=true`, `small_group_ready=true`,
`community_ready=false`, `live_retrieval_result_count=107`,
`live_retrieval_pass_rate=1.0`, `live_answer_result_count=0`,
`live_answer_pass_rate=n/a`, and `live_answer_term_coverage=n/a`, then exited
0. Re-running with `--require-target community` exited 1 as expected. This
confirms the small-group quality lane only when explicit no-secret live report
evidence is supplied; community remains a measured gap until live answer count,
pass-rate, and term-coverage evidence are present alongside the broader corpus
and eval targets.

Server-local `scripts/storage_schema.py --format markdown` returned `ok=true`,
`store_count=9`, `problem_count=0`, and optional `api_key_registry_sqlite` and
`product_registry_sqlite` stores both `ok=true`. Authenticated admin status
returned `storage_schemas.store_count=9`, `problem_count=0`, both registry
store names in the store list, and `content_scanned=false`; no token values,
token hashes, owner IDs, prompts, answers, source paths, or runtime file
contents were exported.

Local API-key registry deploy was refreshed on 2026-06-16 23:36 CST after
pushing `6ad6dbc`, `8f9db56`, `ea1c508`, and `207ba7a`. That earlier slice
added the optional local SQLite API-key lifecycle registry and kept it disabled
by default in production.

Provider-readiness deploy was refreshed on 2026-06-16 19:51 CST after pushing
`938e918` and `0deea23`. `.venv/bin/python scripts/deploy_sync.py --apply
--restart` synced source, docs, tests, and `.env.example` changes to
`/opt/fluxmind`, then restarted `fluxmind-api.service`,
`fluxmind-ui.service`, and `fluxmind-worker.service`; all three returned
`active`.

The follow-up health gate passed with public HTTPS UI 200 at
`https://smy.hyper-dusty.cloud/`, public API health 200 at
`https://api-smy.hyper-dusty.cloud/health`, services active, ports `18501` and
`18502` listening, local API health `{"status":"ok"}`, `active_papers=30`,
`chunk_metadata_rows=1934`, `chunk_metadata_sources=30`, and
`index_fresh=True`. Server-local provider-readiness smoke returned
`local_foundation_ready=true`, `activation_ready=false`,
`local_blockers=none`, and activation blockers
`external_providers_disabled`, `external_image_provider_not_configured`,
`hosted_execution_provider_not_configured`, `matlab_backend_not_configured`,
and `provider_quota_guard_not_enabled`; `--require-activation` exited 1 as
expected. Authenticated admin status returned the same `provider_readiness`
state, and `/admin/metrics` includes `fluxmind_provider_*` gauges while still
omitting `api_key` and `owner_id`.

Community-quality eval/docs sync and live retrieval evaluation were refreshed
on 2026-06-16 17:39 CST after pushing `9b1cbc5` and `8b81c57`.
`.venv/bin/python scripts/deploy_sync.py --apply` synced README, docs, tests,
`eval/rag_baseline.json`, and `scripts/health_check.py` to `/opt/fluxmind`
without restarting services. `fluxmind-api.service`, `fluxmind-ui.service`, and
`fluxmind-worker.service` stayed `active`, local API health returned
`{"status":"ok"}`, and server-local `venv/bin/python scripts/health_check.py`
passed with the local FAISS index non-empty and `active_papers=30`.

Server-local
`venv/bin/python scripts/evaluate_rag.py --retrieval-url http://127.0.0.1:18502
--json-report /tmp/fluxmind-live-octave-eval-report.json` passed all 107 live
retrieval cases, with `minimum_live_retrieval_pass_rate=1.00`. The older
report metrics are `answer_case_count=42`, `retrieval_only_case_count=65`,
`retrieval_eval_question_count=107`, `recorded_answer_count=42`,
`code_output_case_count=13`, `pdf_structure_case_count=20`,
`live_retrieval_result_count=107`, and `seed_paper_count=30`. Public HTTPS
checks with bounded timeouts passed after the sync:
`https://api-smy.hyper-dusty.cloud/health` returned `{"status":"ok"}` and
`https://smy.hyper-dusty.cloud/` returned 200.

At that time, community gaps remained: corpus growth toward 50 papers, 80 answer
cases, 80 recorded answers, 100 retrieval-only cases, 180 total retrieval
questions, 30 PDF structure cases, and live answer evidence. The 13-case
code-output gate and 100-case live retrieval target were already met.

Production readiness foundation deploy was refreshed on 2026-06-16 14:42 CST
after pushing `18200f6`.
`.venv/bin/python scripts/deploy_sync.py --apply --restart` synced source,
docs, tests, and `.env.example` changes to `/opt/fluxmind`, then restarted
`fluxmind-api.service`, `fluxmind-ui.service`, and
`fluxmind-worker.service`; all three returned `active`.

The follow-up SSH health gate passed with services active, ports `18501` and
`18502` listening, local API health `{"status":"ok"}`, `active_papers=30`,
`chunk_metadata_rows=1934`, `chunk_metadata_sources=30`, and
`index_fresh=True`. Admin status smoke reported local metadata/object storage
available and the new distributed job-store readiness surface as
`backend=local`, `available=True`, `external_configured=False`; this is the
expected blocker state until an external job-store backend is chosen and
migration-tested. The same SSH gate passed admin metrics, chunk-filter,
retrieval, and corpus-profile report smokes. Public HTTPS checks also passed:
`https://smy.hyper-dusty.cloud/` returned 200 and
`https://api-smy.hyper-dusty.cloud/health` returned `{"status":"ok"}`.

Small-group quality completion deploy and live retrieval evaluation were
refreshed on 2026-06-16 14:17 CST after pushing `e069873`, `cc705dc`, and
`d80c083`.
Twelve additional official-source open-access `papers/library/` PDFs were added
for integrated SMC/DOB/LPF, second-order adaptive SMO, fast terminal SMPC,
model-free GSTA/FTSMC, fractional super-twisting SMDO, IFTSM adaptive control,
CESO composite SMC, SMC model-predictive current control, prescribed-performance
LESO, ISMO antidisturbance control, and two Frontiers SMO/sensorless-control
surveys. The curated PDFs were synced explicitly because
`scripts/deploy_sync.py` excludes `papers/`; source/eval were synced with
`.venv/bin/python scripts/deploy_sync.py --apply` without restarting services.
All 30 library papers were activated through the local API, and synchronous
index rebuild job `dbf88a8dfa1d` succeeded with `paper_count=30` and
`chunk_count=1934`.

Server-local
`venv/bin/python scripts/evaluate_rag.py --retrieval-url http://127.0.0.1:18502
--json-report /tmp/fluxmind-live-corpus30-report.json` passed all 100 live
retrieval cases, with regression gates `24/24` OK and
`minimum_live_retrieval_pass_rate=1.00`. The report shows
`self_use=met`, `small_group=met`, and `community=gap`.

The same verification pass confirmed public UI 200 at
`https://smy.hyper-dusty.cloud/`, public API health 200 at
`https://api-smy.hyper-dusty.cloud/health`, SSH runtime checks green, and
remote corpus status `papers=30`, `active=30`, `indexed=30`, `chunks=1934`,
`index=fresh`. The SSH health gate reported UI/API/worker/cloudflared/docker
active, ports `18501` and `18502` listening, Docker execution still
`configured=False available=False reason=not_configured`, local storage
available, retrieval/admin smokes passing, and `/dev/vda3` at 36% used. At that
point, small-group gaps were zero and community gaps still included corpus
growth toward 50 papers, 80 recorded answers, 180 retrieval questions, 30 PDF
structure cases, and live answer evidence; code-output breadth is now 13 cases
in the latest 2026-06-17 eval sync.

Corpus-expansion deploy and live retrieval evaluation were refreshed on
2026-06-15 14:29 CST after pushing `b2f543e` and `d1e5326`. Four additional
curated `papers/library/` PDFs were added for super-twisting SMC with ESO,
adaptive quick reaching law with SFTSMO, DSMO/LQR current-loop control, and
NFTSMC with SDOB. The curated PDFs were synced explicitly because
`scripts/deploy_sync.py` excludes `papers/`; source/eval were then synced with
`.venv/bin/python scripts/deploy_sync.py --apply` without restarting services.
All 18 library papers were activated through the local API, and synchronous
index rebuild job `a46499b74e4c` succeeded with `paper_count=18` and
`chunk_count=1216`. Server-local
`venv/bin/python scripts/evaluate_rag.py --retrieval-url http://127.0.0.1:18502
--json-report /tmp/fluxmind-live-corpus18-report.json` passed all 86 live
retrieval cases, with regression gates `24/24` OK and
`minimum_live_retrieval_pass_rate=1.00`.

The same verification pass confirmed public UI 200 at
`https://smy.hyper-dusty.cloud/`, public API health 200 at
`https://api-smy.hyper-dusty.cloud/health`, SSH runtime checks green, and
remote corpus status `papers=18`, `active=18`, `indexed=18`, `chunks=1216`,
`index=fresh`. The SSH health gate reported UI/API/worker/cloudflared/docker
active, ports `18501` and `18502` listening, Docker execution still
`configured=False available=False reason=not_configured`, local storage
available, retrieval/admin smokes passing, and `/dev/vda3` at 36% used. Current
small-group gaps are `seed_paper_count=12`, `answer_case_count=8`,
`retrieval_only_case_count=6`, `retrieval_eval_question_count=14`, and
`recorded_answer_count=8`; the `live_retrieval_result_count`,
`code_output_case_count`, `pdf_structure_case_count`, and `topic_group_count`
small-group metrics are now met.

Corpus-expansion deploy and live retrieval evaluation were refreshed on
2026-06-15 10:23 CST after pushing `745b7d4` and `1b7795a`. The curated
`papers/library/` PDFs were synced explicitly because `scripts/deploy_sync.py`
excludes `papers/`; source/docs/eval were then synced with
`.venv/bin/python scripts/deploy_sync.py --apply` without restarting services.
All 14 library papers were activated through the local API, and synchronous
index rebuild job `7da0aa3167b1` succeeded with `paper_count=14` and
`chunk_count=987`. Server-local
`venv/bin/python scripts/evaluate_rag.py --retrieval-url http://127.0.0.1:18502
--json-report /tmp/fluxmind-live-corpus-expansion-report.json` passed all 74
live retrieval cases, with regression gates `24/24` OK and
`minimum_live_retrieval_pass_rate=1.00`.

The same verification pass confirmed public UI 200 at
`https://smy.hyper-dusty.cloud/`, public API health 200 at
`https://api-smy.hyper-dusty.cloud/health`, SSH runtime checks green, and
remote corpus status `papers=14`, `active=14`, `indexed=14`, `chunks=987`,
`index=fresh`. The SSH health gate reported UI/API/worker/cloudflared/docker
active, ports `18501` and `18502` listening, Docker execution still
`configured=False available=False reason=not_configured`, local storage
available, retrieval/admin smokes passing, and `/dev/vda3` at 36% used. Current
small-group gaps are `seed_paper_count=16`, `answer_case_count=12`,
`retrieval_only_case_count=14`, `retrieval_eval_question_count=26`,
`recorded_answer_count=12`, and `pdf_structure_case_count=3`; the
`live_retrieval_result_count` and `code_output_case_count` small-group metrics
are now met.

Quality-expansion eval sync and live retrieval evaluation were refreshed on
2026-06-15 09:38 CST. After syncing the expanded `eval/rag_baseline.json` to
`/opt/fluxmind` without `--restart`, the server-local command
`venv/bin/python scripts/evaluate_rag.py --retrieval-url http://127.0.0.1:18502
--json-report /tmp/fluxmind-live-quality-expansion-report.json` passed. It
scored all 65 live retrieval cases as OK, with regression gates `24/24` OK,
and wrote a no-secret quality report showing `self_use=met`,
`small_group=gap`, and `community=gap`. Current small-group gaps are
`seed_paper_count=19`, `answer_case_count=15`, `retrieval_only_case_count=20`,
`retrieval_eval_question_count=35`, `recorded_answer_count=15`,
`code_output_case_count=2`, and `pdf_structure_case_count=6`. No service
restart was required for this eval-only sync.

Quality-gate sync and live retrieval evaluation were refreshed on 2026-06-15
09:23 CST. After syncing the local quality-roadmap/eval-report changes to
`/opt/fluxmind` without `--restart`, the server-local command
`venv/bin/python scripts/evaluate_rag.py --retrieval-url http://127.0.0.1:18502
--json-report /tmp/fluxmind-live-quality-report.json` passed. It scored all 50
live retrieval cases as OK, with regression gates `24/24` OK, and wrote a
no-secret quality report showing `self_use=met`, `small_group=gap`, and
`community=gap`. The small-group live-retrieval metric is now met; remaining
small-group gaps are corpus size, answer/retrieval eval breadth, recorded-answer
count, code-output cases, and PDF-structure cases. No service restart was
required for this eval/script sync.

Read-only live checks and a no-restart documentation sync were refreshed on
2026-06-15 08:48 CST. After pushing docs refresh `20d75e5`, `.venv/bin/python
scripts/deploy_sync.py --apply` synced the updated docs, health-check script,
and docs-status test guard to `/opt/fluxmind` without `--restart`. Public HTTPS
UI/API health returned 200, the SSH runtime gate passed, and the remote runtime
still reported UI/API/worker/cloudflared/docker `active`, `active_papers=11`,
`chunk_metadata_rows=800`, `index_fresh=True`, healthy retrieval/admin smokes,
and `/dev/vda3` at 46% used. No service restart was required for this docs-only
sync and verification.

Live checks refreshed on 2026-06-15 08:21 CST after pushing `main` through
`4f27651`, deploying the runtime-state parsing hardening and the `.coverage`
deploy-sync exclude with `.venv/bin/python scripts/deploy_sync.py --apply
--restart`. The deploy sync excludes `.env`, `.coverage`, `.cache/`, `venv/`,
`models/`, `metadata/`, `jobs/`, `artifacts/`, `papers/`, and `faiss_index/`, so
the runtime corpus/index state is preserved while source/docs/test changes are
synced. The one `.coverage` file copied during the previous deploy was removed
from `/opt/fluxmind`, and a follow-up SSH check confirmed `coverage_absent`.

Post-restart public HTTPS UI/API health passed. The SSH runtime gate passed with
UI/API/worker/cloudflared/docker active, ports `18501` and `18502` listening,
local `/health` OK, `active_papers=11`, `faiss_index_bytes=1228845`,
`chunk_metadata_rows=800`, `chunk_metadata_sources=11`, `index_fresh=True`,
Docker execution still `configured=False available=False reason=not_configured`,
local storage available, and retrieval/admin smokes passing. An authenticated
`/corpus/status` smoke returned `papers=11`, `active=11`, `indexed=11`,
`chunks=800`, and `index=fresh`; all three systemd services reported `active`.

Live checks refreshed on 2026-06-15 04:23 CST after pushing `main` through
`391ac7f`, syncing the curated `papers/library/` seed corpus without deleting
runtime uploads/state, activating all 11 library papers through authenticated
`PUT /corpus/active`, and rebuilding the FAISS index through async job
`8c4f1995a02a`. The rebuild succeeded with `paper_count=11` and
`chunk_count=800`.

Live checks refreshed on 2026-06-15 03:15 CST after pushing `main` through
`3cfa426` and deploying with `.venv/bin/python scripts/deploy_sync.py --apply
--restart`. This refresh fixed the two startup findings from the previous
deploy: API retrieval warmup now runs in the background with `/health` as
process liveness and `/ready` as retrieval readiness, and the API import path no
longer imports the PDF/FAISS ingestion stack before uvicorn can bind. Local
verification passed with `.venv/bin/python -m pytest` (286 passed),
`.venv/bin/python scripts/health_check.py`, and
`.venv/bin/python scripts/evaluate_rag.py`. The local API import smoke measured
0:00.88; the remote `/opt/fluxmind` import smoke measured 0:15.52.

The guarded deploy restarted API/UI/worker. An immediate public API probe and
first SSH runtime gate still hit the shortened bind window, but follow-up checks
passed: HTTPS UI and API health returned 200, public `/ready` returned
`{"status":"ready","warmup":{"status":"ready","ready":true,"error":""}}`,
and the SSH gate confirmed active UI/API/worker/cloudflared/docker services,
listeners on `18501` and `18502`, local API health OK, `active_papers=6`,
`faiss_index_bytes=786477`, `chunk_metadata_rows=512`,
`chunk_metadata_sources=6`, `index_fresh=True`, Docker execution still
`configured=False available=False reason=not_configured`, local storage
available, and retrieval/admin metrics smokes passing. The new journal window
from the 03:12 restart showed Uvicorn running about 22 seconds after systemd
start, down from the previous 68-second warmup window; application startup
completed in about one second after process start. The same journal window did
not show FAISS AVX512/AVX2 fallback lines; it only logged SentenceTransformer
model loading during background readiness warmup.

Live checks refreshed on 2026-06-15 02:31 CST after pushing `main` through
`17aacc3` and deploying with `.venv/bin/python scripts/deploy_sync.py --apply
--restart`. The guarded sync excluded `.env`, `.cache/`, `venv/`, `models/`,
`metadata/`, `jobs/`, `artifacts/`, `papers/`, and `faiss_index/`; it synced
source/docs/test changes and restarted API/UI/worker. The first post-restart
probe hit the normal API warmup window: the UI returned 200 and `18501` was
listening, but API health briefly returned 502 externally and `127.0.0.1:18502`
was not yet bound. A follow-up run passed both public HTTPS checks and the full
SSH gate. The verified state was: UI/API/worker/cloudflared/docker active;
listeners on `18501` and `18502`; local API health `{"status":"ok"}`;
`LLM_MODEL=mimo-v2.5-pro`; embedding model
`/opt/fluxmind/models/all-MiniLM-L6-v2`; `active_papers=6`;
`faiss_index_bytes=786477`; `chunk_metadata_rows=512`;
`chunk_metadata_sources=6`; `index_fresh=True`; admin metrics smoke OK; Docker
execution still `configured=False available=False reason=not_configured`;
local metadata/object storage available; chunk-filter, retrieval, and corpus
profile report smokes passed; root disk 21G free. The remote journal still
includes FAISS optional AVX512/AVX2 module fallback lines, but API health,
index/chunk metadata, retrieval smoke, and admin metrics all passed.

Live read-only checks refreshed on 2026-06-14 01:04 CST during git/docs status
cleanup. No deploy sync, service restart, runtime mutation, or push was
performed. `.venv/bin/python scripts/health_check.py --url
https://smy.hyper-dusty.cloud/ --url https://api-smy.hyper-dusty.cloud/health`
passed with both HTTPS endpoints returning 200. The full current-code SSH gate
`.venv/bin/python scripts/health_check.py --ssh-host root@100.100.233.26`
connected and confirmed active systemd services, `18501`/`18502` listeners, and
local API health, but exited nonzero because the current local health checker
expects the latest `origin/main` feature slice to already exist under
`/opt/fluxmind`; that slice has been pushed but has not been deployed or
restarted on Trace-Twin. Targeted read-only SSH probes confirmed
`cloudflared-fluxmind-smy.service`, `fluxmind-ui.service`,
`fluxmind-api.service`, `fluxmind-worker.service`, and `docker.service` active;
local API health `{"status":"ok"}`; `LLM_MODEL=mimo-v2.5-pro`; embedding model
`/opt/fluxmind/models/all-MiniLM-L6-v2`; `active_papers=6`;
`faiss_index_bytes=786477`; `chunk_metadata_rows=512`;
`chunk_metadata_sources=6`; corpus index `fresh`; Docker execution still
`configured=False available=False reason=not_configured`; and local metadata
and object storage available. The running remote `/admin/status` response does
not yet expose the newer `storage_schemas` and `platform_readiness` summaries,
which confirms the latest storage-schema/platform-readiness slice is not active
on the live API process.

Live read-only checks refreshed on 2026-06-08 00:45 CST during production-gap
research and roadmap verification. No deploy sync, service restart, runtime
mutation, or push was performed. `.venv/bin/python scripts/health_check.py --url
https://smy.hyper-dusty.cloud/ --url https://api-smy.hyper-dusty.cloud/health`
passed with both HTTPS endpoints returning 200. `.venv/bin/python
scripts/health_check.py --ssh-host root@100.100.233.26` passed with UI/API/worker,
Cloudflare tunnel, and Docker services active; listeners on `18501` and `18502`;
local API health `{"status":"ok"}`; `LLM_MODEL=mimo-v2.5-pro`;
embedding model `/opt/fluxmind/models/all-MiniLM-L6-v2`; `active_papers=6`;
`faiss_index_bytes=786477`; `chunk_metadata_rows=512`;
`chunk_metadata_sources=6`; `index_fresh=True`; Docker execution backend still
`configured=False available=False reason=not_configured`; local metadata/object
storage available; API-level chunk-filter, retrieval, and corpus-profile report
smokes passing; root disk 24G free.

Live read-only checks refreshed on 2026-06-07 22:57 CST during status-drift
cleanup. No deploy sync, service restart, runtime mutation, or push was
performed. `.venv/bin/python scripts/health_check.py --url
https://smy.hyper-dusty.cloud/ --url https://api-smy.hyper-dusty.cloud/health`
passed with both HTTPS endpoints returning 200. `.venv/bin/python
scripts/health_check.py --ssh-host root@100.100.233.26` passed with UI/API/worker,
Cloudflare tunnel, and Docker services active; listeners on `18501` and `18502`;
local API health `{"status":"ok"}`; `LLM_MODEL=mimo-v2.5-pro`;
embedding model `/opt/fluxmind/models/all-MiniLM-L6-v2`; `active_papers=6`;
`faiss_index_bytes=786477`; `chunk_metadata_rows=512`;
`chunk_metadata_sources=6`; `index_fresh=True`; Docker execution backend still
`configured=False available=False reason=not_configured`; local metadata/object
storage available; API-level chunk, retrieval, and corpus-profile report smokes
passing; root disk 24G free. Direct raw API health returned 200. A first direct
raw UI probe timed out from the local machine, but an immediate retry returned
200 and server-local origin checks returned 200 for both UI and API, so this was
treated as transient public-IP reachability noise rather than a service failure.

Live checks refreshed on 2026-06-03 00:46 CST after confirming the completed
no-key/local platform baseline, pushing `main` to `origin/main`, and deploying
with `python scripts/deploy_sync.py --apply --restart`. The guarded sync
excluded `.env`, `.cache/`, `venv/`, `models/`, `metadata/`, `jobs/`,
`artifacts/`, `papers/`, and `faiss_index/`; it synced source/docs/test changes
and restarted API/UI/worker. Public HTTPS UI/API health returned 200. The first
SSH health check hit the normal API startup warmup window after restart: systemd
services were active and the UI port was listening, but `127.0.0.1:18502` was
not yet bound. A follow-up SSH health check passed with UI/API/worker,
Cloudflare tunnel, and Docker services active; listeners on `18501` and
`18502`; local API health `{"status":"ok"}`; `LLM_MODEL=mimo-v2.5-pro`;
embedding model `/opt/fluxmind/models/all-MiniLM-L6-v2`; `active_papers=6`;
`faiss_index_bytes=786477`; `chunk_metadata_rows=512`;
`chunk_metadata_sources=6`; `index_fresh=True`; local metadata/object storage
available; Docker execution backend not configured; root disk 25G free. The
remote journal still includes FAISS optional AVX512/AVX2 module fallback lines,
but the API health, FAISS index, chunk metadata, and retrieval smoke all passed.

Live read-only checks refreshed on 2026-06-03 00:23 CST during documentation
cleanup. No deploy sync, restart, runtime mutation, or push was performed.
Local health and offline RAG eval passed with `.venv/bin/python`; HTTPS checks
returned 200 for `https://smy.hyper-dusty.cloud/` and
`https://api-smy.hyper-dusty.cloud/health`; SSH health passed with UI/API/worker,
Cloudflare tunnel, and Docker services active. Remote runtime state reported
`LLM_MODEL=mimo-v2.5-pro`, embedding model
`/opt/fluxmind/models/all-MiniLM-L6-v2`, `active_papers=6`,
`faiss_index_bytes=786477`, `chunk_metadata_rows=512`,
`chunk_metadata_sources=6`, `index_fresh=True`, local metadata/object storage
available, Docker execution backend not configured, and 25G free on `/`.

Live checks refreshed on 2026-06-02 03:40 CST after deploying the
`/admin/runtime-manifest`, `/admin/runtime-manifest/report`, and Streamlit
runtime-manifest export UI with `python scripts/deploy_sync.py --apply
--restart`. The guarded sync excluded `.env`, `.cache/`, `venv/`, `models/`,
`metadata/`, `jobs/`, `artifacts/`, `papers/`, and `faiss_index/`; it synced
only source/docs/test changes and restarted API/UI/worker. Public UI/API health
checks returned 200, remote systemd services were active, remote health passed
with `index_fresh=True`, and authenticated local API smoke reported
`local_runtime_backup_manifest False False True False 30`. The authenticated
Markdown report smoke returned `Content exported: false`, `Secrets exported:
false`, `Env file present: true`, `Env file content exported: false`,
`Total files: 30`, and `Total bytes: 93465106`.

Live checks refreshed on 2026-06-02 03:36 CST after deploying the no-secret
runtime backup manifest with `python scripts/deploy_sync.py --apply` and no
service restart. The guarded sync excluded `.env`, `.cache/`, `venv/`,
`models/`, `metadata/`, `jobs/`, `artifacts/`, `papers/`, and `faiss_index/`;
it synced only source/docs/test changes. Public UI/API health checks returned
200, remote systemd services stayed active, remote health passed with
`index_fresh=True`, and `/opt/fluxmind/scripts/runtime_manifest.py --format
markdown` reported `content_exported=false`, `secrets_exported=false`,
`env_file_present=true`, `env_file_content_exported=false`, `total_files=30`,
and `total_bytes=93465106`.

Live checks refreshed on 2026-06-02 03:28 CST after deploying
`scripts/deploy_sync.py` itself with `python scripts/deploy_sync.py --apply
--restart`. The guarded sync excluded `.env`, `.cache/`, `venv/`, `models/`,
`metadata/`, `jobs/`, `artifacts/`, `papers/`, and `faiss_index/`; it synced
only source/docs/test changes, restarted API/UI/worker, and left the Cloudflare
tunnel active. Public UI/API health checks returned 200. The first SSH health
check hit the normal API warmup binding window after restart; the rerun passed
with API/UI listeners on `18502`/`18501`, active corpus count 6, FAISS index
bytes 786477, chunk metadata rows 512, `index_fresh=True`, and safe deploy sync
anchors present on the server.

Live checks refreshed on 2026-06-02 03:22 CST after syncing local storage
inventory admin/UI changes and health-check anchors to `/opt/fluxmind`,
restarting `fluxmind-api.service`, `fluxmind-ui.service`, and
`fluxmind-worker.service`, and confirming the Cloudflare tunnel stayed active.
During the refresh, an initial `rsync --delete` command omitted `venv/`,
`models/`, and `.cache/` from excludes; the deployment was repaired by
rebuilding `/opt/fluxmind/venv` with `torch==2.5.1+cpu`, restoring
`/opt/fluxmind/models/all-MiniLM-L6-v2` from the local Hugging Face snapshot
cache, and switching `/etc/apt/sources.list` from the unreachable
`mirrors.cloud.aliyuncs.com` endpoint to `mirrors.aliyun.com`. Remote health
then passed with public UI/API 200 and SSH runtime checks OK. Authenticated
`/admin/status` reported `storage_inventory mode=local total_files=19
total_bytes=1886739 groups=[metadata,jobs,artifacts,uploads,faiss_index]`,
`content_scanned=false`, and `external_storage_configured=false`.

Live checks refreshed on 2026-06-02 02:50 CST after syncing local
control-engineering execution templates to `/opt/fluxmind`, restarting
`fluxmind-api.service`, `fluxmind-ui.service`, and `fluxmind-worker.service`,
and confirming the Cloudflare tunnel stayed active. Remote health passed with
public UI/API 200, remote source grep found `PYTHON_EXECUTION_TEMPLATES`,
`OCTAVE_EXECUTION_TEMPLATES`, `smc_reaching_law`, and `pmsm_current_decay`, and
authenticated `/jobs/code/python-local` ran the SMC reaching-law template with
`status=succeeded`, producing `smc_reaching_law.csv` as a text artifact and
`smc_reaching_law.svg` as a plot artifact. Live checks refreshed on 2026-06-02 02:45 CST after syncing local SVG diagram
templates to `/opt/fluxmind`, restarting `fluxmind-api.service`,
`fluxmind-ui.service`, and `fluxmind-worker.service`, and confirming the
Cloudflare tunnel stayed active. Remote health passed with public UI/API 200,
remote source grep found `mock_image_template`, `sliding-mode-observer`, and
`paper-figure-redraft`, and authenticated `/jobs/image/mock` generated a
`sliding-mode-observer` SVG artifact with `diagram_template=sliding-mode-observer`,
`mime=image/svg+xml`, and SVG text containing `Sliding-Mode Observer`. Live
checks refreshed on 2026-06-02 02:38 CST after syncing the health-check
warmup retry fix to `/opt/fluxmind` without restarting services. The refreshed
remote health checker passed with public UI/API 200 and remote runtime checks
green against the already-running API/UI/worker services. Live checks refreshed
on 2026-06-02 02:36 CST after syncing worker lease health
visibility to `/opt/fluxmind`, restarting `fluxmind-api.service`,
`fluxmind-ui.service`, and `fluxmind-worker.service`, and confirming the
Cloudflare tunnel stayed active. The first immediate API probe hit the known
startup warmup window before port `18502` was bound; journal showed uvicorn
completed model/FAISS warmup and bound the port about 21 seconds after restart.
The follow-up remote health passed with public UI/API 200, remote source grep
found `worker_lease_health` and `worker_leases`, and authenticated
`/admin/status` reported `worker_leases` with `total=1`, `active=0`,
`expired=0`, `workers=['fluxmind-worker-1']`, and `latest=1`. Live checks
refreshed on 2026-06-02 02:32 CST after syncing optional no-secret
query-cost pricing estimates to `/opt/fluxmind`, restarting
`fluxmind-api.service` and `fluxmind-ui.service`, and confirming
`fluxmind-worker.service` plus the Cloudflare tunnel stayed active. Remote
health passed with public UI/API 200, remote source grep found
`status_cost_pricing` and `summarize_query_cost`, and authenticated
`/admin/status` reported `estimated_cost_usd=0`, `cost_source=not_configured`,
`pricing.configured=false`, provider `mimo-v2.5-pro`, and
`external_billing_enabled=false`. Live checks refreshed on 2026-06-02 02:21 CST after syncing the Streamlit
storage-readiness dashboard refresh to `/opt/fluxmind`, restarting
`fluxmind-ui.service`, and confirming `fluxmind-api.service` and
`fluxmind-worker.service` stayed active. Remote health passed with public UI/API
200, and remote source grep found `status_storage` plus `storage_readiness` in
`/opt/fluxmind/app.py`. The 2026-06-02 02:17 CST refresh synced the durable
storage readiness API/report state, restarted `fluxmind-api.service` and
`fluxmind-ui.service`, and verified `/admin/status` reports
`storage_readiness` with local metadata/object paths writable, metadata backend
`local`, object backend `local`, both available, and
`external_storage_configured=false`, without exposing external database URLs,
buckets, endpoints, or credentials. The 2026-06-02 02:11 CST refresh synced the
aggregate RAG regression-gate files and verified remote offline eval plus
no-LLM retrieval eval. The broader 2026-06-02 01:55 CST source sync restarted
`fluxmind-api.service` and `fluxmind-ui.service`, installed/enabled
`fluxmind-worker.service`, and verified that the service still exposes the
existing FAISS index, chunk
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
local execution templates present in /opt/fluxmind/src/execution_templates.py and /opt/fluxmind/app.py; authenticated smoke ran job=9e3f473b22f1 status=succeeded titles=[smc_reaching_law.csv, smc_reaching_law.svg] kinds=[text, plot]
execution CPU limits     present in /opt/fluxmind/src/providers.py; remote busy-loop smoke returned cpu_limit_enforced=true
execution path containment present in /opt/fluxmind/src/providers.py; remote smoke rejected ../ escape with exit_code=2
execution input limits   present in /opt/fluxmind/src/providers.py; remote smoke rejected >256KiB input with exit_code=2
timeout diagnostics      present in /opt/fluxmind/src/jobs.py; exit 124 is classified as execution_timeout
safe FAISS rebuild       present in /opt/fluxmind/src/ingestion.py; temp save replaces live index only after success
cancellable index rebuild present in /opt/fluxmind/src/ingestion.py and /opt/fluxmind/src/jobs.py; cancelled rebuilds raise before publishing post-cancel state
uploaded metadata extract present in /opt/fluxmind/src/ingestion.py; remote temp-PDF smoke extracted title/authors/year/topic_tags from first-page text, and existing DOI/arXiv extraction remains covered by local tests
uploaded PDF dedup      present in /opt/fluxmind/src/ingestion.py; isolated remote smoke reused existing.pdf with chunk_count=7 and wrote no duplicate file
startup warmup only      present in /opt/fluxmind/api.py; API startup does not call build_vector_store and exposes /ready for retrieval readiness
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
admin worker leases      present in /opt/fluxmind/src/jobs.py, /opt/fluxmind/src/admin.py, and /opt/fluxmind/app.py; authenticated smoke returned total=1 active=0 expired=0 workers=['fluxmind-worker-1'] latest=1
deployed artifact layer  present in /opt/fluxmind/src/artifacts.py
deployed admin layer     present in /opt/fluxmind/src/admin.py
admin status report      present in /opt/fluxmind/src/admin.py and /opt/fluxmind/api.py; authenticated smoke returned text/markdown
admin query usage        present in /opt/fluxmind/src/admin.py, /opt/fluxmind/src/chain.py, /opt/fluxmind/src/costs.py, and /opt/fluxmind/api.py; query events keep estimated token counts and can include provider prompt/completion/total token counts when the upstream response exposes them; optional QUERY_COST_* rates estimate local USD cost without external billing; remote smoke returned estimated_cost_usd=0 cost_source=not_configured pricing_configured=false external_billing=false
admin storage readiness  present in /opt/fluxmind/src/admin.py and /opt/fluxmind/src/config.py; authenticated admin smoke returned metadata_backend=local metadata_available=true object_backend=local object_available=true external_storage_configured=false
admin job-store readiness present in /opt/fluxmind/src/admin.py and /opt/fluxmind/src/config.py; SSH health on 2026-06-16 14:42 CST returned distributed_job_store backend=local available=true external_configured=false
platform migration preflight present in /opt/fluxmind/src/platform_migration.py and /opt/fluxmind/scripts/platform_migration_preflight.py; SSH smoke on 2026-06-16 18:13 CST returned preflight_ok=true activation_ready=false local_blockers=none activation_blockers=[production_metadata_database_not_configured,production_object_storage_not_configured,distributed_job_store_not_configured]
runtime migration rehearsal present in /opt/fluxmind/src/storage_migration.py and /opt/fluxmind/scripts/platform_migration_rehearsal.py; SSH smoke on 2026-06-16 18:34 CST returned rehearsal_ok=true copied_files=19 restore_check_ok=true staged_storage_schema_ok=true blockers=none
product readiness       present in /opt/fluxmind/src/product_readiness.py and /opt/fluxmind/scripts/product_readiness.py; SSH smoke on 2026-06-17 00:10 CST using /opt/fluxmind/venv/bin/python returned local_foundation_ready=true activation_ready=false; local API-key lifecycle and product-registry code are implemented but deployed backends remain none, so activation blockers still include [multi_user_identity_not_configured,api_key_lifecycle_not_configured,identity_quota_store_not_configured,billing_provider_not_configured,billing_attribution_not_enabled]; --require-activation exits 1 as expected; metrics include fluxmind_product_* and still omit token values/token hashes/owner_id
local API key registry  present in /opt/fluxmind/src/api_keys.py and /opt/fluxmind/scripts/api_key_registry.py; SSH smoke on 2026-06-17 00:10 CST returned backend=none available=false active_keys=0 secrets_exported=false; storage_schema.py returned ok=true store_count=9 problem_count=0 and optional api_key_registry_sqlite ok=true; authenticated admin status returned store_count=9 and api_key_registry_sqlite present; production default remains disabled, so product_readiness still reports api_key_lifecycle_not_configured until FLUXMIND_API_KEY_REGISTRY_BACKEND=sqlite is deliberately enabled
local product registry  present in /opt/fluxmind/src/product_registry.py and /opt/fluxmind/scripts/product_registry.py; SSH smoke on 2026-06-17 00:10 CST returned backend=none available=false users=0 workspaces=0 secrets_exported=false; storage_schema.py returned ok=true store_count=9 problem_count=0 and optional product_registry_sqlite ok=true; authenticated admin status returned store_count=9 and product_registry_sqlite present; production default remains disabled, so no multi-user identity, quota, usage, or billing-attribution runtime is activated until FLUXMIND_PRODUCT_REGISTRY_BACKEND=sqlite is deliberately enabled
provider readiness      present in /opt/fluxmind/src/provider_readiness.py and /opt/fluxmind/scripts/provider_readiness.py; SSH smoke on 2026-06-16 19:51 CST using /opt/fluxmind/venv/bin/python returned local_foundation_ready=true activation_ready=false local_blockers=none activation_blockers=[external_providers_disabled,external_image_provider_not_configured,hosted_execution_provider_not_configured,matlab_backend_not_configured,provider_quota_guard_not_enabled]; --require-activation exited 1 as expected; authenticated admin status returned provider_readiness local_foundation_ready=true activation_ready=false; metrics include fluxmind_provider_* and still omit api_key/owner_id
quality readiness       present in /opt/fluxmind/src/quality_readiness.py and /opt/fluxmind/scripts/quality_readiness.py; SSH smoke on 2026-06-17 02:37 CST using /opt/fluxmind/venv/bin/python returned local_foundation_ready=true, default small_group_ready=false without live evidence, community_ready=false, and --require-target community exited 1 as expected; server-local evaluate_rag --retrieval-url wrote /tmp/fluxmind-live-quality-readiness-report.json with 107/107 live retrieval pass, and quality_readiness.py --live-report returned small_group_ready=true, community_ready=false, live_retrieval_pass_rate=1.0, live_answer_result_count=0, live answer quality n/a; live answer count/pass-rate/term-coverage gates are installed but not yet satisfied for community readiness
admin storage inventory  present in /opt/fluxmind/src/admin.py and /opt/fluxmind/app.py; authenticated admin smoke returned mode=local total_files=19 total_bytes=1886739 groups=[metadata,jobs,artifacts,uploads,faiss_index] content_scanned=false external_storage_configured=false
deployed metadata layer  present in /opt/fluxmind/src/metadata.py
corpus SQLite mirror     present in /opt/fluxmind/src/metadata.py; paper metadata mirrors into metadata/corpus.sqlite3
paper enrichment fields  present in /opt/fluxmind/src/metadata.py; DOI/arXiv/venue/topic_tags columns migrated into metadata/corpus.sqlite3
atomic metadata writes   present in /opt/fluxmind/src/metadata.py; corpus/profile JSON writes use temp-file atomic replace, and post-fix concurrent report smoke completed 8/8 requests
chunk SQLite mirror      present in /opt/fluxmind/src/metadata.py; indexed chunk metadata can mirror into metadata/chunks.sqlite3
chunk source listing     present in /opt/fluxmind/src/metadata.py; chunk source paths can be compared with active corpus
deployed eval layer      present in /opt/fluxmind/src/evaluation.py
safe deploy sync         present in /opt/fluxmind/scripts/deploy_sync.py; local apply smoke synced only source/docs/test changes, excluded runtime state, restarted API/UI/worker, and follow-up remote health passed
source/page eval check   present in /opt/fluxmind/src/evaluation.py; expected refs are verified against actual PDFs
live RAG eval gate       present in /opt/fluxmind/src/evaluation.py and scripts/evaluate_rag.py --live-url
live retrieval eval gate present in /opt/fluxmind/src/evaluation.py and scripts/evaluate_rag.py --retrieval-url; remote retrieval eval returned live_retrieval=5/5 context_coverage=1.00
aggregate eval gates     present in /opt/fluxmind/src/evaluation.py and eval/rag_baseline.json; remote gate summary passed minimum_case_count=5, required_answer_modes=5, minimum_expected_source_ref_count=7, provider_fixture_count=4, recorded_answer_count=5, recorded_pass_rate=1.00, average_recorded_coverage=1.00, and live_retrieval_pass_rate=1.00
eval JSON report         present in /opt/fluxmind/src/evaluation.py and scripts/evaluate_rag.py --json-report; remote report summary offline=5/5 provider=4/4 recorded=5/5 live_retrieval=5/5
recorded eval gate       present in /opt/fluxmind/eval/rag_baseline.json; recorded answers scored at coverage=1.00 across all 5 answer modes
hybrid retrieval         present in /opt/fluxmind/src/chain.py; query uses hybrid_retrieve
local BM25 reranker      present in /opt/fluxmind/src/chain.py; hybrid_retrieve uses bm25_relevance_scores
optional local reranker  present in /opt/fluxmind/src/chain.py; learned_rerank_documents uses local RERANKER_MODEL path only, and remote admin status returned reranker_model_configured=false reranker_model_available=false
source-diverse reranking present in /opt/fluxmind/src/chain.py; expanded candidates preserve multiple sources before TOP_K
generated citation check present in /opt/fluxmind/src/chain.py; QueryResult validates numbered refs against source/page context
retrieval diagnostics present in /opt/fluxmind/src/chain.py; retrieve_with_metadata returns no-LLM context refs and source/page completeness
numbered citation guard  present in /opt/fluxmind/src/chain.py; prompt lists valid context-ref range per answer
citation neutralizer     present in /opt/fluxmind/src/chain.py; out-of-range bracket refs are neutralized before validation
artifact references      present in /opt/fluxmind/src/chain.py; RAG prompt includes Generated Artifact References
artifact formatter       present in /opt/fluxmind/src/artifacts.py; stable artifact IDs can be cited and include diagram template metadata
artifact checksum metadata present in /opt/fluxmind/src/providers.py; remote smoke generated checksum_sha256 and byte_count=2
artifact integrity status present in /opt/fluxmind/src/artifacts.py and /opt/fluxmind/src/admin.py; authenticated admin smoke returned checked=1, ok=1, unchecked=1, checksum_mismatch=0
mock image metadata      present in /opt/fluxmind/src/providers.py; local-mock-svg-v1 metadata installed
local diagram templates  present in /opt/fluxmind/src/providers.py and /opt/fluxmind/app.py; authenticated smoke generated job=a4ec9ff27da8 template=sliding-mode-observer svg_has_observer=True
execution artifacts      local code job captured result.txt artifact
artifact export route    present; authenticated local API listed result.txt
artifact gallery         present in /opt/fluxmind/app.py; stable IDs, metadata, local filters, and downloads rendered; authenticated artifact filter smoke created job=d8abf4a5c36e artifact=dbf8747af5f926d3 filtered_count=1 missing_filter_count=0
admin status panel       present in /opt/fluxmind/app.py with no-delete retention preview UI
admin status route       present; authenticated local API returned runtime state and provider_failures
admin retention preview  present in /opt/fluxmind/api.py, /opt/fluxmind/src/admin.py, and /opt/fluxmind/app.py; authenticated smoke returned mode=preview delete_enabled=false limit=25 uploads=0 artifacts=0
admin query usage panel  present in /opt/fluxmind/app.py; status_query_usage rendered in Streamlit source
admin cost pricing panel present in /opt/fluxmind/app.py; status_cost_pricing rendered in Streamlit source
admin storage panel      present in /opt/fluxmind/app.py; status_storage, storage_readiness, and distributed_job_store rendered in Streamlit source
admin inventory panel    present in /opt/fluxmind/app.py; status_storage_inventory rendered in Streamlit source
admin events route       present in /opt/fluxmind/api.py and /opt/fluxmind/app.py; runtime event filters available by kind/code/q; authenticated event filter smoke created event=070664f3ced8 filtered_count=1 missing_filter_count=0
admin index freshness    present in /opt/fluxmind/src/admin.py; authenticated local API returned corpus.index.status=fresh
provider failure history present in /opt/fluxmind/src/runtime.py; /query failures append no-secret runtime events
job SQLite state         /opt/fluxmind/jobs/jobs.sqlite3 exists; 17 current rows
scheduled retry smoke    queued retry executed; parent_job_id/not_before present
job transition logs      present in /opt/fluxmind/src/jobs.py and /opt/fluxmind/api.py; authenticated mock image smoke returned logs=[running,succeeded] with artifact_count=1
job retry/cancel UI      present in /opt/fluxmind/app.py with local q/status/kind filters; authenticated job filter smoke created job=f219605a4f28 filtered_count=1 missing_filter_count=0
offline RAG eval         passed in /opt/fluxmind; 5/5 cases, 4/4 provider fixtures, 5/5 recorded answers, aggregate gates passed
live retrieval eval      passed in /opt/fluxmind against local API; 5/5 cases context_coverage=1.00 without model generation
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
python scripts/deploy_sync.py

python scripts/deploy_sync.py --apply --restart

python scripts/health_check.py \
  --url https://smy.hyper-dusty.cloud/ \
  --url https://api-smy.hyper-dusty.cloud/health

python scripts/health_check.py --ssh-host root@100.100.233.26

ssh -o BatchMode=yes root@100.100.233.26 \
  'cd /opt/fluxmind && venv/bin/python scripts/product_readiness.py --format markdown'

ssh -o BatchMode=yes root@100.100.233.26 \
  'cd /opt/fluxmind && venv/bin/python scripts/provider_readiness.py --format markdown'

ssh -o BatchMode=yes root@100.100.233.26 \
  'cd /opt/fluxmind && venv/bin/python scripts/quality_readiness.py --format markdown'

ssh -o BatchMode=yes root@100.100.233.26 \
  'cd /opt/fluxmind && venv/bin/python scripts/api_key_registry.py status --format markdown'

ssh -o BatchMode=yes root@100.100.233.26 \
  'cd /opt/fluxmind && venv/bin/python scripts/storage_schema.py --format markdown'

ssh -o BatchMode=yes root@100.100.233.26 \
  'cd /opt/fluxmind &&
   venv/bin/python scripts/evaluate_rag.py --retrieval-url http://127.0.0.1:18502 --json-report /tmp/fluxmind-api-key-registry-live-report.json &&
   venv/bin/python scripts/quality_readiness.py --live-report /tmp/fluxmind-api-key-registry-live-report.json --require-target small_group'

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
