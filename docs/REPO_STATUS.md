# FluxMind Repository Status

Snapshot time: 2026-06-15 10:23 CST

This file records the current local repository snapshot plus the last verified
clean repository boundary for the completed no-key/local baseline. It is a repo
snapshot, not a production deployment source of truth. For live service state,
use `docs/DEPLOYMENT_STATUS.md` and re-run the refresh commands there.

## Git State

```text
Branch                         main
Remote                         origin git@github.com:Shallow-dusty/FluxMind.git
Tracking                       origin/main
Verified repo baseline         1b7795a test: calibrate live retrieval refs
Remote status at verification  main...origin/main up to date before this final status-note refresh
Current docs refresh scope     final corpus-expansion deployment status note only
Last deployed source/eval baseline 1b7795a test: calibrate live retrieval refs
Live verification follow-up    14-paper corpus rebuild and 74/74 live retrieval refreshed on 2026-06-15 10:23 CST
Ignored runtime/cache state    .venv, __pycache__, .pytest_cache, jobs, metadata, runtime caches
```

The no-key platform release work is now pushed to `origin/main`. Its main
contents were:

```text
Area                 Main contents
-------------------  ---------------------------------------------------------
RAG/eval             live retrieval gates, aggregate regression gates,
                     recorded-answer checks, JSON eval reports
Jobs/workers         durable leases, explicit local worker loop, systemd worker
                     unit, retries, deadlines, cancellation metadata
Corpus/storage       metadata profiles, paper/chunk SQLite mirrors, runtime
                     backup manifest, storage readiness/inventory
Admin/product shell  status/report endpoints, retention preview, runtime
                     events, query usage/cost visibility
Artifacts/images     artifact metadata mirror/integrity, local SVG diagram
                     templates, stable artifact downloads
Execution            local Python/Octave provider hardening, templates,
                     timeout/resource/path/input-limit metadata
Deployment/docs      guarded deploy sync, health-check expansion, deployment
                     evidence updates
```

Push and deployment completed for `a51a060`; the live verification evidence was
then recorded and pushed in `32fca21`.

The documentation cleanup and completion snapshot are included in `a51a060`; the
deployment record follow-up is included in `32fca21`.

This formerly in-progress local work is now committed, pushed to `origin/main`
through application baseline `4f27651`, deployed to Trace-Twin with the guarded sync/restart path,
and verified after the API readiness window. It expands the deterministic
RAG baseline from answer-only checks into a
74-question no-LLM retrieval gate: 28 answer cases, 28 recorded answers, and 46
retrieval-only source/page cases with topic, lane, and ontology coverage gates.
It also adds a local paper-to-code handoff section to `/query/report` for
implementation and code-generation report exports, plus the first local Python
code-output eval cases that verify generated plot/text artifacts and reusable
execution-template output, one local job-backed code-output eval path, and the
first seeded PDF equation/table/figure structure gates exposed through
`GET /corpus/structure` and the Markdown handoff export at
`GET /corpus/structure/report`. The same local slice adds a no-secret runtime
restore dry-run verifier through `scripts/runtime_manifest.py --restore-check`,
authenticated `POST /admin/runtime-manifest/restore-check`, and the matching
Markdown report route, plus Streamlit runtime-panel upload/report controls. The
verifier checks expected runtime groups, manifest contract flags, byte counts,
known files, and SHA-256 hashes without copying, overwriting, deleting, or
restoring files. The local job layer now also supports optional
`idempotency_key` values for immediate and async job submissions, backed by a
durable SQLite `job_idempotency` claim table. Duplicate submissions for the same
job kind and key return the existing persisted job, while different keys and
missing keys preserve distinct-job behavior. The same local job layer now has
bounded async retry metadata: `max_attempts`, `retry_backoff_s`,
`dead_lettered`, and `dead_lettered_at`, so failed queued attempts can requeue
the same durable job until the configured cap is exhausted. API query and job
requests now also accept optional local `owner_id` and `owner_label` metadata.
Those values persist through durable job records, SQLite owner columns, job
transition logs, query runtime events, generated artifact records, and admin
status owner summaries, with default `local-user` / `Local user` values when
omitted. The fields are local metadata only; they do not provide auth,
tenant-isolation, quotas, or billing. The latest local slice also implements an
opt-in no-key `DockerExecutionProvider` selected by
`CODE_EXECUTION_BACKEND=docker`. It keeps the existing Python/Octave-compatible
job routes stable while running code through `docker run --rm` with network
disabled, a bind-mounted per-run workdir, read-only root filesystem, memory,
CPU, and PID limits, dropped capabilities, and `no-new-privileges`. This local
backend is implemented and tested, but it has not been deployed or enabled on
the live service. The latest local policy slice adds `local-safe-v1` execution
preflight before local child processes, Octave runtime lookup, or Docker
container launch. It enforces a configurable Python import allowlist, rejects
obvious shell/package-manager commands, absolute-path literals in common file
constructors, and Octave shell/network/package-install calls, persists
no-secret policy metadata with execution results, exposes the policy through
admin and Streamlit runtime status, and maps blocked jobs to the stable
`execution_policy_violation` error code. The latest execution-observability
slice appends no-secret `code_execution` runtime events for real code job
attempts, including job id, owner metadata, language, backend, status/error
code, duration, artifact count, exit code, and policy metadata without copying
submitted source files, stdout, or stderr. Admin status/report now summarize
recent code execution outcomes by code, status, backend, policy violations, and
duration. Temporary evaluation job-backed code-output cases opt out of event
recording so local eval runs do not pollute the runtime event log. The latest
output-limit slice adds configurable `CODE_EXECUTION_MAX_STDOUT_BYTES` and
`CODE_EXECUTION_MAX_STDERR_BYTES` caps for local Python, Octave-compatible, and
Docker execution. Subprocess stdout/stderr are read through bounded stream
readers so large outputs cannot accumulate unbounded memory; execution metadata
records total observed bytes plus `stdout_truncated`, `stderr_truncated`, and
`output_truncated` flags. Admin status/report expose the configured output
limits. The latest artifact-limit slice adds configurable
`CODE_EXECUTION_MAX_ARTIFACTS`, `CODE_EXECUTION_MAX_ARTIFACT_BYTES`,
`CODE_EXECUTION_MAX_ARTIFACT_TOTAL_BYTES`, and
`CODE_EXECUTION_MAX_ARTIFACT_CANDIDATES` caps for generated-artifact export
across local Python, Octave-compatible, and Docker execution. Artifact
collection now uses bounded directory traversal instead of materializing the
entire output tree, skips oversized or over-limit files, and records scanned,
exported, skipped, byte-count, and `artifact_collection_truncated` metadata.
The same no-secret metadata is carried through code-execution runtime events,
admin status/report summaries, and the Streamlit execution-policy status panel.
The latest execution-alert slice derives local advisory alerts from recent
`code_execution` runtime events. Admin status/report and the Streamlit admin
panel now summarize failure rate, slow-duration threshold hits, policy
violations, stdout/stderr truncation, and artifact collection truncation with
configurable `CODE_EXECUTION_ALERT_MIN_EVENTS`,
`CODE_EXECUTION_ALERT_FAILURE_RATE`, and `CODE_EXECUTION_ALERT_DURATION_MS`
threshold metadata. This is a local no-secret alert baseline, not a production
metrics or tracing stack.
The latest query-latency slice records `duration_ms` on successful `/query`,
`/query/inspect`, and `/query/report` usage events, plus provider-failure events
on the query path. Admin status/report now summarize recent query duration with
average and max milliseconds, and the local health gate includes a query-latency
anchor. This remains metadata-only local observability; it is not a production
tracing, SLO, or metrics stack.
The latest query-alert slice adds configurable metadata-only advisory alerts for
recent query latency. Admin status/report and the Streamlit query-usage JSON
surface now expose `query_usage.alerts` plus `QUERY_ALERT_MIN_EVENTS` and
`QUERY_ALERT_DURATION_MS` threshold metadata without copying questions, answers,
retrieved chunks, or provider response bodies. This is still local advisory
observability, not production alerting or tracing.
The latest provider-alert slice adds configurable metadata-only advisory alerts
for recent `/query` provider failures. Admin status/report now expose
`provider_failures.alerts`, provider failure rate over recent local query
outcomes, repeated failure-code alerts, and
`PROVIDER_FAILURE_ALERT_MIN_EVENTS` / `PROVIDER_FAILURE_ALERT_RATE` threshold
metadata without copying prompts, answers, retrieved chunks, secrets, or raw
provider response bodies. This remains a local advisory baseline, not
production incident management.
The latest job-alert slice adds configurable metadata-only advisory alerts for
local job and worker health. Admin status/report now expose `jobs.alerts` for
recent failed jobs, dead-lettered jobs, expired queued deadlines, and expired
worker leases, controlled by `JOB_ALERT_FAILED_MIN_EVENTS` and
`JOB_ALERT_EXPIRED_MIN_EVENTS`. This uses existing durable job/queue/lease
metadata and does not copy job requests, logs, artifacts, or runtime files into
alert payloads.
The latest API-access-audit slice adds metadata-only `api_access` runtime
events from FastAPI middleware. Events classify token checks as
`not_configured`, `valid`, `missing`, or `invalid` and record only method,
path, status code, duration, credential type, and request ID when present.
Admin status/report and Streamlit now summarize recent access counts by token
status, HTTP status code, and method. No token values, headers, request bodies,
prompts, answers, client IPs, or uploaded/runtime file contents are copied into
these events. `API_ACCESS_AUDIT_ENABLED` controls this local audit layer.
The latest API-rate-limit slice adds a configurable local in-memory request-rate
guard behind `API_RATE_LIMIT_ENABLED`, `API_RATE_LIMIT_MAX_REQUESTS`, and
`API_RATE_LIMIT_WINDOW_S`. When enabled, FastAPI middleware returns HTTP 429
before route handling once the local bucket is exhausted, emits only
metadata-only `api_access` rate-limit fields, and sets `X-RateLimit-*` response
headers. Admin status/report and Streamlit summarize recent rate-limited access
counts plus the configured local threshold. This is a local guardrail, not
identity-backed quotas, billing, or distributed rate limiting.
The latest upload-scan slice adds a local pre-write PDF upload scan behind
`UPLOAD_SCAN_ENABLED`, `UPLOAD_SCAN_MAX_PAGES`,
`UPLOAD_SCAN_REJECT_ENCRYPTED`, and `UPLOAD_SCAN_BLOCK_ACTIVE_CONTENT`. The
Streamlit upload path now validates PDF magic and PyMuPDF parseability before
writing upload bytes, rejects encrypted PDFs by default, blocks common
active-content markers such as JavaScript, launch actions, embedded files, rich
media, and XFA, and records only metadata-only `upload_scan` runtime events
with reason codes, byte counts, page counts, and threshold config. Admin
status/report and Streamlit summarize recent upload scan outcomes. The event
payloads do not copy filenames, uploaded contents, checksums, request bodies,
client IPs, prompts, or answers. This is a local abuse guardrail, not a
production antivirus, sandbox-scanning, identity-backed quota, or data deletion
system.
The latest retention-delete slice adds a guarded local deletion path behind
`RETENTION_DELETE_ENABLED`, defaulting to disabled. `GET /admin/retention`
remains the preview path, while authenticated `POST /admin/retention/delete`
can delete the same bounded local upload/artifact candidate set only when the
flag is explicitly enabled. The delete path excludes artifact SQLite metadata
files, records aggregate-only `retention_delete` runtime events, and the
Streamlit admin panel only shows the delete action when the same flag is on.
This is local data-retention plumbing, not identity-backed deletion, legal
hold, audit-log retention, or production privacy/compliance automation.
The latest metrics-export slice adds authenticated `GET /admin/metrics` plus a
Streamlit metrics download. The export renders existing admin summaries as
Prometheus/OpenMetrics-style local-window gauges without copying owner IDs,
request IDs, paths, prompts, answers, uploaded content, filenames, or artifact
contents. This is a local scrapeable observability baseline, not a production
metrics backend, retention policy, or alert-routing stack.
The latest retrieval-trace slice adds metadata-only `retrieval_trace` runtime
events for successful `/query`, `/query/inspect`, `/query/report`, and
`/query/retrieve` calls. The events record endpoint, answer mode, context count,
source/page completeness counts, citation status when available, duration, and
whether an LLM provider was called. Admin status/report, Streamlit, and
`/admin/metrics` summarize those local events without prompts, answers,
retrieved text, source paths, owner IDs, or request IDs. This is local
retrieval observability, not production tracing or alert routing.
The latest retrieval-alert slice derives metadata-only advisory alerts from
recent `retrieval_trace` events. Admin status/report, Streamlit, and metrics now
summarize empty-retrieval rate, source/page incomplete rate, citation failure
rate, and alert counts controlled by `RETRIEVAL_TRACE_ALERT_MIN_EVENTS`,
`RETRIEVAL_TRACE_ALERT_EMPTY_RATE`,
`RETRIEVAL_TRACE_ALERT_SOURCE_PAGE_INCOMPLETE_RATE`, and
`RETRIEVAL_TRACE_ALERT_CITATION_FAILURE_RATE`. Alert payloads contain only
counts, rates, thresholds, and codes.
The latest storage-schema slice adds `src.storage_schema` plus admin/report/
metrics/Streamlit exposure for a no-secret local storage-schema inventory. It
checks schema version, JSON/JSONL shape, and expected SQLite table/column
presence across corpus, chunk, job, artifact, and runtime-event stores without
returning row contents, prompts, answers, filenames, owner IDs, request IDs,
source paths, or runtime file contents. `scripts/storage_schema.py` exposes the
same check as a local or target-root CLI preflight with JSON/Markdown output and
a nonzero exit code on drift. The current local CLI/admin snapshot reports
`ok=true`, 7 stores, and 0 schema problems.

The 2026-06-13 pass also refreshed the agent-facing bootstrap docs `CLAUDE.md`
and `AGENTS.md` so they describe the current dual-entrypoint (Streamlit + FastAPI)
runtime, the no-key job/storage/artifact subsystem, the common build/test/run
commands, and the documentation discipline, instead of the earlier
Streamlit-only summary. These are documentation-only changes and keep the
docs-guard tests green.

The same 2026-06-13 pass also adds a CI-safe slice of the previously planned
local eval breadth: a second Python execution template `pmsm_current_step`
(PMSM q-axis current step response producing CSV/SVG), a second job-backed Python
code-output eval case using it (4 code-output cases total, 2 job-backed), a
second Octave template `smc_sign_switching`, and unit tests for the new Python
template and Octave template breadth. Broader Octave *execution* eval and a
real-PDF algorithm-caption acceptance case stay deferred (no `octave` binary in
CI/runtime; no curated library paper with a numbered `Algorithm N` block). Test
count is now 282 and `evaluate_rag.py` reports 4 code-output cases.

## Current Documentation Set

```text
Document                               Role
-------------------------------------  ---------------------------------------
README.md                              Project entrypoint and quick start
AGENTS.md                              Agent (Codex) bootstrap; mirrors CLAUDE.md
CLAUDE.md                              Claude Code bootstrap: commands + architecture
docs/README.md                         Documentation index and ownership map
docs/REPO_STATUS.md                    This git/status snapshot
docs/ARCHITECTURE.md                   Runtime and module architecture
docs/BACKLOG.md                        Work packages and acceptance criteria
docs/DEPLOYMENT_STATUS.md              Live deployment snapshot and commands
docs/FEATURE_AUDIT.md                  Feature inventory, route coverage, and gaps
docs/PLATFORM_AUDIT_AND_ROADMAP.md     Broader platform audit and roadmap
docs/PRODUCTION_GAP_AND_MARKET_RESEARCH.md  Production gap and market research
docs/demo-script.md                    Chinese demo script and Q&A
docs/handover.html                     Single-file presentation handover
```

## Historical Local Verification Run

This 2026-06-13 run verified the platform/eval slice before it was pushed.
Commands ran from
`/home/shallow/04.AI-Prism/11.FluxMind` using `.venv`:

```text
Command                                                                 Result
----------------------------------------------------------------------  ------
.venv/bin/python --version                                             Python 3.13.11
.venv/bin/python -m pytest                                             pass, 282 tests, 2 warnings
.venv/bin/python scripts/health_check.py                               pass, including query-latency,
                                                                        query-alert, provider-alert,
                                                                        job-alert, API-access-audit,
                                                                        API-rate-limit, upload-scan,
                                                                        retention-delete,
                                                                        metrics-export,
                                                                        retrieval-trace,
                                                                        retrieval-alerts,
                                                                        storage-schema,
                                                                        artifact-limit, execution-alert,
                                                                        and docs drift anchors
.venv/bin/python scripts/storage_schema.py --output /tmp/...            pass, ok=true, 7 stores,
                                                                        0 problems, JSON at
                                                                        /tmp/fluxmind-storage-schema-cli.json
.venv/bin/python scripts/storage_schema.py --format markdown --output.. pass, Markdown at
                                                                        /tmp/fluxmind-storage-schema-cli.md
.venv/bin/python scripts/evaluate_rag.py                               pass, 28 answer cases,
                                                                        46 retrieval-only cases,
                                                                        8 code-output cases,
                                                                        12 PDF structure cases,
                                                                        28 recorded answers
.venv/bin/python scripts/evaluate_rag.py --json-report /tmp/...        pass, no-secret JSON report
                                                                        at /tmp/fluxmind-eval-report-storage-schema-cli.json
.venv/bin/python scripts/runtime_manifest.py --output /tmp/...         pass, no-secret manifest at
                                                                        /tmp/fluxmind-runtime-manifest-storage-schema-cli.json
.venv/bin/python scripts/runtime_manifest.py --restore-check ...       pass, ok=true, 6 groups,
                                                                        5 checked files,
                                                                        manifest_errors=0,
                                                                        0 missing/mismatched
git diff --check                                                       pass
git status --short --branch                                            historical pre-push state:
                                                                        main...origin/main [ahead 7],
                                                                        34 modified files,
                                                                        4 untracked files
```

Follow-up status refresh on 2026-06-15 02:31 CST:

```text
Command                                                                 Result
----------------------------------------------------------------------  ------
git status --porcelain=v1 --branch                                     main...origin/main, no local
                                                                        changes before this
                                                                        deployment-record refresh
git rev-parse --short HEAD / origin/main                               both 17aacc3
git rev-list --left-right --count main...origin/main                    0 0
.venv/bin/python scripts/health_check.py --url ...                     pass; public HTTPS UI/API
                                                                        returned 200
.venv/bin/python scripts/health_check.py --ssh-host ...                pass after one normal API
                                                                        warmup-window retry; UI/API/
                                                                        worker/cloudflared/docker
                                                                        active, ports 18501/18502
                                                                        listening, local API health OK,
                                                                        active_papers=6,
                                                                        faiss_index_bytes=786477,
                                                                        chunk_metadata_rows=512,
                                                                        chunk_metadata_sources=6,
                                                                        index fresh, Docker execution
                                                                        not configured, local storage
                                                                        readiness available, retrieval
                                                                        and admin metrics smokes OK
```

Follow-up status refresh on 2026-06-15 03:15 CST:

```text
Command                                                                 Result
----------------------------------------------------------------------  ------
git rev-parse --short HEAD / origin/main                               both 3cfa426
.venv/bin/python -m pytest                                             pass, 286 tests, 2 warnings
.venv/bin/python scripts/health_check.py                               pass, including startup
                                                                        readiness and FAISS log-noise
                                                                        anchors
.venv/bin/python scripts/evaluate_rag.py                               pass, all offline RAG/eval gates
/usr/bin/time ... python -c 'import api'                               local import api 0:00.88,
                                                                        remote import api 0:15.52
.venv/bin/python scripts/deploy_sync.py --apply --restart              synced source to /opt/fluxmind and
                                                                        restarted API/UI/worker
.venv/bin/python scripts/health_check.py --url ...                     pass; public HTTPS UI/API
                                                                        returned 200
.venv/bin/python scripts/health_check.py --ssh-host ...                pass on rerun after the shorter
                                                                        API bind window; UI/API/worker/
                                                                        cloudflared/docker active, ports
                                                                        18501/18502 listening, /health OK,
                                                                        /ready OK, retrieval/admin smokes OK
```

Follow-up status refresh on 2026-06-15 04:23 CST:

```text
Command                                                                 Result
----------------------------------------------------------------------  ------
git rev-parse --short HEAD / origin/main                               both 391ac7f
.venv/bin/python -m pytest                                             pass, 307 tests, 2 warnings
.venv/bin/python -m coverage run -m pytest &&
.venv/bin/python -m coverage report --fail-under=85                    pass, 86% total branch coverage
.venv/bin/python scripts/evaluate_rag.py                               pass, all offline RAG/eval gates
rsync papers/library/ ... /opt/fluxmind/papers/library/                synced 5 new seed PDFs plus
                                                                        manifest without deleting runtime
                                                                        uploads/state
PUT /corpus/active + async index rebuild job 8c4f1995a02a              pass, 11 active papers,
                                                                        800 chunks, index fresh
.venv/bin/python scripts/deploy_sync.py --apply --restart              synced source to /opt/fluxmind and
                                                                        restarted API/UI/worker; runtime
                                                                        state excludes preserved
.venv/bin/python scripts/health_check.py --url ...                     pass; public HTTPS UI/API
                                                                        returned 200
.venv/bin/python scripts/health_check.py --ssh-host ...                pass on rerun after the normal API
                                                                        bind window; active_papers=11,
                                                                        faiss_index_bytes=1228845,
                                                                        chunk_metadata_rows=800,
                                                                        chunk_metadata_sources=11,
                                                                        index fresh, /ready ready,
                                                                        retrieval/admin smokes OK
authenticated /query/retrieve smoke                                    pass; q="MRAS flux linkage observer
                                                                        PMSM" returned ok=true and first
                                                                        context from the new Zhu 2024 MRAS
                                                                        flux-linkage observer paper
```

Local coverage-hardening follow-up after the 04:23 deployment:

```text
Command                                                                 Result
----------------------------------------------------------------------  ------
.venv/bin/python -m pytest                                             pass, 328 tests, 2 warnings
.venv/bin/python -m coverage run -m pytest &&
.venv/bin/python -m coverage report --fail-under=88                    pass, 88% total branch coverage
scope                                                                  local tests/CI/docs only; no runtime
                                                                        service deploy required
```

Runtime-state hardening deployment on 2026-06-15 08:21 CST:

```text
Command                                                                 Result
----------------------------------------------------------------------  ------
git rev-parse --short HEAD / origin/main                               both 4f27651
.venv/bin/python -m pytest                                             pass, 328 tests, 2 warnings
.venv/bin/python -m coverage run -m pytest &&
.venv/bin/python -m coverage report --fail-under=88                    pass, 88% total branch coverage
.venv/bin/python scripts/evaluate_rag.py                               pass, all offline RAG/eval gates
.venv/bin/python scripts/health_check.py                               pass, local/docs/runtime anchors
.venv/bin/python scripts/deploy_sync.py --apply --restart              synced source to /opt/fluxmind and
                                                                        restarted API/UI/worker; .coverage
                                                                        excluded
ssh ... test ! -e /opt/fluxmind/.coverage                              pass, coverage_absent
.venv/bin/python scripts/health_check.py --url ...                     pass; public HTTPS UI/API
                                                                        returned 200
.venv/bin/python scripts/health_check.py --ssh-host ...                pass; services active, active_papers=11,
                                                                        chunk_metadata_rows=800,
                                                                        index fresh, retrieval/admin smokes OK
authenticated /corpus/status smoke                                     pass; papers=11, active=11,
                                                                        indexed=11, chunks=800,
                                                                        index=fresh
```

Documentation sweep on 2026-06-15 08:32 CST:

```text
Command                                                                 Result
----------------------------------------------------------------------  ------
git status --short --branch                                             main...origin/main clean at
                                                                        3b8ecd7 before docs edits
.venv/bin/python scripts/health_check.py --url ...                     pass; public HTTPS UI/API
                                                                        returned 200
.venv/bin/python scripts/health_check.py --ssh-host ...                pass; services active,
                                                                        active_papers=11,
                                                                        chunk_metadata_rows=800,
                                                                        index fresh
authenticated /corpus/status smoke                                     pass; papers=11, active=11,
                                                                        indexed=11, chunks=800,
                                                                        index=fresh
GitHub API competitor snapshot                                         refreshed for market doc
scope                                                                  docs/status/test guards only;
                                                                        deployed app code remains
                                                                        4f27651
```

Documentation sync verification on 2026-06-15 08:48 CST:

```text
Command                                                                 Result
----------------------------------------------------------------------  ------
git push origin main                                                     pushed docs refresh
                                                                        20d75e5
.venv/bin/python scripts/deploy_sync.py --apply                         synced docs, health_check.py,
                                                                        and docs-status test guard to
                                                                        /opt/fluxmind without restart
.venv/bin/python scripts/health_check.py --url ...                     pass; public HTTPS UI/API
                                                                        returned 200
.venv/bin/python scripts/health_check.py --ssh-host ...                pass; services active,
                                                                        active_papers=11,
                                                                        chunk_metadata_rows=800,
                                                                        index fresh
scope                                                                  docs-only sync; deployed
                                                                        application-code baseline
                                                                        remains 4f27651
```

## Latest Deployment Snapshot

The latest live deployment snapshot was refreshed after guarded corpus sync,
index rebuild, live retrieval evaluation, and HTTPS/SSH health checks on
2026-06-15 10:23 CST in
`docs/DEPLOYMENT_STATUS.md`. The platform/eval/API/runtime-restore/
job-idempotency/retry-dead-letter/ownership/Docker-execution/execution-policy/
execution-observability/output-limits/artifact-limits/execution-alerts/
query-latency/query-alerts/provider-alerts/job-alerts/API-access-audit/
API-rate-limit/upload-scan/retention-delete/metrics-export/retrieval-trace/
retrieval-alerts/storage-schema work above plus the API startup readiness,
import-latency fixes, coverage gate, 14-paper seed corpus, API token comparison
hardening, runtime JSON/JSONL state-file tolerance, and `.coverage` deploy
exclude are now synced or rebuilt on Trace-Twin through the current
source/runtime boundary. The latest remote corpus status is `papers=14`,
`active=14`, `indexed=14`, `chunks=987`, and `index=fresh`; live retrieval eval
passes `74/74`. External
providers, hosted sandboxes, MATLAB,
identity-backed quotas/billing, distributed storage, and distributed workers
remain intentionally disabled or planned.
Highlights from that deployment snapshot:

```text
Service state       UI/API/worker/cloudflared/docker active
Listeners           0.0.0.0:18501 and 0.0.0.0:18502
Local API health    {"status":"ok"}
Local API readiness {"status":"ready","warmup":{"status":"ready","ready":true,"error":""}}
Model config        LLM_MODEL=mimo-v2.5-pro
Embedding model     /opt/fluxmind/models/all-MiniLM-L6-v2
Active papers       11
FAISS index bytes   1228845
Chunk rows          800 across 11 source paths
Index freshness     True
Storage readiness   local metadata/object storage available
Docker execution    configured=False available=False reason=not_configured
Disk                /dev/vda3 40G total, 21G free, 46% used
```

## Immediate Boundary

- Runtime state remains git-ignored: `papers/`, `faiss_index/`, `artifacts/`,
  `jobs/`, `metadata/`, `.env`, virtual environments, caches, and bytecode.
- The retrieval-eval/code-output/PDF-structure/report/runtime-restore/
  job-idempotency/retry-dead-letter/ownership/Docker-execution/
  execution-policy/execution-observability/output-limits/artifact-limits/
  execution-alerts/query-latency/query-alerts/provider-alerts/job-alerts/
  API-access-audit/API-rate-limit/upload-scan/retention-delete/
  metrics-export/retrieval-trace/retrieval-alerts/storage-schema/API work plus
  the eval-breadth, coverage/corpus-hardening, runtime-state-hardening, and
  deploy-exclude slices are verified, committed, pushed to `origin/main` through
  application baseline `4f27651`, deployed to Trace-Twin, and post-restart
  verified. Documentation commits may be newer than the deployed application-code
  baseline.
- Deployment facts should not be inferred from git state alone because
  `/opt/fluxmind` is a synchronized source tree, not a git checkout.
