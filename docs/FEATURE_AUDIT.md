# FluxMind Feature Audit

Last updated: 2026-06-20

This document inventories the currently implemented FluxMind feature surface and
the evidence that checks it. It is a no-secret audit document: do not copy API
tokens, `.env` values, uploaded PDF contents, FAISS files, job logs, generated
artifacts, or runtime database contents into it.

For repo/worktree status, use `docs/REPO_STATUS.md`. For live deployment state,
use `docs/DEPLOYMENT_STATUS.md` and re-run the refresh commands there before
making deployment decisions.

## Current Verification Command Set

```text
Command                                                               Result
--------------------------------------------------------------------  -------------------------------
.venv/bin/python -m pytest                                           pass, 602 tests, 2 known warnings
.venv/bin/python -m coverage run -m pytest &&
.venv/bin/python -m coverage report --fail-under=88 --sort=cover    pass, 602 tests, 89% total branch coverage
.venv/bin/python scripts/evaluate_rag.py                             pass, 42 answer cases and
                                                                      65 retrieval-only cases,
                                                                      13 code-output cases,
                                                                      30 PDF structure cases,
                                                                      42 recorded answers
.venv/bin/python scripts/health_check.py                             pass, local/docs/query-latency/query-alert/
                                                                      provider-alert/job-alert/API-access-audit/
                                                                      API-rate-limit/upload-scan/
                                                                      retention-delete/metrics-export/retrieval-trace/
                                                                      retrieval-alerts/storage-schema/artifact-limit/
                                                                      execution-alert/provider-readiness/quality-readiness/
                                                                      API-key-registry/product-registry/
                                                                      product-quota/product-RBAC/
                                                                      remote-SSH safety-anchor sync/
                                                                      product-registry-management/
                                                                      product-registry-read-RBAC/
                                                                      Streamlit-product-registry-flag/
                                                                      API-key-create-output-guard/
                                                                      share-link-registry/
                                                                      product-activation-rehearsal/
                                                                      collaboration-readiness/
                                                                      object-storage-manifest/
                                                                      object-storage-manifest-verifier/
                                                                      job-store-manifest/
                                                                      job-store-manifest-verifier/
                                                                      provider-runtime-rehearsal/
                                                                      activation-suite/
                                                                      OpenAPI-contract/
                                                                      quality-evidence-plan/
                                                                      readiness-CLI-error-sanitizer/
                                                                      live-eval-request-id-redaction/
                                                                      readiness/log-noise anchors
.venv/bin/python scripts/storage_schema.py --output /tmp/...         pass, ok=true, 10 stores, 0 problems
.venv/bin/python scripts/api_key_registry.py status --format...      pass, backend=none, available=false,
                                                                      active_keys=0, secrets_exported=false
.venv/bin/python scripts/product_registry.py status --format...      pass, backend=none, available=false,
                                                                      workspaces=0, secrets_exported=false
.venv/bin/python scripts/share_link_registry.py status --format...   pass, backend=none, available=false,
                                                                      active_links=0, secrets_exported=false
.venv/bin/python scripts/collaboration_readiness.py --format...      pass, ok=true, safe_default_ready=true,
                                                                      activation_ready=false with expected
                                                                      private-corpus/share-link blockers
.venv/bin/python scripts/platform_migration_preflight.py --format... pass, preflight_ok=true,
                                                                      activation_ready=false with expected
                                                                      external-backend blockers
.venv/bin/python scripts/platform_migration_rehearsal.py --format... pass, rehearsal_ok=true,
                                                                      staged restore/schema checks pass
.venv/bin/python scripts/platform_migration_rehearsal.py
  --include-object-manifest --format...                              pass, opaque object manifest
                                                                      without paths, buckets, or contents
.venv/bin/python scripts/platform_migration_rehearsal.py
  --verify-object-manifest /tmp/... --format...                      pass, object manifest verifies
                                                                      against local runtime with 0
                                                                      missing/mismatched/extra objects
.venv/bin/python scripts/platform_migration_rehearsal.py
  --include-job-store-manifest --format...                           pass, no-secret job-store manifest
                                                                      without job payloads or raw IDs
.venv/bin/python scripts/platform_migration_rehearsal.py
  --verify-job-store-manifest /tmp/... --format...                   pass, job-store manifest verifies
                                                                      against local runtime with 0
                                                                      missing/mismatched/extra jobs
GET /admin/platform-migration-rehearsal and report route              pass, on-demand no-secret JSON/Markdown
                                                                      surface; raw manifests, paths,
                                                                      runtime contents, job payloads,
                                                                      and raw IDs not echoed
.venv/bin/python scripts/product_readiness.py --format markdown      pass, local_foundation_ready=true,
                                                                      activation_ready=false with expected
                                                                      identity/quota/billing blockers and
                                                                      product-quota/product-RBAC guard advisories
.venv/bin/python scripts/product_readiness.py --require-activation   pass, exits 1 because activation_ready=false
.venv/bin/python scripts/product_activation_rehearsal.py
  --format markdown --require-activation                             pass, ok=true, local SQLite activation
                                                                      drill with workspace-isolation
                                                                      denial; raw tokens and paths exported=false
GET /admin/product-activation-rehearsal and report route              pass, on-demand no-secret JSON/Markdown
                                                                      surface; raw tokens, paths, prompts,
                                                                      answers, and external account data
                                                                      not echoed
.venv/bin/python scripts/provider_readiness.py --format markdown     pass, local_foundation_ready=true,
                                                                      activation_ready=false with expected
                                                                      provider/MATLAB blockers
.venv/bin/python scripts/provider_readiness.py --require-activation  pass, exits 1 because activation_ready=false
.venv/bin/python scripts/provider_runtime_rehearsal.py
  --format markdown --require-local-foundation                       pass, ok=true, local provider runtime
                                                                      drill including execution abuse-policy
                                                                      denial; external_activation_ready=false
GET /admin/provider-runtime-rehearsal and report route                pass, on-demand no-secret JSON/Markdown
                                                                      surface; raw paths, prompts,
                                                                      content, credentials, and
                                                                      external provider data not echoed
.venv/bin/python scripts/quality_readiness.py --format markdown      pass, local_foundation_ready=true,
                                                                      small_group/community false without supplied
                                                                      live report evidence; target
                                                                      gap summary lists current/
                                                                      expected/gap per maturity target
                                                                      plus evidence source requests
                                                                      and no-secret collection plans
GET/POST /admin/quality-readiness and report route                    pass, on-demand no-secret JSON/Markdown
                                                                      surface with live-report input;
                                                                      raw report paths, prompts, answers,
                                                                      file contents, and concrete
                                                                      API tokens not echoed
.venv/bin/python scripts/activation_suite.py
  --format markdown --require-target local_foundation                 pass, local_foundation_ready=true,
                                                                      full_activation_ready=false with expected
                                                                      product/provider/platform/community blockers;
                                                                      raw child reports, paths, and secrets
                                                                      exported=false; next quality evidence
                                                                      target/gaps/sources and full activation
                                                                      action plan projected
.venv/bin/python scripts/openapi_contract.py
  --format markdown --require-local-contract                          pass, local_contract_ready=true,
                                                                      69 routes, 76 operations,
                                                                      52 required operations,
                                                                      protected auth headers
                                                                      covered; operation fingerprint
                                                                      emitted; raw schema
                                                                      exported=false
GET /admin/openapi-contract and report route                           pass, on-demand no-secret JSON/Markdown
                                                                      surface; raw schema and
                                                                      request examples not echoed
POST /admin/openapi-contract verify/report route                      pass, snapshot drift JSON/Markdown
                                                                      surface compares only no-secret
                                                                      report fields and fingerprints;
                                                                      malformed/raw-schema-shaped
                                                                      snapshot input is reduced to
                                                                      no-secret reason codes/valid flags
GET /admin/activation-suite and report route                          pass, on-demand no-secret JSON/Markdown
                                                                      surface; not collected by default
                                                                      /admin/status refresh
POST /admin/activation-suite live report input                        pass, in-memory eval JSON evidence
                                                                      updates suite gates without echoing
                                                                      raw report paths, prompts, answers,
                                                                      or file contents; next quality
                                                                      evidence target moves from
                                                                      small_group to community when
                                                                      live retrieval evidence is present,
                                                                      and action plan keeps commands
                                                                      placeholder-only
.venv/bin/python scripts/quality_readiness.py --require-target...    pass, exits 1 because community_ready=false
.venv/bin/python scripts/evaluate_rag.py --json-report /tmp/...      pass, /tmp/fluxmind-eval-report-storage-schema-cli.json,
                                                                      includes quality_maturity targets
server-local evaluate_rag.py --retrieval-url ... --json-report ...   02:14 snapshot, 107/107 live retrieval
                                                                      cases and 24/24 regression gates pass
.venv/bin/python scripts/runtime_manifest.py --output /tmp/...       pass, /tmp/fluxmind-runtime-manifest-storage-schema-cli.json
.venv/bin/python scripts/runtime_manifest.py --restore-check ...     pass, ok=true, 6 groups, 5 checked files,
                                                                      manifest_errors=0, 0 missing/mismatched
.venv/bin/python scripts/health_check.py --url ...                   02:14 snapshot, HTTPS UI/API health 200
curl https://api-smy.hyper-dusty.cloud/health                        02:14 snapshot, HTTPS API 200
.venv/bin/python scripts/health_check.py --ssh-host root@100.100...  02:14 snapshot, live runtime green,
                                                                      active_papers=30, chunks=1934
```

The 2026-06-19 local audit added focused coverage for blank and unsafe
`X-Request-ID` handling across query responses and API-access audit events,
`/query/report` download responses preserving request/quota headers, and stable
artifact-ID export through the full current local job history instead of only
the recent job-list window. Unsafe request IDs with bearer/token/secret-like
values or invalid correlation-id characters are not echoed into response IDs or
API-access runtime events; unsafe legacy runtime-event request IDs are projected
as `request_id_present`/`request_id_redacted` booleans instead of raw values.
Live answer/retrieval eval JSON reports now use the same no-secret boundary:
they expose `request_id_present` and `request_id_redacted` evidence flags
instead of copying raw request IDs from live API responses into archived quality
reports.
The same audit now also makes durable job-store manifest verification stable for
scheduled queued jobs by evaluating due/scheduled fields against the manifest
timestamp, and it detects idempotency-claim metadata mismatches without
exporting idempotency keys or raw job IDs.
Provider quota/cost guard, query-cost parsing, and artifact public cost
metadata now reject non-finite `NaN`/`Infinity` values and extreme decimal
exponents as invalid local configuration, so bad no-secret cost settings cannot
crash status/readiness or provider pre-call guard decisions and cannot produce
unbounded public strings.
Provider usage extraction now treats token counts as optional no-secret
metadata: malformed provider usage fields are skipped in favor of the next valid
provider field, zero-token counts are preserved, and totals are derived from
prompt/completion counts when no valid total is returned.
API-access runtime events also replace raw request paths with route presence and
a short route-template fingerprint. Runtime-event messages also redact bare
`sk-...` secret-like tokens.
Artifact SQLite sync also preserves sibling artifacts for the
same job while removing stale artifacts for jobs that were actually refreshed.
The admin runtime-event viewers now sanitize event metadata before returning
`/admin/events` results or rendering the Streamlit runtime-events panel, and
both apply `q` search to that sanitized projection. Prompts, answers, owner
identifiers, product user/workspace/API-key identifiers, source paths,
camelCase/PascalCase path or URL fields, token values, filenames, and raw content
cannot be returned or searched through those viewers. Aggregate
workspace/user/member counts remain visible.
Top-level event messages with URL/path/token/prompt/answer-like value assignments
are also redacted in the admin-facing projection.
Artifact list/search responses, job artifact sub-objects, Streamlit artifact
views, generated-artifact RAG context, and download filenames now use a public
projection that omits raw artifact URIs, source paths, titles, owner IDs/labels,
prompts, and source-reference values while preserving stable artifact IDs,
safe metadata flags/counts, checksum/byte-count presence, and explicit
`owner_id` filtering for local operator workflows. Download filenames reject
unexpected artifact IDs, and public cost summaries accept bounded finite numeric
values only. Local artifact download/export resolution accepts only local
absolute `file://` artifact URIs with no host or `localhost`, returns a
canonical artifact-root path to callers, and rejects nonlocal file artifact URIs
before any file response is built.
`GET /jobs` now uses no-secret job summaries and safe search/status projections
instead of raw request payloads, execution results, logs, owner IDs/labels,
idempotency keys, or artifact metadata; exact `owner_id` filtering remains
available for local operator workflows, owner IDs/labels are reduced to
presence flags, and the Streamlit latest-job panel renders the same no-secret
summary boundary. Bad SQLite job mirror payloads are skipped or refreshed from
append-only JSONL rather than breaking list/get/idempotency/claim paths.
Corpus profile status-report downloads now use one shared API/UI helper that
builds the `Content-Disposition`/download filename from the normalized saved
profile ID instead of the raw path parameter, and hashes secret-like profile IDs
before they reach filenames.
`quality_readiness.py` now
also exposes a no-secret target gap summary so each maturity target reports
current count, expected count, count gap, and live-answer quality gaps directly
in the CLI output without exporting live report paths or filenames. It also
emits evidence requests that classify remaining target gaps by
`corpus_manifest`, `eval_baseline`, or `live_eval_report` source, plus
evidence collection plans that turn next-target and community gaps into
placeholder `evaluate_rag.py`/`quality_readiness.py` commands without embedding
concrete URLs, report paths, prompts, answers, source content, or API tokens.
`activation_suite.py` now also projects a full activation action plan that
groups actual product readiness, provider activation, platform migration
activation, and community-quality blockers into no-secret command/verification
handoffs, while keeping external activation disabled until real operator
configuration exists. CLI/API/UI activation-suite entrypoints now also pass the
generated FastAPI schema into the aggregate, so OpenAPI contract readiness is a
local foundation gate without embedding the raw schema.
`openapi_contract.py` now adds a no-secret OpenAPI contract gate for frontend/API
split work. It validates required route/method coverage, operation summaries and
IDs, response declarations, protected auth header declarations, and route-group
coverage from the generated FastAPI schema without exporting the raw schema. It
also emits a stable operation fingerprint and can compare the current no-secret
report with a prior no-secret JSON snapshot to flag contract drift without
exporting the raw schema. Snapshot verification now normalizes only whitelisted
fields and reports malformed/raw-schema-shaped input through reason codes and
valid flags, not by echoing arbitrary snapshot values; count fields must be
bounded non-negative JSON integers. The explicit readiness/rehearsal admin
routes also emit metadata-only `admin_check` runtime events when API access
auditing is enabled, carrying check names, ok/blocked state, counts, booleans,
and blocker counts instead of uploaded snapshots, raw reports, fingerprints,
paths, prompts, answers, tokens, or child payloads. Admin status/report,
metrics, and Streamlit now summarize those checks, and latest admin-check event
metadata is reduced to a fixed safe-key set before display. Unsafe legacy
admin-check code/check labels are grouped as `invalid`, and negative blocker
counts are clamped before report and metrics totals.
No-secret readiness and rehearsal CLIs now share a safe OSError projection:
output-path or local-file failures keep safe diagnostic text while redacting
paths, URLs, bearer/sk-style tokens, and token/secret-like assignments; direct
CLI tests and `scripts/health_check.py` guard that wiring.
Admin status/report latest-event summaries
share the same runtime-event metadata/request-ID sanitizer and replace raw request IDs with
`request_id_present` and, when needed, `request_id_redacted` booleans; job and artifact summaries now expose owner
counts and ownership-source buckets instead of owner IDs.
This is a local code/test audit update; live deployment state remains owned by
`docs/DEPLOYMENT_STATUS.md`.

## Feature Groups

```text
Group                         Status        Evidence and remaining gap
----------------------------  ------------  ---------------------------------------------------------
RAG query and inspection      verified      /query, /query/inspect, /query/retrieve, /query/report,
                                            /corpus/structure, /corpus/structure/report;
                                            offline eval, citation tests, code-output artifact and
                                            template checks, PDF structure checks, and paper-to-code
                                            report handoff tests pass. quality-readiness now reports
                                            staged self-use/small-group/community gaps and safe
                                            evidence collection plans. Live QA breadth is still
                                            limited.
Corpus and profile control    verified      30-paper curated seed library plus paper/chunk/status/
                                            active/profile routes exist; local
                                            JSON/SQLite store is inspectable. Multi-user ownership and
                                            production DB/object storage remain planned.
                                            Uploaded PDFs have a local pre-write scan guard.
Local job and worker bridge   verified      immediate and async job routes, idempotency keys,
                                            owner metadata, bounded retry/dead-letter state,
                                            cancellation, leases, and worker service exist.
                                            Job-list search/listing uses no-secret
                                            summaries instead of raw request/result/log
                                            payloads or owner labels.
                                            Admin status/report include local job-health alerts.
                                            Distributed queue remains planned.
Artifacts and exports         verified      artifact list/download, checksums, and metadata mirrors
                                            exist, including source-job owner metadata,
                                            full-history stable-ID export, symlink
                                            artifact rejection, nonlocal file-URI
                                            rejection, canonical artifact-root
                                            export paths, store-level
                                            symlink/source write guards,
                                            SQLite-cache fallback, and public no-secret
                                            artifact projections. Durable
                                            object storage remains planned.
Admin and runtime status      verified      status/report/retention/events/runtime-manifest and
                                            restore-check routes plus Streamlit upload/report UI
                                            exist with no-secret output, sanitized admin event
                                            metadata, job/artifact owner-count and
                                            ownership-source summaries,
                                            query duration summaries, query latency
                                            alerts, provider-failure alerts, code-execution event
                                            summaries, job-health alerts, API access audit summaries,
                                            API rate-limit summaries, upload-scan summaries, guarded
                                            retention-delete controls, local advisory alerts, and
                                            no-secret metrics export text plus metadata-only
                                            retrieval trace summaries, retrieval-quality alerts,
                                            platform-readiness blocker summaries, and
                                            provider-readiness blocker summaries.
                                            Identity-aware admin remains planned.
No-key execution providers    verified      local Python, Octave-compatible, and opt-in Docker
                                            providers prove the job/artifact contract. Request-level
                                            package/policy preflight and no-secret execution outcome
                                            events exist; stdout/stderr capture and generated-artifact
                                            export are byte/count-bounded, with local alert summaries.
                                            Hosted execution, deeper abuse controls, and MATLAB
                                            licensing remain planned; provider-readiness now exposes
                                            those activation blockers explicitly.
Mock diagram generation       verified      local SVG templates and artifact capture exist. Real image
                                            provider activation remains disabled and is reported by
                                            provider-readiness blocker codes.
Deployment and health gates   verified      guarded deploy sync, local health, HTTPS health, and SSH
                                            health are present. Live facts must still be refreshed.
Product platform layer        incomplete    local product-readiness CLI/admin/report/metrics/UI
                                            surface now separates local foundations from real
                                            identity/quota/billing activation. Local API-key
                                            lifecycle is implemented through a hash-only SQLite
                                            registry; create requires JSON output so the one-time
                                            raw token is not silently lost in a non-token Markdown
                                            report. Local users/workspaces/quota limits/usage/
                                            billing attribution are implemented through an optional
                                            SQLite product registry, and `/query*` routes can enforce
                                            local request quotas when the product quota guard is
                                            explicitly enabled. The same registry now supplies local
                                            role checks: viewer/member/admin/owner memberships gate
                                            query access, job submit/manage, and corpus/index/admin
                                            destructive writes when the RBAC guard is explicitly
                                            enabled. Local operator management is exposed through
                                            `/admin/product-registry/*` and the Streamlit admin
                                            panel when the SQLite backend is enabled; workspace
                                            list/detail and permission-check reads are also guarded
                                            by local admin-write RBAC when that guard is enabled.
                                            Streamlit's direct local management forms require the
                                            separate explicit
                                            `FLUXMIND_STREAMLIT_PRODUCT_REGISTRY_MANAGEMENT_ENABLED`
                                            flag before they can call the SQLite registry.
                                            Private-corpus/share-link collaboration readiness now
                                            has a separate no-secret policy-matrix preflight and
                                            admin/UI surface. Local hash-only share-link token
                                            lifecycle is implemented through SQLite, CLI, and
                                            `/admin/share-links*`, with Streamlit operator
                                            management behind a separate explicit flag; real
                                            private corpora and public share delivery remain
                                            disabled.
                                            External
                                            identity providers, external billing/payment,
                                            production team administration, and a real frontend
                                            are not implemented.
```

## API Route Coverage

The list below should match the FastAPI route decorators in `api.py`. The
regression test `tests/test_feature_audit_docs.py` parses `api.py` and fails if a
route is added without being listed here.

```text
GET    /artifacts
GET    /artifacts/{artifact_id}

GET    /corpus/papers
GET    /corpus/chunks
GET    /corpus/structure
GET    /corpus/structure/report
GET    /corpus/status
PUT    /corpus/active
GET    /corpus/profiles
POST   /corpus/profiles
GET    /corpus/profiles/{profile_id}/status
GET    /corpus/profiles/{profile_id}/report
POST   /corpus/profiles/{profile_id}/activate
POST   /corpus/profiles/{profile_id}/rebuild

GET    /admin/status
GET    /admin/status/report
GET    /admin/metrics
GET    /admin/platform-migration-rehearsal
GET    /admin/platform-migration-rehearsal/report
GET    /admin/product-activation-rehearsal
GET    /admin/product-activation-rehearsal/report
GET    /admin/collaboration-readiness
GET    /admin/collaboration-readiness/report
GET    /admin/provider-runtime-rehearsal
GET    /admin/provider-runtime-rehearsal/report
GET    /admin/quality-readiness
POST   /admin/quality-readiness
GET    /admin/quality-readiness/report
POST   /admin/quality-readiness/report
GET    /admin/activation-suite
POST   /admin/activation-suite
GET    /admin/activation-suite/report
POST   /admin/activation-suite/report
GET    /admin/openapi-contract
GET    /admin/openapi-contract/report
POST   /admin/openapi-contract/verify
POST   /admin/openapi-contract/verify/report
GET    /admin/runtime-manifest
GET    /admin/runtime-manifest/report
POST   /admin/runtime-manifest/restore-check
POST   /admin/runtime-manifest/restore-check/report
GET    /admin/retention
POST   /admin/retention/delete
GET    /admin/events
GET    /admin/product-registry/status
GET    /admin/product-registry/workspaces
POST   /admin/product-registry/workspaces
GET    /admin/product-registry/workspaces/{workspace_id}
POST   /admin/product-registry/workspaces/{workspace_id}/members
PUT    /admin/product-registry/workspaces/{workspace_id}/quota
PUT    /admin/product-registry/workspaces/{workspace_id}/billing
POST   /admin/product-registry/permissions/check
GET    /admin/share-links/status
GET    /admin/share-links
POST   /admin/share-links
POST   /admin/share-links/{link_id}/revoke
POST   /admin/share-links/resolve

POST   /query
POST   /query/inspect
POST   /query/retrieve
POST   /query/report

POST   /jobs/image/mock
POST   /jobs/async/image/mock
POST   /jobs/code/python-local
POST   /jobs/async/code/python-local
POST   /jobs/code/octave-local
POST   /jobs/async/code/octave-local
POST   /jobs/index/rebuild
POST   /jobs/async/index/rebuild
GET    /jobs
GET    /jobs/{job_id}
POST   /jobs/{job_id}/cancel
POST   /jobs/{job_id}/retry
POST   /jobs/{job_id}/retry-scheduled

GET    /health
GET    /ready
```

## Current Test Coverage Shape

```text
Area                    Primary checks
----------------------  ---------------------------------------------------------------------------
API contract            tests/test_api.py covers auth, constant-time token checks,
                        query modes, corpus, admin,
                        retention preview/delete, jobs, job idempotency, owner metadata,
                        artifacts
Ingestion/upload        tests/test_ingestion.py covers filename safety, PDF metadata,
                        checksum dedup, structure markers, cancellation, and upload scan
                        allowed/blocked paths, symlink skip/write guards, plus
                        manifest/active-selection runtime-state parse hardening
RAG evaluation          tests/test_evaluation.py and scripts/evaluate_rag.py cover citations,
                        recorded answers, retrieval-only source/page cases, provider fixtures,
                        local code-output artifacts/templates/job-backed execution,
                        PDF equation/table/figure/algorithm structure cases, retrieval diagnostics, ontology,
                        topic/lane coverage gates, quality-maturity target reporting,
                        and no-LLM retrieval diagnostics
Jobs and workers        tests/test_jobs.py covers JSONL/SQLite state, durable idempotency,
                        owner metadata, bounded retry/dead-letter behavior, leases, recovery, deadlines,
                        cancellation, malformed JSONL fallback, and durable worker behavior
Providers               tests/test_providers.py covers mock diagrams, Python/Octave execution,
                        resource/path/input limits, Docker readiness, and runtime-unavailable cases
Storage metadata        tests/test_metadata.py, tests/test_storage_manifest.py, and
                        tests/test_storage_schema.py cover corpus, chunks, profiles,
                        atomic writes, no-secret runtime manifests, restore dry-run
                        checks, storage-schema drift checks, and the storage-schema CLI.
                        tests/test_platform_migration.py covers the composed no-secret
                        production migration preflight and activation blocker split.
                        tests/test_storage_migration.py covers local runtime migration
                        rehearsal, staging guards including source/staging overlap
                        rejection, overwrite behavior, symlink skipping, and no-secret
                        report boundaries
UI guardrails           tests/test_translation_guard.py and tests/test_streaming.py cover browser
                        translation guards, runtime restore-check UI anchors, and streaming error handling
Deployment hygiene      tests/test_deploy_sync.py, tests/test_health_check.py, and
                        tests/test_docs_status.py cover safe sync, coverage-data excludes,
                        health checks, retention symlink skip/recheck behavior,
                        and status drift
Feature audit drift     tests/test_feature_audit_docs.py covers route-list completeness
Admin metrics           tests/test_admin.py and tests/test_api.py cover no-secret metrics text
                        formatting, platform-readiness gauges, provider-readiness
                        gauges, and the authenticated metrics route
Retrieval traces        tests/test_api.py and tests/test_admin.py cover metadata-only
                        retrieval trace event emission, admin summaries, and metrics
Retrieval alerts        tests/test_admin.py covers metadata-only retrieval trace
                        alert thresholds, summaries, reports, and metrics
Runtime events          tests/test_runtime.py covers no-secret event listing plus
                        malformed JSONL-line tolerance, sensitive key detection,
                        and admin-facing event projection redaction
```

## Known Evaluation Gaps

- The offline RAG baseline is deterministic and now covers 107 no-LLM retrieval
  questions plus 13 local code-output cases (12 Python, one Octave-compatible
  runtime-aware case; four Python cases are job-backed, across reusable
  templates and paper-specific examples) and 30 seeded PDF
  equation/table/figure/algorithm structure cases, but the live answer QA set
  still lacks enough passing live answer count/pass-rate/term-coverage evidence;
  richer PDF layout extraction remains a future quality lane, and broader real
  Octave execution coverage remains narrow until an `octave` binary is installed
  in CI/runtime.
- The local Python/Octave child-process providers are contract tests, while the
  Docker backend proves a no-key container path and execution policy preflight
  rejects obvious disallowed imports, shell/package-manager commands, absolute
  path literals, and Octave shell/network calls. No-secret code-execution events
  summarize backend/status/policy/output-limit outcomes without copying source.
  Captured stdout/stderr and generated-artifact export are byte/count-bounded
  and record truncation metadata. Admin status/report now include local advisory
  alerts for failure rate, slow duration, policy violations, output/artifact
  truncation, metadata-only retrieval trace summaries/alerts, and a no-secret
  metrics text export for local scraping. Uploaded PDFs now have a local pre-write scan
  for PDF magic, parseability, encryption, active-content markers, and page
  count. Broader
  production sandbox evidence still needs live Docker enablement, antivirus or
  sandbox-backed upload scanning, deeper abuse controls, and production
  metrics/traces beyond the local export baseline.
- Runtime storage and queue state are local SQLite/JSONL/filesystem bridges.
  Admin readiness now has separate external metadata, object-storage, and
  distributed job-store targets, but no production database, object store, or
  distributed queue has been activated. The new migration preflight proves local
  evidence and external blocker reporting. The migration rehearsal now proves a
  local staged runtime copy plus restore/schema verification and can emit and
  verify an opaque object-storage migration manifest for staged files without
  source paths, filenames, buckets, endpoints, credentials, or contents, with
  verifier rejection for nested/camelCase unsafe manifest fields. It is now also
  able to emit and verify a no-secret durable job-store migration
  manifest for staged `jobs.sqlite3` state without job payloads, owner IDs,
  request IDs, worker IDs, idempotency keys, logs, artifacts, stdout/stderr, or
  secrets, with the same nested/camelCase unsafe-field rejection. Its verifier
  now also has an explicit `/admin/platform-migration-rehearsal` and Streamlit
  on-demand surface that exposes only public summary fields, not raw manifests.
  compares idempotency-claim metadata and uses the manifest timestamp for
  scheduled/due job fields so unchanged persisted state does not fail due to
  wall-clock drift. It is still
  not live external database, object-storage, or distributed queue migration.
- Productization readiness now has a no-secret CLI/admin/report/metrics/UI
  surface. The current local foundation passes, and the local API-key lifecycle
  registry is implemented with hash-only storage and API auth integration. The
  local product registry now covers users, workspaces, role permissions, quota
  limits, usage events, and billing attribution as a SQLite ledger for readiness
  checks. FastAPI query routes can also use that ledger as a local request quota
  guard when explicitly enabled, returning `429` before model generation for
  over-limit work. FastAPI query/job/corpus/admin write paths can enforce local
  role permissions when the RBAC guard is explicitly enabled, returning `403`
  before protected work starts. FastAPI and Streamlit now expose a local
  product-registry management surface for workspace/member/quota/permission
  operations when the SQLite backend is enabled. The local product activation
  rehearsal now proves hash-only API-key lifecycle, workspace RBAC,
  cross-workspace isolation denial, quota limiting, billing attribution, and
  `product_readiness` activation against disposable SQLite stores without
  exporting raw tokens, workspace/user identifiers, or paths, and exposes the
  same no-secret drill through `/admin/product-activation-rehearsal` and the
  Streamlit admin panel for explicit operator runs. The collaboration-readiness
  preflight now adds a separate no-secret policy matrix for private corpora and
  share links: the default disabled-safe state passes local foundation, while
  activation remains blocked until feature flags, product registry/RBAC guard,
  and share-link token registry prerequisites are configured. The local
  share-link registry now stores token hashes in SQLite, returns the raw token
  only once on create, supports list/revoke/resolve through CLI and
  `/admin/share-links*`, exposes Streamlit operator management behind a
  separate explicit flag, and records metadata-only `share_link_admin` events.
  It is exposed through
  `scripts/collaboration_readiness.py`, `/admin/collaboration-readiness`, and
  the Streamlit admin panel without returning workspace/user/corpus/share
  identifiers in readiness output, URLs, tokens after create, creator user IDs,
  descriptions, paths, prompts, answers, or contents. The local API-key,
  product, and share-link registry CLIs also sanitize output-write and SQLite
  registry errors instead of leaking output paths or crashing during error
  reporting. Activation
  still remains
  blocked on real
  external identity provider, identity-backed quota enforcement, external
  billing/payment provider, real private-corpus/public-share delivery, and
  tenancy decisions when those are required for
  production.
- Provider activation readiness now has a no-secret CLI/admin/report/metrics/UI
  surface. The current local foundation passes, but activation remains blocked
  on real external image provider configuration, hosted execution provider
  configuration, MATLAB backend/licensing, external-provider enablement, and
  provider quota/cost guard activation. The local provider guard now has a
  reusable pre-call decision for estimated prompt tokens, requested completion
  tokens, and optional cost ceilings, and `src.chain` applies that decision
  before constructing the LLM client. Guard denials now record separate
  metadata-only `provider_quota_guard` events instead of `provider_failure`
  events. Non-finite or extreme-exponent local cost/rate settings are treated
  as invalid disabled pricing/limits rather than runtime exceptions or oversized
  status strings, and artifact public cost metadata applies the same bounded
  output rule. The provider runtime rehearsal now proves the local mock image
  provider, local Python execution with artifact capture, Octave-compatible
  runtime branch, Docker readiness reason code, Python/Octave abuse-policy
  denials, one allowed provider-guard decision, one over-limit provider-guard
  denial, and provider-readiness local foundation without exporting paths,
  unsafe source snippets, stdout/stderr, or claiming external activation, and
  exposes the same no-secret drill through
  `/admin/provider-runtime-rehearsal` and the Streamlit admin panel for explicit
  operator runs.
- The bundled seed corpus has been expanded to 30 open-access papers, and the
  next content milestone is a curated 50+ paper library with richer
  topic coverage and more PDF-layout acceptance cases.
- Streamlit remains acceptable for demo/personal workflows, but platform UX,
  user/workspace state, and account-level admin need a frontend/API split.
- Provider activation, MATLAB licensing, identity, quotas, and billing should
  remain disabled until their runtime boundaries are implemented and tested.

## Next Audit Actions

1. Add live answer evaluation cases with passing live answer pass-rate and
   term-coverage evidence before claiming broad RAG quality beyond the
   small-group retrieval bar.
2. Use the production storage/distributed-worker readiness blockers and the
   object-storage manifest to choose the next real database/object-storage/
   job-store migration tests before moving runtime state out of local files.
3. Add live sandbox and abuse-policy tests before enabling Docker, Cloudflare Sandbox, or any
   MATLAB-compatible hosted execution path.
4. Continue identity/workspace/ownership tests before exposing real private
   corpora, identity-backed keys, quotas, billing, or share links.
