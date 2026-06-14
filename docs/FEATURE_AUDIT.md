# FluxMind Feature Audit

Last updated: 2026-06-15

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
.venv/bin/python -m pytest                                           pass, 286 tests, 2 known warnings
.venv/bin/python scripts/evaluate_rag.py                             pass, 20 answer cases and
                                                                      30 retrieval-only cases,
                                                                      4 code-output cases,
                                                                      6 PDF structure cases,
                                                                      20 recorded answers
.venv/bin/python scripts/health_check.py                             pass, local/docs/query-latency/query-alert/
                                                                      provider-alert/job-alert/API-access-audit/
                                                                      API-rate-limit/upload-scan/
                                                                      retention-delete/metrics-export/retrieval-trace/
                                                                      retrieval-alerts/storage-schema/artifact-limit/
                                                                      execution-alert/readiness/log-noise anchors
.venv/bin/python scripts/storage_schema.py --output /tmp/...         pass, ok=true, 7 stores, 0 problems
.venv/bin/python scripts/evaluate_rag.py --json-report /tmp/...      pass, /tmp/fluxmind-eval-report-storage-schema-cli.json
.venv/bin/python scripts/runtime_manifest.py --output /tmp/...       pass, /tmp/fluxmind-runtime-manifest-storage-schema-cli.json
.venv/bin/python scripts/runtime_manifest.py --restore-check ...     pass, ok=true, 6 groups, 5 checked files,
                                                                      manifest_errors=0, 0 missing/mismatched
.venv/bin/python scripts/health_check.py --url ...                   02:31 snapshot, HTTPS UI/API 200
.venv/bin/python scripts/health_check.py --ssh-host root@100.100...  02:31 snapshot, live runtime green
```

## Feature Groups

```text
Group                         Status        Evidence and remaining gap
----------------------------  ------------  ---------------------------------------------------------
RAG query and inspection      verified      /query, /query/inspect, /query/retrieve, /query/report,
                                            /corpus/structure, /corpus/structure/report;
                                            offline eval, citation tests, code-output artifact and
                                            template checks, PDF structure checks, and paper-to-code
                                            report handoff tests pass. Live QA breadth is still limited.
Corpus and profile control    verified      paper/chunk/status/active/profile routes exist; local
                                            JSON/SQLite store is inspectable. Multi-user ownership and
                                            production DB/object storage remain planned.
                                            Uploaded PDFs have a local pre-write scan guard.
Local job and worker bridge   verified      immediate and async job routes, idempotency keys,
                                            owner metadata, bounded retry/dead-letter state,
                                            cancellation, leases, and worker service exist.
                                            Admin status/report include local job-health alerts.
                                            Distributed queue remains planned.
Artifacts and exports         verified      artifact list/download, checksums, and metadata mirrors
                                            exist, including source-job owner metadata. Durable
                                            object storage remains planned.
Admin and runtime status      verified      status/report/retention/events/runtime-manifest and
                                            restore-check routes plus Streamlit upload/report UI
                                            exist with no-secret output and job/artifact owner
                                            summaries, query duration summaries, query latency
                                            alerts, provider-failure alerts, code-execution event
                                            summaries, job-health alerts, API access audit summaries,
                                            API rate-limit summaries, upload-scan summaries, guarded
                                            retention-delete controls, local advisory alerts, and
                                            no-secret metrics export text plus metadata-only
                                            retrieval trace summaries, retrieval-quality alerts,
                                            and platform-readiness blocker summaries.
                                            Identity-aware admin remains planned.
No-key execution providers    verified      local Python, Octave-compatible, and opt-in Docker
                                            providers prove the job/artifact contract. Request-level
                                            package/policy preflight and no-secret execution outcome
                                            events exist; stdout/stderr capture and generated-artifact
                                            export are byte/count-bounded, with local alert summaries.
                                            Hosted execution, deeper abuse controls, and MATLAB
                                            licensing remain planned.
Mock diagram generation       verified      local SVG templates and artifact capture exist. Real image
                                            provider activation remains disabled.
Deployment and health gates   verified      guarded deploy sync, local health, HTTPS health, and SSH
                                            health are present. Live facts must still be refreshed.
Product platform layer        incomplete    accounts, teams, quotas, billing, identity-backed
                                            ownership, and a real frontend are not implemented.
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
GET    /admin/runtime-manifest
GET    /admin/runtime-manifest/report
POST   /admin/runtime-manifest/restore-check
POST   /admin/runtime-manifest/restore-check/report
GET    /admin/retention
POST   /admin/retention/delete
GET    /admin/events

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
API contract            tests/test_api.py covers auth, query modes, corpus, admin,
                        retention preview/delete, jobs, job idempotency, owner metadata,
                        artifacts
Ingestion/upload        tests/test_ingestion.py covers filename safety, PDF metadata,
                        checksum dedup, structure markers, cancellation, and upload scan
                        allowed/blocked paths
RAG evaluation          tests/test_evaluation.py and scripts/evaluate_rag.py cover citations,
                        recorded answers, retrieval-only source/page cases, provider fixtures,
                        local code-output artifacts/templates/job-backed execution,
                        PDF equation/table/figure structure cases, retrieval diagnostics, ontology,
                        topic/lane coverage gates, and no-LLM retrieval diagnostics
Jobs and workers        tests/test_jobs.py covers JSONL/SQLite state, durable idempotency,
                        owner metadata, bounded retry/dead-letter behavior, leases, recovery, deadlines,
                        cancellation, and durable worker behavior
Providers               tests/test_providers.py covers mock diagrams, Python/Octave execution,
                        resource/path/input limits, Docker readiness, and runtime-unavailable cases
Storage metadata        tests/test_metadata.py, tests/test_storage_manifest.py, and
                        tests/test_storage_schema.py cover corpus, chunks, profiles,
                        atomic writes, no-secret runtime manifests, restore dry-run
                        checks, storage-schema drift checks, and the storage-schema CLI
UI guardrails           tests/test_translation_guard.py and tests/test_streaming.py cover browser
                        translation guards, runtime restore-check UI anchors, and streaming error handling
Deployment hygiene      tests/test_deploy_sync.py, tests/test_health_check.py, and
                        tests/test_docs_status.py cover safe sync, health checks, and status drift
Feature audit drift     tests/test_feature_audit_docs.py covers route-list completeness
Admin metrics           tests/test_admin.py and tests/test_api.py cover no-secret metrics text
                        formatting, platform-readiness gauges, and the
                        authenticated metrics route
Retrieval traces        tests/test_api.py and tests/test_admin.py cover metadata-only
                        retrieval trace event emission, admin summaries, and metrics
Retrieval alerts        tests/test_admin.py covers metadata-only retrieval trace
                        alert thresholds, summaries, reports, and metrics
```

## Known Evaluation Gaps

- The offline RAG baseline is deterministic and now covers 50 no-LLM retrieval
  questions plus four local Python code-output cases (two job-backed, across the
  `smc_reaching_law` and `pmsm_current_step` templates) and six seeded PDF
  equation/table/figure structure cases, but the live answer QA set, richer PDF
  layout extraction, and broader Octave *execution* eval (blocked on an Octave
  binary in CI/runtime) are still narrow for broad control-engineering coverage.
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
- Runtime storage and queue state are local SQLite/JSONL/filesystem bridges, not
  a distributed production database or object store.
- Streamlit remains acceptable for demo/personal workflows, but platform UX,
  user/workspace state, and account-level admin need a frontend/API split.
- Provider activation, MATLAB licensing, identity, quotas, and billing should
  remain disabled until their runtime boundaries are implemented and tested.

## Next Audit Actions

1. Expand live retrieval and live answer evaluation cases before claiming broad
   RAG quality.
2. Use the production storage/distributed-worker readiness blockers to choose
   the next real database/object-storage/queue migration tests before moving
   runtime state out of local files.
3. Add live sandbox and abuse-policy tests before enabling Docker, Cloudflare Sandbox, or any
   MATLAB-compatible hosted execution path.
4. Add identity/workspace/ownership tests before exposing private corpora,
   quotas, billing, or share links.
