# FluxMind Feature Audit

Last updated: 2026-06-07

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
.venv/bin/python -m pytest                                           pass, 200 tests, 2 known warnings
.venv/bin/python scripts/evaluate_rag.py                             pass, 5/5 eval cases
.venv/bin/python scripts/health_check.py                             pass, local/docs anchors
.venv/bin/python scripts/health_check.py --url ...                   pass, HTTPS UI/API 200
.venv/bin/python scripts/health_check.py --ssh-host root@100.100...  pass, live runtime green
```

## Feature Groups

```text
Group                         Status        Evidence and remaining gap
----------------------------  ------------  ---------------------------------------------------------
RAG query and inspection      verified      /query, /query/inspect, /query/retrieve, /query/report;
                                            offline eval and citation tests pass. Live QA breadth is
                                            still limited.
Corpus and profile control    verified      paper/chunk/status/active/profile routes exist; local
                                            JSON/SQLite store is inspectable. Multi-user ownership and
                                            production DB/object storage remain planned.
Local job and worker bridge   verified      immediate and async job routes, retries, cancellation,
                                            leases, and worker service exist. Distributed queue remains
                                            planned.
Artifacts and exports         verified      artifact list/download, checksums, and metadata mirrors
                                            exist. Durable object storage remains planned.
Admin and runtime status      verified      status/report/retention/events/runtime-manifest routes
                                            exist with no-secret output. Identity-aware admin remains
                                            planned.
No-key execution providers    verified      local Python and Octave-compatible providers prove the
                                            contract. They are development providers, not production
                                            sandboxes.
Mock diagram generation       verified      local SVG templates and artifact capture exist. Real image
                                            provider activation remains disabled.
Deployment and health gates   verified      guarded deploy sync, local health, HTTPS health, and SSH
                                            health are present. Live facts must still be refreshed.
Product platform layer        incomplete    accounts, teams, quotas, billing, durable ownership, and a
                                            real frontend are not implemented.
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
GET    /admin/runtime-manifest
GET    /admin/runtime-manifest/report
GET    /admin/retention
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
```

## Current Test Coverage Shape

```text
Area                    Primary checks
----------------------  ---------------------------------------------------------------------------
API contract            tests/test_api.py covers auth, query modes, corpus, admin, jobs, artifacts
RAG evaluation          tests/test_evaluation.py and scripts/evaluate_rag.py cover citations,
                        recorded answers, provider fixtures, retrieval diagnostics, and gates
Jobs and workers        tests/test_jobs.py covers JSONL/SQLite state, retries, leases, recovery,
                        deadlines, cancellation, and durable worker behavior
Providers               tests/test_providers.py covers mock diagrams, Python/Octave execution,
                        resource/path/input limits, Docker readiness, and runtime-unavailable cases
Storage metadata        tests/test_metadata.py and tests/test_storage_manifest.py cover corpus,
                        chunks, profiles, atomic writes, and no-secret runtime manifests
UI guardrails           tests/test_translation_guard.py and tests/test_streaming.py cover browser
                        translation guards and streaming error handling
Deployment hygiene      tests/test_deploy_sync.py, tests/test_health_check.py, and
                        tests/test_docs_status.py cover safe sync, health checks, and status drift
Feature audit drift     tests/test_feature_audit_docs.py covers route-list completeness
```

## Known Evaluation Gaps

- The offline RAG baseline is deterministic and useful, but the live answer QA
  set is still narrow for broad control-engineering coverage.
- The local Python/Octave execution providers are contract tests, not proof of a
  production sandbox with hard filesystem and network isolation.
- Runtime storage and queue state are local SQLite/JSONL/filesystem bridges, not
  a distributed production database or object store.
- Streamlit remains acceptable for demo/personal workflows, but platform UX,
  user/workspace state, and account-level admin need a frontend/API split.
- Provider activation, MATLAB licensing, identity, quotas, and billing should
  remain disabled until their runtime boundaries are implemented and tested.

## Next Audit Actions

1. Expand live retrieval and live answer evaluation cases before claiming broad
   RAG quality.
2. Add production storage and distributed worker acceptance tests before moving
   runtime state out of local files.
3. Add sandbox-specific tests before enabling Docker, Cloudflare Sandbox, or any
   MATLAB-compatible hosted execution path.
4. Add identity/workspace/ownership tests before exposing private corpora,
   quotas, billing, or share links.
