# FluxMind Deployment Status

Last live check: 2026-07-27 02:39 CST

This document records the currently verified deployment boundary. It does not
assume that local working-tree changes are already live.

## Public service

```text
Host            Trace-Twin
Deploy root     /opt/fluxmind
Runtime user    fluxmind
UI service      fluxmind-ui.service
API service     fluxmind-api.service
Worker service  fluxmind-worker.service
UI port         18501
API port        18502
UI URL          https://smy.hyper-dusty.cloud/
API URL         https://api-smy.hyper-dusty.cloud/
```

Read-only public checks on 2026-07-27:

- `GET https://api-smy.hyper-dusty.cloud/health` returned
  `{"status":"ok"}`;
- `HEAD https://smy.hyper-dusty.cloud/` returned HTTP 200.

These checks confirm reachability only. The most recent detailed production
activation recorded before this refactor was the 2026-06-22 52-paper library
deployment with Docker-backed Python/Octave execution.

Read-only SSH preflight at 02:39 CST confirmed:

- UI, API, and worker services are all `active`;
- `/opt/fluxmind/api.py` exists;
- the deploy root is an rsync source tree rather than a Git checkout;
- `metadata/users.sqlite3` is absent, so first-run administrator creation is
  still required after deployment.

## Local release candidate

The current local tree contains a substantial, not-yet-deployed refactor:

- local admin/student accounts and per-user query history;
- provider-backed citation validation and clearer provider failure behavior;
- a smaller operational admin surface;
- removal of the historical product registry, share-link registry, readiness,
  rehearsal, migration, schema-audit, and activation-suite code;
- removal of the API access audit and the runtime keyword-based redaction
  engine;
- direct job and artifact details, including submitted code, stdout/stderr,
  errors, prompts, titles, provider metadata, idempotency keys, and account
  ownership;
- account-scoped Streamlit jobs/artifacts and history recording across
  `/query`, `/query/inspect`, and `/query/report`;
- removal of the obsolete Streamlit component script used for browser
  translation mutation handling;
- removal of placeholder corpus documents, hash-vector embedding substitution,
  and the test-only subprocess output path; startup/readiness and provider
  errors now retain their concrete failure reason;
- removal of the retired workspace/quota request field, empty API-key ownership
  bridge, and source helpers with no runtime caller.

Local verification on 2026-07-27:

```text
pytest                      378 passed
offline RAG evaluation      passed
health check                passed
API local process smoke     /health ok; /ready ready; 49 OpenAPI paths
UI local process smoke      /_stcore/health ok; root HTTP 200
Streamlit admin session     chat rendered; no exception
Streamlit student session   chat rendered; no admin controls; no exception
durable worker smoke        started cleanly; no due jobs
curated papers              52
active papers               52
chunk rows / sources        3497 / 52
public API health           ok
public UI                    HTTP 200
```

The local account database intentionally does not exist until the first
Streamlit setup creates the administrator account.

## Deployment delta

`python scripts/deploy_sync.py` was rerun in its default dry-run mode after the
full local verification. It reports the expected source updates, deletion of
the retired platform modules and legacy coverage configuration, and addition
of `scripts/_cli.py`, `src/users.py`, and `tests/test_users.py`. No files were
synchronized and no services were restarted.

The user-owned untracked files `environment.yml` and `requirements.lock` are
preserved locally but explicitly excluded from source synchronization. The
production environment continues to use `requirements.txt` and its existing
conda environment.

Runtime state remains excluded by the deploy script:

- `.env` and `.env.bak*`;
- local environment snapshots (`environment.yml`, `requirements.lock`);
- `papers/` and `faiss_index/`;
- `metadata/`, `jobs/`, and `artifacts/`;
- models, caches, and virtual environments.

## Required production sequence

Production activation is intentionally pending explicit authorization:

1. finalize and commit the reviewed source scope;
2. run the deploy dry-run again and review every deletion;
3. apply the source sync and restart API, UI, and worker services;
4. verify local and public health, service state, user setup/login, one RAG
   query with citations, query history, and one Python/Octave job;
5. update this document with the deployed commit and observed results.

Use `python scripts/deploy_sync.py --apply --restart` only after steps 1–2 are
complete. The script preserves production runtime data but uses `rsync
--delete` for the source tree.
