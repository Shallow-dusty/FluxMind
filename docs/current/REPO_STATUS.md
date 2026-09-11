# FluxMind Repository Status

Snapshot time: 2026-07-27

This is the current local repository snapshot. Live service state belongs in
`docs/current/DEPLOYMENT_STATUS.md`; product direction belongs in
`DEVELOPMENT.md`.

## Git boundary

```text
Branch             main
Remote             git@github.com:Shallow-dusty/FluxMind.git
HEAD               79e866e fix(deploy): exclude .env.bak* from rsync
Remote divergence  ahead 0 / behind 0
Worktree           dirty: active refactor, not committed or deployed
Diff size          91 tracked paths + 3 intended new project files;
                   about 3.8k additions / 36.0k deletions
```

Untracked files:

- `src/users.py` and `tests/test_users.py` are part of this refactor;
- `environment.yml` and `requirements.lock` are user-owned and were preserved
  without modification; deploy sync excludes both local environment snapshots.

## Current refactor

User-facing additions:

- first-run admin creation and local admin/student accounts;
- account management in Streamlit;
- per-user saved query history in Streamlit and FastAPI;
- current-account ownership for Streamlit jobs and artifacts;
- readable artifact download filenames and visible idempotency keys;
- stricter answer citation validation;
- provider smoke coverage using the configured backend.

Simplification:

- deleted historical readiness, activation, rehearsal, platform migration,
  storage-schema, API-key registry, product registry, and share-link modules,
  scripts, routes, panels, and dedicated tests;
- rewrote `src/admin.py` around corpus, index, jobs, artifacts, users,
  providers, activity, storage, and retention;
- reduced `src/runtime.py` to append/list/filter events, correlation IDs, token
  estimates, and stable user-facing errors;
- removed the API access audit while retaining the optional in-memory API rate
  limit;
- replaced job/artifact fingerprints and presence flags with useful request,
  result, stdout/stderr, error, title, prompt, provider, and owner data;
- removed remaining backup/provider/event presence flags and the deprecated
  browser translation component injection;
- removed the empty-corpus placeholder, incompatible hash embedding
  substitution, and test-only subprocess capture fallback so operational
  failures remain real and diagnosable;
- removed the last workspace/quota request field, empty API-key owner bridge,
  and uncalled BM25/job/artifact/Docker status wrappers; a high-confidence
  static dead-code scan reports no remaining candidates;
- replaced duplicate defensive CLI formatters with one direct error formatter
  and removed the obsolete machine-specific reference update script;
- removed the legacy 88% branch-coverage quota from CI so the gate matches the
  documented behavior-test policy; pytest, offline RAG evaluation, and the
  runtime health check remain mandatory;
- fixed two admin-page integration errors found by real Streamlit session
  execution (translation key and retention preview shape);
- simplified health checks and active project documentation.

The source deletion is intentional: archived roadmap history remains under
`docs/archive/` and retired constraints remain under `docs/legacy/`, but their
implementation is no longer part of the runtime.

## Verification

Run on the current working tree:

```text
python -m pytest                  378 passed, 2 dependency warnings
python scripts/evaluate_rag.py    passed all offline gates
python scripts/health_check.py    passed
python -m compileall              passed
git diff --check                  passed
deploy_sync.py                    dry-run only
API process smoke                 /health and /ready passed on 127.0.0.1:18512
Streamlit process smoke           passed on 127.0.0.1:18511
Streamlit admin/student sessions  rendered without exceptions
durable worker entry              passed; no due jobs
```

Health evidence:

```text
curated papers       52
active papers        52
FAISS vectors        non-empty
chunk rows           3497
chunk sources        52
API public health    {"status":"ok"}
UI public response   HTTP 200
```

The two warnings are upstream deprecations for Starlette's current TestClient
bridge and `langchain-community`; neither is a failing project gate.

## Remaining boundary

The local development slice is complete enough for review, but these external
actions have not been taken:

- no commit or push;
- no production sync or service restart;
- no first production administrator creation;
- no invited-user acceptance test.

Production activation requires explicit authorization because the deploy uses
`rsync --delete` and would remove the retired source files on Trace-Twin.
