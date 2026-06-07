# FluxMind Repository Status

Snapshot time: 2026-06-07 22:57 CST

This file records the last verified clean repository boundary for the completed
no-key/local baseline, plus the repo hygiene expectations for the next working
pass. It is a repo snapshot, not a production deployment source of truth. For
live service state, use `docs/DEPLOYMENT_STATUS.md` and re-run the refresh
commands there.

## Git State

```text
Branch                         main
Remote                         origin git@github.com:Shallow-dusty/FluxMind.git
Tracking                       origin/main
Last clean local/origin state  32fca21 docs: record no-key baseline deployment
Divergence at clean boundary   ahead 0, behind 0
Deployed source baseline       a51a060 docs: confirm no-key platform baseline
Deployment record follow-up    32fca21 after live verification and docs-only sync
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

## Current Documentation Set

```text
Document                               Role
-------------------------------------  ---------------------------------------
README.md                              Project entrypoint and quick start
AGENTS.md                              Codex/project instructions
CLAUDE.md                              Legacy Claude-facing project bootstrap
docs/README.md                         Documentation index and ownership map
docs/REPO_STATUS.md                    This git/status snapshot
docs/ARCHITECTURE.md                   Runtime and module architecture
docs/BACKLOG.md                        Work packages and acceptance criteria
docs/DEPLOYMENT_STATUS.md              Live deployment snapshot and commands
docs/PLATFORM_AUDIT_AND_ROADMAP.md     Broader platform audit and roadmap
docs/demo-script.md                    Chinese demo script and Q&A
docs/handover.html                     Single-file presentation handover
```

## Verification Run

Commands run from `/home/shallow/04.AI-Prism/11.FluxMind` using `.venv`:

```text
Command                                                                 Result
----------------------------------------------------------------------  ------
.venv/bin/python --version                                             Python 3.13.11
.venv/bin/python -m pytest                                             pass, 200 tests
.venv/bin/python scripts/health_check.py                               pass, including docs drift anchors
.venv/bin/python scripts/evaluate_rag.py                               pass, 5 eval cases and 5 recorded answers
.venv/bin/python scripts/health_check.py --url https://smy.hyper-...   pass, both URLs 200
.venv/bin/python scripts/health_check.py --ssh-host root@100.100...    pass
git diff --check                                                       pass
git status --short --branch                                            clean at last verified boundary
```

Remote read-only check highlights:

```text
Service state       UI/API/worker/cloudflared/docker active
Listeners           0.0.0.0:18501 and 0.0.0.0:18502
Local API health    {"status":"ok"}
Model config        LLM_MODEL=mimo-v2.5-pro
Embedding model     /opt/fluxmind/models/all-MiniLM-L6-v2
Active papers       6
FAISS index bytes   786477
Chunk rows          512 across 6 source paths
Index freshness     True
Storage readiness   local metadata/object storage available
Docker execution    configured=False available=False reason=not_configured
Disk                /dev/vda3 40G total, 25G free, 35% used
```

## Immediate Boundary

- Runtime state remains git-ignored: `papers/`, `faiss_index/`, `artifacts/`,
  `jobs/`, `metadata/`, `.env`, virtual environments, caches, and bytecode.
- New development should start from the `32fca21` clean baseline, then refresh
  this file after any commit/release sequence that changes the verified repo
  boundary.
- Deployment facts should not be inferred from git state alone because
  `/opt/fluxmind` is a synchronized source tree, not a git checkout.
