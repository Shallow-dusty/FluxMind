# FluxMind Repository Status

Snapshot time: 2026-06-03 00:43 CST

This file records the current local repository state after confirming the
completed no-key/local baseline. It is a repo snapshot, not a production
deployment source of truth. For live service state, use
`docs/DEPLOYMENT_STATUS.md` and re-run the refresh commands there.

## Git State

```text
Branch        main
Remote        origin git@github.com:Shallow-dusty/FluxMind.git
Tracking      origin/main
HEAD          b9c063e docs(deploy): record runtime manifest API deployment
origin/main   55d16a8
Divergence    ahead 36, behind 0
Baseline      clean for tracked/untracked source files before this docs pass
Current diff  documentation-only completion/status cleanup, unstaged
Ignored only  .venv, __pycache__, .pytest_cache, jobs, metadata, runtime caches
```

The 36 local commits cover the no-key platform work after `origin/main`:

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

Push and deploy are the next steps for this completion pass.

Pending documentation cleanup files:

```text
Modified   AGENTS.md
Modified   CLAUDE.md
Modified   README.md
Modified   docs/ARCHITECTURE.md
Modified   docs/BACKLOG.md
Modified   docs/DEPLOYMENT_STATUS.md
Modified   docs/PLATFORM_AUDIT_AND_ROADMAP.md
Modified   docs/demo-script.md
Modified   docs/handover.html
Added      docs/README.md
Added      docs/REPO_STATUS.md
```

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
.venv/bin/python scripts/health_check.py                               pass
.venv/bin/python scripts/evaluate_rag.py                               pass
.venv/bin/python scripts/health_check.py --url https://smy.hyper-...   pass, both URLs 200
.venv/bin/python scripts/health_check.py --ssh-host root@100.100...    pass
git diff --check origin/main..HEAD                                     pass
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
- The local checkout is ready for a deliberate push or PR decision, but this
  pass only organized and documented the state.
- Deployment facts should not be inferred from the 36 local commits alone
  because `/opt/fluxmind` is a synchronized source tree, not a git checkout.
