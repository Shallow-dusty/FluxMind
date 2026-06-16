# FluxMind Documentation Index

Last updated: 2026-06-17

This directory separates current status, architecture, planning, deployment
evidence, and demo handoff material. Prefer updating the document that owns the
fact instead of repeating the same state in several places.

## Reading Order

1. `docs/REPO_STATUS.md` - current git/worktree snapshot, local/remote
   verification results, and the immediate repo hygiene boundary.
2. `docs/DEPLOYMENT_STATUS.md` - mutable production deployment snapshot and
   refresh commands. Re-check live state before acting on this file.
3. `docs/ARCHITECTURE.md` - current runtime boundaries, module ownership, and
   the next architecture step.
4. `docs/BACKLOG.md` - implementation work packages and acceptance criteria.
5. `docs/PLATFORM_AUDIT_AND_ROADMAP.md` - broader platform audit, product
   direction, and open decisions.
6. `docs/QUALITY_ROADMAP.md` - staged quality bars for self-use, small-group,
   and community readiness.
7. `docs/PRODUCTION_GAP_AND_MARKET_RESEARCH.md` - production readiness gap,
   competitor scan, community demand signals, and staged roadmap.
8. `docs/FEATURE_AUDIT.md` - current feature inventory, route coverage, test
   evidence, and known capability gaps.
9. `docs/demo-script.md` - short Chinese demo script and defense Q&A.
10. `docs/handover.html` - single-file visual handover for presentations.

## Source-Of-Truth Map

```text
Topic                     Owner doc
------------------------  ---------------------------------
Git/worktree snapshot     docs/REPO_STATUS.md
Live deployment state     docs/DEPLOYMENT_STATUS.md
Runtime/module boundary   docs/ARCHITECTURE.md
Work package status       docs/BACKLOG.md
Product/platform roadmap  docs/PLATFORM_AUDIT_AND_ROADMAP.md
Quality maturity gates    docs/QUALITY_ROADMAP.md, eval/rag_baseline.json
Production/market gap     docs/PRODUCTION_GAP_AND_MARKET_RESEARCH.md
Feature coverage audit    docs/FEATURE_AUDIT.md
Demo speaking notes       docs/demo-script.md
Visual delivery page      docs/handover.html
Project bootstrap         README.md, AGENTS.md
```

## Update Rules

- If a fact depends on live services, update it only after running the relevant
  check and record the check date/time.
- If a fact describes planned work, keep it in `docs/BACKLOG.md` or
  `docs/PLATFORM_AUDIT_AND_ROADMAP.md`, not in `README.md`.
- Keep `README.md` as the project entrypoint. It may summarize, but the detailed
  operational status belongs in this docs tree.
- Do not copy secrets, tokens, `.env` values, uploaded PDFs, FAISS indexes,
  metadata databases, job logs, or artifacts into documentation.
