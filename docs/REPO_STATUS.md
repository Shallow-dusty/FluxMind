# FluxMind Repository Status

Snapshot time: 2026-06-20 20:20 CST

This file records the current local repository snapshot plus the last verified
clean repository boundary for the completed no-key/local baseline. It is a repo
snapshot, not a production deployment source of truth. For live service state,
use `docs/DEPLOYMENT_STATUS.md` and re-run the refresh commands there.

## Git State

```text
Branch                         main
Remote                         origin git@github.com:Shallow-dusty/FluxMind.git
Tracking                       origin/main
Source/eval quality baseline   9b1cbc5 test: expand FluxMind community quality eval
Current implementation commit  49cdb82 fix: sanitize share-link UI errors
Current docs/health sync       docs: record share-link UI error audit status (this commit)
Current local app-code HEAD    49cdb82 fix: sanitize share-link UI errors
Remote status at verification  origin/main remains at 675149b; local main is ahead
                               by the twenty-seven local commits below
                               after this share-link UI docs refresh commit
Current local commit stack     docs: record share-link UI error audit status (this commit)
                               49cdb82 fix: sanitize share-link UI errors
                               d39d983 docs: refresh git and documentation drift status
                               042e6d0 fix: redact API key public metadata
                               c7b6d9d docs: refresh git and drift status
                               6066547 docs: record runtime event redaction audit
                               1173ea8 fix: redact runtime event metadata values
                               85eb2b5 docs: record execution input audit status
                               69bb9e7 fix: handle execution input path conflicts
                               1f97b7b docs: record product registry audit status
                               12f4205 fix: guard product registry orphan writes
                               05fae15 docs: record artifact path audit status
                               51fee7e fix: harden local artifact path resolution
                               f2d2da1 docs: record corpus metadata audit status
                               5065418 fix: preserve same-name corpus metadata
                               830d05d docs: record job lease audit status
                               bae5f88 fix: guard terminal job lease release
                               fac2c6b docs: record share-link event evidence audit
                               ea8a7a2 fix: preserve share-link workspace event evidence
                               3ae6842 docs: refresh share-link audit status
                               c56c285 fix: redact share-link workspace identifiers
                               e93dba5 docs: refresh FluxMind activation status
                               b1212e2 feat: expose local activation admin surfaces
                               39ddaee feat: add local activation readiness tools
                               1ebfde3 feat: add durable job-store migration manifests
                               4ea219c fix: harden no-secret local projections
                               ba7c243 feat: add provider quota guard and safe runtime events
Current refresh scope          local audit/forward-development commits for product activation
                               rehearsal, provider runtime rehearsal, job-store migration manifest,
                               standalone platform migration rehearsal admin API/UI,
                               activation suite including in-memory live eval
                               evidence input, OpenAPI contract local
                               foundation gate, OpenAPI contract snapshot
                               drift verifier, and full activation action plan,
                               product activation rehearsal
                               admin API/UI, provider runtime rehearsal
                               admin API/UI, and quality gap/evidence-request
                               summary plus evidence collection plan with
                               standalone admin API/UI surface,
                               admin readiness/rehearsal metadata-only
                               runtime events and OpenAPI snapshot count
                               shape hardening,
                               artifact/admin/runtime/API-access hardening,
                               product-registry read-RBAC and Streamlit management-flag
                               hardening, API-key CLI one-time-token output guard,
                               provider quota/cost numeric-config hardening,
                               no-secret readiness CLI OSError path-sanitization,
                               live eval request-ID redaction in JSON reports,
                               provider runtime execution abuse-policy rehearsal,
                               product activation workspace-isolation rehearsal,
                               collaboration readiness CLI/API/UI policy-matrix gate
                               for private corpora/share links,
                               share-link public projection hardening that replaces
                               raw workspace IDs with presence/fingerprint summaries,
                               share-link admin runtime-event workspace-present
                               evidence restoration after public projection redaction,
                               and share-link terminal-state/status regression coverage,
                               durable job lease-release state guard preserving
                               completed-job worker provenance,
                               source-path-specific corpus metadata for same-filename
                               library/upload PDFs,
                               local artifact file-URI canonicalization and
                               nonlocal host rejection for export/download,
                               product registry workspace/user referential guards
                               and sanitized CLI member output,
                               execution input materialization conflict handling
                               and regular-file entrypoint guards,
                               runtime event metadata-value redaction for
                               legacy and newly written events,
                               git/docs drift refresh,
                               API-key lifecycle public metadata projection
                               hardening,
                               final git/documentation drift refresh with
                               current no-drift gate evidence,
                               Streamlit share-link management error output
                               sanitization,
                               and docs;
                               committed locally in the stack above, not pushed
                               to origin and not deployed to Trace-Twin
Last deployed source/eval baseline 9b1cbc5 test: expand FluxMind community quality eval
Last deployed docs sync base   e4da2e9 docs: document octave-aware eval status
Live verification follow-up    30-paper corpus and 107/107 live retrieval refreshed on 2026-06-17 02:37 CST
Latest deploy follow-up        95f1760/e4da2e9 synced without restart and live-checked on 2026-06-17 02:59 CST
Ignored runtime/cache state    .venv, __pycache__, .pytest_cache, jobs, metadata, runtime caches
```

The no-key platform foundation and current small-group quality baseline are
pushed to `origin/main`. Their main contents are:

```text
Area                 Main contents
-------------------  ---------------------------------------------------------
RAG/eval             live retrieval gates, live answer quality readiness
                     gates, aggregate regression gates, recorded-answer
                     checks, JSON eval reports, staged quality-readiness
                     preflight, Octave-aware code-output fallback
Jobs/workers         durable leases, explicit local worker loop, systemd worker
                     unit, retries, deadlines, cancellation metadata
Corpus/storage       metadata profiles, paper/chunk SQLite mirrors, runtime
                     backup manifest, storage readiness/inventory
Platform readiness   separate metadata/object/job-store readiness targets,
                     blocker codes, and no-secret metrics/report fields
Admin/product shell  status/report endpoints, retention preview, runtime
                     events, query usage/cost visibility, product-readiness
                     and provider-readiness blocker surfaces, local API-key
                     lifecycle registry, local product registry, local
                     share-link token registry, Streamlit share-link
                     management flag
Artifacts/images     artifact metadata mirror/integrity, local SVG diagram
                     templates, stable artifact downloads
Execution            local Python/Octave provider hardening, templates,
                     timeout/resource/path/input-limit metadata
Deployment/docs      guarded deploy sync, health-check expansion, deployment
                     evidence updates
```

Earlier no-key platform deployment completed for `a51a060`; that live
verification evidence was recorded in `32fca21`. The previous small-group live
retrieval baseline was built from the `d80c083` / `cc705dc` quality run. The
current source/eval quality baseline is `9b1cbc5`, and its expanded
107-question retrieval set has now passed against the live deployment.
The latest platform-migration source/docs sync deployed to `/opt/fluxmind` is
`dc2b71a` (`docs: record runtime migration rehearsal`), with implementation
commits `8a4a76f` (`feat: add platform migration preflight`) and `366c1e7`
(`feat: add runtime migration rehearsal`).
The latest product-readiness source/docs sync deployed to `/opt/fluxmind` is
`79be409` (`docs: record product readiness status`), with implementation commit
`e2dc1e3` (`feat: add product readiness preflight`) and deployment record
`b0906df` (`docs: record product readiness deployment`). The latest
provider-readiness source/docs sync deployed to `/opt/fluxmind` is `0deea23`
(`docs: record provider readiness status`), with implementation commit
`938e918` (`feat: add provider readiness preflight`).
The latest quality-readiness source/docs sync deployed to `/opt/fluxmind` is
`35338d2` (`docs: clarify live answer quality readiness`), with implementation
commits `177dd4e` (`feat: gate quality readiness on live answer metrics`) and
`fa512df` (`fix: tolerate partial live quality result objects`). It keeps the
no-secret CLI/module for self-use, small-group, and community maturity checks,
and now merges explicit `--live-report` evidence for live retrieval count/pass
rate and live answer count/pass-rate/term-coverage gates before readiness can
pass.
The latest Octave-aware source/docs/health sync deployed to `/opt/fluxmind` is
`e4da2e9` (`docs: document octave-aware eval status`), with implementation
commit `95f1760` (`test: add octave-aware code-output eval`). It raises the
code-output regression gate to 13 cases, requires Python and Octave language
coverage, requires the `pmsm_current_decay` template, and lets the Octave case
pass only through either real artifact output when an `octave` binary is
installed or a structured runtime-unavailable diagnostic when the binary is
absent. Server-local live retrieval eval now reports 107/107 live retrieval
passes and `code_output_case_count=13` after the deploy sync.
The latest local API-key registry source/docs/health sync deployed to
`/opt/fluxmind` is `207ba7a` (`fix: extend remote health timeout`), with
implementation commit `6ad6dbc` (`feat: add local API key registry`),
documentation boundary commit `8f9db56`, and metrics guard fix `ea1c508`. It
adds an opt-in local SQLite API-key lifecycle registry, a CLI, FastAPI auth
integration, product-readiness visibility, and storage-schema/admin inventory
coverage. The live deployment keeps the registry backend disabled by default
(`none`), so no production credential lifecycle is activated until
`FLUXMIND_API_KEY_REGISTRY_BACKEND=sqlite` is deliberately set.
The latest local product registry source/docs/health sync deployed to
`/opt/fluxmind` is `c41ea94` (`feat: add local product registry`), after the
GitHub README rewrite `bce3ae5` (`docs: rewrite bilingual README`). It adds an
opt-in SQLite user/workspace/quota/usage/billing-attribution ledger, a CLI,
product-readiness visibility, storage-schema/admin inventory coverage, and
no-secret health anchors. The live deployment keeps this registry backend
disabled by default (`none`), so no multi-user identity, quota, or billing
runtime is activated until the corresponding local backends are deliberately
enabled.
The latest local product quota guard source/docs/health sync deployed to
`/opt/fluxmind` is `efe2143` (`docs: document product quota guard`), with
implementation commit `c130778` (`feat: add local product quota guard`). It adds
an opt-in runtime quota guard for `/query`, `/query/inspect`,
`/query/retrieve`, and `/query/report` when the local product registry and
SQLite quota store are explicitly enabled. The live deployment has the guard
code and admin metric installed, but keeps `FLUXMIND_PRODUCT_QUOTA_GUARD_ENABLED`
false by default, so runtime quota enforcement is an explicit operational
activation step rather than an automatic production change.
The latest local product RBAC guard source/docs/health sync deployed to
`/opt/fluxmind` is `3c85999` (`docs: document local product RBAC guard`), with
implementation commit `c7ecbf6` (`feat: add local product RBAC guard`). It adds
opt-in workspace role checks for query, job-submit, corpus-write, and
admin-write actions when the local API-key registry, product registry,
identity/quota/billing runtime, and `FLUXMIND_PRODUCT_RBAC_GUARD_ENABLED=true`
are deliberately enabled. The live deployment has the RBAC guard code, CLI,
runtime-event reporting, and admin metric installed, but keeps
`FLUXMIND_PRODUCT_RBAC_GUARD_ENABLED=false` by default.
The latest local product registry management source/docs/health sync deployed
to `/opt/fluxmind` is `b05c28d` (`docs: document product registry management`),
with implementation commit `645be5d` (`feat: add local product registry management`). It adds `/admin/product-registry/*` management routes and a
Streamlit operator panel for local workspace, member, quota,
billing-attribution, and permission-check metadata when the SQLite registry is
deliberately enabled. The live deployment has the management API/UI code and
health anchors installed, but keeps `FLUXMIND_PRODUCT_REGISTRY_BACKEND=none` by
default, so no external identity, payment, or production tenancy is activated.
The latest object-storage migration manifest verifier source/docs/health sync
deployed to `/opt/fluxmind` is `517756f`
(`docs: document object manifest verifier`), with implementation commit
`45e4cc6` (`feat: verify object storage migration manifests`). It adds an opt-in
`scripts/platform_migration_rehearsal.py --include-object-manifest` path that
turns a staged local runtime rehearsal into opaque object keys, SHA-256 hashes,
byte counts, group names, and path tokens without exporting source paths,
filenames, buckets, endpoints, credentials, `.env`, or file contents, plus
`--verify-object-manifest` to check that opaque manifest or the full rehearsal
JSON against a local/staged runtime tree. The verifier returns only safe
group/token/hash/count differences. The live deployment has the CLI and health
anchors installed, but still keeps external object storage disabled.
The latest PDF structure eval gate source/docs/health sync deployed to
`/opt/fluxmind` is `0aa1919` (`docs: document PDF structure gate expansion`),
with implementation/eval commit `bb9cb76` (`test: expand PDF structure eval
gate`). It raises the aggregate PDF structure regression gate to 30 seeded
equation/table/figure/algorithm cases using local paper fixtures only, so the
community PDF-structure count target is no longer an open blocker.

Share-link Streamlit error-output follow-up on 2026-06-20 20:20 CST:

```text
Command                                                     Result
----------------------------------------------------------  ----------------------------------------
.venv/bin/python -m pytest tests/test_translation_guard.py pass, 13 selected share-link/UI/
  tests/test_share_links.py tests/test_api.py::...         CLI/API guard tests,
  tests/test_cli_scripts.py -k "share_link_registry" -q    83 deselected, 2 known warnings
.venv/bin/python -m pytest -q                              pass, 617 tests, 2 known warnings
.venv/bin/python -m coverage run -m pytest -q &&           pass, 617 tests,
  .venv/bin/python -m coverage report --fail-under=88      89% total branch coverage
.venv/bin/python scripts/evaluate_rag.py                   pass, 42 answer cases,
                                                            65 retrieval-only cases,
                                                            13 code-output cases,
                                                            30 PDF structure cases,
                                                            42 recorded answers
.venv/bin/python scripts/health_check.py                   pass, including
                                                            Streamlit share-link
                                                            management sanitized
                                                            error-output anchor
.venv/bin/python scripts/openapi_contract.py               pass, ok=true, diff_count=0 against the
  --verify-snapshot /tmp/fluxmind-openapi-contract-current.json
  --require-no-drift --format markdown                      just-exported no-secret snapshot
.venv/bin/python scripts/storage_schema.py --format markdown pass, ok=true, 10 stores, 0 problems;
                                                            no storage-schema drift
git diff --check                                           pass
git status --short --branch                                main...origin/main [ahead 26],
                                                            no local changes before this
                                                            docs refresh
```

The follow-up sanitizes Streamlit share-link management exception output. The
share-link registry API/CLI already returned no-secret public projections; this
patch closes the UI-side error path so OSError, SQLite, or validation failures
shown in the explicit operator panel are passed through the existing path,
URL, bearer token, `sk-...`, and token/secret assignment redaction helper
instead of rendering `str(exc)` directly. The guard test and health check now
verify that the share-link management block no longer contains
`st.error(str(exc))` and uses the sanitized error helper for list/create/
resolve/revoke paths.

Git/documentation drift refresh on 2026-06-20 20:10 CST:

```text
Command                                                     Result
----------------------------------------------------------  ----------------------------------------
.venv/bin/python -m pytest tests/test_docs_status.py        pass, 17 docs/feature-audit/
  tests/test_feature_audit_docs.py tests/test_translation_guard.py
                                                            translation guard tests
.venv/bin/python -m pytest -q                              pass, 616 tests, 2 known warnings
.venv/bin/python scripts/evaluate_rag.py                   pass, 42 answer cases,
                                                            65 retrieval-only cases,
                                                            13 code-output cases,
                                                            30 PDF structure cases,
                                                            42 recorded answers
.venv/bin/python scripts/health_check.py                   pass, repo-status and feature anchors;
                                                            local FAISS index and active-paper
                                                            selection absent in this checkout,
                                                            so those runtime checks were skipped
.venv/bin/python scripts/openapi_contract.py               pass, ok=true, diff_count=0 against the
  --verify-snapshot /tmp/fluxmind-openapi-contract-current.json
  --require-no-drift --format markdown                      just-exported no-secret snapshot
.venv/bin/python scripts/storage_schema.py --format markdown pass, ok=true, 10 stores, 0 problems;
                                                            no storage-schema drift
.venv/bin/python scripts/api_key_registry.py status        pass, backend=none, available=false,
  --format markdown                                         active_keys=0, secrets_exported=false
.venv/bin/python scripts/product_registry.py status        pass, backend=none, available=false,
  --format markdown                                         workspaces=0, secrets_exported=false
.venv/bin/python scripts/share_link_registry.py status     pass, backend=none, available=false,
  --format markdown                                         active_links=0, secrets_exported=false,
                                                            share tokens/URLs exported=false
.venv/bin/python scripts/product_readiness.py              pass, local_foundation_ready=true,
  --format markdown                                         activation_ready=false with expected
                                                            identity/quota/billing blockers
.venv/bin/python scripts/provider_readiness.py             pass, local_foundation_ready=true,
  --format markdown                                         activation_ready=false with expected
                                                            provider/MATLAB blockers
.venv/bin/python scripts/quality_readiness.py              pass, local_foundation_ready=true,
  --format markdown                                         small_group/community false without
                                                            supplied live report evidence
git diff --check                                           pass
git status --short --branch                                main...origin/main [ahead 24],
                                                            no local changes before this
                                                            docs refresh
```

This pass refreshes repo and documentation status after the API-key metadata
projection hardening. It confirms no OpenAPI no-secret snapshot drift, no
storage-schema drift, no docs/feature-anchor drift, and no whitespace drift in
the current checkout. No production deployment was performed, and
`docs/DEPLOYMENT_STATUS.md` remains unchanged because no live Trace-Twin service
facts were refreshed in this pass.

API-key public metadata projection follow-up on 2026-06-20 16:16 CST:

```text
Command                                                     Result
----------------------------------------------------------  ----------------------------------------
.venv/bin/python -m pytest tests/test_api_keys.py          pass, 11 selected API-key/CLI tests,
  tests/test_cli_scripts.py -k "api_key_registry" -q       72 deselected, 1 known warning
.venv/bin/python -m pytest tests/test_api.py               pass, 11 selected API auth/product tests,
  -k "api_key or verify_api_token or product_registry      97 deselected, 2 known warnings
  or quota_guard or rbac_guard" -q
.venv/bin/python -m coverage run -m pytest -q              pass, 616 tests, 2 known warnings
.venv/bin/python -m coverage report --fail-under=88        pass, 89% total branch coverage
.venv/bin/python scripts/health_check.py                   pass, repo-status and feature anchors;
                                                            local FAISS index and active-paper
                                                            selection absent in this checkout,
                                                            so those runtime checks were skipped
.venv/bin/python scripts/evaluate_rag.py                   pass, 42 answer cases,
                                                            65 retrieval-only cases,
                                                            13 code-output cases,
                                                            30 PDF structure cases,
                                                            42 recorded answers
.venv/bin/python scripts/openapi_contract.py               pass, local_contract_ready=true,
  --format json --require-local-contract                    69 routes, 76 operations,
                                                            fingerprint=15bdfa2ae5ec34f1d0045c38b7137cf2b31a27857b1571a035a8efc12d61d18c
.venv/bin/python scripts/openapi_contract.py               pass, ok=true, diff_count=0 against the
  --verify-snapshot /tmp/fluxmind-openapi-contract-current.json
  --require-no-drift --format markdown                      just-exported no-secret snapshot
.venv/bin/python scripts/storage_schema.py --format markdown pass, ok=true, 10 stores, 0 problems
git diff --check                                           pass
```

The follow-up hardens local API-key lifecycle public projections. `create`
still returns the one-time raw token at the JSON top level, but key metadata
returned by create/list/verify/revoke now removes raw owner IDs, owner labels,
and descriptions, replacing them with presence booleans and short fingerprints.
The internal `ApiKeyRecord` still carries owner ID/label for FastAPI auth,
request ownership, product RBAC, and quota attribution, so authentication and
local product guards keep their existing behavior.

Git/docs drift refresh on 2026-06-20 16:08 CST:

```text
Command                                                     Result
----------------------------------------------------------  ----------------------------------------
.venv/bin/python -m pytest tests/test_docs_status.py        pass, 17 docs/feature-audit/
  tests/test_feature_audit_docs.py tests/test_translation_guard.py
                                                            translation guard tests
.venv/bin/python scripts/health_check.py                    pass, repo-status, feature-audit,
                                                            and roadmap drift anchors; local
                                                            FAISS index and active-paper
                                                            selection absent in this checkout,
                                                            so those runtime checks were skipped
.venv/bin/python scripts/evaluate_rag.py                    pass, 42 answer cases,
                                                            65 retrieval-only cases,
                                                            13 code-output cases,
                                                            30 PDF structure cases,
                                                            42 recorded answers
.venv/bin/python scripts/openapi_contract.py                pass, local_contract_ready=true,
  --format json --require-local-contract                    69 routes, 76 operations,
                                                            fingerprint=15bdfa2ae5ec34f1d0045c38b7137cf2b31a27857b1571a035a8efc12d61d18c
.venv/bin/python scripts/openapi_contract.py                pass, ok=true, diff_count=0 against the
  --verify-snapshot /tmp/fluxmind-openapi-contract-current.json
  --require-no-drift --format markdown                      just-exported no-secret snapshot
.venv/bin/python scripts/storage_schema.py --format markdown pass, ok=true, 10 stores, 0 problems
git diff --check                                            pass
git status --short --branch                                 main...origin/main [ahead 22],
                                                            no local changes before this
                                                            docs refresh
```

This refresh keeps the git/worktree source of truth current after the
runtime-event metadata-value redaction docs commit. No production deployment was
performed, and `docs/DEPLOYMENT_STATUS.md` remains unchanged because no live
Trace-Twin service facts were refreshed in this pass.

Runtime event metadata-value redaction follow-up on 2026-06-20 16:01 CST:

```text
Command                                                     Result
----------------------------------------------------------  ----------------------------------------
.venv/bin/python -m pytest tests/test_runtime.py -q         pass, 20 runtime tests
.venv/bin/python -m pytest tests/test_api.py                pass, 7 selected API/admin-event tests,
  -k "admin_events or admin_check_event                     101 deselected, 2 known warnings
  or runtime_event or api_access" -q
.venv/bin/python -m pytest tests/test_admin.py              pass, 1 selected admin aggregation test,
  -k "runtime or event or api_access or admin_check" -q     23 deselected
.venv/bin/python -m coverage run -m pytest -q               pass, 616 tests, 2 known warnings
.venv/bin/python -m coverage report --fail-under=88         pass, 89% total branch coverage
.venv/bin/python scripts/health_check.py                    pass, repo-status and feature anchors;
                                                            local FAISS index and active-paper
                                                            selection absent in this checkout,
                                                            so those runtime checks were skipped
.venv/bin/python scripts/openapi_contract.py                pass, local_contract_ready=true,
  --format json --require-local-contract                    69 routes, 76 operations,
                                                            fingerprint=15bdfa2ae5ec34f1d0045c38b7137cf2b31a27857b1571a035a8efc12d61d18c
.venv/bin/python scripts/openapi_contract.py                pass, ok=true, diff_count=0 against the
  --verify-snapshot /tmp/fluxmind-openapi-contract...       just-exported no-secret snapshot
  --require-no-drift --format json
.venv/bin/python scripts/storage_schema.py --format json    pass, ok=true, 10 stores, 0 problems
git diff --check                                            pass
```

The follow-up extends the runtime-event sanitizer from sensitive metadata keys
to sensitive string values under otherwise safe keys. Newly appended events and
legacy JSONL event projections now replace Bearer tokens, `sk-...` secret-like
tokens, URL/file URI values, and local runtime paths with a no-secret metadata
value placeholder while preserving safe route values such as `/query` and
`/admin/status`. The same sanitized projection remains the basis for
`/admin/events` search, so hidden values cannot be rediscovered through `q`.

Execution input materialization follow-up on 2026-06-20 15:50 CST:

```text
Command                                                     Result
----------------------------------------------------------  ----------------------------------------
.venv/bin/python -m pytest tests/test_providers.py          pass, 13 selected provider tests,
  -k "materialization or entrypoint                         29 deselected
  or policy_violation or docker_execution_provider" -q
.venv/bin/python -m pytest tests/test_api.py                pass, 5 selected API execution tests,
  -k "local_python_job_endpoint" -q                         103 deselected, 2 known warnings
.venv/bin/python -m coverage run -m pytest -q               pass, 614 tests, 2 known warnings
.venv/bin/python -m coverage report --fail-under=88         pass, 89% total branch coverage
.venv/bin/python scripts/health_check.py                    pass, repo-status and feature anchors;
                                                            local FAISS index and active-paper
                                                            selection absent in this checkout,
                                                            so those runtime checks were skipped
.venv/bin/python scripts/openapi_contract.py                pass, local_contract_ready=true,
  --format json --require-local-contract                    69 routes, 76 operations,
                                                            fingerprint=15bdfa2ae5ec34f1d0045c38b7137cf2b31a27857b1571a035a8efc12d61d18c
.venv/bin/python scripts/openapi_contract.py                pass, ok=true, diff_count=0 against the
  --verify-snapshot /tmp/fluxmind-openapi-contract...       just-exported no-secret snapshot
  --require-no-drift --format json
.venv/bin/python scripts/storage_schema.py --format json    pass, ok=true, 10 stores, 0 problems
git diff --check                                            pass
```

The follow-up hardens execution input materialization for local Python,
Octave-compatible, and Docker execution providers. Conflicting input names such
as a file and child path under the same name now return a structured failure
without starting Python, Octave, or Docker and without exposing the temporary
workdir path. Execution entrypoints must resolve to regular files, so a
directory entrypoint now fails with the existing structured missing-entrypoint
diagnostic. The API job endpoint preserves that failure as a normal failed
`code_execution` job instead of surfacing a provider exception.

Product registry referential-integrity follow-up on 2026-06-20 15:39 CST:

```text
Command                                                     Result
----------------------------------------------------------  ----------------------------------------
.venv/bin/python -m pytest tests/test_product_registry.py   pass, 89 selected product/CLI tests,
  tests/test_cli_scripts.py -q                              1 known warning
.venv/bin/python -m pytest tests/test_api.py                pass, 7 selected API auth/product tests,
  -k "product_registry or quota_guard or rbac_guard         100 deselected, 2 known warnings
  or api_key" -q
.venv/bin/python -m coverage run -m pytest -q               pass, 610 tests, 2 known warnings
.venv/bin/python -m coverage report --fail-under=88         pass, 89% total branch coverage
.venv/bin/python scripts/health_check.py                    pass, repo-status and feature anchors;
                                                            local FAISS index and active-paper
                                                            selection absent in this checkout,
                                                            so those runtime checks were skipped
.venv/bin/python scripts/openapi_contract.py                pass, local_contract_ready=true,
  --format json --require-local-contract                    69 routes, 76 operations,
                                                            fingerprint=15bdfa2ae5ec34f1d0045c38b7137cf2b31a27857b1571a035a8efc12d61d18c
.venv/bin/python scripts/openapi_contract.py                pass, ok=true, diff_count=0 against the
  --verify-snapshot /tmp/fluxmind-openapi-contract...       just-exported no-secret snapshot
  --require-no-drift --format json
.venv/bin/python scripts/storage_schema.py --format json    pass, ok=true, 10 stores, 0 problems
git diff --check                                            pass
```

The follow-up hardens the local product registry ledger against orphan writes.
Member, quota, usage, billing, and quota-decision writes now require an active
workspace, while usage and quota-decision writes also require an active product
user. `add_member()` returns a no-secret sanitized member projection, and
`scripts/product_registry.py add-member` now emits that projection instead of
reflecting raw CLI `workspace_id` or `user_id` arguments. Regression coverage
verifies that missing workspace/user writes do not inflate registry counts and
that unsafe member user IDs are not echoed by the CLI.

Artifact path-resolution follow-up on 2026-06-20 15:26 CST:

```text
Command                                                     Result
----------------------------------------------------------  ----------------------------------------
.venv/bin/python -m pytest <artifact/API focused slice>     pass, 20 selected artifact/API tests,
                                                            2 known warnings
.venv/bin/python -m coverage run -m pytest -q               pass, 608 tests, 2 known warnings
.venv/bin/python -m coverage report --fail-under=88         pass, 89% total branch coverage
.venv/bin/python scripts/health_check.py                    pass, repo-status and feature anchors;
                                                            local FAISS index and active-paper
                                                            selection absent in this checkout,
                                                            so those runtime checks were skipped
.venv/bin/python scripts/openapi_contract.py                pass, local_contract_ready=true,
  --format json --require-local-contract                    69 routes, 76 operations,
                                                            fingerprint=15bdfa2ae5ec34f1d0045c38b7137cf2b31a27857b1571a035a8efc12d61d18c
.venv/bin/python scripts/openapi_contract.py                pass, ok=true, diff_count=0 against the
  --verify-snapshot /tmp/fluxmind-openapi-contract...       just-exported no-secret snapshot
  --require-no-drift --format json
.venv/bin/python scripts/storage_schema.py --format json    pass, ok=true, 10 stores, 0 problems
git diff --check                                            pass
```

The follow-up hardens local artifact file-URI resolution for export/download
helpers. It now accepts only local `file://` artifact URIs with no host or
`localhost`, requires absolute file paths, rejects symlink/non-regular artifacts
before export, resolves the final file path canonically under the artifact root,
and returns that canonical path to callers. Regression coverage verifies
canonical `..` aliases and rejects nonlocal file artifact URIs such as
`file://remote-host/...`.

Corpus same-name metadata follow-up on 2026-06-20 15:18 CST:

```text
Command                                                     Result
----------------------------------------------------------  ----------------------------------------
.venv/bin/python -m pytest tests/test_ingestion.py          pass, 46 selected ingestion/metadata
  tests/test_metadata.py -q                                 tests, 1 known warning
.venv/bin/python -m pytest tests/test_api.py                pass, 18 selected corpus/API tests,
  -k "corpus or query_retrieval or artifact" -q             89 deselected, 2 known warnings
.venv/bin/python -m coverage run -m pytest -q               pass, 606 tests, 2 known warnings
.venv/bin/python -m coverage report --fail-under=88         pass, 89% total branch coverage
.venv/bin/python scripts/health_check.py                    pass, repo-status and feature anchors;
                                                            local FAISS index and active-paper
                                                            selection absent in this checkout,
                                                            so those runtime checks were skipped
.venv/bin/python scripts/openapi_contract.py                pass, local_contract_ready=true,
  --format json --require-local-contract                    69 routes, 76 operations,
                                                            fingerprint=15bdfa2ae5ec34f1d0045c38b7137cf2b31a27857b1571a035a8efc12d61d18c
.venv/bin/python scripts/openapi_contract.py                pass, ok=true, diff_count=0 against the
  --verify-snapshot /tmp/fluxmind-openapi-contract...       just-exported no-secret snapshot
  --require-no-drift --format json
.venv/bin/python scripts/storage_schema.py --format json    pass, ok=true, 10 stores, 0 problems
git diff --check                                            pass
```

The follow-up fixes corpus metadata enrichment for selectable PDFs that share
the same filename across `papers/library` and `papers/uploads`. The metadata
store now prefers source-path-specific entries and only applies legacy filename
manifest fallback to curated library/root papers, so an uploaded `paper.pdf`
does not inherit the curated library `paper.pdf` title/authors. Regression
coverage builds same-name library/upload PDFs and verifies that each record
keeps its own source-path metadata. The same local sweep confirms no OpenAPI
no-secret snapshot drift, no storage-schema drift, and no
repo-status/feature-anchor drift in the current checkout.

Job lease-release follow-up on 2026-06-20 15:05 CST:

```text
Command                                                     Result
----------------------------------------------------------  ----------------------------------------
.venv/bin/python -m pytest tests/test_jobs.py -q            pass, 47 tests, 1 known warning
.venv/bin/python -m pytest tests/test_docs_status.py        pass, 18 selected docs/health tests
  tests/test_feature_audit_docs.py tests/test_translation_guard.py
  tests/test_health_check.py::test_main_local_health_check_passes -q
.venv/bin/python -m coverage run -m pytest -q               pass, 605 tests, 2 known warnings
.venv/bin/python -m coverage report --fail-under=88         pass, 89% total branch coverage
.venv/bin/python scripts/health_check.py                    pass, repo-status and feature anchors;
                                                            local FAISS index and active-paper
                                                            selection absent in this checkout,
                                                            so those runtime checks were skipped
.venv/bin/python scripts/openapi_contract.py                pass, local_contract_ready=true,
  --format json --require-local-contract                    69 routes, 76 operations,
                                                            fingerprint=15bdfa2ae5ec34f1d0045c38b7137cf2b31a27857b1571a035a8efc12d61d18c
.venv/bin/python scripts/openapi_contract.py                pass, ok=true, diff_count=0 against the
  --verify-snapshot /tmp/fluxmind-openapi-contract...       just-exported no-secret snapshot
  --require-no-drift --format json
.venv/bin/python scripts/storage_schema.py --format json    pass, ok=true, 10 stores, 0 problems
git diff --check                                            pass
```

The follow-up tightens the durable local job-store state machine. `release_job_lease()`
now only clears leases for queued jobs, matching its contract and preserving
worker lease metadata on terminal jobs such as `succeeded`, `failed`,
`cancelled`, or `dead_lettered`. Regression coverage now verifies that a
mismatched worker ID cannot release another worker's active queued lease and
that a completed job keeps its worker provenance even if a release call is made
later. The same local sweep also confirms no OpenAPI no-secret snapshot drift,
no storage-schema drift, and no repo-status/feature-anchor drift in the current
checkout.

Share-link event-evidence follow-up on 2026-06-20 15:00 CST:

```text
Command                                                     Result
----------------------------------------------------------  ----------------------------------------
.venv/bin/python -m pytest tests/test_share_links.py        pass, 8 selected tests, 2 known warnings
  tests/test_api.py::test_admin_share_link_registry_routes_use_local_backend
  tests/test_cli_scripts.py::test_share_link_registry_cli_lifecycle
  tests/test_cli_scripts.py::test_share_link_registry_cli_markdown_shapes_are_no_secret -q
.venv/bin/python -m coverage run -m pytest -q &&            pass, 604 tests, 2 known warnings,
  .venv/bin/python -m coverage report --fail-under=88       89% total branch coverage;
                                                            src/share_links.py now 90%
.venv/bin/python -m pytest tests/test_docs_status.py        pass, 18 selected docs/health tests
  tests/test_feature_audit_docs.py tests/test_translation_guard.py
  tests/test_health_check.py::test_main_local_health_check_passes -q
.venv/bin/python scripts/health_check.py                    pass, repo-status and feature anchors
                                                            aligned with current local stack;
                                                            local FAISS index and active-paper
                                                            selection absent in this checkout,
                                                            so those runtime checks were skipped
.venv/bin/python scripts/openapi_contract.py                pass, local_contract_ready=true,
  --format json --require-local-contract                    69 routes, 76 operations,
                                                            52 required operations,
                                                            protected auth headers covered
.venv/bin/python scripts/storage_schema.py --format json    pass, ok=true, 10 stores, 0 problems
git diff --check                                            pass
```

The follow-up restores metadata-only observability for share-link resolution
after raw workspace IDs were removed from the public projection. Valid and
revoked resolve events now preserve `product_workspace_present=true` through an
explicit boolean rather than by reading a raw workspace ID from the public
payload. The regression test also asserts the event action sequence and that no
raw workspace ID, token, resource ref, owner user, private path, or secret-like
marker is emitted. Additional registry coverage exercises active, expired,
exhausted, and revoked share-link states; status counts; empty resource-ref
rejection; and corrupted SQLite status fallback without exporting DB paths or
stored contents.

Share-link no-secret follow-up on 2026-06-20 12:22 CST:

```text
Command                                                     Result
----------------------------------------------------------  ----------------------------------------
.venv/bin/python -m pytest tests/test_share_links.py        pass, 82 tests, 2 known warnings
  tests/test_api.py::test_admin_share_link_registry_routes_use_local_backend
  tests/test_cli_scripts.py -q
.venv/bin/python -m pytest -q                               pass, 602 tests, 2 known warnings
.venv/bin/python scripts/evaluate_rag.py                    pass, 42 answer cases, 65 retrieval-only
                                                            cases, 13 code-output cases,
                                                            30 PDF structure cases,
                                                            42 recorded answers
.venv/bin/python scripts/health_check.py                    pass, local feature/documentation anchors;
                                                            local FAISS index and active-paper
                                                            selection absent in this checkout,
                                                            so those runtime checks were skipped
.venv/bin/python scripts/storage_schema.py --format json    pass, ok=true, 10 stores, 0 problems
.venv/bin/python scripts/openapi_contract.py                pass, local_contract_ready=true,
  --format json --require-local-contract                    69 routes, 76 operations,
                                                            52 required operations,
                                                            protected auth headers covered
.venv/bin/python scripts/openapi_contract.py                pass, ok=true, diff_count=0 against the
  --verify-snapshot /tmp/fluxmind-openapi-contract-current.json
  --require-no-drift --format json                          just-exported no-secret snapshot
.venv/bin/python scripts/product_readiness.py --format json pass, local_foundation_ready=true,
                                                            activation_ready=false with expected
                                                            identity/quota/billing blockers
.venv/bin/python scripts/provider_readiness.py --format json pass, local_foundation_ready=true,
                                                             activation_ready=false with expected
                                                             provider/MATLAB/quota blockers
.venv/bin/python scripts/collaboration_readiness.py         pass, ok=true, local_foundation_ready=true,
  --format json                                             safe_default_ready=true,
                                                            activation_ready=false with expected
                                                            private-corpus/share-link blockers
.venv/bin/python scripts/product_activation_rehearsal.py    pass, ok=true, activation_ready=true
  --format json --require-activation
.venv/bin/python scripts/provider_runtime_rehearsal.py      pass, ok=true, local mock image,
  --format json --require-local-foundation                  local Python execution, Octave
                                                            unavailable branch, Docker readiness,
                                                            provider quota guard, and abuse-policy
                                                            denial checked
.venv/bin/python scripts/quality_readiness.py --format json pass, local_foundation_ready=true,
                                                            self_use=met; small_group/community
                                                            remain blocked on explicit live/corpus
                                                            evidence gaps
.venv/bin/python scripts/activation_suite.py --format json  pass, ok=true, local_foundation_ready=true,
                                                            full_activation_ready=false with expected
                                                            product/collaboration/provider/
                                                            platform/community activation blockers
.venv/bin/python scripts/platform_migration_rehearsal.py    pass, wrote object + job-store no-secret
  --include-object-manifest --include-job-store-manifest    rehearsal manifest to /tmp
  --output /tmp/fluxmind-object-and-job-rehearsal.json
.venv/bin/python scripts/platform_migration_rehearsal.py    pass, ok=true, checked_objects=9,
  --verify-object-manifest /tmp/fluxmind-object-and-job-rehearsal.json
  --format json                                             missing/mismatched/extra all 0
.venv/bin/python scripts/platform_migration_rehearsal.py    pass, ok=true, expected/current jobs=0,
  --verify-job-store-manifest /tmp/fluxmind-object-and-job-rehearsal.json
  --format json                                             expected/current idempotency claims=0,
                                                            missing/mismatched/extra all 0
.venv/bin/python scripts/share_link_registry.py             pass, backend=none, available=false,
  --format markdown status                                  secrets/tokens/URLs exported=false
.venv/bin/python scripts/api_key_registry.py                pass, backend=none, available=false,
  --format markdown status                                  secrets exported=false
git diff --check                                            pass
```

The follow-up hardens the local share-link public projection: list/create/
resolve/revoke summaries now omit raw `workspace_id` and expose only
`workspace_present` plus a short `workspace_fingerprint`. CLI Markdown uses the
same no-secret fields, and the Streamlit management panel no longer derives its
default workspace input from public share-link summaries. Regression tests now
assert that API, CLI, and registry projections do not echo the sample raw
workspace ID while preserving internal storage and RBAC decisions on the full
record.

Local audit follow-up on 2026-06-20 11:04 CST:

```text
Command                                                     Result
----------------------------------------------------------  ----------------------------------------
.venv/bin/python -m pytest                                  pass, 602 tests, 2 known warnings
.venv/bin/python -m coverage run -m pytest &&
.venv/bin/python -m coverage report --fail-under=88         pass, 602 tests, 89% total branch coverage
.venv/bin/python scripts/evaluate_rag.py                    pass, 42 answer cases, 65 retrieval-only
                                                            cases, 13 code-output cases,
                                                            30 PDF structure cases,
                                                            42 recorded answers
.venv/bin/python scripts/health_check.py                    pass, local feature/documentation anchors;
                                                            local FAISS index and active-paper
                                                            selection absent in this checkout,
                                                            so those runtime checks were skipped;
                                                            includes product-registry
                                                            read-RBAC route anchor
                                                            and explicit Streamlit
                                                            product-registry
                                                            management flag anchor
                                                            and API-key create
                                                            one-time-token
                                                            JSON-output anchor
                                                            and share-link
                                                            registry/API/storage
                                                            and Streamlit
                                                            management flag
                                                            anchors
                                                            and activation-suite
                                                            local foundation/API/UI anchors
                                                            plus OpenAPI-contract
                                                            suite gate anchors
                                                            plus admin-check
                                                            runtime event
                                                            and status-summary
                                                            anchors
                                                            plus full activation
                                                            action plan anchor
                                                            and product activation
                                                            rehearsal admin API/UI
                                                            anchors
                                                            and provider runtime
                                                            rehearsal admin API/UI
                                                            anchors
                                                            and storage migration
                                                            public projection
                                                            plus platform migration
                                                            rehearsal admin API/UI
                                                            anchors
                                                            and live-report
                                                            admin/API upload anchors
                                                            and next-quality-evidence
                                                            summary anchor
                                                            and quality evidence-request
                                                            summary anchor
                                                            and quality evidence
                                                            collection plan anchor
                                                            and quality-readiness
                                                            admin API/UI anchors
                                                            and remote SSH
                                                            source anchors for
                                                            the same deployed
                                                            safety gates,
                                                            provider-readiness
                                                            invalid-limit
                                                            blocker anchor
.venv/bin/python scripts/storage_schema.py --format markdown pass, ok=true, 10 stores, 0 problems
.venv/bin/python scripts/product_readiness.py               pass, local_foundation_ready=true,
                                                            activation_ready=false with expected
                                                            identity/quota/billing blockers
.venv/bin/python scripts/product_activation_rehearsal.py    pass, ok=true and activation_ready=true
  --format markdown --require-activation                    against disposable local SQLite stores;
                                                            workspace-isolation denial checked;
                                                            raw tokens and paths exported=false
Product activation admin route smoke                        pass, HTTP 200 via FastAPI TestClient;
                                                            `/admin/product-activation-rehearsal`
                                                            and report download return
                                                            no-secret JSON/Markdown;
                                                            raw token, workspace ID, SQLite path,
                                                            and file URI markers
                                                            were not echoed
.venv/bin/python scripts/collaboration_readiness.py          pass, ok=true, safe_default_ready=true,
  --format markdown                                          activation_ready=false with expected
                                                            private-corpus/share-link blockers;
                                                            policy_scenario_count=13 and
                                                            identifiers/share tokens/URLs/paths
                                                            exported=false
Collaboration readiness admin route smoke                    pass, HTTP 200 via FastAPI TestClient;
                                                            `/admin/collaboration-readiness`
                                                            and report download return
                                                            no-secret JSON/Markdown;
                                                            workspace/user/corpus/share markers
                                                            were not echoed
.venv/bin/python scripts/provider_readiness.py              pass, local_foundation_ready=true,
                                                            activation_ready=false with expected
                                                            external provider/MATLAB blockers
.venv/bin/python scripts/provider_runtime_rehearsal.py      pass, ok=true, local mock image/Python
  --format markdown --require-local-foundation              execution/Octave branch checked;
                                                            execution abuse-policy denial
                                                            checked without source/stdout/stderr;
                                                            external_activation_ready=false
Provider runtime admin route smoke                          pass, HTTP 200 via FastAPI TestClient;
                                                            `/admin/provider-runtime-rehearsal`
                                                            and report download run
                                                            the actual local drill;
                                                            raw path, file URI,
                                                            and sk-like secret
                                                            markers were not echoed
.venv/bin/python scripts/platform_migration_rehearsal.py    pass, rehearsal_ok=true,
  --include-job-store-manifest --format markdown            job_store_manifest_ready=true,
                                                            jobs=0 and claims=0 in current
                                                            checkout; payloads/raw IDs
                                                            exported=false
.venv/bin/python scripts/platform_migration_rehearsal.py    pass, ok=true,
  --verify-job-store-manifest /tmp/... --format markdown    missing/mismatched/extra jobs
                                                            or claims all 0; payloads/raw
                                                            IDs exported=false
Platform migration rehearsal admin route smoke              pass, HTTP 200 via FastAPI TestClient;
                                                            `/admin/platform-migration-rehearsal`
                                                            and report download run the
                                                            staged local migration drill;
                                                            object/job-store manifest
                                                            summaries are ready while
                                                            raw manifests, file URIs,
                                                            temp paths, and sensitive
                                                            job markers are not echoed
.venv/bin/python scripts/quality_readiness.py               pass, local_foundation_ready=true,
                                                            self_use=met, small_group=false
                                                            without supplied live report,
                                                            community=false with measured
                                                            target gap summary;
                                                            Evidence Requests list
                                                            corpus_manifest,
                                                            eval_baseline, and
                                                            live_eval_report source
                                                            gaps without paths;
                                                            Evidence Collection Plan
                                                            emits placeholder
                                                            evaluate_rag.py and
                                                            quality_readiness.py
                                                            commands without
                                                            concrete URLs, paths,
                                                            prompts, answers,
                                                            or tokens
Quality readiness POST live-report smoke                    pass, HTTP 200 via FastAPI TestClient;
                                                            uploaded 107/107 retrieval live
                                                            report makes small_group_ready=true,
                                                            next_evidence_request=community,
                                                            and report download includes
                                                            Evidence Requests and
                                                            Evidence Collection Plan;
                                                            private path and secret marker
                                                            were not echoed
.venv/bin/python scripts/activation_suite.py                pass, local_foundation_ready=true,
  --format markdown --require-target local_foundation       full_activation_ready=false
                                                            with expected product-readiness,
                                                            collaboration-readiness,
                                                            provider,
                                                            platform-migration,
                                                            and community-quality
                                                            activation blockers;
                                                            next_evidence_target=small_group
                                                            and source=live_eval_report
                                                            without supplied live report;
                                                            Activation Action Plan
                                                            groups product/provider/platform/
                                                            community blockers plus the
                                                            collaboration gate into
                                                            placeholder command and
                                                            verification steps;
                                                            raw child reports,
                                                            paths, and secrets
                                                            exported=false
.venv/bin/python scripts/openapi_contract.py                 pass, local_contract_ready=true,
  --format json --require-local-contract                     69 routes, 76 operations,
                                                            52 required operations,
                                                            protected auth headers
                                                            covered; operation fingerprint
                                                            emitted; raw schema
                                                            exported=false
.venv/bin/python scripts/share_link_registry.py              pass, token_prefix=fms_,
  --db /tmp/... create                                       link_id_prefix=share_,
                                                            resource_ref_present=true,
                                                            raw_ref_exported=false
.venv/bin/python scripts/openapi_contract.py                 pass, ok=true,
  --verify-snapshot /tmp/fluxmind-openapi-contract-current.json
  --require-no-drift --format markdown                       diff_count=0 against the
                                                            just-exported no-secret
                                                            snapshot; raw schema,
                                                            paths, content, and
                                                            secrets exported=false
printf malicious OpenAPI snapshot |                          pass, ok=false with
  .venv/bin/python scripts/openapi_contract.py                snapshot_contract_shape_invalid,
  --verify-snapshot - --format json                           snapshot_raw_schema_included,
                                                            snapshot_valid=false fields;
                                                            injected path/component/
                                                            hunter2 values not echoed
.venv/bin/python scripts/openapi_contract.py                  pass, exits 2 with explicit
  --require-no-drift                                          "--verify-snapshot required"
                                                            error instead of ambiguous
                                                            current-contract output
printf non-object JSON |                                      pass, exits 2 with explicit
  .venv/bin/python scripts/openapi_contract.py                "snapshot JSON must be an object"
  --verify-snapshot -                                        error
Activation suite POST live-report smoke                     pass, HTTP 200 via FastAPI TestClient;
                                                            uploaded 107/107 retrieval live
                                                            report makes small_group_ready=true
                                                            and next_evidence_target=community
                                                            with corpus_manifest/
                                                            eval_baseline/
                                                            live_eval_report sources;
                                                            activation action plan
                                                            remains placeholder-only;
                                                            private path and secret marker
                                                            were not echoed
git diff --check                                            pass
Focused API/admin/artifact/job tests                        pass, tests/test_api.py,
                                                            tests/test_admin.py,
                                                            tests/test_artifacts.py, and
                                                            tests/test_jobs.py cover blank
                                                            request-id sanitation,
                                                            unsafe request-id
                                                            suppression,
                                                            /query/report header retention,
                                                            full-history artifact-ID export,
                                                            artifact public projection,
                                                            safe artifact/profile
                                                            download filenames,
                                                            shared API/UI profile
                                                            report filename helper,
                                                            sanitized public cost
                                                            summaries with bounded
                                                            finite numeric output,
                                                            safe job-list summaries
                                                            and job search
                                                            projections,
                                                            owner-label presence
                                                            flags instead of raw
                                                            owner labels in
                                                            `/jobs` summaries,
                                                            runtime-event request-id
                                                            redaction,
                                                            artifact SQLite sibling
                                                            preservation, and sanitized
                                                            /admin/events metadata/search,
                                                            runtime-event sensitive-key,
                                                            camelCase path/URL, and
                                                            token-value/message redaction,
                                                            including bare sk-like secret
                                                            tokens in event messages,
                                                            including product user/workspace
                                                            identifiers while preserving
                                                            aggregate counts,
                                                            plus no-secret Streamlit
                                                            latest-job summaries,
                                                            runtime-event viewing, and
                                                            admin status/report
                                                            ownership/request-id summaries,
                                                            plus route-fingerprint
                                                            API-access audit events
                                                            without raw request paths,
                                                            plus local product-registry
                                                            workspace list/detail/
                                                            permission-check read
                                                            routes guarded by
                                                            admin-write RBAC when
                                                            the product RBAC guard
                                                            is enabled,
                                                            plus explicit
                                                            opt-in flag for
                                                            Streamlit
                                                            product-registry
                                                            management forms
Focused product activation rehearsal tests                  pass, tests/test_product_activation_rehearsal.py
                                                            and CLI coverage verify hash-only
                                                            API-key lifecycle, local RBAC,
                                                            quota limiting, billing
                                                            attribution, readiness activation,
                                                            no token/path export, and
                                                            fixed-root state isolation
                                                            without deleting caller-root
                                                            SQLite files, plus
                                                            rejection of markdown
                                                            create output before an
                                                            unrecoverable one-time
                                                            token can be generated
Focused collaboration readiness tests                       pass, tests/test_collaboration_readiness.py
                                                            plus API endpoint/report and
                                                            CLI coverage verify the
                                                            disabled-safe default,
                                                            activation blockers,
                                                            product-registry/RBAC/token-store
                                                            prerequisites, role policy matrix,
                                                            and no workspace/user/corpus/share
                                                            identifier, token, URL, or path export
Focused provider runtime rehearsal tests                    pass, tests/test_provider_runtime_rehearsal.py
                                                            and provider-runtime API smoke tests
                                                            and CLI coverage verify local mock
                                                            image generation, Python artifact
                                                            capture, Octave runtime branch,
                                                            Docker readiness code, provider
                                                            quota/cost guard allowed and
                                                            denied decisions, local
                                                            provider foundation, and no
                                                            path/secret export, plus
                                                            fixed-root artifact isolation
                                                            with stale hidden-artifact
                                                            cleanup and without changing caller-root
                                                            artifacts
Focused activation suite tests                              pass, tests/test_activation_suite.py
                                                            plus API/UI source anchors
                                                            and CLI coverage verify the
                                                            aggregate no-secret local
                                                            foundation gate, expected
                                                            full-activation blockers,
                                                            in-memory live eval
                                                            evidence input and
                                                            next-quality-evidence
                                                            target/gap/source
                                                            summary plus legacy
                                                            gap-summary fallback,
                                                            full activation
                                                            action plan grouped
                                                            by product/provider/platform/
                                                            community area,
                                                            OpenAPI contract
                                                            participation in the
                                                            local foundation gate,
                                                            and failure projection
                                                            as an openapi_contract
                                                            local blocker,
                                                            target-root/eval-file
                                                            CLI semantics, Markdown
                                                            output, nonzero
                                                            full-activation require
                                                            behavior, on-demand
                                                            admin JSON/Markdown
                                                            GET/POST routes,
                                                            Streamlit button/download
                                                            panel and live-report
                                                            upload, and no path/token/
                                                            payload/live-report/
                                                            raw-child-report export
Focused quality readiness API/UI tests                      pass, tests/test_api.py and
                                                            tests/test_health_check.py
                                                            verify GET/POST
                                                            /admin/quality-readiness,
                                                            Markdown report download,
                                                            live-report input,
                                                            Streamlit run/upload/download
                                                            anchors, evidence-request
                                                            source output,
                                                            evidence collection
                                                            plan output, and no
                                                            path/secret echo
Focused provider quota/cost guard tests                     pass, tests/test_costs.py,
                                                            tests/test_provider_guard.py, and
                                                            chain/streaming regressions verify
                                                            disabled-by-default behavior,
                                                            prompt/completion/cost limit
                                                            denials, no-secret policy output,
                                                            non-finite and extreme-exponent
                                                            cost/rate config sanitation,
                                                            denial before provider client
                                                            construction, and separate
                                                            provider_quota_guard events
                                                            instead of provider_failure
                                                            pollution
Focused provider usage extraction tests                     pass, tests/test_evaluation.py,
                                                            tests/test_api.py, and
                                                            tests/test_admin.py verify
                                                            provider token usage
                                                            metadata extraction,
                                                            malformed usage-field
                                                            fallback, zero-token
                                                            preservation, derived
                                                            totals, query usage
                                                            events, and admin
                                                            cost aggregation
Focused provider readiness tests                            pass, invalid provider quota
                                                            guard limits block activation
                                                            instead of reporting readiness
Focused job-store manifest tests                            pass, tests/test_storage_migration.py
                                                            and CLI coverage verify durable
                                                            job-store manifest/verify,
                                                            job/idempotency claim token
                                                            matching, changed job-state
                                                            detection, nested/camelCase unsafe
                                                            manifest-field rejection, explicit
                                                            staging-root overlap rejection,
                                                            and no payload/
                                                            owner/request/worker/idempotency
                                                            export
Focused platform migration API/UI tests                     pass, tests/test_storage_migration.py,
                                                            tests/test_api.py, and
                                                            tests/test_health_check.py verify
                                                            standalone
                                                            `/admin/platform-migration-rehearsal`,
                                                            Markdown report download,
                                                            Streamlit button/download
                                                            anchors, public projection
                                                            flags, object/job-store
                                                            manifest summaries, and no
                                                            raw manifest/path/secret/
                                                            payload export
Focused quality-readiness tests                             pass, tests/test_quality_readiness.py
                                                            and CLI/health anchors verify
                                                            per-target count gaps and
                                                            live-answer quality gaps,
                                                            evidence-request
                                                            source classification,
                                                            next/community evidence
                                                            collection plans with
                                                            placeholder commands,
                                                            plus in-memory live report
                                                            summaries without exporting
                                                            report paths, prompts,
                                                            answers, or source content
Focused bootstrap-doc tests                                 pass, tests/test_docs_status.py
                                                            now guards AGENTS/CLAUDE
                                                            current no-secret command
                                                            snippets
Focused README command tests                                pass, tests/test_docs_status.py
                                                            now guards bilingual README
                                                            current no-secret command
                                                            snippets
```

This local audit hardens API request ID normalization, `/query/report` download
headers, artifact export lookup/metadata mirror behavior, and no-secret
runtime-event viewing across the API and Streamlit, plus admin status/report
latest-event summaries. Runtime-event admin projections now also redact common
sensitive metadata key variants such as access-token, API-key, token-value,
camelCase/PascalCase path or URL fields, raw prompt, and raw answer fields while
preserving safe status/count/token metrics for diagnostics.
Top-level runtime-event messages with URL/path/token/prompt/answer-like value
assignments or bare `sk-...` secret-like tokens are also replaced in the
admin-facing projection before events are returned or searched. Unsafe legacy
runtime-event request IDs are also hidden behind
`request_id_present`/`request_id_redacted` booleans.
Admin status/report now use owner counts, ownership-source buckets, sanitized
runtime-event metadata, and `request_id_present` booleans instead of returning
owner IDs or raw request IDs. API-access audit runtime events now report only
route presence and a short route-template fingerprint, not raw request paths or
route strings. `GET /jobs` summaries now expose only `owner_id_present` and
`owner_label_present` instead of raw owner values while preserving exact local
`owner_id` filtering.
Live answer/retrieval eval JSON reports also redact request identifiers into
`request_id_present` and `request_id_redacted` booleans, so deployable quality
evidence can be archived without copying raw request IDs from live API
responses.
The same local pass adds a disposable product
activation rehearsal for the SQLite API-key/product-registry path, exposes it
through `GET /admin/product-activation-rehearsal`,
`GET /admin/product-activation-rehearsal/report`, and a Streamlit
admin-panel button/download path, and now proves cross-workspace access denial
without exporting workspace/user identifiers before private corpora or share
links are enabled. It also adds a local provider runtime rehearsal
for the no-key provider contracts with `GET /admin/provider-runtime-rehearsal`,
`GET /admin/provider-runtime-rehearsal/report`, and a Streamlit
admin-panel button/download path. The provider runtime rehearsal now includes
Python/Octave execution abuse-policy denial checks and exports only counts and
booleans for those unsafe cases. The migration
rehearsal now also emits and verifies a no-secret durable job-store migration manifest for staged
`jobs.sqlite3` state, using hashed job/idempotency-claim tokens and aggregate
status metadata instead of job payloads, owner IDs, request IDs, worker IDs,
idempotency keys, logs, artifacts, or execution output. `quality_readiness.py`
now also emits a no-secret target gap summary for self-use, small-group, and
community maturity targets, including current/expected/gap values and live-answer
quality gaps without exporting report paths, prompts, answers, or source paths.
Live report filenames are also replaced by a generic report label in the
readiness output. The same readiness output now turns the next-target and
community evidence requests into no-secret evidence collection plans with
placeholder `evaluate_rag.py` and `quality_readiness.py` commands, so operators
can collect live retrieval/answer reports and re-check readiness without the
status payload embedding concrete URLs, report paths, prompts, answers, source
content, raw report payloads, or API tokens.
The staged platform migration rehearsal is now also exposed as a standalone
operator surface through `GET /admin/platform-migration-rehearsal`,
`GET /admin/platform-migration-rehearsal/report`, and a Streamlit admin-panel
button/download path. The public projection reports readiness flags, copy/check
summaries, storage-schema status, and object/job-store manifest summaries, but
does not include raw manifests, source paths, temp staging paths, file URIs, job
payloads, owner IDs, request IDs, worker IDs, idempotency keys, logs, artifacts,
or external storage coordinates.
The same local pass now adds `scripts/activation_suite.py` as a single
operator-facing no-secret local gate. It aggregates actual product readiness,
local product activation rehearsal, provider runtime, durable job-store
migration rehearsal, and quality readiness summaries into local foundation,
small-group, community, and full-activation targets without embedding raw child
reports, local paths, tokens, job payloads, artifact URIs, prompts, answers, or
external account data. Its aggregate quality summary also projects the next
no-secret evidence target and metric gaps, so the operator can see the immediate
small-group/community evidence delta without opening raw eval reports. Its full
activation action plan now groups actual product readiness, provider
activation, platform migration activation, and community-quality blockers into
placeholder commands plus verification commands, without enabling external
services or exporting concrete URLs, paths, prompts, answers, source content, or
credentials.
The activation suite is also exposed through explicit on-demand admin surfaces:
`GET /admin/activation-suite`, `GET /admin/activation-suite/report`,
`POST /admin/activation-suite`, `POST /admin/activation-suite/report`, and a
Streamlit admin-panel button/download path. The POST routes and Streamlit
uploader accept no-secret `evaluate_rag.py --json-report` objects as in-memory
live evidence, so small-group quality can be proven without first writing a
server-side report path into the suite. The public output keeps only aggregate
counts/rates/gaps and does not echo live report paths, filenames, prompts,
answers, source content, or raw report payloads. It is intentionally not folded
into the default `/admin/status` refresh because the suite runs local
rehearsals.
The same local pass now adds a no-secret OpenAPI contract gate:
`scripts/openapi_contract.py` checks generated FastAPI schema coverage for
required route/method pairs, operation summaries and IDs, response declarations,
protected auth header declarations, and route-group coverage without embedding
the raw schema. It also emits a stable operation fingerprint and can compare the
current no-secret report with a prior no-secret JSON snapshot for contract
drift. `GET /admin/openapi-contract`, `GET /admin/openapi-contract/report`,
`POST /admin/openapi-contract/verify`,
`POST /admin/openapi-contract/verify/report`, and the Streamlit admin panel
expose the same summary and drift verification for frontend/API split work. The
activation suite CLI/API/UI entrypoints also pass the generated schema into the
aggregate, so `openapi_contract.ok` is included in the local foundation gate
when operators run the suite through those surfaces. The snapshot verifier now
accepts only non-negative bounded JSON integer counts, valid booleans, and
64-hex fingerprints for comparison; negative, stringified, extreme, malformed,
or raw-schema-shaped snapshot values are treated as invalid shape and are not
echoed into JSON/Markdown diffs.
The same local API pass also adds `admin_check` runtime events for explicit
readiness/rehearsal routes when API access auditing is enabled. Those events
record check names, ok/blocked state, count fields, booleans, and blocker
counts for OpenAPI contract, OpenAPI snapshot verification, quality readiness,
product activation rehearsal, provider runtime rehearsal, platform migration
rehearsal, and activation-suite runs. They do not carry uploaded snapshots,
raw live reports, OpenAPI fingerprints, paths, prompts, answers, tokens, or
child payloads, and the Streamlit runtime-event filter now includes
`admin_check`. Admin status/report, metrics, and Streamlit now summarize the
same events by check name, code, ok/blocked state, and blocker-count totals;
latest admin-check event metadata is reduced to a fixed safe-key set so
malformed legacy event fields such as snapshots, raw reports, fingerprints, and
paths are not exposed through the status surface. Unsafe legacy code/check
labels are grouped as `invalid`, and negative blocker counts are clamped before
status/report/metrics totals are emitted.
The agent bootstrap files `AGENTS.md` and `CLAUDE.md` now also list the current
product activation rehearsal, provider runtime rehearsal, platform migration
rehearsal, activation suite, OpenAPI contract, and object/job-store migration
manifest commands, with a docs-status test guarding those snippets.
The no-secret readiness and rehearsal CLI wrappers now share
`scripts/_safe_cli.py` for OSError reporting, so output-path or local-file
failures preserve safe diagnostic text while redacting paths, URLs,
bearer/sk-style tokens, and token/secret-like assignments;
`scripts/health_check.py` guards that wiring.
The bilingual `README.md` verification section now mirrors the same current
no-secret command set for external GitHub readers, with English and Chinese
command blocks both guarded by `tests/test_docs_status.py`.
It is not a deployment record;
`docs/DEPLOYMENT_STATUS.md` remains the source of truth for live Trace-Twin
state.

Previous local verification on 2026-06-17 03:11 CST:

```text
Command                                                     Result
----------------------------------------------------------  ----------------------------------------
.venv/bin/python -m pytest                                  pass, 435 tests, 2 known warnings
.venv/bin/python -m coverage run -m pytest                  pass, 435 tests, 2 known warnings
.venv/bin/python -m coverage report --fail-under=88         pass, 88% total branch coverage
.venv/bin/python -m coverage report --sort=cover            pass, src/product_readiness.py at 97%,
                                                            src/product_registry.py at 92%,
                                                            src/provider_readiness.py at 93%,
                                                            src/storage_migration.py at 86%,
                                                            src/quality_readiness.py at 86%,
                                                            src/evaluation.py at 88%,
                                                            scripts/platform_migration_rehearsal.py at 97%
.venv/bin/python scripts/evaluate_rag.py                    pass, 42 answer cases, 65 retrieval-only
                                                            cases, 13 code-output cases,
                                                            30 PDF structure cases,
                                                            42 recorded answers
.venv/bin/python scripts/health_check.py                    pass, including distributed job-store,
                                                            migration-preflight,
                                                            migration-rehearsal, and
                                                            product/provider/quality-readiness,
                                                            quality live-threshold anchors,
                                                            API-key registry, product registry,
                                                            product quota guard, and product
                                                            RBAC guard plus product registry
                                                            management/object-storage-manifest
                                                            and object-manifest-verifier anchors
.venv/bin/python scripts/platform_migration_rehearsal.py    pass, rehearsal_ok=true,
  --include-object-manifest --format json                    object_manifest_ready=true,
                                                            objects=9, unique=8,
                                                            paths/filenames/bucket/secrets
                                                            exported=false
.venv/bin/python scripts/platform_migration_rehearsal.py    pass, ok=true, checked=9,
  --verify-object-manifest /tmp/... --format json            missing=0, mismatched=0, extra=0,
                                                            manifest_errors=0,
                                                            paths/filenames/bucket/secrets
                                                            exported=false
.venv/bin/python scripts/storage_schema.py --format markdown pass, ok=true, 10 stores, 0 problems
.venv/bin/python scripts/api_key_registry.py status          pass, backend=none, available=false,
  --format markdown                                          active_keys=0, secrets_exported=false
.venv/bin/python scripts/product_registry.py status          pass, backend=none, available=false,
  --format markdown                                          users=0, workspaces=0,
                                                            quota_limits=0, usage_events=0,
                                                            billing_accounts=0,
                                                            secrets_exported=false
.venv/bin/python scripts/platform_migration_preflight.py     pass, preflight_ok=true,
                                                            activation_ready=false with expected
                                                            external-backend blockers
.venv/bin/python scripts/platform_migration_preflight.py     pass, --require-activation exits 1:
  --require-activation                                      preflight_ok=true but activation_ready=false
.venv/bin/python scripts/platform_migration_rehearsal.py     pass, rehearsal_ok=true, copied_files=9,
                                                            restore_check_ok=true,
                                                            staged_storage_schema_ok=true,
                                                            blockers=none
.venv/bin/python scripts/product_readiness.py                pass, local_foundation_ready=true,
                                                            activation_ready=false with expected
                                                            identity/quota/billing blockers
.venv/bin/python scripts/product_readiness.py                pass, --require-activation exits 1:
  --require-activation                                      local_foundation_ready=true but
                                                            activation_ready=false
.venv/bin/python scripts/provider_readiness.py               pass, local_foundation_ready=true,
                                                            activation_ready=false with expected
                                                            external provider/MATLAB blockers
.venv/bin/python scripts/provider_readiness.py               pass, --require-activation exits 1:
  --require-activation                                      local_foundation_ready=true but
                                                            activation_ready=false
.venv/bin/python scripts/quality_readiness.py                pass, local_foundation_ready=true,
                                                            self_use=met, small_group=false
                                                            without supplied live report,
                                                            community=false with measured gaps
.venv/bin/python scripts/quality_readiness.py                pass with /tmp/fluxmind-pdf30-eval-report.json:
  --live-report /tmp/fluxmind-pdf30-eval-report.json         pdf_structure_case_count=30,
  --format markdown                                          small_group=false because the local
                                                            report has no live retrieval evidence;
                                                            community=false without a PDF blocker
.venv/bin/python scripts/quality_readiness.py                pass, --require-target community
  --require-target community                                exits 1 because community_ready=false
git diff --check                                            pass
Admin status smoke                                          distributed_job_store backend=local,
                                                            available=true,
                                                            external_configured=false;
                                                            local worker bridge ready=true,
                                                            distributed_job_store_not_configured
                                                            remains the expected external blocker
Product readiness smoke                                     local_foundation_ready=true,
                                                            activation_ready=false,
                                                            blockers include identity provider,
                                                            product registry, key lifecycle,
                                                            quota store, billing provider, and
                                                            billing attribution
Temporary SQLite API-key registry smoke                     create/verify/list/revoke passed
                                                            against an isolated temp registry;
                                                            list/verify did not return raw
                                                            tokens
Temporary SQLite product-registry smoke                     bootstrap workspace, quota, usage,
                                                            and billing attribution passed
                                                            against an isolated temp registry;
                                                            status did not export secrets or
                                                            content
Temporary SQLite product-RBAC smoke                         owner/admin/member/viewer role
                                                            decisions passed against an isolated
                                                            temp registry; viewer query was
                                                            allowed, viewer job-submit and
                                                            member corpus-write were denied
                                                            with metadata-only reasons
Product readiness temp-registry smoke                       enabling sqlite registry in an
                                                            isolated temp env cleared
                                                            api_key_lifecycle_not_configured;
                                                            identity/quota/billing blockers
                                                            remained expected
Provider readiness smoke                                    local_foundation_ready=true,
                                                            activation_ready=false,
                                                            blockers include external providers
                                                            disabled, external image provider,
                                                            hosted execution provider, MATLAB
                                                            backend, and provider quota guard
Remote product readiness smoke                              `/opt/fluxmind/venv/bin/python
                                                            scripts/product_readiness.py`
                                                            returned local_foundation_ready=true,
                                                            activation_ready=false;
                                                            product_quota_guard_enabled=false
                                                            remains an advisory in the default
                                                            production config;
                                                            --require-activation exited 1;
                                                            admin metrics include
                                                            fluxmind_product_* and omit
                                                            api_key/owner_id
Remote provider readiness smoke                             `/opt/fluxmind/venv/bin/python
                                                            scripts/provider_readiness.py`
                                                            returned local_foundation_ready=true,
                                                            activation_ready=false;
                                                            --require-activation exited 1;
                                                            authenticated admin status returned
                                                            the same provider_readiness state;
                                                            admin metrics include
                                                            fluxmind_provider_* and omit
                                                            api_key/owner_id
Remote quality readiness smoke                              `/opt/fluxmind/venv/bin/python
                                                            scripts/quality_readiness.py`
                                                            returned local_foundation_ready=true,
                                                            small_group_ready=false without
                                                            live report evidence,
                                                            community_ready=false;
                                                            --require-target community exited 1
Remote quality live-report smoke                            server-local evaluate_rag.py
                                                            --retrieval-url wrote a no-secret
                                                            report with 107/107 live retrieval;
                                                            quality_readiness.py --live-report
                                                            returned small_group_ready=true,
                                                            community_ready=false,
                                                            live_retrieval_pass_rate=1.0,
                                                            live_answer_result_count=0,
                                                            live answer quality n/a;
                                                            --require-target small_group
                                                            exited 0 and
                                                            --require-target community exited 1
Remote API-key registry smoke                               `/opt/fluxmind/venv/bin/python
                                                            scripts/api_key_registry.py status`
                                                            returned backend=none,
                                                            available=false, active_keys=0,
                                                            secrets_exported=false
Remote product registry smoke                               `/opt/fluxmind/venv/bin/python
                                                            scripts/product_registry.py status`
                                                            returned backend=none,
                                                            available=false, users=0,
                                                            workspaces=0, rbac_available=false,
                                                            secrets_exported=false
Remote product quota guard smoke                            `/opt/fluxmind/api.py`,
                                                            `/opt/fluxmind/src/product_registry.py`,
                                                            and `/opt/fluxmind/src/admin.py`
                                                            contain enforce_product_quota,
                                                            quota_decision, and
                                                            fluxmind_product_quota_guard_enabled;
                                                            production flag remains false
Remote product RBAC guard smoke                             `/opt/fluxmind/api.py`,
                                                            `/opt/fluxmind/src/product_registry.py`,
                                                            `/opt/fluxmind/src/admin.py`, and
                                                            `/opt/fluxmind/scripts/product_registry.py`
                                                            contain enforce_product_rbac,
                                                            permission_decision,
                                                            fluxmind_product_rbac_guard_enabled,
                                                            and check-permission;
                                                            production flag remains false
Remote product registry management smoke                    `/opt/fluxmind/api.py`,
                                                            `/opt/fluxmind/app.py`, and
                                                            `/opt/fluxmind/src/product_registry.py`
                                                            contain admin_product_registry_create_workspace,
                                                            render_product_registry_management,
                                                            and list_workspace_summaries;
                                                            authenticated
                                                            /admin/product-registry/status
                                                            returned backend=none,
                                                            available=false,
                                                            reason=product_registry_not_configured
Remote object manifest smoke                                `/opt/fluxmind/venv/bin/python
                                                            scripts/platform_migration_rehearsal.py
                                                            --include-object-manifest`
                                                            returned rehearsal_ok=true,
                                                            object_manifest_ready=true,
                                                            objects=19, unique=18,
                                                            source_paths_exported=false,
                                                            filenames_exported=false,
                                                            bucket_exported=false,
                                                            secrets_exported=false
Remote storage-schema smoke                                 `/opt/fluxmind/venv/bin/python
                                                            scripts/storage_schema.py`
                                                            returned ok=true, store_count=9,
                                                            problem_count=0, and optional
                                                            api_key_registry_sqlite ok=true,
                                                            product_registry_sqlite ok=true
Remote deployment health smoke                              public HTTPS and default SSH health
                                                            passed after the SSH command timeout
                                                            floor was raised to 180s for the
                                                            expanded admin/status checks
```

The earlier small-group quality work was deployed and live-verified through the
100/100 retrieval snapshot on 2026-06-16 14:17 CST. The current source/eval
baseline `9b1cbc5` advances the deterministic RAG baseline beyond that snapshot
and is now live-verified as a 107-question no-LLM retrieval gate: 42 answer
cases, 42 recorded answers, and 65 retrieval-only source/page cases with topic,
lane, and ontology coverage gates.
It also adds a local paper-to-code handoff section to `/query/report` for
implementation and code-generation report exports, plus local Python
code-output eval cases that verify generated plot/text artifacts, reusable
execution-template output, paper-specific examples, and job-backed execution
paths, and seeded PDF equation/table/figure/algorithm structure gates exposed through
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
status owner-count/ownership-source summaries, with default `local-user` / `Local user` values when
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
Artifact downloads also reject symlink artifact paths before `FileResponse`
export, even when the symlink target stays inside the artifact root.
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
client IPs, prompts, or answers. Selectable PDF discovery and active-paper
persistence now skip symlink PDFs, and upload filename conflict handling treats
symlinks as occupied paths before creating a new upload file. This is a local abuse guardrail, not a
production antivirus, sandbox-scanning, identity-backed quota, or data deletion
system.
The latest retention-delete slice adds a guarded local deletion path behind
`RETENTION_DELETE_ENABLED`, defaulting to disabled. `GET /admin/retention`
remains the preview path, while authenticated `POST /admin/retention/delete`
can delete the same bounded local upload/artifact candidate set only when the
flag is explicitly enabled. The delete path excludes artifact SQLite metadata
files and symlinks, rechecks candidates as regular files before unlinking,
records aggregate-only `retention_delete` runtime events, and the Streamlit admin
panel only shows the delete action when the same flag is on.
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
presence across corpus, chunk, job, artifact, runtime-event, and optional local
API-key registry stores without returning row contents, prompts, answers,
filenames, owner IDs, request IDs, token hashes, source paths, or runtime file
contents. `scripts/storage_schema.py` exposes the same check as a local or
target-root CLI preflight with JSON/Markdown output and a nonzero exit code on
drift. The current local CLI/admin snapshot reports `ok=true`, 10 stores, and 0
schema problems; `api_key_registry_sqlite`, `product_registry_sqlite`, and
`share_link_registry_sqlite` are allowed to be absent when their backends are
left at the production-safe default `none`.

The 2026-06-13 pass also refreshed the agent-facing bootstrap docs `CLAUDE.md`
and `AGENTS.md` so they describe the current dual-entrypoint (Streamlit + FastAPI)
runtime, the no-key job/storage/artifact subsystem, the common build/test/run
commands, and the documentation discipline, instead of the earlier
Streamlit-only summary. These are documentation-only changes and keep the
docs-guard tests green.

The same 2026-06-13 pass also added a CI-safe slice of the previously planned
local eval breadth: a second Python execution template `pmsm_current_step`
(PMSM q-axis current step response producing CSV/SVG), a second job-backed Python
code-output eval case using it (4 code-output cases total, 2 job-backed), a
second Octave template `smc_sign_switching`, and unit tests for the new Python
template and Octave template breadth. The 2026-06-17 follow-up `95f1760` adds
the first Octave-compatible code-output eval case for `pmsm_current_decay`: it
runs the real template when `octave` is installed and otherwise accepts only the
structured runtime-unavailable diagnostic. Broader real Octave execution coverage
still depends on installing an `octave` binary in CI/runtime, and a real-PDF
algorithm-caption acceptance case still needs a curated library paper with a
numbered `Algorithm N` block.

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

The latest live deployment snapshot was refreshed after syncing `45e4cc6` and
`517756f` with restart, then running SSH, public HTTPS, product-readiness,
product-registry, product-quota/RBAC-guard/product-registry-management/object-manifest
and object-manifest-verifier anchors, and live retrieval checks on
2026-06-17 02:14 CST in
`docs/DEPLOYMENT_STATUS.md`. The platform/eval/API/runtime-restore/
job-idempotency/retry-dead-letter/ownership/API-key-registry/product-registry/product-quota-guard/product-RBAC-guard/Docker-execution/execution-policy/
execution-observability/output-limits/artifact-limits/execution-alerts/
query-latency/query-alerts/provider-alerts/job-alerts/API-access-audit/
API-rate-limit/upload-scan/retention-delete/metrics-export/retrieval-trace/
retrieval-alerts/storage-schema work above plus the API startup readiness,
import-latency fixes, coverage gate, 30-paper seed corpus, API token comparison
hardening, runtime JSON/JSONL state-file tolerance, and `.coverage` deploy
exclude and the distributed job-store readiness foundation are now synced or
rebuilt on Trace-Twin through the current source/runtime boundary. The latest
remote corpus status is `papers=30`,
`active=30`, `indexed=30`, `chunks=1934`, and `index=fresh`; live retrieval eval
passes `107/107`, with `small_group=met`. External
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
Active papers       30
FAISS index bytes   2970669
Chunk rows          1934 across 30 source paths
Index freshness     True
Storage readiness   local metadata/object storage available
Storage schema      ok=true; store_count=9; problems=0; api_key_registry_sqlite ok=true;
                    product_registry_sqlite ok=true
API key registry    backend=none; available=false; active_keys=0; secrets_exported=false
Product registry    backend=none; available=false; workspaces=0; secrets_exported=false
Product registry management installed=true; route ok; backend=none by default
Product quota guard installed=true; enabled=false by default; admin metric present
Product RBAC guard  installed=true; enabled=false by default; admin metric present
Job-store readiness local job store available; external job store configured false
Migration preflight preflight_ok=true; activation_ready=false; local_blockers=none;
                    activation blockers are the expected external metadata DB,
                    object storage, and distributed job-store targets
Migration rehearsal rehearsal_ok=true; copied_files=19; restore_check_ok=true;
                    staged_storage_schema_ok=true; blockers=none
Object manifest     rehearsal_ok=true; objects=19; unique=18;
                    source_paths/filenames/bucket/secrets exported=false
Object verify       ok=true; checked=19; missing=0; mismatched=0; extra=0;
                    source_paths/filenames/bucket/secrets exported=false
Docker execution    configured=False available=False reason=not_configured
Disk                /dev/vda3 40G total, 24G free, 36% used
```

## Immediate Boundary

- Runtime state remains git-ignored: `papers/`, `faiss_index/`, `artifacts/`,
  `jobs/`, `metadata/`, `.env`, virtual environments, caches, and bytecode.
- The retrieval-eval/code-output/PDF-structure/report/runtime-restore/
  job-idempotency/retry-dead-letter/ownership/API-key-registry/product-registry/product-quota-guard/product-RBAC-guard/Docker-execution/
  execution-policy/execution-observability/output-limits/artifact-limits/
  execution-alerts/query-latency/query-alerts/provider-alerts/job-alerts/
  API-access-audit/API-rate-limit/upload-scan/retention-delete/
  metrics-export/retrieval-trace/retrieval-alerts/storage-schema/API work plus
  object-manifest/object-manifest-verifier/eval-breadth,
  coverage/corpus-hardening, runtime-state-hardening, and deploy-exclude slices
  are verified, committed, pushed to `origin/main` through application baseline
  `45e4cc6` plus docs/health sync `517756f`, deployed to Trace-Twin, and
  post-restart verified. The deployed API-key and product registry backends
  plus product quota/RBAC guards remain disabled by default;
  enabling the SQLite registries and the query quota/RBAC guards is an explicit
  operational choice, not an automatic production activation.
- Deployment facts should not be inferred from git state alone because
  `/opt/fluxmind` is a synchronized source tree, not a git checkout.
