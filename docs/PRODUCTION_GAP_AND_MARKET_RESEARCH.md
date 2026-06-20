# FluxMind Production Gap and Market Research

Last updated: 2026-06-21 00:27 CST

This document answers: what should FluxMind build next, what is still missing
before it can be treated as a production-grade product, and what external
competitor/community signals support the plan.

It separates four evidence layers:

```text
Layer                 Current source
--------------------  -------------------------------------------------------
Current repo state    git status/log plus docs/REPO_STATUS.md
Current live state    health_check.py HTTPS/SSH checks plus live retrieval eval
                      refreshed on 2026-06-17 02:14 CST; re-run before deploy claims
Repo snapshots        docs/ARCHITECTURE.md, docs/BACKLOG.md,
                      docs/PLATFORM_AUDIT_AND_ROADMAP.md, docs/FEATURE_AUDIT.md
External research     Public project docs, GitHub API, community/forum search
                      (GitHub counts refreshed 2026-06-15; forum themes remain
                      the 2026-06-08 qualitative sample)
```

External links and GitHub counts are time-sensitive. Re-run public checks before
using this document for investment, deployment, or release decisions.
The 2026-06-21 00:27 refresh updates the current local repo verification counts
and the readiness CLI data-error sanitization boundary. It retains the deploy
sync CLI error-output sanitization boundary, share-link create error-detail
sanitization boundary, Streamlit corpus profile report and job result
failure-message sanitization boundaries, job detail API request-ID,
owner-metadata, and idempotency-key projection boundaries, job detail API
code-output projection boundary, Streamlit validation error-output
sanitization boundary, API request validation error projection boundary, index
rebuild job API projection redaction boundary, API validation/artifact download
error-output redaction boundary, Streamlit admin/artifact error-output
redaction boundary, git/documentation drift evidence after the job request-ID
docs sync, and no-secret registry/readiness default state only. Dated live
deployment and external research snapshots remain scoped to the times shown in
their rows.

## Current Baseline

Documentation-pass state:

```text
Branch          main
Start state     deployed source/eval baseline d80c083 before quality expansion
Source/eval     e069873 test: complete FluxMind small-group quality baseline
Calibration     cc705dc test: recalibrate FluxMind live retrieval expectation
Gate hardening  d80c083 test: tighten FluxMind small-group quality gates
Current source/eval  9b1cbc5 test: expand FluxMind community quality eval
Current implementation 41ca43f fix: sanitize readiness cli data errors
Current docs/health    docs: record readiness cli data error audit status
Status note     local no-key hardening through index rebuild job API projection,
                API request validation projection, API validation/artifact
                download, Streamlit admin/artifact/upload validation
                error-output sanitization, and job detail code-output
                plus idempotency-key, owner-metadata, request-ID projection,
                Streamlit failed-job message sanitization, and Streamlit corpus
                profile report failure-message sanitization, and share-link
                create error-detail sanitization, and deploy sync CLI
                error-output sanitization, and readiness CLI JSON/ValueError
                data-error sanitization is implemented and locally verified;
                the 00:27
                refresh reconfirms no OpenAPI/storage/docs drift; the
                local stack is not pushed to origin and not deployed to
                Trace-Twin
Deployed source/eval  9b1cbc5 test: expand FluxMind community quality eval
Work scope      local no-secret hardening, docs/status refresh, and drift gates;
                external providers, identity, billing, and distributed storage
                remain disabled by default
Diff hygiene    git diff --check passed for the current checkout on 2026-06-21 00:27 CST
```

Current local verification from this docs/status pass plus retained readiness
and deployment snapshots:

```text
Check                                      Result
----------------------------------------  -------------------------------------
pytest                                    640 passed, 2 known warnings
coverage                                  89% total branch coverage over api,
                                          scripts, and src
offline RAG eval                          42 answer cases, 65 retrieval-only
                                          cases, 13 code-output cases,
                                          30 PDF structure cases,
                                          42 recorded answers
local health_check.py                     pass, local/docs/query-latency/query-alert/
                                          provider-alert/job-alert/API-access-audit/
                                          API-rate-limit/upload-scan/retention-delete/
                                          metrics-export/retrieval-trace/
                                          retrieval-alerts/storage-schema/
                                          artifact-limit/execution-alert/
                                          provider-readiness/provider-runtime-
                                          rehearsal/quality-readiness/
                                          API-key-registry/product-registry/
                                          share-link-registry/product-quota/
                                          product-RBAC/product-registry-management/
                                          share-link-management/product-activation-
                                          rehearsal/object-storage-manifest/
                                          object-storage-manifest-verifier/
                                          job-store-manifest/job-store-manifest-
                                          verifier/OpenAPI-contract/execution-input-
                                          materialization/runtime-event-metadata-
                                          value-redaction/readiness-CLI-error-
                                          sanitizer/live-eval-request-ID-redaction,
                                          Streamlit-share-link-error-sanitizer,
                                          and Streamlit-product-registry-error-
                                          sanitizer/admin-on-demand-error-
                                          sanitizer/artifact-gallery-error-
                                          sanitizer/API-validation-error-
                                          sanitizer/request-validation-error-
                                          projection/Streamlit-validation-
                                          error-sanitizer/job-detail-code-output-
                                          projection/job-idempotency-key-
                                          projection/job-owner-metadata-
                                          projection/job-request-id-projection/
                                          Streamlit-job-result-error-sanitizer/
                                          Streamlit-corpus-profile-report-error-
                                          sanitizer/API-share-link-create-error-
                                          sanitizer/deploy-sync-error-sanitizer/
                                          readiness-CLI-data-error-sanitizer
                                          anchors
storage_schema.py                         pass, ok=true, 10 stores, 0 problems
OpenAPI no-secret snapshot verify         pass, ok=true, diff_count=0 against
                                          the just-exported local contract
api_key_registry.py status                pass, backend=none, available=false,
                                          active_keys=0, secrets_exported=false
product_registry.py status                pass, backend=none, available=false,
                                          workspaces=0, secrets_exported=false
share_link_registry.py status             pass, backend=none, available=false,
                                          active_links=0, secrets_exported=false
runtime restore dry-run                   pass, ok=true, 6 groups, 5 checked files, manifest_errors=0 against exported local manifest
object manifest smoke                     pass, rehearsal_ok=true,
                                          object_manifest_ready=true,
                                          objects=9, unique=8,
                                          paths/filenames/bucket/secrets
                                          exported=false
object manifest verify                    pass, ok=true, checked=9,
                                          missing=0, mismatched=0, extra=0,
                                          paths/filenames/bucket/secrets
                                          exported=false
job-store manifest smoke                  pass, rehearsal_ok=true,
                                          job_store_manifest_ready=true,
                                          jobs=0, claims=0 in current checkout;
                                          payloads/raw IDs/secrets exported=false
job-store manifest verify                 pass, ok=true, missing=0,
                                          mismatched=0, extra=0 jobs or claims;
                                          payloads/raw IDs/secrets exported=false
product_readiness.py                      pass, local_foundation_ready=true,
                                          activation_ready=false with expected
                                          identity/quota/billing blockers and
                                          product-quota/RBAC-guard disabled advisories
product_activation_rehearsal.py           pass, ok=true, activation_ready=true
                                          against disposable local SQLite stores;
                                          raw tokens and paths exported=false
provider_readiness.py                     pass, local_foundation_ready=true,
                                          activation_ready=false with expected
                                          provider/MATLAB blockers
provider_runtime_rehearsal.py             pass, ok=true, local mock image/Python
                                          execution/Octave branch checked plus
                                          allowed/blocked provider-guard
                                          decisions;
                                          external_activation_ready=false
quality_readiness.py                      pass, local_foundation_ready=true,
                                          community_ready=false with measured
                                          corpus/eval/live-evidence gaps and
                                          target gap summary
HTTPS UI                                  02:14 snapshot: https://smy.hyper-dusty.cloud/ 200
HTTPS API health                          02:14 snapshot: https://api-smy.hyper-dusty.cloud/health 200
SSH health                                02:14 snapshot: pass on root@100.100.233.26
Remote services                           UI/API/worker/cloudflared/docker active
Remote listeners                          0.0.0.0:18501 and 0.0.0.0:18502
Remote model                              LLM_MODEL=mimo-v2.5-pro
Remote active corpus                       active_papers=30
Remote chunk metadata                     chunk_metadata_rows=1934, sources=30
Remote index freshness                     index_fresh=True
Remote API-key registry                    backend=none, available=false; local SQLite
                                          registry implemented but not activated
Remote product registry                    backend=none, available=false; local SQLite
                                          user/workspace/quota/usage/billing ledger
                                          implemented but not activated
Remote product quota guard                 installed in API/product-registry/admin metric,
                                          but enabled=false by default
Remote product RBAC guard                  installed in API/product-registry/admin metric/CLI,
                                          but enabled=false by default
Remote product registry management         installed in API/UI/product-registry summaries;
                                          /admin/product-registry/status route ok,
                                          backend=none by default
Remote object manifest                     installed in storage_migration/rehearsal CLI;
                                          live smoke objects=19, unique=18,
                                          source_paths/filenames/bucket/secrets
                                          exported=false
Remote object manifest verify              installed in storage_migration/rehearsal CLI;
                                          live smoke ok=true, checked=19,
                                          missing=0, mismatched=0, extra=0,
                                          source_paths/filenames/bucket/secrets
                                          exported=false
Remote live retrieval eval                 02:14 snapshot: 107/107 passed; quality
                                          readiness small_group=true with live report,
                                          community=false with measured gaps
Remote execution sandbox                   Local Docker backend implemented; live Docker execution not configured
```

Interpretation: FluxMind is a healthy deployed no-key/local baseline and a good
single-operator research demo. It is not yet a production-grade multi-user
product. The remaining gap is mostly platform, safety, content operations, and
domain workflow depth, not "can the RAG prototype answer a seeded question".

## Executive Direction

FluxMind should not compete head-on as a generic RAG builder. Dify, Open WebUI,
Langflow, RAGFlow, AnythingLLM, Flowise, Haystack, and LlamaIndex already cover
large parts of generic app building, agent workflows, document chat, ingestion,
and framework plumbing.

The defensible position is narrower and stronger:

> A control-engineering research workspace that turns papers into traceable
> answers, executable modeling snippets, reproducible plots, and implementation
> notes for sliding-mode control, PMSM/FOC, flux estimation, observers, and
> adjacent control workflows.

The product wedge should be "paper-grounded engineering work product", not just
"chat with PDFs".

## Production Gap Matrix

```text
Area                    Current FluxMind state
----------------------  ------------------------------------------------------
RAG core                Local FAISS, hybrid retrieval, citations, inspection,
                        deterministic eval fixtures, deployed API/UI.

Gap to production       Broader live QA set, citation verifier, equation/table
                        parsing, reranker evaluation, domain benchmark tasks,
                        hallucination/error policy, versioned corpora.

Priority                High. This is the trust layer users will judge first.
```

```text
Area                    Current FluxMind state
----------------------  ------------------------------------------------------
Content/corpus          30-paper curated seed library, local metadata, profiles, chunk mirror,
                        DOI/arXiv enrichment fields, upload/index flow.

Gap to production       Curated domain corpus, topic ontology, paper quality
                        rules, benchmark question set, equations/figures/code
                        templates, corpus provenance and refresh process.

Priority                Highest. A domain product wins through domain content,
                        not through having another generic document chatbot.
```

```text
Area                    Current FluxMind state
----------------------  ------------------------------------------------------
Storage                 Local JSON/SQLite/filesystem and FAISS. Good current
                        state visibility, no-secret manifest, and dry-run
                        restore verification through CLI/API/UI. Local
                        storage-schema readiness checks cover JSON, JSONL, and
                        SQLite stores without exposing contents and now have a
                        CLI preflight. Admin/report/metrics/UI surfaces now
                        expose production storage and distributed-worker
                        readiness blockers without connecting to external
                        services. Jobs and generated artifacts now carry local
                        owner metadata. Product-readiness checks now expose the
                        local identity/RBAC/quota/billing foundation, including the
                        hash-only local API-key registry plus a local product
                        registry for users, workspaces, role permissions, quota
                        limits, usage, and billing attribution. `/query*` routes
                        can enforce local request quotas when the product quota
                        guard is explicitly enabled, and query/job/corpus/admin
                        write paths can enforce local workspace roles when the
                        RBAC guard is explicitly enabled, with activation
                        blockers still reporting external systems. Local
                        operator management for workspace/member/quota/
                        billing-attribution metadata is available through
                        `/admin/product-registry/*` and the Streamlit admin
                        panel when the SQLite backend is enabled. Local
                        migration rehearsal can now emit an opaque
                        object-storage manifest with deterministic object keys,
                        hashes, byte counts, group names, and path tokens
                        without exposing source paths, filenames, buckets,
                        endpoints, credentials, or contents.

Gap to production       Relational metadata store, object storage, vector DB or
                        managed vector index, identity-backed ownership,
                        migrations, backups, full restore drills, retention
                        policies.

Priority                High. Identity, quotas, billing, and durable workers all
                        depend on this boundary; the product-readiness preflight
                        is now the gate before real activation.
```

```text
Area                    Current FluxMind state
----------------------  ------------------------------------------------------
Jobs/workers            Local durable worker service, SQLite/JSONL state,
                        idempotency, bounded retries, dead-letter status,
                        leases, cancellation metadata, local owner metadata,
                        and no-secret readiness blockers for distributed
                        worker acceptance.

Gap to production       Distributed queue, concurrent workers, per-tenant quotas,
                        durable cancellation, distributed idempotency,
                        autoscaling, SLO/error budgets.

Priority                High after storage. RAG indexing, execution, and artifact
                        generation all need real job control.
```

```text
Area                    Current FluxMind state
----------------------  ------------------------------------------------------
Execution               Local Python/Octave-compatible providers, request-level
                        execution policy, and the opt-in Docker backend prove
                        the API and artifact contract. No-secret code-execution
                        events/admin summaries expose backend/status/policy
                        outcomes without copying source. Captured stdout/stderr
                        and generated-artifact export are byte/count-bounded
                        with truncation metadata. Admin status/report now expose
                        local advisory alerts for execution failure rate,
                        slow duration, policy violations, and output/artifact
                        truncation. Docker execution is explicitly not configured
                        on live deployment. Provider-readiness now exposes
                        hosted execution and MATLAB backend activation blockers
                        without enabling those backends.

Gap to production       Live sandbox enablement, deeper malware/abuse controls,
                        production metrics/traces/alerts for execution,
                        MATLAB/Simulink license path or explicit Octave-only
                        positioning.

Priority                High if "paper to simulation" is a core feature. Do not
                        enable arbitrary user code in the main API process.
```

```text
Area                    Current FluxMind state
----------------------  ------------------------------------------------------
Product shell           Streamlit UI, FastAPI token boundary, public deployed
                        UI/API, admin/status/report panels with local owner
                        summaries.

Gap to production       External user accounts, production team/RBAC
                        administration, external identity-backed API-key
                        lifecycle, external billing/payment, share/export flows,
                        real frontend, onboarding, team/lab workflows.

Priority                Medium-high. Build after storage/job ownership is clear.
```

```text
Area                    Current FluxMind state
----------------------  ------------------------------------------------------
Observability           No-secret admin status, runtime events, provider failure
                        history, query usage/duration estimates, local
                        job-health, query-latency, provider-failure, and
                        code-execution advisory alerts, metadata-only API
                        access audit summaries, local API rate-limit status,
                        metadata-only retrieval trace summaries/alerts,
                        no-secret local metrics text export, provider-readiness
                        blocker summaries, health checks.

Gap to production       Production traces across retrieval/LLM/jobs, latency
                        SLOs, alerting, cost attribution per workspace,
                        prompt/version tracking, eval dashboards, incident
                        runbooks, and production metrics scrape/retention/alert
                        routing beyond the local export.

Priority                Medium-high. Needed before paid providers and public use.
```

```text
Area                    Current FluxMind state
----------------------  ------------------------------------------------------
Security/compliance     Simple API token, runtime excludes, no-secret manifests,
                        metadata-only API access audit events, configurable
                        local API rate-limit guard, safe request-ID cleaning,
                        local metadata-only upload scan guard, guarded local
                        retention-delete switch, public UI intentionally open
                        in current deployment.

Gap to production       Authentication, production antivirus/sandbox upload
                        scanning, distributed or identity-backed rate limits,
                        abuse protection, secret management, identity-backed
                        audit logs and data deletion, backups/restore,
                        terms/privacy if public.

Priority                High before exposing private corpora or execution.
```

## Competitor Map

GitHub API snapshot collected on 2026-06-15 08:32 CST. Counts are directional, not a
quality ranking.

```text
Project                         Stars   Forks  Open issues  License       Pushed
------------------------------  ------  -----  -----------  ------------  --------------------
langgenius/dify                 145203  22845  753          NOASSERTION   2026-06-14T14:35:41Z
open-webui/open-webui           141519  20336  469          NOASSERTION   2026-06-13T13:19:10Z
langflow-ai/langflow            149666  9267   950          MIT           2026-06-14T01:06:51Z
infiniflow/ragflow              82718   9544   3280         Apache-2.0    2026-06-13T13:37:31Z
Mintplex-Labs/anything-llm      61586   6705   338          MIT           2026-06-14T23:22:48Z
FlowiseAI/Flowise               53573   24511  922          NOASSERTION   2026-06-10T07:32:39Z
run-llama/llama_index           50123   7561   454          MIT           2026-06-12T19:29:18Z
deepset-ai/haystack             25567   2853   137          Apache-2.0    2026-06-12T15:45:23Z
Future-House/paper-qa           8699    882    137          Apache-2.0    2026-06-11T18:43:29Z
simplefoc/Arduino-FOC           2865    710    77           MIT           2026-06-11T08:25:34Z
python-control/python-control   2031    457    103          BSD-3-Clause  2026-06-09T19:53:03Z
do-mpc/do-mpc                   1406    219    93           LGPL-3.0      2025-10-31T09:44:17Z
OpenModelica/OpenModelica       1320    379    2240         NOASSERTION   2026-06-12T21:57:05Z
JuliaControl/ControlSystems.jl  578     91     44           NOASSERTION   2026-06-09T04:05:22Z
```

### Generic RAG and Agent Platforms

These projects define the minimum platform bar:

```text
Project        What it pressures FluxMind to match or avoid
-------------  ---------------------------------------------------------------
Dify           Workflow builder, app templates, datasets/knowledge, hosted and
               self-host app lifecycle. FluxMind should not clone this broadly.

RAGFlow        Document understanding emphasis: layout, tables, OCR, retrieval
               quality. This is directly relevant to engineering PDFs.

AnythingLLM    Workspace/document chat and agent features for local/self-host use.
               FluxMind needs a narrower domain reason to exist.

Open WebUI     Mature local/self-host chat UI with knowledge/document features.
               FluxMind should not spend too long polishing Streamlit as a chat UI.

Langflow       Visual agent/RAG flow design. Competes on low-code composition.
               FluxMind should instead package domain workflows.

Flowise        Low-code LLM app/chatflow builder. Similar lesson as Langflow.

Haystack       Production RAG/agent framework and pipeline concepts. Useful
               reference for pipeline decomposition and evaluation discipline.

LlamaIndex     Data connectors, retrievers, agents, eval/observability ecosystem.
               Useful reference for ingestion/retrieval abstractions.
```

Implication: FluxMind's roadmap should not be "add every agent-builder feature".
It should adopt production-grade platform primitives from this ecosystem while
staying domain-specific in content, evals, execution templates, and artifacts.

### Research Assistant Products

```text
Product/source     Signal
-----------------  ------------------------------------------------------------
PaperQA            Open-source agentic scientific-paper QA with evidence/citation
                   orientation. Strong reference for source-grounded literature QA.

Elicit             Proprietary research workflow product focused on literature
                   search, extraction, and systematic-review style tables.

NotebookLM         Proprietary source-grounded notebook/chat experience. Sets
                   user expectations for citations and uploaded-source handling.
```

Implication: source-grounded answers are becoming table stakes. FluxMind needs
engineering-specific output: equations, block diagrams, simulation-ready code,
plots, assumptions, parameter traces, and "what changed from paper to code".

### Control and Engineering Ecosystem

```text
Project/community       Signal
----------------------  ------------------------------------------------------
python-control          Control-system modeling and design in Python. Good
                        target for generated examples and validation snippets.

do-mpc                  Nonlinear model predictive control in Python. Useful
                        adjacent workflow once FluxMind expands beyond SMC/FOC.

OpenModelica            Open-source modeling/simulation environment. Relevant
                        for Modelica-style physical system models.

ControlSystems.jl       Julia control systems ecosystem. Useful for advanced
                        users, but not the first user-facing target.

SimpleFOC               Open-source FOC library and active motor-control forum.
                        Strong demand signal around practical implementation.

MATLAB/Simulink         Not open-source, but still the dominant educational and
                        engineering reference environment for many control users.
```

Implication: the first execution lane should support Python and Octave well,
then decide whether MATLAB/Simulink is a licensed backend, an export target, or
explicitly out of scope.

## Community Demand Signals

The community/forum search points to recurring needs. These are not statistically
complete market research; they are practical signals for product direction.

```text
Community/source       Recurring signal
---------------------  -------------------------------------------------------
MathWorks Answers      Users ask how to implement sliding mode control,
                       field-oriented control, PMSM models, flux linkage, and
                       observers in MATLAB/Simulink.

TI E2E / ST Community  Motor-control users hit implementation details around
                       PMSM/FOC, observers, flux parameters, sensorless control,
                       workbench configuration, and hardware behavior.

SimpleFOC Community    Practical tuning questions around FOC, current control,
                       motor parameters, loops, sensors, and unstable behavior.

r/ControlTheory        Learning and implementation questions around MATLAB,
                       Simulink, Python, papers, simulation, and real projects.

GitHub issues          Open-source RAG platforms show recurring pressure around
                       PDF ingestion, retrieval quality, citations, deployment,
                       and document parsing edge cases.
```

Demand synthesis:

1. Users do not only need definitions; they need translation from paper concepts
   to runnable models and implementation decisions.
2. PMSM/FOC/observer workflows are parameter-sensitive. Users need assumptions,
   units, motor parameters, sample time, discretization, and failure-mode notes.
3. MATLAB/Simulink remains a common target, but Python/Octave examples are useful
   for no-key/local reproducibility.
4. Engineering PDFs contain figures, equations, tables, and algorithm blocks.
   Plain chunked text RAG is not enough for high-trust answers.
5. Trust depends on traceability: source paper, page, equation, parameter, code
   snippet, generated plot, and execution log should be connected.

Product requirements implied by the forum/community search:

```text
Need observed                               FluxMind feature implication
------------------------------------------  ----------------------------------
Paper concept to runnable implementation    Add paper-to-code reports with
                                            assumptions, equations, code,
                                            outputs, plots, and source refs.

Motor-control debugging depends on context  Add intake templates for motor
                                            parameters, units, sampling/PWM,
                                            sensor mode, gains, and logs.

Observer/FOC questions are failure-prone     Add failure-mode cards for low
                                            speed, parameter mismatch,
                                            chattering, saturation, and noise.

MATLAB/Simulink remains common              Keep Octave/Python runnable paths
                                            no-key, but write MATLAB/Simulink
                                            export notes before licensing work.

PDF layout matters in engineering papers     Prioritize equations, tables,
                                            figures, algorithm blocks, and
                                            source/page/equation provenance.
```

## Content Roadmap

The fastest way to make FluxMind feel valuable is a focused content/eval build,
not a broad SaaS shell first.

### Content Themes

```text
Theme                         Example assets to collect or create
----------------------------- ------------------------------------------------
Sliding-mode control basics   Reaching laws, boundary layers, chattering,
                              Lyapunov proofs, discretization caveats.

PMSM and FOC                  dq model, flux linkage, current loop, speed loop,
                              PWM/sample-time assumptions, saturation.

Observers and estimation      Sliding-mode observer, back-EMF observer, EKF/UKF
                              adjacency, low-speed behavior, sensorless limits.

Implementation templates      Python-control snippets, Octave scripts, Simulink
                              export notes, parameter tables, validation plots.

Failure modes                 Chattering, divergence, low-speed observability,
                              parameter mismatch, noisy derivatives, tuning traps.

Paper-to-code bridges         Equation extraction, assumptions, pseudocode,
                              code scaffold, runnable simulation, plot outputs.
```

### Evaluation Assets

```text
Eval lane              Target
---------------------  -------------------------------------------------------
Retrieval              At least 50 domain questions with expected source/page
                       refs across SMC, PMSM/FOC, observers, flux estimation.

Answer quality         Recorded answers with required terms, citations, caveats,
                       and "unsafe/unsupported claim" checks.

Equation fidelity      Cases where equations or parameter tables must be
                       extracted and preserved accurately.

Code generation        Python/Octave tasks that must execute, produce plots, and
                       cite the paper assumptions used.

Forum-style tasks      Recreate common community questions: tuning, model setup,
                       low-speed sensorless issues, parameter mismatch.
```

### Corpus Growth

Recommended near-term corpus target:

```text
Milestone       Corpus target
--------------  --------------------------------------------------------------
M0 current      30 bundled seed papers; deployed active index is fresh for the
                latest 30-paper runtime rebuild snapshot
M1              50 curated papers, tagged by topic and method
M2              100+ curated papers plus benchmark questions and code templates
M3              Add forum-style implementation notes and failure-mode cards
```

Keep corpus growth curated. A larger undifferentiated PDF pile would make the
product look more generic and less trustworthy.

## Technical Roadmap

Order matters. The recommended sequence is:

```text
Order  Lane                                      Why first
-----  ----------------------------------------  --------------------------------
1      Content/eval expansion                    Creates differentiated value
2      Production storage boundary               Enables ownership and recovery
3      Durable jobs/distributed worker control    Enables indexing/execution scale
4      Observability and cost attribution         Needed before paid providers
5      Isolated execution sandbox                 Needed before public code runs
6      Identity/workspaces/RBAC/quotas            Local registries/guards exist; external
                                                  identity/payment needs storage
                                                  and ownership decisions
7      Frontend/API split                         Avoids overbuilding Streamlit
8      Provider activation/billing                Only after trust and control
```

### 0-30 Days: Domain Trust Sprint

- Expand the curated corpus from 30 papers toward 50 high-quality papers.
- Add topic tags and a control-engineering ontology: SMC, FOC, PMSM, SMO,
  observers, flux estimation, chattering, discretization, parameter tuning.
- Create live answer eval cases with passing live answer pass-rate and
  term-coverage evidence, and broaden recorded/retrieval cases toward the
  community bar.
- Add equation/table/figure/algorithm extraction acceptance tests for representative PDFs.
- Add code-output evals where Python/Octave examples must run and produce plots.
- Add a "paper-to-code report" export: source refs, assumptions, parameters,
  generated code, execution output, plot artifacts.

Current progress on 2026-06-16: the no-key baseline has advanced from 5 to 42
offline/recorded answer cases and 65 retrieval-only cases, for 107 total no-LLM
retrieval questions. The baseline gates 145 source/page refs and 111 topic tags
across retrieval, answer quality, equation fidelity, code generation,
forum-style debugging, failure modes, and paper-to-code reports, and includes
13 local code-output gates that verify expected stdout plus plot/text artifacts
in a temporary artifact store, including reusable execution-template coverage,
store-level symlink write/source guards, paper-specific examples, four local
Python job-backed execution paths, and an Octave-compatible PMSM current-decay
case with structured runtime-unavailable fallback when no `octave` binary is
installed.
The evaluator also has 30 seeded PDF structure gates for
equation/table/figure/algorithm markers on representative
source pages, and `GET /corpus/structure/report` exports filtered structure
anchors as a Markdown handoff report. `POST /query/report` now adds a local
paper-to-code handoff for implementation and code-generation reports, including
source refs, assumption/parameter guardrails, fenced code blocks, cited artifact
IDs, and validation checklist fields. The self-use and small-group targets are
met in the latest deployed live retrieval report; the community target still
needs corpus growth toward 50 papers, 80 recorded answers, 180 retrieval
questions, and live answer count/pass-rate/term-coverage evidence. The 30-case
PDF structure target is now met. Broader Octave
execution remains deferred until an Octave binary
is available in CI/runtime. The 2026-06-15 and 2026-06-16 seed-library
expansions add adaptive-gain SMO, super-twisting SMO, switching-function
comparison, adaptive-parameter IPMSM, MRAS flux-linkage observer, combined
reaching-law SMO, PID/ITSMRL/ESO speed control, fuzzy super-twisted SMO,
super-twisting SMC with ESO feedback, adaptive quick reaching law with SFTSMO,
DSMO/LQR current-loop control, NFTSMC with SDOB validation, integrated
SMC/DOB/LPF, fast terminal SMPC, model-free GSTA/FTSMC, fractional
super-twisting SMDO, IFTSM adaptive control, CESO composite SMC, prescribed
performance LESO, ISMO antidisturbance control, and SMO/sensorless-control
survey coverage.

Success criterion: a skeptical control student or engineer can ask paper-backed
implementation questions and receive traceable, executable outputs.

### 30-60 Days: Production State Foundation

- Choose production storage: PostgreSQL metadata plus object storage, and either
  pgvector/Qdrant/Milvus for vector state or a deliberately local-only vector
  boundary with explicit backup/restore.
- Migrate corpus, chunks, jobs, artifacts, query events, and eval reports behind
  versioned schemas.
- Extend the current no-secret manifest and dry-run verifier into migration
  tests, opaque object-storage manifests, backup/restore drills, retention
  policy, and restore docs.
- Use the current `platform_readiness` blocker summary as the local acceptance
  gate for the storage/queue backend choice; it already confirms clean local
  schema/inventory and local worker-bridge contracts while flagging missing
  external metadata database, object storage, and the separately configured
  distributed job-store target.
- Promote the current local API ownership metadata into versioned production
  schemas once storage/backend ownership is selected.
- Extend local idempotent job submission plus bounded dead-letter/retry policy
  into the future distributed worker queue after the job-store backend is chosen
  and migration-tested.

Success criterion: runtime state can be backed up, restored, migrated, and
owned without reading local JSON/SQLite files by hand.

### 60-90 Days: Execution and Observability

- Decide whether the implemented local Docker backend is sufficient for the next
  release or whether production use needs gVisor/Firecracker-style runtime,
  Cloudflare Sandbox, or another hosted sandbox. Keep a clear "not configured"
  live state until real isolation is verified.
- Extend the local policy and output-limit layer into live sandbox evidence,
  richer package/filesystem controls, and abuse-oriented tests for code
  execution.
- Extend the local retrieval trace and advisory alert baselines into broader
  production traces, metrics, and alerting for reranking, answer generation,
  jobs, artifacts, code execution, and provider failures.
- Promote the local no-secret `/admin/metrics` export into a real metrics
  pipeline only after scrape target, retention, alert routing, and access
  controls are decided.
- Add prompt/version/eval dashboards and per-workspace cost attribution.
- Decide MATLAB path: Octave-compatible only, MATLAB export only, or licensed
  MATLAB backend with isolated execution.
- Keep `scripts/provider_readiness.py --require-activation` failing until the
  selected sandbox, MATLAB path, external provider switch, and quota/cost guard
  are all configured and verified.
- Keep `scripts/quality_readiness.py --require-target community` failing until
  the community-quality corpus, answer/retrieval breadth, and live-answer
  count/pass-rate/term-coverage gaps are closed. Use its target gap summary to
  plan the next corpus, retrieval, recorded-answer, and live-answer additions
  from executable metric deltas.

Success criterion: generated code can run in a controlled environment, and every
expensive or risky operation is observable and attributable.

### 90-120 Days: Product Shell

- Introduce external users, corpora ownership, identity-backed API keys,
  production quotas, and team admin roles.
- Replace or wrap Streamlit with a frontend that supports workspace navigation,
  artifact galleries, job timelines, citation inspection, and exports.
- Add onboarding flows for: upload paper, ask paper-grounded question, generate
  simulation, inspect plot, export report.
- Add share/export controls for lab/team use.

Success criterion: FluxMind can support at least a small private lab/team
without mixing private corpora, jobs, artifacts, or costs.

### 120+ Days: Provider and Business Activation

- Enable paid LLM/image/sandbox providers only after observability, quotas, and
  spend attribution are implemented, and only after provider-readiness
  activation passes.
- Add pricing model if public: per-seat, per-lab, or self-host support. Avoid
  usage-based billing until cost attribution is reliable.
- Add connector/export integrations only where they support the domain wedge:
  Zotero, arXiv, MATLAB/Simulink export, Python notebooks, markdown/LaTeX report.

Success criterion: the product can absorb real provider failures and costs
without corrupting user trust or runtime state.

## Product Positioning

Recommended positioning:

```text
Not this         "A generic RAG/agent app builder for any documents"
Instead         "A control-engineering research copilot for paper-grounded
                 modeling, simulation, and implementation artifacts"

Not this         "Ask PDFs questions"
Instead         "Turn a paper section into assumptions, equations, code,
                 plots, and a cited implementation report"

Not this         "MATLAB replacement"
Instead         "A bridge from literature to reproducible control simulations,
                 with MATLAB/Octave/Python paths chosen explicitly"
```

The first target user should be a graduate student, lab engineer, or controls
developer working through papers and trying to reproduce or adapt controller and
observer designs. That user values citations, equations, runnable examples,
plots, and failure-mode explanations more than a large generic agent UI.

## Open Decisions

```text
Decision                              Recommended default
------------------------------------  -----------------------------------------
Product mode                          Private lab/team self-host first, not
                                      public SaaS first.

Frontend                              Keep Streamlit for current demo; begin API
                                      and frontend split after storage/ownership.

Vector backend                        Evaluate pgvector/Qdrant first; keep FAISS
                                      if single-node local remains intentional.

Execution sandbox                     Keep local providers disabled for public
                                      use; prove isolation before activation.

MATLAB strategy                       Start with Octave-compatible execution and
                                      MATLAB/Simulink export notes; defer real
                                      MATLAB backend until licensing is explicit.

Provider activation                   Defer paid/real provider switches until
                                      observability, quotas, and attribution exist.
```

## Source Register

Repo-local evidence:

- `docs/REPO_STATUS.md`
- `docs/DEPLOYMENT_STATUS.md`
- `docs/FEATURE_AUDIT.md`
- `docs/PLATFORM_AUDIT_AND_ROADMAP.md`
- `.venv/bin/python scripts/health_check.py --url ...`
- `.venv/bin/python scripts/health_check.py --ssh-host root@100.100.233.26`

External competitor and technical sources:

- GitHub API snapshot: https://api.github.com/repos/{owner}/{repo}
- Dify: https://github.com/langgenius/dify and https://docs.dify.ai/
- RAGFlow: https://github.com/infiniflow/ragflow and https://ragflow.io/docs/
- AnythingLLM: https://github.com/Mintplex-Labs/anything-llm and https://docs.anythingllm.com/
- Open WebUI: https://github.com/open-webui/open-webui and https://docs.openwebui.com/
- Langflow: https://github.com/langflow-ai/langflow and https://docs.langflow.org/
- Flowise: https://github.com/FlowiseAI/Flowise and https://docs.flowiseai.com/
- Haystack: https://github.com/deepset-ai/haystack and https://docs.haystack.deepset.ai/
- LlamaIndex: https://github.com/run-llama/llama_index and https://docs.llamaindex.ai/
- PaperQA: https://github.com/Future-House/paper-qa
- Elicit: https://elicit.com/ and https://help.elicit.com/
- NotebookLM: https://support.google.com/notebooklm/
- Cloudflare Sandbox: https://developers.cloudflare.com/sandbox/
- gVisor: https://gvisor.dev/docs/
- Langfuse: https://langfuse.com/docs/
- Ragas: https://docs.ragas.io/
- Qdrant: https://qdrant.tech/documentation/
- Milvus: https://milvus.io/docs/
- pgvector: https://github.com/pgvector/pgvector

External control/domain sources:

- Python Control Systems Library: https://python-control.readthedocs.io/
- do-mpc: https://www.do-mpc.com/
- OpenModelica: https://openmodelica.org/
- ControlSystems.jl: https://juliacontrol.github.io/ControlSystems.jl/stable/
- SimpleFOC docs: https://docs.simplefoc.com/
- SimpleFOC community: https://community.simplefoc.com/
- MathWorks Answers search: https://www.mathworks.com/matlabcentral/answers/
- TI E2E support forums: https://e2e.ti.com/
- ST Community motor-control forum: https://community.st.com/
- Reddit r/ControlTheory: https://www.reddit.com/r/ControlTheory/

Sampled community search themes from the 2026-06-08 qualitative pass: MathWorks
`sliding mode control Simulink`, `PMSM FOC Simulink`, and MathWorks Motor
Control Blockset examples; TI E2E and ST Community `PMSM FOC sensorless flux
observer`; SimpleFOC `current loop`, `motor parameters`, and unstable FOC
tuning; Reddit r/ControlTheory `MATLAB`, `Simulink`, `Python`, papers, and
implementation questions.
