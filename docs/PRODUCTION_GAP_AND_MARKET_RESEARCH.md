# FluxMind Production Gap and Market Research

Last updated: 2026-06-15 08:48 CST

This document answers: what should FluxMind build next, what is still missing
before it can be treated as a production-grade product, and what external
competitor/community signals support the plan.

It separates four evidence layers:

```text
Layer                 Current source
--------------------  -------------------------------------------------------
Current repo state    git status/log plus docs/REPO_STATUS.md
Current live state    health_check.py HTTPS and SSH checks run on 2026-06-15
                      08:48 CST; re-run before deploy claims
Repo snapshots        docs/ARCHITECTURE.md, docs/BACKLOG.md,
                      docs/PLATFORM_AUDIT_AND_ROADMAP.md, docs/FEATURE_AUDIT.md
External research     Public project docs, GitHub API, community/forum search
                      (GitHub counts refreshed 2026-06-15; forum themes remain
                      the 2026-06-08 qualitative sample)
```

External links and GitHub counts are time-sensitive. Re-run public checks before
using this document for investment, deployment, or release decisions.

## Current Baseline

Documentation-pass state:

```text
Branch          main
Start state     main...origin/main up to date at 3b8ecd7 before docs refresh
Docs refresh    20d75e5 docs: refresh FluxMind documentation state
Status note     this final docs-status note is docs-only and may create a newer commit
Deployed code   4f27651 fix: exclude coverage data from deploy sync
Work scope      docs/status/test-guard updates only; no application-code changes
Diff hygiene    git diff --check passed on 2026-06-15 08:48 CST
```

Current local verification from this pass plus the latest deployment snapshot:

```text
Check                                      Result
----------------------------------------  -------------------------------------
pytest                                    332 passed, 2 known warnings
coverage                                  88% total branch coverage over api,
                                          scripts, and src
offline RAG eval                          28 answer cases, 46 retrieval-only
                                          cases, 8 code-output cases,
                                          12 PDF structure cases,
                                          28 recorded answers
local health_check.py                     pass, local/docs/query-latency/query-alert/provider-alert/job-alert/API-access-audit/API-rate-limit/upload-scan/retention-delete/metrics-export/retrieval-trace/retrieval-alerts/storage-schema/artifact-limit/execution-alert anchors
storage_schema.py                         pass, ok=true, 7 stores, 0 problems
runtime restore dry-run                   pass, ok=true, 6 groups, 5 checked files, manifest_errors=0 against exported local manifest
HTTPS UI                                  08:48 snapshot: https://smy.hyper-dusty.cloud/ 200
HTTPS API health                          08:48 snapshot: https://api-smy.hyper-dusty.cloud/health 200
SSH health                                08:48 snapshot: pass on root@100.100.233.26
Remote services                           UI/API/worker/cloudflared/docker active
Remote listeners                          0.0.0.0:18501 and 0.0.0.0:18502
Remote model                              LLM_MODEL=mimo-v2.5-pro
Remote active corpus                       active_papers=11
Remote chunk metadata                     chunk_metadata_rows=800, sources=11
Remote index freshness                     index_fresh=True
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
Content/corpus          14-paper curated seed library, local metadata, profiles, chunk mirror,
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
                        owner metadata.

Gap to production       Relational metadata store, object storage, vector DB or
                        managed vector index, identity-backed ownership,
                        migrations, backups, full restore drills, retention
                        policies.

Priority                High. Identity, quotas, billing, and durable workers all
                        depend on this boundary.
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
                        on live deployment.

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

Gap to production       User accounts, workspaces, RBAC, API-key lifecycle,
                        quotas, billing, share/export flows, real frontend,
                        onboarding, team/lab workflows.

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
                        no-secret local metrics text export, health checks.

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
                        local API rate-limit guard, local metadata-only upload
                        scan guard, guarded local retention-delete switch,
                        public UI intentionally open in current deployment.

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
M0 current      14 bundled seed papers; deployed active index may still reflect
                the latest runtime rebuild snapshot
M1              30-50 curated papers, tagged by topic and method
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
6      Identity/workspaces/API keys/quotas        Needs storage/job ownership
7      Frontend/API split                         Avoids overbuilding Streamlit
8      Provider activation/billing                Only after trust and control
```

### 0-30 Days: Domain Trust Sprint

- Expand the curated corpus from 11 papers to 30-50 high-quality papers.
- Add topic tags and a control-engineering ontology: SMC, FOC, PMSM, SMO,
  observers, flux estimation, chattering, discretization, parameter tuning.
- Create at least 60 retrieval-only eval questions and 40 recorded-answer eval cases for the small-group bar.
- Add equation/table/figure extraction acceptance tests for representative PDFs.
- Add code-output evals where Python/Octave examples must run and produce plots.
- Add a "paper-to-code report" export: source refs, assumptions, parameters,
  generated code, execution output, plot artifacts.

Current progress on 2026-06-15: the no-key baseline has advanced from 5 to 28
offline/recorded answer cases and 46 retrieval-only cases, for 74 total no-LLM
retrieval questions. The baseline gates 96 source/page refs and 80 topic tags
across retrieval, answer quality, equation fidelity, code generation,
forum-style debugging, failure modes, and paper-to-code reports, and includes
eight local Python code-output gates that verify expected stdout plus plot/text
artifacts in a temporary artifact store, including reusable execution-template
coverage plus four local job-backed execution paths. The evaluator also has 12
seeded PDF structure gates for equation/table/figure markers on representative
source pages, and `GET /corpus/structure/report` exports filtered structure
anchors as a Markdown handoff report. `POST /query/report` now adds a local
paper-to-code handoff for implementation and code-generation reports, including
source refs, assumption/parameter guardrails, fenced code blocks, cited artifact
IDs, and validation checklist fields. The self-use target is met; the small-group
target still needs corpus growth from 14 bundled seed papers toward 30 curated
papers, plus 12 more answer/recorded-answer cases, 14 more retrieval-only cases,
26 more total retrieval questions, and 3 more PDF
structure cases. Broader Octave execution remains deferred until an Octave binary
is available in CI/runtime. The 2026-06-15 seed
library expansion adds adaptive-gain SMO, super-twisting SMO, switching-function
comparison, adaptive-parameter IPMSM, MRAS flux-linkage observer, combined
reaching-law SMO, PID/ITSMRL/ESO speed control, and fuzzy super-twisted SMO
coverage.

Success criterion: a skeptical control student or engineer can ask paper-backed
implementation questions and receive traceable, executable outputs.

### 30-60 Days: Production State Foundation

- Choose production storage: PostgreSQL metadata plus object storage, and either
  pgvector/Qdrant/Milvus for vector state or a deliberately local-only vector
  boundary with explicit backup/restore.
- Migrate corpus, chunks, jobs, artifacts, query events, and eval reports behind
  versioned schemas.
- Extend the current no-secret manifest and dry-run verifier into migration
  tests, backup/restore drills, retention policy, and restore docs.
- Use the current `platform_readiness` blocker summary as the local acceptance
  gate for the storage/queue backend choice; it already confirms clean local
  schema/inventory and local worker-bridge contracts while flagging missing
  external metadata database, object storage, and distributed job-store targets.
- Promote the current local API ownership metadata into versioned production
  schemas once storage/backend ownership is selected.
- Extend local idempotent job submission plus bounded dead-letter/retry policy
  into the future distributed worker queue.

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

Success criterion: generated code can run in a controlled environment, and every
expensive or risky operation is observable and attributable.

### 90-120 Days: Product Shell

- Introduce users, workspaces, corpora ownership, API-key lifecycle, quotas, and
  admin roles.
- Replace or wrap Streamlit with a frontend that supports workspace navigation,
  artifact galleries, job timelines, citation inspection, and exports.
- Add onboarding flows for: upload paper, ask paper-grounded question, generate
  simulation, inspect plot, export report.
- Add share/export controls for lab/team use.

Success criterion: FluxMind can support at least a small private lab/team
without mixing private corpora, jobs, artifacts, or costs.

### 120+ Days: Provider and Business Activation

- Enable paid LLM/image/sandbox providers only after observability, quotas, and
  spend attribution are implemented.
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
