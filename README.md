# FluxMind

**A paper-grounded control-engineering copilot for Sliding Mode Control, PMSM drives, observers, and flux-linkage estimation.**

FluxMind turns a curated research corpus into traceable answers, retrieval diagnostics, paper-to-code handoffs, executable local examples, and no-secret operational evidence. It is currently a deployed small-group research baseline, not a fully activated SaaS platform.

[English](#english) | [中文](#中文)

---

## English

### Why FluxMind Exists

Generic document chat is easy to build and hard to trust. FluxMind is narrower:

- It focuses on control-engineering papers and implementation workflows.
- It keeps answers tied to numbered source chunks and source/page evidence.
- It exposes retrieval diagnostics so failures can be inspected before trusting generated text.
- It can turn paper context into implementation notes, validation checklists, local plots, and artifacts.
- It keeps production-risky systems behind explicit readiness checks instead of silently enabling them.

The result is a research workspace for moving from "what does this paper claim?" to "what can I verify, run, export, and revisit?"

### Current Snapshot

```text
Project index       11.FluxMind
Primary domain      SMC, PMSM/FOC, observers, flux estimation, control implementation
Current maturity    small-group research baseline
Deployment          Trace-Twin, independent UI/API/worker systemd services
Corpus baseline     30 curated papers, fresh FAISS index, 1934 chunks
Eval baseline       42 recorded answers, 107 live retrieval questions
Runtime stance      no-key/local by default; external activation is explicit
```

- Public UI: `https://smy.hyper-dusty.cloud/`
- API health: `https://api-smy.hyper-dusty.cloud/health`
- Active workspace directory: `11.FluxMind/`
- Previous temporary index `80` has been retired; the archived pre-formal snapshot lives under `90.Archive/11-FluxMind-PreFormal/`.

Live deployment facts are mutable. Before making deployment claims, refresh [docs/DEPLOYMENT_STATUS.md](docs/DEPLOYMENT_STATUS.md).

### Core Capabilities

```text
Capability                 What is implemented
-------------------------  -----------------------------------------------------
Paper-grounded Q&A         RAG answers with source/page context and citation guard
Retrieval diagnostics      no-LLM /query/retrieve and /query/inspect inspection
Corpus control             library PDFs, uploads, active set, reusable profiles
Paper-to-code reports      assumptions, parameters, source refs, code, artifacts
Local execution            Python and Octave-compatible control examples
Artifacts                  generated plots/files/diagrams with IDs and checksums
Jobs                       local durable JSONL + SQLite job state, worker service
Admin/status               no-secret status, events, metrics, retention preview
API keys                   optional local SQLite registry with hashed tokens only
Product registry           optional local users/workspaces/quotas/billing ledger and query quota guard
Readiness gates            quality, platform, product, provider, migration checks
```

### Architecture

```text
Streamlit UI                         FastAPI
     |                                  |
     +---------------+------------------+
                     |
                     v
              shared RAG core
                     |
       +-------------+-------------+
       |                           |
       v                           v
hybrid retrieval              answer/report layer
FAISS + keyword               citation guard
BM25-lite rerank              source/page evidence
optional local reranker       paper-to-code exports
       |
       v
local no-key platform layer
jobs JSONL/SQLite
corpus JSON/SQLite
artifact SQLite/filesystem
runtime events JSONL
optional hashed API-key registry
optional product registry SQLite ledger
optional query quota guard
```

External databases, object storage, distributed queues, external image providers, hosted sandboxes, real MATLAB integration, identity, quotas, and billing are not enabled by default. They are represented by provider-neutral interfaces, configuration flags, readiness reports, and blocker codes until intentionally activated.

### Quick Start

Use the existing virtual environment if this checkout already has one:

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Fresh clone:

```bash
git clone https://github.com/Shallow-dusty/FluxMind.git
cd FluxMind
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Run the three local processes:

```bash
streamlit run app.py
uvicorn api:app --port 18502
python scripts/run_job_worker.py --loop --max-jobs 5
```

### Verification

Local gates:

```bash
python -m pytest
python scripts/evaluate_rag.py
python scripts/health_check.py
python scripts/storage_schema.py --format markdown
python scripts/api_key_registry.py status --format markdown
python scripts/product_registry.py status --format markdown
python scripts/platform_migration_preflight.py --format markdown
python scripts/platform_migration_rehearsal.py --format markdown
python scripts/product_readiness.py --format markdown
python scripts/provider_readiness.py --format markdown
python scripts/quality_readiness.py --format markdown
```

Deployment checks:

```bash
python scripts/deploy_sync.py
python scripts/deploy_sync.py --apply --restart
python scripts/health_check.py --ssh-host root@100.100.233.26
python scripts/health_check.py \
  --url https://smy.hyper-dusty.cloud/ \
  --url https://api-smy.hyper-dusty.cloud/health
```

`scripts/deploy_sync.py` excludes runtime state such as `.env`, models, metadata, jobs, artifacts, uploaded papers, and FAISS indexes. The production directory `/opt/fluxmind` is a synchronized source tree, not a git checkout.

### Configuration Boundary

Important defaults:

```text
METADATA_STORAGE_BACKEND=local
OBJECT_STORAGE_BACKEND=local
DISTRIBUTED_JOB_STORE_BACKEND=local
CODE_EXECUTION_BACKEND=local
CODE_EXECUTION_POLICY=local-safe-v1
EXTERNAL_PROVIDERS_ENABLED=false
IMAGE_PROVIDER_BACKEND=local-mock
HOSTED_EXECUTION_BACKEND=none
MATLAB_BACKEND=none
PROVIDER_QUOTA_GUARD_ENABLED=false
RETENTION_DELETE_ENABLED=false
FLUXMIND_API_KEY_REGISTRY_BACKEND=none
FLUXMIND_PRODUCT_REGISTRY_BACKEND=none
FLUXMIND_PRODUCT_QUOTA_GUARD_ENABLED=false
IDENTITY_QUOTAS_BILLING_ENABLED=false
```

Setting `FLUXMIND_API_KEY_REGISTRY_BACKEND=sqlite` enables the local hashed-token API-key registry. Setting `FLUXMIND_PRODUCT_REGISTRY_BACKEND=sqlite` enables the local users/workspaces/quotas/billing-attribution ledger. Setting `FLUXMIND_PRODUCT_QUOTA_GUARD_ENABLED=true` lets `/query`, `/query/inspect`, `/query/retrieve`, and `/query/report` enforce the local request quota. These local registries do not connect to an external identity provider or payment processor. Product and provider activation must pass their readiness gates before being treated as production-ready.

### Documentation Map

README is the GitHub entrypoint. Detailed facts live in owner documents:

```text
docs/README.md                         reading order and source-of-truth map
docs/REPO_STATUS.md                    git/worktree snapshot and verification
docs/DEPLOYMENT_STATUS.md              live deployment snapshot and commands
docs/ARCHITECTURE.md                   runtime boundaries and module design
docs/BACKLOG.md                        work packages and acceptance criteria
docs/QUALITY_ROADMAP.md                self-use, small-group, community gates
docs/FEATURE_AUDIT.md                  implemented surface and remaining gaps
docs/PLATFORM_AUDIT_AND_ROADMAP.md     platform audit and open decisions
docs/PRODUCTION_GAP_AND_MARKET_RESEARCH.md production gap and staged roadmap
docs/demo-script.md                    Chinese demo script and defense Q&A
```

### Roadmap

The local research baseline is usable. The remaining path is dependency-driven:

1. Expand community-quality evidence: more curated papers, recorded answers, retrieval questions, PDF structure cases, and live answer evidence.
2. Finish production state foundations: metadata database, object storage, distributed job-store backend, migrations, backups, and restore drills.
3. Harden execution and observability: sandbox decision, abuse controls, production metrics/traces/alerts, cost attribution.
4. Add product identity: users, workspaces, identity-backed API keys, quotas, billing, and audit controls.
5. Replace or wrap the demo-oriented Streamlit surface with a maintainable product UI.

---

## 中文

### FluxMind 是什么

FluxMind 是面向控制理论研究的论文工作台，重点覆盖滑模控制、PMSM/FOC、观测器、磁链估计和控制实现。它不是泛用 PDF 聊天框，而是把论文证据、检索诊断、实现交接、本地执行和运行状态检查串在一起。

它解决的问题是：从“论文说了什么”走到“哪些来源能证明、哪些代码能跑、哪些输出能复现、哪些状态能审计”。

### 当前状态

```text
项目编号           11.FluxMind
核心领域           SMC、PMSM/FOC、观测器、磁链估计、控制实现
当前成熟度         小组研究可用基线
部署形态           Trace-Twin，独立 UI/API/worker systemd 服务
语料基线           30 篇精选论文，FAISS index fresh，1934 个 chunks
评测基线           42 个 recorded answers，107 个 live retrieval questions
运行时边界         默认 no-key/local，外部能力必须显式激活
```

- 公开 UI：`https://smy.hyper-dusty.cloud/`
- API 健康检查：`https://api-smy.hyper-dusty.cloud/health`
- 当前工作区目录：`11.FluxMind/`
- 临时编号 `80` 已退役；迁移前归档在 `90.Archive/11-FluxMind-PreFormal/`。

部署状态会变化。需要对外汇报前，先刷新 [docs/DEPLOYMENT_STATUS.md](docs/DEPLOYMENT_STATUS.md)。

### 已有能力

```text
能力                      当前实现
------------------------  -----------------------------------------------------
论文证据问答              RAG 回答，带来源页和编号引用校验
检索诊断                  no-LLM /query/retrieve 和 /query/inspect
语料控制                  内置论文库、上传 PDF、激活集合、corpus profile
Paper-to-code 报告        假设、参数、来源引用、代码、artifact 和验证清单
本地执行                  Python 与 Octave-compatible 控制工程示例
Artifact 管理             生成图、文件、diagram，带稳定 ID 和 checksum
Job 系统                  本地 durable JSONL + SQLite 状态和 worker 服务
管理面                    no-secret status、events、metrics、retention preview
API key                   可选本地 SQLite registry，只持久化 token hash
Product registry          可选本地 user/workspace/quota/billing ledger 与 query quota guard
Readiness 门禁            quality、platform、product、provider、migration 检查
```

### 架构概览

```text
Streamlit UI                         FastAPI
     |                                  |
     +---------------+------------------+
                     |
                     v
                共享 RAG 核心
                     |
       +-------------+-------------+
       |                           |
       v                           v
hybrid retrieval              answer/report layer
FAISS + keyword               citation guard
BM25-lite rerank              source/page evidence
可选本地 reranker             paper-to-code exports
       |
       v
本地 no-key 平台层
jobs JSONL/SQLite
corpus JSON/SQLite
artifact SQLite/filesystem
runtime events JSONL
可选 hashed API-key registry
可选 product registry SQLite ledger
可选 query quota guard
```

外部数据库、object storage、分布式队列、外部图像 provider、托管 sandbox、真实 MATLAB、身份、配额和计费默认不启用。项目通过 provider-neutral 接口、配置开关、readiness 报告和 blocker code 表示这些边界，直到它们被明确激活和验证。

### 快速启动

已有 checkout 优先使用 `.venv`：

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
```

全新克隆：

```bash
git clone https://github.com/Shallow-dusty/FluxMind.git
cd FluxMind
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

运行三个本地进程：

```bash
streamlit run app.py
uvicorn api:app --port 18502
python scripts/run_job_worker.py --loop --max-jobs 5
```

### 验证命令

本地门禁：

```bash
python -m pytest
python scripts/evaluate_rag.py
python scripts/health_check.py
python scripts/storage_schema.py --format markdown
python scripts/api_key_registry.py status --format markdown
python scripts/product_registry.py status --format markdown
python scripts/platform_migration_preflight.py --format markdown
python scripts/platform_migration_rehearsal.py --format markdown
python scripts/product_readiness.py --format markdown
python scripts/provider_readiness.py --format markdown
python scripts/quality_readiness.py --format markdown
```

部署检查：

```bash
python scripts/deploy_sync.py
python scripts/deploy_sync.py --apply --restart
python scripts/health_check.py --ssh-host root@100.100.233.26
python scripts/health_check.py \
  --url https://smy.hyper-dusty.cloud/ \
  --url https://api-smy.hyper-dusty.cloud/health
```

`scripts/deploy_sync.py` 会排除 `.env`、models、metadata、jobs、artifacts、上传论文和 FAISS index 等运行时状态。生产目录 `/opt/fluxmind` 是同步后的源码树，不是 git checkout。

### 配置边界

关键默认值：

```text
METADATA_STORAGE_BACKEND=local
OBJECT_STORAGE_BACKEND=local
DISTRIBUTED_JOB_STORE_BACKEND=local
CODE_EXECUTION_BACKEND=local
CODE_EXECUTION_POLICY=local-safe-v1
EXTERNAL_PROVIDERS_ENABLED=false
IMAGE_PROVIDER_BACKEND=local-mock
HOSTED_EXECUTION_BACKEND=none
MATLAB_BACKEND=none
PROVIDER_QUOTA_GUARD_ENABLED=false
RETENTION_DELETE_ENABLED=false
FLUXMIND_API_KEY_REGISTRY_BACKEND=none
FLUXMIND_PRODUCT_REGISTRY_BACKEND=none
FLUXMIND_PRODUCT_QUOTA_GUARD_ENABLED=false
IDENTITY_QUOTAS_BILLING_ENABLED=false
```

设置 `FLUXMIND_API_KEY_REGISTRY_BACKEND=sqlite` 可以启用本地 hashed-token API-key registry。设置 `FLUXMIND_PRODUCT_REGISTRY_BACKEND=sqlite` 可以启用本地 user/workspace/quota/billing-attribution ledger。设置 `FLUXMIND_PRODUCT_QUOTA_GUARD_ENABLED=true` 后，`/query`、`/query/inspect`、`/query/retrieve`、`/query/report` 会执行本地请求 quota guard。这些本地 registry 不连接外部身份 provider 或支付系统。生产级 product/provider 激活必须先通过对应 readiness 门禁。

### 文档地图

README 是 GitHub 入口页，详细事实由 owner 文档维护：

```text
docs/README.md                         阅读顺序和事实归属图
docs/REPO_STATUS.md                    git/worktree 快照和验证记录
docs/DEPLOYMENT_STATUS.md              live 部署快照和刷新命令
docs/ARCHITECTURE.md                   运行时边界和模块设计
docs/BACKLOG.md                        工作包和验收标准
docs/QUALITY_ROADMAP.md                self-use / small-group / community 门槛
docs/FEATURE_AUDIT.md                  已实现功能和剩余缺口
docs/PLATFORM_AUDIT_AND_ROADMAP.md     平台审计和开放决策
docs/PRODUCTION_GAP_AND_MARKET_RESEARCH.md 生产缺口和分阶段路线
docs/demo-script.md                    中文演示脚本和答辩问答
```

### 后续路线

当前本地研究基线已经可用，后续按依赖关系推进：

1. 扩展社区质量证据：更多精选论文、recorded answers、retrieval questions、PDF structure cases 和 live answer evidence。
2. 完成生产状态基础：metadata database、object storage、distributed job-store、迁移、备份和恢复演练。
3. 强化执行安全和观测：sandbox 方案、滥用控制、生产 metrics/traces/alerts、成本归因。
4. 建立产品身份层：用户、workspace、identity-backed API key、quota、billing 和审计控制。
5. 替换或包裹 demo-oriented Streamlit，形成可维护的正式产品 UI。
