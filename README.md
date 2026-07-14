# FluxMind

**A paper-grounded control-engineering copilot for Sliding Mode Control, PMSM drives, observers, and flux-linkage estimation.**

FluxMind turns a curated research corpus into traceable answers, retrieval diagnostics, paper-to-code handoffs, executable local examples, and no-secret operational evidence. It is currently a deployed small-group research baseline, not a fully activated SaaS platform.

> **For Contributors:** Before starting development, read [CODE_PRINCIPLES.md](CODE_PRINCIPLES.md) and [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md). The project has been refocused (2026-06-21) from over-engineering to delivering real user value.

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
Corpus baseline     52 curated papers, local and Trace-Twin FAISS index rebuilt, 3497 chunks
Eval baseline       42 recorded answers, 129 retrieval questions, offline and live retrieval eval passing
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
Product registry           optional local users/workspaces/RBAC/quotas/billing ledger, admin API/UI, query quota guard, and write-path RBAC guard
Share links                optional local hash-only share-token registry with admin API/CLI
Collaboration readiness    private-corpus/share-link preflight with no-secret policy matrix
Migration readiness        local restore rehearsal and opaque object-storage manifest verifier
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
optional share-link token registry
local product registry admin API/UI
optional query quota and local RBAC guards
local migration rehearsal and object manifest verifier
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
python scripts/import_seed_papers.py --require-count 52
python scripts/rebuild_seed_index.py --require-count 52
IMAGE_PROVIDER_BACKEND=openai python scripts/openai_image_smoke.py
python scripts/health_check.py
python scripts/storage_schema.py --format markdown
python scripts/platform_migration_rehearsal.py --include-object-manifest --format markdown
python scripts/platform_migration_rehearsal.py --include-object-manifest --output /tmp/fluxmind-object-manifest.json
python scripts/platform_migration_rehearsal.py --verify-object-manifest /tmp/fluxmind-object-manifest.json --format markdown
python scripts/platform_migration_rehearsal.py --include-object-manifest --include-job-store-manifest --output /tmp/fluxmind-object-and-job-rehearsal.json
python scripts/platform_migration_rehearsal.py --verify-job-store-manifest /tmp/fluxmind-object-and-job-rehearsal.json --format markdown
python scripts/api_key_registry.py status --format markdown
python scripts/share_link_registry.py status --format markdown
python scripts/product_registry.py status --format markdown
python scripts/platform_migration_preflight.py --format markdown
python scripts/platform_migration_rehearsal.py --format markdown
python scripts/product_readiness.py --format markdown
python scripts/product_activation_rehearsal.py --format markdown --require-activation
python scripts/collaboration_readiness.py --format markdown
python scripts/provider_readiness.py --format markdown
python scripts/provider_runtime_rehearsal.py --format markdown --require-local-foundation
python scripts/quality_readiness.py --format markdown
python scripts/activation_suite.py --format markdown --require-target local_foundation
python scripts/openapi_contract.py --format markdown --require-local-contract
python scripts/openapi_contract.py --verify-snapshot /tmp/fluxmind-openapi-contract.json --require-no-drift
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
DOCKER_PYTHON_EXECUTION_IMAGE=python:3.11-slim
DOCKER_OCTAVE_EXECUTION_IMAGE=gnuoctave/octave:latest
CODE_EXECUTION_POLICY=local-safe-v1
EXTERNAL_PROVIDERS_ENABLED=false
IMAGE_PROVIDER_BACKEND=local-mock
OPENAI_IMAGE_MODEL=gpt-image-2
OPENAI_IMAGE_OUTPUT_FORMAT=png
HOSTED_EXECUTION_BACKEND=none
MATLAB_BACKEND=none
PROVIDER_QUOTA_GUARD_ENABLED=false
PROVIDER_QUOTA_MAX_PROMPT_TOKENS_PER_REQUEST=128000
PROVIDER_QUOTA_MAX_COMPLETION_TOKENS_PER_REQUEST=4096
PROVIDER_QUOTA_MAX_COST_USD_PER_REQUEST=0
RETENTION_DELETE_ENABLED=false
FLUXMIND_API_KEY_REGISTRY_BACKEND=none
FLUXMIND_PRODUCT_REGISTRY_BACKEND=none
FLUXMIND_SHARE_LINK_TOKEN_STORE_BACKEND=none
FLUXMIND_STREAMLIT_PRODUCT_REGISTRY_MANAGEMENT_ENABLED=false
FLUXMIND_STREAMLIT_SHARE_LINK_MANAGEMENT_ENABLED=false
FLUXMIND_PRODUCT_QUOTA_GUARD_ENABLED=false
FLUXMIND_PRODUCT_RBAC_GUARD_ENABLED=false
IDENTITY_QUOTAS_BILLING_ENABLED=false
```

Set `IMAGE_PROVIDER_BACKEND=openai` and provide `OPENAI_IMAGE_API_KEY` or `OPENAI_API_KEY`
to make `POST /jobs/image`, `POST /jobs/async/image`, and the Streamlit image panel call
the OpenAI Images API. `POST /jobs/image/mock` and `POST /jobs/async/image/mock` remain
deterministic local SVG fallbacks.

Setting `FLUXMIND_API_KEY_REGISTRY_BACKEND=sqlite` enables the local hashed-token API-key registry. Setting `FLUXMIND_PRODUCT_REGISTRY_BACKEND=sqlite` enables the local users/workspaces/RBAC/quotas/billing-attribution ledger plus the local `/admin/product-registry/*` management API. Streamlit product-registry management remains additionally gated by `FLUXMIND_STREAMLIT_PRODUCT_REGISTRY_MANAGEMENT_ENABLED=true`, is disabled by default, and sanitizes operator-facing error output through the no-secret redaction boundary. Setting `FLUXMIND_SHARE_LINK_TOKEN_STORE_BACKEND=sqlite` enables the local hash-only share-link token registry and `/admin/share-links*` API/CLI lifecycle; create returns a one-time token, while list/revoke/resolve outputs omit raw tokens, URLs, resource refs, creator user IDs, descriptions, paths, and content. Streamlit share-link management remains additionally gated by `FLUXMIND_STREAMLIT_SHARE_LINK_MANAGEMENT_ENABLED=true`, is disabled by default, and sanitizes operator-facing error output through the same no-secret redaction boundary. Setting `FLUXMIND_PRODUCT_QUOTA_GUARD_ENABLED=true` lets `/query`, `/query/inspect`, `/query/retrieve`, and `/query/report` enforce the local request quota. Setting `FLUXMIND_PRODUCT_RBAC_GUARD_ENABLED=true` makes query routes require an active workspace membership, lets member/admin/owner roles submit or manage local jobs, and limits corpus/index/admin destructive writes to admin/owner roles. Setting `PROVIDER_QUOTA_GUARD_ENABLED=true` enables the no-secret provider pre-call guard for estimated prompt tokens, requested completion tokens, and optional cost ceilings before LLM/provider clients are constructed. These local registries and guards do not connect to an external identity provider or payment processor. Product and provider activation must pass their readiness gates before being treated as production-ready.

### Documentation Map

README is the GitHub entrypoint. Detailed facts live in owner documents:

```text
# Development Guides (Start Here for Contributors)
CODE_PRINCIPLES.md                     development principles, defensive code freeze, red lines
DEVELOPMENT_PLAN.md                    3-week detailed plan (2026-06-21 to 07-12)
NEXT_STEPS.md                          quick action checklist and checkpoints
DISCUSSION.md                          project diagnosis and decision records (reference)

# Current Documentation
docs/README.md                         reading order and source-of-truth map
docs/current/ARCHITECTURE.md           runtime boundaries and module design
docs/current/DEPLOYMENT_STATUS.md      live deployment snapshot and commands
docs/current/REPO_STATUS.md            git/worktree snapshot and verification

# Archived Documentation (Read-Only)
docs/archive/BACKLOG.md                past work packages
docs/archive/FEATURE_AUDIT.md          feature audit
docs/archive/PLATFORM_AUDIT_AND_ROADMAP.md  platform audit
docs/archive/PRODUCTION_GAP_AND_MARKET_RESEARCH.md  production gap analysis
docs/archive/QUALITY_ROADMAP.md        quality roadmap

# Demo Materials
docs/demo-script.md                    Chinese demo script and defense Q&A
```

### Roadmap

The local research baseline is usable. The remaining path is dependency-driven:

1. Expand community-quality evidence: more curated papers, recorded answers, retrieval questions, and live answer count/pass-rate/term-coverage evidence. The current PDF structure target is met; add more PDF cases when new source anchors are useful.
2. Finish production state foundations: metadata database, object storage, distributed job-store backend, migrations, backups, and restore drills.
3. Harden execution and observability: sandbox decision, abuse controls, production metrics/traces/alerts, cost attribution.
4. Add external product identity: identity-backed API keys, external quotas, billing/payment, team workflows, and audit controls.
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
语料基线           52 篇精选论文，本地与 Trace-Twin FAISS index 已重建，3497 个 chunks
评测基线           42 个 recorded answers，129 个 retrieval questions，离线与 live retrieval eval 通过
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
Product registry          可选本地 user/workspace/RBAC/quota/billing ledger、admin API/UI、query quota guard 与写路径 RBAC guard
Share links               可选本地 hash-only share-token registry，提供 admin API/CLI
协作就绪检查              私有语料/share-link 预检，输出 no-secret policy matrix
迁移准备                  本地 restore rehearsal 与 opaque object-storage manifest verifier
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
可选 share-link token registry
本地 product registry admin API/UI
可选 query quota 与本地 RBAC guard
本地 migration rehearsal 与 object manifest verifier
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
python scripts/import_seed_papers.py --require-count 52
python scripts/rebuild_seed_index.py --require-count 52
IMAGE_PROVIDER_BACKEND=openai python scripts/openai_image_smoke.py
python scripts/health_check.py
python scripts/storage_schema.py --format markdown
python scripts/platform_migration_rehearsal.py --include-object-manifest --format markdown
python scripts/platform_migration_rehearsal.py --include-object-manifest --output /tmp/fluxmind-object-manifest.json
python scripts/platform_migration_rehearsal.py --verify-object-manifest /tmp/fluxmind-object-manifest.json --format markdown
python scripts/platform_migration_rehearsal.py --include-object-manifest --include-job-store-manifest --output /tmp/fluxmind-object-and-job-rehearsal.json
python scripts/platform_migration_rehearsal.py --verify-job-store-manifest /tmp/fluxmind-object-and-job-rehearsal.json --format markdown
python scripts/api_key_registry.py status --format markdown
python scripts/share_link_registry.py status --format markdown
python scripts/product_registry.py status --format markdown
python scripts/platform_migration_preflight.py --format markdown
python scripts/platform_migration_rehearsal.py --format markdown
python scripts/product_readiness.py --format markdown
python scripts/product_activation_rehearsal.py --format markdown --require-activation
python scripts/collaboration_readiness.py --format markdown
python scripts/provider_readiness.py --format markdown
python scripts/provider_runtime_rehearsal.py --format markdown --require-local-foundation
python scripts/quality_readiness.py --format markdown
python scripts/activation_suite.py --format markdown --require-target local_foundation
python scripts/openapi_contract.py --format markdown --require-local-contract
python scripts/openapi_contract.py --verify-snapshot /tmp/fluxmind-openapi-contract.json --require-no-drift
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
DOCKER_PYTHON_EXECUTION_IMAGE=python:3.11-slim
DOCKER_OCTAVE_EXECUTION_IMAGE=gnuoctave/octave:latest
CODE_EXECUTION_POLICY=local-safe-v1
EXTERNAL_PROVIDERS_ENABLED=false
IMAGE_PROVIDER_BACKEND=local-mock
OPENAI_IMAGE_MODEL=gpt-image-2
OPENAI_IMAGE_OUTPUT_FORMAT=png
HOSTED_EXECUTION_BACKEND=none
MATLAB_BACKEND=none
PROVIDER_QUOTA_GUARD_ENABLED=false
PROVIDER_QUOTA_MAX_PROMPT_TOKENS_PER_REQUEST=128000
PROVIDER_QUOTA_MAX_COMPLETION_TOKENS_PER_REQUEST=4096
PROVIDER_QUOTA_MAX_COST_USD_PER_REQUEST=0
RETENTION_DELETE_ENABLED=false
FLUXMIND_API_KEY_REGISTRY_BACKEND=none
FLUXMIND_PRODUCT_REGISTRY_BACKEND=none
FLUXMIND_SHARE_LINK_TOKEN_STORE_BACKEND=none
FLUXMIND_STREAMLIT_PRODUCT_REGISTRY_MANAGEMENT_ENABLED=false
FLUXMIND_STREAMLIT_SHARE_LINK_MANAGEMENT_ENABLED=false
FLUXMIND_PRODUCT_QUOTA_GUARD_ENABLED=false
FLUXMIND_PRODUCT_RBAC_GUARD_ENABLED=false
IDENTITY_QUOTAS_BILLING_ENABLED=false
```

设置 `IMAGE_PROVIDER_BACKEND=openai` 并提供 `OPENAI_IMAGE_API_KEY` 或 `OPENAI_API_KEY`
后，`POST /jobs/image`、`POST /jobs/async/image` 和 Streamlit 图像面板会调用
OpenAI Images API。`POST /jobs/image/mock` 和 `POST /jobs/async/image/mock` 保持为
确定性的本地 SVG fallback。

设置 `FLUXMIND_API_KEY_REGISTRY_BACKEND=sqlite` 可以启用本地 hashed-token API-key registry。设置 `FLUXMIND_PRODUCT_REGISTRY_BACKEND=sqlite` 可以启用本地 user/workspace/RBAC/quota/billing-attribution ledger，以及本地 `/admin/product-registry/*` 管理 API。Streamlit product-registry 管理面还需要 `FLUXMIND_STREAMLIT_PRODUCT_REGISTRY_MANAGEMENT_ENABLED=true`，默认关闭，并且 operator 侧错误输出会走 no-secret redaction 边界。设置 `FLUXMIND_SHARE_LINK_TOKEN_STORE_BACKEND=sqlite` 可以启用本地 hash-only share-link token registry 和 `/admin/share-links*` API/CLI 生命周期；create 只返回一次 raw token，list/revoke/resolve 不导出 raw token、URL、resource ref、creator user ID、description、路径或内容。Streamlit share-link 管理面还需要 `FLUXMIND_STREAMLIT_SHARE_LINK_MANAGEMENT_ENABLED=true`，默认关闭，并且 operator 侧错误输出也走同一条 no-secret redaction 边界。设置 `FLUXMIND_PRODUCT_QUOTA_GUARD_ENABLED=true` 后，`/query`、`/query/inspect`、`/query/retrieve`、`/query/report` 会执行本地请求 quota guard。设置 `FLUXMIND_PRODUCT_RBAC_GUARD_ENABLED=true` 后，查询路由需要 active workspace membership，member/admin/owner 可提交或管理本地 job，corpus/index/admin 破坏性写操作限制为 admin/owner。设置 `PROVIDER_QUOTA_GUARD_ENABLED=true` 后，LLM/provider client 构造前会执行 no-secret provider 调用前 guard，检查估算 prompt token、请求 completion token 和可选 cost ceiling。这些本地 registry/guard 不连接外部身份 provider 或支付系统。生产级 product/provider 激活必须先通过对应 readiness 门禁。

### 文档地图

README 是 GitHub 入口页，详细事实由 owner 文档维护：

```text
# 开发指南（贡献者必读）
CODE_PRINCIPLES.md                     开发原则、防御代码冻结、红线警报
DEVELOPMENT_PLAN.md                    3周详细计划（2026-06-21 至 07-12）
NEXT_STEPS.md                          快速行动清单和检查点
DISCUSSION.md                          项目诊断和决策记录（参考）

# 当前文档
docs/README.md                         阅读顺序和事实归属图
docs/current/ARCHITECTURE.md           运行时边界和模块设计
docs/current/DEPLOYMENT_STATUS.md      live 部署快照和刷新命令
docs/current/REPO_STATUS.md            git/worktree 快照和验证记录

# 归档文档（只读）
docs/archive/BACKLOG.md                过去的工作包
docs/archive/FEATURE_AUDIT.md          功能审计
docs/archive/PLATFORM_AUDIT_AND_ROADMAP.md  平台审计
docs/archive/PRODUCTION_GAP_AND_MARKET_RESEARCH.md  生产缺口分析
docs/archive/QUALITY_ROADMAP.md        质量路线图

# 演示材料
docs/demo-script.md                    中文演示脚本和答辩问答
```

### 后续路线

当前本地研究基线已经可用，后续按依赖关系推进：

1. 扩展社区质量证据：论文库已达到 52 篇，本地与 Trace-Twin 生产索引均已重建为 3497 chunks，生产 live retrieval eval 已通过；下一步是补 live answer eval 与 recorded answers 数量、通过率和术语覆盖证据。当前 PDF structure 目标已达到；后续只在出现有价值的新源码锚点时继续增加 PDF cases。
2. 完成生产状态基础：metadata database、object storage、distributed job-store、迁移、备份和恢复演练。
3. 强化执行安全和观测：sandbox 方案、滥用控制、生产 metrics/traces/alerts、成本归因。
4. 建立外部产品身份层：identity-backed API key、外部 quota、billing/payment、team workflow 和审计控制。
5. 替换或包裹 demo-oriented Streamlit，形成可维护的正式产品 UI。

## Docker execution image policy for Trace-Twin

Trace-Twin is hosted in Hangzhou. Production Docker execution should not depend
on direct Docker Hub or GHCR pulls for large runtime images. The current live
configuration uses:

```text
CODE_EXECUTION_BACKEND=docker
DOCKER_PYTHON_EXECUTION_IMAGE=m.daocloud.io/docker.io/library/python:3.11-slim
DOCKER_OCTAVE_EXECUTION_IMAGE=fluxmind/octave:trixie-slim
```

The Octave runtime image is built locally from
`deploy/docker/octave-trixie-slim.Dockerfile`.

## Trace-Twin 的 Docker 执行镜像策略

Trace-Twin 位于杭州。生产 Docker 执行不应依赖 Docker Hub 或 GHCR 直连拉取大镜像。当前线上配置为：

```text
CODE_EXECUTION_BACKEND=docker
DOCKER_PYTHON_EXECUTION_IMAGE=m.daocloud.io/docker.io/library/python:3.11-slim
DOCKER_OCTAVE_EXECUTION_IMAGE=fluxmind/octave:trixie-slim
```

Octave runtime 镜像通过 `deploy/docker/octave-trixie-slim.Dockerfile` 在服务器本机构建。
