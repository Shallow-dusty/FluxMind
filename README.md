# FluxMind

**Control-theory research copilot for Sliding Mode Control, PMSM drives, and flux-linkage estimation.**

FluxMind is a RAG-based research assistant for reading control-engineering papers, checking source-backed claims, generating paper-to-code handoffs, and running local no-key development workflows. The current deployed baseline is designed for personal and small-group research use, with explicit readiness surfaces for later production storage, distributed jobs, identity, quotas, billing, and provider activation.

Languages: [English](#english) | [中文](#中文)

---

## English

### Current Status

```text
Project number                 11.FluxMind
Current maturity               small-group research baseline
Deployment                     live on Trace-Twin, UI/API/worker services active
Quality gate                   30 papers, 107 live retrieval questions, 42 recorded answers
Runtime model                  no-key/local platform foundation with guarded external readiness
Production platformization     not activated yet
```

- AI-Prism formal project number: `11`
- Active workspace directory: `11.FluxMind/`
- Previous temporary index `80` has been retired; the pre-formal snapshot is kept under `90.Archive/11-FluxMind-PreFormal/`.
- Public UI: `https://smy.hyper-dusty.cloud/`
- Public API health: `https://api-smy.hyper-dusty.cloud/health`

### What It Does

FluxMind helps a control researcher move from paper reading to implementation evidence:

- **Paper-grounded Q&A** with numbered citations and source/page checks.
- **Retrieval diagnostics** through no-LLM `/query/retrieve` and citation inspection.
- **Corpus management** for curated library papers, uploaded PDFs, active selections, and reusable corpus profiles.
- **Paper-to-code handoffs** with assumptions, source references, validation checklists, and local artifacts.
- **Local execution workflows** for Python and Octave-compatible control examples.
- **Artifact tracking** for generated plots, files, and mock diagrams with stable IDs and checksums.
- **No-secret admin status** for jobs, corpus state, storage schema, runtime events, metrics, retention preview, and platform readiness.

### Current Architecture

```text
Streamlit UI / FastAPI
        |
        v
Shared RAG core
        |
        +--> hybrid retrieval: FAISS vectors + metadata/docstore keyword signals
        +--> deterministic BM25-lite reranking, optional local CrossEncoder
        +--> numbered citation guard and source/page validation
        |
        v
Local no-key platform layer
        |
        +--> jobs: JSONL history + SQLite current-state mirror
        +--> corpus metadata: JSON + SQLite mirrors
        +--> artifacts: filesystem + SQLite registry
        +--> runtime events: metadata-only JSONL observability
```

External production components are intentionally **not** enabled by default. Metadata database, object storage, and distributed job-store targets are exposed as readiness configuration and blocker codes, not as active migrations.

### Quick Start

Use the existing `.venv` in this checkout when present:

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
```

For a fresh clone:

```bash
git clone https://github.com/Shallow-dusty/FluxMind.git
cd FluxMind
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Run the UI and API:

```bash
streamlit run app.py
uvicorn api:app --port 18502
python scripts/run_job_worker.py --loop --max-jobs 5
```

### Verification

Core local gates:

```bash
python -m pytest
python scripts/evaluate_rag.py
python scripts/health_check.py
python scripts/storage_schema.py --format markdown
```

Deployment and live checks:

```bash
python scripts/deploy_sync.py
python scripts/deploy_sync.py --apply --restart
python scripts/health_check.py --ssh-host root@100.100.233.26
python scripts/health_check.py --url https://smy.hyper-dusty.cloud/
curl -fsS https://api-smy.hyper-dusty.cloud/health
```

`scripts/deploy_sync.py` excludes runtime state such as `.env`, models, metadata, jobs, artifacts, uploaded papers, and FAISS indexes. The production host `/opt/fluxmind` is a synchronized source tree, not a git checkout.

### Configuration Boundary

Important no-key/local defaults:

```text
METADATA_STORAGE_BACKEND=local
OBJECT_STORAGE_BACKEND=local
DISTRIBUTED_JOB_STORE_BACKEND=local
CODE_EXECUTION_BACKEND=local
CODE_EXECUTION_POLICY=local-safe-v1
RETENTION_DELETE_ENABLED=false
```

External providers, hosted sandboxes, real MATLAB integration, identity, quotas, billing, production database/object storage, and distributed job-store activation remain disabled until their runtime boundaries are implemented and verified.

### Documentation

The README is only the project entrypoint. Detailed status and design facts live in owner documents:

```text
docs/README.md                         reading order and source-of-truth map
docs/REPO_STATUS.md                    git/worktree snapshot and local verification
docs/DEPLOYMENT_STATUS.md              live deployment snapshot and refresh commands
docs/ARCHITECTURE.md                   runtime boundaries and module architecture
docs/BACKLOG.md                        work packages and acceptance criteria
docs/QUALITY_ROADMAP.md                self-use, small-group, and community quality gates
docs/FEATURE_AUDIT.md                  implemented features, routes, and remaining gaps
docs/PLATFORM_AUDIT_AND_ROADMAP.md     platform roadmap and open decisions
docs/PRODUCTION_GAP_AND_MARKET_RESEARCH.md production gap and staged roadmap
docs/demo-script.md                    demo script and defense Q&A
```

### Roadmap

The completed baseline covers no-key/local research use. Remaining work is staged by dependency:

1. Community-quality expansion: 50+ curated papers, 80 recorded answers, 180 retrieval questions, and live answer evidence.
2. Production state foundation: external metadata database, object storage, distributed job-store backend, migration tests, and backup/restore drills.
3. Execution safety and observability: live sandbox decision, abuse policy, production metrics, tracing, and alert routing.
4. Identity and commercialization: users, workspaces, API keys, quotas, billing, and audit controls.
5. Product frontend/API split: replace the demo-oriented Streamlit surface with a maintainable product UI.

---

## 中文

### 当前状态

```text
项目编号                     11.FluxMind
当前成熟度                   小组研究可用基线
部署状态                     已部署在 Trace-Twin，UI/API/worker 服务可用
质量门槛                     30 篇论文、107 个 live retrieval 问题、42 个 recorded answers
运行时形态                   no-key/local 平台基础，外部能力只做 readiness
完整生产平台化               尚未激活
```

- AI-Prism 正式项目编号：`11`
- 当前工作区目录：`11.FluxMind/`
- 临时编号 `80` 已退役；迁移前快照保留在 `90.Archive/11-FluxMind-PreFormal/`。
- 公开 UI：`https://smy.hyper-dusty.cloud/`
- 公开 API 健康检查：`https://api-smy.hyper-dusty.cloud/health`

### 项目能做什么

FluxMind 面向控制理论研究，把“读论文、查证据、转实现”串成一个可验证流程：

- **基于论文的问答**：回答带编号引用，并检查来源页。
- **检索诊断**：通过 `/query/retrieve` 和 `/query/inspect` 检查上下文与引用质量。
- **语料管理**：支持内置论文库、上传 PDF、激活论文集合、可复用 corpus profile。
- **Paper-to-code 交接**：输出假设、参数边界、来源引用、验证清单和本地 artifact。
- **本地代码执行**：支持 Python 和 Octave-compatible 控制工程示例。
- **Artifact 管理**：生成图、文件、mock diagram，记录稳定 ID、校验和与元数据。
- **No-secret 管理面**：展示 job、corpus、storage schema、runtime events、metrics、retention preview 和 platform readiness。

### 当前架构

```text
Streamlit UI / FastAPI
        |
        v
共享 RAG 核心
        |
        +--> hybrid retrieval: FAISS 向量 + metadata/docstore 关键词信号
        +--> 确定性 BM25-lite rerank，可选本地 CrossEncoder
        +--> 编号引用校验和来源页检查
        |
        v
本地 no-key 平台层
        |
        +--> jobs: JSONL 历史 + SQLite 当前态镜像
        +--> corpus metadata: JSON + SQLite 镜像
        +--> artifacts: 文件系统 + SQLite registry
        +--> runtime events: 仅元数据 JSONL 观测层
```

外部生产组件默认**不启用**。Metadata database、object storage、distributed job-store 目前只通过配置、readiness 和 blocker code 暴露，不代表已经迁移或激活。

### 快速启动

当前 checkout 优先使用已有 `.venv`：

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

运行 UI、API 和本地 worker：

```bash
streamlit run app.py
uvicorn api:app --port 18502
python scripts/run_job_worker.py --loop --max-jobs 5
```

### 验证命令

本地核心门禁：

```bash
python -m pytest
python scripts/evaluate_rag.py
python scripts/health_check.py
python scripts/storage_schema.py --format markdown
```

部署与 live 检查：

```bash
python scripts/deploy_sync.py
python scripts/deploy_sync.py --apply --restart
python scripts/health_check.py --ssh-host root@100.100.233.26
python scripts/health_check.py --url https://smy.hyper-dusty.cloud/
curl -fsS https://api-smy.hyper-dusty.cloud/health
```

`scripts/deploy_sync.py` 会排除 `.env`、模型、metadata、jobs、artifacts、上传论文和 FAISS index 等运行时状态。生产目录 `/opt/fluxmind` 是同步后的源码树，不是 git checkout。

### 配置边界

关键 no-key/local 默认值：

```text
METADATA_STORAGE_BACKEND=local
OBJECT_STORAGE_BACKEND=local
DISTRIBUTED_JOB_STORE_BACKEND=local
CODE_EXECUTION_BACKEND=local
CODE_EXECUTION_POLICY=local-safe-v1
RETENTION_DELETE_ENABLED=false
```

真实外部 provider、托管 sandbox、真实 MATLAB 集成、身份、配额、计费、生产数据库/object storage、distributed job-store 激活都仍处于禁用状态；需要先实现和验证对应运行时边界。

### 文档入口

README 只是 GitHub 入口页。详细状态、架构和计划由以下 owner 文档维护：

```text
docs/README.md                         阅读顺序和事实归属图
docs/REPO_STATUS.md                    git/worktree 快照和本地验证
docs/DEPLOYMENT_STATUS.md              live 部署快照和刷新命令
docs/ARCHITECTURE.md                   运行时边界和模块架构
docs/BACKLOG.md                        工作包和验收标准
docs/QUALITY_ROADMAP.md                self-use / small-group / community 质量门槛
docs/FEATURE_AUDIT.md                  已实现功能、API route 和剩余缺口
docs/PLATFORM_AUDIT_AND_ROADMAP.md     平台路线图和开放决策
docs/PRODUCTION_GAP_AND_MARKET_RESEARCH.md 生产缺口和分阶段路线
docs/demo-script.md                    演示脚本和答辩问答
```

### 后续路线

已完成的是 no-key/local 研究基线。剩余工作按依赖关系推进：

1. 社区质量扩展：50+ 篇精选论文、80 个 recorded answers、180 个 retrieval questions、live answer evidence。
2. 生产状态基础：外部 metadata database、object storage、distributed job-store、迁移测试、备份恢复演练。
3. 执行安全和观测：live sandbox 方案、滥用策略、生产 metrics、tracing、alert routing。
4. 身份和商业化：用户、workspace、API key、quota、billing、审计控制。
5. 产品前端/API split：用可维护的正式产品 UI 替代 demo-oriented Streamlit 表面。
