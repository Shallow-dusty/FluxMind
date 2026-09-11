# FluxMind

**A paper-grounded control-engineering copilot for Sliding Mode Control, PMSM
drives, observers, and flux-linkage estimation.**

FluxMind turns a curated paper corpus into traceable answers, retrieval
diagnostics, paper-to-code handoffs, executable examples, artifacts, and saved
per-user research history. It is built for a 5–20 person research group, not as
an enterprise SaaS platform.

> Contributors must read [DEVELOPMENT.md](DEVELOPMENT.md) before changing the
> project. It is the source of truth for scope, status, conventions, and roadmap.

[English](#english) · [中文](#中文)

## English

### What it does

```text
Capability              Implementation
----------------------  -------------------------------------------------------
Paper-grounded Q&A      Hybrid retrieval, rerank, inline [n] citation validation
Retrieval inspection    /query/retrieve and /query/inspect diagnostics
Corpus management       52-paper library, uploads, active set, saved profiles
Local accounts          admin/student login, history, owned jobs/artifacts
Paper-to-code           implementation reports, Python/Octave examples, plots
Jobs                    durable local JSONL + SQLite queue and worker
Artifacts               generated files with IDs and integrity metadata
Operations              corpus/index drift, jobs, artifacts, users, failures
```

The active runtime stays deliberately local:

```text
Streamlit UI (:18501)            FastAPI (:18502)
           \                       /
            \                     /
                  src/chain.py
         retrieval → rerank → generation
                     |
       +-------------+-------------+
       |             |             |
    metadata        jobs        artifacts
  JSON/SQLite   JSONL/SQLite   files/SQLite
```

SSO, billing, tenant isolation, external databases, and distributed queues are
outside the current product scope.

### Quick start

Use the existing project environment:

```bash
conda activate fluxmind
pip install -r requirements-dev.txt
cp .env.example .env
```

If it does not exist, create a Python 3.11+ environment first. Then start the
three processes:

```bash
streamlit run app.py
uvicorn api:app --port 18502
python scripts/run_job_worker.py --loop --max-jobs 5
```

On the first Streamlit visit, create the administrator account. Administrators
can create or disable students and reset passwords. Each account can reload or
clear its own saved query history.

### Verification

Supported local gates:

```bash
python -m pytest
python scripts/evaluate_rag.py
python scripts/health_check.py
```

Corpus reconstruction:

```bash
python scripts/import_seed_papers.py --require-count 52
python scripts/rebuild_seed_index.py --require-count 52
```

Explicit provider smokes:

```bash
IMAGE_PROVIDER_BACKEND=openai python scripts/openai_image_smoke.py
CODE_EXECUTION_BACKEND=docker python scripts/docker_execution_smoke.py --language all
```

### Configuration

Copy `.env.example` to `.env`; never commit credentials. Important settings:

```text
LLM_* / OPENAI_*                 LLM endpoint and model
IMAGE_PROVIDER_BACKEND           mock or openai
OPENAI_IMAGE_*                   image endpoint, model, quality, timeout
CODE_EXECUTION_BACKEND           local or docker
DOCKER_*_EXECUTION_IMAGE         Python and Octave runtime images
FLUXMIND_API_TOKEN               optional shared API token
FLUXMIND_USER_STORE_FILE         optional account DB path
PROVIDER_QUOTA_GUARD_ENABLED     optional provider token/cost ceiling
RETENTION_DELETE_ENABLED         enables explicit retention deletion
```

Image and code work is job-backed. The real image provider and Docker execution
are used only when explicitly configured; deterministic mock and local
backends remain available as explicit choices. There is no silent backend
substitution after a configured provider fails. Job details include the
submitted request, stdout/stderr, errors, logs, and artifact metadata so failed
runs are directly debuggable.
Streamlit jobs are attached to the signed-in account; students see their own
runs while administrators can inspect all runs.

### API highlights

```text
POST /query                         answer with citation metadata
POST /query/inspect                 answer plus detailed retrieval inspection
POST /query/retrieve                retrieval only, no model call
POST /query/report                  downloadable Markdown report
GET  /query/history/{user_id}       saved history for an active local user
GET  /corpus/*                      paper/chunk/profile state
POST /jobs/*                        image, Python, Octave, and index jobs
GET  /jobs                          recent jobs
GET  /artifacts                     generated artifacts
GET  /admin/status                  real local runtime status
GET  /admin/events                  recent runtime events
GET  /health and /ready             liveness and retrieval readiness
```

Set `FLUXMIND_API_TOKEN` to protect API calls with either
`Authorization: Bearer ...` or `X-API-Key: ...`. Leaving it blank keeps local
development open.

### Deployment

Production runs on Trace-Twin with independent UI, API, and worker systemd
services:

- UI: `https://smy.hyper-dusty.cloud/`
- API: `https://api-smy.hyper-dusty.cloud/`

Deployment is dry-run by default:

```bash
python scripts/deploy_sync.py
python scripts/deploy_sync.py --apply --restart
```

Runtime state (`.env`, PDFs, FAISS index, metadata, jobs, and artifacts) is
excluded from source synchronization.

## 中文

FluxMind 是面向滑模控制、PMSM、观测器和磁链估计研究的 RAG Copilot。它把
精选论文库变成可追溯答案、检索诊断、论文到代码的衔接、可执行示例、工件和
按用户保存的研究历史。

### 当前能力

```text
能力                    实现
----------------------  -------------------------------------------------------
论文问答                混合检索、重排、内联 [n] 引用校验
检索诊断                /query/retrieve 与 /query/inspect
语料管理                52 篇论文库、上传、active set、可复用 profile
本地账户                admin/student 登录、个人历史、任务/工件归属
论文到代码              实现报告、Python/Octave 示例、图像和产物
任务系统                本地 JSONL + SQLite durable job 与独立 worker
运行状态                语料/index 漂移、任务、工件、用户和近期失败
```

项目定位是 5–20 人课题组内部工具。当前不做多租户、SSO、计费、外部数据库或
分布式队列；存储和任务状态保持本地，重活由独立 worker 执行。

### 启动

```bash
conda activate fluxmind
pip install -r requirements-dev.txt
cp .env.example .env

streamlit run app.py
uvicorn api:app --port 18502
python scripts/run_job_worker.py --loop --max-jobs 5
```

首次打开 Streamlit 时创建管理员。管理员可以新增/停用学生账户、重置密码；
每个用户可以载入或清空自己的查询历史。

### 验证

```bash
python -m pytest
python scripts/evaluate_rag.py
python scripts/health_check.py
```

真实图像 provider 和 Docker smoke 必须显式执行，因为它们会使用外部配额或
本机 Docker：

```bash
IMAGE_PROVIDER_BACKEND=openai python scripts/openai_image_smoke.py
CODE_EXECUTION_BACKEND=docker python scripts/docker_execution_smoke.py --language all
```

### 文档

```text
DEVELOPMENT.md                    项目定位、状态、约定与路线图
docs/current/ARCHITECTURE.md      活跃架构与数据流
docs/current/DEPLOYMENT_STATUS.md 生产部署快照
docs/current/REPO_STATUS.md       仓库快照
docs/legacy/                      已废止约束，只读
docs/archive/                     历史审计，只读
```
