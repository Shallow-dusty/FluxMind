# AGENTS.md

本文件为在 FluxMind 仓库工作的 AI agent 提供指引，内容应与
`CLAUDE.md` 保持同步。

FluxMind 是基于 RAG 的控制理论研究 Copilot（滑模控制、PMSM、观测器与
磁链估计）。

## 开始前必读

任何开发工作前必须完整阅读 [DEVELOPMENT.md](DEVELOPMENT.md)。它是项目
定位、功能、架构、开发约定、状态和路线图的单一事实来源。改文档前再读
[docs/README.md](docs/README.md)。

核心原则：

- 功能优先：改动应让用户更容易完成研究任务。
- 项目是 5–20 人内部工具，不是企业级 SaaS。
- 不新增 no-secret 投影层、错误 sanitize、复杂审计或 readiness 面板。
- 保留真正有用的基础防护：参数化 SQL、上传校验、简单权限、请求限流、
  provider 成本上限和代码执行策略。
- secret 只写 `.env`；`.env.example` 只留空模板。
- `docs/legacy/` 与 `docs/archive/` 只读。

## 环境与常用命令

优先使用已有 `.venv`；当前开发机也可使用 conda `fluxmind`：

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt

streamlit run app.py
uvicorn api:app --port 18502
python scripts/run_job_worker.py --loop --max-jobs 5
```

质量门禁：

```bash
python -m pytest
python scripts/evaluate_rag.py
python scripts/health_check.py
```

按需工具：

```bash
python scripts/evaluate_rag.py --json-report artifacts/eval/latest.json
python scripts/evaluate_rag.py --retrieval-url http://127.0.0.1:18502
python scripts/import_seed_papers.py --require-count 52
python scripts/rebuild_seed_index.py --require-count 52
python scripts/runtime_manifest.py --format markdown
IMAGE_PROVIDER_BACKEND=openai python scripts/openai_image_smoke.py
CODE_EXECUTION_BACKEND=docker python scripts/docker_execution_smoke.py --language all
python scripts/deploy_sync.py
```

## 架构要点

`app.py`（Streamlit）和 `api.py`（FastAPI）共用 `src/chain.py`：

```text
hybrid_retrieve()
  ├─ FAISS vector candidates
  ├─ docstore keyword candidates
  └─ source-level manifest metadata
        ↓
rerank_documents()
        ↓
LLM generation + inline [n] citation validation
```

- `query_stream()` 服务 UI 流式回答。
- `query_with_metadata()` 服务 `/query`、`/query/inspect`、`/query/report`。
- `retrieve_with_metadata()` 服务无 LLM 的 `/query/retrieve`。
- 有检索上下文却没有有效 `[n]` 引用时，引用校验必须失败。

活跃模块：

- `src/chain.py`：RAG 核心。
- `src/ingestion.py` / `src/metadata.py`：论文、active set、chunk 和 profile。
- `src/users.py`：本地 admin/student 与个人查询历史。
- `src/jobs.py`：本地 JSONL + SQLite durable job、租约、重试和 worker。
- `src/artifacts.py`：job artifact 注册、完整性和导出。
- `src/providers.py` / `src/capabilities.py`：图像与代码 provider。
- `src/execution_policy.py`：本地代码执行策略。
- `src/provider_guard.py`：可选 provider token/cost 上限。
- `src/runtime.py`：用于真实诊断的运行事件。
- `src/admin.py`：语料/index、job/worker、artifact、用户与近期失败状态。
- `src/storage_manifest.py`：gitignored 运行状态的备份完整性检查。
- `src/config.py`：从项目根 `.env` 读取配置。

不要恢复已删除的 enterprise readiness/rehearsal、workspace registry、
share-link registry、quota/billing ledger 或 platform migration 投影。若将来出现
真实需求，应从用户工作流重新设计，而不是复活旧层。

## 数据与安全边界

运行目录均 gitignored：

```text
papers/
faiss_index/
metadata/
jobs/
artifacts/
```

不要提交 secret、token、上传 PDF、FAISS index、数据库或 job/artifact 内容。
内部环境互信，但所有写路径仍应限定到项目目录；删除必须精确且可解释。

## 测试约定

- 优先测试用户行为和核心数据流。
- 不为理论上的信息泄露或静态源码形状堆测试。
- 改 RAG 时运行离线评估。
- 改 UI 时至少验证首次建管理员、admin 登录、student 权限和查询历史。
- 改 job/provider 时验证同步和异步路径，以及持久化结果。

## 文档维护

- `DEVELOPMENT.md`：定位、状态、约定和路线图。
- `docs/current/ARCHITECTURE.md`：活跃架构。
- `docs/current/DEPLOYMENT_STATUS.md`：生产状态；更新前先跑 health check。
- `docs/current/REPO_STATUS.md`：仓库快照。
- `docs/legacy/`、`docs/archive/`：历史资料，只读。

每个事实只在 owner 文档维护，避免复制大段状态。

## 部署

生产实例位于 Trace-Twin 的 `/opt/fluxmind`，由三个 systemd 服务运行：

```text
fluxmind-ui.service       :18501
fluxmind-api.service      :18502
fluxmind-worker.service
```

Cloudflare Tunnel：

- UI `https://smy.hyper-dusty.cloud/`
- API `https://api-smy.hyper-dusty.cloud/`

`scripts/deploy_sync.py` 默认 dry-run；只有显式 `--apply --restart` 才同步和重启。
