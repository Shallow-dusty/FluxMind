# AGENTS.md

本文件为在此仓库工作的 AI agent（Codex 等）提供指引，内容应与 `CLAUDE.md` 保持同步。

FluxMind 是基于 RAG 的控制理论研究 Copilot（滑模控制 SMC + 磁链估计）。

## ⚠️ 重要：先读开发指南

**在开始任何开发工作前，必须先阅读 [DEVELOPMENT.md](DEVELOPMENT.md)**（项目定位、功能、架构、开发约定、状态、路线图的单一事实来源）。

**核心约定**（详见 DEVELOPMENT.md）：
- 功能优先：每个改动回答"这能让用户更容易完成研究任务吗？"
- 不过度防御：内部互信环境，不再添加 no-secret 投影层 / 错误 sanitize / 复杂审计
- 配置：secret 写入 `.env`（gitignored），`.env.example` 留模板
- 旧的防御冻结/红线/比例约束已废止，归档于 `docs/legacy/`（只读）

**项目定位**：轻量级多用户工具（5-20人内部使用），不是企业级 SaaS。

## 常用命令

```bash
# 环境（优先用已存在的 .venv；否则 conda create -n fluxmind python=3.11）
source .venv/bin/activate
pip install -r requirements-dev.txt        # = requirements.txt + pytest

# 运行两个入口（生产端口为 18501/18502，本地默认 8501）
streamlit run app.py                        # Streamlit UI
uvicorn api:app --port 18502                # FastAPI（/query、/jobs、/corpus、/admin/* 等）
python scripts/run_job_worker.py --loop --max-jobs 5   # 显式 durable worker（生产用 --forever）

# 测试
python -m pytest                            # 全量
python -m pytest tests/test_jobs.py         # 单文件
python -m pytest tests/test_jobs.py -k retry  # 按名筛选单测

# CI 门禁（.github/workflows/ci.yml 会依次跑这三个）
python -m pytest
python scripts/evaluate_rag.py              # 离线 RAG 基线（无网络）
python scripts/health_check.py              # 本地运行时检查

# 其他门禁/工具
python scripts/evaluate_rag.py --json-report artifacts/eval/latest.json
python scripts/evaluate_rag.py --retrieval-url http://127.0.0.1:18502  # 调用 /query/retrieve 评检索
python scripts/import_seed_papers.py --require-count 52 # 导入 curated open-access 论文库种子
python scripts/rebuild_seed_index.py --require-count 52 # 用内置论文库重建本地 FAISS index
python scripts/storage_schema.py --format markdown   # 存储 schema 漂移检测（drift 时非零退出）
python scripts/api_key_registry.py status --format markdown # 本地 API key registry no-secret 状态
python scripts/share_link_registry.py status --format markdown # 本地 share-link token registry no-secret 状态
python scripts/product_registry.py status --format markdown # 本地 user/workspace/RBAC/quota/billing ledger 与 guard 状态
python scripts/product_readiness.py --format markdown # identity/quota/billing product readiness
python scripts/product_activation_rehearsal.py --format markdown --require-activation # 本地 SQLite product activation 演练
python scripts/collaboration_readiness.py --format markdown # 私有语料/share-link collaboration readiness
python scripts/provider_readiness.py --format markdown # 外部 provider/MATLAB activation readiness
python scripts/provider_runtime_rehearsal.py --format markdown --require-local-foundation # 本地 provider runtime 合约演练
IMAGE_PROVIDER_BACKEND=openai python scripts/openai_image_smoke.py # 显式 OpenAI 图像生成 smoke（需 key）
CODE_EXECUTION_BACKEND=docker python scripts/docker_execution_smoke.py --language all # 显式 Docker Python/Octave 执行 smoke
python scripts/quality_readiness.py --format markdown # self-use/small-group/community 质量 readiness
python scripts/activation_suite.py --format markdown --require-target local_foundation # 聚合本地激活演练
python scripts/openapi_contract.py --format markdown --require-local-contract # no-secret OpenAPI 合约检查
python scripts/openapi_contract.py --verify-snapshot /tmp/fluxmind-openapi-contract.json --require-no-drift # 校验 no-secret OpenAPI 合约快照漂移
python scripts/platform_migration_rehearsal.py --include-object-manifest --format markdown # 本地迁移演练 + opaque object-storage 清单
python scripts/platform_migration_rehearsal.py --include-object-manifest --output /tmp/fluxmind-object-manifest.json # 写入可校验 JSON 清单
python scripts/platform_migration_rehearsal.py --verify-object-manifest /tmp/fluxmind-object-manifest.json --format markdown # 校验 rehearsal/object 清单
python scripts/platform_migration_rehearsal.py --include-object-manifest --include-job-store-manifest --output /tmp/fluxmind-object-and-job-rehearsal.json # 写入 object + job-store 清单
python scripts/platform_migration_rehearsal.py --verify-job-store-manifest /tmp/fluxmind-object-and-job-rehearsal.json --format markdown # 校验 durable job-store 清单
python scripts/deploy_sync.py                # rsync 部署（默认 dry-run，--apply --restart 才真正执行）
```

## 架构要点（需读多文件才能理解的部分）

**双入口、共用一个 RAG 核心。** `app.py`（Streamlit UI）和 `api.py`（FastAPI）都调用 `src/chain.py`。chain 做检索 + 生成：`hybrid_retrieve()`（FAISS 向量池 + docstore 关键词）→ `rerank_documents()`（确定性 BM25-lite，无密钥；`RERANKER_MODEL` 指向本地路径时才用 CrossEncoder）→ 编号引用校验。`query_stream()` 供 UI 流式，`query()`/`query_with_metadata()`/`retrieve_with_metadata()` 供 API 的 `/query`、`/query/inspect`、`/query/retrieve`。

**运行时默认保持 no-key/local，但后续开发以真实用户功能为先。** 现有 provider-neutral、job、artifact、metadata 层可以继续复用；但不要再把"no-secret"扩展成新的投影层、sanitize/redact 代码或企业级 readiness 面板。真实 OpenAI 图像生成、Docker 执行等用户可见功能可以通过显式配置、job-backed provider 和清晰错误提示接入；外部 key 不写入仓库，也不要把重活塞进 UI 进程或同步 `/query` 路径。

**Job / Storage / Artifact 子系统是为"未来迁移到分布式平台"预留的本地形态。** 关键模块：
以下模块说明是现有系统地图，不是下一阶段待扩展清单；除非 Priority 1-3 功能确实需要，避免继续扩展 readiness/no-secret/审计类表面。
- `src/jobs.py` — append-only JSONL（`jobs/jobs.jsonl`）+ SQLite 当前态镜像（`jobs/jobs.sqlite3`）；即时 runner、进程内队列、显式 durable worker；幂等键、租约（lease）、重试/退避、死信、deadline；`GET /jobs` 和 Streamlit 最近任务面板
  只返回 no-secret summary 并用安全投影搜索，详情走精确 `GET /jobs/{job_id}`。这是**本地**契约，不是分布式队列。
- `src/metadata.py` — `metadata/corpus.json`（+ `corpus.sqlite3`）语料注册表、`chunks.sqlite3` chunk 镜像、`corpus_profiles.json` 可复用选集。JSON 写入走同目录临时文件 + 原子替换。
- `src/artifacts.py` — artifact 注册表（`artifacts/artifacts.sqlite3`）+ 公共 no-secret 投影 + 安全导出；只导出落在 `ARTIFACTS_DIR` 下的 `file://` artifact，API/UI/RAG 不展示 raw URI、路径、owner、prompt 或 source reference。
- `src/api_keys.py` — 可选本地 SQLite API key 生命周期 registry；只持久化 token hash，原始 token 仅在创建时输出一次，API 鉴权可在 `FLUXMIND_API_KEY_REGISTRY_BACKEND=sqlite` 时使用它。
- `src/product_registry.py` — 可选本地 SQLite product registry；提供 user/workspace/RBAC/quota/usage/billing attribution ledger、本地 `/admin/product-registry/*` 管理 API 和 Streamlit operator 面板，在显式启用 `FLUXMIND_STREAMLIT_PRODUCT_REGISTRY_MANAGEMENT_ENABLED=true` 时才开放 Streamlit 直写管理面，operator 错误输出走 no-secret redaction 边界；在显式启用 `FLUXMIND_PRODUCT_QUOTA_GUARD_ENABLED=true` 时供 `/query*` 路径做本地 quota guard，在显式启用 `FLUXMIND_PRODUCT_RBAC_GUARD_ENABLED=true` 时按 workspace role 守护查询、job/corpus/admin 写路径；不连接外部身份或支付系统。
- `src/share_links.py` — 可选本地 SQLite share-link token registry；只持久化 token hash，创建时一次性返回 raw token，list/revoke/resolve/API 事件只输出 no-secret summary、presence/fingerprint、计数和布尔位，不导出 URL、resource ref、creator user ID、description、路径或内容；Streamlit 管理面还需要显式 `FLUXMIND_STREAMLIT_SHARE_LINK_MANAGEMENT_ENABLED=true`。
- `src/storage_migration.py` — 本地 runtime migration rehearsal；可选生成并校验 opaque object-storage migration manifest（object key/hash/byte count/group/token only）和 durable job-store migration manifest（job/idempotency claim token + aggregate metadata only），不输出源路径、文件名、bucket、endpoint、credential、job payload、owner ID、request ID、worker ID、idempotency key 或内容。
- `src/providers.py` + `src/capabilities.py` + `src/execution_policy.py` — 执行/图像 provider 与契约；执行前 `local-safe-v1` 策略用 `ast` 校验 Python、import 白名单、拦截 shell/绝对路径。`CODE_EXECUTION_BACKEND=local|docker`。
- `src/product_readiness.py` + `src/provider_readiness.py` — no-secret activation readiness：前者覆盖身份/API-key/RBAC/配额/计费并能检查本地 SQLite key registry、product registry、本地 quota guard 与 RBAC guard，后者覆盖外部图像 provider、托管执行、MATLAB backend 和 provider quota guard。默认只报告 blocker code，不启用外部调用。
- `src/provider_guard.py` — no-secret provider 调用前 quota/cost guard；默认关闭，启用后在 LLM/provider client 构造前检查估算 prompt tokens、请求 completion tokens 和可选 cost ceiling，只输出计数、阈值和 reason code。
- `src/product_activation_rehearsal.py` + `src/provider_runtime_rehearsal.py` — disposable no-secret rehearsal：前者演练本地 SQLite API key/product registry/RBAC/跨 workspace 隔离/quota/billing attribution activation，后者演练 mock image、local Python、Octave 分支、执行 abuse-policy denial、Docker readiness、provider quota/cost guard 和 provider local foundation；都不导出 raw token、路径或外部账号。
- `src/collaboration_readiness.py` — no-secret collaboration readiness：私有语料和 share links 默认关闭；上线前检查 product registry、RBAC guard、本地 share-link token registry 和角色矩阵，只输出 role/reason/count/boolean，不导出 workspace/user/corpus/share 标识或 URL。
- `src/quality_readiness.py` — no-secret 质量成熟度 readiness：复用 `eval/rag_baseline.json` 的 `quality_maturity_targets`，可合并显式传入的 no-secret live eval report 中的 live retrieval/live answer 数量、通过率和 live answer 术语覆盖，区分 self-use、small-group、community 缺口，并输出 per-target current/expected/gap 摘要。
- `src/activation_suite.py` — no-secret 聚合入口：复用 product activation、collaboration readiness、provider runtime、job-store migration manifest、quality readiness 的摘要；CLI/API/UI 传入生成的 OpenAPI schema 时也把 OpenAPI contract 纳入 local foundation gate；输出 local foundation/full activation/small-group/community gate，不包含 raw 子报告、token、路径、payload 或外部账号。
- `src/openapi_contract.py` — no-secret OpenAPI contract readiness：检查 required route/method、operation summary/id、responses、protected auth header 声明和 route-group 覆盖，生成 stable operation fingerprint，并可用旧 no-secret JSON 报告校验 snapshot drift；FastAPI/Streamlit/CLI 只输出摘要，不导出 raw schema。
- `src/admin.py`（最大模块）— 聚合上述所有状态为 `/admin/status`、`/admin/status/report`(Markdown)、`/admin/metrics`(Prometheus)、`/admin/events`。全部 no-secret：只出计数/布尔/code，绝不返回 prompt、答案、源路径、owner ID、文件内容。

**runtime event 是已有观测层，不是新增审计任务的默认理由。** `metadata/runtime_events.jsonl` 记录 `retrieval_trace`、`query_usage`、`code_execution`、`upload_scan`、`provider_failure` 等事件；admin/Streamlit/metrics 从中派生 advisory alert。增量功能只有在对调试、用户反馈或运行稳定性有实际价值时才追加事件，不再为了理论上的 no-secret 审计覆盖而扩展事件面。

**配置全在 `src/config.py`**（从 `.env` 读，`PROJECT_ROOT` 锚定）。RAG 参数硬编码：`CHUNK_SIZE=1000`、`CHUNK_OVERLAP=200`、`TOP_K=5`。运行时目录（`papers/`、`faiss_index/`、`artifacts/`、`jobs/`、`metadata/`）均 gitignored。

## 文档结构（2026-06-21 整理后）

### 顶层指导文档
- **DEVELOPMENT.md** - 开发指南（定位/功能/架构/约定/状态/路线图，单一入口）
- `docs/legacy/` - 旧约束文档归档（CODE_PRINCIPLES/DEVELOPMENT_PLAN/NEXT_STEPS/DISCUSSION/CLEANUP_SUMMARY，只读）

### docs/ 目录结构
- **docs/README.md** - 文档导航索引
- **docs/current/** - 活跃维护的文档
  - `ARCHITECTURE.md` - 系统架构设计
  - `DEPLOYMENT_STATUS.md` - 生产部署状态（更新前需先跑 health check）
  - `REPO_STATUS.md` - Git 仓库快照
- **docs/archive/** - 历史参考文档（只读）
  - `BACKLOG.md` - 过去的待办清单
  - `FEATURE_AUDIT.md` - 功能审计
  - `PLATFORM_AUDIT_AND_ROADMAP.md` - 平台审计
  - `PRODUCTION_GAP_AND_MARKET_RESEARCH.md` - 生产差距分析
  - `QUALITY_ROADMAP.md` - 质量路线图

### 文档维护原则
- 单一事实来源：每个事实只在其 owner 文档中写
- 改动前先读 `docs/README.md`
- 严禁写入 secret、token、上传的 PDF、FAISS 索引、metadata/job/artifact 内容

## 部署

生产实例跑在 **Trace-Twin（06 项目）** 的 `/opt/fluxmind`，UI/API 各自独立 systemd 服务（端口 18501/18502），worker 见 `deploy/systemd/fluxmind-worker.service`。Cloudflare Tunnel 暴露：`https://smy.hyper-dusty.cloud/`（UI）、`https://api-smy.hyper-dusty.cloud/`（API）。

## Workspace / Linear

正式编号 `11.FluxMind`，Public 仓库。无 Linear（独立个人项目）。
