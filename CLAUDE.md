# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

FluxMind 是基于 RAG 的控制理论研究 Copilot（滑模控制 SMC + 磁链估计）。

> **AGENTS.md 与本文件应保持同步**——它面向其他 agent，内容应与此处一致。

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
python scripts/storage_schema.py --format markdown   # 存储 schema 漂移检测（drift 时非零退出）
python scripts/provider_readiness.py --format markdown # 外部 provider/MATLAB activation readiness
python scripts/deploy_sync.py                # rsync 部署（默认 dry-run，--apply --restart 才真正执行）
```

## 架构要点（需读多文件才能理解的部分）

**双入口、共用一个 RAG 核心。** `app.py`（Streamlit UI）和 `api.py`（FastAPI）都调用 `src/chain.py`。chain 做检索 + 生成：`hybrid_retrieve()`（FAISS 向量池 + docstore 关键词）→ `rerank_documents()`（确定性 BM25-lite，无密钥；`RERANKER_MODEL` 指向本地路径时才用 CrossEncoder）→ 编号引用校验。`query_stream()` 供 UI 流式，`query()`/`query_with_metadata()`/`retrieve_with_metadata()` 供 API 的 `/query`、`/query/inspect`、`/query/retrieve`。

**"本地无密钥（no-key / no-secret）"是核心架构约束，不是临时状态。** 所有重活——图像生成、代码执行、向量存储、metadata、job 队列——都通过 provider-neutral 接口 + 本地 mock/真实实现完成，**不依赖任何外部 key**。真实外部 provider（图像、托管沙箱、真 MATLAB、多租户身份/配额/计费）被刻意禁用，只在 admin 里以"配置/可用"布尔位与 reason code 体现。新功能必须沿这条路走：接口 + 本地实现 + fixture + runtime flag，**绝不**把外部依赖硬塞进 UI 进程或同步 `/query` 路径。

**Job / Storage / Artifact 子系统是为"未来迁移到分布式平台"预留的本地形态。** 关键模块：
- `src/jobs.py` — append-only JSONL（`jobs/jobs.jsonl`）+ SQLite 当前态镜像（`jobs/jobs.sqlite3`）；即时 runner、进程内队列、显式 durable worker；幂等键、租约（lease）、重试/退避、死信、deadline。这是**本地**契约，不是分布式队列。
- `src/metadata.py` — `metadata/corpus.json`（+ `corpus.sqlite3`）语料注册表、`chunks.sqlite3` chunk 镜像、`corpus_profiles.json` 可复用选集。JSON 写入走同目录临时文件 + 原子替换。
- `src/artifacts.py` — artifact 注册表（`artifacts/artifacts.sqlite3`）+ 安全导出；只导出落在 `ARTIFACTS_DIR` 下的 `file://` artifact。
- `src/providers.py` + `src/capabilities.py` + `src/execution_policy.py` — 执行/图像 provider 与契约；执行前 `local-safe-v1` 策略用 `ast` 校验 Python、import 白名单、拦截 shell/绝对路径。`CODE_EXECUTION_BACKEND=local|docker`。
- `src/product_readiness.py` + `src/provider_readiness.py` — no-secret activation readiness：前者覆盖身份/API-key/配额/计费，后者覆盖外部图像 provider、托管执行、MATLAB backend 和 provider quota guard。默认只报告 blocker code，不启用外部调用。
- `src/admin.py`（115KB，最大模块）— 聚合上述所有状态为 `/admin/status`、`/admin/status/report`(Markdown)、`/admin/metrics`(Prometheus)、`/admin/events`。全部 no-secret：只出计数/布尔/code，绝不返回 prompt、答案、源路径、owner ID、文件内容。

**runtime event 是横切观测层。** `metadata/runtime_events.jsonl` 记录 `retrieval_trace`、`query_usage`、`code_execution`、`upload_scan`、`provider_failure` 等元数据-only 事件；admin/Streamlit/metrics 从中派生 advisory alert。增量功能若产生可观测行为，应追加对应 event（同样不得含敏感内容）。

**配置全在 `src/config.py`**（从 `.env` 读，`PROJECT_ROOT` 锚定）。RAG 参数硬编码：`CHUNK_SIZE=1000`、`CHUNK_OVERLAP=200`、`TOP_K=5`。运行时目录（`papers/`、`faiss_index/`、`artifacts/`、`jobs/`、`metadata/`）均 gitignored。

## 文档纪律（重要）

`docs/` 严格遵循**单一事实来源**：每个事实只在其 owner 文档里写，不跨文件复制。改动前先读 `docs/README.md` 的 Source-Of-Truth Map。有专门测试守护文档一致性（`tests/test_docs_status.py`、`test_feature_audit_docs.py`、`test_translation_guard.py`），改文档可能需同步改测试。

- 架构/模块边界 → `docs/ARCHITECTURE.md`（深度参考，本文件是其摘要）
- git/worktree 快照 → `docs/REPO_STATUS.md`
- 生产部署快照 → `docs/DEPLOYMENT_STATUS.md`（**依赖 live 服务的事实，必须先跑 health check 再更新并记录时间**）
- 工作包/验收 → `docs/BACKLOG.md`
- 严禁把 secret、token、上传的 PDF、FAISS 索引、metadata/job/artifact 内容写进任何文档。

## 部署

生产实例跑在 **Trace-Twin（06 项目）** 的 `/opt/fluxmind`，UI/API 各自独立 systemd 服务（端口 18501/18502），worker 见 `deploy/systemd/fluxmind-worker.service`。Cloudflare Tunnel 暴露：`https://smy.hyper-dusty.cloud/`（UI）、`https://api-smy.hyper-dusty.cloud/`（API）。

## Workspace / Linear

正式编号 `11.FluxMind`，Public 仓库。无 Linear（独立个人项目）。
