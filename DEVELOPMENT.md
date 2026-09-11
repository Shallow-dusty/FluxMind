# FluxMind 开发指南

> **单一事实来源**：项目定位、功能、架构、开发约定、状态、路线图。
> 替代旧版 CODE_PRINCIPLES / DEVELOPMENT_PLAN / NEXT_STEPS / DISCUSSION / CLEANUP_SUMMARY（已归档至 `docs/legacy/`，只读）。
>
> 最后更新：2026-07-27

---

## 1. 项目定位

FluxMind 是基于 RAG 的控制理论研究 Copilot，聚焦 **滑模控制（SMC）、PMSM 驱动、观测器与磁链估计**。

- **是什么**：把精选研究语料变成可溯源的答案、检索诊断、论文→代码衔接、可执行示例与工件
- **不是什么**：不是企业级 SaaS —— 不做多租户隔离 / SSO / 精确计费 / 公开发售
- **用户场景**：5–20 人实验室/课题组内部使用，互信、非对抗环境
- **部署**：Trace-Twin，UI / API / worker 独立 systemd 服务，Cloudflare Tunnel
  - UI  `https://smy.hyper-dusty.cloud/`
  - API `https://api-smy.hyper-dusty.cloud/`

---

## 2. 功能模块与状态

> 2026-07-14 端到端验证（检索→生成、图像、Docker 均跑通）。

| 模块 | 状态 | 实现要点 |
|------|------|---------|
| RAG 检索 | ✅ | 真实 HuggingFace 语义 embedding + FAISS 向量 / docstore 关键词混合检索 + BM25-lite 重排 + 源级 manifest 增强；52 篇语料 |
| RAG 生成 | ✅ | `ChatOpenAI`（elysiver 中转，主力 `deepseek-v4-pro`、flash fallback），引用编号校验 |
| 图像生成 | ✅ | 显式选择 OpenAI（huyunapi 中转 `gpt-image-2`）或 deterministic mock backend；job→artifact |
| 代码执行 | ✅ | 显式选择 Docker（python:3.11-slim / octave）或 local backend；产物入 artifact |
| 语料库 | ✅ | 52 篇 curated PDF + manifest + 可复现导入脚本（PDF 不入 git，308MB） |
| 作业系统 | ✅ | JSONL + SQLite，即时 runner + durable worker，重试/退避/死信 |
| 工件 | ✅ | artifact 注册表 + 导出 |
| 多用户 / 查询历史 | ✅ | 本地 admin/student 登录、管理员账户管理、按用户查询历史；API 生成类入口可显式记录/读取历史 |
| 任务 / 工件归属 | ✅ | UI 任务绑定当前账户；学生只看自己的任务与工件，管理员可总览；下载使用可读文件名 |
| 引用标注 | ✅ | 提示词明确要求内联 `[n]`；有检索上下文却无有效编号时校验失败；中英文及流式入口已验证 |

---

## 3. 架构概要

双入口共用一个 RAG 核心：

```
Streamlit UI (app.py, :18501)      FastAPI (api.py, :18502)
              \                    /
               \                  /
              src/chain.py  (RAG 核心)
              ├─ hybrid_retrieve      (FAISS 向量 + docstore 关键词)
              ├─ rerank_documents     (BM25-lite；RERANKER_MODEL 指向本地路径时用 CrossEncoder)
              ├─ 引用校验 / citation guard
              ├─ query() / query_stream() / query_with_metadata() / retrieve_with_metadata()
              └─ 源级 manifest 检索增强

本地平台层：jobs.py / metadata.py / artifacts.py / runtime 事件
配置：     src/config.py（从 .env 加载）
```

当前平台层只服务本地小组运行：SQLite/JSONL 作业、语料与工件存储，
本地账户，以及可选的共享 API token。外部数据库、对象存储、分布式队列、
外部身份和支付不在当前产品范围。

---

## 4. 技术栈

- Python 3.11+（当前开发环境为 conda `fluxmind` / Python 3.12）
- UI：Streamlit；API：FastAPI
- RAG：LangChain + FAISS + `langchain_openai.ChatOpenAI`
- Provider：OpenAI SDK（图像生成 + streaming）
- 存储：SQLite（用户 / 作业 / 语料 / 工件）+ 少量 JSON/JSONL
- 执行：Docker（Python / Octave 沙箱）
- 测试：pytest

---

## 5. 开发约定（基于功能实现要求）

> 旧版"防御代码冻结 / 200 行红线 / 1:1.5 比例 / 红线警报"等约束**已废止**。
> 本约定围绕"让 5–20 人更高效做控制理论研究"这一功能目标。

### 5.1 功能优先
- 每个改动回答："这能让用户更容易完成研究任务吗？"
- 用户可见功能优先；性能 / 重构 / 覆盖率为次，除非阻碍功能

### 5.2 代码质量
- 清晰、可维护、命名准确
- 不强加行数红线；但单文件过大应考虑按真实职责拆分（参考 §8 平台边界）
- **不过度防御**：内部互信环境，不再添加 no-secret 投影层 / 错误 sanitize / 复杂审计追踪
- 基础防护保留：SQL 参数化、上传校验、简单权限、请求限流

### 5.3 测试
- 功能行为测试为主（确保功能正确）
- 不为理论上的"信息泄露"写测试；文档审计类测试精简
- 测试代码量合理即可，无硬性比例要求

### 5.4 配置与凭据
- **配置文件优于环境变量**（区分同源不同用处、便于配额统计）
- secret 写入 `.env`（gitignored），`.env.example` 留模板（key 留空）
- 当前凭据（见 `.env`，不入库）：
  - LLM 生成：elysiver 中转（别名"烁"）`https://elysiver.h-e.top/v1`，主力 `deepseek-v4-pro`（质量严谨，~18s）；辅助 `deepseek-v4-flash`（小任务/辅助/最终兜底，~13s）
  - 图像：huyunapi 中转 `https://www.huyunapi.com/v1`，`gpt-image-2`，`quality=low`
  - 注意：中转对 `output_format` 参数会挂起 → `OPENAI_IMAGE_SEND_OUTPUT_FORMAT=false`
  - ⚠️ 中转 prompt 会过境第三方；控制框图 prompt 无敏感数据，可接受

### 5.5 提交
- 一功能一 commit，message 清晰（`feat` / `fix` / `docs` / `chore`）
- 不提交 secret / `.env` / PDF / FAISS index / metadata 与 job 内容

### 5.6 文档
- 单一事实来源：本文件为开发入口
- 部署状态：`docs/current/DEPLOYMENT_STATUS.md`（人工维护）
- 架构：`docs/current/ARCHITECTURE.md`
- 历史参考：`docs/legacy/`（只读）

---

## 6. 运行与验证

```bash
conda activate fluxmind

# 三进程
streamlit run app.py                       # UI
uvicorn api:app --port 18502               # API
python scripts/run_job_worker.py --loop --max-jobs 5   # worker

# 质量门禁
python -m pytest                           # 测试
python scripts/evaluate_rag.py             # 离线 RAG 基线（无网络）
python scripts/health_check.py             # 核心导入、语料/index、用户库与可选 HTTP/SSH 运行检查

# 真实 provider smoke（需对应 key 在 .env）
IMAGE_PROVIDER_BACKEND=openai python scripts/openai_image_smoke.py
CODE_EXECUTION_BACKEND=docker python scripts/docker_execution_smoke.py --language all

# 语料重建（PDF 经 .gitignore 排除，靠脚本复现）
python scripts/import_seed_papers.py --require-count 52
python scripts/rebuild_seed_index.py --require-count 52
```

---

## 7. 路线图（里程碑驱动，不绑死日期）

- **[M1] RAG 端到端可用** ✅ 检索+生成（2026-07-14 验证）
- **[M2] 引用标注优化** ✅ 答案稳定带 `[n]` 编号 + 零引用校验生效（2026-07-27 验证）
- **[M3] 多用户** ✅ 简单 admin/student + 按用户查询历史 + 任务/工件归属（2026-07-27 验证）
- **[M4] 文档/测试与代码对齐** ✅ 静态源码审计改为真实运行检查，活跃文档同步（2026-07-27）
- **[M4.1] 历史平台瘦身** ✅ 删除 readiness/rehearsal、旧 registry/RBAC/quota 与迁移投影；admin 状态收敛到真实运行问题（2026-07-27）
- **[M4.2] 任务结果可见性** ✅ job/artifact API 与 UI 直接展示代码、stdout/stderr、错误、提示与 provider 元数据；支持按研究内容搜索（2026-07-27）
- **[M4.3] 真实 UI 收口** ✅ 首次管理员、管理员完整页和学生页实跑；修复旧翻译键/保留预览结构，删除废弃的浏览器脚本注入与残余 presence/no-secret 字段（2026-07-27）
- **[M4.4] 真实失败语义** ✅ 空语料、embedding 加载、执行管道与 readiness 直接失败并给出原因；删除伪文档、哈希 embedding 和测试专用输出回退（2026-07-27）
- **[M4.5] 死代码收口** ✅ 删除旧 workspace/quota 请求字段、空 API-key owner 上下文，以及无调用方的 BM25/job/artifact/Docker 状态包装；高置信度静态死代码扫描无候选（2026-07-27）
- **[M5] 生产同步** Trace-Twin 更新 source tree + 初始化生产管理员
- **[M6] 真实用户试用** 邀请 3–5 人 + 收集反馈

---

## 8. 平台边界

2026-07-27 已删除不符合当前产品定位的历史实现：

- activation/readiness/rehearsal 聚合面
- 旧 API-key/product/share-link registry
- workspace RBAC、quota/billing ledger
- platform migration/schema 投影
- 对应 API、Streamlit 管理面、CLI 与防御性测试

保留的基础边界都有直接用途：

- `provider_guard.py`：可选的真实 provider token/cost 上限
- `storage_manifest.py`：gitignored 运行数据的备份完整性检查
- `admin.py`：语料/index、job/worker、artifact、用户与近期失败的运行状态

**活跃核心**：`chain` / `ingestion` / `embeddings` / `metadata` / `users` /
`jobs` / `providers` / `capabilities` / `execution_policy` / `artifacts` /
`runtime` / `admin` / `storage_manifest` / `provider_guard` / `costs` /
`evaluation` / `config`

---

## 9. 外部依赖

| 用途 | 来源 | 说明 |
|------|------|------|
| LLM 生成 | elysiver 中转（别名"烁"） | 主力 `deepseek-v4-pro`；辅助 `deepseek-v4-flash`（兜底） |
| 图像生成 | huyunapi 中转 | `gpt-image-2`，`quality=low`，`send_output_format=false` |
| Docker Python | `python:3.11-slim` | 生产用 `m.daocloud.io` 镜像加速 |
| Docker Octave | `fluxmind/octave:trixie-slim` | Trace-Twin 本地构建（`deploy/docker/octave-trixie-slim.Dockerfile`） |
