# FluxMind 下一步行动清单

**生成时间**: 2026-06-21  
**状态**: Docker 执行后端已在 Trace-Twin 启用并验证，当前改动待提交

---

## ✅ 已完成

1. **文档整理**
   - 活跃文档 → `docs/current/`
   - 历史文档 → `docs/archive/`
   - 更新导航 → `docs/README.md`
   - Commit: 173b546

2. **指导文档创建**
   - `CODE_PRINCIPLES.md` - 开发原则和防御代码冻结
   - `DEVELOPMENT_PLAN.md` - 3周详细计划
   - `DISCUSSION.md` - 项目分析和决策记录
   - Commit: ed56760

3. **Git 状态**
   - 当前 HEAD: `1b59d1e docs: align FluxMind refocus status`
   - `main` 相对 `origin/main`：ahead 1 / behind 0
   - 工作区包含 Docker 执行后端、生产启用记录和可复现镜像构建文件，待提交

4. **Docker 代码执行生产启用**
   - Trace-Twin `/opt/fluxmind` 已启用 `CODE_EXECUTION_BACKEND=docker`
   - API/UI/worker systemd 服务通过 `SupplementaryGroups=docker` 访问 Docker daemon
   - Python 镜像：`m.daocloud.io/docker.io/library/python:3.11-slim`
   - Octave 镜像：`fluxmind/octave:trixie-slim`
   - Python/Octave smoke 和 `/jobs/code/*-docker` API smoke 均通过

---

## 🚀 立即行动（今天）

### 1. 提交当前 Docker 执行后端切片

当前有新改动，应该先提交 Docker execution slice，再进入下一项功能开发。

### 2. 进入下一个功能边界

Docker 代码执行已在 Trace-Twin 验证通过。下一步进入 `DEVELOPMENT_PLAN.md`
后续 Priority 1-3 用户可见功能，优先选择论文库或图像生成改进。

保留 Docker 回归命令：

```bash
cd /home/shallow/04.AI-Prism/11.FluxMind
CODE_EXECUTION_BACKEND=docker python scripts/docker_execution_smoke.py --language all
python -m pytest tests/test_providers.py tests/test_jobs.py tests/test_docker_execution_smoke.py tests/test_api_docker_execution.py
```

完成标准已达成：Python/Octave 代码能在 Docker 后端执行，生成文件能进入 artifact，失败时错误信息清晰。

生产启用时，API/UI/worker 三类进程都可能触发代码执行；需要让对应 systemd 服务的运行用户能访问 Docker daemon。参考：

```text
deploy/systemd/fluxmind-docker-execution.dropin.example.conf
```

最短验证顺序：

```bash
# 1. 确认 Docker daemon 对当前用户可用
docker --version
docker ps

# 2. 验证 provider/artifact 路径
CODE_EXECUTION_BACKEND=docker python scripts/docker_execution_smoke.py --language python
CODE_EXECUTION_BACKEND=docker python scripts/docker_execution_smoke.py --language octave

# 3. 如默认镜像不可用，只替换 smoke 镜像先定位问题
python scripts/docker_execution_smoke.py --language octave --octave-image fluxmind/octave:trixie-slim

# 4. 如需绕过 job 层直接调试 provider，可显式指定 provider mode
python scripts/docker_execution_smoke.py --mode provider --language python
```

注意：`/admin/status` 中的 Docker available 只表示 Docker daemon 对运行用户可达，不等于 Python/Octave 镜像都已拉取并能执行；镜像和 artifact 路径必须用 `scripts/docker_execution_smoke.py` 验证。

API 入口：

```text
POST /jobs/code/python-docker
POST /jobs/async/code/python-docker
POST /jobs/code/octave-docker
POST /jobs/async/code/octave-docker
```

这些 Docker API 入口要求 `CODE_EXECUTION_BACKEND=docker`；否则返回 `409` 并提示当前 backend，避免误把 local 执行当作 Docker 执行。

常见失败码：

```text
docker_container_start_failed  Docker daemon/权限/镜像启动问题
docker_container_run_denied    Docker 拒绝运行容器
runtime_unavailable            容器内缺少 python/octave runtime
execution_timeout              执行超时
execution_failed               用户代码自身失败
```

提交前最小检查：

```bash
python -m pytest tests/test_providers.py tests/test_jobs.py tests/test_docker_execution_smoke.py tests/test_api_docker_execution.py
CODE_EXECUTION_BACKEND=docker python scripts/docker_execution_smoke.py --language python
CODE_EXECUTION_BACKEND=docker python scripts/docker_execution_smoke.py --language octave
```

只有上述检查覆盖到 Docker job/provider/artifact 边界后，才把本轮 Docker 执行后端改动标记为完成。

---

## 📋 开发执行顺序

### Week 1 (2026-06-21 - 2026-06-28)

**Day 1-2: Docker 代码执行**
```bash
# 目标：替换本地 subprocess/mock，使用真实 Docker
cd /home/shallow/04.AI-Prism/11.FluxMind

# 检查 Docker 可用性
docker --version
docker ps

# 启动开发
# 参考：DEVELOPMENT_PLAN.md "Day 1-2" 部分
```

**Day 3-4: OpenAI 图像生成**
```bash
# 目标：接入 OpenAI 图像 API，生成真实控制系统图
# 需要：OpenAI API Key

# 参考：DEVELOPMENT_PLAN.md "Day 3-4" 部分
```

**Day 5: 简化用户管理**
```bash
# 目标：简单的 admin/student 区分
# 参考：DEVELOPMENT_PLAN.md "Day 5" 部分
```

Day 1-5 完成前，不新增 sanitize/redact/no-secret 投影类工作。

---

## 📖 Codex 使用指南

### 启动新任务时

1. **先读指导文档**
   ```
   请先阅读 CODE_PRINCIPLES.md 和 DEVELOPMENT_PLAN.md
   ```

2. **设置明确目标**
   ```
   /goal 接入 OpenAI 图像生成，能生成 PMSM 控制框图
   ```

3. **定期检查进度**
   - 每天下班前：检查是否偏离计划
   - 每周五：回顾本周完成情况

### 遇到"是否添加安全措施"时

**自问清单**（参考 CODE_PRINCIPLES.md）：
- [ ] 这是 5-20 个互信用户的真实威胁吗？
- [ ] 这能防止什么实际问题？
- [ ] 实现成本是否超过 100 行代码？

**如果任何一项答案是"否"或"不确定" → 不添加**

---

## 🎯 成功标准（3 周后验证）

### 必须完成
- [ ] Docker 代码执行可用
- [ ] OpenAI 图像生成可用
- [ ] 论文库 ≥ 50 篇
- [ ] 基础用户登录和权限
- [ ] 查询历史可用

### 过程指标
- [ ] 没有新的"sanitize/redact"提交
- [ ] 每周至少 1 个用户可见功能
- [ ] 代码量增长 < 20%

### 用户指标
- [ ] 至少 5 个真实用户
- [ ] 用户反馈"有用"
- [ ] 每天有查询活动

---

## 🚨 红线警报

**如果出现以下情况，立即停止**：

1. ❌ 连续 3 天没有功能进展
2. ❌ 又开始写"fix: sanitize XX"
3. ❌ 测试代码超过功能代码 3 倍
4. ❌ 纠结"是否需要 no-secret 投影"

**停止后的行动**：
1. 重新阅读 `CODE_PRINCIPLES.md`
2. 检查是否在做 Priority 1-3 任务
3. 如果不是，立即切换回去

---

## 📞 检查点

### 第一周末（2026-06-28 周五）
- [ ] Docker 执行演示成功
- [ ] OpenAI 图像演示成功
- [ ] 基础登录演示成功

### 第二周末（2026-07-05 周五）
- [ ] 论文库 ≥ 50 篇
- [ ] 检索质量测试通过
- [ ] 生产环境更新索引

### 第三周末（2026-07-12 周五）
- [ ] 5+ 真实用户试用
- [ ] 收集反馈
- [ ] 规划下一阶段

---

## 📚 文档结构（整理后）

```
FluxMind/
├── README.md                 # 项目入口
├── CLAUDE.md                 # Claude Code 指南
├── CODE_PRINCIPLES.md        # ⭐ 开发原则（必读）
├── DEVELOPMENT_PLAN.md       # ⭐ 3周计划（必读）
├── DISCUSSION.md             # 决策记录
├── NEXT_STEPS.md            # 本文件
│
└── docs/
    ├── README.md             # 文档导航
    ├── current/              # 活跃文档
    │   ├── ARCHITECTURE.md
    │   ├── DEPLOYMENT_STATUS.md
    │   └── REPO_STATUS.md
    └── archive/              # 历史文档
        ├── BACKLOG.md
        ├── FEATURE_AUDIT.md
        └── ...
```

---

## 💡 最后提醒

### 给 Codex 的话

**你的工作不是**：
- ❌ 追求完美的代码覆盖率
- ❌ 防范所有理论上的风险
- ❌ 构建企业级 SaaS 平台

**你的工作是**：
- ✅ 让 5-20 个研究人员更高效
- ✅ 提供真实可用的功能
- ✅ 保持代码简单可维护

### 给 Shallow 的话

如果感觉 Codex 又走偏了：
1. 让它重读 `CODE_PRINCIPLES.md`
2. 问它："这是 Priority 1-3 的任务吗？"
3. 如果不是，让它停下来

---

**祝开发顺利！🚀**

## Docker 镜像策略（Trace-Twin）

Trace-Twin 在杭州，Docker Hub/GHCR 大镜像直连超时属于预期网络条件，不作为功能失败处理。
生产使用以下镜像：

```text
Python image  m.daocloud.io/docker.io/library/python:3.11-slim
Octave image  fluxmind/octave:trixie-slim
```

Octave 镜像的可复现构建文件在：

```text
deploy/docker/octave-trixie-slim.Dockerfile
```

如果以后重建生产镜像，在 Trace-Twin 上执行：

```bash
cd /opt/fluxmind
docker build -t fluxmind/octave:trixie-slim -f deploy/docker/octave-trixie-slim.Dockerfile deploy/docker
```
