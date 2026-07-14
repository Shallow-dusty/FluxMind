# FluxMind 开发计划 (2026-06-21 至 2026-07-12)

**制定日期**: 2026-06-21  
**计划周期**: 3 周  
**目标**: 从 mock 工具转变为实用的研究助手

---

## 📋 总览

### 核心目标
1. ✅ **激活真实功能**：替换所有 mock，接入真实服务
2. ✅ **扩充内容**：论文库从 30 篇 → 50+ 篇
3. ✅ **基础多用户**：简单的用户管理和历史追溯

### 成功标准
- 能生成真实的控制系统图表
- 能安全执行用户的 Python/Octave 代码
- 论文检索质量明显提升
- 5+ 个真实用户在日常使用

---

## 🗓️ 第一周：激活真实功能 (2026-06-21 - 2026-06-28)

### Day 1-2: Docker 代码执行环境 🔴 Priority 3

#### 目标
替换 `LocalPythonExecutionProvider` 和 `LocalOctaveExecutionProvider`，使用真实的 Docker 沙箱。

#### 具体任务
```bash
# 1. 配置 Docker 环境
- 在 Trace-Twin 服务器上安装/验证 Docker
- 准备执行镜像：python:3.11-slim, octave:latest
- 测试 Docker 基本权限
- 给可能执行代码的 API/UI/worker systemd 服务配置 Docker daemon 访问权限，可参考 deploy/systemd/fluxmind-docker-execution.dropin.example.conf

# 2. 更新代码
- 检查 src/providers.py 中的 DockerExecutionProvider
- 确认已实现：网络隔离、资源限制、临时目录
- 删除或废弃 Local*ExecutionProvider (保留作为 fallback)

# 3. 配置和测试
- 设置环境变量：CODE_EXECUTION_BACKEND=docker
- 配置语言镜像：
  - DOCKER_PYTHON_EXECUTION_IMAGE=python:3.11-slim
  - DOCKER_OCTAVE_EXECUTION_IMAGE=gnuoctave/octave:latest
- 运行最小 smoke：python scripts/docker_execution_smoke.py --language all
- 默认 smoke 走 job/provider/artifact 完整边界；如需只调 provider，可加 --mode provider
- 若默认镜像不可用，可只在 smoke 中临时覆盖：
  - python scripts/docker_execution_smoke.py --language python --python-image python:3.11-slim
  - python scripts/docker_execution_smoke.py --language octave --octave-image gnuoctave/octave:latest
- 测试执行简单 Python 代码
- 测试执行 Octave 代码
- API 入口使用：
  - POST /jobs/code/python-docker
  - POST /jobs/async/code/python-docker
  - POST /jobs/code/octave-docker
  - POST /jobs/async/code/octave-docker
- Docker API 入口要求 CODE_EXECUTION_BACKEND=docker；否则返回 409 并提示当前 backend
- 测试生成图表和文件输出

# 4. 更新文档
- 更新 CLAUDE.md 中的执行后端说明
- 记录 Docker 配置要求
```

#### 验收标准
- [x] 能在 Docker 中执行用户提交的 Python 代码
- [x] 能在 Docker 中执行 Octave 代码
- [x] 生成的图表和文件能正确保存为 artifact
- [x] 执行失败有清晰的错误提示
- [x] 本地和生产环境都能工作

#### 预估时间
16 小时（2 天）

---

### Day 3-4: OpenAI 图像生成 🟡 Priority 2

#### 目标
在保留 `MockImageGenerationProvider` 本地 fallback 的前提下，接入 OpenAI Images API，
让 `/jobs/image`、`/jobs/async/image` 和 Streamlit 图像面板可通过
`IMAGE_PROVIDER_BACKEND=openai` 生成真实图像。

#### 具体任务
```bash
# 1. 准备 API 密钥
- 获取 OpenAI API Key
- 配置到运行环境：OPENAI_IMAGE_API_KEY=... 或 OPENAI_API_KEY=...
- [x] 在 src/config.py 中读取 OpenAI 图像配置

# 2. 实现真实 Provider
- [x] 创建 OpenAIImageGenerationProvider
- [x] 使用 OpenAI Python SDK 的 client.images.generate()
- [x] 将 b64_json/url 图像结果保存到 artifacts/
- [x] 添加 scripts/openai_image_smoke.py 作为显式 live smoke 入口
- [ ] 配置真实 key 后做 live smoke

# 3. 集成到系统
- [x] 更新 job runner 的 provider 选择逻辑
- [x] 新增 /jobs/image 与 /jobs/async/image 配置后端入口
- [x] 保留 /jobs/image/mock 与 /jobs/async/image/mock 本地 SVG fallback
- [x] Streamlit 图像面板改为配置后端入口
- [x] OpenAI 后端缺 key 时，API 返回 409，Streamlit 禁用按钮并提示配置缺失
- [ ] 配置真实 key 后测试 API 和 UI 生成图像

# 4. 特定于控制系统的优化
- [x] 优化 prompt 模板，适配控制系统领域
- [x] 预设常用图表类型：
  - PMSM 控制框图
  - 滑模控制器结构
  - 观测器架构
  - 波形图

# 5. 文档更新
- [x] 记录 API 配置方法和 API/UI 入口
- [ ] 添加真实 OpenAI 图像生成示例
```

#### 验收标准
- [ ] 配置真实 key 后，能通过 API 生成真实图像
- [ ] 配置真实 key 后，能通过 Streamlit UI 生成图像
- [x] 图像 provider 输出保存为 artifact
- [ ] live 图像针对"PMSM 控制框图"生成合理的图表
- [x] API 失败时有友好的错误提示

#### 预估时间
16 小时（2 天）

---

### Day 5: 简化用户管理 🟠 Priority 4

#### 目标
简化 `product_registry`，实现最小化的多用户支持。

#### 具体任务
```bash
# 1. 简化数据模型
- 保留 product_registry.py 的核心功能
- 简化为：users 表 + roles（admin/student）
- 移除复杂的：billing, quota 细粒度控制

# 2. 实现简单登录
- 在 Streamlit 侧边栏添加登录表单
- 基于 session_state 的简单认证
- admin 能看所有查询，student 只能看自己的

# 3. 查询历史隔离
- 在 query 表中记录 user_id
- /query API 记录当前用户
- Streamlit 根据用户显示历史

# 4. 基础权限检查
- 删除数据需要 admin
- 上传论文需要 admin
- 普通查询 student 也能做
```

#### 验收标准
- [ ] 有简单的登录界面
- [ ] admin 和 student 看到不同的内容
- [ ] 每个人能看到自己的查询历史
- [ ] 导师能看到所有学生的查询

#### 预估时间
8 小时（1 天）

---

### 第一周检查点

**周五下午评估**：
- [x] Docker 执行可用
- [ ] OpenAI 图像可用
- [ ] 基础登录可用
- [x] 在 Trace-Twin 上部署并测试 Docker 执行

---

## 🗓️ 第二周：扩充内容 (2026-06-29 - 2026-07-05)

### Day 6-10: 论文库扩充 🔵 Priority 1

#### 目标
从 30 篇扩充到 50+ 篇高质量论文，并让评估基线覆盖新增论文。

#### 具体任务

**Day 6-7: 滑模控制方向（10 篇）**
```bash
# 搜索关键词：
- Sliding Mode Control
- Super-twisting Algorithm
- Higher-order Sliding Mode
- Terminal Sliding Mode

# 选择标准：
- 经典论文（引用 > 500）
- 近期综述（2020 年后）
- 实用算法（有伪代码）
- [x] 新增 seed-paper importer：scripts/import_seed_papers.py
- [x] 本地 curated library 已从 30 篇扩充到 52 篇
```

**Day 8-9: PMSM 控制方向（10 篇）**
```bash
# 搜索关键词：
- PMSM Sensorless Control
- Flux Linkage Observation
- Model Reference Adaptive System
- Extended Kalman Filter for PMSM

# 选择标准：
- 工程应用（有实验结果）
- 算法创新（新方法）
- 对比分析（多种方法对比）
```

**Day 10: 检索优化**
```bash
# 1. 测试检索质量
- 对每个新论文问 2-3 个问题
- 检查是否能正确检索
- 调整 chunk size 和 overlap

# 2. 优化 metadata
- 补充缺失的 DOI/arXiv ID
- 标记论文的子领域标签
- 添加关键贡献摘要

# 3. 更新评估基线
- [x] 扩展 eval/rag_baseline.json：新增 22 个 seed-library retrieval-only cases
- [x] 添加新论文相关的问题
- [x] 运行 evaluate_rag.py
- [x] 重建本地 FAISS index，52 篇论文生成 3497 chunks
- [x] no-LLM retrieve smoke 命中新论文
- [x] 在生产环境更新索引并运行 live retrieval eval
```

#### 验收标准
- [x] 论文库达到 52 篇
- [x] 每篇新增论文有 manifest/source/pdf_url/topic metadata，并可由 PDF 抽取补充书目信息
- [x] 本地离线检索质量评估通过
- [x] 评估基线更新
- [x] 生产 live retrieval 评估通过

#### 预估时间
40 小时（5 天）

---

### 第二周检查点

**周五下午评估**：
- [x] 论文库 ≥ 50 篇
- [x] 本地检索质量明显提升
- [x] 本地离线评估测试通过
- [x] 在生产环境更新索引

---

## 🗓️ 第三周：用户体验与迭代 (2026-07-06 - 2026-07-12)

### Day 11-12: 查询历史功能 🟢 Priority 5

#### 目标
实现可追溯的查询历史。

#### 具体任务
```bash
# 1. 数据模型
- 扩展现有的 query events
- 记录：user, query, answer, timestamp, sources
- 保留最近 1000 条

# 2. 历史查询界面
- Streamlit 侧边栏显示"我的查询"
- 点击可查看完整对话
- 支持搜索和筛选

# 3. Admin 视图
- 导师能看所有学生的查询
- 按用户、时间、主题筛选
- 导出为 CSV 或 Markdown

# 4. 隐私设置
- 学生默认看不到其他人的查询
- 导师能设置"公开优秀查询"
```

#### 验收标准
- [ ] 每个用户能看到自己的查询历史
- [ ] 导师能看到所有查询
- [ ] 查询历史可搜索
- [ ] 能导出历史记录

#### 预估时间
16 小时（2 天）

---

### Day 13: 文档和演示

#### 具体任务
```bash
# 1. 更新用户文档
- README.md 更新功能列表
- 添加使用教程（带截图）
- 常见问题 FAQ

# 2. 写使用案例
- 案例 1: 查询滑模控制算法
- 案例 2: 生成 PMSM 控制框图
- 案例 3: 执行参数调优代码

# 3. 准备演示
- 录制 5 分钟演示视频
- 截图关键功能
- 准备演示脚本
```

#### 验收标准
- [ ] README.md 更新完整
- [ ] 有 3 个实际使用案例
- [ ] 有演示视频或 GIF

#### 预估时间
8 小时（1 天）

---

### Day 14: 真实用户测试

#### 具体任务
```bash
# 1. 邀请真实用户
- 邀请 3-5 个学生试用
- 让导师试用 admin 功能

# 2. 收集反馈
- 观察他们如何使用
- 记录遇到的问题
- 询问改进建议

# 3. 快速修复
- 修复阻碍使用的 bug
- 优化不流畅的体验
- 调整不清晰的文案

# 4. 部署到生产
- 合并所有功能到 main
- 部署到 Trace-Twin
- 通知用户新版本上线
```

#### 验收标准
- [ ] 至少 3 个真实用户试用
- [ ] 收集到具体反馈
- [ ] 修复关键问题
- [ ] 新版本部署成功

#### 预估时间
8 小时（1 天）

---

### 第三周检查点

**周五下午评估**：
- [ ] 查询历史可用
- [ ] 文档完整
- [ ] 真实用户在使用
- [ ] 收到正面反馈

---

## 📊 监控指标

### 开发过程指标

**每周检查**：
```bash
# 代码量变化
git diff --stat origin/main | tail -1

# 测试通过率
python -m pytest --tb=short

# 功能完成度
cat DEVELOPMENT_PLAN.md | grep "^\- \[x\]" | wc -l
```

### 使用指标

**每天检查**（第三周开始）：
- 查询次数
- 活跃用户数
- 平均查询长度
- 图像生成次数
- 代码执行次数

### 质量指标

**每周检查**：
- 检索准确率（evaluate_rag.py）
- API 响应时间
- 错误率
- 用户反馈评分

---

## 🚧 风险与应对

### 风险 1: Docker 配置困难
**应对**: 
- 预留额外 1 天调试时间
- 如果不行，先用 subprocess + 严格资源限制
- 第二周再解决

### 风险 2: OpenAI API 限流
**应对**:
- 使用 exponential backoff 重试
- 实现本地缓存
- 考虑使用其他 API（如 Stability AI）

### 风险 3: 找不到高质量论文
**应对**:
- 降低标准：40 篇也可以接受
- 扩展领域：加入相关的控制理论论文
- 求助：让导师推荐论文列表

### 风险 4: 用户不愿试用
**应对**:
- 导师先试用并推荐
- 提供小激励（如帮忙跑实验）
- 降低门槛：准备好示例问题

---

## 📅 每日工作流程

### 早晨（9:00-12:00）
1. 查看计划，确认今天任务
2. 专注开发核心功能
3. 不做计划外的事情

### 下午（14:00-18:00）
1. 继续完成功能
2. 写测试和文档
3. 本地测试

### 晚上（可选）
1. 提交代码
2. 部署到测试环境
3. 更新进度

---

## ✅ 完成标准（3 周后）

### 必须完成
- [x] Docker 代码执行可用
- [ ] OpenAI 图像生成可用
- [x] 论文库 ≥ 50 篇
- [ ] 基础用户登录和权限
- [ ] 查询历史可用

### 期望完成
- [ ] 至少 5 个真实用户在使用
- [ ] 检索质量提升 20%
- [ ] 用户反馈"非常有用"

### 可以推迟
- [ ] 完美的 UI 设计
- [ ] 100% 测试覆盖
- [ ] 详尽的文档

---

## 🔄 每周回顾

### 第一周末（2026-06-28）
- 回顾完成情况
- 调整第二周计划
- 识别阻碍因素

### 第二周末（2026-07-05）
- 回顾完成情况
- 准备用户测试
- 调整第三周计划

### 第三周末（2026-07-12）
- 总结 3 周成果
- 收集用户反馈
- 制定下一阶段计划

---

## 📞 需要帮助时

**遇到以下情况，立即停下来重新评估**：
1. 某个任务超过预估时间 2 倍
2. 发现计划有重大遗漏
3. 用户反馈方向完全错误
4. 再次陷入"安全清理"循环

**重新评估流程**：
1. 阅读 `CODE_PRINCIPLES.md`
2. 检查是否偏离核心目标
3. 调整计划或寻求反馈

---

**初稿**: Claude Code  
**校订/方向确认**: Shallow  
**生效日期**: 2026-06-21  
**下次审查**: 2026-07-05（第二周末）

## Docker 镜像策略补充（Trace-Twin / 国内服务器）

Trace-Twin 位于杭州，生产 Docker 执行不应依赖 Docker Hub 或 GHCR 直连拉取大镜像。
当前生产策略是：

```text
Python image  m.daocloud.io/docker.io/library/python:3.11-slim
Octave image  fluxmind/octave:trixie-slim
```

Octave 镜像通过 `deploy/docker/octave-trixie-slim.Dockerfile` 在 Trace-Twin 本机构建，
使用 USTC Debian mirror 安装 Debian trixie 的 Octave。后续 Docker 执行相关验收，
应优先验证服务运行用户、systemd `EnvironmentFile`、Docker group、artifact 输出目录和
`/jobs/code/*-docker` API 路由，而不是把 Docker Hub 直连作为前置条件。
