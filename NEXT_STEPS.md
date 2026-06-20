# FluxMind 下一步行动清单

**生成时间**: 2026-06-21  
**状态**: 文档整理完成，等待推送

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
   - 领先 origin/main 65 个提交
   - 工作区干净
   - 准备推送

---

## 🚀 立即行动（今天）

### 1. 推送到 GitHub
```bash
git push origin main
```

### 2. 创建提醒标签
```bash
# 标记当前重构起点
git tag -a v0.9.0-refocus -m "Refocus: From over-engineering to real features"
git push origin v0.9.0-refocus
```

---

## 📋 下周开始执行

### Week 1 (2026-06-21 - 2026-06-28)

**Day 1-2: Docker 代码执行**
```bash
# 目标：替换 mock，使用真实 Docker
cd /home/shallow/04.AI-Prism/11.FluxMind

# 检查 Docker 可用性
docker --version
docker ps

# 启动开发
# 参考：DEVELOPMENT_PLAN.md "Day 1-2" 部分
```

**Day 3-4: OpenAI 图像生成**
```bash
# 目标：接入 DALL-E API
# 需要：OpenAI API Key

# 参考：DEVELOPMENT_PLAN.md "Day 3-4" 部分
```

**Day 5: 简化用户管理**
```bash
# 目标：简单的 admin/student 区分
# 参考：DEVELOPMENT_PLAN.md "Day 5" 部分
```

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
