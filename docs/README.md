# FluxMind 文档索引

**最后更新**: 2026-07-14

## 📖 开发指南（单一入口）

- **[../DEVELOPMENT.md](../DEVELOPMENT.md)** - 项目定位、功能、架构、开发约定、状态、路线图
  （替代旧的 CODE_PRINCIPLES / DEVELOPMENT_PLAN / NEXT_STEPS / DISCUSSION，已归档）

## 📕 用户文档

- **[../README.md](../README.md)** - 项目介绍、安装和快速开始
- **[../CLAUDE.md](../CLAUDE.md)** / **[../AGENTS.md](../AGENTS.md)** - agent harness 开发约定（内容同步）

## 📘 当前文档（活跃维护）

- **[current/ARCHITECTURE.md](current/ARCHITECTURE.md)** - 系统架构设计
- **[current/DEPLOYMENT_STATUS.md](current/DEPLOYMENT_STATUS.md)** - 生产部署状态
- **[current/REPO_STATUS.md](current/REPO_STATUS.md)** - Git 仓库状态快照（脚本生成）

## 📦 归档文档（只读）

### legacy/ — 旧约束文档（2026-07-14 作废归档）

| 文档 | 说明 |
|------|------|
| `legacy/CODE_PRINCIPLES.md` | 旧开发原则、防御代码冻结、红线（已废止） |
| `legacy/DEVELOPMENT_PLAN.md` | 旧 3 周计划（已废止） |
| `legacy/NEXT_STEPS.md` | 旧行动清单（已废止） |
| `legacy/DISCUSSION.md` | 项目诊断与决策记录（参考） |
| `legacy/CLEANUP_SUMMARY.md` | 文档整理总结（参考） |

### archive/ — 历史审计文档

| 文档 | 大小 | 说明 |
|------|------|------|
| `archive/BACKLOG.md` | 94KB | 实现待办清单 |
| `archive/FEATURE_AUDIT.md` | 66KB | 功能审计报告 |
| `archive/PLATFORM_AUDIT_AND_ROADMAP.md` | 59KB | 平台审计和路线图 |
| `archive/PRODUCTION_GAP_AND_MARKET_RESEARCH.md` | 48KB | 生产差距分析 |
| `archive/QUALITY_ROADMAP.md` | 8KB | 质量路线图 |

## 🎯 快速导航

**想要...**

- 开始开发 → [../DEVELOPMENT.md](../DEVELOPMENT.md)
- 快速上手 → [../README.md](../README.md)
- 理解架构 → [current/ARCHITECTURE.md](current/ARCHITECTURE.md)
- 查看部署状态 → [current/DEPLOYMENT_STATUS.md](current/DEPLOYMENT_STATUS.md)
- 了解历史 → `legacy/` / `archive/`

## 📊 文档维护约定

1. **DEVELOPMENT.md** - 开发单一事实来源，重大变更时更新
2. **DEPLOYMENT_STATUS.md** - 生产状态，部署后手动更新
3. **REPO_STATUS.md** - Git 快照，脚本生成
4. **ARCHITECTURE.md** - 架构设计，重大变更时更新
5. **legacy/ + archive/** - 只读，不再更新

**注意**: 不在文档中包含 secrets、tokens、`.env` 值、PDF 内容、FAISS 索引或数据库内容。