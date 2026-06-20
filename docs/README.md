# FluxMind 文档索引

**最后更新**: 2026-06-21

## 📖 核心文档（活跃维护）

### 用户文档
- **[../README.md](../README.md)** - 项目介绍、安装和快速开始
- **[../CLAUDE.md](../CLAUDE.md)** - Claude Code 开发约定

### 开发文档
- **[current/ARCHITECTURE.md](current/ARCHITECTURE.md)** - 系统架构设计（56KB）
- **[current/DEPLOYMENT_STATUS.md](current/DEPLOYMENT_STATUS.md)** - 生产部署状态（74KB）
- **[current/REPO_STATUS.md](current/REPO_STATUS.md)** - Git 仓库状态快照（207KB）

### 示例
- **[demo-script.md](demo-script.md)** - 演示脚本
- **[handover.html](handover.html)** - 可视化交付页面

## 📦 存档文档（仅供参考）

以下文档记录了开发过程中的详细审计和路线图，已移至 `archive/` 供参考：

| 文档 | 大小 | 说明 |
|------|------|------|
| `archive/BACKLOG.md` | 94KB | 实现待办清单 |
| `archive/FEATURE_AUDIT.md` | 66KB | 功能审计报告 |
| `archive/PLATFORM_AUDIT_AND_ROADMAP.md` | 59KB | 平台审计和路线图 |
| `archive/PRODUCTION_GAP_AND_MARKET_RESEARCH.md` | 48KB | 生产差距分析 |
| `archive/QUALITY_ROADMAP.md` | 8KB | 质量路线图 |

**总计**: ~275KB 开发历史记录

## 🎯 快速导航

**想要...**

- 快速上手 → [../README.md](../README.md)
- 理解架构 → [current/ARCHITECTURE.md](current/ARCHITECTURE.md)
- 查看部署状态 → [current/DEPLOYMENT_STATUS.md](current/DEPLOYMENT_STATUS.md)
- 了解开发历史 → `archive/` 目录

## 📊 文档维护约定

1. **REPO_STATUS.md** - Git 快照，由脚本自动更新
2. **DEPLOYMENT_STATUS.md** - 生产状态，部署后手动更新
3. **ARCHITECTURE.md** - 架构设计，重大变更时更新
4. **存档文档** - 保持只读，不再更新

**注意**: 不要在文档中包含 secrets、tokens、`.env` 值、PDF 内容、FAISS 索引或数据库内容。
