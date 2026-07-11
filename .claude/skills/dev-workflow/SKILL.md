---
name: dev-workflow
description: Use when starting or finishing development work in the PU Learning Toolbox project — before writing code, after completing a task, before committing, or when syncing with GitHub Issues
---

# PU Learning Toolbox 开发流程

## 开发前

1. `git fetch origin && git status` — 确认本地与远程同步状态
2. 如果是新功能/fix，先 `git checkout -b feature/<name>` 或 `fix/<name>`
3. 检查 `docs/project_management/process_checklist.md` 确认当前 Phase 进度
4. 检查 [GitHub Issues](https://github.com/shuidisjtu/pu-learning-toolbox/issues) 是否有新的 assign 或待处理

## 开发完成后

1. `pytest tests/ -v` — 确认全部测试通过
2. 更新 `docs/project_management/process_checklist.md`（勾选完成项）
3. 检查是否需要同步更新以下文档：
   - `docs/project_structure.md` — 新增/删除文件
   - `docs/architecture.md` — API 变更
   - `docs/resources_optimized.md` — 源码状态变更
   - 方法卡 — 实现细节与设计有偏差
4. 如有已完成的 GitHub Issue，手动 close

## 提交前

1. `git branch` — 确认不在 `main` 上直接提交代码
2. 检查 commit message（不加 `Co-Authored-By`）
3. `git push -u origin <branch>`
4. 更新对应的 GitHub Issue 状态（如果使用）

## GitHub Issues 同步

- 进度清单: [Issue #1](https://github.com/shuidisjtu/pu-learning-toolbox/issues/1)（置顶）
- 交接任务以 Issue assign 形式跟踪
- 完成后 close issue 并勾选 checklist
