# pu-workflow Skill 设计

> 状态：已批准（2026-08-06）。实现走 `feature/pu-workflow-skill` 分支，完成后本设计蒸馏进 decision_log（参照 cli_design.md 先例）。

## 背景与目标

`cli_design.md` 范围边界规划了 "skill 集成（`pu-workflow` skill 自然语言驱动，内部调 PUPipeline 或 CLI）"，当时明确不在 v1。现在 Pipeline/CLI 执行内核已就绪（648 测试全绿），条件成熟。

**关键需求：通用 skill——不绑定 Claude Code。** 受众 = 队友（使用不同 AI 工具）+ 公开发布。因此：

- 载体 = **开放 Agent Skills 规范**（Anthropic 2025-12-18 发布为开放标准，agentskills.io 治理，25+ 工具采用：Claude Code、OpenAI Codex、Gemini CLI、GitHub Copilot、Cursor、VS Code、Windsurf、Warp、OpenCode 等）
- 仅用开放标准 frontmatter 字段（Claude Code 扩展字段会被其他工具静默忽略）
- 目录策略覆盖全部主流工具的扫描路径

## 需求决策记录

| 决策点 | 结论 |
|---|---|
| 受众 | 队友（多工具）+ 公开发布 |
| 载体 | 开放 Agent Skills 规范（SKILL.md + frontmatter） |
| 执行层 | 混合：CLI 能干的用 CLI（训练/评估），CLI 缺的写仓库级环节脚本（画像/推荐/敏感性） |
| 覆盖范围 | 四环节全流程：画像诊断 → 推荐先验 → 训练评估 → 结果解读 |
| 发布形态 | 随仓库发布（目录进 git，clone 即得） |
| 自动化程度 | 自适应检查点（无异常连续跑完，需用户输入/有警告时停下） |
| 语言 | SKILL.md 英文主体 + 中文解读指南（references） |

## 跨工具目录事实（实现依据）

没有任何一个目录被所有工具扫描，覆盖全部主流工具至少需要两个目录：

| 目录 | 原生扫描工具 |
|---|---|
| `.claude/skills/` | Claude Code、Cursor |
| `.agents/skills/` | OpenAI Codex、Gemini CLI、Windsurf、OpenCode、Warp |

> 来源：Agent Skills 开放标准（agentskills.io）、anthropics/skills 生态文档。

## 架构

**skill = 说明书（SKILL.md，英文）+ 执行内核（仓库级环节脚本，可测试）**。skill 不复制任何 docs 内容，references 全部链接已有文档——零漂移，符合项目文档一致性治理。

```
pu-workflow（随仓库发布）
│
├── .claude/skills/pu-workflow/SKILL.md     ← 完整版 ①（Claude Code + Cursor 原生加载）
├── .agents/skills/pu-workflow/SKILL.md     ← 完整版 ②（其余工具原生加载）
│        （① ② 内容逐字节一致，由门禁保证）
│
├── scripts/pu_workflow/                    ← 执行内核（仓库级、pytest 覆盖）
│     ├── profile.py        # 画像 + SCAR/SAR 诊断（CLI 无此能力）
│     ├── recommend.py      # 推荐 + 先验估计（CLI 无此能力）
│     └── sensitivity.py    # 敏感性分析（CLI 无此能力）
│     # 训练评估不写脚本 —— 直接调已有 CLI `pu-toolbox run`
│
├── .claude/skills/pu-workflow/references/
│     └── interpret.zh-CN.md   # 中文结果解读指南（唯一一份，两份 SKILL.md 以仓库路径引用）
│
└── 其余 references 直接链接 docs/user/howto/*.md、concepts/method_selection.md
```

## SKILL.md 规格

### frontmatter（只用开放标准字段）

```yaml
---
name: pu-workflow          # 必须与目录名一致
description: >-
  End-to-end PU learning workflow: data profiling, assumption
  diagnosis, algorithm recommendation, class-prior estimation,
  training/evaluation, and result interpretation.
license: MIT
compatibility: Python >=3.10, uv, pu-toolbox
metadata:
  author: shuidisjtu
  version: 0.1.0
---
```

禁止添加 Claude Code 扩展字段（`argument-hint` / `model` / `context: fork` / `agent` / `user-invocable` / `disable-model-invocation` / `hooks`）。

### 正文结构（英文，目标 <500 行，渐进披露）

```
# pu-workflow

## When to use          # 触发场景与关键词
## Prerequisites        # 环境（uv run）、输入契约（CSV/npy 格式约定）
## Workflow overview    # 4 环节表：做什么/用什么执行/输出/失败怎么办
## Step 1 — Profile & diagnose      → uv run python scripts/pu_workflow/profile.py
## Step 2 — Recommend & prior      → uv run python scripts/pu_workflow/recommend.py
## Step 3 — Train & evaluate       → pu-toolbox run ...（现有命令模板）
## Step 4 — Interpret & advise     → 解读规则 + references/interpret.zh-CN.md
## Checkpoints          # 自适应停止规则
## Failure handling     # 错误码 → 建议动作
## References           # 链接 docs/user/ 相关文档
```

### 自适应检查点规则

1. **环节 1 后**：画像有警告（稀疏/正样本过少/SCAR 不可识别）→ 停下，解释含义 + 修复建议，等确认
2. **环节 2 前**：用户未提供 `class_prior` → 停下，列"自动估计 vs 手动指定"选项（km1/km2 说明），等用户选
3. **环节 3 后**：report 有 error/warning 条目 → 停下解读，不自动继续
4. **无异常时**：连续跑完 4 环节，最后统一输出结论 + 行动建议
5. **任何环节失败**：按 Failure handling 给用户 3 选（修数据重跑 / 换方法 / 完整 traceback）

## 环节脚本契约

输入文件 + 输出 JSON 文件（agent 读输出文件，不解析 stdout）：

| 脚本 | 输入 | 输出 | 内部调 |
|---|---|---|---|
| `profile.py` | X.csv + y_pu.csv（或 .npy） | `profile.json`（SCAR/SAR 诊断、问题清单） | `profile_pu_data` + `build_diagnostic_report` |
| `recommend.py` | profile.json + 可选 `--class-prior` | `recommendation.json`（候选排序 + 全局警告 + 先验估计） | `recommend_from_profile` + prior 估计器 |
| `sensitivity.py` | X.csv + y_pu.csv + y_true.csv | `sensitivity.json`（假设扫描区间） | `sensitivity_analysis` |

- 支持 `--out-dir`，输出文件名固定（skill 指令引用具体文件名）
- 复用 `utils/serialization.py` 严格 JSON 序列化
- 退出码沿用 CLI 约定：0 成功；1 用户/运行错误（stderr 清晰消息、无 traceback）
- pytest 覆盖输出契约（新测试目录 `tests/unit/workflow_scripts/`）

## 一致性门禁（第 5 道）

新脚本 `scripts/check_skill_sync.py`：
- 逐字节比对两份 SKILL.md 内容一致
- 校验 frontmatter 只用开放标准字段白名单（防未来加 Claude Code 扩展字段破坏可移植性）

升为第 5 道质量门禁，同步更新：README / README.zh-CN / CONTRIBUTING / PR 模板 / project_structure / development_compatibility 命令清单 + CI step + dev-workflow SKILL.md（参照 check_math_rendering 加门禁先例）。

## 测试

- `tests/unit/workflow_scripts/`：三脚本输出契约（JSON 结构、文件名、退出码、--out-dir）+ demo 数据端到端
- 门禁脚本自身测试

## 文档更新

- `cli_design.md`：范围边界"skill 集成"改为已完成；注明 **profile/recommend/sensitivity 以脚本而非 CLI 子命令提供**（多命令全面版仍未做，范围决策记录在案）
- `process_checklist.md`：完成记录行
- 根 README（双语惯例）：加"AI workflow skill"一句话入口

## 验证流程

1. 全量测试 + 5 道门禁 + ruff（check + format）
2. **实战演练**：按 SKILL.md 指令对 demo 数据完整跑一遍 4 环节，确认每一步指令与脚本实现一致（skill 类工作核心验收，防"文档写得好但 agent 照做会失败"）
3. process_checklist 记录

## 范围外（明确不做）

- 不补 CLI 子命令（profile/recommend/sensitivity 以脚本形式，不动 argparse 面）
- 不做 ONNX 导出等 v1 外功能
- 不写双语 SKILL.md（英文主体 + 中文 interpret 参考已定）
- 不做多语言 interpret（只 zh-CN）
