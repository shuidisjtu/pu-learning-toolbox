# 启用与使用 pu-workflow Skill

> 前置条件：已安装 `pu-toolbox`（>= 1.2，含技能资源）与 `uv`；使用 Claude Code /
> Cursor / Codex / Gemini CLI / Windsurf 等支持 [Agent Skills](https://agentskills.io) 的
> 客户端。

`pu-workflow` 是一个 Agent Skills 标准技能文件（`SKILL.md`），让 AI 客户端以自然
语言驱动完整 PU 分析流程：数据画像与 SCAR/SAR 假设诊断 → 算法推荐与类先验估计 →
训练评估 → 结果解读。

技能文件随发布包分发：自 v1.2 起 wheel 内置技能资源，`pu-toolbox skill install`
一键安装；技能内所有命令均为包自带 CLI 子命令（`profile` / `recommend` /
`sensitivity` / `run`），因此**安装包 + 一条命令即可启用完整工作流**，无需源码仓库。

## 1. 启用技能

```bash
uv pip install "pu-toolbox>=1.2"   # 或 pip install pu-toolbox
pu-toolbox skill install
```

命令把技能安装到用户级两个位置：

- `~/.claude/skills/pu-workflow/` — Claude Code / Cursor
- `~/.agents/skills/pu-workflow/` — Codex / Gemini CLI / Windsurf

任何目录下的客户端会话都能触发。常用变体：

- `pu-toolbox skill install --dest <目录>`：自定义安装根目录
- `pu-toolbox skill install --force`：覆盖已有安装（默认跳过，重复执行安全）

克隆仓库的开发者无需安装 —— 仓库内 `.claude/skills/pu-workflow/` 与
`.agents/skills/pu-workflow/` 已原生可用（双份由 `check_skill_sync` 门禁保证一致）。

## 2. 使用

在客户端会话中直接用自然语言描述 PU 分析任务，例如：

> 分析这份 PU 数据 `X.csv` / `y_pu.csv`，给我完整的分析报告

客户端匹配技能后自动执行四步,每步之间**强制检查点**会停下等你确认：

| 步骤 | 做什么 | 产出 |
|---|---|---|
| 1. 画像与诊断 | 数据质量 + SCAR/SAR 假设证据 | `profile.json` |
| 2. 推荐与先验 | 算法排序 + 类先验估计 | `recommendation.json` |
| 3. 训练与评估 | 分层交叉验证下的完整 PU 流水线 | `report.json` + `report.md` |
| 4. 解读与建议 | 面向结论的中文解读与下一步行动 | 对话摘要 |

手动执行这些步骤也可以直接用 CLI（等价）：

```bash
uv run pu-toolbox profile --data X.csv --labels y_pu.csv --out-dir work/
uv run pu-toolbox recommend --profile work/profile.json --out-dir work/
uv run pu-toolbox run --data X.csv --labels y_pu.csv --out-dir work/run
uv run pu-toolbox sensitivity --data X.csv --labels y_pu.csv --out-dir work/
```

## 3. 已知边界

- 技能的 Step 4 解读模板（`references/interpret.zh-CN.md`）与部分参考文档
  （`docs/user/...`）按仓库相对路径引用，非仓库环境（`skill install` 安装）下客户端
  找不到时会跳过模板、直接用对话补全，不影响四步主流程。
- 技能前提中 "Working directory: repository root" 主要针对仓库开发者；安装用户按
  自己的数据目录操作，命令中的相对路径指向自己的数据文件。

## 4. 故障排查

| 症状 | 处理 |
|---|---|
| 客户端不触发技能 | 确认安装目录存在（`~/.claude/skills/pu-workflow/SKILL.md`）、客户端支持 Agent Skills、任务描述与技能触发词匹配 |
| `skill install` 提示 `skill assets not found` | 包版本过旧（< 1.2 无内置资源），`uv pip install --upgrade pu-toolbox` 后重试 |
| `skill install` 提示 already installed | 已存在安装，默认跳过；需要覆盖加 `--force` |
| 命令 exit 1 | 输入/用法错误，读 stderr 的 `error:` 消息；`pu-toolbox <子命令> --help` 查参数 |
| 技能要求版本不符 | 技能要求 CLI `pu-toolbox >= 1.1`；`pip show pu-toolbox` 确认后升级 |
