# 启用与使用 pu-workflow Skill

> 前置条件：已安装 `pu-toolbox`（>= 1.1）与 `uv`；使用 Claude Code / Cursor /
> Codex / Gemini CLI / Windsurf 等支持 [Agent Skills](https://agentskills.io) 的
> 客户端。

`pu-workflow` 是一个 Agent Skills 标准技能文件（`SKILL.md`），让 AI 客户端以自然
语言驱动完整 PU 分析流程：数据画像与 SCAR/SAR 假设诊断 → 算法推荐与类先验估计 →
训练评估 → 结果解读。

技能文件**不在 PyPI 包里**（wheel 只含 Python 代码），需要单独获取。自 v1.1 起，
技能内的所有命令均为 `pu-toolbox` 包自带的 CLI 子命令（`profile` / `recommend` /
`sensitivity` / `run`），因此**只要安装包 + 拿到技能文件即可运行完整工作流**，无需
源码仓库。

## 1. 启用技能(三选一)

**方式 A:克隆仓库(最简单,仓库内即用)**

```bash
git clone https://github.com/shuidisjtu/pu-learning-toolbox.git
cd pu-learning-toolbox && uv sync
```

在仓库根目录打开客户端,技能自动可用。适合同时想读源码/文档的用户。

**方式 B:复制到项目(推荐给 pip 用户,单项目使用)**

```bash
# 从仓库复制技能目录(Claude Code / Cursor 用 .claude,Codex / Gemini CLI / Windsurf 用 .agents)
git clone --depth 1 https://github.com/shuidisjtu/pu-learning-toolbox.git /tmp/tb
mkdir -p <你的项目>/.claude/skills
cp -r /tmp/tb/.claude/skills/pu-workflow <你的项目>/.claude/skills/
```

在 `<你的项目>` 目录打开客户端即可使用。

**方式 C:用户级全局安装(所有项目可用)**

把 `pu-workflow/` 复制到 `~/.claude/skills/`（Windows: `C:\Users\<你>\.claude\skills\`；
Codex 等客户端对应 `~/.agents/skills/`）。任何目录下的客户端会话都能触发。

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
  （`docs/user/...`）为仓库内文件,纯 pip 环境（方式 B/C）中没有;客户端找不到时会
  跳过模板、直接用对话补全,不影响四步主流程。
- 技能前提中 "Working directory: repository root" 主要针对方式 A 的仓库用户;
  方式 B/C 的用户按自己的数据目录操作,命令中的相对路径指向自己的数据文件。

## 4. 故障排查

| 症状 | 处理 |
|---|---|
| 客户端不触发技能 | 确认 `SKILL.md` 位于 `skills/pu-workflow/` 下、客户端支持 Agent Skills、任务描述与技能触发词匹配 |
| 命令 exit 1 | 输入/用法错误,读 stderr 的 `error:` 消息;`pu-toolbox <子命令> --help` 查参数 |
| 技能要求版本不符 | 技能要求 `pu-toolbox >= 1.1`;`pip show pu-toolbox` 确认后升级 |
| 命令不存在 | 确认安装的是发布包而非仅克隆源码(仓库内命令带 `uv run` 前缀) |
