# 启用与使用 pu-workflow Skill

> 前置条件：Python 3.10+、`pu-toolbox` 及支持 [Agent Skills](https://agentskills.io)
> 的客户端。项目环境优先使用 `uv`；图像与 encoder 路线安装 torch extra。

`pu-workflow` 根据自然语言任务选择工具箱的 CLI 或 Python API，覆盖日常 PU 分析、
数据构造、先验解析、调参比较、四角色研究、PA/OA 选模和图像表征，以及漂移、监控、
人工复核和基准审计。主入口按任务读取六份参考文件，结果解释使用实际保存的产物。

本轮 skill 修订版本为 1.11.0，验证环境为当前源码构建的 `pu-toolbox 1.11.0` wheel。
客户端按已安装包的命令帮助、Python 签名和能力声明选择执行路线。

## 1. 安装与启用

使用发布包的用户在自己的环境执行：

```bash
uv pip install "pu-toolbox>=1.11.0"
pu-toolbox skill install
```

源码工作区使用当前检出内容：

```bash
uv sync --python 3.11 --extra dev --extra torch
uv run pu-toolbox skill install --dest work/agent-home
```

安装器将完整主入口和参考文件复制到两个目标：

- `~/.claude/skills/pu-workflow/`：Claude Code / Cursor。
- `~/.agents/skills/pu-workflow/`：Codex / Gemini CLI / Windsurf。

`--dest` 指定上述两个目录的共同安装根目录。已有安装默认保留；用户选择更新已有安装时，
使用 `pu-toolbox skill install --force`。仓库内的两套 skill 也可由客户端直接发现。

## 2. 按任务使用

向客户端说明数据路径、目标、已有先验和预算，例如：

> 分析 `X.csv` 和 `y_pu.csv`。数据来自同一总体池，已知总体先验 0.4，
> 使用 CPU、3 折 CV、seed 42，报告写到 `work/analysis`。

> 我有全标注数据，先按来源切分，再按固定数 SCAR 与特征相关 SAR 构造 PU 数据，
> 比较两种标记设置，保存来源索引与采样记录。

> 为四路数据运行 PA 和 OA，分别选择 checkpoint 和阈值，再评测 test。

| 任务 | 入口与产物 |
|---|---|
| 日常 OS 表格分析 | CLI/Pipeline；画像、推荐、先验和训练报告 |
| 独立 TS 与显式先验目标 | Python `input_sampling="ts"`；目标换算与逐折消费记录 |
| OS/TS、SCAR/SAR 构造 | preprocessing API；PU 标签、采样政策、来源与先验元数据 |
| 调参或多方法比较 | `PUTuner` / `PUModelComparator` 或 CLI 配置；候选及选择结果 |
| 四角色研究与 PA/OA | `ExperimentRunner`；各协议选择、checkpoint、阈值和 test |
| 图像、encoder、冻结特征 | 注册能力匹配的模型与表征路径；模型和表示身份 |
| 漂移、监控、复核、审计 | 对应 CLI；结构化诊断与执行产物 |

skill 复用用户已有选择和授权，按任务推进。诊断结果进入方法选择和报告；涉及缺失的研究设计、
数据来源或额外预算时，客户端提出具体问题。日常完整分析可自动选择适用的先验估计器并记录来源。

## 3. 采样、先验与图像设置

当前 `profile`、`recommend`、`run` CLI 使用 OS 输入语义。独立 TS 通过 Python API
显式指定采样设计，见[Pipeline 指南](pipeline.md)和[采样教程](pu_sampling.md)。
Pipeline 调用级 `class_prior` 和报告的 `prior.value` 使用总体目标；报告保存原始估计目标、
总体换算及每折实际训练消费。独立推荐使用目标明确的总体先验，自动估计分析通过 Pipeline
统一解析先验链。详细定义见[先验语义](../concepts/class_prior_semantics.md)。

原始图像使用有限值 NCHW 数组，并选择显式 CNN 分类器，例如：

```bash
pu-toolbox run --data X.npy --labels y_pu.csv --classifier nnpu --architecture cnn --class-prior 0.4 --device cpu --max-epochs 2 --cv 2 --out-dir work/images
```

自定义 encoder 和冻结特征的生命周期见[表征原理](../concepts/representations.md)及
[encoder 教程](custom_encoders.md)。四角色和选择记录见[自定义实验](custom_experiments.md)。

## 4. 安装后的参考文件

每个安装目标都包含相同的七份资源：

| 资源 | 内容 |
|---|---|
| `SKILL.md` | 任务识别、能力发现、路由与交付条件 |
| `references/analysis.md` | 日常分析、调参比较、敏感性计算 |
| `references/sampling-and-priors.md` | 采样、标记、先验目标和消费链 |
| `references/research-experiments.md` | 四角色、PA/OA、checkpoint、自定义策略 |
| `references/representations.md` | CNN、encoder 注入和冻结特征 |
| `references/extensions.md` | 漂移、监控、复核、审计 |
| `references/interpret.zh-CN.md` | 中文字段解释和结论书写 |

内部引用相对于 skill 所在目录解析。安装用户可在自己的数据目录工作，参考文件自带最小示例；
源码用户可进一步运行[示例 14–18](../../../examples/minimal/)。

## 5. 结果与故障处理

客户端读取实际 JSON，报告数据设置、方法、先验来源与消费、可用指标和后续动作。
Pipeline schema 1.1 保存 `prior.training_consumption` 与指标有效折数；研究 manifest
分别保存 PA/OA 的选择和最终测试。文档和总结直接陈述证据支持的结论及其适用条件。

| 情况 | 动作 |
|---|---|
| 客户端未触发 | 核对安装目录、客户端发现设置与任务描述；显式调用 `$pu-workflow` |
| 已有安装 | 保留当前目录，或在选择更新后使用 `--force` |
| 命令或 Python 符号与路线要求不同 | 查看安装版本、命令帮助和签名，安装包含所需能力的构建 |
| 输入报错 | 按实际错误修正文件格式、维数、标签或参数，然后重跑 |
| 诊断提示或指标待补充 | 读取具体字段与建议，按已授权路线推进并报告相应数据动作 |

维护者使用 `uv run python scripts/check_skill_sync.py` 核对完整资源树、内部引用和
可移植 frontmatter。安装、wheel 内容与外部目录执行共同验证分发体验。
