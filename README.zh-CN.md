[English](README.md) | [中文](README.zh-CN.md)

# PU Learning Toolbox

**正例-无标记学习 Python 工具箱** -- 17 个注册算法、联合漂移研究适配，支持 SCAR 与 SAR。

![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![Status](https://img.shields.io/badge/status-1.11.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 特性

- **17 个注册算法**，来自近年 PU 学习研究论文，全部为 native clean-room 实现；另有隔离的联合漂移研究求解器（[方法卡](docs/research/method_cards/)）
- **兼容 sklearn API** -- `fit(X, y)` / `predict(X)` / `decision_function(X)`，支持 Pipeline 与交叉验证
- **SCAR & SAR** -- 常数与实例相关两种标记机制，附数据模拟器
- **数据画像 + 算法推荐** -- 自动质量检查、SCAR/SAR 证据，以及七维评分推荐器为你的数据选方法
- **可审计流水线** -- 一次调用的 `PUPipeline`（画像 → 先验 → 训练 → PU 分层 CV → 评估），加结构化诊断报告与先验/标记倾向敏感性分析
- **分布漂移护栏** -- OOF 源/目标漂移审计、有界协变量权重、覆盖诊断与受保护的 `ShiftAwarePUPipeline`
- **部署监控** -- 可恢复的窗口告警、coverage/拒绝预测控制，以及 CLI/UI 主动复核导出
- **CLI** -- `pu-toolbox` 把整条流水线变成终端命令
- **模型调整** -- 统一 `classifier_params`、命令行参数入口与 PU-aware 网格搜索
- **评估指标** -- PU 原生风险，以及带明确可用性契约的排序、平衡准确率与概率校准指标
- **可复现传统 PU benchmark** -- 七方法锁定协议、数据泄露预检、断点续跑、配对比较与调优证据
- **图形界面** -- 上传数据、配置/比较模型、查看诊断并下载报告与模型

## 快速开始

```bash
pip install pu-toolbox                # 核心依赖（Python ≥ 3.10）
pip install "pu-toolbox[torch]"       # + 基于 PyTorch 的方法（nnPU、Dist-PU、Self-PU 等）
pip install "pu-toolbox[ui]"          # + Streamlit 图形界面
```

### 安装环境

任何 Python ≥ 3.10 的解释器均可使用：本包是纯 Python 通用 wheel（无编译扩展），
解释器来源不影响。各环境注意事项：

- **venv / uv**（推荐）：标准隔离环境，无需特殊处理。
- **系统 Python**（python.org / Ubuntu / Homebrew）：版本必须 ≥ 3.10。
  Ubuntu 22.04+ 与 Debian 12+ 会阻止向系统环境 `pip install`（PEP 668）——
  请改用 venv。
- **Anaconda / Miniconda**：在 conda 环境内 `pip install pu-toolbox`（本包只在
  PyPI 发布，`conda install` 找不到）。若已通过 conda 安装 torch，普通
  `pip install pu-toolbox`（不带 `[torch]` extra）仍可启用基于 PyTorch 的方法——
  torch 是延迟导入的可选依赖。

### Hello World

```python
import numpy as np
from pu_toolbox.preprocessing import make_scar_dataset
from pu_toolbox import PUPipeline

# 合成 SCAR 数据（标记与特征无关——所有类先验估计器的前提）：
# 部分正例被标记（1），其余为无标记（0）。SAR 数据用
# make_sar_dataset(mechanism="linear")。
X, y_pu, y_true = make_scar_dataset(
    n=500, c=0.5, n_features=8, separation=1.0, random_state=42,
)

# 一次调用：画像 → 先验 → 训练 → PU 分层 CV → 评估
report = PUPipeline().fit_evaluate(X, y_pu, y_true=y_true)
print(report.summary())
```

完整文档（中文）：[docs/README.md](docs/README.md)。更多可运行示例：[`examples/minimal/`](examples/minimal/)。

## 命令行

`pu-toolbox` 命令把整条流水线封装为终端命令。完整指南：[`docs/user/howto/cli.md`](docs/user/howto/cli.md)。

```bash
# 1. 生成 SCAR 演示数据（X.csv / y_pu.csv / y_true.csv）
pu-toolbox make-demo-data --out-dir demo/ --n 200 --seed 42

# 2. 一键式全流程训练评估（auto 模式自动选算法）
pu-toolbox run --data demo/X.csv --labels demo/y_pu.csv --out-dir results/

# 3. 查看结果
#    results/report.md    完整 Markdown 报告
#    results/report.json  严格 JSON（无 NaN），可程序化消费
```

## 图形界面与模型调整

```bash
pip install "pu-toolbox[ui]"
pu-toolbox-ui
```

界面支持自动推荐、手动模型参数、PU 分层网格搜索、指标图表、诊断提示，以及报告、
预测和模型下载。Python 与 CLI 调参方法见
[模型调整指南](docs/user/howto/model_tuning.md)，界面说明见
[图形界面指南](docs/user/howto/ui.md)。

## 文档

文档按受众分层，完整索引见 [`docs/README.md`](docs/README.md)。

| 入口 | 内容 |
|------|------|
| [`docs/user/quickstart.md`](docs/user/quickstart.md) | 5 分钟快速开始（CLI + Python） |
| [`docs/user/concepts/`](docs/user/concepts/) | PU 问题设定、SCAR/SAR、方法选择 |
| [`docs/user/howto/`](docs/user/howto/) | 任务指南：模拟、画像、流水线、CLI、报告、敏感性、分布漂移 |
| [`docs/user/reference/api.md`](docs/user/reference/api.md) | 精确 API 契约 |
| [`docs/dev/`](docs/dev/) | 贡献者文档：架构、结构、路线图、兼容性 |
| [`docs/research/method_cards/`](docs/research/method_cards/) | 各论文方法卡 |

## AI 工作流 Skill

`pu-workflow`（Agent Skills 开放标准）以自然语言驱动完整 PU 分析流程：
数据画像、假设诊断、算法推荐、训练评估与结果解读。Claude Code / Cursor
（`.claude/skills/`）与 Codex / Gemini CLI / Windsurf（`.agents/skills/`）
原生加载。技能随 PyPI wheel 分发：`pip install "pu-toolbox>=1.2" && pu-toolbox
skill install` 一键启用；详见
[启用与使用 pu-workflow Skill](docs/user/howto/using_skill.md)。

## 开发

```bash
git clone https://github.com/shuidisjtu/pu-learning-toolbox.git
cd pu-learning-toolbox
pip install -e ".[dev,torch]"   # 开发安装
uv run pytest tests/ -v -m "not slow and not e2e"   # 快速测试（e2e 由 nightly 跑）
uv run ruff check pu_toolbox/               # Lint 检查
uv run ruff format --check pu_toolbox/      # 格式检查

# 质量门禁
uv run python scripts/check_test_quality.py
uv run python scripts/check_doc_links.py
uv run python scripts/check_project_metadata.py
uv run python scripts/check_math_rendering.py
uv run python scripts/check_skill_sync.py
uv run python scripts/check_format.py        # 格式门禁（ruff check + format --check，全目录）
uv run python scripts/generate_structure.py --check    # 结构文档一致性(--update 重新生成)
```

贡献指南见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## License

MIT
