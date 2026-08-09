[English](README.md) | [中文](README.zh-CN.md)

# PU Learning Toolbox

**正例-无标记学习 Python 工具箱** -- 兼容 sklearn API，17 篇论文方法，支持 SCAR 与 SAR。

![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![Status](https://img.shields.io/badge/status-0.1.0--dev-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## 特性

- **17 个算法**，来自近年 PU 学习研究论文，全部为 native clean-room 实现（[方法卡](docs/research/method_cards/)）
- **兼容 sklearn API** -- `fit(X, y)` / `predict(X)` / `decision_function(X)`，支持 Pipeline 与交叉验证
- **SCAR & SAR** -- 常数与实例相关两种标记机制，附数据模拟器
- **数据画像 + 算法推荐** -- 自动质量检查、SCAR/SAR 证据，以及七维评分推荐器为你的数据选方法
- **可审计流水线** -- 一次调用的 `PUPipeline`（画像 → 先验 → 训练 → PU 分层 CV → 评估），加结构化诊断报告与先验/标记倾向敏感性分析
- **CLI** -- `pu-toolbox` 把整条流水线变成终端命令

## 快速开始

> **注意**：尚未发布到 PyPI，需要从源码安装。

```bash
git clone https://github.com/shuidisjtu/pu-learning-toolbox.git
cd pu-learning-toolbox
pip install -e .          # 核心依赖
pip install -e ".[torch]" # + 基于 PyTorch 的方法（nnPU、Dist-PU、Self-PU 等）
```

### Hello World

```python
import numpy as np
from pu_toolbox.preprocessing import make_sar_dataset
from pu_toolbox import PUPipeline

# 合成 PU 数据：部分正例被标记（1），其余为无标记（0）
X, y_pu, y_true, _ = make_sar_dataset(
    n_samples=1000, n_features=8, class_prior=0.3, random_state=42,
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

## 文档

文档按受众分层，完整索引见 [`docs/README.md`](docs/README.md)。

| 入口 | 内容 |
|------|------|
| [`docs/user/quickstart.md`](docs/user/quickstart.md) | 5 分钟快速开始（CLI + Python） |
| [`docs/user/concepts/`](docs/user/concepts/) | PU 问题设定、SCAR/SAR、方法选择 |
| [`docs/user/howto/`](docs/user/howto/) | 任务指南：模拟、画像、流水线、CLI、报告、敏感性 |
| [`docs/user/reference/api.md`](docs/user/reference/api.md) | 精确 API 契约 |
| [`docs/dev/`](docs/dev/) | 贡献者文档：架构、结构、路线图、兼容性 |
| [`docs/research/method_cards/`](docs/research/method_cards/) | 各论文方法卡 |

## AI 工作流 Skill

`pu-workflow`（Agent Skills 开放标准）以自然语言驱动完整 PU 分析流程：
数据画像、假设诊断、算法推荐、训练评估与结果解读。Claude Code / Cursor
（`.claude/skills/`）与 Codex / Gemini CLI / Windsurf（`.agents/skills/`）
原生加载。

## 开发

```bash
uv run pytest tests/ -v -m "not slow"       # 运行测试
uv run ruff check pu_toolbox/               # Lint 检查
uv run ruff format --check pu_toolbox/      # 格式检查

# 质量门禁
uv run python scripts/check_test_quality.py
uv run python scripts/check_doc_links.py
uv run python scripts/check_project_metadata.py
uv run python scripts/check_math_rendering.py
uv run python scripts/check_skill_sync.py
```

贡献指南见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## License

MIT
