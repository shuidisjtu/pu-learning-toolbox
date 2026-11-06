[English](README.md) | [中文](README.zh-CN.md)

# PU Learning Toolbox

**正例-无标记学习 Python 工具箱** -- 内置注册算法、联合漂移研究适配，支持 SCAR 与 SAR。

![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![Status](https://img.shields.io/badge/status-1.11.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 特性

- **近年 PU 学习研究的注册算法**，全部为 native clean-room 实现；另有隔离的联合漂移研究求解器（[方法卡](docs/research/method_cards/)）；当前集合见 `pu-toolbox list-methods`
- **兼容 sklearn API** -- `fit(X, y)` / `predict(X)` / `decision_function(X)`，支持 Pipeline 与交叉验证
- **SCAR & SAR** -- 常数与实例相关两种标记机制，附数据模拟器
- **数据画像 + 算法推荐** -- 自动质量检查、SCAR/SAR 证据，以及七维评分推荐器为你的数据选方法
- **可审计流水线** -- 一次调用的 `PUPipeline`（画像 → 先验 → 训练 → PU 分层 CV → 评估），加结构化诊断报告与先验/标记倾向敏感性分析
- **可组合 PU 研究** -- OS 与独立 TS 采样、显式先验目标、可注入实验策略、自定义 encoder 与研究设置留痕（[教程](docs/user/howto/custom_experiments.md)）
- **分布漂移护栏** -- OOF 源/目标漂移审计、有界协变量权重、覆盖诊断与受保护的 `ShiftAwarePUPipeline`
- **部署监控** -- 可恢复的窗口告警、coverage/拒绝预测控制，以及 CLI/UI 主动复核导出
- **CLI** -- `pu-toolbox` 把整条流水线变成终端命令
- **模型调优** -- 统一分类器参数与 PU-aware 网格搜索
- **评估指标** -- PU 原生风险，以及带明确可用性契约的排序、平衡准确率与概率校准指标
- **可复现传统 PU benchmark** -- 七方法锁定协议、数据泄露预检、断点续跑、配对比较与调优证据
- **图形界面** -- 上传数据、配置/比较模型、查看诊断并下载结果

## 快速开始

```bash
pip install pu-toolbox                # 核心依赖（Python ≥ 3.10）
pip install "pu-toolbox[torch]"       # + 基于 PyTorch 的方法（nnPU、Dist-PU、Self-PU 等）
pip install "pu-toolbox[text]"        # + Survey 文本数据集的固定 SBERT 预处理
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
from pu_toolbox.preprocessing import make_scar_dataset
from pu_toolbox import PUPipeline

# Synthetic OS data with SCAR labeling: each positive has the same labeling
# probability. Observed labels are 1 (labeled positive) and 0 (unlabeled).
X, y_pu, y_true = make_scar_dataset(
    n=500, c=0.5, n_features=8, separation=1.0, random_state=42,
)

# One call: profile -> class prior -> train -> PU-stratified CV -> evaluate
report = PUPipeline().fit_evaluate(X, y_pu, y_true=y_true)
print(report.summary())
```

完整文档（中文）：[docs/README.md](docs/README.md)。更多可运行示例：[`examples/minimal/`](examples/minimal/)。

`PUPipeline` 默认采用 OS 输入采样；独立取得的 P 与总体 U 使用 `input_sampling="ts"`。
显式传入的 `class_prior` 指总体先验，报告记录其来源、换算及每个训练折的实际消费值。
详见[先验语义](docs/user/concepts/class_prior_semantics.md)与
[流水线指南](docs/user/howto/pipeline.md)。

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

界面支持自动推荐、手动模型参数、PU 分层网格搜索、指标图表、诊断提示，以及报告和模型下载。
相关说明见
[模型调整指南](docs/user/howto/model_tuning.md)，界面说明见
[图形界面指南](docs/user/howto/ui.md)。

## 文档

文档按受众分层，完整索引见 [`docs/README.md`](docs/README.md)。

| 入口 | 内容 |
|------|------|
| [`docs/user/quickstart.md`](docs/user/quickstart.md) | 5 分钟快速开始（CLI + Python） |
| [`docs/user/concepts/`](docs/user/concepts/) | 采样、SCAR/SAR、先验目标、PA/OA 选模、表征与方法选择 |
| [`docs/user/howto/`](docs/user/howto/) | 模拟、流水线、DIY 实验、encoder、CLI、报告、敏感性与分布漂移 |
| [`docs/user/reference/api.md`](docs/user/reference/api.md) | 精确 API 契约 |
| [`docs/dev/`](docs/dev/) | 贡献者文档：架构、结构、兼容性 |
| [`docs/dev/project_structure.md`](docs/dev/project_structure.md) | 源码与本地实验资产分类、归档清理边界、脚本用途导航 |
| [`docs/research/method_cards/`](docs/research/method_cards/) | 各论文方法卡 |

### 可组合 PU 研究

在仓库检出目录中，可以通过以下 CPU 示例逐步组合研究流程：

| 任务 | 指南 | 可运行示例 |
|---|---|---|
| 构造 OS 与独立 TS 数据 | [采样操作](docs/user/howto/pu_sampling.md) | [15_pu_sampling.py](examples/minimal/15_pu_sampling.py) |
| 跟踪已知与估计先验进入训练 | [先验目标](docs/user/concepts/class_prior_semantics.md) | [16_prior_targets.py](examples/minimal/16_prior_targets.py) |
| 比较 Bernoulli 与固定数 SAR 政策 | [SAR 模拟](docs/user/howto/sar_simulation.md) | [17_sar_policies.py](examples/minimal/17_sar_policies.py) |
| 训练自定义 encoder 或使用冻结特征 | [自定义 encoder](docs/user/howto/custom_encoders.md) | [18_custom_encoders.py](examples/minimal/18_custom_encoders.py) |
| 注入实验策略并通过 PA/OA 独立选模 | [DIY 实验](docs/user/howto/custom_experiments.md) | [14_diy_experiment.py](examples/minimal/14_diy_experiment.py) |

实验 manifest 记录采样、标记、风险池、先验输入与实际消费、监督和表征。
汇总文件、工作簿及诊断卡展示可读设置，历史结果沿用原记录身份。
操作流程见[运行与报告指南](docs/research/pu_survey/runbook.md)。

### PU Survey 实验

Survey 脚本通过 `ExperimentRunner` 执行仓库选定的研究设置，使用四路数据角色、
独立 PA/OA 选模和最终 test 评测。可通过以下命令准备可复现的数据切分并查看参数。

LBE 诊断使用 `scripts/_lbe_diagnostic_reference.py` 中单独的 SCAR 约束参考实现。

```bash
uv run python scripts/prepare_survey_splits.py --help
uv run python scripts/run_survey_experiment.py --help
uv run python scripts/run_lbe_comparison.py --help
uv run python scripts/run_lbe_diagnostics.py --help
uv run python scripts/prepare_clean_support.py --help
uv run python scripts/run_clean_support_experiment.py --help
uv run python scripts/run_genpu_recipe.py --help
uv run python scripts/run_candidate_image_experiment.py --help

# SCAR 实验行（PA + OA 选模）
uv run python scripts/run_survey_experiment.py path/to/splits \
    --method lbe --c 0.1,0.3,0.5

# SAR 压力测试（LBE-A/LBE-B 标记机制，仅 OA，c ∈ {0.05, 0.5}）
uv run python scripts/run_survey_experiment.py path/to/splits \
    --method lbe --labeling-mechanism sar_lbe_a --c 0.05,0.5
```

registry 声明需要先验的方法在运行入口使用 `--class-prior`，构造参数通过
`--model-params` 提供。各参数结合[先验目标](docs/user/concepts/class_prior_semantics.md)
与记录的训练视图解释，可运行配置见[runbook](docs/research/pu_survey/runbook.md)。
`--labeling-mechanism` 选择 PU 标签视图的生成方式（SCAR 或 SAR LBE 变体），
它与 `--method` 相互独立：标记机制是实验自变量。支持所选标记机制的方法在 SAR
实验行中报告 OA。

结果记录执行模式与资格状态。协议、执行状态和报告流程见
[`docs/research/pu_survey/`](docs/research/pu_survey/)。

配套脚本（范围与限制见各脚本 docstring 和 Survey 文档）：

| 阶段 | 脚本 |
|---|---|
| 数据与切分 | `collect_survey_public_data.py`（获取缺失的公开原始数据，不含受控访问的 ADNI）、`survey_splits_archive.py`（split 传输） |
| 运行 | `run_survey_pilot.py`（范围规划、执行与续跑） |
| 报告 | `summarize_survey_results.py`、`export_survey_results.py`（工作簿、诊断与结论；需要 `report` extra）、`_source_provenance.py`（内部代码与环境身份辅助工具） |
| 配方 | `prepare_survey_recipe_registry.py`、`check_survey_recipe_registry.py`（校验注册表；CI 检查冻结的锁） |
| 独立图像实验 | `run_candidate_image_experiment.py`（PULDA/GradPU 四角色 PA/OA smoke、实测快照与重放） |
| 独立传统方法实验 | `run_classical_candidate_experiment.py`（PUET/RP/KLDCE 四角色训练、显式参数与实际成本留痕） |
| 独立 support 实验 | `run_clean_support_experiment.py`（Self-PU/LaGAM/PULNS 额外标签审计与显式 Self-PU TS meta-checkpoint） |

这些脚本记录技术检查与实测结果。报告读取已有结果，将产物写入选定的 `--out-dir`。

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
uv run python scripts/check_api_docs.py       # api.md 覆盖全部公共符号（防漂移）
uv run python scripts/check_project_metadata.py
uv run python scripts/check_math_rendering.py
uv run python scripts/check_skill_sync.py
uv run python scripts/check_baseline_configs.py    # 基线配置与源码默认参数一致性
uv run python scripts/check_format.py        # 格式门禁（ruff check + format --check，全目录）
uv run python scripts/check_comment_quality.py  # 注释卫生：遗留标记 / 裸 TBD；行尾注释在已迁移分区为 error，未迁移分区为 advisory 并在通过时自报
uv run python scripts/check_survey_recipe_registry.py  # P4.1 recipe registry 门禁；CI 对冻结的注册表以 --require-present --require-locked 运行
uv run python scripts/generate_structure.py --check    # 结构文档一致性(--update 重新生成)
uv run python scripts/generate_layer_deps.py --check  # architecture.md §2.1 分层表（--update 重新生成）
```

贡献指南见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## License

MIT
