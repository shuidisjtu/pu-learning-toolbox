# Contributing

本文档定义 PU Learning Toolbox 的代码、论文方法、benchmark 和项目状态管理流程。

## 1. 权威来源

发生冲突时按以下顺序处理：

1. `pyproject.toml`：Python、运行依赖、可选依赖和工具配置。
2. `docs/dev/process_checklist.md`：当前任务完成状态。
3. `docs/dev/project_structure.md`：目录结构。
4. `docs/dev/architecture.md`：公共 API、依赖方向和数据流。
5. `requirements.txt`：开发环境快照，仅用于问题复查，不是安装规范。
6. `docs/adr/`：架构与流程决策记录（决策的权威来源；版本/进度状态见
   `docs/dev/process_checklist.md` 与 `docs/dev/release_process.md`）。

Method Card 描述论文、公式、实现边界和复现规格，但不能单独证明算法或论文复现已经完成。

## 2. 开发环境

项目支持 Python 3.10、3.11 和 3.12，推荐使用 3.11。

```bash
uv venv --python 3.11
uv sync --python 3.11 --extra dev --extra torch
```

不使用 uv 时：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,torch]"
```

依赖策略：

- 可安装依赖只在 `pyproject.toml` 维护。
- 本项目是 library，CI 需要验证声明范围内的最新可解析依赖，因此不提交 `uv.lock`。
- `requirements.txt` 是一次开发环境的精确快照，不用于 CI，也不要与 `pyproject.toml` 手工双向同步。
- 新增仅开发期工具放入 `dev`；模型运行依赖放入对应 runtime extra。

## 3. 分支与提交

不要直接在 `main` 上开发。每个独立变更使用一个分支：

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/<short-name>
```

推荐提交前缀：`feat:`、`fix:`、`test:`、`docs:`、`ci:`、`refactor:`、`chore:`。

一个 PR 应只处理一个清晰主题。不要混入无关重构、生成文件或本地配置；不得提交 SSH 私钥、token、`.env` 或数据访问凭据。

## 4. 公共 API 规则

- 标签必须通过公共规范化/校验接口处理，PU 公共表示为 `{1, 0}`。
- 分类器遵循 `BasePUClassifier`；类先验估计器遵循 `BasePriorEstimator`。
- 新增方法必须登记 registry metadata，并明确 assumption、scenario、backend、source status 和 implementation status。
- 可选依赖不得让基础包导入失败；缺少依赖时应延迟导入并给出可行动错误。
- `api_only` 不得伪装成可训练实现；clean-room 核心不得标为官方数值复现。
- 破坏性 API 修改必须更新架构、示例、测试和版本说明。

## 5. 测试与质量门禁

提交前运行：

```bash
python -m pytest -q
python scripts/check_format.py       # ruff check + ruff format --check（全目录）
python scripts/check_test_quality.py
python scripts/check_doc_links.py
python scripts/check_project_metadata.py
python scripts/check_math_rendering.py
python scripts/check_skill_sync.py
uv build
git diff --check
```

测试应按风险选择 marker：

- `unit`：局部行为、参数和边界。
- `math`：手工可计算的公式 golden test。
- `property`：数学不变量。
- `contract`：所有 native estimator 共享的 API 契约。
- `paper`：依赖论文数据、源码或配置的复现实验。
- `slow`：不适合每次本地反馈的测试。

每个普通测试文件最多 15 个测试方法，并覆盖 basic、param、edge、deterministic 全部四类（缺任一分类即门禁失败，严格模式为本地与 CI 共同默认）；具体规则由 `scripts/check_test_quality.py` 执行。

### 5.1 单源助手（必须复用）

以下助手是跨模块的单一实现，新代码必须复用而非内联重写：

| 助手 | 位置 | 用途 |
|---|---|---|
| `canonical_hash` | `pu_toolbox/utils/serialization.py`（`benchmarks/_common.py` 为兼容 re-export） | 严格 JSON 规范化哈希 |
| `json_safe` | `pu_toolbox/utils/serialization.py` | 严格 JSON 兼容转换 |
| `sigmoid_stable` | `pu_toolbox/utils/activations.py` | 数值稳定 sigmoid |
| `rbf_weights` | `pu_toolbox/utils/basis.py` | RBF 核权重（六处收敛单源） |
| `validate_true_binary_labels` | `pu_toolbox/core/validation.py` | y_true 值域校验 |
| `check_scalar_in_range` | `pu_toolbox/core/validation.py` | 标量范围校验（`inclusive=False` 为开区间） |
| `solve_prior_from_positive_fraction` / `stable_centroid_denominator` | `pu_toolbox/estimators/risk/_class_prior.py` | 类先验推导与质心项 1−2ph 稳定性检查 |
| `git_worktree_dirty` | `benchmarks/_common.py` | git 脏状态检测（`exclude` 参数排除 runner 自身输出） |

**代谢率红线**：PR 评审时对增量代码做单源检查——发现 **>1 处单源违规为黄线**（该 PR 必须包含收敛治理）；**≥3 处或同一概念第 3 次分裂为红线**，触发该区域的结构性重构评估。历史治理记录见 `docs/dev/architecture_principles.md` §5 与 `docs/adr/0001-architecture-governance.md`。

## 6. 论文方法和 benchmark

状态必须严格区分：

- **Method Card 完成**：论文内容和接口规格已整理。
- **native/clean-room 完成**：本项目独立实现可运行，并通过代码测试。
- **paper-like 完成**：按论文规格设计实验，但可能使用替代数据或现代环境。
- **官方复现完成**：官方数据、源码/配置、历史环境和统计协议均已锁定并实际运行。

Benchmark runner、配置、来源锁和结果应一起更新。提交结果 CSV/manifest 时必须能够由仓库内配置和命令重新生成，并在 README 中说明 seed 数、数据来源和环境限制。

禁止使用测试集真值选择类先验、阈值或超参数。合成数据中的 `y_true` 和 propensity 只用于评价或明确标记的 oracle sensitivity。

## 7. 文档与进度

完成任务时至少检查：

- `README.md` 的用户入口和测试数量。
- `docs/README.md` 的文档索引。
- `docs/dev/architecture.md` 的 API 和依赖方向。
- `docs/dev/project_structure.md` 的真实文件结构。
- `docs/dev/process_checklist.md` 的勾选与发布状态。

不要一次性把所有待办标记完成。只有代码、测试、文档和要求的实际实验都完成后才能勾选对应任务。

## 8. Pull Request

推送功能分支后创建 PR：

```bash
git push -u origin feature/<short-name>
```

PR 描述应包含问题、实现边界、验证命令、实验产物和未解决风险。CI 的 Python 矩阵、静态质量门禁及 wheel 安装冒烟全部通过后再合并。
