# 双架构能力契约（阶段 2）设计规范：nnpu encoder 试点

- 日期：2026-08-30
- 状态：已与用户确认（brainstorming 三决策 + 两批设计呈现均获批）
- 上游：dual_architecture_plan.md §5 阶段 2（"以 nnpu 为首个试点"）
- 前置：阶段 0（能力契约）与阶段 1（双架构整理）已完成；本机无 CUDA，GPU 执行级测试在 `D:\CodexProject\FunctionTest` 验收环境执行（用户已配置 CUDA）

## 1. 背景与范围

上游计划阶段 2「以 `nnpu` 为首个试点」：增加可选 `encoder=None`；`None`
继续使用原有默认网络；传入 encoder 时只替换表征网络，不改变 PU loss
数学目标；并定义 encoder→head 的组合结构及与 `model` 的共存/优先级。

nnpu 现状（`pu_toolbox/estimators/risk/nnpu.py`）：

- `model: torch.nn.Module | None` 为**完整网络**（输出 raw score g(x)），
  默认 `nn.Linear(d, 1)`；fit 内 `copy.deepcopy(self.model)` 后训练；
- 能力声明：`native_architectures = {"mlp"}`、`input_ndims = {2}`、
  `encoder_parameter = None`、`trains_encoder = False`；
- 训练循环、early stopping、optimizer 重建、decision_function 全部围绕
  单一 `self.model_`；
- 传统 PU benchmark 的 nnpu 调优基线（r1，beta 0.3）基于 Linear 默认。

**范围内**：

1. nnpu 新增 `encoder` 参数 + encoder→head 组合结构（fit 内组合）；
2. 能力字段升级为 MLP/CNN 双架构声明（随之自动获得 pipeline/UI/报告
   联动，见 §4）；
3. 回归测试：二维/四维输入、CV fold 隔离、seed、设备、保存加载；
4. CPU/GPU 执行级测试：新增 `gpu` pytest marker 入 tests/，在
   FunctionTest（CUDA）环境执行。

**范围外（明确不做）**：

- `dist_pu`：计划原文"试点稳定后再评估，避免同时改造多个训练循环"；
- 阶段 3（self_pu / dgpu，长期搁置）、阶段 4（adapter_architectures，
  YAGNI）；
- 公共 compose helper 提取：试点阶段组合逻辑留在 nnpu 内，dist_pu
  接入时再提取（届时本阶段测试即安全网，YAGNI）。

## 2. 已确认决策

| 决策点 | 结论 |
|---|---|
| encoder 与 model 共存语义 | **model 复用为 head**：encoder 提供时 model 被解释为 head（接在 encoder 后输出 score），`nn.Sequential(encoder, head)` 组合；encoder=None 时 model 语义不变（完整网络）。同传合法，非互斥 |
| 默认网络 | encoder=None 时保持现状 `nn.Linear(d, 1)`。计划原文"原有 MLP"为笔误（泛指原有默认网络），本文档勘误；表格行为、benchmark 基线、调优结果全部不动 |
| 组合实现方式 | 方案 A：fit 内组合为单一 `model_ = nn.Sequential(encoder_, head)`，训练循环/early stopping/optimizer/decision_function 零改动 |
| 默认 head | model 为 None 时 head = `nn.Linear(representation_dim, 1)`（representation_dim 由 probe 得出） |
| head 维度不匹配 | 不预校验，由 torch 前向自然报 shape error（与现状 model 语义一致，YAGNI） |
| GPU 测试组织 | 新增 `gpu` marker 注册进 pyproject，测试写入仓库 tests/，无 CUDA 自动 skip；FunctionTest 环境装包后 `pytest -m gpu` 真执行 |
| CV 隔离测试 | 参数化扩展现有 `tests/integration/test_cv_fold_isolation.py`（数据协议泛化为 classifier 工厂），nnpu 复用同一断言套件 |
| 既有测试策略 | 新测试放新文件；既有 `test_nnpu.py`、红线文件（test_pipeline_deep.py / test_classifier_baseline.py）零改动 |

## 3. nnpu 类改造设计

### 3.1 构造函数

`__init__` 新增 keyword-only 参数（紧跟 `model` 之后，与 infomax/wconpu
先例同为 keyword-only）：

```python
def __init__(
    self,
    model: torch.nn.Module | None = None,
    *,
    encoder: torch.nn.Module | None = None,
    class_prior: float | None = None,
    ...  # 其余参数逐字不变
)
```

docstring 补充 encoder 语义：替换表征网络；提供时 model 退化为 head；
encoder 与 model 同传合法，组合使用。Module 对象被 sklearn
clone/get_params 深拷贝——与 infomax 既有模式一致（已验证可用）。

### 3.2 fit 变更

仅 `self.encoder is not None` 时走新分支，encoder=None 分支逐字不动：

1. `validate_pu_X_y(..., allow_nd=self.encoder is not None)`——
   四维输入放行（与 infomax 先例一致）；**validation_data 分支的
   `validate_pu_X_y` 同样传入 `allow_nd`**（nnpu.py:301-305 现状无该
   参数，否则 encoder 模式下带 early stopping 的四维 X_val 会被拒绝）；
2. `encoder_ = copy.deepcopy(self.encoder).to(device)`——fit 内深拷贝，
   与 CV fold 隔离语义（"共享构造 + 逐 fit 深拷贝"）一致；
3. probe：`encoder_.eval()` + `X[:1]` 前向 →
   `validate_encoder_features(probe.flatten(start_dim=1),
   encoder_param_name="encoder")` 得 `representation_dim`
   （BatchNorm 需 eval 模式，同 infomax 先例）；
4. head 确定：`model` 非 None → `copy.deepcopy(self.model)` 复用为
   head；否则 `nn.Linear(representation_dim, 1)`；head 与 encoder 同
   device；
5. `self.model_ = nn.Sequential(encoder_, head)`——此后训练循环、
   early stopping（validation_data 路径）、optimizer 重建
   （`type(self.optimizer)(self.model_.parameters(), **defaults)` 自动
   含 encoder 参数）、`_decision_function`、`evaluate_pu_risk` 全部
   复用现状代码，零改动。

PU loss 数学目标不变：`_nnpu_train_step` 的输入仍是
`self.model_` 输出的 scores，nnPU 风险估计器（Kiryo et al. 2017）语义
不受影响——只替换了表征网络。

### 3.3 能力字段升级

```python
native_architectures = frozenset({"mlp", "cnn"})
input_ndims = frozenset({2, 4})
encoder_parameter = "encoder"
trains_encoder = True
```

## 4. 平台联动（声明驱动，零额外代码）

- **Pipeline**：`check_architecture_capability` + `fresh_estimator`
  自动注入共享 encoder → CV fold 隔离语义自动继承（fit 内 deepcopy
  保证）；
- **UI**：`cnn_candidates()` 自动包含 nnpu（阶段 1 纯函数从 registry
  元数据推导）；骨架 selectbox 已有；
- **`list-methods`**：能力列自动更新；
- **报告 provenance**：CNN 运行时自动报 `native_cnn` + backbone +
  encoder 摘要（`in_channels` 为图像通道数）；MLP 运行时
  `native_mlp` + encoder None；
- **注册表**：能力字段类属性权威 + `_SYNC_FIELDS` 自动镜像，
  `builtin_methods.py` 无需改动（已确认条目无显式能力字段）。

**声明变化的涟漪（既有测试必然更新）**：

| 文件 | 变更 |
|---|---|
| `tests/unit/ui/test_cnn_candidates.py` | 锁定集合 `{"infomax_pu", "weighted_contrastive_pu"}` → 增加 `"nnpu"` |
| `tests/contract/test_capability_declarations.py` | `_EXPECTED_DECLARATIONS` 的 nnpu 条目 → `({"mlp","cnn"}, {2,4}, "encoder", True)` |
| `tests/unit/cli/test_info.py:87` | list-methods Input/Arch 列精确断言同步为 `"2,4"` / `"mlp,cnn"`（Task 1 实施发现，与 infomax 同行风格一致） |

## 5. 测试设计

| 文件 | 内容 |
|---|---|
| `tests/unit/estimators/test_nnpu_encoder.py`（新建） | ① encoder + 二维输入：训练生效、`model_` 为 Sequential、encoder 权重在变；② encoder + 四维输入：训练 + predict round-trip；③ encoder + 自定义 head 同传：组合正确、head 参数参与训练；④ encoder 输出非法（非 2D）→ ValueError；⑤ 无 encoder + 四维输入 → 拒绝（fail-fast）；⑥ encoder=None 默认路径 = `nn.Linear(d,1)` 回归；⑦ 同 seed 两次 fit 权重一致（seed 确定性） |
| `tests/integration/test_cv_fold_isolation.py`（参数化扩展） | 现有数据协议（`[1×4, 0×8, 1×4, 0×8]` 交错布局）泛化为 classifier 工厂参数，nnpu 复用断言套件（防假阳性 / 起点无污染 / 折间隔离 / 模板不被训练） |
| `tests/integration/test_nnpu_pipeline_cnn.py`（新建） | `PUPipeline(architecture="cnn", classifier="nnpu")` 端到端：fit_evaluate 成功、provenance = `native_cnn` + backbone + encoder 摘要；`architecture="mlp"` 对照报 `native_mlp` + encoder None；save/load round-trip（pickle dump/load 后 predict 一致） |
| `tests/unit/estimators/test_nnpu_gpu.py`（新建，gpu marker） | CUDA 可用时：nnpu + cnn13 真实训练 + predict + 参数设备一致性；`torch.cuda.is_available()` 不满足自动 skip。本机 skip，FunctionTest 环境 `pytest -m gpu` 真执行 |
| `pyproject.toml` | markers 列表注册 `"gpu: tests that require a CUDA GPU (auto-skip without one)"` |

## 6. 兼容性影响清单

| 项 | 影响 |
|---|---|
| encoder=None 默认路径 | 逐字不动：现有 `tests/unit/estimators/test_nnpu.py`、`check_baseline_configs` 门禁锁定的 nnpu 基线配置（beta=0.3 等）、七轮调优结果、benchmark 基线全部有效 |
| 红线文件 | `tests/integration/test_pipeline_deep.py`、`tests/contract/test_classifier_baseline.py` 零改动 |
| 既有 test_nnpu.py | 零改动（新 encoder 分支全在新文件） |
| sklearn clone / get_params | encoder 为 Module 被 deepcopy——与 infomax 既有模式一致 |
| UI / CLI / run_config | 声明驱动自动生效；encoder 由 pipeline 管理（`_MANAGED_CLASSIFIER_PARAMS` 已含），用户侧无感知 |

## 7. 验收标准

1. 新增测试全绿（单元 + CV 隔离 + pipeline 端到端）；
2. 快速测试零回归，两个红线文件全程未改；
3. 7 项质量门禁全过（pyproject gpu marker 注册后 format 等门禁同步）；
4. 报告目检：nnpu + `architecture="cnn"` → provenance `native_cnn` +
   encoder 摘要；`architecture="mlp"` → `native_mlp` + encoder None；
5. `list-methods` 中 nnpu 行显示 cnn 能力；
6. GPU 测试在 FunctionTest（CUDA）环境实际执行通过（用户环境验收）；
7. 设计文档蒸馏进 `dual_architecture_plan.md` §5 阶段 2「实施结果」，
   `process_checklist.md` 登记未发布条目。

## 8. 实施计划（TDD 六任务）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** nnpu 支持可选 encoder 注入（MLP/CNN 双架构声明），encoder→head 在 fit 内组合为单一 model_，训练循环零改动。

**Architecture:** nnpu 类内新增 `encoder` 参数；encoder 提供时 fit 深拷贝 encoder、probe 得 representation_dim、复用 `model` 为 head（默认 `Linear(rep_dim, 1)`），组合 `nn.Sequential(encoder_, head)` 后走现状训练路径。能力字段升级后 pipeline/UI/报告声明驱动自动联动。

**Tech Stack:** Python + torch（现有依赖，不新增）；pytest（unit/integration/gpu markers）；uv 命令前缀。

## Global Constraints

- 红线文件零改动：`tests/integration/test_pipeline_deep.py`、`tests/contract/test_classifier_baseline.py`
- 既有 `tests/unit/estimators/test_nnpu.py` 零改动（新 encoder 分支测试全在新文件）
- nnpu 的 encoder=None 分支代码逐字不动；默认网络保持 `nn.Linear(d, 1)`
- 每新建测试文件必须登记 `scripts/check_test_quality.py` 的 PARTIAL_COVERAGE（4 类别 basic/param/edge/determ，理由英文、与兄弟条目风格一致）
- 锁定声明（与契约测试一致）：`nnpu = (frozenset({"mlp","cnn"}), frozenset({2,4}), "encoder", True)`；cnn_candidates 锁定集合 `{"infomax_pu", "weighted_contrastive_pu", "nnpu"}`
- 所有 Python 命令前缀 `uv run`；git commit 不加 Co-Authored-By；工作分支 `feature/dual-arch-phase2`
- 修改任何文件前先 Read 该文件

---

### Task 1: 能力声明升级（契约测试先行）

**Files:**
- Modify: `tests/contract/test_capability_declarations.py:82`
- Modify: `tests/unit/ui/test_cnn_candidates.py:27-29`
- Modify: `pu_toolbox/estimators/risk/nnpu.py:111-114`

**Interfaces:**
- Produces: nnpu 类属性 `native_architectures={"mlp","cnn"}`、`input_ndims={2,4}`、`encoder_parameter="encoder"`、`trains_encoder=True`（Task 2-6 全部依赖此声明）

- [ ] **Step 1: 先改契约测试（RED）**

`test_capability_declarations.py:82`：

```python
    "nnpu": (frozenset({"mlp"}), frozenset({2}), None, False),
```
改为：

```python
    "nnpu": (frozenset({"mlp", "cnn"}), frozenset({2, 4}), "encoder", True),
```

`test_cnn_candidates.py:27-29`：

```python
def test_cnn_candidates_matches_current_declarations():
    """Phase-0 declarations: only infomax/wconpu support cnn today."""
    assert cnn_candidates() == {"infomax_pu", "weighted_contrastive_pu"}
```
改为：

```python
def test_cnn_candidates_matches_current_declarations():
    """Phase-2 declarations: infomax/wconpu/nnpu support cnn today."""
    assert cnn_candidates() == {"infomax_pu", "weighted_contrastive_pu", "nnpu"}
```

- [ ] **Step 2: 跑测试验证失败**

Run: `uv run pytest tests/contract/test_capability_declarations.py::test_deep_capability_declarations tests/unit/ui/test_cnn_candidates.py -v`
Expected: 两处 FAIL（nnpu 声明为旧值 / 候选集缺 nnpu）

- [ ] **Step 3: 更新 nnpu 类属性（GREEN）**

`nnpu.py:111-114`：

```python
    native_architectures = frozenset({"mlp"})
    input_ndims = frozenset({2})
    encoder_parameter = None
    trains_encoder = False
```
改为：

```python
    native_architectures = frozenset({"mlp", "cnn"})
    input_ndims = frozenset({2, 4})
    encoder_parameter = "encoder"
    trains_encoder = True
```

- [ ] **Step 4: 跑测试验证通过**

Run: `uv run pytest tests/contract/test_capability_declarations.py tests/unit/ui/test_cnn_candidates.py -v`
Expected: PASS。注意：此时 `test_declarations_are_legal` 会因 `encoder_parameter="encoder"` 不在 `__init__` 签名而失败（`encoder_parameter in inspect.signature(cls.__init__).parameters` 断言）——预期 RED 的一部分，Task 2 完成后转绿。

- [ ] **Step 5: Commit**

```bash
git add tests/contract/test_capability_declarations.py tests/unit/ui/test_cnn_candidates.py pu_toolbox/estimators/risk/nnpu.py
git commit -m "test(phase2): 契约测试先行——nnpu 能力声明升级为 MLP/CNN 双架构（RED）"
```

---

### Task 2: encoder 参数 + fit 组合逻辑（核心 TDD）

**Files:**
- Create: `tests/unit/estimators/test_nnpu_encoder.py`
- Modify: `pu_toolbox/estimators/risk/nnpu.py`（__init__ 签名 + docstring + fit 的 validate 行 + Build model 段 + validation_data 段 + import）
- Modify: `scripts/check_test_quality.py`（PARTIAL_COVERAGE 登记）

**Interfaces:**
- Consumes: Task 1 的 nnpu 类属性声明
- Produces: `clf.encoder_` 属性（fit 后，深拷贝的 encoder；与 infomax/wconpu 属性先例一致——Task 3 CV 断言依赖此属性）；`clf.model_` 为 `nn.Sequential(encoder_, head)`（encoder 路径）

- [ ] **Step 1: 写测试文件（RED）**

新建 `tests/unit/estimators/test_nnpu_encoder.py`：

```python
# ruff: noqa: N802, N803, N806
"""nnPU encoder-injection tests (dual_architecture_plan.md §5 阶段 2).

Covers: Sequential composition, custom head via ``model``, 4-D input paths,
fail-fast without encoder, default-Linear regression, seed determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier  # noqa: E402

pytestmark = [pytest.mark.unit]


class _TinyMLPEncoder(torch.nn.Module):
    """2-D in → rep_dim out (no torchvision dependency)."""

    def __init__(self, in_features=5, rep_dim=8):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, 16), torch.nn.ReLU(), torch.nn.Linear(16, rep_dim)
        )

    def forward(self, x):
        return self.net(x)


class _TinyCNNEncoder(torch.nn.Module):
    """4-D (C,6,6) in → rep_dim out (no torchvision dependency)."""

    def __init__(self, in_channels=1, rep_dim=8):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, 4, 3, padding=1)
        self.fc = torch.nn.Linear(4 * 6 * 6, rep_dim)

    def forward(self, x):
        return self.fc(self.conv(x).relu().flatten(start_dim=1))


class _ScalarEncoder(torch.nn.Module):
    """Outputs 1-D per-sample scalars — invalid representation.

    ``validate_encoder_features`` rejects any output that is not 2-D
    after ``flatten(start_dim=1)``; a 1-D output survives flattening
    unchanged and triggers the ValueError (a 3-D output would be
    flattened into a legal 2-D shape, so 1-D is the right trigger).
    """

    def forward(self, x):
        return x.mean(dim=-1) if x.ndim == 2 else x.mean(dim=(-3, -2, -1))


def _table_data(n=40, d=5, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.normal(0.0, 1.0, size=(n, d)).astype(np.float32)
    y_pu = np.concatenate([np.ones(10, dtype=int), np.zeros(n - 10, dtype=int)])
    return X, y_pu


def _image_data(n=24, channels=1, size=6, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.normal(0.5, 0.3, size=(n, channels, size, size)).astype(np.float32)
    y_pu = np.concatenate([np.ones(6, dtype=int), np.zeros(n - 6, dtype=int)])
    return X, y_pu


def _snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _fit_encoder_clf(X, y_pu, *, encoder, model=None, seed=42, val=None):
    clf = NonNegativePUClassifier(
        model=model,
        encoder=encoder,
        class_prior=0.3,
        max_epochs=1,
        batch_size=16,
        random_state=seed,
        device="cpu",
    )
    clf.fit(X, y_pu, validation_data=val)
    return clf


def test_encoder_with_2d_input_composes_sequential_and_trains():
    X, y_pu = _table_data()
    encoder = _TinyMLPEncoder()
    initial = _snapshot(encoder)
    clf = _fit_encoder_clf(X, y_pu, encoder=encoder)
    assert isinstance(clf.model_, torch.nn.Sequential)
    assert clf.encoder_ is clf.model_[0]
    assert isinstance(clf.model_[-1], torch.nn.Linear)
    assert clf.model_[-1].in_features == 8  # rep_dim
    assert clf.model_[-1].out_features == 1
    # Training took effect on the encoder and the default head.
    assert not torch.equal(next(iter(_snapshot(clf.encoder_).values())), next(iter(initial.values())))
    assert clf.decision_function(X[:8]).shape == (8,)


def test_custom_head_via_model_parameter_is_deepcopied_and_trained():
    X, y_pu = _table_data()
    head = torch.nn.Linear(8, 1)
    head_initial = {k: v.detach().clone() for k, v in head.state_dict().items()}
    clf = _fit_encoder_clf(X, y_pu, encoder=_TinyMLPEncoder(), model=head)
    assert clf.model_[1] is not head  # deepcopied, not the caller's instance
    after = _snapshot(clf.model_[1])
    assert not torch.equal(after["weight"], head_initial["weight"])


def test_invalid_encoder_output_raises_value_error():
    X, y_pu = _table_data()
    with pytest.raises(ValueError, match="encoder"):
        _fit_encoder_clf(X, y_pu, encoder=_ScalarEncoder())


def test_default_linear_model_preserved_without_encoder():
    X, y_pu = _table_data(d=5)
    clf = NonNegativePUClassifier(
        class_prior=0.3, max_epochs=1, batch_size=16, random_state=42, device="cpu"
    )
    clf.fit(X, y_pu)
    assert isinstance(clf.model_, torch.nn.Linear)
    assert clf.model_.in_features == 5


def test_encoder_with_4d_input_trains_and_predicts():
    X, y_pu = _image_data()
    clf = _fit_encoder_clf(X, y_pu, encoder=_TinyCNNEncoder())
    scores = clf.decision_function(X)
    assert scores.shape == (24,)


def test_4d_input_without_encoder_is_rejected():
    X, y_pu = _image_data()
    clf = NonNegativePUClassifier(
        class_prior=0.3, max_epochs=1, batch_size=16, random_state=42, device="cpu"
    )
    with pytest.raises(ValueError):
        clf.fit(X, y_pu)


def test_4d_validation_data_accepted_with_encoder():
    X, y_pu = _image_data()
    X_val, y_val = _image_data(n=8, seed=2)
    clf = _fit_encoder_clf(X, y_pu, encoder=_TinyCNNEncoder(), val=(X_val, y_val))
    assert clf.decision_function(X_val).shape == (8,)


def test_seed_determinism_with_encoder():
    X, y_pu = _table_data()
    clf1 = _fit_encoder_clf(X, y_pu, encoder=_TinyMLPEncoder(), seed=42)
    clf2 = _fit_encoder_clf(X, y_pu, encoder=_TinyMLPEncoder(), seed=42)
    for (k1, v1), (k2, v2) in zip(clf1.model_.state_dict().items(), clf2.model_.state_dict().items()):
        assert k1 == k2
        assert torch.equal(v1, v2)
```

- [ ] **Step 2: 跑测试验证失败**

Run: `uv run pytest tests/unit/estimators/test_nnpu_encoder.py -v`
Expected: FAIL——`NonNegativePUClassifier` 无 `encoder` 参数（TypeError: unexpected keyword argument 'encoder'）

- [ ] **Step 3: 实现（nnpu.py）**

import 段（`from ...core.validation import ...` 之后、`from ...losses...` 之前）增加：

```python
from ..deep._validation import validate_encoder_features
```

`__init__` 签名（`model` 之后）：

```python
    def __init__(
        self,
        model: torch.nn.Module | None = None,  # noqa: F821
        *,
        encoder: torch.nn.Module | None = None,  # noqa: F821
        class_prior: float | None = None,
        ...
```
并在 `super().__init__()` 之后、`self.model = model` 之后增加 `self.encoder = encoder`。

docstring Parameters 段（`model` 条目之后）追加：

```python
    encoder : torch.nn.Module or None, default None
        Feature encoder replacing the default raw-input model.  When
        provided, ``model`` (if any) is used as the score head stacked
        on the encoder; otherwise a default ``nn.Linear(rep_dim, 1)``
        head is created.  ``encoder=None`` keeps the original behaviour.
```

fit 主校验行改为：

```python
        X, y_pu = validate_pu_X_y(
            X,
            y_pu,
            allow_nd=self.encoder is not None,
            estimator_name="NonNegativePUClassifier",
        )
```

Build model 段（现状 `if self.model is not None:` 三行 + `# ── Device ──` 段）整体替换为：

```python
        # ── Build model ───────────────────────────────────────────
        device = resolve_device(self.device)
        if self.encoder is not None:
            self.encoder_ = copy.deepcopy(self.encoder).to(device)
            # Probe in eval mode: fresh BatchNorm layers reject 1-sample
            # batches in training mode and must not accumulate stats
            # from the probe.
            self.encoder_.eval()
            with torch.no_grad():
                probe = self.encoder_(
                    torch.as_tensor(X[:1], dtype=torch.float32, device=device)
                )
            representation_dim = validate_encoder_features(
                probe.flatten(start_dim=1), encoder_param_name="encoder"
            )
            if self.model is not None:
                head = copy.deepcopy(self.model)
            else:
                head = torch.nn.Linear(representation_dim, 1)
            self.model_ = torch.nn.Sequential(self.encoder_, head)
        elif self.model is not None:
            self.model_ = copy.deepcopy(self.model)
        else:
            self.model_ = torch.nn.Linear(d, 1)
        self.model_.to(device)
```

validation_data 段（`X_val, y_val_pu = validate_pu_X_y(...)`）改为：

```python
            X_val, y_val_pu = validate_pu_X_y(
                X_val,
                y_val_pu,
                allow_nd=self.encoder is not None,
                estimator_name="NonNegativePUClassifier[val]",
            )
```

- [ ] **Step 4: 跑测试验证通过**

Run: `uv run pytest tests/unit/estimators/test_nnpu_encoder.py tests/unit/estimators/test_nnpu.py tests/contract/test_capability_declarations.py -v`
Expected: PASS（含 Task 1 遗留的 `test_declarations_are_legal` RED 转绿）

- [ ] **Step 5: 登记测试质量门禁**

`scripts/check_test_quality.py` PARTIAL_COVERAGE 增加：

```python
    "test_nnpu_encoder.py": {
        "basic": (
            "encoder-injection behavior assertions on tiny local stubs (fit/predict/Sequential "
            "composition); pipeline-path behavior is covered by test_nnpu_pipeline_cnn and "
            "test_cv_fold_isolation"
        ),
        "param": (
            "constructor-parameter error paths (invalid encoder output, missing encoder) are "
            "the param/edge cases here; scalar-parameter validation is covered by the "
            "existing test_nnpu.py"
        ),
        "edge": (
            "input-boundary scenarios (4-D without encoder rejected, 3-D representation "
            "rejected, 4-D validation_data accepted) are covered; ratio/boundary warnings "
            "live in validate_pu_X_y tests"
        ),
        "determ": (
            "same-seed double-fit state_dict equality is asserted (test_seed_determinism_with_encoder); "
            "no new randomness sources beyond the existing manual_seed path"
        ),
    },
```

Run: `uv run python scripts/check_test_quality.py`
Expected: exit 0

- [ ] **Step 6: Commit**

```bash
git add tests/unit/estimators/test_nnpu_encoder.py pu_toolbox/estimators/risk/nnpu.py scripts/check_test_quality.py
git commit -m "feat(nnpu): encoder 注入试点——fit 内组合 Sequential(encoder, head)

- __init__ 新增 keyword-only encoder 参数；model 复用为 head
  （encoder=None 分支逐字不动，默认 Linear(d,1) 保持）
- fit：深拷贝 encoder + eval probe → validate_encoder_features 得
  representation_dim → 组合 nn.Sequential；训练循环零改动
- validation_data 分支 validate_pu_X_y 同样传 allow_nd（4D X_val 放行）
- 存 encoder_ 属性（与 infomax/wconpu 先例一致，供 CV 隔离断言）
- 测试 8 项：组合/自定义 head/非法输出/默认回归/4D/拒绝/确定性"
```

---

### Task 3: CV fold 训练隔离参数化扩展

**Files:**
- Modify: `tests/integration/test_cv_fold_isolation.py:39-72`

**Interfaces:**
- Consumes: Task 2 的 `clf.encoder_` 属性（nnpu 与 wconpu 同名断言）
- Produces: 无（测试锁定 nnpu 的 CV fold 隔离语义）

- [ ] **Step 1: 参数化测试函数**

`test_cv_fold_isolation.py` 的测试函数改为：

```python
@pytest.mark.integration
@pytest.mark.parametrize("classifier_name", ["wconpu", "nnpu"])
def test_cv_folds_do_not_leak_encoder_weights(classifier_name):
    X, y_pu = _image_data()
    pipe = PUPipeline(
        classifier=classifier_name,
        architecture="cnn",
        backbone="cnn13",
        cv=2,
        max_epochs=1,
        random_state=42,
        device="cpu",
    )
    pipe._encoder = build_encoder("cnn", backbone="cnn13", in_channels=3)
    template_initial = _snapshot(pipe._encoder)
    ...
```
（函数体其余断言逐字不动——`clf1.encoder_` / `clf2.encoder_` 两个算法都有）

- [ ] **Step 2: 跑测试验证通过**

Run: `uv run pytest tests/integration/test_cv_fold_isolation.py -v`
Expected: PASS ×2（wconpu 与 nnpu 各一）。nnpu 侧 max_epochs=1 经 pipeline 传参（pipeline.py:135 已声明 nnpu 接受 max_epochs；_models.py:145-146 传递）

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_cv_fold_isolation.py
git commit -m "test(nnpu): CV fold 隔离断言参数化覆盖 nnpu

共享 encoder 模板 2 折真实训练断言套件泛化为 classifier 工厂参数；
nnpu（fit 内深拷贝 encoder 语义）与 wconpu 同断言锁定'共享构造 +
逐 fit 深拷贝'契约（dual_architecture_plan §5 阶段 2）"
```

---

### Task 4: pipeline 端到端 + save/load round-trip

**Files:**
- Create: `tests/integration/test_nnpu_pipeline_cnn.py`
- Modify: `scripts/check_test_quality.py`（PARTIAL_COVERAGE 登记）

**Interfaces:**
- Consumes: Task 1 声明（pipeline CNN gate 放行）、Task 2 实现；`build_encoder`（包根导出，阶段 1）；provenance 4 字段（阶段 1 `_reporting.py`）
- Produces: 无

- [ ] **Step 1: 写测试文件（RED）**

新建 `tests/integration/test_nnpu_pipeline_cnn.py`：

```python
# ruff: noqa: E402, N802, N803, N806
"""nnPU CNN end-to-end through PUPipeline: provenance mapping + save/load
round-trip (dual_architecture_plan.md §5 阶段 2)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox import build_encoder  # noqa: E402
from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier  # noqa: E402
from pu_toolbox.workflows import PUPipeline  # noqa: E402


def _image_data(n=24, channels=3, size=8, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.normal(0.5, 0.3, size=(n, channels, size, size)).astype(np.float32)
    y_pu = np.concatenate(
        [
            np.ones(4, dtype=int),
            np.zeros(8, dtype=int),
            np.ones(4, dtype=int),
            np.zeros(8, dtype=int),
        ]
    )
    return X, y_pu


def _table_data(n=40, d=5, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.normal(0.0, 1.0, size=(n, d)).astype(np.float32)
    y_pu = np.concatenate([np.ones(10, dtype=int), np.zeros(n - 10, dtype=int)])
    return X, y_pu


@pytest.mark.integration
def test_nnpu_cnn_pipeline_reports_native_cnn_provenance():
    X, y_pu = _image_data()
    report = PUPipeline(
        classifier="nnpu",
        architecture="cnn",
        backbone="cnn13",
        cv=2,
        max_epochs=1,
        random_state=42,
        device="cpu",
    ).fit_evaluate(X, y_pu, class_prior=0.3, refit=False)
    p = report.provenance
    assert p["architecture"] == "native_cnn"
    assert p["backbone"] == "cnn13"
    assert p["encoder"] == {"backbone": "cnn13", "in_channels": 3}
    assert p["device"] == {"requested": "cpu", "resolved": "cpu"}


@pytest.mark.integration
def test_nnpu_mlp_pipeline_reports_native_mlp_provenance():
    X, y_pu = _table_data()
    report = PUPipeline(
        classifier="nnpu",
        architecture="mlp",
        cv=2,
        max_epochs=1,
        random_state=42,
    ).fit_evaluate(X, y_pu, class_prior=0.3, refit=False)
    p = report.provenance
    assert p["architecture"] == "native_mlp"
    assert p["backbone"] is None
    assert p["encoder"] is None


@pytest.mark.integration
def test_nnpu_encoder_model_survives_pickle_roundtrip():
    X, y_pu = _image_data()
    clf = NonNegativePUClassifier(
        encoder=build_encoder("cnn", backbone="cnn13", in_channels=3),
        class_prior=0.3,
        max_epochs=1,
        random_state=42,
        device="cpu",
    )
    clf.fit(X, y_pu)
    expected = clf.decision_function(X[:8])
    restored = pickle.loads(pickle.dumps(clf))
    np.testing.assert_allclose(restored.decision_function(X[:8]), expected)
```

- [ ] **Step 2: 跑测试验证失败**

Run: `uv run pytest tests/integration/test_nnpu_pipeline_cnn.py -v`
Expected: FAIL——`architecture='cnn' requires classifier 'nnpu' to declare an 'encoder' constructor parameter`（Task 1/2 尚未在分支上的假想 RED 文案；若在 Task 2 之后执行则可能已 PASS，直接进入 Step 3 验证）

- [ ] **Step 3: 跑测试验证通过**

Run: `uv run pytest tests/integration/test_nnpu_pipeline_cnn.py -v`
Expected: PASS ×3

- [ ] **Step 4: 登记测试质量门禁**

`scripts/check_test_quality.py` PARTIAL_COVERAGE 增加：

```python
    "test_nnpu_pipeline_cnn.py": {
        "basic": (
            "pipeline end-to-end provenance mapping and pickle round-trip assertions "
            "(single-intent integration locks); unit-level field assembly is covered by "
            "test_report_provenance.py"
        ),
        "param": (
            "no constructor-error surface beyond the pipeline gates themselves; pipeline "
            "error paths are covered by test_pipeline_deep.py"
        ),
        "edge": (
            "no new boundary scenarios; 4-D boundary behavior is covered by "
            "test_nnpu_encoder.py and validate_pu_X_y tests"
        ),
        "determ": (
            "seeded pipeline runs assert provenance values, not reproducibility; seed "
            "determinism is covered by test_nnpu_encoder.py and test_pipeline_deep.py"
        ),
    },
```

Run: `uv run python scripts/check_test_quality.py`
Expected: exit 0

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_nnpu_pipeline_cnn.py scripts/check_test_quality.py
git commit -m "test(nnpu): pipeline 端到端 provenance 映射 + pickle round-trip

CNN 运行报 native_cnn + backbone + encoder 摘要；MLP 运行报
native_mlp + backbone/encoder None；Sequential(encoder, head)
模型 pickle dump/load 后 predict 一致"
```

---

### Task 5: gpu marker + GPU 执行级测试

**Files:**
- Modify: `pyproject.toml:111-120`（markers 列表）
- Modify: `scripts/check_test_quality.py:48-56`（REGISTERED_MARKERS 同步）
- Create: `tests/unit/estimators/test_nnpu_gpu.py`
- Modify: `scripts/check_test_quality.py`（PARTIAL_COVERAGE 登记）

**Interfaces:**
- Consumes: Task 1/2（nnpu encoder 路径）、`build_encoder` 包根导出
- Produces: `gpu` marker（本机/CI 自动 skip；`D:\CodexProject\FunctionTest` 环境 `pytest -m gpu` 真执行）

- [ ] **Step 1: 注册 marker**

`pyproject.toml` markers 列表末尾追加：

```toml
    "gpu: tests that require a CUDA GPU (auto-skip without one)",
```

`scripts/check_test_quality.py` REGISTERED_MARKERS 集合追加 `"gpu",`（该集合注释为 "Registered markers from pyproject.toml (must stay in sync)"，AST 分析用其识别 marker 装饰器）。

- [ ] **Step 2: 写 GPU 测试文件**

新建 `tests/unit/estimators/test_nnpu_gpu.py`：

```python
# ruff: noqa: E402, N802, N803, N806
"""GPU execution-level tests for nnPU with a CNN encoder
(dual_architecture_plan.md §5 阶段 2). Auto-skip without CUDA; run on a
CUDA machine with ``pytest -m gpu``."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox import build_encoder  # noqa: E402
from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.unit,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


def _image_data(n=32, channels=3, size=8, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(0.5, 0.3, size=(n, channels, size, size)).astype(np.float32)
    y_pu = np.concatenate([np.ones(8, dtype=int), np.zeros(n - 8, dtype=int)])
    return X, y_pu


def test_nnpu_cnn_trains_and_predicts_on_gpu():
    X, y_pu = _image_data()
    clf = NonNegativePUClassifier(
        encoder=build_encoder("cnn", backbone="cnn13", in_channels=3),
        class_prior=0.4,
        max_epochs=1,
        batch_size=16,
        device="cuda",
        random_state=7,
    )
    clf.fit(X, y_pu)
    assert isinstance(clf.model_, torch.nn.Sequential)
    assert all(p.device.type == "cuda" for p in clf.model_.parameters())
    scores = clf.decision_function(X[:8])
    assert scores.shape == (8,)
```

- [ ] **Step 3: 本机验证 skip 路径**

Run: `uv run pytest tests/unit/estimators/test_nnpu_gpu.py -v`
Expected: SKIPPED（"CUDA not available"）——本机无 CUDA 的正确行为，不报错

- [ ] **Step 4: 登记测试质量门禁**

`scripts/check_test_quality.py` PARTIAL_COVERAGE 增加：

```python
    "test_nnpu_gpu.py": {
        "basic": (
            "GPU execution smoke (fit/predict/device assertions) behind a CUDA skipif; "
            "the same behavior is covered CPU-side by test_nnpu_encoder.py"
        ),
        "param": (
            "no parameter-error surface; parameter validation is covered by "
            "test_nnpu_encoder.py and test_nnpu.py"
        ),
        "edge": (
            "no boundary scenarios; boundary behavior is covered by "
            "test_nnpu_encoder.py"
        ),
        "determ": (
            "seeded GPU smoke asserts behavior, not reproducibility; seed determinism "
            "is covered by test_nnpu_encoder.py test_seed_determinism_with_encoder"
        ),
    },
```

Run: `uv run python scripts/check_test_quality.py`
Expected: exit 0

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml scripts/check_test_quality.py tests/unit/estimators/test_nnpu_gpu.py
git commit -m "test(nnpu): gpu marker 与 CUDA 执行级测试（无 CUDA 自动 skip）

pyproject 注册 gpu marker；check_test_quality REGISTERED_MARKERS 同步；
nnpu+cnn13 在 CUDA 上真实训练 + 设备一致性断言。本机/CI skip，
D:\\CodexProject\\FunctionTest（CUDA）环境 pytest -m gpu 真执行"
```

- [ ] **Step 6（验收标准 6，用户环境执行）**: 在 `D:\CodexProject\FunctionTest` 的 CUDA 环境安装本仓库包后运行 `pytest -m gpu tests/unit/estimators/test_nnpu_gpu.py -v`，结果留档（Task 6 汇报时引用）。

---

### Task 6: 文档收尾与全量验收

**Files:**
- Modify: `docs/dev/dual_architecture_plan.md`（§5 阶段 2 追加实施结果）
- Modify: `docs/dev/process_checklist.md`（发布状态登记）
- Modify: `docs/user/reference/api.md:19`（nnpu 行参数列加 encoder）
- Modify: `docs/README.md:46`（后追加阶段 2 设计文档索引行）
- Modify: `docs/dev/project_structure.md`（generate_structure.py --update 重新生成）

**Interfaces:**
- Consumes: Task 1-5 全部产出
- Produces: 无（收尾）

- [ ] **Step 1: 上游计划追加实施结果**

`dual_architecture_plan.md` §5 阶段 2 末尾追加：

```markdown
**实施结果（已完成，2026-08-30）**：nnpu 新增 keyword-only `encoder`
参数——提供时在 fit 内深拷贝 + eval probe 得 representation_dim，
`model` 复用为 score head（默认 `Linear(rep_dim, 1)`），组合为
`nn.Sequential(encoder_, head)`，训练循环与 PU loss 数学目标零改动；
`encoder=None` 分支逐字不动（默认 `Linear(d, 1)` 保持，计划原文
"原有 MLP"系笔误）。能力声明升级 MLP/CNN 双架构，pipeline
CV fold 隔离/UI 候选集/报告 provenance 声明驱动自动生效；新增 gpu
marker 与 CUDA 执行级测试（无 CUDA 自动 skip）。设计细节见
[2026-08-30-dual-arch-phase2-design.md](2026-08-30-dual-arch-phase2-design.md)。
```

- [ ] **Step 2: process_checklist 登记**

发布状态节追加：

```markdown
- 双架构阶段 2 nnpu encoder 试点（未发布，随下一版本发布）：nnpu
  新增 encoder 参数（model 复用为 head，fit 内 Sequential 组合）、
  MLP/CNN 双架构声明、CV fold 隔离与 pipeline 端到端测试、
  gpu marker + CUDA 执行级测试（详见
  docs/dev/dual_architecture_plan.md §5）
```

- [ ] **Step 3: api.md 更新**

`docs/user/reference/api.md:19` 的 nnpu 行参数列 `` `model` / `class_prior` / `loss` / `optimizer` `` 改为 `` `model` / `encoder` / `class_prior` / `loss` / `optimizer` ``。

- [ ] **Step 4: docs/README.md 索引**

`docs/README.md:46`（阶段 1 设计文档行）之后追加：

```markdown
| [dev/2026-08-30-dual-arch-phase2-design.md](dev/2026-08-30-dual-arch-phase2-design.md) | 双架构阶段 2 设计规范与实施计划：nnpu encoder 试点、encoder→head 组合契约、gpu marker |
```

- [ ] **Step 5: structure 重新生成**

Run: `uv run python scripts/generate_structure.py --update`
Expected: 新增 test_nnpu_encoder.py / test_nnpu_pipeline_cnn.py / test_nnpu_gpu.py 三项登记，无无关差异

- [ ] **Step 6: 全量验证**

Run:
```bash
uv run pytest tests/ -v -m "not slow"
uv run python scripts/check_test_quality.py
uv run python scripts/check_doc_links.py
uv run python scripts/check_project_metadata.py
uv run python scripts/check_math_rendering.py
uv run python scripts/check_skill_sync.py
uv run python scripts/check_baseline_configs.py
uv run python scripts/check_format.py
uv run python scripts/generate_structure.py --check
```
Expected: 快速测试零回归（含两个红线文件）、7 门禁全部 exit 0。红线文件确认：`git diff HEAD~1 -- tests/integration/test_pipeline_deep.py tests/contract/test_classifier_baseline.py` 为空

- [ ] **Step 7: 验收目检（验收标准 4/5）**

Run:
```bash
uv run python -c "from pu_toolbox.registry import list_algorithms, register_all_builtin_methods; register_all_builtin_methods(); m = next(x for x in list_algorithms() if x.name == 'nnpu'); print(m.native_architectures, m.input_ndims, m.encoder_parameter, m.trains_encoder)"
```
Expected: 输出 `frozenset({'cnn', 'mlp'}) frozenset({2, 4}) encoder True`

Run: `uv run pu-toolbox list-methods | grep -A1 "nnpu"`
Expected: nnpu 行的能力列显示 cnn（Arch 列含 cnn）

- [ ] **Step 8: Commit**

```bash
git add docs/dev/dual_architecture_plan.md docs/dev/process_checklist.md docs/user/reference/api.md docs/README.md docs/dev/project_structure.md
git commit -m "docs(phase2): 阶段 2 收尾——计划实施结果蒸馏 + 文档同步

- dual_architecture_plan.md §5 阶段 2 追加实施结果摘要（对称阶段 0/1）
- process_checklist.md 登记阶段 2 未发布条目（随下一版本发布）
- api.md nnpu 参数列补 encoder；docs/README.md 索引设计文档
- structure 重新生成登记 3 个新测试文件"
```
