# 双架构能力契约（阶段 0）设计规范

- 日期：2026-08-29
- 状态：已与用户确认（brainstorming 两个部分均获批）
- 上游：[`dual_architecture_plan.md`](dual_architecture_plan.md)（已对照 v1.11.0 代码现状核实修订，commit 65e7ee9）
- 分支：`feature/dual-arch-capability-contract`

## 1. 背景与范围

上游计划分为 5 个阶段。本规范仅覆盖**阶段 0：能力契约和文档，不改变默认行为**。

**范围内**：

1. Registry 增加架构能力字段（4 个）；
2. Pipeline 的 CNN gate 增加元数据并行校验（与签名检查结论一致，漂移时报错）；
3. `list-methods` 展示能力信息（加 2 列）；
4. 统一 encoder 输出校验（新 helper，接入 infomax_pu / wconpu）；
5. 契约测试强制新算法声明能力字段；
6. 文档：`architecture.md` schema 节更新 + 新算法接入模板。

**范围外（明确不做，归属后续阶段）**：

- 阶段 1：`build_encoder` 公共导出、报告 provenance 字段、UI 元数据驱动、CV fold 隔离测试；
- 阶段 2：nnpu `encoder` 参数试点；
- 阶段 3：self_pu / dgpu（长期搁置）；
- 阶段 4：`adapter_architectures` 字段（YAGNI，待真正有适配算法时再加）；
- 逐 fold 独立随机初始化（可选改进，非本阶段）。

## 2. 已确认决策

| 决策点 | 结论 |
|---|---|
| 本轮范围 | 仅阶段 0 |
| 字段集合 | 4 字段：`native_architectures` / `input_ndims` / `encoder_parameter` / `trains_encoder`；`tabular_only` 派生；`adapter_architectures` 留待阶段 4 |
| 消费点 | Pipeline 校验 + `list-methods` 展示（Registry API 查询自动获得新字段） |
| 接入模板 | 文档 + 契约测试强制 |
| 字段权威来源 | 类属性权威 + `_SYNC_FIELDS` 同步进 Registry |

## 3. 数据模型

### 3.1 基类默认值（`core/base.py`，tabular 默认）

```python
native_architectures: frozenset[str] = frozenset()   # 原生架构路径 ⊆ {"mlp", "cnn"}
input_ndims: frozenset[int] = frozenset({2})         # 支持输入维度 ⊆ {2, 4}
encoder_parameter: str | None = None                 # 接收 encoder 的构造参数名
trains_encoder: bool = False                         # 是否端到端训练注入的 encoder
```

用 `frozenset` 而非 `set`：类属性在类间与注册表间共享，必须不可变。

### 3.2 各算法声明

仅深度算法需显式声明；传统算法继承默认值：

| 算法 | native_architectures | input_ndims | encoder_parameter | trains_encoder |
|---|---|---|---|---|
| infomax_pu、weighted_contrastive_pu | frozenset({"mlp","cnn"}) | frozenset({2,4}) | "encoder" | True |
| self_pu | frozenset({"mlp"}) | frozenset({2,4}) | None（`backbone` 是完整模型） | False |
| nnpu、dist_pu、dgpu | frozenset({"mlp"}) | frozenset({2}) | None | False |
| 传统 9 个（elkan_noto/upu/pnu/centroid_pu/kldce/llsvm/pusb/pusb_kernel/lbe） | 继承默认 ∅ | 继承默认 {2} | None | False |

`tabular_only` 不单独声明，派生自 `native_architectures == frozenset()`。

### 3.3 Registry 侧（`registry/metadata.py`）

`AlgorithmMetadata` 增加同名 4 字段，默认值与基类一致，并增加派生 property：

```python
@property
def is_tabular_only(self) -> bool:
    return self.native_architectures == frozenset()
```

`_SYNC_FIELDS`（`registry/registry.py:124-133`）增加这 4 个字段名。现有同步机制**只同步类显式声明的字段**（`_sync_class_metadata_to_registry` 跳过基类继承默认），因此：

- 传统算法：类不声明 → 注册表条目保持 tabular 默认，两者天然一致；
- 深度算法：类显式声明 → 注册表自动镜像。

`frozenset` 引用共享安全（不可变）。同步函数中现有 tuple→list 转换逻辑不影响 frozenset。

## 4. 消费点

### 4.1 Pipeline gate 并行校验（`workflows/pipeline.py:219-236` 区域）

将并行校验提取为 `workflows/_models.py` 中的纯函数（便于单测）：

```python
def check_architecture_capability(cls, architecture: str) -> None:
    """architecture="cnn" 时校验签名检查与能力声明结论一致，不一致抛错。"""
```

`architecture="cnn"` 时的逻辑：

1. 签名结论：`declares_encoder_parameter(cls)`（现有逻辑不变）；
2. 能力结论：`"cnn" in cls.native_architectures and 4 in cls.input_ndims and cls.encoder_parameter is not None`；
3. **两者不一致 → `PipelineError`**（声明与签名漂移，fail-loud）；
4. 能力不支持 CNN 时，错误信息引用能力声明（如 "declares tabular_only: input_ndims={2}")。

`architecture="mlp"` 路径行为完全不变（默认值）。输入侧维度校验复用现有 `validate_pu_X_y(allow_nd=...)` 机制，阶段 0 仅在错误路径上引用能力声明增强提示信息，不新增输入校验逻辑。

### 4.2 `list-methods` 加列（`cli/info.py`）

表格从 5 列扩为 7 列：

```
Name / Family / Prior / Status / Input / Arch / Auto-inst
```

- Input：`2` 或 `2,4`；
- Arch：`mlp` / `mlp,cnn` / `tabular`（`meta.is_tabular_only` 时）。

数据直接从 `meta` 读取（同步后已有），无需逐行 resolve 类。

## 5. 统一 encoder 输出校验

新建 pu_toolbox/estimators/deep/_validation.py（`core/validation.py` 保持 torch-free）：

```python
def validate_encoder_features(features, *, encoder_param_name: str) -> int:
    """校验 encoder 输出：ndim == 2、torch.isfinite 全真、feature_dim >= 1；返回 feature_dim。"""
```

接入点（现有 probe 逻辑委托）：

- `infomax_pu.py:165-167`（InfoMaxPURepresentation.fit 一处 probe；InfoMaxPUClassifier.fit 经 `representation.shape[1]` 取维）；
- `weighted_contrastive_pu.py:174-178`。

有限性检查与项目输入契约（拒绝 NaN/Inf）对齐，属预期内的行为收紧。

## 6. 契约测试（tests/contract/test_capability_declarations.py，测试先行）

四组不变量，对全部 17 个注册分类器断言：

1. **声明合法性**：
   - `input_ndims` 非空且 ⊆ {2, 4}；
   - `native_architectures` ⊆ {"mlp", "cnn"}；
   - `encoder_parameter` 为 None 或真实存在于 `__init__` 签名；
   - `trains_encoder=True ⇒ encoder_parameter is not None`；
2. **跨机制一致性**：`"cnn" in native_architectures ⇐ encoder_parameter is not None`，且与 `declares_encoder_parameter(cls)` 结论一致；
3. **注册表同步**：`get_metadata(name)` 的 4 字段 == 类属性值（同时保证默认值对齐）；
4. **派生语义**：`is_tabular_only ⇔ native_architectures == frozenset()`。

另加 `check_architecture_capability` 的单元测试：签名/能力漂移场景抛错、一致场景通过。

## 7. 文档变更

- `docs/dev/architecture.md` 注册表 schema 节补充 4 字段 + `is_tabular_only`（`metadata.py` docstring 引用该节，保持引用准确）；
- 新建 docs/dev/new_algorithm_template.md：声明清单 + 测试清单（§9 最低要求 + 契约测试自动门禁说明）+ 最小示例片段；
- 完成时更新 `docs/dev/process_checklist.md`（dev-workflow 步骤 4）。

## 8. 兼容性影响清单

| 现有测试 | 影响 |
|---|---|
| `tests/unit/cli/test_info.py` | 表头/列断言更新（+Input/+Arch 两列） |
| 注册表元数据相关单测 | 若存在字段枚举/数量断言需更新（实现时 grep 确认） |
| `tests/integration/test_pipeline_deep.py` | 不变——签名检查仍生效，现有算法能力一致，无新错误路径 |
| 传统算法全部测试 | 零变化（继承默认） |

## 9. 验收标准

1. 快速测试全绿（预期内更新仅 test_info.py 等）；新增契约测试 4 组不变量全绿；
2. 现有 WConPU/InfoMax MLP/CNN 集成测试**不改一行且通过**（零回归的最强证明）；
3. 7 项质量门禁全过；
4. `list-methods` 7 列输出对齐可读。

## 10. 实施顺序

1. 契约测试先行（红）；
2. 基类字段 + `AlgorithmMetadata` 字段 + `_SYNC_FIELDS`（绿）；
3. 各深度算法类显式声明（绿）；
4. `check_architecture_capability` 纯函数 + pipeline 接入；
5. `list-methods` 加列 + `test_info.py` 更新；
6. `validate_encoder_features` + 两估计器接入；
7. 文档（architecture.md、new_algorithm_template.md、process_checklist.md）；
8. 全量验证：快速测试 + 7 门禁。

---

# 第二部分：实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地阶段 0 能力契约——Registry 4 个架构能力字段 + Pipeline 并行校验 + list-methods 能力列 + encoder 输出校验 helper + 契约测试强制，不改变任何默认行为。

**Architecture:** 能力字段以估算器类属性为唯一权威（`frozenset` 不可变），经 `_SYNC_FIELDS` 镜像进 `AlgorithmMetadata`；Pipeline 在签名检查旁新增能力并行校验（结论漂移即 fail-loud）；encoder 输出校验提取为共享 helper 接入 infomax/wconpu。

**Tech Stack:** Python 3.13 / uv / pytest / sklearn / PyTorch（懒加载，保持 torch-free 导入链）

## Global Constraints

- 所有 Python 命令前缀 `uv run`；Python 代码中 Windows 路径用 `r'E:\...'` 或 `'E:/...'`
- 修改文件前必须先 Read（hooks 强制）
- commit message 不加 `Co-Authored-By` 署名；代码提交在 `feature/dual-arch-capability-contract` 分支（本计划所有 Task 的第一步之后保持在该分支）
- **零回归红线**：`tests/integration/test_pipeline_deep.py`、`tests/contract/test_classifier_baseline.py` 不改一行且必须通过；传统算法行为零变化；`architecture="mlp"` 默认路径不变
- **Rule 1 门禁约束**：本计划中"待创建"的文件路径一律不包裹反引号（check_doc_links 只豁免已存在文件）；待文件创建后再恢复引用
- 集合类能力字段一律 `frozenset`（类属性共享必须不可变）
- 每任务结束跑一次完整快速测试 `uv run pytest tests/ -v -m "not slow and not e2e"` 确认零回归，再提交

**Spec corrections（对照代码现状的勘误）**：规范 §5 称 infomax 有"两处 probe（Classifier.fit 与 Representation.fit）"——实际 probe 只在 `InfoMaxPURepresentation.fit`（infomax_pu.py:165-167）一处，Classifier.fit 通过 `representation.shape[1]` 取维（:445-447）。Task 5 一并修正规范原文。

---

### Task 1: 能力字段落地（基类 + AlgorithmMetadata + 同步 + 契约测试）

**Files:**
- Create: tests/contract/test_capability_declarations.py
- Modify: pu_toolbox/core/base.py:62-71（Metadata 块末尾）
- Modify: pu_toolbox/registry/metadata.py:80-85（training_cost 之后）
- Modify: pu_toolbox/registry/registry.py:124-133（_SYNC_FIELDS）

**Interfaces:**
- Consumes: 现有 `BasePUClassifier` 类属性块、`AlgorithmMetadata` dataclass、`_sync_class_metadata_to_registry`
- Produces: 类属性 `native_architectures: frozenset[str]` / `input_ndims: frozenset[int]` / `encoder_parameter: str | None` / `trains_encoder: bool`（BasePUClassifier 上，tabular 默认）；`AlgorithmMetadata` 同名 4 字段 + property `is_tabular_only -> bool`；Task 2-4 依赖这些名字

- [ ] **Step 1: 建分支**

```bash
git checkout -b feature/dual-arch-capability-contract
```

- [ ] **Step 2: 写契约测试（红）**

Create tests/contract/test_capability_declarations.py:

```python
# ruff: noqa: N802, N803, N806
"""Capability-declaration contract tests for registered classifiers.

Invariants from the phase-0 design spec §6: declaration legality, registry
sync, tabular_only derivation, and cross-mechanism consistency with the
constructor-signature check (dual_architecture_plan.md §4.2).
"""

from __future__ import annotations

import pytest

from pu_toolbox.core.base import BasePUClassifier
from pu_toolbox.registry import list_algorithms, register_all_builtin_methods
from pu_toolbox.registry.registry import get_algorithm

torch = pytest.importorskip("torch", reason="PyTorch not installed")

_LEGAL_NDIMS = {2, 4}
_LEGAL_ARCHS = {"mlp", "cnn"}


def _classifier_entries():
    """Yield (metadata, class) for every registered PU classifier."""
    register_all_builtin_methods()
    for meta in list_algorithms():
        cls = get_algorithm(meta.name)
        if isinstance(cls, type) and issubclass(cls, BasePUClassifier):
            yield meta, cls


@pytest.mark.contract
def test_declarations_are_legal():
    import inspect

    for meta, cls in _classifier_entries():
        assert cls.input_ndims, f"{meta.name}: input_ndims must be non-empty"
        assert cls.input_ndims <= _LEGAL_NDIMS, f"{meta.name}: input_ndims {cls.input_ndims}"
        assert cls.native_architectures <= _LEGAL_ARCHS, (
            f"{meta.name}: native_architectures {cls.native_architectures}"
        )
        if cls.encoder_parameter is not None:
            assert cls.encoder_parameter in inspect.signature(cls.__init__).parameters, (
                f"{meta.name}: encoder_parameter {cls.encoder_parameter!r} "
                "not in __init__ signature"
            )
        if cls.trains_encoder:
            assert cls.encoder_parameter is not None, (
                f"{meta.name}: trains_encoder=True requires encoder_parameter"
            )


@pytest.mark.contract
def test_registry_sync_matches_class():
    for meta, cls in _classifier_entries():
        assert meta.native_architectures == cls.native_architectures, meta.name
        assert meta.input_ndims == cls.input_ndims, meta.name
        assert meta.encoder_parameter == cls.encoder_parameter, meta.name
        assert meta.trains_encoder == cls.trains_encoder, meta.name


@pytest.mark.contract
def test_tabular_only_derived_from_empty_native_architectures():
    for meta, cls in _classifier_entries():
        expected = cls.native_architectures == frozenset()
        assert meta.is_tabular_only == expected, meta.name


@pytest.mark.contract
def test_cnn_capability_consistent_with_signature():
    from pu_toolbox.workflows._models import declares_encoder_parameter

    for meta, cls in _classifier_entries():
        if "cnn" in cls.native_architectures:
            assert cls.encoder_parameter is not None, meta.name
            assert declares_encoder_parameter(cls), meta.name
```

- [ ] **Step 3: 运行测试确认失败**

Run: `uv run pytest tests/contract/test_capability_declarations.py -v`
Expected: FAIL with `AttributeError: type object 'BasePUClassifier' has no attribute 'native_architectures'`（首条），后续为 `AlgorithmMetadata` 无字段/无 `is_tabular_only`

- [ ] **Step 4: 实现基类字段**

Modify pu_toolbox/core/base.py — 在 `sample_weight_support`（:71）之后、`# ── Internal state ──` 分隔注释之前插入：

```python
    # ── Architecture capability (dual_architecture_plan.md §4.2) ─────
    native_architectures: frozenset[str] = frozenset()
    input_ndims: frozenset[int] = frozenset({2})
    encoder_parameter: str | None = None
    trains_encoder: bool = False
```

- [ ] **Step 5: 实现 AlgorithmMetadata 字段与派生属性**

Modify pu_toolbox/registry/metadata.py — 在 `training_cost` 字段（:80-85）之后、`trainable` property 之前插入：

```python
    native_architectures: frozenset[str] = frozenset()
    """Native architecture paths; subset of {"mlp", "cnn"} (empty = tabular-only)."""

    input_ndims: frozenset[int] = frozenset({2})
    """Supported input dimensionalities; subset of {2, 4}."""

    encoder_parameter: str | None = None
    """Constructor parameter name that receives an injected encoder."""

    trains_encoder: bool = False
    """Whether the algorithm trains an injected encoder end-to-end."""
```

并在 `trainable` property 之后增加：

```python
    @property
    def is_tabular_only(self) -> bool:
        """True when the algorithm natively supports only 2-D table input."""
        return self.native_architectures == frozenset()
```

- [ ] **Step 6: 扩展 _SYNC_FIELDS**

Modify pu_toolbox/registry/registry.py:124-133，在 `"maturity"` 之后追加 4 个字段名：

```python
_SYNC_FIELDS = (
    "family",
    "assumption",
    "scenario",
    "requires_class_prior",
    "implementation_status",
    "source_status",
    "backend",
    "maturity",
    "native_architectures",
    "input_ndims",
    "encoder_parameter",
    "trains_encoder",
)
```

（同步函数只同步类显式声明的字段，frozenset 共享引用安全，无需改同步逻辑。）

- [ ] **Step 7: 运行契约测试确认通过**

Run: `uv run pytest tests/contract/test_capability_declarations.py -v`
Expected: PASS（传统算法继承默认值，同步默认一致，4 组不变量全绿）

- [ ] **Step 8: 全量快速测试零回归**

Run: `uv run pytest tests/ -v -m "not slow and not e2e"`
Expected: 全部 PASS（现有测试无字段枚举/数量断言触及这些变更——Step 8 若发现例外，修正该测试并在此任务提交中说明）

- [ ] **Step 9: 提交**

```bash
git add pu_toolbox/core/base.py pu_toolbox/registry/metadata.py pu_toolbox/registry/registry.py tests/contract/test_capability_declarations.py
git commit -m "feat(registry): 架构能力字段（native_architectures/input_ndims/encoder_parameter/trains_encoder）

- BasePUClassifier 增加 4 个能力类属性，tabular 默认值（frozenset 不可变）
- AlgorithmMetadata 增加同名字段 + 派生 is_tabular_only
- _SYNC_FIELDS 扩展同步；契约测试 4 组不变量先行落地"
```

---

### Task 2: 深度算法显式声明能力

**Files:**
- Modify: pu_toolbox/estimators/deep/infomax_pu.py（InfoMaxPUClassifier 类属性块，sample_weight_support 行 :318 之后）
- Modify: pu_toolbox/estimators/deep/weighted_contrastive_pu.py（:53 之后）
- Modify: pu_toolbox/estimators/deep/self_pu.py（:249 之后）
- Modify: pu_toolbox/estimators/deep/dgpu.py（:37 之后）
- Modify: pu_toolbox/estimators/risk/nnpu.py（:110 之后）
- Modify: pu_toolbox/estimators/risk/dist_pu.py（:41 之后）
- Modify: tests/contract/test_capability_declarations.py

**Interfaces:**
- Consumes: Task 1 的 4 个类属性与 `AlgorithmMetadata.is_tabular_only`
- Produces: 六个深度分类器的具体声明值（Task 3 gate 与 Task 4 展示依赖）

- [ ] **Step 1: 写期望声明测试（红）**

Modify tests/contract/test_capability_declarations.py — 文件末尾追加：

```python
_EXPECTED_DECLARATIONS = {
    "infomax_pu": (frozenset({"mlp", "cnn"}), frozenset({2, 4}), "encoder", True),
    "weighted_contrastive_pu": (frozenset({"mlp", "cnn"}), frozenset({2, 4}), "encoder", True),
    "self_pu": (frozenset({"mlp"}), frozenset({2, 4}), None, False),
    "nnpu": (frozenset({"mlp"}), frozenset({2}), None, False),
    "dist_pu": (frozenset({"mlp"}), frozenset({2}), None, False),
    "dgpu": (frozenset({"mlp"}), frozenset({2}), None, False),
}


@pytest.mark.contract
def test_deep_capability_declarations():
    register_all_builtin_methods()
    for name, (archs, ndims, enc_param, trains) in _EXPECTED_DECLARATIONS.items():
        cls = get_algorithm(name)
        assert cls.native_architectures == archs, name
        assert cls.input_ndims == ndims, name
        assert cls.encoder_parameter == enc_param, name
        assert cls.trains_encoder == trains, name
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/contract/test_capability_declarations.py::test_deep_capability_declarations -v`
Expected: FAIL——首个断言 `frozenset() != frozenset({'mlp','cnn'})`（infomax_pu 仍继承默认 ∅）

- [ ] **Step 3: 声明 6 个深度算法**

在六个文件各自的类属性块（锚点见 Files）`sample_weight_support` 行之后追加（infomax_pu 与 weighted_contrastive_pu）：

```python
    native_architectures = frozenset({"mlp", "cnn"})
    input_ndims = frozenset({2, 4})
    encoder_parameter = "encoder"
    trains_encoder = True
```

self_pu 追加：

```python
    native_architectures = frozenset({"mlp"})
    input_ndims = frozenset({2, 4})
    encoder_parameter = None
    trains_encoder = False
```

nnpu / dist_pu / dgpu 追加（self_pu 那组中 `input_ndims` 换成 `frozenset({2})`）：

```python
    native_architectures = frozenset({"mlp"})
    input_ndims = frozenset({2})
    encoder_parameter = None
    trains_encoder = False
```

注意：`encoder_parameter = None` 显式声明（而非省略继承），契约测试的注册表同步与声明合法性都因此对该类生效。

- [ ] **Step 4: 运行契约测试确认通过**

Run: `uv run pytest tests/contract/test_capability_declarations.py -v`
Expected: 5 项全部 PASS（含注册表镜像同步断言——`_sync_class_metadata_to_registry` 对显式声明字段自动镜像）

- [ ] **Step 5: 全量快速测试零回归**

Run: `uv run pytest tests/ -v -m "not slow and not e2e"`
Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add pu_toolbox/estimators/deep/infomax_pu.py pu_toolbox/estimators/deep/weighted_contrastive_pu.py pu_toolbox/estimators/deep/self_pu.py pu_toolbox/estimators/deep/dgpu.py pu_toolbox/estimators/risk/nnpu.py pu_toolbox/estimators/risk/dist_pu.py tests/contract/test_capability_declarations.py
git commit -m "feat: 六个深度算法显式声明架构能力

infomax/wconpu: mlp+cnn/{2,4}/encoder/端到端；self_pu: mlp/{2,4}（backbone 为完整模型）；
nnpu/dist_pu/dgpu: mlp/{2}。契约测试锁定期望声明表"
```

---

### Task 3: Pipeline 架构能力并行校验

**Files:**
- Create: tests/unit/workflows/test_architecture_capability.py
- Modify: pu_toolbox/workflows/_models.py（新增纯函数）
- Modify: pu_toolbox/workflows/pipeline.py:42-50（import）与 :227-236（调用）

**Interfaces:**
- Consumes: Task 1 的 `declares_encoder_parameter`（_models.py:43-44）、Task 2 的类属性
- Produces: `check_architecture_capability(cls, architecture, classifier_name) -> None`（cnn 时校验签名结论与能力结论一致，漂移抛 `PipelineError`）

- [ ] **Step 1: 写单测（红）**

Create tests/unit/workflows/test_architecture_capability.py:

```python
# ruff: noqa: N802, N803, N806
"""Unit tests for check_architecture_capability (signature vs capability)."""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.core.base import BasePUClassifier
from pu_toolbox.workflows._errors import PipelineError
from pu_toolbox.workflows._models import check_architecture_capability


class _Capable(BasePUClassifier):
    native_architectures = frozenset({"cnn"})
    input_ndims = frozenset({2, 4})
    encoder_parameter = "encoder"
    trains_encoder = True

    def __init__(self, *, encoder=None):
        self.encoder = encoder

    def fit(self, X, y_pu, *, class_prior=None, sample_weight=None):
        return self

    def _predict(self, X):
        return np.zeros(len(X))

    def _decision_function(self, X):
        return np.zeros(len(X))


class _SigYesCapNo(BasePUClassifier):
    """Signature declares encoder but capability metadata does not (drift)."""

    def __init__(self, *, encoder=None):
        self.encoder = encoder

    def fit(self, X, y_pu, *, class_prior=None, sample_weight=None):
        return self

    def _predict(self, X):
        return np.zeros(len(X))

    def _decision_function(self, X):
        return np.zeros(len(X))


class _SigNoCapYes(BasePUClassifier):
    """Capability claims cnn but signature has no encoder param (drift)."""

    native_architectures = frozenset({"cnn"})
    input_ndims = frozenset({2, 4})
    encoder_parameter = "encoder"
    trains_encoder = True

    def fit(self, X, y_pu, *, class_prior=None, sample_weight=None):
        return self

    def _predict(self, X):
        return np.zeros(len(X))

    def _decision_function(self, X):
        return np.zeros(len(X))


@pytest.mark.unit
def test_capable_class_passes_cnn_check():
    check_architecture_capability(_Capable, "cnn", "fake")


@pytest.mark.unit
def test_signature_yes_capability_no_raises():
    with pytest.raises(PipelineError, match="mismatch"):
        check_architecture_capability(_SigYesCapNo, "cnn", "fake")


@pytest.mark.unit
def test_signature_no_capability_yes_raises():
    with pytest.raises(PipelineError, match="mismatch"):
        check_architecture_capability(_SigNoCapYes, "cnn", "fake")


@pytest.mark.unit
def test_mlp_architecture_never_checked():
    check_architecture_capability(_SigYesCapNo, "mlp", "fake")
    check_architecture_capability(_SigNoCapYes, "mlp", "fake")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/unit/workflows/test_architecture_capability.py -v`
Expected: FAIL with `ImportError: cannot import name 'check_architecture_capability'`

- [ ] **Step 3: 实现纯函数**

Modify pu_toolbox/workflows/_models.py — 在 `declares_encoder_parameter`（:43-44）之后追加：

```python
def check_architecture_capability(
    cls: type, architecture: str, classifier_name: str
) -> None:
    """Validate ``architecture`` against the class capability declaration.

    Runs alongside the constructor-signature check; both conclusions must
    agree.  A mismatch means the capability declaration has drifted from
    the implementation and must be fixed (fail-loud, never silently pick
    one side).
    """
    if architecture != "cnn":
        return
    signature_ok = declares_encoder_parameter(cls)
    capability_ok = (
        "cnn" in cls.native_architectures
        and 4 in cls.input_ndims
        and cls.encoder_parameter is not None
    )
    if signature_ok != capability_ok:
        raise PipelineError(
            f"architecture='cnn' capability mismatch for classifier "
            f"{classifier_name!r}: constructor signature says "
            f"{'encoder supported' if signature_ok else 'no encoder'}, but the "
            f"capability declaration says "
            f"{'cnn supported' if capability_ok else 'no cnn'} "
            f"(native_architectures={set(cls.native_architectures)!r}, "
            f"input_ndims={set(cls.input_ndims)!r}, "
            f"encoder_parameter={cls.encoder_parameter!r}). "
            "Fix the class capability declaration."
        )
```

- [ ] **Step 4: 运行单测确认通过**

Run: `uv run pytest tests/unit/workflows/test_architecture_capability.py -v`
Expected: 4 项 PASS

- [ ] **Step 5: Pipeline 接入**

Modify pu_toolbox/workflows/pipeline.py:

- :42-50 的 `from ._models import (...)` 列表中 `declares_encoder_parameter` 之前加一行 `check_architecture_capability,`（字母序）；
- 在 :227-236 现有签名检查 `if not declares_encoder_parameter(encoder_cls): raise PipelineError(...)` 块之后追加：

```python
            check_architecture_capability(
                encoder_cls, architecture, self._classifier_name
            )
```

（顺序保证现有错误信息不变：签名不支持时先抛原 `"requires classifier ... 'encoder'"` 错误，能力漂移时再抛 mismatch。）

- [ ] **Step 6: 集成测试确认零回归**

Run: `uv run pytest tests/integration/test_pipeline_deep.py -v`
Expected: 全部 PASS（wconpu/infomax 能力一致不触发新路径；self_pu/dgpu 仍走原签名错误路径，match="encoder" 不变）

- [ ] **Step 7: 全量快速测试零回归**

Run: `uv run pytest tests/ -v -m "not slow and not e2e"`
Expected: 全部 PASS

- [ ] **Step 8: 提交**

```bash
git add pu_toolbox/workflows/_models.py pu_toolbox/workflows/pipeline.py tests/unit/workflows/test_architecture_capability.py
git commit -m "feat(workflows): 架构能力并行校验 gate

check_architecture_capability 纯函数 + pipeline cnn 路径接入；签名结论与
能力声明漂移时 fail-loud（PipelineError mismatch）"
```

---

### Task 4: list-methods 能力列

**Files:**
- Modify: pu_toolbox/cli/info.py:30-62（run_list_methods）
- Modify: tests/unit/cli/test_info.py

**Interfaces:**
- Consumes: Task 1 的 `meta.input_ndims` / `meta.is_tabular_only` / `meta.native_architectures`
- Produces: 7 列表格（Name/Family/Prior/Status/Input/Arch/Auto-inst）

- [ ] **Step 1: 写测试（红）**

Modify tests/unit/cli/test_info.py — 在 `test_deterministic_list_methods_output_stable` 之前追加：

```python
@pytest.mark.unit
def test_list_methods_shows_capability_columns(capsys):
    """Input and Arch columns reflect the capability declarations."""
    args = build_parser().parse_args(["list-methods"])
    args.func(args)
    out = capsys.readouterr().out
    assert "Input" in out and "Arch" in out
    lines = [line for line in out.splitlines()[2:] if line.strip()]
    by_name = {line.split()[0]: line.split() for line in lines}
    assert "2,4" in by_name["infomax_pu"] and "mlp,cnn" in by_name["infomax_pu"]
    assert "2" in by_name["nnpu"] and "mlp" in by_name["nnpu"]
    assert "tabular" in by_name["elkan_noto"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/unit/cli/test_info.py::test_list_methods_shows_capability_columns -v`
Expected: FAIL with `AssertionError: 'Input' not found`（表头尚无 Input/Arch）

- [ ] **Step 3: 实现 7 列表格**

Modify pu_toolbox/cli/info.py `run_list_methods`：

- `rows` 类型改为 `list[tuple[str, str, str, str, str, str, str]]`；
- 行组装改为：

```python
        for name in (meta.name, *meta.aliases):
            auto_inst = "yes" if not _missing_required_params(cls) else "no"
            input_dims = ",".join(str(d) for d in sorted(meta.input_ndims))
            arch = (
                "tabular"
                if meta.is_tabular_only
                else ",".join(a for a in ("mlp", "cnn") if a in meta.native_architectures)
            )
            rows.append(
                (
                    name,
                    meta.family.value,
                    "yes" if meta.requires_class_prior else "no",
                    meta.implementation_status.value,
                    input_dims,
                    arch,
                    auto_inst,
                )
            )
```

- 表头与分隔线改为：

```python
    print(f"{'Name':<{name_width}}{'Family':<22}{'Prior':<6}{'Status':<8}{'Input':<7}{'Arch':<11}{'Auto-inst':<10}")
    print("-" * (name_width + 64))
```

- 行打印改为：

```python
    for name, family, prior, status, input_dims, arch, auto in rows:
        print(f"{name:<{name_width}}{family:<22}{prior:<6}{status:<8}{input_dims:<7}{arch:<11}{auto:<10}")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/unit/cli/test_info.py -v`
Expected: 全部 PASS（既有断言只查 Name/Family/Auto-inst 表头与首/末列 token，不冲突）

- [ ] **Step 5: 全量快速测试零回归**

Run: `uv run pytest tests/ -v -m "not slow and not e2e"`
Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add pu_toolbox/cli/info.py tests/unit/cli/test_info.py
git commit -m "feat(cli): list-methods 增加 Input/Arch 能力列

从注册表元数据读取（input_ndims / native_architectures / is_tabular_only），
7 列对齐输出；新增能力列契约测试"
```

---

### Task 5: encoder 输出校验 helper

**Files:**
- Create: pu_toolbox/estimators/deep/_validation.py
- Create: tests/unit/estimators/test_encoder_validation.py
- Modify: pu_toolbox/estimators/deep/infomax_pu.py（导入 + :158-165 probe 替换）
- Modify: pu_toolbox/estimators/deep/weighted_contrastive_pu.py（导入 + :169-173 probe 替换）
- Modify: docs/dev/2026-08-29-dual-arch-phase0-design.md（§5 勘误：probe 只有一处）

**Interfaces:**
- Consumes: 现有 probe 代码（eval 模式 + no_grad 前向）
- Produces: `validate_encoder_features(features, *, encoder_param_name: str) -> int`（2-D、有限、feature_dim>=1，返回 feature_dim）

- [ ] **Step 1: 写单测（红）**

Create tests/unit/estimators/test_encoder_validation.py:

```python
# ruff: noqa: N802, N803, N806
"""Unit tests for validate_encoder_features."""

from __future__ import annotations

import pytest

from pu_toolbox.estimators.deep._validation import validate_encoder_features

torch = pytest.importorskip("torch", reason="PyTorch not installed")


@pytest.mark.unit
def test_valid_2d_output_returns_feature_dim():
    features = torch.zeros(4, 7)
    assert validate_encoder_features(features, encoder_param_name="encoder") == 7


@pytest.mark.unit
def test_ndim_not_2_raises():
    with pytest.raises(ValueError, match="2-D"):
        validate_encoder_features(torch.zeros(2, 3, 4), encoder_param_name="encoder")


@pytest.mark.unit
def test_non_finite_raises():
    features = torch.zeros(2, 3)
    features[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        validate_encoder_features(features, encoder_param_name="encoder")


@pytest.mark.unit
def test_non_tensor_raises():
    with pytest.raises(TypeError, match="torch.Tensor"):
        validate_encoder_features([1.0, 2.0], encoder_param_name="encoder")


@pytest.mark.unit
def test_zero_feature_dim_raises():
    with pytest.raises(ValueError, match="empty"):
        validate_encoder_features(torch.zeros(2, 0), encoder_param_name="encoder")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/unit/estimators/test_encoder_validation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pu_toolbox.estimators.deep._validation'`

- [ ] **Step 3: 实现 helper**

Create pu_toolbox/estimators/deep/_validation.py:

```python
"""Shared validation for encoder feature outputs (deep estimators)."""

from __future__ import annotations


def validate_encoder_features(features, *, encoder_param_name: str) -> int:
    """Validate an encoder's output and return its feature dimension.

    Encoder contract (dual_architecture_plan.md §4.1): output must be a
    2-D ``(batch, feature_dim)`` tensor with finite values.
    """
    import torch  # lazy: keep the torch-free import chain of deep estimators

    if not torch.is_tensor(features):
        raise TypeError(
            f"encoder {encoder_param_name!r} must return a torch.Tensor; "
            f"got {type(features).__name__}"
        )
    if features.ndim != 2:
        raise ValueError(
            f"encoder {encoder_param_name!r} must output a 2-D "
            f"(batch, feature_dim) tensor; got shape {tuple(features.shape)}"
        )
    if not torch.isfinite(features).all():
        raise ValueError(
            f"encoder {encoder_param_name!r} output contains NaN or Inf values"
        )
    feature_dim = int(features.shape[1])
    if feature_dim < 1:
        raise ValueError(
            f"encoder {encoder_param_name!r} output has empty feature dimension"
        )
    return feature_dim
```

（torch 必须在函数内导入：infomax/wconpu 模块级不导入 torch，顶层导入 torch 会破坏 base wheel 无 torch 可导入的契约。）

- [ ] **Step 4: 运行单测确认通过**

Run: `uv run pytest tests/unit/estimators/test_encoder_validation.py -v`
Expected: 5 项 PASS

- [ ] **Step 5: 接入 infomax_pu**

Modify pu_toolbox/estimators/deep/infomax_pu.py：

- 顶层导入区追加 `from ._validation import validate_encoder_features`；
- :163-165 probe 替换为：

```python
            with torch.no_grad():
                probe = self.encoder_(torch.as_tensor(X[:1], dtype=torch.float32, device=device))
            representation_dim = validate_encoder_features(
                probe.flatten(start_dim=1), encoder_param_name="encoder"
            )
```

（原代码 `representation_dim = int(probe.flatten(start_dim=1).shape[-1])` 删除。）

- [ ] **Step 6: 接入 weighted_contrastive_pu**

Modify pu_toolbox/estimators/deep/weighted_contrastive_pu.py：

- 顶层导入区追加 `from ._validation import validate_encoder_features`；
- :169-173 替换为：

```python
        self.encoder_.eval()
        with torch.no_grad():
            probe = _flatten_features(self.encoder_(torch.as_tensor(X[:1], device=device)))
        self.encoder_.train()
        feature_dim = validate_encoder_features(probe, encoder_param_name="encoder")
```

（原代码 `feature_dim = int(probe.shape[-1])` 删除。）

- [ ] **Step 7: 深度集成测试零回归**

Run: `uv run pytest tests/integration/test_pipeline_deep.py tests/unit/estimators/test_deep_pu.py tests/unit/estimators/test_deep_pu_vision.py -v`
Expected: 全部 PASS（helper 校验与原有 flatten+shape[-1] 语义等价）

- [ ] **Step 8: 勘误规范 §5**

Modify docs/dev/2026-08-29-dual-arch-phase0-design.md §5 接入点第二条：
原文 "`infomax_pu.py:158-165`（InfoMaxPUClassifier.fit 与 InfoMaxPURepresentation.fit 两处 probe）" 改为 "`infomax_pu.py:158-165`（InfoMaxPURepresentation.fit 一处 probe；InfoMaxPUClassifier.fit 经 `representation.shape[1]` 取维）"。

- [ ] **Step 9: 全量快速测试零回归**

Run: `uv run pytest tests/ -v -m "not slow and not e2e"`
Expected: 全部 PASS

- [ ] **Step 10: 提交**

```bash
git add pu_toolbox/estimators/deep/_validation.py pu_toolbox/estimators/deep/infomax_pu.py pu_toolbox/estimators/deep/weighted_contrastive_pu.py tests/unit/estimators/test_encoder_validation.py docs/dev/2026-08-29-dual-arch-phase0-design.md
git commit -m "feat(deep): 统一 encoder 输出校验 helper

validate_encoder_features（2-D/有限/feature_dim>=1），接入 infomax 表示层
probe 与 wconpu probe；torch 函数内懒导入保持无 torch 可导入契约；
规范 §5 勘误 probe 位置"
```

---

### Task 6: 文档收尾与全量验收

**Files:**
- Modify: docs/dev/architecture.md:126-129（§4 注册元信息段）
- Create: docs/dev/new_algorithm_template.md
- Modify: docs/README.md（新增模板文档索引行 + 更新设计规范行描述）
- Modify: docs/dev/process_checklist.md（发布状态节增加完成条目）
- Regenerate: docs/dev/project_structure.md（generate_structure.py --update，登记新 .py 文件）

**Interfaces:**
- Consumes: Task 1-5 全部产物（字段名、helper 名、CLI 列名）
- Produces: 无代码接口；阶段 0 文档闭环

- [ ] **Step 1: 更新 architecture.md §4**

Modify docs/dev/architecture.md:126-129 段落后追加：

```markdown
架构能力字段（native_architectures / input_ndims / encoder_parameter /
trains_encoder + 派生 is_tabular_only）以估算器类属性为权威，注册时经
`_SYNC_FIELDS` 镜像进 registry；语义与消费点见 `dual_architecture_plan.md`
§3-§4 与阶段 0 设计规范。
```

- [ ] **Step 2: 新建接入模板**

Create docs/dev/new_algorithm_template.md（要点齐全、无占位符；路径不裹反引号）：

```markdown
# 新算法接入模板

## 1. 必做声明

新算法必须：

1. 实现 API 契约（fit/predict/decision_function/get_params/set_params，
   y 标签语义由算法决定并在 fit 内校验）；
2. 在类属性块声明 4 个能力字段（`BasePUClassifier` 有 tabular 默认值，
   深度算法必须显式声明）：

   | 字段 | 合法值 | 说明 |
   |---|---|---|
   | native_architectures | ⊆ {"mlp","cnn"} | 原生架构路径；∅ = tabular_only（派生） |
   | input_ndims | ⊆ {2,4}，非空 | 支持输入维度 |
   | encoder_parameter | None 或构造函数参数名 | 接收注入 encoder 的参数 |
   | trains_encoder | bool | 是否端到端训练注入的 encoder |

3. 在 registry/builtin_methods.py 注册 AlgorithmMetadata（含
   implementation_status=NATIVE 仅当有真实训练逻辑；未实现必须 API_ONLY）；
4. 声明 sample_weight_support / backend / requires_class_prior 等既有字段。

## 2. 自动门禁（无需手写）

- 契约测试 tests/contract/test_capability_declarations.py：声明合法性、
  注册表同步、tabular_only 派生、签名一致性——新算法漏声明直接失败；
- tests/contract/test_classifier_baseline.py：API 契约 + 基线行为；
- check_doc_links Rule 1：文档引用路径必须真实存在。

## 3. 若声明支持 CNN（native_architectures 含 "cnn"）

必须提供：

- CNN smoke training 测试；
- 输入/输出形状测试（encoder 输出经 validate_encoder_features 校验）；
- CV fold 隔离测试（fold 间权重不泄漏）；
- 固定 seed 测试；
- CPU/GPU（可用时）测试；
- save/load/predict round-trip 测试；
- 不支持架构的 fail-fast 测试（PUPipeline architecture 校验）。

## 4. 最小示例

深度算法类属性块（假设 MLP+CNN 双架构）：

    native_architectures = frozenset({"mlp", "cnn"})
    input_ndims = frozenset({2, 4})
    encoder_parameter = "encoder"
    trains_encoder = True

fit 内 probe（若 encoder 非 None）：

    representation_dim = validate_encoder_features(
        probe.flatten(start_dim=1), encoder_param_name="encoder"
    )

传统表格算法：不声明能力字段（继承 tabular 默认），只声明既有元数据。
```

- [ ] **Step 3: 更新 docs/README.md**

- 开发者文档表新增一行（在阶段 0 设计规范行之后，链接格式仿照既有行，目标为 dev/new_algorithm_template.md，用途描述：新算法接入模板——能力声明清单、自动门禁与 CNN 最低测试要求）；

- 将阶段 0 设计规范行描述改为 "双架构阶段 0 能力契约设计规范与实施计划：字段模型、消费点、契约测试、任务分解与验收标准"。

- [ ] **Step 4: 更新 process_checklist.md**

Read docs/dev/process_checklist.md 发布状态节，按其条目风格追加一条：

```markdown
- 双架构阶段 0 能力契约：Registry 4 能力字段 + Pipeline 并行校验 +
  list-methods 能力列 + encoder 输出校验 helper + 契约测试（详见
  docs/dev/2026-08-29-dual-arch-phase0-design.md）
```

- [ ] **Step 5: 重新生成结构文档**

Run: `uv run python scripts/generate_structure.py --update`
Expected: project_structure.md 登记 _validation.py 与两个新测试文件（无其他差异；若出现无关差异需检查是否误改）

- [ ] **Step 6: 全量验收**

```bash
uv run pytest tests/ -v -m "not slow and not e2e"
uv run python scripts/check_test_quality.py
uv run python scripts/check_doc_links.py
uv run python scripts/check_project_metadata.py
uv run python scripts/check_math_rendering.py
uv run python scripts/check_skill_sync.py
uv run python scripts/check_baseline_configs.py
uv run python scripts/check_format.py
```

Expected: 快速测试全部 PASS；7 项门禁全部通过（check_doc_links 五规则全过——本计划文档中的待创建路径已全部落地，不再触发 Rule 1）。

- [ ] **Step 7: 提交**

```bash
git add docs/dev/architecture.md docs/dev/new_algorithm_template.md docs/README.md docs/dev/process_checklist.md docs/dev/project_structure.md
git commit -m "docs: 阶段 0 文档收尾（architecture.md 能力字段、接入模板、索引与进度）

- architecture.md §4 补充能力字段权威来源与镜像机制
- 新增 new_algorithm_template.md（声明清单 + 自动门禁 + CNN 最低要求）
- docs/README.md 索引登记；process_checklist.md 发布状态条目
- project_structure.md 重生成（登记 _validation.py 与新测试文件）"
```

---

## 验收总览（对照规范 §9）

1. 契约测试 4 组不变量 + 期望声明表全绿（Task 1-2）；
2. tests/integration/test_pipeline_deep.py 与 tests/contract/test_classifier_baseline.py 全程未改一行且通过（Task 3/5 Step 验证 + Task 6 全量）；
3. list-methods 7 列对齐可读（Task 4，`uv run pu-toolbox list-methods` 目检）；
4. 快速测试 + 7 门禁全过（Task 6 Step 6）；
5. 分支 feature/dual-arch-capability-contract 共 6 个提交，PR 合并回 main。
