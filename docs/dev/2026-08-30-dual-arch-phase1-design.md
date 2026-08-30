# 双架构能力契约（阶段 1）设计规范

- 日期：2026-08-30
- 状态：已与用户确认（brainstorming 设计呈现两批均获批）
- 上游：dual_architecture_plan.md §5 阶段 1（commit 65e7ee9 修订版）
- 前置：阶段 0 已全部完成并合入 main（commit 9fa9f68 为收尾）

## 1. 背景与范围

上游计划阶段 1「整理现有双架构实现」。阶段 0 已落地能力契约字段与消费点，本阶段补齐四个缺口：

**范围内**：

1. `build_encoder` 公共导出（保留 WConPU backbone 函数兼容）；
2. 报告 provenance 增加 architecture/backbone/device/encoder 字段；
3. UI 的 CNN 算法候选集改由注册表能力元数据驱动（移除硬编码集合）；
4. 测试缺口：CV fold 训练隔离测试（fold 间权重不泄漏）+ build_encoder 公共导出契约测试。

**范围外（明确不做，归属后续阶段）**：

- 阶段 2：nnpu `encoder` 参数试点；
- 阶段 3：self_pu / dgpu（长期搁置）；
- 阶段 4：`adapter_architectures` 字段（YAGNI）；
- CPU/GPU 执行级测试：仅 CUDA 可用时补充（计划原文约定），本阶段本机无 CUDA 不实现。

## 2. 已确认决策

| 决策点 | 结论 |
|---|---|
| 本轮范围 | 按上游计划 §5 阶段 1 四项，不增不减 |
| architecture 字段语义 | 单值：`"native_mlp"` / `"native_cnn"`，由 `pipeline.architecture` 映射；「分别报告」指不同运行各自体现，不在单次报告内分节 |
| encoder 字段内容 | 构造摘要 dict（`{"backbone": ..., "in_channels": ...}`）；MLP 路径（无注入 encoder）为 `None` |
| provenance 层级 | 顶层平铺 4 键，与 classifier/random_state 并列；device 为 `{"requested", "resolved"}` 两键嵌套 |
| backbone 清单权威 | vision.py 新增常量 `CNN_BACKBONES`，app.py 与 run_config.py 均从此导入；registry 不扩字段（YAGNI，与阶段 4 才加 adapter 字段的决策一致） |
| CV 隔离测试策略 | 方案 C：pipeline 集成测试 2 折真实训练 + 权重快照断言 + 防假阳性（折内权重确实在变） |
| 报告 schema_version | 保持 "1.0" 不变——纯 additive 扩展，旧键不破坏旧消费者 |

## 3. 组件设计

### 3.1 build_encoder 公共导出

现状：build_encoder 已存在（`estimators/deep/vision.py:207-235`），签名
`build_encoder(architecture, *, backbone="cnn13", in_channels, normalization_mean=None, normalization_std=None)`
返回 `nn.Sequential | None`；vision.py 已用 try/except ImportError 保持无 torch 可导入契约；生产与测试调用方全部走模块路径。

变更：

1. `pu_toolbox/estimators/deep/__init__.py` 导出 `build_encoder`（与 build_wconpu_augmentation / build_wconpu_backbone 并列）；
2. `pu_toolbox/__init__.py` 的 `__all__` 登记 `build_encoder`；
3. WConPU backbone 函数与现有导出保持不动（兼容保留，计划原文要求）；
4. 新增契约测试（tests/contract/test_build_encoder_export.py）：
   - 包根 `from pu_toolbox import build_encoder` 可用；
   - `("mlp", ...)` 返回 None；
   - 非法 architecture 抛 ValueError；
   - `("cnn", backbone=...)` 返回 nn.Sequential 且与 build_wconpu_backbone 同参构造等价（结构一致 + 前向形状一致；权重随机初始化不比较 state_dict）；
   - 无 torch 环境下包根导入不炸：vision.py 既有 try/except ImportError 机制保持不变，本阶段不新增模拟导入的测试（仓库无现成 torch-free 导入测试模式），由 CI wheel 安装冒烟覆盖。

### 3.2 报告 provenance 4 字段

现状：build_pipeline_report（`workflows/_reporting.py:72-84`）构建扁平 provenance dict；PUPipeline 持有 architecture / backbone / device（`workflows/pipeline.py:239-241`，device 为请求值）但未传入报告；`core/device.py:11` 的 resolve_device_name 可解析请求值→实际设备名。

变更（顶层平铺）：

```python
"architecture": "native_mlp" | "native_cnn",   # pipeline.architecture 映射
"backbone": "cnn13" | None,                     # MLP 时为 None（backbone 不适用）
"device": {"requested": ..., "resolved": ...},  # resolved = resolve_device_name(device)
"encoder": None | {"backbone": ..., "in_channels": ...},  # MLP 为 None
```

- build_pipeline_report 签名增加 architecture / backbone / device 三个参数，由 pipeline.fit_evaluate 调用点（pipeline.py:564-582）传入；
- encoder 摘要仅在 CNN 时非空：`{"backbone": backbone, "in_channels": 实际输入通道数}`；
- 文档：`docs/user/reference/api.md` provenance 节补 4 字段说明（保持引用准确）。

### 3.3 UI 元数据驱动

现状：`ui/app.py` 硬编码点 5 处——:117（4-D 输入不合格项比较）、:151/:166（两处相同筛选集合 `{"infomax_pu", "weighted_contrastive_pu"}`）、:176（信息文本）、:212-214（骨架 selectbox `["cnn13","resnet18","resnet50"]`）；`run_config.py:89` backbone 白名单同集。UI 算法目录本身已元数据驱动（`ui/parameters.py:15-50` classifier_catalog 为现成先例）。

变更：

1. **算法候选集**：app.py 三处硬编码集合改为从 registry 元数据推导——`{m.name for m in list_algorithms() if "cnn" in m.native_architectures}`；复用 classifier_catalog 的注册流程，不重复注册；
2. **骨架清单**：vision.py 新增 `CNN_BACKBONES: tuple[str, ...] = ("cnn13", "resnet18", "resnet50")`；app.py selectbox 与 run_config.py 白名单均从此导入；build_wconpu_backbone 的 Literal 类型注解与常量保持一致；
3. **信息文本**：:176 改为动态列出当前支持 CNN 的算法名（渲染时从注册表取）；
4. **候选集推导提取为纯函数**（放 ui/parameters.py，如 `cnn_candidates()`），新增单测断言：返回与 registry 声明一致；当前声明集恰为 {"infomax_pu", "weighted_contrastive_pu"}（锁定与阶段 0 声明的一致性）。UI 层不引入 streamlit 依赖，现有 AppTest 测试不变。

### 3.4 CV fold 训练隔离测试（方案 C）

现状：共享 encoder 每 fit_evaluate 构建一次（pipeline.py:491-493），fresh_estimator 注入同一引用（workflows/_models.py:130-142），各估计器 fit 内 copy.deepcopy（infomax_pu.py:160 / weighted_contrastive_pu.py:172,185-186）。泄漏风险点 = deepcopy 语义被未来改动破坏。

变更：新增 tests/integration/test_cv_fold_isolation.py（WConPU + cnn13 最小配置、1 epoch、小合成数据）：

1. 构造 PUPipeline(architecture="cnn", cv=2)，取 pipeline._encoder（共享模板）；
2. 经 pipeline._fresh_estimator 取得两折 estimator，分别 fit 折 1 / 折 2 数据；
3. 断言：
   - **防假阳性**：折 1 fit 后其 encoder_ 权重 ≠ 初始（训练确实生效）；
   - **起点无污染**：折 2 fit 前其 encoder_ 权重 == 共享 encoder 权重（折 1 训练未泄漏进折 2 起点）；
   - **折间隔离**：折 1 encoder_ 权重快照在折 2 fit 后不变；两折 encoder_ 非同一对象；
   - **模板不被训练**：共享 pipeline._encoder 权重在全部折 fit 后保持未训练状态。

## 4. 兼容性影响清单

| 现有测试 | 影响 |
|---|---|
| tests/integration/test_pipeline_deep.py / tests/contract/test_classifier_baseline.py | 红线：不改一行 |
| tests/unit/estimators/test_vision.py 等模块路径调用方 | 不变（模块路径仍可用） |
| report 相关测试 | 若断言 provenance 键集合需更新（实现时 grep 确认） |
| UI 测试（test_ui_app.py 等，streamlit skip） | 不变；新增纯函数单测不依赖 streamlit |
| run_config.py 白名单行为 | 不变（同一集合，来源改为常量导入） |

## 5. 验收标准

1. 新增 3 个测试文件（导出契约 + CV 隔离 + UI 候选纯函数）全绿；
2. 快速测试零回归（tests/integration/test_pipeline_deep.py 与 tests/contract/test_classifier_baseline.py 全程未改）；
3. 7 项质量门禁全过；
4. 报告 JSON 目检 4 新字段（native_mlp 与 native_cnn 各一次运行分别体现）；
5. UI 硬编码算法集清零（grep 确认无 {"infomax_pu", "weighted_contrastive_pu"} 字面量残留于筛选逻辑）。

## 6. 实施顺序（粗粒度，细节由实施计划展开）

1. 契约测试先行（build_encoder 导出，红）→ 导出实现（绿）；
2. provenance 字段（测试更新 + _reporting.py + pipeline 传参）；
3. UI 元数据驱动（cnn_candidates 纯函数 + 单测 → app.py 替换 + vision.py 常量 + run_config 导入）；
4. CV fold 隔离测试（新增，直接绿——锁定现状语义）；
5. 文档（api.md provenance 节）与全量验证（快速测试 + 7 门禁）。

---

# 第二部分：实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地阶段 1 四项——build_encoder 公共导出、报告 provenance 4 字段、UI CNN 候选集元数据驱动、CV fold 训练隔离测试，不改变任何默认行为。

**Architecture:** backbone 清单以 vision.py 常量 `CNN_BACKBONES` 为单一权威（UI 与 run_config 消费）；CNN 算法候选集由注册表能力元数据推导（ui.parameters.cnn_candidates 纯函数）；provenance 4 字段由 pipeline 传入原始值、_reporting 组装（架构前缀 native_、device 解析复用 resolve_device_name）；CV 隔离测试锁定"共享构造 + 逐 fit 深拷贝"现状语义。

**Tech Stack:** Python 3.13 / uv / pytest / sklearn / PyTorch（懒加载，保持 torch-free 导入链）

## Global Constraints

- 所有 Python 命令前缀 `uv run`；Python 代码中 Windows 路径用 `r'E:\...'` 或 `'E:/...'`
- 修改文件前必须先 Read（hooks 强制）
- commit message 不加 `Co-Authored-By` 署名；代码提交在 `feature/dual-arch-phase1` 分支（Task 1 Step 1 建分支）
- **零回归红线**：tests/integration/test_pipeline_deep.py、tests/contract/test_classifier_baseline.py 不改一行且必须通过
- **Rule 1 门禁约束**：本计划中"待创建"的文件路径一律不包裹反引号（check_doc_links 只豁免已存在文件）；待文件创建后再恢复引用
- **torch-free 契约**：build_encoder 导出不得破坏 base wheel 无 torch 可导入（vision.py 已有 try/except ImportError 保护，导出时不得在包根引入顶层 torch import）
- 每任务结束跑一次完整快速测试 `uv run pytest tests/ -v -m "not slow and not e2e"` 确认零回归，再提交

---

### Task 1: build_encoder 公共导出

**Files:**
- Create: tests/contract/test_build_encoder_export.py
- Modify: `pu_toolbox/estimators/deep/__init__.py:6,9-17`（import 行 + __all__）
- Modify: `pu_toolbox/__init__.py:25-31`（import 区）与 `:44-82`（__all__）

**Interfaces:**
- Consumes: 现有 `build_encoder`（vision.py:207-235，不修改其实现）
- Produces: `pu_toolbox.build_encoder` 与 `pu_toolbox.estimators.deep.build_encoder` 两个公共导出（同一对象）；Task 4 测试可改用包导出导入

- [ ] **Step 1: 建分支**

```bash
git checkout -b feature/dual-arch-phase1
```

- [ ] **Step 2: 写契约测试（红）**

Create tests/contract/test_build_encoder_export.py:

```python
# ruff: noqa: N802, N803, N806
"""Public export contract for build_encoder (dual_architecture_plan.md §5 阶段 1)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pu_toolbox import build_encoder  # noqa: E402
from pu_toolbox.estimators.deep import build_encoder as deep_export  # noqa: E402
from pu_toolbox.estimators.deep.vision import build_wconpu_backbone  # noqa: E402


@pytest.mark.contract
def test_exported_from_package_root():
    assert build_encoder is deep_export


@pytest.mark.contract
def test_mlp_returns_none():
    assert build_encoder("mlp", in_channels=3) is None


@pytest.mark.contract
def test_invalid_architecture_raises():
    with pytest.raises(ValueError, match="architecture"):
        build_encoder("lstm", in_channels=3)


@pytest.mark.contract
def test_cnn_matches_wconpu_backbone_structure():
    encoder = build_encoder("cnn", backbone="cnn13", in_channels=3)
    reference = build_wconpu_backbone("cnn13", in_channels=3)
    assert isinstance(encoder, torch.nn.Sequential)
    assert isinstance(reference, torch.nn.Sequential)
    x = torch.randn(2, 3, 8, 8)
    assert tuple(encoder(x).shape) == tuple(reference(x).shape)
```

- [ ] **Step 3: 运行测试确认失败**

Run: `uv run pytest tests/contract/test_build_encoder_export.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_encoder' from 'pu_toolbox'`（首条）

- [ ] **Step 4: 实现导出**

Modify pu_toolbox/estimators/deep/__init__.py — :6 改为（字母序 build_encoder 在 build_wconpu 之前）：

```python
from .vision import build_encoder, build_wconpu_augmentation, build_wconpu_backbone
```

`__all__`（:9-17）在 `"WeightedContrastivePUClassifier",` 之后、`"build_wconpu_augmentation",` 之前插入：

```python
    "build_encoder",
```

Modify pu_toolbox/__init__.py — import 区（:25 `from .estimators.deep.weighted_contrastive_pu import ...` 之后）追加：

```python
from .estimators.deep.vision import build_encoder
```

`__all__` 在 `"build_diagnostic_report",`（:77）之后插入（build_d < build_e 字母序）：

```python
    "build_encoder",
```

（不加顶层 torch import；build_encoder 定义不依赖 torch，调用时才需要，契约保持。）

- [ ] **Step 5: 运行测试确认通过**

Run: `uv run pytest tests/contract/test_build_encoder_export.py -v`
Expected: 4 项 PASS

- [ ] **Step 6: 全量快速测试零回归**

Run: `uv run pytest tests/ -v -m "not slow and not e2e"`
Expected: 全部 PASS（test_import.py 冒烟覆盖各子包导入不受影响）

- [ ] **Step 6.5: 门禁登记**

`scripts/check_test_quality.py` 为 tests/contract/test_build_encoder_export.py 登记 PARTIAL_COVERAGE（缺失 basic/determ 类别，理由：跨分类器契约套件，参数/边界行为由 test_classifier_baseline 与深度单测覆盖），条目格式仿 test_capability_declarations.py；Run: `uv run python scripts/check_test_quality.py`，Expected: 全部通过。

- [ ] **Step 7: 提交**

```bash
git add pu_toolbox/estimators/deep/__init__.py pu_toolbox/__init__.py tests/contract/test_build_encoder_export.py
git commit -m "feat(deep): build_encoder 公共导出

包根 pu_toolbox 与 estimators.deep 双层导出同一对象；契约测试锁定
mlp→None / 非法架构 ValueError / cnn 与 wconpu backbone 结构一致；
vision.py 既有 try/except 保持 torch-free 导入契约"
```

---

### Task 2: 报告 provenance 4 字段

**Files:**
- Create: tests/unit/workflows/test_report_provenance.py
- Modify: `pu_toolbox/workflows/_reporting.py:18-36`（签名）与 `:72-84`（provenance dict）
- Modify: `pu_toolbox/workflows/pipeline.py:565-582`（调用点传参）

**Interfaces:**
- Consumes: 现有 `build_pipeline_report` 唯一生产调用点（pipeline.py:565，grep 已确认）；`resolve_device_name`（core/device.py:11）
- Produces: `build_pipeline_report` 新增 keyword-only 参数 `architecture: str` / `backbone: str | None` / `device: str | None` / `encoder_in_channels: int | None`；provenance 新增 4 键

- [ ] **Step 1: 写单测（红）**

Create tests/unit/workflows/test_report_provenance.py:

```python
# ruff: noqa: N802, N803, N806
"""Provenance field tests: architecture/backbone/device/encoder
(dual_architecture_plan.md §5 阶段 1)."""

from __future__ import annotations

import pytest

from pu_toolbox.preprocessing.data_profiler import PUDataProfile
from pu_toolbox.workflows._reporting import build_pipeline_report
from pu_toolbox.workflows.report import PriorInfo

pytestmark = [pytest.mark.unit]


def _report(*, architecture, backbone, device, encoder_in_channels):
    profile = PUDataProfile(
        summary={"n_samples": 40, "n_features": 5, "positive_fraction": 0.25},
        feature_statistics={},
        selection_diagnostic={
            "separability_auc": None,
            "is_identifying": False,
            "status": "inconclusive",
        },
        issues=(),
        assumption_hints=(),
    )
    return build_pipeline_report(
        profile=profile,
        prior_info=PriorInfo(value=0.3, source="user", method_requires_prior=True),
        recommendation=None,
        cv_metrics={},
        classifier_name="wconpu",
        auto_mode=False,
        classifier_cls=None,
        skipped_candidates=[],
        y_true=None,
        splitter=None,
        n_splits=2,
        final_model=None,
        diagnostic=None,
        random_state=42,
        classifier_params={},
        sample_weight=None,
        architecture=architecture,
        backbone=backbone,
        device=device,
        encoder_in_channels=encoder_in_channels,
    )


@pytest.mark.unit
def test_mlp_provenance_reports_native_mlp_without_backbone_or_encoder():
    report = _report(
        architecture="mlp", backbone=None, device="auto", encoder_in_channels=None
    )
    p = report.provenance
    assert p["architecture"] == "native_mlp"
    assert p["backbone"] is None
    assert p["encoder"] is None
    assert p["device"]["requested"] == "auto"
    assert p["device"]["resolved"] in {"cpu", "cuda"}


@pytest.mark.unit
def test_cnn_provenance_reports_native_cnn_with_backbone_and_encoder_summary():
    report = _report(
        architecture="cnn", backbone="cnn13", device="cpu", encoder_in_channels=3
    )
    p = report.provenance
    assert p["architecture"] == "native_cnn"
    assert p["backbone"] == "cnn13"
    assert p["encoder"] == {"backbone": "cnn13", "in_channels": 3}
    assert p["device"] == {"requested": "cpu", "resolved": "cpu"}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/unit/workflows/test_report_provenance.py -v`
Expected: FAIL with `TypeError: build_pipeline_report() got an unexpected keyword argument 'architecture'`

- [ ] **Step 3: 实现 _reporting.py**

:18-36 签名在 `sample_weight: np.ndarray | None,` 之后追加 4 个参数：

```python
    architecture: str,
    backbone: str | None,
    device: str | None,
    encoder_in_channels: int | None,
```

文件顶部 import 区（:10-15）追加：

```python
from ..core.device import resolve_device_name
```

:72-84 provenance dict 在 `"skipped_candidates": skipped_candidates,` 之后追加：

```python
        "architecture": f"native_{architecture}",
        "backbone": backbone,
        "device": {"requested": device, "resolved": resolve_device_name(device)},
        "encoder": (
            {"backbone": backbone, "in_channels": encoder_in_channels}
            if architecture == "cnn"
            else None
        ),
```

- [ ] **Step 4: 运行单测确认通过**

Run: `uv run pytest tests/unit/workflows/test_report_provenance.py -v`
Expected: 2 项 PASS

- [ ] **Step 5: pipeline 传参**

Modify pu_toolbox/workflows/pipeline.py — :565-582 调用点在 `sample_weight=sample_weight,` 之后追加：

```python
            architecture=self.architecture,
            backbone=self.backbone,
            device=self.device,
            encoder_in_channels=int(X.shape[1]) if self.architecture == "cnn" else None,
```

（`X` 为 fit_evaluate 参数，作用域内可用；与 :492 `int(X.shape[1])` 同源。）

- [ ] **Step 6: 红线验证 + 全量快速测试**

Run: `uv run pytest tests/integration/test_pipeline_deep.py tests/unit/workflows/test_report_provenance.py -v`
Expected: 全部 PASS（现有 provenance 断言均为按键访问，无键集合断言，新增键不破坏）
Then Run: `uv run pytest tests/ -v -m "not slow and not e2e"`
Expected: 全部 PASS

- [ ] **Step 7: 门禁登记**

`scripts/check_test_quality.py` 为 tests/unit/workflows/test_report_provenance.py 登记（缺失类别按实际补齐，条目格式仿既有 PARTIAL_COVERAGE）；Run: `uv run python scripts/check_test_quality.py`，Expected: 全部通过。

- [ ] **Step 8: 提交**

```bash
git add pu_toolbox/workflows/_reporting.py pu_toolbox/workflows/pipeline.py tests/unit/workflows/test_report_provenance.py scripts/check_test_quality.py
git commit -m "feat(workflows): 报告 provenance 增加架构能力 4 字段

architecture(native_mlp/native_cnn)/backbone/device(requested+resolved)/
encoder(构造摘要或 null)顶层平铺；pipeline 传原始值、_reporting 组装；
schema_version 保持 1.0（纯 additive 扩展）"
```

---

### Task 3: UI 元数据驱动

**Files:**
- Modify: `pu_toolbox/estimators/deep/vision.py`（CNN_BACKBONES 常量）
- Modify: `pu_toolbox/ui/parameters.py:12`（_MANAGED_PARAMS 之后新增函数）
- Create: tests/unit/ui/test_cnn_candidates.py
- Modify: `pu_toolbox/ui/app.py:19,92-93,117,151,166,176,214`
- Modify: `pu_toolbox/run_config.py:11,89`

**Interfaces:**
- Consumes: 阶段 0 的 `AlgorithmMetadata.native_architectures`（registry/metadata.py:87）
- Produces: `CNN_BACKBONES: tuple[str, ...]`（vision.py 模块常量）；`cnn_candidates() -> set[str]`（ui/parameters.py）；Task 4 不依赖、Task 5 文档提及两者

- [ ] **Step 1: vision.py 新增常量**

Modify pu_toolbox/estimators/deep/vision.py — 在 build_wconpu_backbone 定义之前（:77 前）插入：

```python
CNN_BACKBONES: tuple[str, ...] = ("cnn13", "resnet18", "resnet50")
"""Supported CNN backbone names (single source of truth for UI and config)."""
```

（build_wconpu_backbone 的 `name: Literal["cnn13","resnet18","resnet50"]` 注解已与常量一致，无需修改。）

- [ ] **Step 2: ui/parameters.py 新增 cnn_candidates**

在 `_MANAGED_PARAMS`（:12）之后、`classifier_catalog` 之前插入：

```python
def cnn_candidates() -> set[str]:
    """Return trainable classifier names whose capability declaration includes cnn."""
    register_all_builtin_methods()
    return {
        metadata.name
        for metadata in list_algorithms(trainable_only=True)
        if "cnn" in metadata.native_architectures
    }
```

（register_all_builtin_methods 幂等，UI 侧与 classifier_catalog 复用同一注册流程。）

- [ ] **Step 3: 写单测**

Create tests/unit/ui/test_cnn_candidates.py:

```python
# ruff: noqa: N802, N803, N806
"""UI candidate-set derivation from registry capability metadata."""

from __future__ import annotations

import pytest

from pu_toolbox.registry import list_algorithms, register_all_builtin_methods
from pu_toolbox.ui.parameters import cnn_candidates

pytestmark = [pytest.mark.unit]


@pytest.mark.unit
def test_cnn_candidates_matches_registry_declarations():
    register_all_builtin_methods()
    candidates = cnn_candidates()
    for name in candidates:
        meta = next(m for m in list_algorithms(trainable_only=True) if m.name == name)
        assert "cnn" in meta.native_architectures, name
    for meta in list_algorithms(trainable_only=True):
        if "cnn" in meta.native_architectures:
            assert meta.name in candidates, meta.name


@pytest.mark.unit
def test_cnn_candidates_matches_current_declarations():
    """Phase-0 declarations: only infomax/wconpu support cnn today."""
    assert cnn_candidates() == {"infomax_pu", "weighted_contrastive_pu"}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/unit/ui/test_cnn_candidates.py -v`
Expected: 2 项 PASS

- [ ] **Step 5: app.py 替换硬编码**

:19 import 行改为：

```python
from pu_toolbox.ui.parameters import cnn_candidates, classifier_catalog, render_parameter_form
```

:14 附近（`from pu_toolbox.run_config import RunConfiguration` 之前，字母序 estimators < run_config）插入：

```python
from pu_toolbox.estimators.deep.vision import CNN_BACKBONES
```

`catalog = classifier_catalog()`（:92）之后追加：

```python
    cnn_names = cnn_candidates()
```

:117 改为（原 `or (X.ndim == 4 and name not in {"infomax_pu", "weighted_contrastive_pu"})`）：

```python
                or (X.ndim == 4 and name not in cnn_names)
```

:151 与 :166 两处相同的 `(not image_mode or name in {"infomax_pu", "weighted_contrastive_pu"})` 均改为：

```python
            (not image_mode or name in cnn_names)
```

:176 改为：

```python
    if image_mode:
        st.info("图像输入使用 CNN 模式，目前支持：" + "、".join(sorted(cnn_names)) + "。")
```

:214 改为：

```python
        backbone = st.selectbox("CNN 骨架", list(CNN_BACKBONES), key="backbone")
```

- [ ] **Step 6: run_config.py 白名单改来源**

:11（`from .workflows import DEFAULT_METRICS`）之后追加：

```python
from .estimators.deep.vision import CNN_BACKBONES
```

:89 改为：

```python
        if backbone not in CNN_BACKBONES:
```

（错误信息文案不变；test_run_config.py 无 backbone 断言，grep 已确认。）

- [ ] **Step 7: 全量快速测试零回归**

Run: `uv run pytest tests/ -v -m "not slow and not e2e"`
Expected: 全部 PASS（UI streamlit 测试本地 skip，不受影响）

- [ ] **Step 8: 门禁登记**

`scripts/check_test_quality.py` 为 tests/unit/ui/test_cnn_candidates.py 登记（缺失类别按实际补齐）；Run: `uv run python scripts/check_test_quality.py`，Expected: 全部通过。

- [ ] **Step 9: 提交**

```bash
git add pu_toolbox/estimators/deep/vision.py pu_toolbox/ui/parameters.py pu_toolbox/ui/app.py pu_toolbox/run_config.py tests/unit/ui/test_cnn_candidates.py scripts/check_test_quality.py
git commit -m "feat(ui): CNN 候选集与骨架清单元数据驱动

- cnn_candidates 纯函数从 registry 能力声明推导（app.py 3 处硬编码集合
  与信息文本替换）
- CNN_BACKBONES 常量移入 vision.py 单一权威，selectbox 与 run_config
  白名单消费
- 单测锁定推导与声明双向一致 + 当前声明集"
```

---

### Task 4: CV fold 训练隔离测试

**Files:**
- Create: tests/integration/test_cv_fold_isolation.py

**Interfaces:**
- Consumes: Task 1 导出的 build_encoder（包导出）；现有 `PUPipeline._fresh_estimator`（pipeline.py:711-728）与 `pipe._classifier_cls`（test_pipeline_deep.py:95 有使用先例）；WConPU fit 内 deepcopy 语义（weighted_contrastive_pu.py:172,185-186）
- Produces: 无代码接口；锁定"共享构造 + 逐 fit 深拷贝"隔离语义

- [ ] **Step 1: 写集成测试**

Create tests/integration/test_cv_fold_isolation.py:

```python
# ruff: noqa: E402, N802, N803, N806
"""CV fold training isolation: the shared encoder template must not leak
weights across folds (dual_architecture_plan.md §5 阶段 1)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox import build_encoder  # noqa: E402
from pu_toolbox.workflows import PUPipeline  # noqa: E402


def _image_data(n=24, channels=3, size=8, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.normal(0.5, 0.3, size=(n, channels, size, size)).astype(np.float32)
    y_pu = np.concatenate([np.ones(8, dtype=int), np.zeros(n - 8, dtype=int)])
    return X, y_pu


def _snapshot(model):
    """Detached weight snapshot of a torch module."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.integration
def test_cv_folds_do_not_leak_encoder_weights():
    X, y_pu = _image_data()
    pipe = PUPipeline(
        classifier="wconpu",
        architecture="cnn",
        backbone="cnn13",
        cv=2,
        max_epochs=1,
        random_state=42,
    )
    pipe._encoder = build_encoder("cnn", backbone="cnn13", in_channels=3)
    template_initial = _snapshot(pipe._encoder)

    # Fold 1 trains a deep copy; the shared template must stay untouched
    # (it is fold 2's starting point).
    clf1 = pipe._fresh_estimator(pipe._classifier_cls, None, 0.3)
    clf1.fit(X[:12], y_pu[:12], class_prior=0.3)
    assert not _same(_snapshot(clf1.encoder_), template_initial)  # training took effect
    assert _same(_snapshot(pipe._encoder), template_initial)  # template untainted
    fold1_after = _snapshot(clf1.encoder_)

    # Fold 2 trains its own copy; fold 1's weights must not move.
    clf2 = pipe._fresh_estimator(pipe._classifier_cls, None, 0.3)
    clf2.fit(X[12:], y_pu[12:], class_prior=0.3)
    assert _same(_snapshot(clf1.encoder_), fold1_after)  # fold 2 did not touch fold 1
    assert not _same(_snapshot(clf2.encoder_), template_initial)  # fold 2 trained
    assert _same(_snapshot(pipe._encoder), template_initial)  # template never trained

    # Object isolation: three distinct module objects.
    assert clf1.encoder_ is not clf2.encoder_
    assert clf1.encoder_ is not pipe._encoder
    assert clf2.encoder_ is not pipe._encoder
```

- [ ] **Step 2: 运行测试确认通过（直接绿——锁定现状语义）**

Run: `uv run pytest tests/integration/test_cv_fold_isolation.py -v`
Expected: 1 项 PASS

- [ ] **Step 3: 全量快速测试零回归**

Run: `uv run pytest tests/ -v -m "not slow and not e2e"`
Expected: 全部 PASS（新增约 20-40s 训练时间）

- [ ] **Step 4: 门禁登记**

`scripts/check_test_quality.py` 为 tests/integration/test_cv_fold_isolation.py 登记（缺失类别按实际补齐）；Run: `uv run python scripts/check_test_quality.py`，Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add tests/integration/test_cv_fold_isolation.py scripts/check_test_quality.py
git commit -m "test(deep): CV fold 训练隔离测试（fold 间权重不泄漏）

共享 encoder 模板 2 折真实训练：防假阳性（折内权重确实在变）、
模板不被训练（折 2 起点无污染）、折间权重互不影响、对象隔离。
锁定'共享构造 + 逐 fit 深拷贝'现状语义（dual_architecture_plan §5）"
```

---

### Task 5: 文档收尾与全量验收

**Files:**
- Modify: `docs/user/reference/api.md:87`（provenance 节）
- Modify: `docs/dev/2026-08-30-dual-arch-phase1-design.md`（§5 验收勾选无——设计文档不改，本任务仅 api.md）

**Interfaces:**
- Consumes: Task 2 的 4 字段名与取值语义
- Produces: 无代码接口；阶段 1 文档闭环

- [ ] **Step 1: 更新 api.md provenance 节**

Modify docs/user/reference/api.md — :87（sample_weight provenance 段之后）追加：

```markdown
`provenance["architecture"]` 为 `"native_mlp"` / `"native_cnn"`；
`provenance["backbone"]` 为 CNN 骨架名（MLP 为 `None`）；`provenance["device"]`
保存 `{"requested", "resolved"}` 两键；`provenance["encoder"]` 为注入 encoder 的
构造摘要（`{"backbone", "in_channels"}`，MLP 无注入时为 `None`）。
```

- [ ] **Step 2: 登记结构文档**

各任务提交时已随任务完成 check_test_quality 登记（Task 1/2/3/4 各带 Step 6.5/7.5），本步仅核验：Run `uv run python scripts/check_test_quality.py`，Expected: 全部通过（若仍有遗漏，补登并说明）。

- Run: `uv run python scripts/generate_structure.py --update`
- Expected: project_structure.md 登记 4 个新测试文件（无其他无关差异；若出现无关差异检查是否误改）

- [ ] **Step 3: 全量验收**

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

Expected: 快速测试全部 PASS；7 项门禁全部通过（本计划中的待创建路径已全部落地，不触发 Rule 1）。

- [ ] **Step 3: 报告目检**

```bash
uv run python -c "
import numpy as np
from pu_toolbox.workflows import PUPipeline
rng = np.random.RandomState(1)
X = rng.normal(0.5, 0.3, size=(24, 3, 8, 8)).astype(np.float32)
y = np.concatenate([np.ones(8, int), np.zeros(16, int)])
r = PUPipeline(classifier='wconpu', architecture='cnn', cv=2, max_epochs=1, refit=False).fit_evaluate(X, y, class_prior=0.3)
print({k: r.provenance[k] for k in ('architecture', 'backbone', 'device', 'encoder')})
"
```

Expected: 输出 `architecture: native_cnn, backbone: cnn13, device: {requested: auto, resolved: cpu 或 cuda}, encoder: {backbone: cnn13, in_channels: 3}`（native_mlp 对应运行同理，可在上一命令改 architecture 参数复核）。

- [ ] **Step 4: 提交**

```bash
git add docs/user/reference/api.md
git commit -m "docs: api.md provenance 节补充架构能力 4 字段说明"
```

---

## 验收总览（对照设计规范 §5）

1. 新增测试全绿：导出契约 4 项 + provenance 单测 2 项 + cnn_candidates 单测 2 项 + CV 隔离 1 项（Task 1-4）；
2. tests/integration/test_pipeline_deep.py 与 tests/contract/test_classifier_baseline.py 全程未改一行且通过（Task 2 Step 6 + Task 5 全量）；
3. UI 硬编码算法集清零：app.py 无 {"infomax_pu", "weighted_contrastive_pu"} 字面量残留于筛选逻辑（Task 3，grep 确认）；
4. 快速测试 + 7 门禁全过（Task 5 Step 2）；
5. 报告 JSON 目检 4 新字段（Task 5 Step 3）；
6. 分支 feature/dual-arch-phase1 共 5 个提交，PR 合并回 main。
