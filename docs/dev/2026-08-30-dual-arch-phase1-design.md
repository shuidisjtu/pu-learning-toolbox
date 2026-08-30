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
   - 无 torch 环境下包根导入不炸（沿用现有 torch-free 导入测试模式）。

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
