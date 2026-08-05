# Deep PU 算法接入 Pipeline/CLI：架构选择功能设计

> 状态：已批准（2026-08-05）。实现后本文件蒸馏进 `decision_log.md` 并删除。

## 背景与目标

工具箱已有四个深度 PU 算法（WConPU / InfoMax PU / DGPU / Self-PU，backend=TORCH），其中 WConPU 具备图像 CNN backbone 接入（`estimators/deep/vision.py`：cnn13 / resnet18 / resnet50 + 图像增强），但其 CNN 逻辑目前仅在 benchmark 层使用。PUPipeline 与 CLI 完全不支持深度算法：auto 模式经 `_missing_required_params` 跳过它们，显式指定也被拒绝；CLI 输入只有 2D 表格 CSV。

目标：在 Pipeline 与 CLI 中支持用户**选择 MLP / CNN 架构**训练和使用深度算法，同时支持表格（2D）与图像（4D NCHW）两种数据形态。

## 需求决策（已与用户确认）

| 决策点 | 结论 |
|---|---|
| 数据形态 | 表格 + 图像都支持：表格走 MLP；图像走 CNN |
| 算法范围 | WConPU + InfoMax PU（DGPU 为 GAN 结构、Self-PU 为三网络蒸馏，无单骨架插拔概念，不接入） |
| 参数形态 | 两级：`architecture`（mlp/cnn 主开关）+ `backbone`（cnn13/resnet18/resnet50，仅 cnn 有效） |
| 图像输入格式 | CLI `--data` 接受 `.npy` 4D NCHW 数组文件（零新依赖，复用 numpy） |

## 设计

### 1. 算法层（`pu_toolbox/estimators/deep/`）

**WConPU（`weighted_contrastive_pu.py`）——已天然支持，不改核心逻辑**：
- `encoder=None` → 默认 MLP（单隐藏层），现有行为不变
- `encoder=build_wconpu_backbone(...)` → CNN（cnn13/resnet18/50）
- `validate_pu_X_y(allow_nd=self.encoder is not None)` 已放行 4D 图像输入

**InfoMax PU（`infomax_pu.py`）——需加 encoder 插拔**：
- `InfoMaxPURepresentation` 加 `encoder=None` 参数：None → 现有 MLP 构建（向后兼容）；传入 → 外置 encoder 替代 `nn.Sequential(Linear...)` 部分，`ratio_head_` 接在 encoder 特征之后
- `InfoMaxPUClassifier` 加 `encoder=None` 透传；`validate_pu_X_y` 增加 `allow_nd=self.encoder is not None`
- CNN 图像输入的标准化由 backbone 内嵌 `ChannelNormalize` 处理（复用 vision.py 现有逻辑）

**`vision.py` 补统一构建入口**：

```python
def build_encoder(architecture: Literal["mlp", "cnn"], *, backbone: str = "cnn13",
                  in_channels: int, normalization_mean, normalization_std):
    """mlp → None（WConPU 默认 MLP / InfoMax 内部 MLP）；
    cnn → build_wconpu_backbone(backbone, ...)。"""
```

### 2. Pipeline 层（`pu_toolbox/workflows/pipeline.py`）

**新参数**：`PUPipeline(..., architecture="mlp", backbone="cnn13")`
- `architecture`: `"mlp" | "cnn"`；`backbone`: `"cnn13" | "resnet18" | "resnet50"`（仅 cnn 时有效）

**实例化逻辑扩展**（`_resolve_classifier_name` / `_pick_first_instantiable` 附近）：
- 显式指定 `wconpu` / `infomax_pu` 时：放行必填参数检查；WConPU 的 `class_prior` 由 pipeline 注入（复用现有 prior 解析顺序：显式 > 估计）；`architecture="cnn"` 时构建 encoder 注入
- auto 模式行为不变：推荐器候选中的 deep 算法仍被跳过（WConPU 构造必填 `class_prior` 的检查保留）

**输入维度校验**：
- 2D 表格 + mlp → 正常
- 4D 图像（NCHW）+ cnn → 正常（CV splitter 按索引切分，天然兼容）
- 2D + cnn 或 4D + mlp → 明确报错
- deep 算法 + cv>1 时打印训练成本提示（800 epoch × folds）

### 3. CLI 层（`pu_toolbox/cli/run.py`）

- **`--data` 按扩展名分发**：`.csv` → 现有加载；`.npy` → `np.load` + 校验（float32、4D NCHW、有限值）
- **新参数**：`--architecture {mlp,cnn}`（默认 mlp）、`--backbone {cnn13,resnet18,resnet50}`（默认 cnn13）、`--device`（默认 cpu，透传 deep 算法）
- **校验**：`--architecture cnn` 但 `--classifier` 非 wconpu/infomax_pu → 报错；`--backbone` 在 mlp 下指定 → 报错（不静默忽略）

### 4. 测试

| 层 | 覆盖 |
|---|---|
| vision.py | `build_encoder` 两架构分支、参数校验 |
| infomax_pu.py | encoder 插拔（None 兼容 + CNN 4D 输入）、fit 全流程 |
| pipeline | wconpu 显式指定 + prior 注入、infomax 接入、架构/维度校验、auto 跳过 deep 不变 |
| CLI | .npy 加载、参数透传、错误路径（cnn+浅层算法、mlp+4D） |

### 5. 文档

`docs/user/howto/cli.md`、`docs/user/howto/pipeline.md`、`docs/user/reference/api.md` 补 deep 算法与架构参数说明；`docs/project_management/process_checklist.md` 追加完成记录行。

## 范围外

- DGPU / Self-PU 不接入 Pipeline/CLI
- 图片目录输入不支持（仅 `.npy` 4D 数组）
- auto 推荐行为不改变（deep 算法不进入推荐候选）
- 不新增 torch 依赖声明（已为可选依赖）
