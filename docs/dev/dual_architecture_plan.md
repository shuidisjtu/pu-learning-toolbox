# PU 双架构渐进式升级方案

适用范围：显式 estimator、Registry 能力声明、encoder 构造和实验层图像评估。

## 1. 背景与目标

深度 PU 方法可以共享表征网络的构造方式，但仍各自决定预测头、损失和训练循环。
例如 `infomax_pu`、`weighted_contrastive_pu` 支持原生 MLP/CNN；要处理图像时，调用方
按约定将 CNN encoder 交给具体估计器。对传统算法使用 CNN 特征，是另一类实验路径。

本方案让这两类路径都可测试、可比较、可追踪，不要求所有估计器共用构造参数或 trainer。
新增架构时要保留原有二维路径的算法语义，并明确新路径的表示、拟合范围、预算和评估方法。

## 2. 参考项目及吸收内容

### PUBench

PUBench 通过公共 `Algorithm` 基类和 `Featurizer`，让不同 PU loss 复用 MLP 或
ResNet 特征网络。工具箱借鉴“共享 encoder、算法 head/loss 解耦”的思想，但不
直接复制其算法代码或按数据集名称隐式选模型的方式。

- [PUBench `networks.py`](https://github.com/wu-dd/PUBench/blob/main/core/networks.py)
- [PUBench `algorithms.py`](https://github.com/wu-dd/PUBench/blob/main/core/algorithms.py)

### PU-Bench

PU-Bench 的 backbone factory、统一 trainer 生命周期、方法配置与训练代码分离，
可作为工具箱整理 `vision.py`、Registry 和实验配置的参考。工具箱保留现有
scikit-learn 风格的 estimator 契约，不强迫所有算法使用同一个 trainer。

- [PU-Bench 项目](https://github.com/XiXiphus/PU-Bench)

### LIBSVM

LIBSVM 的 problem/parameter/model 分离、训练前参数校验、统一 train/predict/CV
接口和稳定保存加载契约，作为本方案的工程设计参考。只借鉴工具箱组织方式，
不复用其 SVM 数学求解器。

- [LIBSVM 官方 README](https://github.com/cjlin1/libsvm/blob/master/README)

## 3. 能力分类

| 路径 | 含义 |
|---|---|
| `native_mlp` | 估计器的原生 MLP 路径。 |
| `native_cnn` | 估计器的原生 CNN 路径；是否训练传入的 encoder 由独立字段声明。 |
| `cnn_feature_adapter` | 先提取 CNN 特征，再用二维特征拟合估计器；不算该方法的原生 CNN 实现。 |
| `tabular_only` | Registry 对未声明任何原生架构的标记；实际接受的维度另看 `input_ndims`。 |

`input_ndims` 说明能接受哪些维度，不说明怎样处理这些维度。估计器可能接受四维输入但将其展平，
不能因此标记为 `native_cnn`。完整能力清单以 Registry 为准，不在本计划另维护一份名单。

## 4. 总体架构

### 4.1 Encoder 契约

支持注入的估计器要求 encoder 将其支持的输入映射为 `(batch, feature_dim)`，输出不是最终类别分数。
调用方可以用公开的 `build_encoder` 构造共享骨架，或提供满足具体估计器要求的自定义网络。
内置 CNN 名称以 `vision.py` 的 `CNN_BACKBONES` 为准。

训练时，拟合后的状态不得在候选或分区之间泄漏。对于在 `fit` 内深拷贝 encoder 的估计器，
复用同一个未拟合模板可共享初始化而不共享更新后的权重。不能把这种实现假定为所有方法的保证；
新方法应验证其 clone、实际拟合、预测及所承诺的保存恢复行为。

### 4.2 Registry 能力声明

四个字段分别表达原生结构、输入维度、注入参数及训练责任，例如：

```python
native_architectures = {"mlp", "cnn"}
input_ndims = {2, 4}
encoder_parameter = "encoder"
trains_encoder = True
```

Registry 同步估计器类上实际声明的字段，用于查询及 `list-methods` 展示。
`encoder_parameter` 命名构造参数，不是自动注入指令。声明、构造签名和实际接受的输入必须相符。

### 4.3 估计器与实验层的分工

- **直接拟合**：调用方选择 estimator、构造及传入 encoder、指定设备与先验，自己管理数据划分。
  估计器校验自身输入，并实现模型头、PU 目标与训练循环。
- **四角色实验**：调用方把已构造的 estimator 交给 `ExperimentRunner`。
  Runner 按实验数据维度、所请求的架构和模型声明预检，并记录候选、验证选择和独立测试结果。
  Survey 的骨架、表征和训练预算由对应配方定义，不能从 Registry 的能力字段推断。
- **固定特征适配**：实验层可用同一份冻结 encoder 对不同数据角色提取二维特征；
  若 encoder 需要训练，只能使用许可的训练数据，不能让验证或测试参与拟合。

## 5. 渐进式实施阶段

### 阶段 0：能力契约和文档

2026-08-30 完成四项能力字段及 Registry 同步、`list-methods` 展示、`validate_encoder_features`
共享校验和能力声明测试。新增方法沿用[接入模板](new_algorithm_template.md)。

### 阶段 1：整理现有双架构实现

2026-08-30 将通用 `build_encoder` 从包根及 `estimators.deep` 导出，保留原有 WConPU 构造入口。
CNN13 和 ResNet 系列的模型结构由 encoder 工厂维护；算法一致性仍通过各自的输入形状、
拟合、确定性及保存恢复用例验证，而非由工厂统一保证。

### 阶段 2：以 `nnpu` 为首个试点

2026-08-30 接入可选的 `encoder`。提供时，nnPU 在 `fit` 内深拷贝 encoder，探测输出特征维度，
将传入的 `model` 用作分数头（未提供时构造线性头）。未提供 encoder 时，`model` 仍是完整网络，
默认 `Linear(d, 1)`。因此可以替换表征而不改变 nnPU 风险目标。

2026-10-20 接入 Dist-PU 的可选 encoder，仅用于显式 `training_recipe="author_staged"` 和
`distribution_loss="author_histogram"`；默认保留 clean-room 全量路径，冻结 Pilot 的图像条目仍用固定特征。
前者在训练时复制 encoder，按批执行并保留数值保护；细节见 [Dist-PU 方法卡](../research/method_cards/Dist-PU.md)。

### 阶段 3：逐个评估复杂深度算法

Self-PU 可接受二维和四维输入，但声明的原生架构只有 MLP，四维可由默认网络展平。
它的元学习仍需要独立的干净验证监督。DGPU 有自身的生成器协议。是否接入原生 CNN 应逐算法评估，
不能把完整 `backbone` 或 `model` 仅重命名为 encoder，也不能因新表征消除既有监督要求。

### 阶段 4：传统算法图像特征适配

`experiment/feature_adapter.py` 的 `adapt_image_bundle_to_features` 生成二维特征和表示记录；
`LeaderboardRunSpec` 区分原生和特征适配分组。该路径按 Survey 协议 §2.5 单独调参及报告，
不替换已有 `tabular_native` 基准。

## 6. 架构演进要求

保留估计器既有二维输入和构造参数语义。接入新架构时，明确特征维度、归一化、初始权重、
encoder 是否可训练以及在哪个数据分区拟合。需要可复现实验或模型恢复时，记录 seed、设备、
实际配置及权重来源，并在承诺的设备和序列化路径下验证。新架构的实验预算应独立核对，不能复用
旧架构的调参结果冒充验证证据。

## 7. 当前实验的公平性协议

具体数据集、标记预算、训练风险视图及评估口径以 [Survey 协议](../research/pu_survey/pu_survey_protocol.md)
为准。PA/OA 选择、OS 输入下的 TS-compatible 风险视图和图像架构是不同的设计维度；
同样的 CNN 结构不保证训练目标或验证监督一样。表格和文本 SBERT 特征不依赖原生 CNN 接入。

### 轨道 A：原生算法比较

在同一表示和协议组内比较各算法；按方法声明区分二维输入和原生 CNN 路径。

### 轨道 B：架构比较

比较同一算法支持的 MLP/CNN 实现，固定源样本划分、标记策略、seed、选择证据和预算，
分别报告 `native_mlp` 与 `native_cnn`。

### 轨道 C：CNN 特征适配

先固定 encoder 的来源或训练范围，再提取特征供二维方法使用。
单独记录表示、训练范围和调参预算，不把其结果与端到端原生 CNN 混为同一组。

各轨道记录实际先验消费、候选与 epoch 预算、使用的验证监督、最终评估样本及有效 seed 数。
配置允许真实先验和估计先验对照时，分别记明来源和额外监督，不把不同条件合并报告。

## 8. 传统算法引入 CNN 的风险

- **线性算法**：CNN 特征上的线性模型不再是原始输入空间中的线性模型。
- **核算法**：CNN 改变距离、内积、核宽度和矩阵条件数；KLDCE 的 ACS/SMO
  求解语义不能简单视为增加输入层。
- **数值优化/闭式求解算法**：可能破坏原有设计矩阵、凸性、闭式解或数值稳定性。
- **比较解释**：性能变化可能来自 CNN 表征，而不是 PU 算法本身。

因此，传统算法图片适配可能提升也可能降低性能，必须与原生表格结果分开解释。

## 9. 新算法接入最低要求

新算法注册时必须声明：

- 支持的输入维度和模态；
- `native_architectures`；
- 是否接收和训练 encoder；
- 是否支持 GPU、稀疏输入和 sample weight。

若声明支持 CNN，至少提供：

- CNN smoke training；
- 输入和输出形状测试；
- 独立拟合之间的模型状态隔离测试；
- 固定 seed 测试；
- CPU/GPU（可用时）测试；
- save/load/predict round-trip 测试；
- 不支持架构的 fail-fast 测试。

## 10. 最终决策

本项目采用以下长期结构：

```text
能力声明
  + 统一 encoder factory
  + 深度算法按需适配
  + 传统算法保持 tabular_native
  + 必要时单独提供 cnn_feature_adapter
  + 显式数据划分、seed、预算和报告约束
```

架构扩展以显式 estimator 为训练入口，以实验协议管理比较边界，并保留 scikit-learn 风格的估计器契约。
