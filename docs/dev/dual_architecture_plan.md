# PU 双架构渐进式升级方案

状态：设计方案（不改变当前默认行为；已对照 v1.11.0 代码现状核实修订）  
适用范围：`PUPipeline`、Registry、深度估计器、实验与结果报告

## 1. 背景与目标

当前工具箱已经为 `infomax_pu` 和 `weighted_contrastive_pu`（别名
`wconpu`）提供了可用的 MLP/CNN 双架构：表格输入使用内置 MLP，图像输入由
Pipeline 构造并注入 CNN encoder。

本方案的目标不是让所有算法表面上拥有同一组参数，而是：

1. 为研究者提供可解释、可复现的架构比较能力；
2. 为实验型开发者提供能力感知的统一配置和错误提示；
3. 保留传统 PU 算法的原始实现、调参结果和论文语义；
4. 让未来适合的深度算法能够低成本接入 MLP/CNN；
5. 保证交叉验证、随机种子、预算、保存加载和报告的一致性。

本方案遵循 YAGNI：先解决当前实验和接口治理问题，不进行一次性“大一统”重构。

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
sklearn estimator/Pipeline 契约，不强迫所有算法使用同一个 trainer。

- [PU-Bench 项目](https://github.com/XiXiphus/PU-Bench)

### LIBSVM

LIBSVM 的 problem/parameter/model 分离、训练前参数校验、统一 train/predict/CV
接口和稳定保存加载契约，作为本方案的工程设计参考。只借鉴工具箱组织方式，
不复用其 SVM 数学求解器。

- [LIBSVM 官方 README](https://github.com/cjlin1/libsvm/blob/master/README)

## 3. 能力分类

每个算法应明确属于以下一种或多种能力模式：

| 能力模式 | 含义 |
|---|---|
| `native_mlp` | 算法原生使用二维表格/MLP 表征 |
| `native_cnn` | PU 目标端到端训练 CNN encoder |
| `cnn_feature_adapter` | CNN 提取特征后运行传统算法；不是原生 CNN |
| `tabular_only` | 当前正式实现只接受二维表格特征 |

当前建议分类：

- `native_mlp` + `native_cnn`：`infomax_pu`、`weighted_contrastive_pu`；
- 优先评估 encoder 适配：`nnpu`、`dist_pu`；
- 需要逐个改造复杂接口：`self_pu`、`dgpu`；
- 当前保持 `tabular_only`：`elkan_noto`、`upu`、`pnu`、`centroid_pu/ldce`、
  `kldce`、`llsvm`、`pusb`、`pusb_kernel`、`lbe`。

`tabular_only` 是当前实现和算法语义的正式边界，不表示数学上绝对不能把图片
转换为特征。若以后增加传统算法的图像能力，必须单独标记为
`cnn_feature_adapter`。

## 4. 总体架构

```text
2-D table ──┐
            ├─ EncoderFactory ─→ features ─→ PU algorithm head/loss ─→ model
4-D image ──┘
```

### 4.1 Encoder 契约

统一 encoder 的最低契约：

- 输入为二维表格或 NCHW 图像；
- 输出形状为 `(batch, feature_dim)`；
- 不输出最终 PU logits；
- 可被深拷贝、sklearn clone 相关流程和 pickle 使用；
- 不在全量数据上提前拟合；
- CV 中的构造语义采用"共享构造 + 逐 fit 深拷贝"：Pipeline 以同一
  random_state 构造一次 encoder，每个 fold 的估计器在 fit 内深拷贝出
  独立副本，训练互不泄漏；初始化随机源在 fold 间共享（同一 seed 下各
  fold 初始权重一致，现有测试锁定该语义）。"逐 fold 独立随机初始化"
  是可选改进，仅在有明确实验需求（如研究 fold 间初始化多样性）时实施，
  并同步更新锁定语义的测试。

`architecture="mlp"` 仍返回/使用算法内置 MLP；`architecture="cnn"` 使用公共
factory 构造 CNN/ResNet。现有 `build_encoder` 函数继续兼容（当前仅在模块内
使用，阶段 1 补公共导出），必要时增加别名而不删除旧入口。

### 4.2 Registry 能力声明

在现有 `AlgorithmMetadata` 基础上增加最小能力信息，例如：

```python
native_architectures = {"mlp", "cnn"}
adapter_architectures = set()
input_ndims = {2, 4}
encoder_parameter = "encoder"
trains_encoder = True
```

不支持 CNN 的算法显式声明 `tabular_only` 或仅 `{2}`。当前唯一的架构 gate 是
构造函数签名检查；元数据能力字段是本方案的新增内容。落地顺序：阶段 0 新增
元数据字段，与签名检查并行生效（二者结论应一致）；稳定后签名检查降级为
兼容性兜底。

新增字段归属沿用现有分工：与实现强绑定、适合类属性声明的能力字段
（`input_ndims`、`encoder_parameter`、`trains_encoder`、
`native_architectures` 等）以估算器类属性为权威来源（与
`sample_weight_support` 同模式），注册表条目同步镜像；避免能力信息继续在
注册表字段、类属性与签名检查三处分裂。

### 4.3 Pipeline 责任

Pipeline 统一负责：

- 输入维度、通道数和架构兼容性校验；
- encoder 的构造和注入（构造语义见 §4.1）；
- 先验、设备、随机种子和训练预算传递；
- CV、模型选择与报告；
- 对不支持的架构快速失败。

保存加载沿用现状分工：Pipeline 产出报告与最终模型对象，模型落盘与加载由
CLI 负责（pickle 契约）；本方案只要求保存加载契约稳定可测，不迁移职责。

算法自身只负责 PU 目标、算法 head 和特殊训练循环。

## 5. 渐进式实施阶段

### 阶段 0：能力契约和文档，不改变默认行为

- 在 Registry 增加架构能力字段；
- 统一输入和 encoder 输出校验；
- 补充 CLI/API/报告中的能力说明；
- 保留 `architecture="mlp"` 默认值和已有构造参数；
- 增加新算法接入模板。

验收：现有表格流程、WConPU/InfoMax MLP/CNN 流程和旧报告均不回归。

### 阶段 1：整理现有双架构实现

- 明确 `vision.py` 的通用 encoder factory 职责，补 `build_encoder`
  公共导出；
- 兼容保留现有 WConPU backbone 函数；
- 报告 provenance 增加 architecture/backbone/device/encoder 字段，
  分别报告 `native_mlp` 与 `native_cnn`；
- UI 的 CNN 算法候选集改由注册表能力元数据驱动（移除硬编码集合）；
- 补齐测试缺口：CV fold 训练隔离测试（fold 间权重不泄漏）、
  `build_encoder` 公共导出契约测试；其余项（CNN13/ResNet 形状、
  seed 确定性、pickle/save-load round-trip、fail-fast）已有覆盖，
  CPU/GPU 执行级测试在 CUDA 可用时补充。

### 阶段 2：以 `nnpu` 为首个试点

- 增加可选 `encoder=None`；
- `None` 继续使用原有 MLP；
- 传入 encoder 时只替换表征网络，不改变 PU loss 数学目标；
- 注意 nnpu 现有 `model` 参数语义为完整网络（输出原始 score），新增
  `encoder` 后需定义 encoder→head 的组合结构及与 `model` 的共存/优先级；
- 完成二维/四维输入、CV、seed、设备和保存加载回归测试。

`dist_pu` 在试点稳定后再评估，避免同时改造多个训练循环。

### 阶段 3：逐个评估复杂深度算法

分别处理 `self_pu`、`dgpu` 等使用完整 `backbone/model` 或特殊状态的算法。
不得仅通过参数重命名把完整模型伪装成 encoder。self_pu 的元学习再加权依赖
干净验证标签（真值 y），encoder 适配不解决该前置条件；dgpu 依赖外部
generator 协议，与 encoder 注入无法直接映射。二者建议长期搁置，待有明确
实验需求再单独立项。

### 阶段 4：按实际需求增加传统算法图像适配

仅在实验或用户需求明确时实现：

```text
固定/折内训练的 CNN encoder → 传统 PU 算法
```

该路径单独调参、单独报告为 `cnn_feature_adapter`，不替换
`tabular_native` 结果，也不默认进入原生算法主榜单。

## 6. 兼容性策略

必须保持：

- `architecture="mlp"` 为默认值；
- 现有二维输入行为和传统算法构造函数不变；
- 旧调参结果保留为 `tabular_native` 基准；
- 新增参数不覆盖已有参数语义；
- 旧报告字段和模型文件尽量可解析/加载；
- CNN 模型保存输入通道、backbone、归一化、encoder 权重和算法参数；
- CPU/GPU 加载支持设备映射；
- 同一 `random_state` 的初始化和训练满足可复现要求。

架构升级后，传统算法的既有调优结果不被替代：新适配路径必须重新调参，旧结果
作为回归测试和比较基线继续保留。

## 7. 当前实验的公平性协议

实验设计包含 CIFAR10（cnn13，可扩展至现有 resnet18/resnet50；ResNet-34
需先新增 backbone 支持）、USPS（MLP）、OS 数据生成、TS-OS 校准、正类比例
`{0.1, 0.3, 0.5}`、PA/PAUC/OA 模型选择以及全监督 oracle。其中 PA/PAUC/OA
模型选择与 TS-OS 校准当前不在工具箱实现中，属于本协议的前置基础设施，
需单独立项，不计入双架构改造范围；正类比例集合与既有深度协议
（pi=0.3/0.5/0.7）不一致，实施前需明确补充运行或对齐口径。

建议分为三条轨道：

### 轨道 A：原生算法比较

- 深度算法使用已验证的原生 MLP/CNN；
- 传统算法保持二维输入；
- 比较算法目标本身的性能和稳定性。

### 轨道 B：架构比较

- 只比较声明支持 MLP/CNN 的同一算法；
- 固定数据划分、seed、模型选择和调参预算；
- 分别报告 `native_mlp` 和 `native_cnn`。

### 轨道 C：CNN 特征适配

- 传统算法通过 CNN 特征处理图片；
- encoder 和预处理必须在训练 fold 内拟合；
- 单独调参和报告，不与原生实现混合排名。

所有轨道统一记录 Accuracy、AUC、F1、Precision、Recall、标准差（主报告已
具备）；训练时间、失败率和有效 seed 数为待补齐字段（时间与 seed 数已存在
于 benchmark trials，失败率需增加 trial 级异常捕获）。需要 class prior 的
算法同时报告真实先验和估计先验结果（该并列目前仅 benchmark trials 具备，
主报告待补齐）。

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
- `adapter_architectures`（如有）；
- 是否接收和训练 encoder；
- 是否支持 GPU、稀疏输入和 sample weight。

若声明支持 CNN，至少提供：

- CNN smoke training；
- 输入和输出形状测试；
- CV fold 隔离测试；
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
  + 严格 CV、seed、预算和报告约束
```

这是一项面向多架构扩展的渐进式架构改进，不是一次性重构。它吸收 PUBench 的
共享 featurizer 思路、PU-Bench 的 backbone factory/配置分层和 LIBSVM 的参数与
模型契约，同时保留当前工具箱的 sklearn/Pipeline 兼容性。

