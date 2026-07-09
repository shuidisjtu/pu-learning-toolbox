# PU Learning Toolbox

## 1. 项目定位

本项目是一个面向 **Positive-Unlabeled Learning（PU Learning）** 的 Python Toolbox，目标是在只有“已标记正样本 + 无标记样本”的条件下，帮助用户完成数据校验、类先验估计、模型训练、预测、评估、算法选择、诊断与论文方法复现。

Toolbox 的核心目标是：

1. **易用**：接口尽量兼容 `scikit-learn`，让非 PU Learning 专家也能快速训练基线模型；
2. **可靠**：内置 PU 数据校验、类先验估计、标记倾向建模、PU 专用切分、指标与诊断工具；
3. **可扩展**：先搭建稳定框架，再逐步集成经典 PU、风险估计 PU、SAR / Instance-Dependent PU、深度 PU、表征学习 PU 与生成式 PU；
4. **可复现**：提供统一配置、随机种子控制、benchmark、paper-like examples 与测试基准；
5. **尊重论文实现**：如果某一论文方法存在作者/官方源码，Toolbox 应优先使用作者源码对应的训练逻辑、损失函数、默认超参数与数据处理流程；没有明确源码时再进行 clean-room reimplementation；
6. **有差异化**：重点支持 selection bias、labeling bias、SAR / Instance-Dependent PU Learning，不只停留在 SCAR 假设下的传统算法。

---

## 2. 基本问题设定

PU Learning 中，训练数据只包含：

- `P`：已标记为正类的样本；
- `U`：无标记样本，其中混合了真实正类和真实负类。

统一记号如下：

| 符号 | 含义 |
|---|---|
| `x` | 样本特征 |
| `y ∈ {0,1}` | 真实类别标签，训练时通常不可完全观测 |
| `s ∈ {0,1}` | 是否被标记为正类 |
| `π = P(y=1)` | 类先验，即总体中真实正类比例 |
| `c = P(s=1 \| y=1)` | SCAR 假设下正类被标记的常数概率 |
| `c(x) = P(s=1 \| y=1, x)` | SAR / Instance-Dependent 假设下实例相关的标记倾向 |

---

## 3. 支持的数据场景

| 场景 | 含义 | 代表方法 |
|---|---|---|
| Single-training-set | 从总体中采样一批数据，再对部分正样本打标 | Elkan-Noto、PU Bagging、Spy、Rocchio |
| Case-control | 正样本集合与无标记集合分别采样 | uPU、nnPU、PNU、Dist-PU |
| Selection-biased PU | 已标记正样本不代表全部正样本 | PUSB、LBE、propensity-aware PU |
| Representation-heavy PU | 图像、文本、高维表征学习主导 | Self-PU、InfoMax PU、Weighted Contrastive PU、DGPU |

| 假设 | 含义 | 代表算法 |
|---|---|---|
| SCAR | 正样本是否被标记与特征 `x` 无关 | Elkan-Noto、uPU、nnPU、PU Bagging |
| SAR / Instance-Dependent | 正样本是否被标记依赖特征，即 `c(x)` 不同 | PUSB、LBE、Centroid Estimation、LLSVM |
| Unknown | 无法确认 SCAR 是否成立 | 先跑稳健 baseline，再做敏感性分析和 SAR 对照实验 |

---

## 4. 论文方法覆盖范围

Toolbox 覆盖 15 篇 PU Learning 论文，按方向分为 8 个算法族。完整的算法族总览、按场景/假设选型指南见 [`method_selection.md`](method_selection.md) §2–§5，论文方法到代码模块的映射见 [`architecture.md`](architecture.md) §9。

---

## 5. 分层依赖设计

| 层级 | 依赖 | 内容 |
|---|---|---|
| Core | `numpy`、`scipy`、`pandas`、`scikit-learn` | 标签处理、数据校验、经典算法、类先验估计、指标、交叉验证、算法注册表 |
| Torch Extension | `torch` | uPU、nnPU、Dist-PU、Self-PU、深度表征学习方法 |
| Research Extension | `torchvision`、`lightning`、`tqdm` 等 | 图像 benchmark、生成式模型、复杂论文复现 |
| Optional Extension | `numba`、`joblib`、`matplotlib`、`networkx` 等 | 加速、并行、可视化、图 PU、诊断报告 |
| External Source Adapters | 视作者源码而定 | 包装作者源码、旧框架代码、MATLAB/RAR 代码包或独立仓库 |

建议安装方式：

```bash
pip install pu-toolbox
pip install pu-toolbox[torch]
pip install pu-toolbox[research]
pip install pu-toolbox[benchmark]
pip install pu-toolbox[all]
```

---

## 6. 基础使用示例

```python
from pu_toolbox.registry import get_algorithm

Model = get_algorithm("elkan_noto")

model = Model()
model.fit(X_train, y_pu_train)

scores = model.decision_function(X_test)
proba = model.predict_proba(X_test)
pred = model.predict(X_test)
```

如果用户不知道该选择哪个算法，可以使用推荐器：

```python
from pu_toolbox.advisor import recommend_algorithms

recommendations = recommend_algorithms(
    X,
    y_pu,
    scenario="unknown",
    assumption="unknown",
    class_prior=None,
    hardware="cpu",
    prefer_official_source=True
)
```

---

## 7. 核心能力清单

### 7.1 数据处理

- 支持 `numpy.ndarray`、`pandas.DataFrame`、`scipy.sparse`；
- Torch 扩展支持 `torch.Tensor`；
- 自动规范化 PU 标签；
- 检查正样本、无标记样本、P/U 比例与 `class_prior` 合法性；
- 标记 SCAR、SAR、case-control、selection bias 等假设信息；
- 支持生成已知 `π`、`c`、`c(x)` 的 synthetic benchmark。

### 7.2 算法

MVP 版本优先支持框架和稳定基线：

- `ElkanNotoPUClassifier`
- `BaggingPUClassifier`
- `WeightedLogisticPUClassifier`
- `BiasedSVMClassifier`
- `TIcEEstimator`
- `AlphaMaxEstimator`
- `PUStratifiedKFold`
- `PUStratifiedShuffleSplit`
- `RuleBasedMethodAdvisor`

后续版本按论文和源码优先级扩展：

- `ReCPEEstimator`
- `UPULoss`
- `NNPULoss`
- `PNUClassifier`
- `CentroidPUClassifier`
- `LLSVMClassifier`
- `DistPUClassifier`
- `PUSBClassifier`
- `LBEClassifier`
- `SelfPUClassifier`
- `InfoMaxPURepresentation`
- `WeightedContrastivePU`
- `DGPUClassifier`

### 7.3 源码集成策略

源码集成策略（adapter 优先、许可证检查、clean-room 兜底）详见 [`architecture.md`](architecture.md) §10 源码 adapter 设计，各论文官方源码的可用性统计见 [`resources_optimized.md`](resources_optimized.md)。

---

## 8. 项目文档结构

| 文件 | 作用 |
|---|---|
| `README.md` | 项目总览、目标、安装方式、基础使用、论文方法范围 |
| `method_selection.md` | 方法分类、算法选择逻辑、推荐器设计 |
| `development_roadmap.md` | 开发阶段、任务拆分、版本规划、源码集成策略 |
| `architecture.md` | 包结构、数据流、API、模块边界、adapter 设计 |
| `resources_optimized.md` | 论文源码状态、URL、官方实现统计与适配优先级 |

---

## 9. 最终设计原则

本项目允许并且建议采用 **framework-first** 开发方式：

```text
先定义稳定、可测试、sklearn-compatible 的框架与 API 契约
        ↓
用轻量 baseline 和 mock estimator 跑通 registry、advisor、metrics、benchmark
        ↓
为论文算法预留 metadata、adapter、测试接口和复现目录
        ↓
优先接入有作者源码的算法，做 adapter 与复现对齐
        ↓
对无源码算法进行 clean-room reimplementation
        ↓
逐步扩展 SAR、深度表征、生成式和高级研究方向
```
