# PU Learning Toolbox 论文源码资源统计

> 来源：`positive_unlabeled_learning.xlsx` 中列出的 15 篇 PU Learning 相关论文。  
> 更新时间：2026-07-08。  
> 目的：统计论文作者/作者团队是否公开了对应源码，并为 toolbox 的算法集成策略提供依据。

---

## 1. 统计口径

本文件只统计与论文算法/模型直接相关的公开代码资源。源码状态（`source_status`）采用以下分类：

| 状态 | 含义 | Toolbox 使用原则 |
|---|---|---|
| `official_exact` | 论文作者、作者组织、论文页面或作者主页明确给出的对应论文源码 | 优先作为实现依据；优先 adapter/wrapper 集成；必要时迁移到统一后端 |
| `official_bundle` | 作者或同研究线维护的工具包中包含该方法实现，但不是该论文单独发布的仓库 | 可作为官方级参考实现；优先复用接口和测试逻辑 |
| `official_related` | 作者或同团队给出了相关实现，但不能确认是该论文精确源码 | 可作为参考；核心实现建议 clean-room reimplementation |
| `third_party_only` | 只发现第三方实现，未能确认作者/官方关系 | 不作为默认实现来源；可作为接口和测试参考 |
| `not_found` | 未发现公开作者源码或可信对应仓库 | 按论文公式和伪代码重新实现 |
| `unknown` | 源码状态尚未调查或暂时无法确定 | 待后续版本确认后更新为上述具体状态 |

> 注意：`official_exact` 不等于可以直接复制代码进入 toolbox。仍需检查 license、依赖、框架版本、数据处理方式和再分发限制。

算法实现状态（`implementation_status`）采用以下分类：

| 状态 | 含义 |
|---|---|
| `api_only` | 只有 API 契约和占位类，尚未实现训练逻辑 |
| `native` | Toolbox 内部 clean-room 实现 |
| `official_adapter` | 通过 adapter 调用作者/官方源码 |
| `official_aligned_native` | 参考官方训练逻辑后完成原生实现，并有对齐测试 |
| `third_party_reference_only` | 只把第三方实现作为参考，不作为核心依赖 |
| `experimental` | 研究版，API 可能变化 |

许可证字段（`license`）允许以下值：

| 值 | 含义 |
|---|---|
| `"MIT"` / `"BSD"` / `"Apache"` / `"GPL"` 等 | 已知的 OSI 许可证标识符 |
| `"unknown"` | 尚未调查或无法确定许可证 |
| `"needs_review"` | 已获取源码但许可证需要人工确认 |
| `"proprietary"` | 明确为专有/非开放许可证 |
| `None` | 不存在源码（如 `not_found` 方法）或尚未填写 |

---

## 2. 总体统计

| 类别 | 数量 | 说明 |
|---|---:|---|
| 明确作者/官方对应源码：`official_exact` | 8 | 可作为优先适配对象 |
| 官方工具包或作者相关实现：`official_bundle` / `official_related` | 3 | 可作为实现参考或可选 backend |
| 仅第三方实现：`third_party_only` | 1 | 不计入作者源码 |
| 未发现作者源码：`not_found` | 3 | 建议 clean-room reimplementation |
| 合计 | 15 | 与上传表格一致 |

---

## 3. Toolbox 源码使用原则

对于某一具体模型或算法，如果作者提供了源码，toolbox 应优先使用作者源码作为实现依据。推荐优先级如下：

```text
official_exact 作者源码
    ↓
official_bundle 官方/作者团队工具包实现
    ↓
official_related 作者相关实现
    ↓
third_party_only 第三方实现，仅参考
    ↓
not_found 根据论文 clean-room 重新实现
```

工程上区分”源码依据”和”最终实现形式”，完整设计原则见 [`architecture.md`](architecture.md) §10。核心要点：优先 adapter（不急于重写）、旧框架做隔离层、许可证不清不进发行包、作者源码作为 correctness oracle。

推荐的算法注册字段：

```yaml
method_name: nnPU
paper: Positive-Unlabeled Learning with Non-Negative Risk Estimator
source_status: official_exact
upstream_url: https://github.com/kiryor/nnPUlearning
license: unknown_or_to_verify
integration_mode: adapter_then_native_reimplementation
validation_target: reproduce_author_default_experiments
```

---

## 4. 论文源码清单

| # | 类型 | 论文 | 源码状态 | 作者/官方源码或相关 URL | Toolbox 处理建议 |
|---:|---|---|---|---|---|
| 1 | class prior estimation | Class-Prior Estimation for Learning from Positive and Unlabeled Data | `official_related` |  http://www.mcduplessis.com/index.php/software/| 作者主页提供了相关 class-prior estimation MATLAB 代码，但更接近早期 IEICE 版本；toolbox 中建议 clean-room 实现 pen-L1 / PE，并将作者代码作为数值参考 |
| 2 | class prior estimation | Rethinking Class-Prior Estimation for Positive-Unlabeled Learning | `official_exact` | https://github.com/a5507203/Rethinking-Class-Prior-Estimation-for-Positive-Unlabeled-Learning | 优先适配 ReCPE；设计为 CPE wrapper，可接入 KM、PE、TIcE、AlphaMax 等 base estimator |
| 3 | risk-consistent / calibration | Learning Classifiers from Only Positive and Unlabeled Data | `third_party_only` | https://github.com/aldro61/pu-learning；https://github.com/pulearn/pulearn | 未发现 Elkan/Noto 作者源码；第三方实现只作接口参考。核心包中建议 clean-room 实现 Elkan-Noto calibration |
| 4 | risk-consistent loss function | Convex Formulation for Learning from Positive and Unlabeled Data | `official_bundle` | https://github.com/t-sakai-kure/pywsl；https://github.com/kiryor/nnPUlearning | `pywsl` 中包含 uPU；`nnPUlearning` 也包含 uPU/nnPU 相关实现。建议先通过 `pywsl` 或 nnPU 代码验证公式，再实现统一 PyTorch / sklearn 风格版本 |
| 5 | risk-consistent loss function | Positive-Unlabeled Learning with Non-Negative Risk Estimator | `official_exact` | https://github.com/kiryor/nnPUlearning；https://github.com/t-sakai-kure/pywsl | 优先适配作者 Chainer 代码；同时参考 `pywsl` 的 Python 包装。核心目标是保留 nnPU 的非负风险修正和默认训练策略 |
| 6 | risk-consistent loss function | Semi-supervised Classification Based on Classification from Positive and Unlabeled Data | `official_exact` | https://t-sakai-kure.github.io/software.html；https://github.com/t-sakai-kure/pywsl | 优先使用 `pywsl` 中 PNU 实现作为官方工具包来源；将 PNU 风险组合抽象为可配置 risk-composer |
| 7 | risk-consistent loss function | Loss Decomposition and Centroid Estimation for Positive and Unlabeled Learning | `official_related` | https://gcatnjust.github.io/ChenGong/code/CEGE_PAMI20.rar | 未发现该 TPAMI 论文精确源码；CEGE 是同作者 centroid estimation 相关代码。建议 clean-room 实现 LDCE/KLDCE，并用 CEGE 代码理解 centroid 子模块 |
| 8 | risk-consistent / large-margin | Large-Margin Label-Calibrated Support Vector Machines for Positive and Unlabeled Learning | `official_exact` | https://gcatnjust.github.io/ChenGong/code/LLSVM_TNNLS19.rar | 作者主页给出代码包。建议作为 legacy adapter 集成；核心包中暴露 sklearn-style `LLSVMClassifier` |
| 9 | label-distribution alignment | Dist-PU: Positive-Unlabeled Learning from a Label Distribution Perspective | `official_exact` | https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective | 优先适配 PyTorch 代码；保留 label distribution consistency、entropy minimization、Mixup 和 class prior 配置 |
| 10 | bias-aware PU | Learning from Positive and Unlabeled Data with a Selection Bias | `official_exact` | https://github.com/MasaKat0/PUlearning | 优先适配 `BiasedPUlearning` 目录；作为 selection-biased PU / nnPUSB 基线 |
| 11 | bias-aware PU | Instance-Dependent Positive and Unlabeled Learning with Labeling Bias Estimation | `official_exact` | https://gcatnjust.github.io/ChenGong/code/LBE_TPAMI21.rar | 作者主页给出代码包。建议抽象为 `PropensityEstimator` / `LabelingBiasEstimator`，并与风险估计器解耦 |
| 12 | deep / self-training | Self-PU: Self Boosted and Calibrated Positive-Unlabeled Training | `official_exact` | https://github.com/VITA-Group/Self-PU；论文中历史链接：https://github.com/TAMU-VITA/Self-PU | 优先适配 PyTorch 代码；拆分 self-paced sample mining、self-calibrated loss、self-distillation 三个组件 |
| 13 | deep / representation | Information-Theoretic Representation Learning for Positive-Unlabeled Classification | `not_found` | - | 未发现作者源码。建议实现为 `PURepresentationLearner` 或 `TransformerMixin`，用于高维数据的 CPE 前处理 |
| 14 | deep / contrastive representation | Weighted Contrastive Learning with Hard Negative Mining for Positive and Unlabeled Learning | `not_found` | - | 未发现作者源码。建议作为 research-stage 模块；先实现 weighted contrastive loss 和 hard negative mining，再进入稳定 API |
| 15 | deep / generative-discriminative hybrid | Discriminative-Generative Positive and Unlabeled Learning | `not_found` | - | 未发现作者源码。生成式成本高，建议放入 post-MVP；优先完成 Dist-PU / Self-PU 后再实现 |

---

## 5. 推荐集成优先级

算法实施优先级以 [`development_roadmap.md`](development_roadmap.md) §10 为权威来源。本文件仅从**源码可用性**角度给出各方法的推荐集成方式：

| 方法 | 源码状态 | 推荐集成方式 |
|---|---|---|
| uPU / nnPU | `official_exact` / `official_bundle` | `pywsl` / `nnPUlearning` adapter + native PyTorch loss |
| PNU | `official_exact` | `pywsl` adapter + risk-composer |
| ReCPE | `official_exact` | wrapper + base CPE interface |
| PUSB | `official_exact` | official repo adapter |
| Self-PU | `official_exact` | PyTorch adapter，逐步拆分组件 |
| Dist-PU | `official_exact` | PyTorch adapter，保留增强策略 |
| LBE | `official_exact` | legacy adapter + clean-room estimator |
| LLSVM | `official_exact` | legacy adapter + sklearn-style wrapper |
| LDCE/KLDCE | `official_related` | clean-room 实现 + 相关代码参考 |
| InfoMax PU / WConPU / DGPU | `not_found` | post-MVP，method card 先行 |

---

## 6. 对 toolbox 架构的直接影响

建议在 toolbox 中增加一个 `SourcePolicy` 或等价元数据层：

```python
@dataclass
class SourcePolicy:
    source_status: Literal[
        "official_exact",
        "official_bundle",
        "official_related",
        "third_party_only",
        "not_found",
        "unknown",
    ]
    upstream_url: str | None
    license: str | None
    integration_mode: Literal[
        "adapter",
        "native_reimplementation",
        "adapter_then_native_reimplementation",
        "reference_only",
    ]
    validation_target: str | None
```

每个算法注册时同时注册源码来源，例如：

```python
register_method(
    name="dist_pu",
    estimator_cls=DistPUClassifier,
    source_policy=SourcePolicy(
        source_status="official_exact",
        upstream_url="https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective",
        license=None,
        integration_mode="adapter_then_native_reimplementation",
        validation_target="match official CIFAR/MNIST protocol where feasible",
    ),
)
```

这样可以让 toolbox 在还没有完全实现所有论文算法时，先完成统一框架、接口、注册表、测试协议和外部 adapter 机制；之后再逐个把算法替换为 native implementation。

---

## 7. 需要人工复核的事项

1. **许可证**：GitHub 项目中标注 MIT 的可优先复用；`.rar` 或作者主页压缩包若无 license，默认只作参考，不直接再分发。
2. **仓库活跃度**：旧仓库可能依赖过时 Python、Chainer、MATLAB 或 CUDA 版本，需容器化复现。
3. **数据协议**：深度方法通常把数据预处理、PU split、class prior 设置写在训练脚本里，集成时必须抽象为统一 dataset protocol。
4. **作者身份**：若仓库没有从论文页面直接链接，但能从作者主页或作者账号确认，则可标记为 `official_exact`；否则降级为 `official_related` 或 `third_party_only`。
5. **测试基线**：所有官方源码 adapter 都应至少提供 smoke test、loss-level test 和小规模复现实验脚本。

---

## 8. 简短结论

当前 15 篇论文中，已有相当一部分可以通过作者源码或官方工具包加速集成。toolbox 的实现策略不应等待所有论文算法全部重写完成，而应先搭建稳定的统一框架：`dataset protocol + class-prior estimator + risk estimator + estimator API + source policy registry + adapter layer`。在此基础上，优先把 `official_exact` 方法作为 adapter 接入，再逐步做 native reimplementation。
