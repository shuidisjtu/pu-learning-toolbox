# PU 研究能力整合：语义基线

## 1. 范围与证据

调查日期为 2026-11-06，实施起点为合并后 `main` 的 `68555f9e`。
本页记录 P0/P1 语义基线及 P2 采样与先验工作包：当前行为、公共知识、来源和实施验证。
原始方案位于工作区临时材料，正式仓库以本页、[ADR-0021](../adr/0021-pu-research-semantics.md)
和[进度清单](process_checklist.md)保留可维护的依据。

证据分为论文定义、固定参考实现和工具箱当前代码，分别登记公式、实现与行为。

| 材料 | 固定来源 | 核对位置（PDF 文件页序） |
|---|---|---|
| R1 Bekker & Davis (2020) | [DOI](https://doi.org/10.1007/s10994-020-05877-5) | p4–6 §2.3–2.5、式(5)–(7)，p7/p9 Definition 1/2 |
| R2 Mielniczuk & Wawrzeńczyk (2024) | [arXiv v3](https://arxiv.org/abs/2312.02095v3) | p2–3 §1、式(2)，p6–8 §2.1、式(16)–(19) |
| R3 Wang et al. (2026) | [arXiv v2](https://arxiv.org/abs/2509.24228v2) | p4–5 §3、式(3)/(5)/(6)，p6–7 §4，p15 Appendix B.1 |
| R4 Chen et al. (2026) | [OpenReview](https://openreview.net/forum?id=tb8DabMbMq) | p21 C.2，p22 C.3，p23–24 D.1 |
| PUBench（R3） | [a9a62b05b0f222c72aff4df8992307376c78d682](https://github.com/wu-dd/PUBench/tree/a9a62b05b0f222c72aff4df8992307376c78d682) | datasets / misc / model_selection |
| PU-Bench（R4） | [2d95a19eefd72e66ff30128ec2f65e1d1d4cc077](https://github.com/XiXiphus/PU-Bench/tree/2d95a19eefd72e66ff30128ec2f65e1d1d4cc077) | data_utils / metrics / model_factory |

公共定义及条件分别位于[采样](../user/concepts/pu_sampling.md)、
[SCAR/SAR](../user/concepts/scar_sar.md)、[先验](../user/concepts/class_prior_semantics.md)、
[选模](../user/concepts/pu_model_selection.md)和[表征](../user/concepts/representations.md)。

## 2. 生成与抽样事实

| 入口 | 当前行为 | 后续边界 |
|---|---|---|
| OS SCAR 固定比例配额 / `SCARGenerator` | 同池、正类均匀无放回；`max(1, round(c*n_pos))`，不超过正类数；无正例返回全 U | 共享 `preprocessing/_sampling.py` 计数与抽样，seed/标签/后续随机流一致；R4 的 floor 配额另外登记 |
| OS SCAR 显式定额 | 同池选定 n_labeled 个正例，其余原位 U，支持全部为正类的池 | `construct_pu_dataset` 校验正整数及可用正例数 |
| `make_case_control_labels` | 同池定额标记的旧名字，延续双类别输入校验与随机轨迹 | 历史上有兼容包装；现已移除旧名字 |
| 旧标签分派 | 历史上有同池定额分派 | 现由 `PULabelingConfig` 显式声明，不改历史采样身份 |
| `make_gaussian_pu_data` | 先生成 n_p 个正类并全部可见；再生成 n_u 个负类作为 U；返回拼接表正类比例 | S2 已固定旧场景与 seed 回归，docstring 明确分布与返回含义 |
| `make_gaussian_case_control_data` | P 从正类条件分布抽取，U 独立从总体混合分布抽取；返回真值及来源/先验元数据 | S1/S2 已实现；总体参数与经验先验分别记录 |
| OS Bernoulli / `make_sar_propensity` | 正类均值校准；可显式请求零标记时补标 | 构造记录区分原本的倾向、实际预算与补标 |
| `SARLBEAGenerator` | 默认 k=10，权重 `0.9*p^k/sum(p^k)+0.1/n_pos`，固定配额无放回 | 保留参考实现身份；论文无平滑版本为独立配置 |
| `SARLBEBGenerator` | `clip(1.5+shrink_coef-p,0)^k`，默认 shrink_coef=1、k=10 | 论文 `(1-p)^10` 与当前实现分开 |

实现：[pu_labeling.py](../../pu_toolbox/preprocessing/pu_labeling.py)、
[selection_bias.py](../../pu_toolbox/preprocessing/selection_bias.py)、
[strategies.py](../../pu_toolbox/experiment/strategies.py)。
LBE 辅助 LogisticRegression 在调用所在分区的真实标签上拟合；归一化权重不是最终入选概率。
参考库 PUSB 接受/拒绝、R4 的 top-N 及有限池 CC 的差异见公共概念页的来源对照。

S1/S2 新入口由 preprocessing 门面导出；dispatcher 增加 `fixed_count`。
[示例 15](../../examples/minimal/15_pu_sampling.py)和[采样教程](../user/howto/pu_sampling.md)
使用新入口展示 OS/TS。旧入口的调用保留在兼容回归与本页历史基线中。
既有标签测试覆盖计数、seed、形状；新增的
[采样语义回归](../../tests/unit/preprocessing/test_sampling_semantics.py)覆盖独立 TS、混合 U、
目标/经验先验、来源身份、参数域及旧别名迁移。

## 3. 先验估计与传递

| 对象 | 实际估计池/输出 | 所需审查与消费者 |
|---|---|---|
| KM1 / KM2 | `X[y_pu==0]` 为 F，`X[y_pu==1]` 为 H；返回 F 中 H 的 mixture proportion κ | 代表性与可辨识条件成立时解释为 U 正类比例；需要总体先验时按输入设计换算 |
| penL1（`ClassPriorEstimator`） | P、原生 U 的 RBF 均值，优化 `theta*mean(phi_P)-mean(phi_U)` | 输出 mixture 目标 θ；OS 经质量换算，TS 目标与总体相同 |
| `ReCPEEstimator` | ranker 取高分 `max(1,ceil(copy_fraction*n_U))` 行复制进 P；默认 KM2 消费重组 P 与完整 U | base 值经 `[0,1]` clip 后作为原生 U 先验近似返回；元数据记录辅助目标与复制政策 |
| 独立 `BBEMixtureEstimator` | 观测 P/U 分别切 train/holdout；holdout U 输出 `alpha_U` | 要求 `input_sampling="ts"`；使用独立的 `estimate_unlabeled_prior()` 接口 |

实现：[kernel_mean.py](../../pu_toolbox/prior/kernel_mean.py)、
[pen_l1.py](../../pu_toolbox/prior/pen_l1.py)、[recpe.py](../../pu_toolbox/prior/recpe.py)、
[bbe.py](../../pu_toolbox/prior/bbe.py)。
mixture 估计按各方法的代表性/可辨识条件解释。

先验估计结果先按 `estimate_target` 和 `estimation_pool` 解释，再决定是否转换。对于 OS，记
训练分区中已标记正例占比为 `q`，则总体先验等于 `q + (1-q) * pi_U`；独立 TS 下原生 U
本身来自总体。质量换算由 `prior.semantics` 提供，不检查方法是否满足代表性或可辨识条件。

直接拟合时，按具体估计器和实际风险视图传参。`prior_consumers` 分用途声明默认目标及可选 TS-compatible 目标，
例如 uPU 默认消费原生 U 的先验；不能把该目标当作所有视图和参数的统一规则。PUSBKernel
分别消费分位阈值的 `training_table_prior` 和风险的 `risk_prior`。验证与诊断另按其目标总体
选取先验：OS 的总体评估池是完整输入，独立 TS 的总体评估池是原生 U。实验应记录输入设计、
原始估计值、估计池、换算和每个消费者的实际值。

## 4. 历史 os_or_ts 的逐实现清单

以下为所有 `fit` 签名含 `os_or_ts` 的 estimator（AST 全量扫描，共 19 个）。
它们表达风险角色；输入采样设计和先验消费目标分别记录。

| 方法 | 默认 | 当前 ts 行为与先验消费 | 实现 |
|---|---|---|---|
| uPU | os | 完整池替换 U 风险；`risk_prior` 不变，RBF 中心池随之变化 | [upu.py](../../pu_toolbox/estimators/risk/upu.py) |
| nnPU | os | mini-batch 补回 P；`risk_prior` 不变；拒绝与 sample_weight 同用 | [nnpu.py](../../pu_toolbox/estimators/risk/nnpu.py) |
| VPU | ts | 使用完整风险池；训练不使用先验 | [vpu.py](../../pu_toolbox/estimators/risk/vpu.py) |
| CVIR | os | input_sampling 与风险视图分别声明；负例筛选和加权目标使用实际风险池的 unlabeled_positive_prior | [cvir.py](../../pu_toolbox/estimators/risk/cvir.py) |
| GradPU | os | 训练视图适配；U 承担负例角色，采用无先验目标 | [grad_pu.py](../../pu_toolbox/estimators/deep/grad_pu.py) |
| RobustPU | os | 仅 warm-up 适配，后续 episode 仍取 native U；`risk_prior` 指 warm-up U 风险的正类比例 | [robust_pu.py](../../pu_toolbox/estimators/deep/robust_pu.py) |
| GenPU | os | 支持风险适配并使用先验；生成/分类组件需分别解释 | [gen_pu.py](../../pu_toolbox/estimators/deep/gen_pu.py) |
| SplitPU | os | 仅 teacher 的 nnPU marginal 补回 P；先验用于 teacher 风险；easy/hard 仍取原生 U | [split_pu.py](../../pu_toolbox/estimators/deep/split_pu.py) |
| PULDA | os | 分布与 two-way margin 风险池补回 P；先验用于二者目标，伪标签空间不变 | [pulda.py](../../pu_toolbox/estimators/risk/pulda.py) |
| PUET | os | P 副本作为 U-risk 角色行；先验用于 P 质量权重，不换算数值 | [puet.py](../../pu_toolbox/estimators/risk/puet.py) |
| DistPU | os | expectation / histogram 对齐池补回 P；先验是被对齐分布的目标比例 | [dist_pu.py](../../pu_toolbox/estimators/risk/dist_pu.py) |
| SelfPU | os | 负 PU 项补回 P，先验用于风险修正；可信集仍基于原生 U，clean support 单独记录 | [self_pu.py](../../pu_toolbox/estimators/deep/self_pu.py) |
| PUSBKernel | os | 适配拟合风险池，先验同时用于风险和分位预测；中心池/CV/阈值池不变 | [pusb_kernel.py](../../pu_toolbox/estimators/bias_aware/pusb_kernel.py) |
| RankPruning | os | 原生 OS；由折外概率估计噪声率 | [rank_pruning.py](../../pu_toolbox/estimators/classic/rank_pruning.py) |
| HolisticPU | os | 拒绝 ts；训练不使用先验 | [holistic_pu.py](../../pu_toolbox/estimators/deep/holistic_pu.py) |
| LBE-neural | os | 拒绝 ts；内部先验为自身推断值 | [lbe_neural.py](../../pu_toolbox/estimators/deep/lbe_neural.py) |
| P3Mix | os | 拒绝 ts；训练不使用先验 | [p3mix.py](../../pu_toolbox/estimators/deep/p3mix.py) |
| PAN | os | 拒绝 ts；训练不使用先验 | [pan.py](../../pu_toolbox/estimators/deep/pan.py) |
| PULNS | os | 拒绝 ts；训练不使用先验 | [pulns.py](../../pu_toolbox/estimators/deep/pulns.py) |

registry 的现行能力由实际类声明给出。

其余 12 个注册 classifier 如下，与前表合计 31 个（另有 2 个注册 prior 条目）。
它们没有 `os_or_ts`，因此没有可继承的视图默认值；具名采样参数单独列出。

| 注册名 | 标签语义 | 当前先验消费与缺省 | 实现 |
|---|---|---|---|
| pnu | pnu | 构造必需 pi，fit 可覆盖；pi/1-pi 加权 PN/PU/NU 风险，U 为边际项；默认 eta=0 退化 PN 风险但标签声明仍为 pnu | [pnu.py](../../pu_toolbox/estimators/risk/pnu.py) |
| llsvm | pu | registry 要求外部先验；直接 fit 可在缺省时内部 penL1 估计；`t=2*pi-1` 用于原生 U 校准损失，无目标换算 | [llsvm.py](../../pu_toolbox/estimators/classic/llsvm.py) |
| centroid_pu / kldce | pu | 消费可选 pi；缺省 `n_P/(n*(1-h))`，构造必填 flip_probability=h；公式对应 OS 完整输入总体，进入 centroid/优化常数，无 pool 参数 | [ldce.py](../../pu_toolbox/estimators/risk/ldce.py)、[kldce.py](../../pu_toolbox/estimators/risk/kldce.py) |
| weighted_contrastive_pu | pu | 构造必需 pi，fit 可覆盖；初始化 `[1-pi,pi]`、对齐原生 U 正类均值并加权正项，无目标换算 | [weighted_contrastive_pu.py](../../pu_toolbox/estimators/deep/weighted_contrastive_pu.py) |
| dgpu | pu | 构造必需 pi，fit 可覆盖；一个值用于原生 U 均值对齐、类别分布与生成样本配额，无 pool 参数 | [dgpu.py](../../pu_toolbox/estimators/deep/dgpu.py) |
| infomax_pu | pu | 可指定 pi，缺省在学得表征上用 penL1（可替换）；直接传内部 nnPU 默认 os 风险，无目标换算 | [infomax_pu.py](../../pu_toolbox/estimators/deep/infomax_pu.py) |
| tedn | pu | 不接受外部先验；构造必填 input_sampling=ts；内部 BBE 的 holdout alpha_U 用于负例筛选，有 prior_target，不做 OS 适配 | [tedn.py](../../pu_toolbox/estimators/deep/tedn.py) |
| elkan_noto | pu | 内部估计 c 后计算/裁剪 `n_L/(n*c)`，解释依赖 OS/SCAR；q 是 OS 总体的标记比例 | [elkan_noto.py](../../pu_toolbox/estimators/classic/elkan_noto.py) |
| lbe | pu | 可选 pi 用于原生 U 潜在后验初始化，缺省 `clip(2*mean(S),0.05,0.95)`；不固定训练后的先验 | [lbe.py](../../pu_toolbox/estimators/bias_aware/lbe.py) |
| pusb | pu | observed-P-vs-U LogisticRegression baseline；外部 pi 存储为属性，分类目标由观测 P/U 标签定义 | [pusb.py](../../pu_toolbox/estimators/bias_aware/pusb.py) |
| lagam | pu | 外部 pi 仅兼容校验，训练不使用；support_data 是额外监督角色 | [lagam.py](../../pu_toolbox/estimators/deep/lagam.py) |

`required_prior_parameters` 为空表示调用者可省略外部先验；
LDCE/KLDCE、InfoMax 和 LBE 分别展示派生、内部估计及初始化三类消费。
DGPU/WConPU 的 `unlabeled_prior` 对应原生 U；直接训练时应针对允许使用的训练分区换算。
当前没有注册 PN classifier；Survey 的私有 `PilotOracleMLP` 声明 pn，
以真实标签优化监督 BCE，无 PU prior 或风险视图参数，入口为
[survey_execution.py](../../pu_toolbox/experiment/survey_execution.py)。

[core.training_views](../../pu_toolbox/core/training_views.py)负责 positive/native_unlabeled/
loss_unlabeled 与来源，experiment 负责准入与 Survey 配方。
已有局部先验契约的参考是
[cvir_risk_prior](../../pu_toolbox/experiment/cvir_adapter.py)：OS 原生 U 用
`(pi*n-n_P)/(n-n_P)`，完整风险池用 pi，非法域拒绝而非 clip。

## 5. S1–S3 的最小复现

以下断言固定原有行为，作为后续兼容回归的基准。
可将各片段分别保存成临时脚本并使用 `uv run python` 执行。

### S1：同池构造的 U 先验改变

```python
import numpy as np
from pu_toolbox.preprocessing import PULabelingConfig, SourcePartition, construct_pu_dataset

truth = np.array([1] * 40 + [0] * 60)
features = np.arange(100, dtype=float)[:, None]
result = construct_pu_dataset(
    SourcePartition(features, truth, "example-v1", "train"),
    PULabelingConfig("scar", "fixed_count", random_state=7, n_labeled=20),
    positive_classes=(1,),
    negative_classes=(0,),
)
observed = result.observations
assert observed.labels.sum() == 20
assert result.manifest["audit"]["empirical_native_unlabeled_prior"] == 0.25
```

这个构造仍是 OS 固定配额。独立 TS 从指定总体取得混合 U，并独立取得额外 P。

### S2：旧 Gaussian 的 U 来源为负类

```python
import numpy as np
from pu_toolbox.preprocessing import make_gaussian_pu_data

X, y_pu, prior = make_gaussian_pu_data(
    n_p=30, n_u=70, n_features=2, separation=2.0, random_state=7,
)
rng = np.random.RandomState(7)
expected_p = rng.randn(30, 2) + 1.0
expected_negative_u = rng.randn(70, 2) - 1.0
assert np.array_equal(X[y_pu == 1], expected_p)
assert np.array_equal(X[y_pu == 0], expected_negative_u)
assert prior == 0.3
```

此处精确重建随机流用于确认旧生成器的类别来源；新混合 U 生成器单独验收。

### S3：按输入设计换算先验目标

```python
from math import isclose

from pu_toolbox.prior import (
    native_unlabeled_to_population,
    population_to_native_unlabeled,
)

q, pi_u = 0.2, 0.25
pi_population = native_unlabeled_to_population(pi_u, q, input_sampling="os")
assert isclose(pi_population, 0.4)
assert isclose(population_to_native_unlabeled(pi_population, q, input_sampling="os"), pi_u)
assert native_unlabeled_to_population(pi_u, q, input_sampling="ts") == pi_u
```

这是纯目标换算，不拟合估计器。若上述 OS 分区用于默认 uPU，应传入原生 U 先验 0.25；
总体评估需要总体值时才使用 0.4。独立 TS 的原生 U 先验与总体先验相同。
[示例 16](../../examples/minimal/16_prior_targets.py)演示如何将 OS 总体先验换成 uPU 的训练先验。

## 6. 分阶段退出条件

| 阶段 | 交付与退出条件 |
|---|---|
| P0 | 本语义基线、来源版本、S1–S3 复现与调用范围、最小 ADR |
| P1 | 公共采样/先验/选模/表征概念，SCAR/SAR 扩充，入口导航；片段执行和链接门禁 |
| P2 | S1/S2/S3 分别先有失败回归再修复；准确新接口、兼容迁移与先验来源记录；专项核对 ReCPE 的重组目标与原始目标换算 |
| P3 | 同义采样与先验换算单源化，registry 能力事实与具名政策分离；seed 与层间契约 |
| P4 | OS/TS、SAR、先验训练、独立 PA/OA、encoder/冻结特征及完整 DIY 六类 CPU 示例与教程 |
| P5 | 协议引用公共概念、报告设置摘要、方法卡/API/开发文档收口与适用回归 |

PAUC、新 SAR 预设和更多 backbone 为后续扩展。正式数据重训练和历史结果审核另行安排。
各阶段实际完成状态与测试结果见[进度清单](process_checklist.md)。

## 7. 本批验证记录（2026-11-06）

以下为 P0/P1 验证，覆盖公共知识与当时实现的对应关系。

| 验证 | 实际结果 |
|---|---|
| 采样/先验/风险/选模及层边界专项 | 215 个用例，通过，3 条 uPU 既有 intercept 警告 |
| 文档链接、API 覆盖、测试质量、Ruff 格式、目录、层依赖 | 全部通过；API 覆盖 86 个公开符号 |
| `git diff --check` | 通过 |

专项实际命令：

```powershell
uv run pytest tests/unit/preprocessing/test_pu_labeling.py tests/unit/preprocessing/test_selection_bias.py tests/unit/experiment/test_strategies_labeling.py tests/unit/experiment/test_strategies_selection.py tests/unit/experiment/test_training_views.py tests/unit/experiment/test_runner_class_prior.py tests/unit/prior/test_kernel_mean.py tests/unit/prior/test_recpe.py tests/unit/prior/test_pen_l1.py tests/unit/prior/test_bbe.py tests/contract/test_layer_boundaries.py -o addopts='--tb=short' -q
```

文档片段逐段在独立命名空间执行，临时核验脚本为
`tmp/verify_pu_research_foundations.py`；本页片段也可分别存为脚本后运行。
静态门禁实际命令：

```powershell
uv run python scripts/check_doc_links.py
uv run python scripts/check_api_docs.py
uv run python scripts/check_test_quality.py
uv run python scripts/check_format.py
uv run python scripts/generate_structure.py --update
uv run python scripts/generate_structure.py --check
uv run python scripts/generate_layer_deps.py --check
git diff --check
```

目录生成器首次写入遭遇 PermissionError；按其生成差异补齐条目并标注后，
`--update` 与 `--check` 均正常退出。uv 缓存受限的命令在获准的环境重跑后通过。

## 8. S3 实施依据

估计值的分布目标与实际消费者分开记录。例如 RobustPU warm-up 使用 U 正类比例，
总体指标使用总体先验。

ReCPE 的[官方论文 v2](https://arxiv.org/html/2002.03673v2) Algorithm 1 第 3 步将所选 U
复制到 P，第 4 步仍使用完整原生 U 作为 mixture；
[官方固定实现](https://github.com/a5507203/Rethinking-Class-Prior-Estimation-for-Positive-Unlabeled-Learning/blob/8ea9324fda4afc1c860b211602d1137b8eb1e55e/run_experiments.py)
的 `runALGS` 将 `tr_positive_sample + estimated_set_A` 与完整 `tr_unlabeled_sample`
传给底层 KM。该流程是 S3 修正复制政策的依据。
论文 §3.3 式(7)给出 $`\pi'=\pi+(1-\pi)P_n(A)`$；其中 $`P_n(A)`$ 表示潜在负类在
被选区域的概率质量。辅助分布身份和近似条件随估计结果记录。

显式 TS 路径使用与其输入设计对应的画像和指标。OS 画像通过 `q/pi`
审计标记率；TS 的额外 P 占比由采样设计确定。专项用例采用 P 占拼接表 0.8、总体先验 0.4，
验收该合法 TS 构造的画像与评分。

S3 的回归验收如下：

- native-U stub 为 0.25、OS q 为 0.2 时，总体解析为 0.4；严格 TS 使用相同的总体/U 数值。
- 外部估计器的标量需要结合估计目标和估计池解释，不默认视为总体先验。
- ReCPE 的 `n_P=10,n_U=20,copy_fraction=0.2` 保留 P=14、U=20，共 34 行，核对复制行来源。
- 不同训练分区或风险池分别按其实际比例换算；验证与诊断按各自契约消费先验。
- 换算覆盖零标记率、空 U、非法域、非有限值与小样本；TS 画像和指标按采样设计验收。

## 9. P2-S1/S2 验证记录

新增回归首先因缺少新入口失败，随后实现定额标记、兼容包装与独立 TS 生成。
旧 Gaussian 的负类 U 来源通过随机流复现固定。新 TS 逐项验证目标先验、U/拼接表经验先验、
来源身份、参数有效域、整数 seed 与 RandomState 随机流。

受影响范围命令：

```powershell
uv run pytest tests/unit/preprocessing/ tests/unit/cli/test_demo.py tests/unit/core/test_random.py tests/integration/test_iris_smoke.py tests/contract/test_layer_boundaries.py -o addopts='' -q --junitxml=tmp/pu-sampling-p2.xml
uv run python examples/minimal/15_pu_sampling.py --seed 7
uv run python tmp/verify_pu_research_foundations.py
```

示例 seed=7 的目标总体先验为 0.4；独立 TS 的 U 经验先验为 0.415，
从同一总体池标记后 OS 的原生 U 经验先验为 0.2618296529968454。
两组输出分别记录总体参数、实际计数及经验比例。

最终回归结果为 249 通过、24 跳过、0 失败；共 273 个用例。
跳过项沿用 Iris fixture 的条件：方法需要独立 clean support，或方法提供的接口集合中
没有 predict_proba。14 条 warning 包括旧名字迁移提示及既有 uPU/Self-PU 训练提示。
结果保存在 `tmp/pu-sampling-p2.xml`。

9 段文档 Python 片段全部执行通过。文档链接、API 覆盖、Ruff 格式、测试质量、目录、
层依赖、注释质量和 `git diff --check` 全部通过；静态命令沿用 §7，并新增
`uv run python scripts/check_comment_quality.py`。

## 10. P2-S3 先验证据与验证入口

S3 分别核对了 ReCPE 复制所选 U 而保留原生 U、OS/TS 总体池不同及先验参数消费目标。
KM1/KM2 和 penL1 在分离的混合分布上用 0.25、0.5、0.75 三档原生 U 先验校验。
PUSBKernel 需要分别核对风险优化、内部验证和训练表分位阈值，不能仅依据参数名称判定先验目标。

现有验证入口包括 `tests/unit/prior/test_prior_semantics.py`、`tests/unit/prior/test_recpe.py`，
以及 [ReCPE 示例](../../examples/minimal/05_recpe_pipeline.py)和 [uPU 先验示例](../../examples/minimal/16_prior_targets.py)。
示例中的训练表准确率不等同于独立测试成绩。

PU 零一风险函数默认使用原生 U 公式，PU 精度函数默认使用完整 OS 公式；按目标池显式传入
`input_sampling`。Survey 的训练风险配方及结果身份仍由实验适配器维护。

## 11. P3 共享原语验证记录

固定正例计数与均匀无放回标记在 `preprocessing/_sampling.py` 单源维护。
OS 固定配额构造与 `SCARGenerator` 复用抽样；
LBE-A/B 的配额通过 `_n_labeled` 委托计数，各自保留权重与加权抽样政策。
重构前的 59 个基线用例全部通过，锁定标签、计数、标签哈希和后续随机流。
全选同样执行随机抽样；seed=19 的后续五个随机整数保持为
`[834, 3973, 266, 2279, 4048]`。

`cvir_risk_prior` 的 OS 原生 U 分支委托 `population_to_native_unlabeled`，
完整 OS 风险池校准分支使用总体值。包装器保留严格域、计数、视图和异常契约；
实际参数绑定、选模总体先验和来源记录继续由实验层维护。

```powershell
uv run pytest tests/unit/preprocessing/ tests/unit/experiment/test_shared_sampling.py tests/unit/experiment/test_strategies_labeling.py tests/unit/prior/test_prior_semantics.py tests/unit/experiment/test_cvir_population_adapter.py tests/unit/experiment/test_cvir_image_candidate.py tests/unit/experiment/test_cvir_bn_cardinality.py tests/unit/scripts/test_cvir_image_execution.py tests/contract/test_layer_boundaries.py tests/contract/test_ledger_registry_consistency.py -o addopts='--tb=short' -q --maxfail=6 --junitxml=tmp/pu-research-p3-shared.xml
uv run python examples/minimal/15_pu_sampling.py --seed 7
```

共享实现专项 281 个用例全部通过。采样示例的计数、经验先验和来源身份与 S1/S2 记录一致。
格式、测试质量、文档链接、生成目录、层依赖、注释质量及 `git diff --check` 通过。

## 12. P4 组合示例与教程验证记录

E1/E3 沿用示例 15/16。E2 由示例 17 在同一 OS 池对照 linear/nonlinear Bernoulli
与 LBE-A/B 定额抽样；seed=7、160 个真正例、请求标记率 0.4 时，实际计数为
63、61、64、64，并记录机制参数、评分来源与标签哈希。
E4 在示例 14 的 `--fixed-selection` 路径中使用两个固定单点候选，
PA 选择 A、阈值 0.5、准则值 1.5，OA 选择 B、阈值 0.5、验证 Accuracy 1；
改变 test 标签后选择记录相同。

E5 的示例 18 使用相同初始化模板运行 nnPU 端到端 CNN 和冻结特征 uPU。
OS 总体先验 0.5、标记比例 0.25，对应实际训练先验 1/3。回归核对模板不变、
拟合副本更新、四角色来源保留及 CPU 模型 pickle 预测一致。
默认 tiny 路径实际训练两轮；共享 CNN13-no-BN/ResNet18 工厂均实际前向验证，
完整骨干训练通过命令行选项运行。
E6 的示例 14 保留完整三策略注入，uPU 显式使用完整 OS 风险池并消费总体先验。

```powershell
uv run pytest tests/integration/test_diy_experiment_example.py tests/integration/test_sar_policy_example.py tests/integration/test_custom_encoder_example.py tests/unit/experiment/test_checkpoint_selection.py -o addopts='--tb=short' -q --maxfail=5 --junitxml=tmp/pu-research-p4-examples.xml
uv run python examples/minimal/14_diy_experiment.py
uv run python examples/minimal/14_diy_experiment.py --fixed-selection
uv run python examples/minimal/17_sar_policies.py
uv run python examples/minimal/18_custom_encoders.py --output-dir tmp/pu-research-p4-encoder-output
uv run python tmp/verify_pu_research_p4_docs.py
```

31 个用例全部通过。encoder 教程 5 段 Python 片段与 DIY 完整片段实际执行通过；
输出目录中的模型、冻结 encoder/分类器组合及设置文件实际生成。
格式、测试质量、文档链接、API 覆盖和生成目录检查通过。
概念、用户入口和文档总导航已连接到新增教程与示例。

## 13. P5 运行设置与报告

`experiment.research_settings` 将生成元数据、风险视图请求、先验参数输入、监督角色和
表征身份组织为 `research_settings`（schema `1.0`）。生成与训练契约校验后建立设置，
候选训练结束时补入实际训练留痕；成功及全部候选失败的运行均保存该记录。
SCAR/LBE-A/LBE-B 额外声明 OS 和具名定额抽样政策，保留已有标签与随机轨迹。

先验记录分为基础输入、逐候选有效输入、候选实际训练留痕及选模总体先验。
CVIR 的实际风险池、先验来源和总体到风险池换算从现有训练留痕取得。
自定义选模策略的验证角色按实际返回的协议名记录。
表征设置使用运行记录中的路径、表示身份、输入维度及 adapter/image 元数据。

结果读取按 seed 保存设置，在 JSON、CSV、Markdown、工作簿和诊断卡中展示摘要。
历史记录的设置保留为 `null`，摘要为“历史记录：按原 manifest 设置读取”。
旧、新及混合记录的身份、指标、成本和原文件字节回归通过。
Survey 协议选择具体研究条件，并链接公共采样、标记、先验、选模与表征概念；
runbook、API、nnPU/CVIR 方法卡与实验层设计同步说明字段及使用路径。

报告与历史身份专项回归 109 项通过，见 `tmp/p5-report-regressions.xml`。
实验层、三个组合示例与层间契约首次回归共 1922 项：1915 项通过、4 项跳过，
3 项失败定位为自定义协议名读取及调整期间的子进程导入；相应修复使用定向回归验证。
4 项跳过分别对应当前 NumPy 的 promotion 兼容测试钩子，以及 KLDCE 的 SCAR 配方范围。

最终定向回归 80 项全部通过，覆盖上述三个失败点、候选先验覆盖、具名方法风险池、
CVIR 两种适配器的实际消费留痕和 episodic 子进程执行。
完整 DIY 示例实际生成 `basic.json`、`custom.json` 两份 manifest；其中保存新增设置。

```powershell
uv run pytest tests/unit/experiment tests/integration/test_diy_experiment_example.py tests/integration/test_sar_policy_example.py tests/integration/test_custom_encoder_example.py tests/contract/test_layer_boundaries.py -m 'not slow and not e2e' -o addopts='--tb=short' -q --maxfail=8 --junitxml=tmp/pu-research-p5-regressions.xml
uv run pytest tests/unit/experiment/test_research_settings.py tests/unit/experiment/test_runner.py tests/unit/experiment/test_cvir_known_prior_adapter.py tests/unit/experiment/test_cvir_population_adapter.py tests/unit/experiment/test_survey_episodic_budget.py::test_basic_actual_split_pilot_resumes_after_native_early_stop -o addopts='--tb=short' -q --junitxml=tmp/pu-research-p5-final-focused.xml
uv run python examples/minimal/14_diy_experiment.py --output-dir tmp/pu-research-p5-diy-output
```

格式、文档链接、API 覆盖、测试质量、生成目录和层依赖门禁通过。
测试明细分别保存于上述 XML 与 `tmp/p5-report-regressions.xml`。

## 14. GenPU 的先验校验

GenPU 的 `risk_prior` 在默认 OS 视图对应原生 U，且必须是 `(0, 1)` 内的有限值。
`tests/unit/estimators/test_gen_pu.py` 覆盖缺失值、0、1 和非有限值。

## 15. PA 在 OS 采样设计下的参考实现差异

本节记录 PA 选模分数的实现对照。工具箱的
[`proxy_accuracy`](../../pu_toolbox/experiment/strategies.py) 采用 Wang et al. (2026)
在 OS 采样设计下的 PA 定义，第二项为完整验证分区 $`L\cup U`$ 上的预测负类比例。
定义和使用条件见[PU 选模](../user/concepts/pu_model_selection.md#2-pa-conditions-and-formulas)。

[PUBench 的固定版本实现](https://github.com/wu-dd/PUBench/blob/a9a62b05b0f222c72aff4df8992307376c78d682/lib/misc.py)
与 [PU-Bench 的固定版本实现](https://github.com/XiXiphus/PU-Bench/blob/2d95a19eefd72e66ff30128ec2f65e1d1d4cc077/train/utils/metrics.py)
在这一项使用观测 PU 标签准确率。令 $`q=n_L/(n_L+n_U)`$，
$`a=\widehat P_L(\hat Y=1)`$，$`d=\widehat P_U(\hat Y=0)`$，则两种第二项分别为：

```math
\text{预测负类比例}=q(1-a)+(1-q)d,\qquad
\text{观测 PU 标签准确率}=qa+(1-q)d.
```

因此，使用观测 PU 标签准确率的 PA 分数比论文定义多出 $`q(2a-1)`$。
该差值随候选模型与阈值变化，可能改变选模排序。复现实验记录所采用的 PA 定义、
参考仓库版本和阈值选择方式。
