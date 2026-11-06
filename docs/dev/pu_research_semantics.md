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
| `make_scar_labels` / `SCARGenerator` | 同池、正类均匀无放回；`max(1, round(c*n_pos))`，不超过正类数；无正例返回全 U | P3 共享计算前锁定 seed/计数；R4 的 floor 配额另外登记 |
| `make_fixed_count_labels` | 同池选定 n_labeled 个正例，其余原位 U，支持全部为正类的池 | S1 已实现；n_labeled 校验正整数及可用正例数 |
| `make_case_control_labels` | 同池定额标记的旧名字，延续双类别输入校验与随机轨迹 | S1 已实现兼容包装、FutureWarning 与 2.0 移除计划 |
| `make_pu_labels(case_control)` | 直接委托上项 | 与 S1 一并处理，不单独改采样身份 |
| `make_gaussian_pu_data` | 先生成 n_p 个正类并全部可见；再生成 n_u 个负类作为 U；返回拼接表正类比例 | S2 已固定旧场景与 seed 回归，docstring 明确分布与返回含义 |
| `make_gaussian_case_control_data` | P 从正类条件分布抽取，U 独立从总体混合分布抽取；返回真值及来源/先验元数据 | S1/S2 已实现；总体参数与经验先验分别记录 |
| `make_sar_propensity` / `make_sar_labels` | 正类均值校准，Bernoulli；ensure_labeled 默认补标最高 propensity 正例 | 生成概率与补标后的入选概率分别解释；元数据后续收敛 |
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
| KM1 / KM2 | `X[y_pu==0]` 为 F，`X[y_pu==1]` 为 H；返回 F 中 H 的 mixture proportion κ | 代表性与可辨识条件成立时解释为 U 正类比例；Pipeline 按输入设计解析为总体值 |
| penL1（`ClassPriorEstimator`） | P、原生 U 的 RBF 均值，优化 `theta*mean(phi_P)-mean(phi_U)` | 输出 mixture 目标 θ；OS 经质量换算，TS 目标与总体相同 |
| `ReCPEEstimator` | ranker 取高分 `max(1,ceil(copy_fraction*n_U))` 行复制进 P；默认 KM2 消费重组 P 与完整 U | base 值经 `[0,1]` clip 后作为原生 U 先验近似返回；元数据记录辅助目标与复制政策 |
| 独立 `BBEMixtureEstimator` | 观测 P/U 分别切 train/holdout；holdout U 输出 `alpha_U` | 要求 `input_sampling="ts"`；有具名目标/来源，不走 Pipeline `.estimate()` 链 |

实现：[kernel_mean.py](../../pu_toolbox/prior/kernel_mean.py)、
[pen_l1.py](../../pu_toolbox/prior/pen_l1.py)、[recpe.py](../../pu_toolbox/prior/recpe.py)、
[bbe.py](../../pu_toolbox/prior/bbe.py)。
mixture 估计按各方法的代表性/可辨识条件解释。

`PUPipeline._resolve_prior` 在 `fit(X,y_pu)` 后读取 `estimate_target` 与 `estimation_pool`，
按 `input_sampling` 将估计解析为总体值。显式总体 class_prior > 分类器构造值 > 估计值；
构造值按分类器默认风险视图的 `class_prior_target` 解释。第三方先验估计器及分类器
继承 `population` 默认目标。

[PriorInfo](../../pu_toolbox/workflows/report.py) 保存原值/目标、估计池、输入设计、转换名、
估计器细节和逐折消费。训练边界根据各折 P 占比换算原生 U 或拼接表目标，
`os_or_ts` 沿用方法默认值。PUSBKernel 的 `risk_class_prior` 与分位阈值先验分别消费。
总体评估池在 OS 为完整输入、在 TS 为原生 U，画像及诊断同步使用输入设计。

## 4. 历史 os_or_ts 的逐实现清单

以下为所有 `fit` 签名含 `os_or_ts` 的 estimator（AST 全量扫描，共 19 个）。
它们表达风险角色；输入采样设计和先验消费目标分别记录。

| 方法 | 默认 | 当前 ts 行为与先验消费 | 实现 |
|---|---|---|---|
| uPU | os | 完整池替换 U 风险；class_prior 不变，RBF 中心池随之变化 | [upu.py](../../pu_toolbox/estimators/risk/upu.py) |
| nnPU | os | mini-batch 补回 P；class_prior 不变；拒绝与 sample_weight 同用 | [nnpu.py](../../pu_toolbox/estimators/risk/nnpu.py) |
| VPU | ts | 使用完整风险池；class_prior 仅兼容校验，训练不使用 | [vpu.py](../../pu_toolbox/estimators/risk/vpu.py) |
| CVIR | os | input_sampling 与风险视图分开；消费风险池 unlabeled_positive_prior，拒绝 class_prior | [cvir.py](../../pu_toolbox/estimators/risk/cvir.py) |
| GradPU | os | 训练视图适配；class_prior 仅兼容，训练不使用 | [grad_pu.py](../../pu_toolbox/estimators/deep/grad_pu.py) |
| RobustPU | os | 仅 warm-up 适配，后续 episode 仍取 native U；class_prior 指 warm-up U 风险的正类比例 | [robust_pu.py](../../pu_toolbox/estimators/deep/robust_pu.py) |
| GenPU | os | 支持风险适配并使用先验；生成/分类组件需分别解释 | [gen_pu.py](../../pu_toolbox/estimators/deep/gen_pu.py) |
| SplitPU | os | 仅 teacher 的 nnPU marginal 补回 P；先验用于 teacher 风险；easy/hard 仍取原生 U | [split_pu.py](../../pu_toolbox/estimators/deep/split_pu.py) |
| PULDA | os | 分布与 two-way margin 风险池补回 P；先验用于二者目标，伪标签空间不变 | [pulda.py](../../pu_toolbox/estimators/risk/pulda.py) |
| PUET | os | P 副本作为 U-risk 角色行；先验用于 P 质量权重，不换算数值 | [puet.py](../../pu_toolbox/estimators/risk/puet.py) |
| DistPU | os | expectation / histogram 对齐池补回 P；先验是被对齐分布的目标比例 | [dist_pu.py](../../pu_toolbox/estimators/risk/dist_pu.py) |
| SelfPU | os | 负 PU 项补回 P，先验用于风险修正；可信集仍基于原生 U，clean support 单独记录 | [self_pu.py](../../pu_toolbox/estimators/deep/self_pu.py) |
| PUSBKernel | os | 适配拟合风险池，先验同时用于风险和分位预测；中心池/CV/阈值池不变 | [pusb_kernel.py](../../pu_toolbox/estimators/bias_aware/pusb_kernel.py) |
| RankPruning | os | 拒绝 ts；class_prior 仅兼容校验 | [rank_pruning.py](../../pu_toolbox/estimators/classic/rank_pruning.py) |
| HolisticPU | os | 拒绝 ts；class_prior 仅兼容校验 | [holistic_pu.py](../../pu_toolbox/estimators/deep/holistic_pu.py) |
| LBE-neural | os | 拒绝 ts；外部 class_prior 仅校验，内部先验为自身推断值 | [lbe_neural.py](../../pu_toolbox/estimators/deep/lbe_neural.py) |
| P3Mix | os | 拒绝 ts；class_prior 仅兼容校验 | [p3mix.py](../../pu_toolbox/estimators/deep/p3mix.py) |
| PAN | os | 拒绝 ts；class_prior 仅兼容校验 | [pan.py](../../pu_toolbox/estimators/deep/pan.py) |
| PULNS | os | 拒绝 ts；class_prior 仅兼容校验 | [pulns.py](../../pu_toolbox/estimators/deep/pulns.py) |

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
| tedn | pu | 拒绝 class_prior；构造必填 input_sampling=ts；内部 BBE 的 holdout alpha_U 用于负例筛选，有 prior_target，不做 OS 适配 | [tedn.py](../../pu_toolbox/estimators/deep/tedn.py) |
| elkan_noto | pu | 内部估计 c 后计算/裁剪 `n_L/(n*c)`，解释依赖 OS/SCAR；q 是 OS 总体的标记比例 | [elkan_noto.py](../../pu_toolbox/estimators/classic/elkan_noto.py) |
| lbe | pu | 可选 pi 用于原生 U 潜在后验初始化，缺省 `clip(2*mean(S),0.05,0.95)`；不固定训练后的先验 | [lbe.py](../../pu_toolbox/estimators/bias_aware/lbe.py) |
| pusb | pu | observed-P-vs-U LogisticRegression baseline；外部 pi 存储为属性，分类目标由观测 P/U 标签定义 | [pusb.py](../../pu_toolbox/estimators/bias_aware/pusb.py) |
| lagam | pu | 外部 pi 仅兼容校验，训练不使用；support_data 是额外监督角色 | [lagam.py](../../pu_toolbox/estimators/deep/lagam.py) |

`requires_class_prior=False` 表示调用者可省略外部先验；
LDCE/KLDCE、InfoMax 和 LBE 分别展示派生、内部估计及初始化三类消费。
DGPU/WConPU 等方法声明 `class_prior_target="native_unlabeled"`，Pipeline 按每个训练分区换算。
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
from pu_toolbox.preprocessing import make_case_control_labels, make_pu_labels

y_true = np.array([1] * 40 + [0] * 60)
y_pu = make_case_control_labels(y_true, n_labeled=20, random_state=7)
assert y_pu.shape == y_true.shape
assert np.all(y_pu[y_true == 0] == 0)
assert np.mean(y_pu == 1) == 0.2
assert np.mean(y_true[y_pu == 0]) == 0.25
assert np.array_equal(y_pu, make_pu_labels(
    y_true, mechanism="case_control", n_labeled=20, random_state=7,
))
```

独立 TS 新入口从指定总体取得混合 U，并独立取得额外 P。

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

### S3：Pipeline 解析 U 估计值

```python
import numpy as np
from pu_toolbox.core.base import BasePriorEstimator
from pu_toolbox.workflows import PUPipeline

class NativeUStub(BasePriorEstimator):
    estimate_target = "native_unlabeled"
    estimation_pool = "native_unlabeled"

    def fit(self, X, y_pu):
        return self

    def estimate(self):
        return 0.25

y_pu = np.array([1, 1] + [0] * 8)
pipeline = PUPipeline(classifier="upu", prior_estimator=NativeUStub())
value, info = pipeline._resolve_prior(
    X=np.arange(20).reshape(10, 2), y_pu=y_pu,
    class_prior=None, classifier_instance=None, needs_prior=True,
)
assert abs(value - 0.4) < 1e-12 and info.source == "estimated"
assert info.input_value == 0.25 and info.input_target == "native_unlabeled"
assert info.conversion == "os_native_unlabeled_to_population"
```

该开发复现使用私有解析入口，验证总体解析值为 0.4；
[公共示例 16](../../examples/minimal/16_prior_targets.py)验证 uPU 默认风险消费原生 U 先验 0.25。
S3 回归保留用户显式值优先级，并验证每个训练折的实际消费。

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
| 本页 S1–S3、四篇概念页中的 Python 片段 | 7 段全部执行通过 |
| 采样/先验/风险/选模及层边界专项 | 215 个用例，通过，3 条 uPU 既有 intercept 警告 |
| encoder/冻结特征/DIY/Pipeline | 通过；14 条既有 intercept / 深度训练成本警告 |
| 文档链接、API 覆盖、测试质量、Ruff 格式、目录、层依赖 | 全部通过；API 覆盖 86 个公开符号 |
| `git diff --check` | 通过 |

专项实际命令：

```powershell
uv run pytest tests/unit/preprocessing/test_pu_labeling.py tests/unit/preprocessing/test_selection_bias.py tests/unit/experiment/test_strategies_labeling.py tests/unit/experiment/test_strategies_selection.py tests/unit/experiment/test_training_views.py tests/unit/experiment/test_runner_class_prior.py tests/unit/prior/test_kernel_mean.py tests/unit/prior/test_recpe.py tests/unit/prior/test_pen_l1.py tests/unit/prior/test_bbe.py tests/contract/test_layer_boundaries.py -o addopts='--tb=short' -q
uv run pytest tests/unit/estimators/test_nnpu_encoder.py tests/unit/estimators/test_encoder_validation.py tests/unit/experiment/test_feature_adapter.py tests/integration/test_diy_experiment_example.py tests/integration/test_pipeline.py -o addopts='--tb=short' -q
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

先验传递按“估计目标 → 总体记录 → 实际消费目标”组织。Pipeline 分别记录 classifier、
PU metrics、profile 和 diagnostics 的目标。
例如 RobustPU warm-up 使用 U 正类比例，总体指标使用总体先验。

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
- 第三方估计器按其声明目标解析，保留显式值与构造值的优先级。
- ReCPE 的 `n_P=10,n_U=20,copy_fraction=0.2` 保留 P=14、U=20，共 34 行，核对复制行来源。
- 各训练折使用实际消费者所需的先验，报告保存原估计值、目标、换算和消费者值。
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

## 10. P2-S3 验证记录

先以失败回归固定 ReCPE 移走原生 U、Pipeline 混用估计目标与训练目标、
OS/TS 指标使用不同总体池以及分类器采样配置冲突的行为，随后逐项修复。
KM1/KM2、penL1 在可辨识的分离混合上分别核对 0.25、0.5、0.75 三档原生 U 先验。
uPU、nnPU、RobustPU 的实际 fit 参数与每折标记比例一同验证，
PUSBKernel 分别核对风险优化、内部 CV 和训练表分位阈值。

验证命令与结果：

```powershell
uv run pytest tests/ -m 'not slow and not e2e' -o addopts='--tb=short' -q --maxfail=15 --junitxml=tmp/pu-research-s3-suite.xml
uv run pytest tests/integration/test_nnpu_pipeline_cnn.py tests/integration/test_pipeline_deep.py tests/integration/test_pipeline_device.py tests/unit/estimators/test_pulda.py tests/unit/workflows/test_pipeline_prior_targets.py tests/unit/prior/test_prior_semantics.py -o addopts='--tb=short' -q --maxfail=6 --junitxml=tmp/pu-research-s3-corrected.xml
uv run python examples/minimal/05_recpe_pipeline.py
uv run python examples/minimal/16_prior_targets.py
uv run python tmp/verify_pu_research_foundations.py
```

常规完整回归首轮为 5836 通过、5 失败、30 跳过、36 取消选择。
5 个失败均来自旧 fixture 中总体先验低于 OS 已标记比例；调整为合法总体先验后，
覆盖这 5 个用例及相关 CNN、设备、PULDA、目标传播和质量换算的 130 个用例全部通过。
扩展专项为 342 通过；最终目标/指标/敏感性专项为 95 通过。
实际数量由 JUnit 记录统计。

示例 16 的 OS 路径报告总体 0.4、训练原生 U 先验 0.25；TS 路径二者均为 0.4。
示例 05 记录 ReCPE 的原生 U 近似 0.446、换算总体 0.593，以及训练准确率 1.000。
9 段文档 Python 片段全部执行通过。格式、文档链接、API 覆盖、测试质量、目录、
层依赖、注释质量及 `git diff --check` 通过。

迁移时，Pipeline 的显式 `class_prior` 使用总体值，分类器构造值按其消费目标解析；
直接调用估计器继续返回其声明目标的标量。PU 零一风险函数默认保留原生 U 公式，
PU 精度函数默认保留完整 OS 公式，研究调用通过 `input_sampling` 显式选择总体池。
Survey 的风险配方与结果身份通过 P3 共享实现工作包继续衔接。
