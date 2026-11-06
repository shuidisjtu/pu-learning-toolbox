# PU 输入采样与训练风险视图

## 1. 先确定 U 来自哪里

PU 标签 `1/0` 表示已标记正例和未标记角色；U 的分布由采样设计决定。
设真实类别为 $`Y`$，目标总体先验为 $`\pi_{population}=P(Y=1)`$。

| 输入设计 | P（已标记正例）怎样取得 | 原生 U 的分布 |
|---|---|---|
| OS / SS（one-sample / single-sample） | 在同一个总体样本中标记一部分正例 | 剩余未标记样本，服从 $`X\mid S=0`$ |
| 严格 TS / CC（two-sample / case-control） | 从正类条件分布独立取得额外 P | 从目标总体取得混合样本，先验为 $`\pi_{population}`$ |
| 有限池 benchmark 构造 | 从已有池抽取或复用 P | 取决于是否移除 P、是否放回、是否复用完整池 |

OS 中 $`S`$ 是总体内的标记事件；TS 中 P/U 身份及其拼接比例由采样设计决定。
两种设计都可以叠加 SCAR 或具名 SAR 选择机制；机制和采样政策见
[SCAR/SAR](scar_sar.md)。
若额外 P 经 SAR 筛选，其分布一般不再代表完整 $`X\mid Y=1`$，
表中正例代表性及下文风险分解的条件需另外检查。

有限池设计应记录来源池、原始索引、放回政策及角色间重用。R4 Appendix C.2
将完整训练池作为 U，并从该池选取 P，形成具名的样本重用设计。

## 2. OS 与 TS 的先验算例

OS 目标总体先验为 0.4，正类平均标记率为 0.5，则总体已标记比例
$`q=0.4\times0.5=0.2`$。移除已标记正例后，原生 U 的正类比例为：

```math
\pi_U=\frac{0.4-0.2}{1-0.2}=0.25.
```

严格 TS 的 U 直接取自同一个目标总体，其目标先验仍为 0.4。
如果额外 P 占 TS 拼接训练表的 20%，该表的正类比例目标为
$`0.2+0.8\times0.4=0.52`$。总体、U 与拼接表应分别命名；有限样本中的
经验比例还会受到抽样波动影响。完整推导见[先验语义](class_prior_semantics.md)。

## 3. 风险分解决定实际损失池

当 P 代表真实正类分布时，期望分类风险可写为：

```math
R(f)=\pi_{population} E_{X\mid Y=1}[\ell(f,+1)-\ell(f,-1)]
     +E_{population}[\ell(f,-1)].
```

严格 TS 用原生 U 估计最后一项；OS 用完整 $`L\cup U`$ 估计。
SAR 下已标记正例服从选择后的正类分布，风险估计还需处理该选择偏差。
各方法的统计假设和风险分解见具体实现。

当前 [TrainingView 实现](../../../pu_toolbox/core/training_views.py)明确三个角色：

| 角色 | `requested_view="os"` | `requested_view="ts"` |
|---|---|---|
| `positive` | 已标记 P | 已标记 P |
| `native_unlabeled` | 输入中标签为 0 的 U | 同一原生 U |
| `loss_unlabeled` | 原生 U | 原生 U 后接 P |

这里的 `ts` 构造 **OS 输入上的 TS-compatible 训练风险池**。
正例同时参与正例损失和总体混合损失，PU 标签不被重写。该构造只允许用于 train，
验证与测试沿用各自原始视图。来源索引随角色保留。

```python
import numpy as np
from pu_toolbox.core.training_views import build_training_view

X = np.arange(20).reshape(10, 2)
y_pu = np.array([1, 1] + [0] * 8)
view = build_training_view(X, y_pu, requested_view="ts")
assert len(view.positive_indices) == 2
assert len(view.native_unlabeled_indices) == 8
assert len(view.loss_unlabeled_indices) == 10
```

这是角色构造示例。实际 estimator 是否支持该视图、整池与 mini-batch 如何加权，
仍须查看方法实现。R3 Algorithm 1 的 mini-batch 补回与整池估计只有在对应比例、
重采样及权重条件下才可比较；[实现基线](../../dev/pu_research_semantics.md)登记当前差异。

## 4. 当前生成入口及边界

- `make_scar_labels` 在同一 clean 池中定额标记正例，属于 OS 构造。
- `make_fixed_count_labels` 在同池中均匀选择指定数量的 P，保留来源行序。
  `make_pu_labels(mechanism="fixed_count")` 提供分派入口。
- `make_case_control_labels` 及 `make_pu_labels(mechanism="case_control")` 是该同池政策的
  兼容名字，发出 `FutureWarning`，计划在 2.0 移除。
- `make_gaussian_pu_data` 当前将全部生成正例标为 P，U 全部来自负类。
  返回值 `n_p/(n_p+n_u)` 是该拼接表的真实正类比例。
- 混合 U 的 OS 教学数据可使用 `make_scar_dataset`，见
  [API 参考](../reference/api.md)。
- `make_gaussian_case_control_data` 独立生成正类 P 与总体混合 U，返回总体设定先验、
  U/拼接表经验先验及样本来源身份。[采样教程](../howto/pu_sampling.md)给出可运行用法和迁移表。

真实数据应先调查获得方式。对照研究先按原始来源切分 train / pu_val / clean_val / test，
再在允许的分区构造 PU 标签。四角色示例见[自定义实验](../howto/custom_experiments.md)。

## 5. 来源

- Bekker & Davis (2020), §2.3–2.5、式(5)–(7)：
  [Learning from positive and unlabeled data: a survey](https://doi.org/10.1007/s10994-020-05877-5)。
- Mielniczuk & Wawrzeńczyk (2024), §1、式(2)，§2–2.1、式(16)–(19)：
  [Single-sample versus case-control sampling scheme](https://arxiv.org/abs/2312.02095v3)。
- Wang et al. (2026), §4、式(7)–(9)、Algorithm 1：
  [Accessible, Realistic, and Fair Evaluation](https://openreview.net/forum?id=5R11h5o44C)。
- Chen et al. (2026), Appendix C.2：
  [PU-Bench](https://openreview.net/forum?id=tb8DabMbMq)。
