# 类先验的目标、来源与换算

## 1. 每个先验值都要说明目标分布

同一个 `0.4` 可以代表不同分布中的正类比例，传入训练前先确定消费者需要什么。

| 名称 | 定义 | 典型消费者 |
|---|---|---|
| 总体先验 $`\pi_{population}`$ | 目标总体中的 $`P(Y=1)`$ | 总体风险、OS 的 PA |
| 原生 U 先验 $`\pi_U`$ | 输入 U 内的正类比例 | 原生 U mixture 估计、相应 U 风险 |
| 风险池先验 | 实际进入某项损失的池内正类比例 | CVIR 的 `unlabeled_positive_prior` |
| 拼接表先验 $`\pi_{train}`$ | P 与 U 拼接训练表内的正类比例 | 拼接训练表的构造审计 |

建议研究记录同时写明数值、目标、来源（known / estimated / sensitivity setting）、
估计池、输入采样设计和换算依据。当前公共 `PriorInfo` 的字段见
[API 参考](../reference/api.md)，目标字段与转换记录在 S3 中接入。

## 2. 同一 OS 总体中的质量恒等式

设 $`S=1`$ 只发生在正类中，$`e(x)=P(S=1\mid Y=1,X=x)`$，
$`\bar c=E[e(X)\mid Y=1]`$，总体已标记比例为 $`q=P(S=1)`$。则：

```math
q=\pi_{population}\bar c,\qquad
\pi_U=\frac{\pi_{population}-q}{1-q},\qquad
\pi_{population}=q+(1-q)\pi_U.
```

这些是单边标记下的质量分解，在 SCAR 与 SAR 下均成立。SCAR 时 $`\bar c=c`$。
估计 $`\pi_U`$ 还需满足对应估计器的代表性与可辨识条件。

算例：$`\pi_{population}=0.4,c=0.5`$，所以 $`q=0.2,\pi_U=0.25`$。
若得到的是 U 先验 0.25，且目标确为同一 OS 总体，则：

```python
q = 0.2
pi_u = 0.25
pi_population = q + (1 - q) * pi_u
assert abs(pi_population - 0.4) < 1e-12
```

对经验计数，完整 OS 分区的正类比例等于
$`n_L/n+(n_U/n)\widehat\pi_U`$；它与生成总体的设定参数在有限样本中可以不同。
此处真实标签计数用于合成构造与审计。未知先验实验通过独立估计或预注册设置确定训练先验。

有效域为 $`0\le q<1`$、$`q\le\pi_{population}\le1`$、$`0\le\pi_U\le1`$。
$`q=0`$ 时两个目标数值相等；$`q=1`$ 时 U 为空，U 先验没有定义。
非法值应报告其输入、目标及有效域，便于核对来源。具体算法按其契约校验 `(0, 1)` 等范围。

## 3. TS 与拼接比例

严格 TS 中 U 来自目标总体，因此 $`\pi_U=\pi_{population}`$。
令额外 P 占训练表比例为 $`r=n_P/(n_P+n_U)`$，则：

```math
\pi_{train}=r+(1-r)\pi_U.
```

$`r=0.2,\pi_U=0.4`$ 时，拼接表先验为 0.52。
$`r`$ 是 TS 拼接表中的 P 占比，$`q`$ 是 OS 总体中的标记比例。
有限池重用设计需要按其实际计数和分布解释，见[采样原理](pu_sampling.md)。

## 4. 当前估计器实际估计什么

| 入口 | 当前估计池与输出目标 | 使用条件或限制 |
|---|---|---|
| `KernelMeanPriorEstimator`，KM1/KM2 | 原生 U 为 mixture F，P 为 component H，返回 mixture proportion κ | 正例代表性与相应可辨识条件成立时才解释为 U 正类比例 |
| `ClassPriorEstimator`（penL1） | 比较 P 与原生 U 的基函数均值，返回 mixture 目标 θ | 在相应密度/可辨识条件下解释为 U mixture proportion；总体换算由消费者解析 |
| `ReCPEEstimator` | 将高分 U 重标成 P，base 估计经 `[0,1]` clip 后返回 | 输出对应重组估计池；S3 专项处理复制流程及原始 U 目标 |
| `BBEMixtureEstimator` | 独立 TS 的 holdout U，输出 `alpha_U` | 要求显式 `input_sampling="ts"`，不接入普通 Pipeline 自动估计链 |

实现定位与后续专项验收见[语义基线](../../dev/pu_research_semantics.md)。
各公开 `.estimate()` 当前仍返回标量，未统一携带 target/pool 信息。

## 5. 传入训练时的当前边界

`PUPipeline` 当前优先使用调用时指定值，其次分类器构造值，最后自动估计。
自动链将 `.estimate()` 的标量原样传入消费者。总体风险使用前，按估计目标和
采样设计完成换算并记录计算依据。S3 将目标声明、换算与消费记录接入自动链。
已按适用条件完成换算的数值可通过 `class_prior` 显式传入；
优先级见 [Pipeline 指南](../howto/pipeline.md)。

方法间也存在契约差异：CVIR 使用所选风险池的 `unlabeled_positive_prior` 并拒绝
`class_prior`；RobustPU 的 `class_prior` 用于 warm-up 的 U 风险；
VPU 和 GradPU 将此参数作为兼容字段。各方法按实际消费点解释其参数。

known-prior、estimated-prior 和先验敏感性扫描回答不同问题，结果应分别呈现。
先验、估计方法及超参数由允许的训练/验证信息选择，test 在选择完成后评测。

## 6. 来源

- Bekker & Davis (2020), §2.3–2.5、式(5)–(7)：
  [PU survey](https://doi.org/10.1007/s10994-020-05877-5)，标记率与总体先验关系。
- Mielniczuk & Wawrzeńczyk (2024), §1、式(2)，§2：
  [Single-sample versus case-control](https://arxiv.org/abs/2312.02095v3)，OS/TS 中 U 分布。
- Wang et al. (2026), §4、式(7)–(9)：
  [Accessible, Realistic, and Fair Evaluation](https://openreview.net/forum?id=5R11h5o44C)，内部标签漂移与风险适配。

数学质量恒等式与具体估计器的统计保证分开使用；各估计方法的论文来源见
[先验估计方法卡](../../research/method_cards/class_prior_estimation.md)。
