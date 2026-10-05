# Method Card: Rank Pruning (RP)

## 来源

Northcutt, Wu, Chuang, UAI 2017：[原文](https://auai.org/uai2017/proceedings/papers/35.pdf)，
[作者代码](https://github.com/cgnorthcutt/rankpruning)，核查 commit
`40ae17cc50fcfeec8d7a345a1c5426036d2bdb1a`，MIT。
作者旧包已弃用而推荐 cleanlab；本项目保留 RP 论文基线，不将新 cleanlab 当作同一算法。

## PU 特例

只实现无误标正例的 PU 特例，ρ₀=P(s=1|Y=0)=0；不是通用双向 noisy-PN。
先对 s 作分层 CV，heldout 概率 q=P(s=1|x)，正例均值阈值 t 决定 confident-positive 集合：

```math
\widehat\rho_1=\frac{\#\{s=0,q\ge t\}}{\#\{q\ge t\}},\quad
\widehat\pi=\overline s/(1-\widehat\rho_1),\quad
\widehat\pi_0=\widehat\rho_1\widehat\pi/(1-\overline s).
```

ρ₁ 取各 CV fold confident-count 估计的均值（不改成合并 OOF 计数）；可显式给定
`frac_pos2neg`，但排序仍使用 OOF。按 U 中概率最高的样本删去估计噪声数，默认每类至少留 10 个；
作者使用严格概率阈值，ties 保留，因此实际删除数可能小于预算。P 完全保留。
最终在保留集重新拟合，P 权重 1/(1−ρ₁)，U 权重 1。内部重权不是外部 sample_weight 支持。

## 校准与实现差异

原生 OS、无同形未标记风险项；显式 ts fail-loud，不给 P 添加 s=0 复制行。
推导 π 只是训练观测/噪声模型诊断，不需要外部 π。若退化模型推导 π≥1，保留作者原估计、
夹断 inverse noise π₀ 并在 `prior_diagnostic_` 显式报告；不能拿该值作合法类先验。
默认 min_retained=10/严格 ties 行为与源码一致；为 fold 隔离每次 clone，给 CV 固定 seed，
而旧作者代码在 folds 中复用 classifier、未固定 shuffle seed。源忠实性因此为 related，而非完整 exact。

`RankPruningClassifier`，注册 `rp` / `rank_pruning` / `rankpruning`；默认 sklearn LogisticRegression。
可注入模型必须有二分类 predict_proba/classes_ 和 fit(sample_weight)。仅稠密二维/CPU；
支持 clone、pickle、预测与训练诊断。每候选成本包含 n_cv_folds+1 次训练，不能只算最后一次。

独立公式、strict ties、fold 互斥、权重、保存加载及 OS-only 门禁测试见
`tests/unit/estimators/test_rank_pruning.py`。未进入旧 frozen matrix，公开结果和正式预算仍待登记。
