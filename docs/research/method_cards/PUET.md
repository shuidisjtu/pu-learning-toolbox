# Method Card: PU Extra Trees (PUET)

> 参数签名以 [API 参考](../../user/reference/api.md) 为准。本卡区分论文/作者实现和工具箱当前子集；注册成功不表示 Survey P3.2 验收完成。

## 论文与来源

| 字段 | 内容 |
|---|---|
| Paper | Positive-Unlabeled Learning using Random Forests via Recursive Greedy Risk Minimization |
| Authors | Jonathan Wilton, Abigail M. Y. Koay, Ryan K. L. Ko, Miao Xu, Nan Ye |
| Venue | NeurIPS 2022 |
| 原文 | [NeurIPS 论文](https://proceedings.neurips.cc/paper_files/paper/2022/file/98257285340854262185500e59bc0f28-Paper-Conference.pdf) |
| 作者代码 | [PUExtraTrees](https://github.com/jonathanwilton/PUExtraTrees)，MIT 许可 |
| 本批锁定 | `5cb15d7e021a6e24d4cd300278500c38729e2775`；源码字节摘要见五方法证据包 |
| 类先验 | 必需，$\pi=P(Y=1)\in(0,1)$ |
| 数据假设 | case-control P/U；P 从正类条件分布采样，U 从总体边缘分布采样（SCAR） |

PUET 是**树集成**，不是深度网络。Survey 执行计划将其列入 C 类/P3.2，是当前任务分组而非模型结构事实；正式分组和该项的 GPU 验收适用性须由合作者复核。

## nnPU 二次损失分裂核心

在任意节点 $S$，设 $n_{P,S}$ 与 $n_{U,S}$ 为节点内 P/U 样本数，$n_P,n_U$ 为训练组总数：

```math
W_P(S)=\pi\frac{n_{P,S}}{n_P},\qquad
W_U(S)=\frac{n_{U,S}}{n_U},\qquad
W_N(S)=W_U(S)-W_P(S).
```

论文 Proposition 2(a) 给出 nnPU 二次损失的最优节点风险；采用 $W_U=0$ 时的连续边界值：

```math
\widehat R^*_{\mathrm{nnPU}}(S)=
\begin{cases}
0,& W_N(S)\le 0\ \text{或}\ W_P(S)=0,\\
4W_P(S)W_N(S)/W_U(S),& W_N(S)>0.
\end{cases}
```

候选划分的风险下降为父节点风险减去两个子节点风险。每节点随机取最多 `max_features` 个非恒定特征，每特征随机取 `max_candidates` 个阈值，选择正风险下降最大的有效划分。叶节点在 $2W_P>W_U$ 时投正类，否则投负类；森林取多数票。`feature_importances_` 是各特征风险下降之和在树间的平均，**不是**归一化 Gini 重要性。

## 当前实现边界

- 公开类 `PUExtraTreesClassifier`、注册名 `puet`（别名 `pu_extra_trees`）；仅稠密二维表格数据，CPU/NumPy 实现，无 GPU 路径。
- 当前只实现作者默认的 **nnPU + quadratic** 分支；作者代码另有 uPU 与 logistic 分支，本组件未宣称覆盖。`source_status=official_related` 表示作者代码可追溯，但本实现是独立编写的受限子集。
- 默认 `bootstrap=False` 与作者发布代码一致，树间随机性来自特征/阈值；`bootstrap=True` 可启用论文 §4 描述的 P/U 分组重采样。为保证真正的贪心风险最小化，仅接受正风险下降的划分；与论文实验的完整逐节点决策/数值结果尚未对齐。
- `decision_function` 返回 $[-1,1]$ 多数票边际，不是校准概率；`predict_proba` 不提供。`sample_weight` 非空时报错。
- 已有节点风险 golden、训练/预测、随机性、Bootstrap、参数/边界、pickle 与 pipeline 接入测试。方法台账已登记；`fit(os_or_ts="ts")` 将正例行复制为边缘 U 风险角色，原行仍保留正例角色；U 节点质量分母改为 `n_P+n_U`，P 质量与先验不变。OS/TS 节点风险与分裂增益已有独立数值测试。正式 P3.2 仍需冻结矩阵之外的候选规格、公开结果对照和合作者审阅；本算法为 CPU 树方法，GPU smoke 不适用，任务分组调整仍需复核。

## 准入证据补全（2026-10-05）

2026-10-07 [阶段/预算规格](../pu_survey/p3_stage_budget_review_20261007.md)另列
CPU 单次森林 fit、100棵默认树与独立模型存储，不伪造神经 epoch/provenance 或 GPU 验收。
正式预算、表示、GPU 条款豁免仍待决定。

P1-2 [公开对照准备](../pu_survey/p3_public_comparison_20261005.md)登记原表 Table2
nonnegative/quadratic CIFAR accuracy79.74(0.37)，5次拟合；不是邻列 logistic79.86。
原图3072维/固定P1000/全体训练U与共享冻结特征、c扫描、PA/OA不同，数字保持 pending。

见[五方法证据包](../pu_survey/p3_admission_evidence_20261005.md)。作者 `tree.py:293-298`
在叶节点质量并列时随机投票；我方固定负类。作者 `tree.py:347-348` 按第二小/第二大
值裁边抽阈值，我方用完整 min/max 区间。作者 `tree.py:357-363` 未要求正风险下降；
我方要求 gain>1e-12。上游 `min_samples_leaf` 在该处只检查父节点，我方约束两个子节点。
这四项是明确的适配，不由节点风险公式一致推出完整树等价；森林多数票并列均取负类。
新增 tie/零增益/CPU 非 epoch 预算测试，不能把树数套用神经快照预算。
