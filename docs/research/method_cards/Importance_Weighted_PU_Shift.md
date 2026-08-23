# Importance-weighted PU Learning for Distribution Shift Adaptation

## 来源与状态

- 论文：Kumagai et al., AISTATS 2025，
  [PMLR 原文](https://proceedings.mlr.press/v258/kumagai25a.html)
- 工具箱实现：`DynamicJointShiftPUClassifier`
- 复现等级：公开公式的 clean-room research 实现
- 作者源码：论文 reproducibility checklist 标明 proprietary，未公开

因此本实现可以核对公开公式与训练顺序，但不能声明 `official_exact` 或论文数值复现。

## 权重目标

论文使用相对联合密度比

\[
w_\alpha(x,y)=\frac{p_{te}(x,y)}
{\alpha p_{te}(x,y)+(1-\alpha)p_{tr}(x,y)},
\qquad 0<\alpha\le 1.
\]

权重头输出为 `sigmoid / alpha`，因此位于 `[0, 1/alpha]`。工具箱的
`paper_importance_weight_objective` 对应论文式 (19)：目标域负类替换项使用其理论下界
`-(1-pi_te)/alpha` 做 `abs(z-c)+c` 修正，源域负类替换项使用 `abs(z)` 修正。

## 分类目标与动态训练

分类目标对应论文式 (20)：

\[
\widehat L(f,m)=\beta\widehat R_{te}(f)
+(1-\beta)\widehat R^w_{tr}(f,m).
\]

两个 PU risk 都使用 sigmoid loss 和绝对值负风险修正。模型共享特征提取器 `h`，分类头为
`u(h(x))`，权重头为 `v([h(x),y])`。每个 epoch 严格分两步：

1. 固定 `h`，按式 (19) 只更新 `v`；
2. 固定当前权重，按式 (20) 更新 `h` 和 `u`。

`training_mode="two_step"` 与 correction 开关用于消融。`trPU`、`tePU`、fine-tune 和
五核 RBF-MMD 对照由 `build_joint_shift_estimator` 以相同神经规模构造。

## 已有证据与边界

- 公式金标准覆盖式 (13)、(19)、(20) 的手算值、修正与 beta 端点；
- 固定种子验证动态训练、权重范围和预测可重复；
- 公开 Wisconsin 数据提供多 seed、95% CI 和样本重叠 smoke；
- 尚未复跑 MNIST/FMNIST/CIFAR10/DIABETES/FoodStamp 原协议、验证集调参和 10 次完整实验；
- 尚未建立与作者私有实现的参数/梯度逐项一致性证据。
