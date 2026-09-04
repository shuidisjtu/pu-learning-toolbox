# Method Card: WConPU

> 参数契约（签名、参数表、返回结构）以 [API 参考](../../user/reference/api.md) 为权威；本文档只记论文研究内容、实现边界与复现状态。

## 1. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Weighted Contrastive Learning with Hard Negative Mining for Positive and Unlabeled Learning |
| Authors | Botai Yuan, Chen Gong, Dacheng Tao, Jie Yang |
| Venue | IEEE Transactions on Neural Networks and Learning Systems |
| Volume/Issue | 36(6) |
| Pages | 10515-10529 |
| Year | 2025 |
| DOI | `10.1109/TNNLS.2025.3530427` |
| 作者公开 PDF | `yuan_tnnls25.pdf` |
| 方法类型 | 原型驱动的 PU 对比表示学习与分类联合训练 |
| 场景 | case-control PU |
| 类先验 | 必需 |
| 核心模块 | SAT、momentum encoder/queue、prototype、weighted hard negatives、soft pseudo-label、distribution alignment |
| 官方源码 | 未发现公开仓库 |
| Source status | `not_found` |

### Assumptions

```math
\mathcal X_L=\{x_i\}_{i=1}^{n_P}\sim p_P(x),
\qquad
\mathcal X_U=\{x_i\}_{i=1}^{n_U}\sim p(x),
```

```math
p(x)=\pi_Pp_P(x)+(1-\pi_P)p_N(x).
```

---

## 2. 问题设定与符号

`o_i=1` 表示 labeled positive，`o_i=0` 表示 unlabeled。分类器
`f(x)` 输出两类 logits/softmax，query encoder 与 key encoder 分别记为
`g_q`、`g_k`。

| 论文符号 | 含义 |
|---|---|
| $\`\mathcal X_L\`$ / $\`\mathcal X_U\`$ | labeled positive set / unlabeled marginal set |
| $\`\pi_P\`$ | 正类先验（必需） |
| $\`f(x)\`$ | 分类器，输出两类 logits/softmax `z_i` |
| $\`g_q\`$ / $\`g_k\`$ | query encoder / key encoder（EMA 更新） |
| $\`q_i\`$ / $\`k_i\`$ | 样本的 query / key embedding（L2 归一化） |
| $\`\mathcal Q\`$ / $\`\mathcal A\`$ | momentum queue / 当前对比池 |
| $\`\tau_t\`$ / $\`\tau_t(c)\`$ | 全局阈值 / 类别阈值 |
| $\`\mu_c\`$ | 类别 prototype |
| $\`s_i\`$ / $\`h_i\`$ | soft pseudo-label / 最近 prototype 的 one-hot |
| $\`\rho\`$ | contrastive temperature |
| $\`\omega_j\`$ | hard negative 的逆距离权重 |
| $\`\gamma_0\`$ / $\`\gamma_1\`$ | 对比损失 / 分布对齐损失权重 |

---

## 3. 核心公式

### 3.1 Query/Key Encoder 与 Momentum Queue

对样本 `x_i`：

```math
q_i=g_q(\mathrm{Aug}_q(x_i)),
\qquad
k_i=g_k(\mathrm{Aug}_k(x_i)),
```

两者均进行 L2 归一化。key encoder 参数按 EMA 更新：

```math
\theta_k
\leftarrow
\lambda\theta_k+(1-\lambda)\theta_q.
```

维护最近 key embeddings 的 FIFO queue：

```math
\mathcal Q=\{k_1,\ldots,k_m\}.
```

当前对比池：

```math
\mathcal A=\mathcal B_q\cup\mathcal B_k\cup\mathcal Q.
```

### 3.2 Self-Adaptive Threshold

令 `z_i=softmax(f(x_i))`。全局阈值：

```math
\tau_t
=
\lambda\tau_{t-1}
+(1-\lambda)\frac1b\sum_i\max(z_i).
```

类别预测分布 EMA：

```math
\widetilde p_t(c)
=
\lambda\widetilde p_{t-1}(c)
+(1-\lambda)\frac1b\sum_i z_i(c).
```

类别阈值：

```math
\tau_t(c)
=
\frac{\widetilde p_t(c)}
{\max_{c'}\widetilde p_t(c')}
\tau_t.
```

只有置信度达到对应类别阈值的预测才进入 positive peer set。labeled positive 的
peer set 还必须显式包含 labeled-positive embeddings，避免珍贵监督被伪标签淹没。

### 3.3 Prototype 与伪标签

class prototype 按预测类别更新：

```math
\mu_c
\leftarrow
\mathrm{Normalize}
\left(
\lambda\mu_c+(1-\lambda)q_i
\right).
```

初始化 soft pseudo-label：

```math
s_i=
\begin{cases}
[0,1]^\top,&o_i=1,\\
[1-\pi_P,\pi_P]^\top,&o_i=0.
\end{cases}
```

由最近 prototype 得到 one-hot `h_i`：

```math
h_{i,j}
=
\mathbf 1
\left[
j=\arg\max_{c\in\{0,1\}}q_i^\top\mu_c
\right].
```

随后：

```math
s_i\leftarrow\alpha s_i+(1-\alpha)h_i.
```

labeled positive 的 pseudo-label 固定，不参与该更新。

### 3.4 Prototype-Based Hard Negative Mining

论文定义归一化 dissimilarity：

```math
\mathrm{DisSim}(q_i,k_j)
=
\frac14
\left\|
\frac{q_i}{\|q_i\|}
-
\frac{k_j}{\|k_j\|}
\right\|_2^2
\in[0,1].
```

用最近 prototype 分配临时类别：

```math
\widetilde y_i=\arg\max_c q_i^\top\mu_c,
\qquad
\widetilde y_j=\arg\max_c k_j^\top\mu_c.
```

hard negative 同时满足：

- prototype 类别不同；
- dissimilarity 不超过当前 queue 距离的第一四分位数。

```math
\mathcal B_i^{\mathrm{neg}}
=
\{
k_j\in\mathcal Q:
\widetilde y_i\ne\widetilde y_j,\,
\mathrm{DisSim}(q_i,k_j)\le Q_{1/4}(x_i)
\}.
```

对应权重：

```math
\omega_j
=
\frac{1}
{\mathrm{DisSim}(q_i,k_j)}.
```

### 3.5 Weighted Contrastive Loss

对 positive peer `k_+`：

```math
\widetilde{\mathcal L}(q_i,k_+)
=
\log
\frac{\exp(q_i^\top k_+/\rho)}
{
\sum_{k'\in\mathcal A\setminus\mathcal B_i^{neg}}
\exp(q_i^\top k'/\rho)
+
\sum_{k_j\in\mathcal B_i^{neg}}
\omega_j\exp(q_i^\top k_j/\rho)
}.
```

对 labeled-positive 和 unlabeled anchor，分别在各自 positive peer set 上取负平均，
再对全体样本平均得到 `L_con`。

### 3.6 分类损失

soft pseudo-label cross entropy：

```math
\mathcal L_{\mathrm{class}}
=
-\frac1n
\sum_i\sum_{c=0}^1s_{i,c}\log z_{i,c}.
```

该损失使 prototype 产生的语义信号反向促进 classifier；classifier 的置信预测又决定
下一轮 positive peer set。

### 3.7 标签分布对齐

论文复用 Dist-PU 风险：

```math
\mathcal L_{\mathrm{dis}}
=
2\pi_P
\left|
\frac1{n_P}\sum_{x_i\in\mathcal X_L}z_{i,1}-1
\right|
+
\left|
\frac1{n_U}\sum_{x_i\in\mathcal X_U}z_{i,1}-\pi_P
\right|.
```

### 3.8 总目标

```math
\mathcal L_{\mathrm{WConPU}}
=
\mathcal L_{\mathrm{class}}
+\gamma_0\mathcal L_{\mathrm{con}}
+\gamma_1\mathcal L_{\mathrm{dis}}.
```

三个损失不能被相互替代：

- `L_class` 训练最终分类器；
- `L_con` 改善表示几何；
- `L_dis` 抑制负预测偏好。

---

## 4. 算法概要

WConPU 在每个训练迭代中协同执行：

1. 对输入生成 weak query view 和 strong key view；
2. 由 query/key encoder 生成归一化 embedding；
3. 使用 classifier 预测与 SAT 构造 positive peer set；
4. 由 prototype 划分类别并挖掘 hard negatives；
5. 计算 weighted contrastive loss；
6. 根据 prototype 更新 U 样本 soft pseudo-label；
7. 计算分类损失和标签分布对齐损失；
8. 更新 query 网络、classifier、prototype、key encoder 与 queue。

---

## 5. 论文边界

- 论文从 EM 角度解释其交替过程（classifier/pseudo-label 更新近似 E-step，
  contrastive representation 更新近似 M-step），但这只是解释性类比，不意味着
  非凸训练具有全局收敛保证。
- 对比学习结果高度依赖 augmentation，论文没有公开 SimAugment/RandAugment 参数。
- 论文只给出“13-layer CNN”名称，没有逐层 topology；当前适配器属于 clean-room，
  不能称为精确官方网络。
- 论文未公开 clean validation 的选择指标；accuracy 是显式、可替换的暂定选择。
- 原文训练样本计数（`nP=1000`、`nU=50000`）与额外保留 10% validation 在 50000 张
  训练图像上不能同时保持互斥；当前协议优先保证无泄漏。
- early-stage classifier 错误会污染 peer set 和 prototype，形成确认偏差。
- hard-negative inverse distance 可能产生大梯度，必须做 `eps`/clamp。

---

## 6. 实现注记

> 见 ADR-0014 #13（标签分布对齐项含
> $`2\pi_P\left|\frac1{n_P}\sum_{x_i\in\mathcal X_L}z_{i,1}-1\right|+\left|\frac1{n_U}\sum_{x_i\in\mathcal X_U}z_{i,1}-\pi_P\right|`$，
> 正则惩罚依赖先验 `pi_P`，仓库旧元数据 `requires_class_prior=False` 必须修正）

- **除零保护**：`DisSim` 计算必须设置数值下界 `eps`，防止 embedding 极近时权重
  `1/DisSim` 除零。
- **无 positive peer 的 anchor**：如果某个 anchor 没有可靠 positive peer，应跳过该
  anchor，而不是制造空集合 NaN。
- **默认 epoch**：工具箱默认 100 epoch（论文为 800 epoch，不使用 early stopping），
  已实测 50 epoch 即收敛，可用 `--max-epochs` 调整。
- **默认增强**：默认 tabular augmentation 仅用于接口验证，不等价于论文
  SimAugment/RandAugment。

---

## 7. 论文实验参考

数据集：CIFAR-10、SVHN、STL-10、Alzheimer MRI。

| 项目 | 论文设置 |
|---|---|
| 骨干网络 | CIFAR-10/SVHN 使用 13-layer CNN，STL-10 使用 ResNet-18，Alzheimer 使用 ResNet-50 |
| projection head | 2-layer MLP，输出 128 维 |
| 增强 | weak augmentation 为 SimAugment，strong augmentation 为 RandAugment |
| queue size | Alzheimer 4096，其余 8192 |
| 核心超参数 | `alpha=0.9`、`lambda=0.999`、`rho=0.07` |
| 优化器 | SGD momentum 0.9，初始学习率 `1e-2`，cosine annealing |
| batch size | Alzheimer 8，其余 256 |
| 损失权重 | `gamma_0`、`gamma_1` 从 `{1e-3,1e-2,1e-1,1}` 网格选择 |
| 训练轮数 | 800 epoch，不使用 early stopping |
| 指标协议 | 每组独立运行 5 次并报告 6 项分类指标 |

---

## 8. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `not_found`：论文未提供公开源码，当前实现是 clean-room |
| 实现状态 | `WeightedContrastivePUClassifier` + 视觉适配层，状态 `NATIVE`；视觉链路与 clean validation 选参已接入；未公开参数及长周期 GPU 训练待完成 |
| runner 状态 | `benchmarks/deep_pu/` 提供统一 runner、锁定论文配置、3-seed 表格合成结果和 CIFAR-10 visual paper-protocol；`official_data_wconpu_cifar10_protocol.json` 锁定 clean validation 选择协议：每 seed 从 CIFAR-10 canonical training split 隔离 10% 带真值验证集，再从其余样本构造 `1000 P + 44000 U` 训练集，`gamma_0`/`gamma_1` 的 `4 x 4` Cartesian grid 逐候选训练评估并写入 `model_selection.csv`，支持断点续跑，最优参数从头 refit |
| 结果声明 | 论文未说明 clean validation 选择指标，当前暂定 accuracy；互斥计数可审计并记录差异；尚未执行 5-seed × 800 epoch，不能标记为论文结果 |
| 复现风险 | 完整 800-epoch 视觉 benchmark 成本高，不应纳入普通 CI；早期分类器错误会污染 peer set 和 prototype |

参考资料：

1. Yuan et al. *Weighted Contrastive Learning with Hard Negative Mining for Positive and Unlabeled Learning*. IEEE TNNLS, 2025.
2. DOI: <https://doi.org/10.1109/TNNLS.2025.3530427>
3. 作者 PDF：<https://gcatnjust.github.io/ChenGong/paper/yuan_tnnls25.pdf>
