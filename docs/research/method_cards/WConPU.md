# Method Card: WConPU

## 1. 方法定位与当前状态

| 项目 | 内容 |
|---|---|
| 全称 | Weighted Contrastive Learning with Hard Negative Mining for Positive and Unlabeled Learning |
| 工具箱注册名 | `weighted_contrastive_pu` |
| 方法类型 | 原型驱动的 PU 对比表示学习与分类联合训练 |
| 场景 | case-control PU |
| 类先验 | 必需 |
| 核心模块 | SAT、momentum encoder/queue、prototype、weighted hard negatives、soft pseudo-label、distribution alignment |
| 当前实现 | `WeightedContrastivePUClassifier` + 视觉适配层，状态 `NATIVE` |
| 完整论文复现 | 视觉链路与 clean validation 选参已接入；未公开参数及长周期 GPU 训练待完成 |

## 2. 论文信息

| 字段 | 内容 |
|---|---|
| Authors | Botai Yuan, Chen Gong, Dacheng Tao, Jie Yang |
| Venue | IEEE Transactions on Neural Networks and Learning Systems |
| Volume/Issue | 36(6) |
| Pages | 10515-10529 |
| Year | 2025 |
| DOI | `10.1109/TNNLS.2025.3530427` |
| 作者公开 PDF | `yuan_tnnls25.pdf` |
| 官方源码 | 未发现公开仓库 |
| Source status | `not_found` |

## 3. 问题设定与符号

```math
\mathcal X_L=\{x_i\}_{i=1}^{n_P}\sim p_P(x),
\qquad
\mathcal X_U=\{x_i\}_{i=1}^{n_U}\sim p(x),
```

```math
p(x)=\pi_Pp_P(x)+(1-\pi_P)p_N(x).
```

`o_i=1` 表示 labeled positive，`o_i=0` 表示 unlabeled。分类器
`f(x)` 输出两类 logits/softmax，query encoder 与 key encoder 分别记为
`g_q`、`g_k`。

## 4. 整体结构

WConPU 在每个训练迭代中协同执行：

1. 对输入生成 weak query view 和 strong key view；
2. 由 query/key encoder 生成归一化 embedding；
3. 使用 classifier 预测与 SAT 构造 positive peer set；
4. 由 prototype 划分类别并挖掘 hard negatives；
5. 计算 weighted contrastive loss；
6. 根据 prototype 更新 U 样本 soft pseudo-label；
7. 计算分类损失和标签分布对齐损失；
8. 更新 query 网络、classifier、prototype、key encoder 与 queue。

论文从 EM 角度解释其交替过程：classifier/pseudo-label 更新近似 E-step，
contrastive representation 更新近似 M-step，但这不意味着非凸训练具有全局收敛保证。

## 5. Query/Key Encoder 与 Momentum Queue

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

## 6. Self-Adaptive Threshold

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

## 7. Prototype 与伪标签

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

## 8. Prototype-Based Hard Negative Mining

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

实现必须设置数值下界 `eps`，防止 embedding 极近时除零。

## 9. Weighted Contrastive Loss

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
再对全体样本平均得到 `L_con`。如果某个 anchor 没有可靠 positive peer，应跳过该
anchor，而不是制造空集合 NaN。

## 10. 分类损失

soft pseudo-label cross entropy：

```math
\mathcal L_{\mathrm{class}}
=
-\frac1n
\sum_i\sum_{c=0}^1s_{i,c}\log z_{i,c}.
```

该损失使 prototype 产生的语义信号反向促进 classifier；classifier 的置信预测又决定
下一轮 positive peer set。

## 11. 标签分布对齐

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

这说明 WConPU 实际上需要 `pi_P`。仓库旧元数据中的
`requires_class_prior=False` 必须修正。

## 12. 总目标

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

## 13. 工具箱 API

```python
WeightedContrastivePUClassifier(
    class_prior,
    encoder=None,
    hidden_dim=128,
    embedding_dim=128,
    queue_size=8192,
    temperature=0.07,
    momentum=0.999,
    pseudo_label_momentum=0.9,
    contrastive_weight=0.1,
    distribution_weight=0.1,
    hard_negative_quantile=0.25,
    weak_augmentation=None,
    strong_augmentation=None,
    batch_size=256,
    max_epochs=100,
    learning_rate=1e-2,
    optimizer_momentum=0.9,
    scheduler="none",
    random_state=None,
    device=None,
)
```

### 13.1 数据协议

- `fit(X, y_pu, class_prior=None)`；
- `X` 默认是二维 feature matrix；传入 encoder 时也接受 NCHW 图像；
- 图像级协议可由调用方传入 encoder/augmentation，或由 runner 的结构化 `vision` 配置构造；
- 默认增强只用于 tabular smoke test，不等价于论文 SimAugment/RandAugment；
- `predict`、`decision_function`、`predict_proba` 使用 query classifier。

### 13.2 视觉构建器

```python
from pu_toolbox.estimators.deep import (
    build_wconpu_augmentation,
    build_wconpu_backbone,
)

encoder = build_wconpu_backbone(
    "cnn13",  # 也支持 resnet18 / resnet50
    normalization_mean=(0.4914, 0.4822, 0.4465),
    normalization_std=(0.2470, 0.2435, 0.2616),
)
weak = build_wconpu_augmentation("simaugment", image_size=32)
strong = build_wconpu_augmentation(
    "randaugment",
    image_size=32,
    randaugment_num_ops=2,
    randaugment_magnitude=10,
)
```

- `cnn13` 是 13 个 convolution layer、三次 max-pooling 和 global average pooling 的
  clean-room adapter；论文没有公开逐层结构，不能称为精确官方网络。
- ResNet-18/50 使用 torchvision topology、随机初始化和无分类头 feature 输出；
  `small_input_stem=True` 可用于小图像。
- SimAugment 采用 supervised contrastive 常用的 random resized crop、flip、color jitter、
  grayscale 组合；RandAugment 使用 crop/flip 后的 torchvision RandAugment。
- 每张图像独立采样增强；channel normalization 位于 encoder 内，因此训练和推理一致。

### 13.3 拟合属性

- `model_`、`key_encoder_`；
- `prototypes_`；
- `pseudo_labels_`；
- `queue_embeddings_`、`queue_labels_`；
- `sat_global_`、`sat_classwise_`；
- `history_`；
- `class_prior_`。

## 14. 论文实验设置

数据集：CIFAR-10、SVHN、STL-10、Alzheimer MRI。

主要配置：

- CIFAR-10/SVHN 使用 13-layer CNN，STL-10 使用 ResNet-18，Alzheimer 使用 ResNet-50；
- projection head 为 2-layer MLP，输出 128 维；
- weak augmentation 为 SimAugment，strong augmentation 为 RandAugment；
- queue size：Alzheimer 4096，其余 8192；
- `alpha=0.9`、`lambda=0.999`、`rho=0.07`；
- SGD momentum 0.9，初始学习率 `1e-2`，cosine annealing；
- batch size：Alzheimer 8，其余 256；
- `gamma_0`、`gamma_1` 从 `{1e-3,1e-2,1e-1,1}` 网格选择；
- 训练 800 epoch，不使用 early stopping（论文协议；工具箱默认 100 epoch，
  已实测 50 epoch 即收敛，可用 `--max-epochs` 调整）；
- 每组独立运行 5 次并报告 6 项分类指标。

项目已在 `benchmarks/deep_pu/` 提供统一 runner、锁定论文配置、3-seed 表格合成结果和
CIFAR-10 visual paper-protocol。runner 已支持 NCHW 数据、13-layer CNN/ResNet-18/50、
SimAugment/RandAugment、cosine annealing，并将 anchor-wise contrastive loss 向量化。
`official_data_wconpu_cifar10_protocol.json` 还锁定了 clean validation 模型选择协议：每个
seed 先从 CIFAR-10 canonical training split 隔离 10% 带真值验证集，再从其余样本构造
`1000 P + 44000 U` 的训练集；验证样本不会传入 estimator 的 `fit`。runner 对
`gamma_0`/`gamma_1` 的 `4 x 4` Cartesian grid 逐候选训练和评估，将中间结果写入
`model_selection.csv`，采用稳定的候选顺序打破并列，并以最优参数从头 refit 最终模型。
候选级结果支持断点续跑。

论文没有说明 clean validation 的选择指标，当前配置明确、暂定使用 accuracy；论文报告的
`nP=1000`、`nU=50000` 与额外保留 10% validation 在 50000 张训练图像上不能同时保持互斥，
因此本协议采用可审计的互斥计数并记录差异。配置尚未执行 5-seed × 800 epoch，不能标记为
论文结果。

## 15. 测试与验收

- dissimilarity 范围为 `[0,1]`；
- hard-negative 权重有限且随 dissimilarity 减小而增大；
- prototype 和 queue embeddings 保持 L2 归一化；
- pseudo-label 每行和为 1，labeled positives 始终为 `[0,1]`；
- queue 严格遵守最大容量和 FIFO；
- SAT 状态在 `[0,1]`；
- 三项 loss 均为有限标量；
- fixed seed 可复现；
- 三种视觉 backbone 返回二维 feature，增强保持 NCHW shape 且固定 seed 可复现；
- NCHW 图像可完成一轮 WConPU 训练，cosine scheduler 状态正确；
- clean validation 与 PU 训练索引严格不重叠，并保留真实二分类标签；
- 二维参数网格按确定顺序展开，候选逐项落盘、可恢复且最优参数会重新训练；
- 模型选择期间及最终 refit 均不把 clean validation 传入 `fit`；
- registry 元数据要求 class prior；
- 合成 PU 数据上可完成 fit/predict。

## 16. 局限与复现风险

- 对比学习结果高度依赖 augmentation，论文没有公开 SimAugment/RandAugment 参数；
- 论文只给出“13-layer CNN”名称，没有逐层 topology；当前适配器属于 clean-room；
- 论文未公开 clean validation 的选择指标；accuracy 是显式、可替换的暂定选择；
- 原文训练样本计数与 10% clean validation 存在互斥性歧义；当前协议优先保证无泄漏；
- 默认 tabular augmentation 仅用于接口验证；
- early-stage classifier 错误会污染 peer set 和 prototype；
- hard-negative inverse distance 可能产生大梯度，必须做 `eps`/clamp；
- 论文未提供公开源码，当前实现是 clean-room；
- 完整 800-epoch 视觉 benchmark 成本高，不应纳入普通 CI。

## 17. 参考资料

1. Yuan et al. *Weighted Contrastive Learning with Hard Negative Mining for Positive and Unlabeled Learning*. IEEE TNNLS, 2025.
2. DOI: <https://doi.org/10.1109/TNNLS.2025.3530427>
3. 作者 PDF：<https://gcatnjust.github.io/ChenGong/paper/yuan_tnnls25.pdf>
