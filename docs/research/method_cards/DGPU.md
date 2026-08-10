# Method Card: DGPU

## 1. 方法定位与当前状态

| 项目 | 内容 |
|---|---|
| 全称 | Discriminative-Generative Positive and Unlabeled Learning |
| 工具箱注册名 | `dgpu` |
| 方法类型 | 判别式 PU + 条件扩散生成的协同迭代 |
| 生成模型 | 论文采用 EDM 条件扩散模型 |
| 初始化分类器 | Dist-PU |
| 类先验 | 必需 |
| 当前实现 | `DGPUClassifier` 原生训练编排、判别式损失和 generator protocol |
| 完整论文复现 | 需要外部 EDM 训练后端与图像数据流水线 |

## 2. 论文信息

| 字段 | 内容 |
|---|---|
| Authors | Botai Yuan, Chen Gong, Dacheng Tao, Jie Yang |
| Venue | IEEE Transactions on Image Processing |
| Volume | 35 |
| Pages | 2969-2980 |
| Year | 2026 |
| DOI | `10.1109/TIP.2026.3672381` |
| Received | 2025-08-12 |
| Published | 2026-03-16 |
| 作者公开 PDF | `yuan_tip26.pdf` |
| 官方源码 | 未发现公开仓库 |
| Source status | `not_found` |

仓库原先把 DGPU 作为未发表占位方法。正式论文现已可用，因此 Method Card 和元数据
应以 2026 TIP 版本为准。

## 3. 问题设定

```math
\mathcal X_P
=
\{(x_i,y_i=1):x_i\sim p_P(x)\}_{i=1}^{n_P},
```

```math
\mathcal X_U
=
\{x_i:x_i\sim p(x)\}_{i=1}^{n_U},
```

```math
p(x)=\pi_Pp_P(x)+(1-\pi_P)p_N(x).
```

论文采用 case-control 设定并在实验中假设 `pi_P` 已知。

## 4. 总体训练闭环

DGPU 由两个交替阶段组成：

```text
Dist-PU 初始化 classifier
  -> 从 U 构造 class-balanced pseudo-labeled set
  -> 训练/微调 conditional diffusion model
  -> 按 class prior 生成正负样本
  -> weighted supervised + debiased SSL 更新 classifier
  -> 重新构造 pseudo-labeled set
  -> 下一轮生成
```

关键点：

- classifier 在各轮之间增量更新，不重新初始化；
- diffusion model 接收不断改进的 pseudo labels；
- synthetic data 质量通过前一轮 classifier 置信度加权；
- 生成和判别两个阶段必须形成反馈闭环。

## 5. 初始化：Dist-PU

弱增强记为 `A_w`，分类器 softmax 输出：

```math
z_i=\mathrm{softmax}(f(A_w(x_i))).
```

初始化目标：

```math
\mathcal L_{\mathrm{align}}
=
2\pi_P
\left|
\frac1{n_P}\sum_{x_i\in\mathcal X_P}z_i^1-1
\right|
+
\left|
\frac1{n_U}\sum_{x_i\in\mathcal X_U}z_i^1-\pi_P
\right|.
```

该阶段得到初始判别模型，并用于从 U 中产生第一轮 pseudo-labeled set。

## 6. 条件扩散生成

### 6.1 Forward process

```math
q(x_t\mid x_0)
=
\mathcal N
\left(
x_t;\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I
\right),
```

```math
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
```

### 6.2 Reverse process

```math
p_\theta(x_{t-1}\mid x_t)
=
\mathcal N
\left(
x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t)
\right).
```

DDPM noise-prediction objective：

```math
\mathcal L_{\mathrm{diff}}
=
\mathbb E_{x_0,t,\epsilon}
\left[
\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2
\right].
```

### 6.3 Classifier-Free Guidance

条件与无条件噪声预测组合：

```math
\widetilde\epsilon_\theta(x_t,y,t)
=
(1+s)\epsilon_\theta(x_t,y,t)
-s\epsilon_\theta(x_t,t),
```

其中 `s` 为 guidance strength。论文实际采用 EDM 以减少每轮生成的采样成本。

## 7. Generated Set

生成集合：

```math
\mathcal X_G
=
\bigcup_{y\in\{0,1\}}
\{(\widehat x_{i,y},y)\}_{i=1}^{K_y}.
```

为匹配真实类别先验：

```math
K_1=n_G\pi_P,
\qquad
K_0=n_G(1-\pi_P).
```

实现中需要定义整数舍入策略，并保证 `K_0+K_1=n_G`。

## 8. Weighted Supervised Loss

增广 labeled set：

```math
\widetilde{\mathcal X}_L=\mathcal X_P\cup\mathcal X_G.
```

以前一阶段分类器 `\widehat f` 的置信度定义样本权重：

```math
\omega_i
=
\left[
\max
\mathrm{softmax}
(\widehat f(A_w(x_i)))
\right]^{1/2}.
```

监督损失：

```math
\mathcal L_s
=
\frac1{|\widetilde{\mathcal X}_L|}
\sum_{x_i\in\widetilde{\mathcal X}_L}
\omega_i H(y_i,z_i).
```

平方根提供次线性缩放，降低低质量 synthetic samples 的影响，同时避免权重过度尖锐。

## 9. Debiased Pseudo Labeling

### 9.1 Prediction-distribution EMA

```math
\eta
\leftarrow
m\eta+(1-m)\frac1b\sum_{i=1}^{b}z_i.
```

论文 PDF 的排版可能省略均值中的 `/b`，但 `eta` 必须保持类别概率分布语义，工程实现
应显式取 batch mean 并归一化。

### 9.2 Debiased logits

```math
\widetilde f_i
=
f(A_w(x_i))-\lambda\log\eta.
```

以 `softmax(\widetilde f_i)` 进行 confidence thresholding：

```math
\mathbf 1[\max(\widetilde z_i)>\tau].
```

### 9.3 Debiased marginal loss

强增强输出记为：

```math
Z_i=\mathrm{softmax}(f(A_s(x_i))).
```

论文定义：

```math
\mathcal L_{\mathrm{DML}}(\widehat y_i,Z_i)
=
-\log
\frac{
\exp(Z_i^{\widehat y_i}+\lambda\log\eta_{\widehat y_i})
}{
\sum_{k=0}^1
\exp(Z_i^k+\lambda\log\eta_k)
}.
```

需要注意：论文公式直接对概率 `Z_i` 再做 exponential，而不是把它写成 logits。
clean-room 实现应忠实保留此定义，并在代码中清楚命名，避免与普通 cross entropy 混淆。

### 9.4 Unsupervised loss

```math
\widetilde{\mathcal L}_u
=
\frac1{|\mathcal X_U|}
\sum_{x_i\in\mathcal X_U}
\mathbf 1[\max(\widetilde z_i)>\tau]
\mathcal L_{\mathrm{DML}}(\widehat y_i,Z_i).
```

判别阶段总损失：

```math
\mathcal L_{\mathrm{dis}}
=
\mathcal L_s+\widetilde{\mathcal L}_u.
```

## 10. Pseudo-Labeled Set 更新

对未标记样本，入选概率与 classifier confidence 成正比：

```math
p(o_i=1)\propto\max(z_i).
```

论文每类选择固定数量 `N`：

```math
\mathcal X_L
=
\bigcup_{\widehat y_i\in\{0,1\}}
\{(x_i,\widehat y_i):x_i\in\mathcal X_U,o_i=1\}_{i=1}^{N}.
```

与固定 confidence threshold 不同，概率采样使低置信样本仍有非零参与机会，意在减轻
生成数据偏置和 mode collapse。工程实现应：

- 分类别采样；
- 无放回；
- 使用置信度归一化为 sampling probability；
- 当某类候选少于 `N` 时安全截断；
- 记录每轮两类实际样本数量。

## 11. Generator Protocol

工具箱不内置大型 EDM 训练代码，而定义最小协议：

```python
class ConditionalGeneratorProtocol:
    def fit(self, X, y, *, warm_start=True):
        ...

    def sample(self, n_samples, *, class_label, random_state=None):
        ...
```

`DGPUClassifier` 负责编排迭代，用户负责传入满足协议的 conditional diffusion backend。
测试可使用轻量 mock generator；论文复现应接入 EDM。

未传 generator 时接口必须明确报错，不能静默退化成 Gaussian jitter 后仍宣称是 DGPU。

## 12. 工具箱 API

```python
DGPUClassifier(
    class_prior,
    generator,
    model=None,
    rounds=3,
    initialization_epochs=100,
    annotation_epochs=100,
    generated_samples=5000,
    pseudo_label_fraction=0.1,
    confidence_threshold=0.95,
    debias_strength=0.8,
    distribution_momentum=0.999,
    batch_size=256,
    learning_rate=1e-4,
    weak_augmentation=None,
    strong_augmentation=None,
    random_state=None,
    device=None,
)
```

拟合属性：

- `model_`；
- `generator_`；
- `class_prior_`；
- `predicted_distribution_`；
- `pseudo_labeled_indices_`；
- `generated_counts_`；
- `history_`；
- `classes_`。

## 13. 论文实验协议

数据集：

- Fashion-MNIST：tops 为 positive；
- CIFAR-10：vehicles 为 positive；
- CelebA：male 为 positive。

主要配置：

- 已知 `pi_P`；
- weak augmentation：random crop + horizontal flip；
- strong augmentation：RandAugment；
- `lambda=0.8`；
- confidence threshold `tau=0.95`；
- 每轮每类 pseudo-labeled 数量为 U 的 10%；
- classifier 使用 Adam，学习率 `1e-4`，batch size 256；
- EDM batch size 512，学习率 `1e-3`；
- 每个数据集运行 3 次并报告 test accuracy mean/std；
- 论文还评估 instance-dependent labeling：

```math
p(o=1\mid x,y=1)
\propto
p(y=1\mid x)^{10}.
```

项目已在 `benchmarks/deep_pu/` 提供统一 runner、锁定论文配置和 3-seed 合成
clean-room 结果。runner 使用可复现 Gaussian conditional generator 验证 DGPU 编排，
不使用 EDM，也不构成 Fashion-MNIST/CIFAR-10/CelebA paper-like 结果。

## 14. 测试与验收

### 14.1 核心损失

- confidence weight 落在 `[sqrt(0.5),1]`；
- `eta` 始终有限、为正且和为 1；
- debiased logits 与手工计算一致；
- DML 与 unsupervised mask 可手工核对；
- 没有样本超过 threshold 时 unsupervised loss 为零而非 NaN。

### 14.2 生成编排

- 每轮调用 generator 的 `fit` 和两个类别的 `sample`；
- 生成正负数量符合 class prior；
- classifier 参数跨轮 warm start；
- pseudo-label 每类采样无重复；
- fixed seed 时采样可复现。

### 14.3 API

- generator 缺失或协议不完整时给出明确错误；
- `fit/predict/decision_function/predict_proba` 契约通过；
- registry 要求 `class_prior=True`；
- metadata 标记为 experimental、source `not_found`。

## 15. 局限与复现风险

- DGPU 的计算瓶颈是每轮训练/微调条件扩散模型；
- 生成质量直接影响判别阶段，普通 CI 无法验证视觉 fidelity；
- paper-like 结果依赖 EDM 实现、数据增强和大规模 GPU；
- pseudo-label 与 synthetic error 可能形成反馈放大；
- 论文于 2026 年刚发表，目前未发现官方代码；
- 只有 mock generator 的测试证明编排正确，不证明完整论文指标。

## 16. 参考资料

1. Yuan et al. *Discriminative-Generative Positive and Unlabeled Learning*. IEEE TIP 35, 2026.
2. DOI: <https://doi.org/10.1109/TIP.2026.3672381>
3. 作者 PDF：<https://gcatnjust.github.io/ChenGong/paper/yuan_tip26.pdf>
