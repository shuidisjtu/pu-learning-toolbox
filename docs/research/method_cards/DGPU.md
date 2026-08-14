# Method Card: DGPU

## 1. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Discriminative-Generative Positive and Unlabeled Learning |
| Authors | Botai Yuan, Chen Gong, Dacheng Tao, Jie Yang |
| Venue | IEEE Transactions on Image Processing |
| Volume | 35 |
| Pages | 2969-2980 |
| Year | 2026 |
| DOI | `10.1109/TIP.2026.3672381` |
| Received | 2025-08-12 |
| Published | 2026-03-16 |
| 作者公开 PDF | `yuan_tip26.pdf` |
| 方法类型 | 判别式 PU + 条件扩散生成的协同迭代 |
| 生成模型 | 论文采用 EDM 条件扩散模型 |
| 初始化分类器 | Dist-PU |
| 类先验 | 必需 |
| 官方源码 | 未发现公开仓库 |
| Source status | `not_found` |

### Assumptions

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

---

## 2. 问题设定与符号

| 论文符号 | 含义 |
|---|---|
| $\`\mathcal X_P\`$ / $\`\mathcal X_U\`$ | 可靠正样本集 / 未标记集 |
| $\`\pi_P\`$ | 正类先验（实验假设已知） |
| $\`A_w\`$ / $\`A_s\`$ | 弱增强（random crop + flip）/ 强增强（RandAugment） |
| $\`f(x)\`$ / $\`z_i\`$ | 分类器 / softmax 输出 |
| $\`\mathcal X_G\`$ / $\`K_1\`$ / $\`K_0\`$ | 生成集合 / 生成正样本数 / 生成负样本数 |
| $\`\omega_i\`$ | 样本置信度权重 |
| $\`\eta\`$ | 类别预测分布 EMA |
| $\`\lambda\`$ | debias strength |
| $\`\tau\`$ | confidence threshold |
| $\`\widetilde f_i\`$ / $\`Z_i\`$ | debiased logits / 强增强 softmax 输出 |
| $\`o_i\`$ | 样本是否入选 pseudo-labeled set |
| $\`s\`$ | CFG guidance strength |

---

## 3. 核心公式

### 3.1 初始化：Dist-PU

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

### 3.2 条件扩散生成

Forward process：

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

Reverse process：

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

Classifier-Free Guidance：

```math
\widetilde\epsilon_\theta(x_t,y,t)
=
(1+s)\epsilon_\theta(x_t,y,t)
-s\epsilon_\theta(x_t,t),
```

其中 `s` 为 guidance strength。论文实际采用 EDM 以减少每轮生成的采样成本。

### 3.3 Generated Set

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

### 3.4 Weighted Supervised Loss

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

### 3.5 Debiased Pseudo Labeling

Prediction-distribution EMA：

```math
\eta
\leftarrow
m\eta+(1-m)\frac1b\sum_{i=1}^{b}z_i.
```

Debiased logits：

```math
\widetilde f_i
=
f(A_w(x_i))-\lambda\log\eta.
```

以 `softmax(\widetilde f_i)` 进行 confidence thresholding：

```math
\mathbf 1[\max(\widetilde z_i)>\tau].
```

Debiased marginal loss（强增强输出 `Z_i=\mathrm{softmax}(f(A_s(x_i)))`）：

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

Unsupervised loss：

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

### 3.6 Pseudo-Labeled Set 更新

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
生成数据偏置和 mode collapse。

---

## 4. 算法概要

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

---

## 5. 论文边界

- DGPU 假设 `pi_P` 已知（case-control 设定），不是类先验估计器。
- DGPU 的计算瓶颈是每轮训练/微调条件扩散模型。
- 生成质量直接影响判别阶段，普通 CI 无法验证视觉 fidelity。
- paper-like 结果依赖 EDM 实现、数据增强和大规模 GPU。
- pseudo-label 与 synthetic error 可能形成反馈放大。
- 论文于 2026 年刚发表，目前未发现官方代码。

---

## 6. 实现注记

- **论文版本基线**：仓库原先把 DGPU 作为未发表占位方法。正式论文现已可用，因此
  Method Card 和元数据应以 2026 TIP 版本为准。
- **EMA 归一化（关键）**：论文 PDF 的排版可能省略均值中的 `/b`，但 `eta` 必须保持
  类别概率分布语义，工程实现应显式取 batch mean 并归一化。
- **DML 忠实保留（关键）**：论文公式直接对概率 `Z_i` 再做 exponential，而不是把它
  写成 logits。clean-room 实现应忠实保留此定义，并在代码中清楚命名，避免与普通
  cross entropy 混淆。
- **generator 缺失必须报错（关键）**：未传 generator 时接口必须明确报错，不能静默
  退化成 Gaussian jitter 后仍宣称是 DGPU。
- **Generated Set 整数舍入**：实现中需要定义整数舍入策略，并保证 `K_0+K_1=n_G`。
- **Pseudo-Labeled Set 采样**：工程实现应分类别采样、无放回、使用置信度归一化为
  sampling probability；当某类候选少于 `N` 时安全截断；记录每轮两类实际样本数量。

---

## 7. 论文实验参考

数据集：

- Fashion-MNIST：tops 为 positive；
- CIFAR-10：vehicles 为 positive；
- CelebA：male 为 positive。

主要配置：

| 项目 | 论文设置 |
|---|---|
| 先验 | 已知 `pi_P` |
| 增强 | weak：random crop + horizontal flip；strong：RandAugment |
| 超参数 | `lambda=0.8`；confidence threshold `tau=0.95` |
| pseudo-labeled 规模 | 每轮每类为 U 的 10% |
| classifier | Adam，学习率 `1e-4`，batch size 256 |
| EDM | batch size 512，学习率 `1e-3` |
| 指标协议 | 每个数据集运行 3 次并报告 test accuracy mean/std |
| instance-dependent labeling | 论文还评估 $`p(o=1\mid x,y=1)\propto p(y=1\mid x)^{10}`$ |

---

## 8. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `not_found`：未发现公开仓库 |
| 实现状态 | `DGPUClassifier` 原生训练编排、判别式损失和 generator protocol；完整论文复现需要外部 EDM 训练后端与图像数据流水线 |
| runner 状态 | `benchmarks/deep_pu/` 提供统一 runner、锁定论文配置和 3-seed 合成 clean-room 结果；runner 使用可复现 Gaussian conditional generator 验证 DGPU 编排，不使用 EDM，也不构成 Fashion-MNIST/CIFAR-10/CelebA paper-like 结果 |
| 复现风险 | 计算瓶颈是每轮训练/微调条件扩散模型；生成质量直接影响判别阶段；只有 mock generator 的测试证明编排正确，不证明完整论文指标 |

参考资料：

1. Yuan et al. *Discriminative-Generative Positive and Unlabeled Learning*. IEEE TIP 35, 2026.
2. DOI: <https://doi.org/10.1109/TIP.2026.3672381>
3. 作者 PDF：<https://gcatnjust.github.io/ChenGong/paper/yuan_tip26.pdf>
