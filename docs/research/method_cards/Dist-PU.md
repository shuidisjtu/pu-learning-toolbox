# Method Card: Dist-PU

## 1. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Dist-PU: Positive-Unlabeled Learning from a Label Distribution Perspective |
| Authors | Yunrui Zhao, Qianqian Xu, Yangbangyan Jiang, Peisong Wen, Qingming Huang |
| Venue | CVPR 2022 |
| Year | 2022 |
| Family | `risk_estimation` / deep PU |
| Scenario | `case_control` |
| Requires class prior | `True` |
| Requires propensity | `False` |
| Requires negative samples | `False` |
| GPU required | 论文实验需要；项目接口通过可选 PyTorch 支持 CPU/GPU |
| Official source | [Dist-PU GitHub repository](https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective) |
| Paper source | [arXiv:2212.02801](https://arxiv.org/abs/2212.02801) |

### Assumptions

令 `Y in {0,1}`，正类先验为 `pi=P(Y=1)`。可靠正样本和未标记样本分别满足：

```math
X_P \sim p_P(x)=p(x\mid Y=1),
\qquad
X_U \sim p(x)=\pi p_P(x)+(1-\pi)p_N(x).
```

模型输出 logit `f_theta(x)` 和 soft prediction：

```math
q_\theta(x)=\mathrm{sigmoid}(f_\theta(x)).
```

论文的分布对齐直觉为：

```math
E_{x\sim p_P}[q_\theta(x)]\approx 1,
\qquad
E_{x\sim p}[q_\theta(x)]\approx \pi.
```

---

## 2. 问题设定与符号

论文摘要明确指出：传统 cost-sensitive PU 方法显式优化把未标记样本判为负类的风险，柔性模型可能因此出现负类预测偏好；Dist-PU 改为对齐预测标签分布的期望，并加入熵最小化和 Mixup。上述三项组成是本方法卡的算法范围。

### 2.1 数据假设

| 假设 | 论文需要 |
|---|---|
| P 的标签可靠 | 是 |
| U 是边缘分布 | 是 |
| `pi` 已知 | 是 |
| 负类标签可用 | 否 |
| 模型可微 | 是 |

| 论文符号 | 含义 |
|---|---|
| `P` | labeled positive set |
| `U` | unlabeled marginal set |
| `pi` | positive class prior |
| `f_theta(x)` | classifier logit |
| `q_theta(x)` | sigmoid prediction |
| `R_P` | positive prediction loss |
| `R_lab` | label distribution alignment loss |
| `R_ent` | entropy minimization term |
| `R_mix` | Mixup consistency term |
| `mu` | entropy weight |
| `nu` | Mixup strength/weight |
| `alpha` | Beta distribution 参数 |

---

## 3. 核心公式

### 3.1 正样本监督项

对正样本施加正类监督：

```math
\widehat R_P(\theta)
=-
\frac{1}{n_P}\sum_{i=1}^{n_P}
\log q_\theta(x_i^P).
```

该项防止模型只通过满足未标记集均值来得到任意的常数输出。

### 3.2 Label distribution alignment

论文将未标记数据的真实标签分布期望固定为 `pi`，从而使用：

```math
\widehat R_{lab}(\theta)
=
\left(
\frac{1}{n_U}\sum_{j=1}^{n_U}q_\theta(x_j^U)-\pi
\right)^2.
```

### 3.3 熵最小化

对每个未标记样本的二元预测分布：

```math
H(q)=-q\log q-(1-q)\log(1-q).
```

论文使用未标记样本熵的平均值：

```math
\widehat R_{ent}(\theta)
=\frac{1}{n_U}\sum_{j=1}^{n_U}H(q_\theta(x_j^U)).
```

最小化该项会鼓励输出接近 0 或 1，缓解均值对齐的常数解；权重过大时会放大确认偏差。

### 3.4 Mixup 一致性

从样本对 `(x_i, x_j)` 及其当前 soft label `(q_i, q_j)` 构造：

```math
\tilde x=\lambda x_i+(1-\lambda)x_j,
\qquad
\tilde q=\lambda q_i+(1-\lambda)q_j,
```

其中 `lambda` 通常来自 Beta 分布。Mixup 训练项为：

```math
\widehat R_{mix}(\theta)
=\mathrm{BCE}
\left(f_\theta(\tilde x),\tilde q\right).
```

### 3.5 总目标

```math
\widehat J(\theta)=
\widehat R_P(\theta)
+\lambda_{lab}\widehat R_{lab}(\theta)
+\mu\widehat R_{ent}(\theta)
+\nu\widehat R_{mix}(\theta).
```

---

## 4. 算法概要

```text
输入：P、U、先验 pi、网络 f_theta、训练轮数 T

1. 校验 P/U 非空，校验 0 < pi < 1。
2. 初始化 MLP 和 Adam optimizer。
3. 每轮训练：
   a. 计算所有 P/U 的 logits 和 sigmoid outputs；
   b. 计算 P 上的正类 BCE；
   c. 计算 U 上的均值对齐损失 (mean(q_U)-pi)^2；
   d. 计算 U 上的熵最小化损失；
   e. 对样本对进行 Mixup，计算 soft-target BCE；
   f. 加权求和并反向传播。
4. 保存训练记录与模型参数。
5. 用 sigmoid(logit) 输出预测概率，用 logit 的符号输出预测标签。
```

---

## 5. 论文边界

- Dist-PU 不是一个类先验估计器；`pi` 必须由用户提供，或在训练外部先通过 `ReCPE`/其他 CPE 得到。
- `U` 来自边缘分布 `p(x)`。如果 `U` 是人为筛选、时间漂移或选择偏置样本，均值约束 `E_U[q(x)] = pi` 不再直接对应论文设定。
- 仅使用标签分布对齐会产生平凡解：所有未标记样本都输出 `pi`。熵最小化用于推动预测远离常数解，Mixup 用于缓解确认偏差。
- 预测概率是模型 sigmoid 输出，不自动等同于校准的 `P(y=1|x)`；论文核心约束是分布期望，而非概率校准定理。

---

## 6. 实现注记

- 当前项目实现使用全量张量训练，`batch_size` 是兼容性参数，尚未复现论文的 batch-level training protocol。
- 当前实现使用小型 MLP，不应直接与论文的图像 backbone 结果比较。
- 当前实现使用随机 `lambda`，并对构造 soft target 使用 stop-gradient；这是工程侧简化，尚未暴露论文中的完整 `alpha/gamma/nu` 超参数协议。
- 数值稳定性：
  - logits 在熵计算前限制在 `[-10, 10]`，避免 `log(0)` 和指数溢出；
  - sigmoid 输出的熵计算增加 `1e-6` epsilon；
  - 推理时 sigmoid 输入限制在 `[-40, 40]`；
  - `random_state` 只控制 PyTorch 初始化和 Mixup 随机性；GPU 上的完全确定性仍需额外配置。

结果记录（benchmark 数字与官方配置 commit 锁以 benchmarks 产物为准）：

> 结果见 `benchmarks/assigned_methods/results/clean_room_multiseed/`（当前 clean-room 运行结果）；官方配置 commit 锁见 `benchmarks/assigned_methods/configs/official_sources.lock.json`

---

## 7. 论文实验参考

### 7.1 超参数（论文对应）

| 参数 | 论文对应 |
|---|---|
| `class_prior` | `pi_P` |
| `alignment_weight` | 论文 `R_lab` 权重 |
| `entropy_weight` | 论文 `mu` |
| `mixup_weight` | 论文 `nu` 的工程化参数 |
| `learning_rate` | 参考论文训练习惯 |

### 7.2 论文实验

论文实验使用 Fashion-MNIST、CIFAR-10 与 Alzheimer 三个 benchmark，官方协议含 backbone、mini-batch、数据增强与两阶段训练；当前实现仅覆盖 clean-room 档，不得与论文图像 backbone 结果直接比较。

---

## 8. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `official_exact` |
| Implementation status | `NATIVE`，PyTorch optional dependency |
| 当前实现 | 论文核心损失的 clean-room 小型 MLP 版本 |
| 尚未对齐 | 官方 backbone、mini-batch、图像增强、数据 split、完整超参数搜索 |
| 主要风险 | `pi` 错误会直接改变 U 集分布约束；熵权重过大可能强化确认偏差；全量训练与官方 batch 训练结果不可直接比较 |
