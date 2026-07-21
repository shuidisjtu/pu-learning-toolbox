# Method Card: Dist-PU

## 1. 待办与注意

### 1.1 待办

- [x] 明确 Dist-PU 的输入协议：可靠正样本 `P`、边缘分布未标记样本 `U` 和已知正类先验 `pi`。
- [x] 实现标签分布对齐项：正样本预测期望接近 1，未标记样本预测期望接近 `pi`。
- [x] 实现未标记集熵最小化和 Mixup 正则项。
- [x] 接入 `BasePUClassifier`、registry 和可选 PyTorch 依赖。
- [x] 编写小型合成数据上的 fit/predict、有限值和 sklearn-style 参数测试。
- [ ] 按官方仓库复现 Fashion-MNIST、CIFAR-10 和 Alzheimer benchmark。
- [ ] 补充真正的 mini-batch、数据增强和论文 backbone 配置。

### 1.2 注意

- Dist-PU 不是一个类先验估计器；`pi` 必须由用户提供，或在训练外部先通过 `ReCPE`/其他 CPE 得到。
- `U` 来自边缘分布 `p(x)`。如果 `U` 是人为筛选、时间漂移或选择偏置样本，均值约束 `E_U[q(x)] = pi` 不再直接对应论文设定。
- 仅使用标签分布对齐会产生平凡解：所有未标记样本都输出 `pi`。熵最小化用于推动预测远离常数解，Mixup 用于缓解确认偏差。
- 预测概率是模型 sigmoid 输出，不自动等同于校准的 `P(y=1|x)`；论文核心约束是分布期望，而非概率校准定理。
- 当前项目实现使用全量张量训练，`batch_size` 是兼容性参数，尚未复现论文的 batch-level training protocol。
- 当前实现使用小型 MLP，不应直接与论文的图像 backbone 结果比较。

## 2. 论文信息

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

论文摘要明确指出：传统 cost-sensitive PU 方法显式优化把未标记样本判为负类的风险，柔性模型可能因此出现负类预测偏好；Dist-PU 改为对齐预测标签分布的期望，并加入熵最小化和 Mixup。上述三项组成是本方法卡的算法范围。

## 3. 问题设定与假设

令 `Y in {0,1}`，正类先验为 `pi=P(Y=1)`。可靠正样本和未标记样本分别满足：

```math
X_P \sim p_P(x)=p(x\mid Y=1),
\qquad
X_U \sim p(x)=\pi p_P(x)+(1-\pi)p_N(x).
```

模型输出 logit `f_theta(x)` 和 soft prediction：

```math
q_\theta(x)=\operatorname{sigmoid}(f_\theta(x)).
```

论文的分布对齐直觉为：

```math
E_{x\sim p_P}[q_\theta(x)]\approx 1,
\qquad
E_{x\sim p}[q_\theta(x)]\approx \pi.
```

项目实现中，`p_P` 的期望用 labeled positive 样本均值估计，`p` 的期望用 unlabeled 样本均值估计。

### 3.1 数据假设检查

| 假设 | 论文需要 | 项目处理 |
|---|---|---|
| P 的标签可靠 | 是 | `y_pu=1` 视为可靠正例 |
| U 是边缘分布 | 是 | `y_pu=0` 视为 `p(x)` 样本 |
| `pi` 已知 | 是 | 构造函数必填，可由 `fit` 覆盖 |
| 负类标签可用 | 否 | 不允许把 U 当作真实负类标签 |
| 模型可微 | 是 | 当前实现为 PyTorch MLP |

## 4. 符号与记号

| 论文符号 | 含义 | 开发侧对应 |
|---|---|---|
| `P` | labeled positive set | `X[y_pu == 1]` |
| `U` | unlabeled marginal set | `X[y_pu == 0]` |
| `pi` | positive class prior | `class_prior` / `_class_prior` |
| `f_theta(x)` | classifier logit | `model_(x)` |
| `q_theta(x)` | sigmoid prediction | `predict_proba(X)[:, 1]` |
| `R_P` | positive prediction loss | `positive_loss` |
| `R_lab` | label distribution alignment loss | `alignment` |
| `R_ent` | entropy minimization term | `entropy` |
| `R_mix` | Mixup consistency term | `mixup_weight * mixup_loss` |
| `mu` | entropy weight | `entropy_weight` |
| `nu` | Mixup strength/weight | `mixup_weight` |
| `alpha` | Beta distribution parameter | 当前实现内部随机 `lambda`，未暴露完整参数 |

## 5. 核心公式

### 5.1 正样本监督项

项目用 sigmoid/BCE 对正样本施加正类监督：

```math
\widehat R_P(\theta)
=-
\frac{1}{n_P}\sum_{i=1}^{n_P}
\log q_\theta(x_i^P).
```

该项防止模型只通过满足未标记集均值来得到任意的常数输出。

### 5.2 Label distribution alignment

论文将未标记数据的真实标签分布期望固定为 `pi`，从而使用：

```math
\widehat R_{lab}(\theta)
=
\left(
\frac{1}{n_U}\sum_{j=1}^{n_U}q_\theta(x_j^U)-\pi
\right)^2.
```

正样本也需要具有正类预测期望。当前实现将正样本监督作为 BCE，而未标记样本使用上式的均值对齐；两者共同实现论文的 label-distribution consistency。

### 5.3 熵最小化

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

### 5.4 Mixup 一致性

从样本对 `(x_i, x_j)` 及其当前 soft label `(q_i, q_j)` 构造：

```math
\tilde x=\lambda x_i+(1-\lambda)x_j,
\qquad
\tilde q=\lambda q_i+(1-\lambda)q_j,
```

其中 `lambda` 通常来自 Beta 分布。Mixup 训练项为：

```math
\widehat R_{mix}(\theta)
=\operatorname{BCE}
\left(f_\theta(\tilde x),\tilde q\right).
```

当前实现使用随机 `lambda`，并对构造 soft target 使用 stop-gradient；这是工程侧简化，尚未暴露论文中的完整 `alpha/gamma/nu` 超参数协议。

### 5.5 总目标

项目当前实现的目标为：

```math
\widehat J(\theta)=
\widehat R_P(\theta)
+\lambda_{lab}\widehat R_{lab}(\theta)
+\mu\widehat R_{ent}(\theta)
+\nu\widehat R_{mix}(\theta).
```

对应代码参数为：

| 公式参数 | 构造参数 |
|---|---|
| `lambda_lab` | `alignment_weight` |
| `mu` | `entropy_weight` |
| `nu` | `mixup_weight` |

## 6. 算法概要

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
4. 保存 loss_history_、模型参数和 class_prior_。
5. 用 sigmoid(logit) 输出预测概率，用 logit 的符号输出预测标签。
```

### 6.1 数值稳定性

- logits 在熵计算前限制在 `[-10, 10]`，避免 `log(0)` 和指数溢出。
- sigmoid 输出的熵计算增加 `1e-6` epsilon。
- 推理时 sigmoid 输入限制在 `[-40, 40]`。
- `random_state` 只控制 PyTorch 初始化和 Mixup 随机性；GPU 上的完全确定性仍需额外配置。

## 7. 超参数与实验协议

| 参数 | 当前默认值 | 含义 | 论文对应 |
|---|---:|---|---|
| `class_prior` | 必填 | 正类先验 | `pi_P` |
| `hidden_dim` | 64 | 项目 MLP 隐藏层宽度 | 项目适配 |
| `epochs` | 100 | 全量训练轮数 | 需 benchmark 调整 |
| `batch_size` | 128 | 预留 batch 参数，当前未切 batch | 项目适配 |
| `learning_rate` | `1e-3` | Adam 学习率 | 参考论文训练习惯 |
| `alignment_weight` | 1.0 | 标签分布对齐强度 | 论文 `R_lab` 权重 |
| `entropy_weight` | 0.05 | 熵最小化强度 | 论文 `mu` |
| `mixup_weight` | 0.1 | Mixup 强度 | 论文 `nu` 的工程化参数 |
| `random_state` | 0 | 可复现随机种子 | 项目适配 |

正式复现实验应在每个训练 fold 内固定 `pi`，不得用测试集真实比例调参；若 `pi` 通过 CPE 得到，CPE 也必须只使用对应训练 fold。

## 8. API 接口与项目落点

### 8.1 构造函数

```python
class DistPUClassifier(BasePUClassifier):
    def __init__(
        self,
        class_prior,
        *,
        hidden_dim=64,
        epochs=100,
        batch_size=128,
        learning_rate=1e-3,
        alignment_weight=1.0,
        entropy_weight=0.05,
        mixup_weight=0.1,
        random_state=0,
        device="cpu",
    ):
        ...
```

### 8.2 API 语义

| API / 属性 | 约定 |
|---|---|
| `fit(X, y_pu, *, class_prior=None, sample_weight=None)` | `y_pu=1` 为 P，`0` 为 U；`class_prior` 可覆盖构造参数；当前忽略 `sample_weight` |
| `predict(X)` | `decision_function(X) >= 0` 输出 1，否则 0 |
| `decision_function(X)` | 返回 logit，不是概率 |
| `predict_proba(X)` | 返回两列 sigmoid 概率 |
| `loss_history_` | 每轮总优化目标 |
| `model_` | PyTorch `nn.Sequential` 网络 |
| `device_` | 实际使用的 `torch.device` |
| `get_pu_metadata()` | 基类元数据；包括 class prior、backend 和 fitted 状态 |

### 8.3 模块落点

| 模块 | 责任 | 状态 |
|---|---|---|
| `pu_toolbox/estimators/risk/dist_pu.py` | `DistPUClassifier` | ✅ 核心实现 |
| `pu_toolbox/estimators/risk/__init__.py` | 导出分类器 | ✅ |
| `pu_toolbox/registry/builtin_methods.py` | `dist_pu` 元数据和 lazy binding | ✅ |
| `tests/unit/estimators/test_dist_pu.py` | fit/predict/有限值 smoke test | ✅ |
| `benchmarks/paper_like/dist_pu/` | 官方数据集和 backbone 复现 | ⏳ |

## 9. 测试与验收标准

### 9.1 单元测试

- `class_prior <= 0` 或 `>= 1` 时抛出 `ValueError`。
- 没有 P 或没有 U 时由统一 PU validator 拒绝。
- `predict_proba` 形状为 `(n_samples, 2)`，每行和为 1。
- 训练损失、logits 和概率均为有限值。
- 固定 `random_state` 时 CPU 结果可重复到测试容差。

### 9.2 数学性质测试

- 人工 logits 下，`alignment` 在 `mean(q_U)=pi` 时为 0。
- `entropy` 在概率趋近 0/1 时小于概率趋近 0.5 的情形。
- 禁用 entropy 和 Mixup 后，模型仍能最小化 P 监督项和均值对齐项。

### 9.3 Paper-like benchmark

- Fashion-MNIST、CIFAR-10、Alzheimer 数据协议与官方代码一致。
- 报告 accuracy、AUC、precision、recall，并报告 U 集预测正比例。
- 做 `alignment only`、`alignment+entropy`、`alignment+entropy+Mixup` 三组消融。
- 多随机种子报告均值和标准差，不把单次结果写成论文复现结论。

## 10. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `official_exact` |
| Implementation status | `NATIVE`，PyTorch optional dependency |
| 当前实现 | 论文核心损失的 clean-room 小型 MLP 版本 |
| 尚未对齐 | 官方 backbone、mini-batch、图像增强、数据 split、完整超参数搜索 |
| 主要风险 | `pi` 错误会直接改变 U 集分布约束；熵权重过大可能强化确认偏差；全量训练与官方 batch 训练结果不可直接比较 |

