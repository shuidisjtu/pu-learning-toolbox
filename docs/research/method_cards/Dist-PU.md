# Method Card: Dist-PU

> 参数契约（签名、参数表、返回结构）以 [API 参考](../../user/reference/api.md) 为权威；本文档只记论文研究内容、实现边界与复现状态。

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
3. 默认 `training_recipe="clean_room"` 每轮训练：
   a. 计算所有 P/U 的 logits 和 sigmoid outputs；
   b. 计算 P 上的正类 BCE；
   c. 计算 U 上的均值对齐损失 (mean(q_U)-pi)^2；
   d. 计算 U 上的熵最小化损失；
   e. 对样本对进行 Mixup，计算 soft-target BCE；
   f. 加权求和并反向传播。
4. 保存训练记录与模型参数。
5. 用 sigmoid(logit) 输出预测概率，用 logit 的符号输出预测标签。
```

显式 opt-in `training_recipe="author_staged"` 且 `distribution_loss="author_histogram"` 时，切换到两阶段来源适配：
先执行 warm-up histogram + U entropy；再重建 Adam/scheduler，初始化并按训练行身份更新持久伪标签，逐 batch
计算 histogram、动态 U entropy、mixup entropy 与双目标 BCE。详见 §8.1；该路径不改变默认 clean-room 结果。

---

## 5. 论文边界

- Dist-PU 不是一个类先验估计器；`pi` 必须由用户提供，或在训练外部先通过 `ReCPE`/其他 CPE 得到。
- `U` 来自边缘分布 `p(x)`。如果 `U` 是人为筛选、时间漂移或选择偏置样本，均值约束 `E_U[q(x)] = pi` 不再直接对应论文设定。
- 仅使用标签分布对齐会产生平凡解：所有未标记样本都输出 `pi`。熵最小化用于推动预测远离常数解，Mixup 用于缓解确认偏差。
- 预测概率是模型 sigmoid 输出，不自动等同于校准的 `P(y=1|x)`；论文核心约束是分布期望，而非概率校准定理。

---

## 6. 实现注记

- 默认 `clean_room` 路径使用全量张量训练，`batch_size` 不生效；显式 `author_staged` 才使用真实 mini-batch。
- 当前默认使用 toolbox 小型 MLP；显式 staged encoder 路径也是独立工程适配，不应直接与论文的图像 backbone 结果比较。
- 默认 clean-room 路径使用工程化随机 Mixup；author-staged 则显式暴露 `alpha`、mixed entropy 和 mixup BCE 系数，并按来源 Beta mixup；两者不可混称。
- 数值稳定性：
  - 两条路径的熵项都在 sigmoid 前限制 logits 到 `[-10, 10]`；clean-room 的 `log` 内另有 `1e-6`，author-staged 按来源定义直接对受限概率取 `log`；
  - clean-room 中的 `1e-6` **不是数值守卫**（真正的兜底是上面的 clamp，概率被夹在 `[4.5e-5, 0.99995]`），重构时须逐字符保留其位置，不得因「看起来没用」而删改；
  - 推理时 sigmoid 输入限制在 `[-40, 40]`；
  - `random_state` 控制网络初始化、batch permutation 与 Beta mixup 抽样；不同训练路径/设备的全序列 bitwise replay 不作保证，GPU 上完全确定性仍需额外配置。

结果记录（benchmark 数字与官方配置 commit 锁以 benchmarks 产物为准）：

> 结果见 `benchmarks/assigned_methods/results/clean_room_multiseed/`（当前 clean-room 运行结果）；官方配置 commit 锁见 `benchmarks/assigned_methods/configs/official_sources.lock.json`

- **训练视图叠加（协议 §2.3，实验层叠加而非算法语义）**：`fit(..., os_or_ts=...)` 默认 `"os"`。`"ts"`（方法原生 case-control 采样）把 **label-distribution alignment 与 entropy 的角色集合**由 $`X_U`$ 换成 $`X_U \cup X_P`$（alignment 均值的分母随之取 $`n_P + n_U`$）。正例 BCE 只消费原始 $`P`$，不因 P 同时进入 marginal 角色而扩大分母或重复行；class prior 保持总体 $`\pi`$，**不**按观测 P/U 比例重估。直接调用 estimator 时默认 OS；Survey 未显式覆盖时的默认视图由 resolver 依台账原生假设与 `fit` 签名决定（本方法台账已是原生 TS，故接线后其默认行转 `ts` ；按 D20 单方放行，manifest 不再挂 `ts_view_collaborator_review`，但该路径的方法学复核仍未获得，呈报不得读作已签署）。
- **alignment 的依据是可核验的 population identity，不是「约定如此」**：$`R_{lab}`$ 约束的是 $`E_{p(x)}[f(x)] = \pi`$，而 OS 视图下的 $`X_U`$ 恰好排除了已标记正例、不再代表完整 marginal，约束因此落在错的值上；并集把它搬回正确落点。**量化（本次 Spambase `split_0` 制品观测，不是本方法的一般性质——换 seed、数据集或 `c` 数值会不同）**：OS 角色集真实正例率 0.369265 vs 目标 $`\pi`$ = 0.394045（偏离 −0.024780）；TS 角色集 0.394022（偏离 −0.000023，即分层切分的舍入残差）；前者约为后者的 1076 倍。一般成立的是那条恒等式：并集均值被约束到 $`\pi`$ 时，U 行被要求达到的条件均值是 $`\left((n_P+n_U)\pi - n_P \overline{f}_P\right)/n_U`$；在本制品上它等于 0.369，正是 U 行的真实正例率。
- **entropy 扩到 P 上只有协议一致性依据，须与 alignment 分开标注**：协议 §2.3 对「未标记损失输入」作整体规则（PUBench 同样如此），这不是数学必然。P 的概率已被权重 1.0 的正例损失推向 1，entropy 权重仅 0.05，两者方向一致、不冲突。按 D17 ⑤ 的口径，这是**协议选择（protocol choice）**，不是论文事实。
- **其余一概不随视图变化**：模型结构、优化器、epoch、先验决策、callback 与 checkpoint 契约；Mixup 物理池**本已消费完整 X**，故保持原样、不追加 P。Mixup 池不随视图变化还有一条更硬的理由：**单变量消融**——池子若在 OS 下缩回 $`X_U`$，两跑的差异就不止「正则项角色集」这一个变量（与 D17 否决「验证折并入 P」同源）。随机侧只锁定 **RNG API 的调用次数、调用顺序、`randperm` 长度、`lam` 的形状与抽样位置**；**不**锁定后续 epoch 的 loss 或梯度——模型参数在首轮更新后即分化，那是消融的目的而非缺陷。
- **`sample_weight` 的处置与相邻方法都不同**：本估计器是 `SampleWeightSupport.IGNORED`——既不是 `nnpu` 的「需要互斥门」，也不是 `pusb_kernel` 的「整体拒绝」。并集确实会引入没有权重定义的 P 行，但本估计器整体忽略权重，故无害。三种处置的措辞**不可互相照抄**。
- **与 PUBench 的关系是类比，不是逐行对应**：PUBench 的公共 union 只把 P 追加进 U batch，而它的 label-distribution loss 是**双样本直方图**损失（`l_p + frac_prior * l_u`，只有 `l_u` 吃扩充后的 U），本实现是**单样本矩约束**；其 Mixup 因外层 union 使 P 在池中出现两次，本项目不复制。`ts` 路径尚未经合作者复核。

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

上表仅描述默认 clean-room 路径的映射。`author_staged` 使用独立参数 `warmup_*`、`pu_*`、`mixup_alpha`、
`mixup_entropy_weight` 和 `mixup_bce_weight`；其默认值按锁定作者脚本可见配置设置，但仍是 MLP adapter 的
工程配置，不代表原始数据集 recipe 或已批准的超参搜索。

### 7.2 论文实验

论文实验使用 Fashion-MNIST、CIFAR-10 与 Alzheimer 三个 benchmark，官方协议含数据集专属 backbone、mini-batch、数据处理与两阶段训练；toolbox 不得与论文图像 backbone 结果直接比较。

---

## 8. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `official_exact` |
| Implementation status | `NATIVE`，PyTorch optional dependency |
| 当前实现 | 默认 clean-room 小型 MLP；另有显式 `training_recipe="author_staged"` + `distribution_loss="author_histogram"` 的两阶段 mini-batch 适配 |
| 尚未对齐 | 官方 dataset-specific backbone/预处理/augmentation、公开 benchmark 重放、完整超参数搜索及作者 test-based 选模（后者不得照搬） |
| 主要风险 | `pi` 错误会直接改变 U 集分布约束；熵权重过大可能强化确认偏差；全量训练与官方 batch 训练结果不可直接比较 |

### 8.1 锁定作者程序的训练行为差异

本节记录 2026-10-19 的 MLP 适配交付；2026-10-20 的显式 encoder 扩展见 §8.2。

2026-10-19 对锁定作者提交 `cb74be1a87176fd38270873c06374e53905b7354` 完成只读交叉核对，
见[来源行为审计](../pu_survey/admission/dist_pu_source_behavior_review_20261019.md)。作者 `train.py`
不是仅将上述几项核心损失放入一个训练循环：程序先做 60 epoch warm-up，再做 60 epoch Dist-PU；
使用 batch_size 256 的 shuffled DataLoader；PU 阶段每个 batch 基于持久伪标签做 mixup，优化后更新当前行
伪标签；进入 PU 阶段会重建 Adam、切换学习率/weight decay，并运行余弦调度和逐 epoch 熵系数日程。
作者把两个阶段的 `CosineAnnealingLR` 都设为 `T_max=pu_epochs`，但计步 cadence 不同：warm-up 的
`utils.train()` 在每个 epoch 的 DataLoader 循环结束后 step 一次；PU 阶段 `train_mixup()` 在每个
mini-batch 更新后 step。源码位置与差异见[来源行为审计](../pu_survey/admission/dist_pu_source_behavior_review_20261019.md)。
作者 `losses/factory.py::create_loss()` 的 warm-up 默认使用直方图损失加 `co_mu=0.002` 的 U 熵；PU 阶段把
`entropy=0`，用同一基础直方图损失另加动态熵与 mixup 项。作者 `distributionLoss.py` 使用 batch 内直方图分布距离，目标为 P 直方图与单点正类 proxy 的距离，加上
`1/(2*pi)` 加权的 U 直方图与边缘先验混合 proxy 的距离；不等同于本卡 §3 的单样本均值平方对齐 clean-room
目标。estimator 现提供显式 `distribution_loss="author_histogram"` 选项，逐项实现作者
`LabelDistributionLoss(num_bins=1, proxy="polar", dist="L1")` 的 P/U histogram 项；TS 时仅按项目
协议将 P 并入 marginal-role histogram，不改 `pi` target。该选项仍运行现有单阶段全量训练，并保留当前熵/
Mixup工程路径；默认值仍是既有 `expectation`。另增加 opt-in `training_recipe="author_staged"`，默认复用作者
warm-up/PU 两阶段长度、Adam/weight decay、batch size、熵权重、mixup 系数和分阶段 scheduler cadence；持久伪标签
按训练行身份初始化与批次更新；CPU 常驻完整输入，只逐 batch 搬运到训练设备，伪标签按 128 行 chunks 初始化。
该适配仍使用 toolbox MLP、调用方传入 prior，并且不复现原作者的数据、图像模型、
增强和训练/test split，故 `upstream_training_recipe_reproduced=false`。作者脚本每个 epoch 在 test loader
上按 accuracy 记录 best，这种选择不能移入本项目。

因此，本卡和台账中的 `official_exact` 仅表示已锁定作者来源身份，不表示 toolbox estimator 忠实重放作者
训练程序。默认仍是 clean-room 小型 MLP；其 `batch_size` 不生效。只有 `author_staged` 中该参数才控制真实
mini-batch。拟合后的 `training_provenance_` 会逐 run 记录训练路径、两阶段参数、scheduler 实际步数、伪标签时序、
OS/TS 角色适配、优化步数与 epoch 数。`author_staged` 不重放作者 test accuracy 选模，也不改变默认训练路径、
历史 Pilot 或正式准入。仍待完成的是正式数据/backbone/PA-OA 选择协议、批准预算与资源验收；不得把 MLP 短拟合
或该适配器称作论文数值复现。

### 8.2 显式 encoder 与 CPU-stream（2026-10-20）

新增可选 `encoder=None`。显式 `author_staged + author_histogram` 路径可接收二维特征或
四维 NCHW 图像，deepcopy/解冻 encoder 后接 MLP head，两个阶段都训练 encoder；
clean-room 默认仍是原全量 MLP，不隐式切换损失或训练 recipe。Registry、台账和 Pipeline
能力声明同步为 MLP/CNN；四维输入缺 encoder 或 encoder 搭配 clean-room 均提前拒绝。

输入/标签/持久伪标签留在 CPU，训练与推理按 `batch_size` 搬运，encoder 的初始伪标签
扫描按 `min(128,batch_size)`；普通 MLP 仍保留原 128 行扫描。单行尾批保留，BN 使用运行
统计；prototype、in-place 输入、CV 副本与评分模式均作隔离。额外 provenance 披露这些
工程决定；旧 MLP pickle 回填缺失可选参数，原模型参数和已保存结果不作改写。

冻结 Pilot 的 fullbatch/mlp128/CIFAR feature-adapter 路径保持原定义；新 CNN 结构、
阶段资格、预算与资源仍需新协议登记和复核。具体接口、UI 限制与验证见
[Dist-PU encoder 接入记录](../pu_survey/admission/distpu_encoder_stream_20261020.md)。
