# Method Card: PUSB

> 参数契约（签名、参数表、返回结构）以 [API 参考](../../user/reference/api.md) 为权威；本文档只记论文研究内容、实现边界与复现状态。

## 1. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Learning from Positive and Unlabeled Data with a Selection Bias |
| Authors | Masahiro Kato, Takeshi Teshima, Junya Honda |
| Venue | ICLR 2019 |
| Year | 2019 |
| Family | `bias_aware` |
| Scenario | `selection_biased` |
| Assumption | SAR / selected-at-random 条件机制 |
| Requires class prior | 官方核实现的分位数预测和 PU 风险需要 `pi`；linear baseline 不依赖 |
| Requires propensity | 论文建模 selection bias；当前 baseline 不显式输出 propensity |
| Requires negative samples | 否 |
| Official source | [MasaKat0/PUlearning](https://github.com/MasaKat0/PUlearning) |
| Paper page | [ICLR 2019 poster](https://iclr.cc/virtual/2019/poster/1024) |

### Assumptions

令 `Y in {0,1}` 为真实标签，`S in {0,1}` 为是否被标记。观测数据为：

```math
P_{+}^{obs}=p(x\mid y=1,s=1),
\qquad
P_U=p(x).
```

PUSB 处理的是 selection-biased PU（SAR）：普通 SCAR 假设要求标记概率为常数

```math
p(s=1\mid y=1,x)=c,
```

PUSB 放宽为选择机制依赖于特征：

```math
e(x)=p(s=1\mid y=1,x),
\qquad
p(x\mid y=1,s=1)
\propto e(x)p(x\mid y=1).
```

---

## 2. 问题设定与符号

已标记正样本会过度代表 `e(x)` 较大的正例区域。若不处理该项，训练得到的 score 可能学习“容易被标记”而不是“真实为正”。

### 2.1 SAR 与 SCAR 对照

| 机制 | 标记概率 | 已标记正例代表性 | 适用方法 |
|---|---|---|---|
| SCAR | `e(x)=c` | 理论上代表全部正类 | Elkan-Noto、uPU、nnPU 等 |
| SAR | `e(x)` 随 `x` 变化 | 有 selection bias | PUSB、LBE |

PUSB 的价值在于承认第二行，而不是把所有偏差归因于模型容量。

| 论文符号 | 含义 |
|---|---|
| `X` | 输入特征 |
| `Y` | 潜在真实类别（不可直接观测） |
| `S` | 是否被标记 |
| `P_+` | 无偏真实正类分布（不可直接观测） |
| `P_+^obs` | 被选择的已标记正类分布 |
| `P_U` | 边缘未标记分布 |
| `e(x)` | 正类样本的 labeling propensity |
| `eta(x)` | 真实后验/目标 score 的单调变换 |
| `f(x)` | 训练得到的 scoring function |
| `tau` | 决策阈值 |
| `C_j` | RBF 基中心 |
| `sigma` | RBF 带宽 |
| `lambda` | L2 正则系数 |

---

## 3. 核心公式

### 3.1 直接训练 P/U 分类器的问题

把 `P_+^obs` 标为 1、把 `P_U` 标为 0 的普通二分类器，实际上区分的是：

```math
p(x\mid y=1,s=1)
\quad\text{与}\quad
p(x),
```

而不是直接区分真实的 `p(x|y=1)` 与 `p(x|y=0)`。在 `e(x)` 非常不均匀时，来源分类器会把“更容易被标记”当成“更可能为真实正类”。

### 3.2 Density-ratio 与排序保持

PUSB 的可计算对象可以写成观察到的正例与 U 的密度比：

```math
\rho(x)=
\frac{p(x\mid y=1,s=1)}{p_U(x)}.
```

论文在其 order/invariance 条件下研究 `rho(x)` 与真实 class posterior `p(y=1|x)` 的排序关系：虽然 `rho` 一般不是 posterior 本身，但在满足该条件时，按 `rho` 排序可以得到与 posterior 一致的顺序。分类器因此需要额外选择阈值：

```math
\hat y(x)=\mathbf 1\{f(x)\ge t\}.
```

### 3.3 Official-aligned RBF 目标与梯度

从 P/U 合并训练集随机选择最多 `B=300` 个中心，构造：

```math
\phi_j(x)=\exp\left(-\frac{\lVert x-C_j\rVert^2}{2\sigma^2}\right),
\qquad
g(x)=\tilde\phi(x)^T b,
```

其中 `tilde phi` 额外包含截距项。对已标记正例 `P` 和未标记集 `U`，适配器优化：

```math
\widehat R(b)=
-\pi\frac{1}{n_P}\sum_{x\in P}g(x)
+\frac{1}{n_U}\sum_{x\in U}\log(1+\exp(g(x)))
+\frac{\lambda}{2}\lVert b\rVert^2.
```

对应梯度为：

```math
\nabla_b\widehat R=
-\pi\,\overline{\tilde\phi}_P
+\frac{1}{n_U}\sum_{x\in U}\mathrm{sigmoid}(g(x))\tilde\phi(x)
+\lambda b.
```

---

## 4. 算法概要

### 4.1 Linear baseline

当前 linear baseline 对 P/U 来源标签训练 `StandardScaler + LogisticRegression`，对新样本输出来源分类器的 decision score，以 `sigmoid(score)` 提供工程侧 `predict_proba`，按概率阈值产生二元预测。它提供 selection-biased PU 的统一接口和可运行 baseline，不能用于声称“已经完成 PUSB 论文复现”。

### 4.2 Official-aligned RBF 适配器

超参数选择复刻官方结构：中心从全训练集选择一次，样本随机分配到折，遍历 `sigma` 与 `lambda` 网格，累加验证 PU 风险，按网格顺序稳定打破平局，最后在全训练集重训。决策阈值的双语义与梯度系数兼容修正见 §6。

---

## 5. 论文边界

在 selection-bias 下，单靠 `(P_+^obs, P_U)` 一般不能无条件识别完整 posterior probability。论文因此研究在较弱假设下可识别的分类器/排序函数：目标是使 scoring function 对真实 class posterior 保持正确顺序，或在可识别区间内给出分类决策。

- PUSB 讨论的是已标记正样本有选择偏差的情形：`P(x|y=1,s=1)` 不一定等于 `P(x|y=1)`。
- 在 selection bias 下，普通 PU 风险分解使用的 SCAR 常数 propensity `c` 不成立；直接套用 Elkan-Noto、uPU 或 nnPU 会引入不可忽略偏差。
- 论文的主要识别目标是后验排序/部分识别的分类器，不等于完整恢复 `P(y=1|x)` 数值。应区分以下三个陈述：
  1. **排序结论**：`f(x_1) > f(x_2)` 可对应真实后验更高的样本。
  2. **部分识别**：数据只能确定一组可能的分类器/决策边界，而不是唯一的后验概率。
  3. **概率校准**：不由 PUSB 的排序结论自动推出，必须额外有标注验证集或更强机制假设。
- 决策阈值 `t` 不是由 `0.5` 自动推出。若没有带真实标签的验证集，阈值只能基于业务成本、先验约束或论文给出的部分识别区间选择。

---

## 6. 实现注记

> 见 ADR-0014 #10（官方源码的正则目标与梯度相差系数 2，适配器使用与官方梯度一致的 `0.5 * lambda * ||b||^2`，该修正必须进入结果 manifest；实现用 `logaddexp`/`expit` 避免指数溢出并用有限差分锁定梯度）

- **决策阈值的双语义**：官方代码在每个测试批次中排序 score，并取索引 `floor(n(1-pi))` 处为阈值，随后使用严格 `score > threshold`。这依赖整个测试批次的组成，不符合 sklearn 单样本预测的稳定性预期。
  - 公共 `predict`：在训练 score 上计算一次先验分位数并冻结为 `threshold_`；
  - `predict_with_prior_quantile`：显式复刻官方批次分位数；
  - benchmark：使用后一接口，并在 trial 中保存实际阈值。
- **README 与代码入口差异**：仓库 README 仍称 `main_linear_kernel.py` 复现 Table 2，但该入口当前默认 `ijcnn1`；最早可见提交中也已如此，后续提交只修改 README/删除文件，没有修正入口。项目因此以论文正文作为 Table 2 数据集权威，以仓库代码作为算法、采样和超参数实现证据。
- **Table 2 单元可行性审计**：按官方循环从 seed 2018 连续执行 100 次重复的静态审计发现，72 个 `dataset × U × prior` 单元中只有 45 个在所有重复中严格满足声明样本量：USPS 与 connect-4 为 12/12，mushrooms 为 11/12，shuttle 为 6/12，pageblocks 为 1/12，spambase 为 3/12。根因是官方脚本对训练池和固定 3000 行 holdout 进行无放回切片，却未校验切片结果长度。例如 pageblocks 划出 holdout 后训练池仅 2473 行，任何 `U=3200` 单元都不可能得到 3200 个未标记样本。兼容模式可记录官方实际长度，严格论文模式则必须阻止该 trial；批准重采样政策前不能声称完整复现 Table 2。
- `predict_proba` 返回 logistic ranking model 的工程概率，不能宣称是 PUSB 理论保证的校准后验。

结果记录（benchmark 数字、SHA/commit 锁与试次统计以 benchmarks 产物为准）：

> 结果见 `benchmarks/assigned_methods/results/clean_room_multiseed/`（clean-room SAR）；10-seed 配对 ranking 见 `benchmarks/assigned_methods/results/scar_sar_comparison/`；官方配置 commit 锁见 `benchmarks/assigned_methods/configs/official_sources.lock.json`
>
> 结果见 `benchmarks/assigned_methods/results/pusb_official_data_smoke/` 与 `benchmarks/assigned_methods/results/pusb_official_data_feasible_multiseed/`（IJCNN1 仓库扩展 smoke 与完整网格）
>
> 结果见 `benchmarks/assigned_methods/results/pusb_table2_strict_full/REPORT.md`（Table 2 严格可行子集 45 单元/4500 trials）

---

## 7. 论文实验参考

### 7.1 可控 SAR 数据生成

先生成带真实标签的数据，再只暴露 PU 标签：

```text
Y ~ Bernoulli(pi)
X | Y 由可分离程度 delta 控制
S | X,Y=1 ~ Bernoulli(e(X))
S=0 when Y=0
observed y_pu = S
```

至少包含以下 selection mechanism：

```text
SCAR:       e(x) = c
linear SAR: e(x) = sigmoid(a^T x + b)
nonlinear:  e(x) = sigmoid(a1*x1 + a2*x2^2 + b)
```

### 7.2 对照与评估对象

同一 split 上运行 P/U logistic、uPU、nnPU、Dist-PU 和 official-aligned PUSB；后面三种方法若依赖 SCAR 或已知先验，必须在表头明确。另加入使用隐藏真实标签训练的 PN classifier 作为诊断上界，不作为可部署基线。

评估分为两层：

- 排序：ROC-AUC、PR-AUC、Spearman/Kendall 相关性、pairwise ranking accuracy；
- 决策：在独立验证集选择阈值后报告 balanced accuracy、F1 和风险。

### 7.3 官方核实现参数（来源：官方源码）

| 参数 | 默认值 | 含义 | 来源 |
|---|---:|---|---|
| kernel `n_basis` | 300 | RBF 中心数上限 | 官方源码 |
| kernel `cv` | 5 | 随机交叉验证折数 | 官方源码 |
| kernel `sigma_grid` | 9 个值 | `0.01` 至 `20` 的带宽网格 | 官方源码 |
| kernel `reg_grid` | 8 个值 | `0.001` 至 `5` 的正则网格 | 官方源码 |
| kernel `random_state` | 2018 | 中心和折分配随机源 | 官方实验初始 seed |
| kernel `class_prior` | 必填 | PU 风险与分位数规则中的 `pi` | 官方算法 |

### 7.4 论文 Table 2 数据协议

论文第 5.2 节和 Table 2 明确使用六个线性模型数据集：mushrooms、shuttle、pageblocks、usps、connect-4 和 spambase。每个数据集构造：

```text
class_prior in {0.2, 0.4, 0.6, 0.8}
unlabeled_size in {800, 1600, 3200}
positive_size = 400
test_size = 1000
repetitions = 100
```

---

## 8. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `official_exact`：作者源码已锁定；该字段表示源码可获得性，不表示 toolbox 逐行复制 |
| Implementation status | `NATIVE`；linear baseline 与 official-aligned kernel 双实现 |
| 当前实现可声称 | 统一接口、核 PU 目标/CV/分位数规则、IJCNN1 仓库扩展，以及 Table 2 的 45 单元/4500-trial 严格可行子集 |
| 当前实现不可声称 | 已复现论文 Table 2 六数据集比较表 |
| 主要风险 | 观察到的 P 受到 `e(x)` 加权；直接来源分类会把 selection preference 与 class posterior 混合 |
