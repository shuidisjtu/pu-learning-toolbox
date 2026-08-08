# Method Card: PUSB

## 1. 待办与注意

### 1.1 待办

- [x] 明确 PUSB 处理的是 selection-biased PU，而不是 SCAR 下的普通 PU。
- [x] 在 registry 中标记为 `BIAS_AWARE`、`SAR`、`SELECTION_BIASED`。
- [x] 提供统一的 sklearn-compatible `PUSBClassifier`，支持 `fit/predict/decision_function/predict_proba`。
- [x] 在方法卡中区分论文的 partial-identification/ranking 结论与当前工程实现。
- [x] 补充 SCAR/SAR 生成、排序与决策评估、官方对齐和统计汇总协议。
- [x] 移植官方 RBF scoring、PU 风险、随机 CV 和先验分位数决策，并补公式级测试。
- [x] 在配对 SCAR/线性 SAR/非线性 SAR 合成数据上验证 posterior ranking preservation。
- [x] 使用官方仓库默认 IJCNN1 完成缩小网格 smoke，并保存配置、hash 和 manifest。
- [x] 对 IJCNN1 可行 `pi=0.2` 完成 3 seeds × 3 U sizes 的完整网格与 densratio 对照。
- [x] 核对论文 Table 2，确认 IJCNN1 是仓库扩展而非论文表格数据集。
- [x] 锁定并接入 mushrooms、shuttle、pageblocks、usps、connect-4、spambase，审计采样可行性。
- [ ] 明确不可行单元政策，执行严格可行单元并单独报告官方静默截断单元。

### 1.2 注意

- PUSB 讨论的是已标记正样本有选择偏差的情形：`P(x|y=1,s=1)` 不一定等于 `P(x|y=1)`。
- 在 selection bias 下，普通 PU 风险分解使用的 SCAR 常数 propensity `c` 不成立；直接套用 Elkan-Noto、uPU 或 nnPU 会引入不可忽略偏差。
- 论文的主要识别目标是后验排序/部分识别的分类器，不等于完整恢复 `P(y=1|x)` 数值。
- `predict_proba` 在当前项目中返回 logistic ranking model 的工程概率，不能宣称是 PUSB 理论保证的校准后验。
- `PUSBClassifier` 保留为可运行的 linear ranking baseline；`PUSBKernelClassifier` 是独立的
  official-aligned clean-room 适配器，避免已有用户升级后行为突变。
- 官方源码的正则目标与梯度相差系数 2。适配器使用与官方梯度一致的
  `0.5 * lambda * ||b||^2`，该修正必须进入结果 manifest。

## 2. 论文信息

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

## 3. 问题设定

令 `Y in {0,1}` 为真实标签，`S in {0,1}` 为是否被标记。观测数据为：

```math
P_{+}^{obs}=p(x\mid y=1,s=1),
\qquad
P_U=p(x).
```

普通 SCAR 假设要求：

```math
p(s=1\mid y=1,x)=c,
```

其中 `c` 与 `x` 无关。PUSB 放宽为选择机制依赖于特征：

```math
e(x)=p(s=1\mid y=1,x),
```

因此：

```math
p(x\mid y=1,s=1)
\propto e(x)p(x\mid y=1).
```

已标记正样本会过度代表 `e(x)` 较大的正例区域。若不处理该项，训练得到的 score 可能学习“容易被标记”而不是“真实为正”。

### 3.1 SAR 与 SCAR 对照

| 机制 | 标记概率 | 已标记正例代表性 | 适用方法 |
|---|---|---|---|
| SCAR | `e(x)=c` | 理论上代表全部正类 | Elkan-Noto、uPU、nnPU 等 |
| SAR | `e(x)` 随 `x` 变化 | 有 selection bias | PUSB、LBE |

PUSB 的价值在于承认第二行，而不是把所有偏差归因于模型容量。

## 4. 符号与记号

| 符号 | 含义 | 开发侧对应 |
|---|---|---|
| `X` | 输入特征 | `X` |
| `Y` | 潜在真实类别 | 不可直接观测 |
| `S` | 是否被标记 | `y_pu == 1` 近似 `S=1` |
| `P_+` | 无偏真实正类分布 | 不可直接观测 |
| `P_+^obs` | 被选择的已标记正类分布 | `X[y_pu == 1]` |
| `P_U` | 边缘未标记分布 | `X[y_pu == 0]` |
| `e(x)` | 正类样本的 labeling propensity | 当前 baseline 未显式估计 |
| `eta(x)` | 真实后验/目标 score 的单调变换 | `decision_function(X)` 的目标 |
| `f(x)` | 训练得到的 scoring function | `PUSBClassifier.decision_function` |
| `tau` | 工程决策阈值 | baseline 的 `threshold` 或 kernel 的 `threshold_` |
| `C_j` | RBF 基中心 | `PUSBKernelClassifier.centers_` |
| `sigma` | RBF 带宽 | `sigma_` |
| `lambda` | L2 正则系数 | `reg_lambda_` |

## 5. 方法目标与理论边界

### 5.1 为什么直接训练 P/U 分类器有问题

把 `P_+^obs` 标为 1、把 `P_U` 标为 0 的普通二分类器，实际上区分的是：

```math
p(x\mid y=1,s=1)
\quad\text{与}\quad
p(x),
```

而不是直接区分真实的 `p(x|y=1)` 与 `p(x|y=0)`。在 `e(x)` 非常不均匀时，来源分类器会把“更容易被标记”当成“更可能为真实正类”。

### 5.2 PUSB 的 partial-identification 视角

在 selection-bias 下，单靠 `(P_+^obs, P_U)` 一般不能无条件识别完整 posterior probability。论文因此研究在较弱假设下可识别的分类器/排序函数：目标是使 scoring function 对真实 class posterior 保持正确顺序，或在可识别区间内给出分类决策。

应区分以下三个陈述：

1. **排序结论**：`f(x_1) > f(x_2)` 可对应真实后验更高的样本。
2. **部分识别**：数据只能确定一组可能的分类器/决策边界，而不是唯一的后验概率。
3. **概率校准**：不由 PUSB 的排序结论自动推出，必须额外有标注验证集或更强机制假设。

### 5.3 Density-ratio 与 order preservation

PUSB 的可计算对象可以写成观察到的正例与 U 的密度比：

```math
\rho(x)=
\frac{p(x\mid y=1,s=1)}{p_U(x)}.
```

论文在其 order/invariance 条件下研究 `rho(x)` 与真实 class posterior `p(y=1|x)` 的排序关系：虽然 `rho` 一般不是 posterior 本身，但在满足该条件时，按 `rho` 排序可以得到与 posterior 一致的顺序。因此分类器需要额外选择阈值：

```math
\hat y(x)=\mathbf 1\{f(x)\ge t\}.
```

这里的 `t` 不是由 `0.5` 自动推出。若没有带真实标签的验证集，阈值只能基于业务成本、先验约束或论文给出的部分识别区间选择。当前代码的 `threshold=0.5` 是 sklearn baseline 的工程默认值，不是 PUSB 理论阈值。

### 5.4 项目当前可计算的对象

当前代码输出：

```math
f_{logistic}(x)=w^T\mathrm{scale}(x)+b,
```

以及：

```math
q_{logistic}(x)=\mathrm{sigmoid}(f_{logistic}(x)).
```

它是 PUSB family 的可运行接口 baseline，不等价于论文完整 partial-identification solver。该边界必须在实验报告中保留。

## 6. 当前实现的算法流程

### 6.1 Linear baseline

```text
输入：X、PU 标签、阈值 tau、正则 C

1. 统一校验 y_pu：1 为 labeled positive，0 为 unlabeled。
2. 对 P/U 来源标签训练 StandardScaler + LogisticRegression。
3. 对新样本输出来源分类器的 decision score。
4. 用 sigmoid(score) 提供工程侧 predict_proba。
5. 用 probability >= tau 产生二元预测。
```

这个流程的定位是：

- 提供 selection-biased PU 的公共 API 和 baseline；
- 让 registry、benchmark 可以先调用 PUSB；
- 为后续官方算法替换保留相同的 estimator contract。

它不能用于声称“已经完成 PUSB 论文复现”。

### 6.2 Official-aligned RBF 适配器

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

实现使用 `logaddexp` 和 `expit` 避免指数溢出，并用有限差分锁定梯度。官方源码目标最后
一项写成 `lambda * b^T b`，但梯度写成 `lambda * b`；上述 `lambda/2` 是显式兼容性
修正，而不是无记录地改变公式。

超参数选择复刻官方结构：中心从全训练集选择一次，样本随机分配到折，遍历 `sigma` 与
`lambda`，累加验证 PU 风险，按网格顺序稳定打破平局，最后在全训练集重训。

### 6.3 决策阈值的双语义

官方代码在每个测试批次中排序 score，并取索引 `floor(n(1-pi))` 处为阈值，随后使用严格
`score > threshold`。这依赖整个测试批次的组成，不符合 sklearn 单样本预测的稳定性预期。

- 公共 `predict`：在训练 score 上计算一次先验分位数并冻结为 `threshold_`；
- `predict_with_prior_quantile`：显式复刻官方批次分位数；
- benchmark：使用后一接口，并在 trial 中保存实际阈值。

## 7. 参数与项目协议

| 实现/参数 | 默认值 | 含义 | 来源 |
|---|---:|---|---|
| baseline `threshold` | 0.5 | logistic 工程预测阈值 | 项目适配 |
| baseline `C` | 1.0 | logistic L2 正则倒数 | 项目适配 |
| kernel `n_basis` | 300 | RBF 中心数上限 | 官方源码 |
| kernel `cv` | 5 | 随机交叉验证折数 | 官方源码 |
| kernel `sigma_grid` | 9 个值 | `0.01` 至 `20` 的带宽网格 | 官方源码 |
| kernel `reg_grid` | 8 个值 | `0.001` 至 `5` 的正则网格 | 官方源码 |
| kernel `random_state` | 2018 | 中心和折分配随机源 | 官方实验初始 seed |
| kernel `max_iter` | 200 | BFGS 最大迭代次数 | 项目显式化 |
| kernel `class_prior` | 必填 | PU 风险与分位数规则中的 `pi` | 官方算法 |

正式 PUSB 复现还需要记录：selection propensity 的生成函数、SAR 参数、正类先验、P/U 样本量、真实测试标签和排序评估指标。

## 8. API 接口与项目落点

### 8.1 构造函数

```python
class PUSBClassifier(BasePUClassifier):
    def __init__(self, *, threshold=0.5, C=1.0, max_iter=1000):
        ...

class PUSBKernelClassifier(BasePUClassifier):
    def __init__(self, *, n_basis=300, cv=5, sigma_grid=..., reg_grid=...,
                 random_state=2018, max_iter=200, tol=1e-5):
        ...
```

### 8.2 API 语义

| API / 属性 | 约定 |
|---|---|
| baseline `fit(X, y_pu, ..., sample_weight=None)` | `y_pu=1` 是观察到的正样本，`0` 是 U；支持样本权重 |
| `decision_function(X)` | score 越高，越倾向于正类；用于排序 |
| `predict(X)` | `predict_proba(X)[:,1] >= threshold` |
| `predict_proba(X)` | logistic 工程概率，不是理论校准后验 |
| `model_` | `StandardScaler + LogisticRegression` pipeline |
| `get_pu_metadata()` | 返回 `SAR`、`selection_biased` 和 fitted 状态 |
| `PUSBKernelClassifier.fit(..., class_prior=pi)` | 运行官方对齐核 CV 与全量重训；拒绝缺失/非法先验 |
| `PUSBKernelClassifier.predict(X)` | 使用训练集冻结阈值，预测不依赖批次组成 |
| `predict_with_prior_quantile(X)` | 使用官方测试批次先验分位数规则 |
| `cv_scores_` / `cv_convergence_` | 全网格验证风险和各候选折收敛状态 |
| `optimization_result_` | 最终 BFGS 诊断，不隐藏未收敛状态 |

### 8.3 模块落点

| 模块 | 责任 | 状态 |
|---|---|---|
| `pu_toolbox/estimators/bias_aware/pusb.py` | 当前 linear ranking baseline | ✅ |
| `pu_toolbox/estimators/bias_aware/pusb_kernel.py` | official-aligned RBF 适配器 | ✅ |
| `pu_toolbox/estimators/bias_aware/__init__.py` | 导出 baseline 与 kernel 两个分类器 | ✅ |
| `pu_toolbox/registry/builtin_methods.py` | `pusb` 元数据和 lazy binding | ✅ |
| `tests/unit/estimators/test_bias_aware.py` | API smoke test | ✅ |
| `benchmarks/assigned_methods/` | SAR 合成 runner、官方参数锁和结果 | ✅ baseline + official-data smoke |
| `benchmarks/assigned_methods/pusb_official_data.py` | IJCNN1 扩展校验、官方抽样、uLSIF、checkpoint 与 provenance | ✅ repository extension |
| `benchmarks/assigned_methods/pusb_table2_data.py` | Table 2 六数据集锁定加载与官方 seed/采样可行性审计 | ✅ |

## 9. 测试与验收标准

### 9.1 API 测试

- 缺少正样本或缺少未标记样本时拒绝输入。
- `threshold` 不在 `(0,1)` 或 `C <= 0` 时抛出 `ValueError`。
- `predict` 只返回 `{0,1}`。
- `decision_function` 和 `predict_proba` 形状正确且为有限值。
- `sample_weight` 能正确路由到 pipeline 中的 logistic step。
- registry 中 `pusb` 的 `assumption` 含 `SAR`，`backend=SKLEARN`，状态为 `NATIVE`。
- kernel 目标梯度与中心有限差分一致，固定 seed 的中心、fold、CV score 和预测完全可复现。
- kernel 公共预测不随同批次其他样本变化；官方 batch quantile 只能通过显式 API 调用。

### 9.2 SAR 性质测试

必须额外构造真实标签 `Y` 和 propensity `e(x)`，分别检查：

- SCAR (`e(x)=c`) 时与普通 PU baseline 的行为接近；
- SAR (`e(x)` 随 `x` 变化) 时比较 score 的 Spearman/Kendall 排序相关性；
- 不把 accuracy 单独作为 ranking preservation 的证明；
- 改变 selection mechanism 后记录 score shift 和 AUC 变化。

### 9.3 完整论文复现

- 使用官方代码的数据划分和 selection-bias 参数；
- 对照论文中的 partial-identification/classification metric；
- 多随机种子报告均值、标准差和置信区间；
- 明确报告当前实现是 baseline 还是 official-aligned 版本。

## 10. 复现实验协议

### 10.1 前置门槛

PUSB 的 official-aligned scoring 已有独立实现，但论文级实验必须使用第 10.7 节列出的
Table 2 六数据集。实验报告必须用 `implementation_variant` 区分 linear baseline 与
kernel adapter，并用 `fidelity_level` 区分仓库扩展和论文协议。

### 10.2 可控 SAR 数据生成

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

扫描 `pi in {0.1, 0.3, 0.5}`、平均标记率 `{0.1, 0.3, 0.5}`、偏置强度
`{weak, medium, strong}` 和至少 20 个 seed。通过调节截距使不同机制具有相近平均标记率，
避免把“标记样本更多”误判为 selection-bias 方法更好。

### 10.3 对照与评估对象

同一 split 上运行 P/U logistic、uPU、nnPU、Dist-PU 和 official-aligned PUSB；后面三种
方法若依赖 SCAR 或已知先验，必须在表头明确。另加入使用隐藏真实标签训练的 PN classifier
作为诊断上界，不作为可部署基线。

评估分为两层：

- 排序：ROC-AUC、PR-AUC、Spearman/Kendall 相关性、pairwise ranking accuracy；
- 决策：在独立验证集选择阈值后报告 balanced accuracy、F1 和风险。

`predict_proba` 的 Brier score/校准曲线只作诊断；除非完成额外校准与识别假设验证，否则
不能将其解释为真实 posterior。阈值不得在测试集上根据真实标签选择。

### 10.4 真实数据与官方对齐

官方数据集、selection-bias 构造、预处理、超参数和 split 必须从论文附录与官方仓库生成
版本化 manifest。每个数据集同时运行 SCAR 控制组和论文 SAR 设置，保证性能差异可归因
于选择机制。官方仓库 commit、运行环境和任何修复补丁必须随结果保存。

### 10.5 统计、产物与验收

- 每个配置至少 20 个合成 seed；真实数据使用论文重复次数，缺失时项目默认 10；
- 方法共享 split/seed，报告均值、标准差和 paired bootstrap 95% CI；
- 保存 `Y/S/e(X)` 的生成参数，但训练入口只能接收 `X/S`；
- 保存逐样本 score，便于独立复算排序指标和阈值；
- 验收时应看到 SCAR 与 SAR 强度变化的完整曲线，而不是只挑选有利设置；
- 只有 official-aligned 实现通过公式级测试、排序性质测试并跑完官方 manifest 后，结果才可
  标为 `paper_like`。

建议落点为 `benchmarks/sar/pusb/`，至少包含 `synthetic.yaml`、
`official.yaml`、`dataset_manifest.json`、`trials.csv` 和可重建汇总表的脚本。

当前 clean-room SAR 运行使用 seed `0..4`，来源 Logistic Regression 得到 ROC-AUC
`1.0000 ± 0.0001`。额外的 10-seed 配对 benchmark 在 SCAR、线性 SAR、非线性 SAR
下得到 posterior pairwise ranking accuracy `0.9148`、`0.9590`、`0.9537`。该合成数据
可分性较强，因此这些数字用于验证 ranking 链路，不作为真实数据性能结论。官方 PUSB
配置已锁到 commit
`3401b77ccdd653d39f4f3a6258a42c7938fa9ede`，包括 100 次重复、四个先验、三种 U
样本量和 kernel CV 网格。

### 10.6 IJCNN1 官方仓库扩展

解压后的完整 LIBSVM IJCNN1 文件 SHA-256 为
`16506cad788cf7c9607454150ed1994788204bac2ff4c9cb3b320036b6950d3f`，形状为
`(49990, 22)`，其中 `+1` 为 4,853 条。seed 2018 的官方 3,000 条 holdout 只有 315 条
正例，因此 1,000 条测试集仅能满足 `pi=0.2` 所需的 200 条正例，不能满足
`pi=0.4/0.6/0.8` 所需的 400/600/800 条。runner 对这些组合直接报错，不实施替换采样。
这一限制属于仓库 IJCNN1 扩展，不是论文 Table 2 的协议阻塞项。

已执行 smoke 使用 `pi=0.2`、400 P、800 U、1,000 test、30 个基、3 折以及
`sigma={0.5,1.0}`、`lambda={0.01,0.1}`。选中 `sigma=1.0`、`lambda=0.01`，官方
batch-quantile accuracy 为 `0.7470`，balanced accuracy 为 `0.6038`，ROC-AUC 为
`0.6664`。结果 manifest 固定为 `paper_claim=false`。

进一步完成 seed `2018..2020` 与 U `{800,1600,3200}` 共 9 个完整 trial：300 个 RBF 基、
5 折、9×8 PUSB 网格，以及 `densratio 0.3.0` 默认 100 kernels 与 13×13 uLSIF 搜索。
所有 CV 候选和最终 BFGS 重训均成功，全部 trial 选择 `sigma=1.0`、`lambda=0.001`。

| U | PUSB quantile accuracy | PUSB ROC-AUC | uLSIF quantile accuracy | uLSIF ROC-AUC |
|---:|---:|---:|---:|---:|
| 800 | `0.7730 ± 0.0151` | `0.7061 ± 0.0093` | `0.7543 ± 0.0050` | `0.6519 ± 0.0325` |
| 1600 | `0.7683 ± 0.0170` | `0.6982 ± 0.0280` | `0.7543 ± 0.0050` | `0.6519 ± 0.0326` |
| 3200 | `0.7657 ± 0.0114` | `0.6983 ± 0.0210` | `0.7543 ± 0.0061` | `0.6520 ± 0.0326` |

该批次验证了仓库扩展的完整计算链路，固定为 `fidelity_level=official_repo_extension` 和
`paper_claim=false`。runner 已支持逐 trial 原子 checkpoint、配置一致性检查和 `--resume`。

### 10.7 论文 Table 2 数据协议

论文第 5.2 节和 Table 2 明确使用六个线性模型数据集：mushrooms、shuttle、pageblocks、
usps、connect-4 和 spambase。每个数据集构造：

```text
class_prior in {0.2, 0.4, 0.6, 0.8}
unlabeled_size in {800, 1600, 3200}
positive_size = 400
test_size = 1000
repetitions = 100
```

仓库 README 仍称 `main_linear_kernel.py` 复现 Table 2，但该入口当前默认 `ijcnn1`；最早
可见提交中也已如此，后续提交只修改 README/删除文件，没有修正入口。项目因此以论文正文
作为 Table 2 数据集权威，以仓库代码作为算法、采样和超参数实现证据。六个数据集的来源、
hash、形状、标签映射和类别计数现已锁定，统一 loader 已接入。

按官方循环从 seed 2018 连续执行 100 次重复的静态审计发现，72 个
`dataset × U × prior` 单元中只有 45 个在所有重复中严格满足声明样本量：USPS 与
connect-4 为 12/12，mushrooms 为 11/12，shuttle 为 6/12，pageblocks 为 1/12，
spambase 为 3/12。根因是官方脚本对训练池和固定 3000 行 holdout 进行无放回切片，却未
校验切片结果长度。例如 pageblocks 划出 holdout 后训练池仅 2473 行，任何 `U=3200`
单元都不可能得到 3200 个未标记样本。兼容模式可记录官方实际长度，严格论文模式则必须
阻止该 trial；批准重采样政策前不能声称完整复现 Table 2。

## 11. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `official_exact`：作者源码已锁定；该字段表示源码可获得性，不表示 toolbox 逐行复制 |
| Implementation status | `NATIVE`；linear baseline 与 official-aligned kernel 双实现 |
| 当前实现可声称 | 统一接口、核 PU 目标/CV/分位数规则、IJCNN1 仓库扩展完整网格与 uLSIF 对照 |
| 当前实现不可声称 | 已复现论文 Table 2 六数据集比较表 |
| 主要风险 | 观察到的 P 受到 `e(x)` 加权；直接来源分类会把 selection preference 与 class posterior 混合 |
| 下一步 | 明确严格/兼容采样政策；先运行 45 个全重复可行单元，单列其余 27 个单元 |
