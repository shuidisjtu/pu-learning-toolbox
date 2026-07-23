# Method Card: PUSB

## 1. 待办与注意

### 1.1 待办

- [x] 明确 PUSB 处理的是 selection-biased PU，而不是 SCAR 下的普通 PU。
- [x] 在 registry 中标记为 `BIAS_AWARE`、`SAR`、`SELECTION_BIASED`。
- [x] 提供统一的 sklearn-compatible `PUSBClassifier`，支持 `fit/predict/decision_function/predict_proba`。
- [x] 在方法卡中区分论文的 partial-identification/ranking 结论与当前工程实现。
- [ ] 将官方 PUSB 的 partial-identification 目标和非参数 scoring procedure 逐式移植。
- [ ] 在 SAR 合成数据上验证 ranking preservation，而不是只验证分类准确率。
- [ ] 复现官方仓库的实验和与 uPU、nnPU、Dist-PU 的比较。

### 1.2 注意

- PUSB 讨论的是已标记正样本有选择偏差的情形：`P(x|y=1,s=1)` 不一定等于 `P(x|y=1)`。
- 在 selection bias 下，普通 PU 风险分解使用的 SCAR 常数 propensity `c` 不成立；直接套用 Elkan-Noto、uPU 或 nnPU 会引入不可忽略偏差。
- 论文的主要识别目标是后验排序/部分识别的分类器，不等于完整恢复 `P(y=1|x)` 数值。
- `predict_proba` 在当前项目中返回 logistic ranking model 的工程概率，不能宣称是 PUSB 理论保证的校准后验。
- 当前 `PUSBClassifier` 是一个可运行的 linear ranking baseline，不是官方 PUSB 完整优化器；正式 benchmark 前必须完成第 9 节的替换工作。

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
| Requires class prior | 论文目标不要求用户直接给定 `pi`；当前接口允许可选传入但不依赖它 |
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
| `tau` | 工程决策阈值 | `threshold` |

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
f_{logistic}(x)=w^T\operatorname{scale}(x)+b,
```

以及：

```math
q_{logistic}(x)=\operatorname{sigmoid}(f_{logistic}(x)).
```

它是 PUSB family 的可运行接口 baseline，不等价于论文完整 partial-identification solver。该边界必须在实验报告中保留。

## 6. 当前实现的算法流程

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

## 7. 参数与项目协议

| 参数 | 当前默认值 | 含义 | 论文是否规定 |
|---|---:|---|---|
| `threshold` | 0.5 | 工程预测阈值 | 不是论文识别结论 |
| `C` | 1.0 | logistic L2 正则倒数 | 项目适配 |
| `max_iter` | 1000 | sklearn 优化最大迭代次数 | 项目适配 |
| `sample_weight` | `None` | P/U 样本权重 | 项目扩展 |
| `class_prior` | `None` | 可选元数据，不参与当前 baseline 优化 | 当前不要求 |

正式 PUSB 复现还需要记录：selection propensity 的生成函数、SAR 参数、正类先验、P/U 样本量、真实测试标签和排序评估指标。

## 8. API 接口与项目落点

### 8.1 构造函数

```python
class PUSBClassifier(BasePUClassifier):
    def __init__(self, *, threshold=0.5, C=1.0, max_iter=1000):
        ...
```

### 8.2 API 语义

| API / 属性 | 约定 |
|---|---|
| `fit(X, y_pu, *, class_prior=None, sample_weight=None)` | `y_pu=1` 是观察到的正样本，`0` 是 U；可接受样本权重 |
| `decision_function(X)` | score 越高，越倾向于正类；用于排序 |
| `predict(X)` | `predict_proba(X)[:,1] >= threshold` |
| `predict_proba(X)` | logistic 工程概率，不是理论校准后验 |
| `model_` | `StandardScaler + LogisticRegression` pipeline |
| `get_pu_metadata()` | 返回 `SAR`、`selection_biased` 和 fitted 状态 |

### 8.3 模块落点

| 模块 | 责任 | 状态 |
|---|---|---|
| `pu_toolbox/estimators/bias_aware/pusb.py` | 当前 linear ranking baseline | ✅ |
| `pu_toolbox/estimators/bias_aware/__init__.py` | 导出 `PUSBClassifier` | ✅ |
| `pu_toolbox/registry/builtin_methods.py` | `pusb` 元数据和 lazy binding | ✅ |
| `tests/unit/estimators/test_bias_aware.py` | API smoke test | ✅ |
| `benchmarks/sar/pusb/` | SAR ranking benchmark | ⏳ |
| `pu_toolbox/estimators/bias_aware/pusb_official.py` | 论文算法逐式移植 | ⏳ |

## 9. 测试与验收标准

### 9.1 API 测试

- 缺少正样本或缺少未标记样本时拒绝输入。
- `threshold` 不在 `(0,1)` 或 `C <= 0` 时抛出 `ValueError`。
- `predict` 只返回 `{0,1}`。
- `decision_function` 和 `predict_proba` 形状正确且为有限值。
- `sample_weight` 能正确路由到 pipeline 中的 logistic step。
- registry 中 `pusb` 的 `assumption` 含 `SAR`，`backend=SKLEARN`，状态为 `NATIVE`。

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

## 10. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `official_exact` |
| Implementation status | `NATIVE`，但当前为 clean-room ranking baseline |
| 当前实现可声称 | 统一接口、SAR 元数据、可运行 P/U scoring baseline |
| 当前实现不可声称 | 已复现论文 partial-identification 理论或官方实验结果 |
| 主要风险 | 观察到的 P 受到 `e(x)` 加权；直接来源分类会把 selection preference 与 class posterior 混合 |
| 下一步 | 从官方仓库整理论文目标函数、识别区间、阈值选择和实验 protocol，再替换 `model_` 训练逻辑 |
