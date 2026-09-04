# API 参考

> **定位**：本文档是公共 API 的**权威契约参考**——每个入口自含签名、参数表、
> 返回结构与最小示例。需要理解原理与完整走查时，从这里**单向**跳转：
> 术语与整体设计见 [concepts](../concepts/)，操作教程见 [howto](../howto/)，
> 完整示例见 [examples/minimal](../../../examples/minimal/)。
> 反向不会发生：查参数契约不需要离开本页。
>
> **版本**：与 `pyproject.toml` 同步的 **v1.11.0**（随包发布更新；变更记录见
> [dev/release_process.md](../../dev/release_process.md)）。
> **防漂移**：`scripts/check_api_docs.py` 门禁校验本页覆盖每个公共导出与注册算法名，
> 新增公共 API 必须登记本页，否则门禁失败。

## 阅读指引

| 你想做什么 | 去哪儿 |
|---|---|
| 5 分钟跑通第一个实验 | [快速开始](../quickstart.md) |
| 理解 PU 设定 / SCAR-SAR / 选型原理 | [concepts](../concepts/) |
| 按任务走完整教程 | [howto](../howto/)（入口见 [README](../README.md)） |
| **查某个 API 的签名 / 参数 / 返回** | **本页** |
| 看运行示例脚本 | [examples/minimal/](../../../examples/minimal/) |
| 算法原理、公式与复现状态 | [method cards](../../research/method_cards/) |

> 信息分区：**参数契约**在本页；**行为细节**真相源是 docstring（ADR-0013）；
> **走查示例**在 howto/examples；**算法研究内容**在 method card。每类信息
> 只有一个权威位置，互以单向链接联系。

## 分类器与估计器

所有分类器遵守统一契约：`fit(X, y)` + `predict(X)` + `decision_function(X)` +
`get_params()`/`set_params()`；类先验估计器实现 `fit` + `estimate()`。
标签语义由分类器决定（PU 为 `{+1, 0}`，PNU 为 `{+1, -1, 0}`）。

### 注册表索引

下表是注册表索引（快速导航与别名提示）；**每个方法的完整签名、参数表与
返回行为见下方按族划分的分组小节**（参数契约的权威位置在本页，方法卡
不再重复参数契约）。

| 注册名（别名） | 类 | 族 | 核心构造参数摘要 | Method Card |
|---|---|---|---|---|
| `class_prior_estimation`（`cpe`, `pen_l1`） | `ClassPriorEstimator` | class-prior | `sigma` / `reg_lambda` / `theta_grid` / `n_centers` | [class_prior_estimation](../../research/method_cards/class_prior_estimation.md) |
| `recpe`（`re_cpe`） | `ReCPEEstimator` | class-prior | `copy_fraction` / `base_estimator` | [ReCPE](../../research/method_cards/ReCPE.md) |
| `elkan_noto`（`en`） | `ElkanNotoClassifier` | classic | `base_estimator` / `calibration_method` / `n_cv_folds` / `eps` | [Elkan_Noto](../../research/method_cards/Elkan_Noto.md) |
| `upu`（`convex_pu`） | `UPUClassifier` | risk | `class_prior` / `loss` / `reg_lambda` | [Convex uPU](../../research/method_cards/Convex_Formulation_for_PU_DATA_Learning.md) |
| `nnpu`（`nn-pu`） | `NonNegativePUClassifier` | risk | `model` / `encoder` / `class_prior` / `loss` / `optimizer` | [nnPU](../../research/method_cards/nnpu.md) |
| `pnu` | `PNUClassifier` | risk | `class_prior` / `eta` / `reg_lambda` | [PNU](../../research/method_cards/PNU.md) |
| `centroid_pu`（`ldce`） | `LDCEClassifier` | risk | `flip_probability` / `reg_strength` / `centroid_radius` | [LDCE](../../research/method_cards/LDCE.md) |
| `kldce`（`kernelized_ldce`） | `KLDCEClassifier` | risk | `flip_probability` / `sigma` / `reg_strength` | [KLDCE](../../research/method_cards/KLDCE.md) |
| `llsvm` | `LLSVMClassifier` | risk | `alpha` / `beta` / `gamma` / `reg_lambda` / `max_epochs` | [LLSVM](../../research/method_cards/LLSVM.md) |
| `dist_pu`（`distpu`） | `DistPUClassifier` | risk | `class_prior` / `hidden_dim` / `epochs` / `learning_rate` | [Dist-PU](../../research/method_cards/Dist-PU.md) |
| `pusb`（`biased_pu`） | `PUSBClassifier` | bias-aware | `threshold` / `C` / `max_iter` | [PUSB](../../research/method_cards/PUSB.md) |
| `pusb_kernel`（`kernelized_pusb`） | `PUSBKernelClassifier` | bias-aware | `n_basis` / `cv` / `sigma_grid` / `reg_grid` | [PUSB §6.2](../../research/method_cards/PUSB.md) |
| `lbe` | `LBEClassifier` | bias-aware | `max_iter` / `n_em_iter` / `C` | [LBE](../../research/method_cards/LBE.md) |
| `self_pu` | `SelfPUClassifier` | deep | `class_prior` / `backbone` / `warmup_epochs` / `self_paced_start` | [Self-PU](../../research/method_cards/Self-PU.md) |
| `infomax_pu` | `InfoMaxPUClassifier` | deep | `class_prior` / `representation_*` / `classifier_*`（详见下方深度分类器小节） | [InfoMax-PU](../../research/method_cards/InfoMax-PU.md) |
| `weighted_contrastive_pu`（`wconpu`） | `WeightedContrastivePUClassifier` | deep | `class_prior` / `encoder` / `hidden_dim` / `embedding_dim` | [WConPU](../../research/method_cards/WConPU.md) |
| `dgpu` | `DGPUClassifier` | deep | `class_prior` / `generator` / `model` / `hidden_dim` | [DGPU](../../research/method_cards/DGPU.md) |
| —（`km1` / `km2` variant） | `KernelMeanPriorEstimator` | class-prior | `variant="km1"/"km2"` 等（Kernel-mean 类先验，`PUPipeline` 的 `prior_estimator` 支持） | [Kernel_Mean](../../research/method_cards/Kernel_Mean_Class_Prior.md) |

> 注册名可直接用于 `PUPipeline(classifier="...")` 与 CLI `--classifier`；别名大小写不敏感。
> 族列为友好简称；CLI 与 JSON 输出使用枚举值：`class_prior_estimation` / `classic_calibration` / `risk_estimation` / `bias_aware` / `deep_pu`。
> 构造器有必填非 `class_prior` 参数的方法可用 `classifier_params` 补齐；需要 Python 对象协议的参数也可直接传配置好的实例。
> 旧类先验别名 `pe` 仍可解析，但会发出 `FutureWarning`；请迁移到 `class_prior_estimation` 或 `cpe`。

### `sample_weight` 三档语义

所有分类器都保留统一的 `fit(..., sample_weight=None)` 签名，并通过
`get_pu_metadata()["sample_weight_support"]` 明确声明行为：

| 值 | 行为 | 当前分类器 |
|---|---|---|
| `supported` | 权重进入训练目标 | `elkan_noto`, `nnpu`, `pusb`, `infomax_pu`, `weighted_contrastive_pu`, `dgpu` |
| `ignored` | 为 sklearn API 兼容而接受，但不参与训练 | `llsvm`, `upu`, `pnu`, `centroid_pu`, `kldce`, `dist_pu`, `lbe` |
| `not_implemented` | 非 `None` 时抛出 `NotImplementedError` | `pusb_kernel`, `self_pu` |

依赖样本权重时，应在训练前检查该字段；`ignored` 不会把用户传入的权重误报为已生效。

### 类先验估计器

#### `ClassPriorEstimator`（注册名 `class_prior_estimation`，别名 `cpe` / `pen_l1`）

penL1（惩罚 L1 风险）类先验估计器；`sigma=None` 时自适应选择中位数尺度。

```python
ClassPriorEstimator(*, sigma=None, reg_lambda=0.01, theta_grid=None, n_centers=200, standardize=True)
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `sigma` | `float \| None` | `None`（数据自适应中位数距离） | RBF 核宽；显式 `sigma` 保留历史固定尺度行为 |
| `reg_lambda` | `float` | `0.01` | 惩罚系数 |
| `theta_grid` | `np.ndarray \| None` | `None`（自动生成） | 搜索网格（θ 候选值） |
| `n_centers` | `int \| None` | `200` | 子采样的中心数 |
| `standardize` | `bool` | `True` | 训练前对 X 标准化 |

- 实现 `fit(X)` + `estimate()`；旧别名 `pe` 已弃用（`FutureWarning`）。
- 文档：[class_prior_estimation 方法卡](../../research/method_cards/class_prior_estimation.md) · 示例：[05_recpe_pipeline.py](../../../examples/minimal/05_recpe_pipeline.py)（经 `PUPipeline` 使用）

#### `ReCPEEstimator`（注册名 `recpe`，别名 `re_cpe`）

拷贝正例模型（Copy Positive Estimator，ReCPE）类先验估计器。

```python
ReCPEEstimator(copy_fraction=0.1, base_estimator=None, classifier=None, classifier_max_iter=1000)
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `copy_fraction` | `float` | `0.1` | 拷贝进正样本的未标记样例比例 |
| `base_estimator` | estimator \| None | `None` | 实现 `estimate()` 的 CPE 估计器；`None` 使用内置混合比例基线 |
| `classifier` | estimator \| None | `None` | 区分正例与未标记样本的二分类器；其正类概率用于排序未标记样本 |
| `classifier_max_iter` | `int` | `1000` | 默认 logistic 分类器最大迭代数 |

- 文档：[ReCPE 方法卡](../../research/method_cards/ReCPE.md) · 示例：[05_recpe_pipeline.py](../../../examples/minimal/05_recpe_pipeline.py)

#### `KernelMeanPriorEstimator`（注册名 `km1` / `km2` 变体）

核均值（Kernel-mean）类先验估计器（KM1 / KM2 算法）。

```python
KernelMeanPriorEstimator(*, variant="km1", kernel_width=None, width_selection="relative",
                         kernel_width_scale=0.1, width_factors=(0.1, 0.316227766, 1.0, 3.16227766, 10.0),
                         epsilon=0.04, lambda_upper_bound=8.0, km2_final_slope_weight=0.2,
                         max_qp_iter=2000, qp_tolerance=1e-7, max_samples_per_group=None,
                         standardize=False, random_state=None)
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `variant` | `"km1"` / `"km2"` | `"km1"` | 算法变体 |
| `kernel_width` | `float \| None` | `None` | 核宽；`None` 按 `width_selection` 选择 |
| `width_selection` | `"relative"` / `"mmd_grid"` | `"relative"` | 核宽选择策略（相对启发式或 MMD 网格） |
| `kernel_width_scale` | `float` | `0.1` | `relative` 策略的缩放系数 |
| `width_factors` | `tuple[float, ...]` | 默认网格 | `mmd_grid` 的宽度因子网格 |
| `epsilon` | `float` | `0.04` | KM2 松弛参数 |
| `lambda_upper_bound` | `float` | `8.0` | 标度约束上界 |
| `km2_final_slope_weight` | `float` | `0.2` | KM2 最终斜率权重 |
| `max_qp_iter` / `qp_tolerance` | `int` / `float` | `2000` / `1e-7` | QP 求解迭代上限与容差 |
| `max_samples_per_group` | `int \| None` | `None` | 每组最大样本数（`None` 不限制） |
| `standardize` | `bool` | `False` | 训练前标准化 |
| `random_state` | `int \| None` | `None` | 随机种子 |

- `PUPipeline` 的 `prior_estimator="km1"/"km2"` 映射到本估计器。
- 文档：[Kernel_Mean 方法卡](../../research/method_cards/Kernel_Mean_Class_Prior.md)

### 经典包装器

#### `ElkanNotoClassifier`（注册名 `elkan_noto`，别名 `en`）

Elkan-Noto 经验估计包装：单模型校准 + 概率修正或加权重训。

```python
ElkanNotoClassifier(base_estimator=None, calibration_method="sigmoid", n_cv_folds=3,
                    eps=1e-12, mode="weighted_retraining", random_state=None)
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `base_estimator` | sklearn 估计器 | `None`（`LogisticRegression()`） | 基础二分类器；必须实现 `predict_proba` |
| `calibration_method` | `"sigmoid"` / `"isotonic"` | `"sigmoid"` | 基估计器非线性概率输出时的校准方法（线性模型可有自然概率输出时不用） |
| `n_cv_folds` | `int` | `3` | 估计 `c` 的 OOF 分层 CV 折数；须 ≥ 2 |
| `eps` | `float` | `1e-12` | 权重计算中防除零的数值裁剪阈值 |
| `mode` | `"probability_correction"` / `"weighted_retraining"` | `"weighted_retraining"` | 概率修正（`g(x)/c`）或加权重训；工具箱默认加权重训（ADR-0016） |
| `random_state` | `int \| None` | `None` | K 折切分种子 |

- 文档：[Elkan_Noto 方法卡](../../research/method_cards/Elkan_Noto.md) · 示例：[01_elkan_noto.py](../../../examples/minimal/01_elkan_noto.py)

### 风险估计分类器

#### `UPUClassifier`（注册名 `upu`，别名 `convex_pu`）

无偏 PU（uPU / convex PU）分类器：最小化凸 PU 风险。

```python
UPUClassifier(class_prior, *, loss="squared", reg_lambda=1e-3, basis="linear",
              kernel_width=None, n_centers=None, fit_intercept=True,
              max_iter=1000, tol=1e-6, random_state=None)
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `class_prior` | `float` | 必填 | 类先验 π=P(y=1)；可在 `fit` 经 `class_prior` kwarg 覆盖 |
| `loss` | `"double_hinge"` / `"logistic"` / `"squared"` | `"squared"` | 边缘损失变体：C-DH（凸 QP，推荐主变体）/ C-LL（平滑凸，L-BFGS）/ squared（闭式解，最快但惩罚正确大边缘；工具箱默认，ADR-0016） |
| `reg_lambda` | `float` | `1e-3` | α 的 ℓ₂ 正则系数；须 > 0；截距 *b* 不参与正则 |
| `basis` | `"linear"` / `"rbf"` | `"linear"` | 基函数类型；`"rbf"` 用未标记数据子采样 *n_centers* 个中心 |
| `kernel_width` | `float \| None` | `None` | RBF 核宽；`basis="rbf"` 时必填 |
| `n_centers` | `int \| None` | `None`（`min(200, n_U)`） | RBF 中心数；`basis="linear"` 时忽略 |
| `fit_intercept` | `bool` | `True` | 是否拟合截距 *b* |
| `max_iter` / `tol` | `int` / `float` | `1000` / `1e-6` | 优化迭代上限与收敛容差 |
| `random_state` | `int \| None` | `None` | 中心子采样种子 |

- 文档：[Convex uPU 方法卡](../../research/method_cards/Convex_Formulation_for_PU_DATA_Learning.md) · 示例：[02_upu.py](../../../examples/minimal/02_upu.py)

#### `NonNegativePUClassifier`（注册名 `nnpu`，别名 `nn-pu`）

非负 PU（nnPU）分类器（深度网络，PyTorch）。

```python
NonNegativePUClassifier(model=None, *, encoder=None, class_prior=None, loss="sigmoid",
                        beta=0.0, gamma=1.0, optimizer=None, batch_size=256,
                        max_epochs=200, patience=20, random_state=None, device=None)
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `model` | `torch.nn.Module \| None` | `None`（`nn.Linear(d, 1)`） | 输出原始分数 g(x) 的 PyTorch 模型 |
| `encoder` | `torch.nn.Module \| None` | `None` | 特征编码器替代默认原始输入模型；提供时 `model` 作为叠加的分数头，否则创建默认 `nn.Linear(rep_dim, 1)` 头。`None` 保持原行为 |
| `class_prior` | `float \| None` | `None` | π∈(0,1)；也可经 `fit` 提供/覆盖 |
| `loss` | `"sigmoid"` | `"sigmoid"` | 替代损失；MVP 仅支持 sigmoid |
| `beta` | `float` | `0.0` | 非负阈值；须 ≥ 0 |
| `gamma` | `float` | `1.0` | 校正分支步长折扣；∈[0,1] |
| `optimizer` | `torch.optim.Optimizer \| None` | `None`（`Adam(lr=1e-3)`） | 优化器 |
| `batch_size` | `int` | `256` | mini-batch 大小；P/U 批独立取 `min(batch_size, n_P/n_U)` |
| `max_epochs` | `int` | `200` | 最大训练轮数 |
| `patience` | `int` | `20` | 早停耐心（仅当 `fit` 传 `validation_data` 时生效） |
| `random_state` / `device` | `int \| None` / `str \| None` | `None` / `None` | 种子与 torch 设备 |

- 文档：[nnPU 方法卡](../../research/method_cards/nnpu.md) · 示例：[03_nnpu.py](../../../examples/minimal/03_nnpu.py)

#### `PNUClassifier`（注册名 `pnu`）

PNU（positive-negative-unlabeled）风险分类器：统一插件式 λ 损失与正则项。

```python
PNUClassifier(class_prior, *, eta=0.0, reg_lambda=1e-3, basis="linear",
              kernel_width=None, n_centers=None, fit_intercept=True, random_state=None)
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `class_prior` | `float` | 必填 | 类先验 θ_P=P(y=1)；须 ∈(0,1) |
| `eta` | `float` | `0.0` | PNU 权衡参数 ∈[-1,1]：`0`=PN（监督）、`+1`=PU、`-1`=NU |
| `reg_lambda` | `float` | `1e-3` | ℓ₂ 正则系数；截距 *b* 不参与 |
| `basis` / `kernel_width` / `n_centers` | 见 uPU | `"linear"` / `None` / `None` | 基函数与 RBF 参数（同 uPU 语义） |
| `fit_intercept` | `bool` | `True` | 经由常数基列实现（pywsl 惯例） |
| `random_state` | `int \| None` | `None` | 中心子采样种子 |

- 文档：[PNU 方法卡](../../research/method_cards/PNU.md) · 示例：[04_pnu.py](../../../examples/minimal/04_pnu.py)

#### `LDCEClassifier`（注册名 `centroid_pu`，别名 `ldce`）

LDCE（Label Density Centroid Estimation）分类器。

```python
LDCEClassifier(flip_probability, *, reg_strength=1.0, centroid_radius=0.1, mom_groups=10,
               covariance_ridge=0.01, learning_rate=0.01, n_inner_iter=50,
               max_iter=10000, tol=1e-6, random_state=None)
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `flip_probability` | `float` | 必填 | 真正例被翻转为观测负例的概率 *h*（截尾率）；∈(0,1) |
| `reg_strength` | `float` | `1.0` | 线性权重 ℓ₂ 正则系数 λ |
| `centroid_radius` | `float` | `0.1` | 质心约束的椭球半径 *b*；工具箱默认 tuned（ADR-0016） |
| `mom_groups` | `int` | `10` | 中位数质心估计的分组数 *g*；`1` 退化为普通均值 |
| `covariance_ridge` | `float` | `1e-2` | 质心协方差对角脊；与 `centroid_radius` 强交互（ADR-0016） |
| `learning_rate` | `float` | `0.01` | 次梯度下降初始步长 |
| `n_inner_iter` | `int` | `50` | 每次外迭代的内层梯度步数 |
| `max_iter` / `tol` | `int` / `float` | `10000` / `1e-6` | 交替优化上限与容差 |
| `random_state` | `int \| None` | `None` | 种子 |

- 文档：[LDCE 方法卡](../../research/method_cards/LDCE.md)

#### `KLDCEClassifier`（注册名 `kldce`，别名 `kernelized_ldce`）

核化 LDCE（Kernelized LDCE，ACP 组合交替求解）。

```python
KLDCEClassifier(flip_probability, *, sigma="scale", reg_strength=1.0, centroid_radius=1.0,
                mom_groups=10, covariance_ridge=0.0, max_acs_iter=50, max_inner_iter=2000,
                inner_tol=1e-6, tol=1e-6, random_state=None)
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `flip_probability` | `float` | 必填 | 截尾率 *h*；∈(0,1) |
| `sigma` | `float \| "scale"` | `"scale"` | RBF 带宽；`"scale"` 用启发式 `σ = 1/sqrt(n_features)` |
| `reg_strength` | `float` | `1.0` | ℓ₂ 正则系数 λ |
| `centroid_radius` | `float` | `1.0` | 椭球半径 *b* |
| `mom_groups` | `int` | `10` | 中位数质心分组数 |
| `covariance_ridge` | `float` | `0.0` | 质心协方差对角脊；`0.0` 与论文一致，>0 为数值稳定化变体 |
| `max_acs_iter` / `max_inner_iter` / `inner_tol` | `int` / `int` / `float` | `50` / `2000` / `1e-6` | ACS 外循环上限、内层 QP 对更新上限与 KKT 容差 |
| `tol` / `random_state` | `float` / `int \| None` | `1e-6` / `None` | 收敛容差与种子 |

- 文档：[KLDCE 方法卡](../../research/method_cards/KLDCE.md)

#### `LLSVMClassifier`（注册名 `llsvm`）

标签相关线性 SVM（LLSVM，LPSVM 变体）。

```python
LLSVMClassifier(*, alpha=2.0, beta=1.0, gamma=10.0, squash_scale=10.0, reg_lambda=1.0,
                learning_rate=5e-6, max_epochs=3000, n_batches=20, fit_intercept=True,
                intercept_scale=10.0, shuffle=True, random_state=None, early_stopping=True,
                patience=100, tol=5e-4, min_epochs=200)
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `alpha` | `float` | `2.0` | 正类平方铰链损失权重 |
| `beta` | `float` | `1.0` | 未标记 hat 损失权重 |
| `gamma` | `float` | `10.0` | 标签校准损失权重 |
| `squash_scale` | `float` | `10.0` | squash 函数缩放参数 *A*（A/π·arctan(f)） |
| `reg_lambda` | `float` | `1.0` | ℓ₂ 正则强度 |
| `learning_rate` | `float` | `5e-6` | 固定 SGD 步长 |
| `max_epochs` | `int` | `3000` | 最大训练轮数 |
| `n_batches` | `int` | `20` | 每 epoch 的 mini-batch 数 |
| `fit_intercept` / `intercept_scale` | `bool` / `float` | `True` / `10.0` | 常数特征增广实现截距及其值 |
| `shuffle` | `bool` | `True` | 每 epoch 是否打乱 |
| `random_state` | `int \| None` | `None` | 初始化与打乱种子 |
| `early_stopping` / `patience` / `tol` / `min_epochs` | `bool`/`int`/`float`/`int` | `True` / `100` / `5e-4` / `200` | 全数据目标平台期早停（官方代码固定 3000 轮，故默认开启） |

- `get_pu_metadata()["n_epochs"]` 反映实际停止轮数。
- 文档：[LLSVM 方法卡](../../research/method_cards/LLSVM.md)

#### `DistPUClassifier`（注册名 `dist_pu`，别名 `distpu`）

Dist-PU 小 MLP 分类器（标签分布目标）。

```python
DistPUClassifier(class_prior, *, hidden_dim=64, epochs=100, batch_size=128, learning_rate=1e-3,
                 alignment_weight=1.0, entropy_weight=0.05, mixup_weight=0.1,
                 random_state=0, device=None)
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `class_prior` | `float` | 必填 | 类先验 |
| `hidden_dim` | `int` | `64` | 隐藏层维度 |
| `epochs` | `int` | `100` | 训练轮数 |
| `batch_size` | `int` | `128` | mini-batch 大小 |
| `learning_rate` | `float` | `1e-3` | 学习率 |
| `alignment_weight` | `float` | `1.0` | 分布对齐损失权重 |
| `entropy_weight` | `float` | `0.05` | 熵损失权重 |
| `mixup_weight` | `float` | `0.1` | Mixup 损失权重 |
| `random_state` / `device` | `int \| None` / `str \| None` | `0` / `None` | 种子与 torch 设备 |

- 文档：[Dist-PU 方法卡](../../research/method_cards/Dist-PU.md)

### Bias-Aware 分类器

#### `PUSBClassifier`（注册名 `pusb`，别名 `biased_pu`）

Selection-bias 鲁棒后验排序（PUSB）分类器。

```python
PUSBClassifier(*, threshold=0.5, C=1.0, max_iter=1000)
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `threshold` | `float` | `0.5` | 决策阈值（sigmoid 化概率阈值）；`predict` 冻结为先验分位数 `threshold_`（单样本稳定） |
| `C` | `float` | `1.0` | 正则化强度 |
| `max_iter` | `int` | `1000` | 优化最大迭代 |

- 文档：[PUSB 方法卡](../../research/method_cards/PUSB.md)

#### `PUSBKernelClassifier`（注册名 `pusb_kernel`，别名 `kernelized_pusb`）

官方对齐的 RBF PUSB（确定性 CV）。

```python
PUSBKernelClassifier(*, n_basis=300, cv=5, sigma_grid=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
                     reg_grid=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
                     random_state=2018, max_iter=200, tol=1e-5)
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `n_basis` | `int` | `300` | 核基函数数 |
| `cv` | `int` | `5` | 确定性 CV 折数 |
| `sigma_grid` | `Sequence[float]` | 官方网格 | RBF 带宽候选 |
| `reg_grid` | `Sequence[float]` | 官方网格 | 正则候选 |
| `random_state` | `int \| None` | `2018` | 种子（官方值） |
| `max_iter` / `tol` | `int` / `float` | `200` / `1e-5` | 迭代上限与容差 |

- 正则梯度与官方释放实现一致（`0.5·reg_lambda·‖coef‖²`）。
- 文档：[PUSB 方法卡 §6.2](../../research/method_cards/PUSB.md)

#### `LBEClassifier`（注册名 `lbe`）

LBE（Label Bias Estimation）分类器：估计类后验与实例相关标记倾向。

```python
LBEClassifier(*, max_iter=1000, n_em_iter=20, C=1.0)
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `max_iter` | `int` | `1000` | 主优化最大迭代 |
| `n_em_iter` | `int` | `20` | EM 迭代数 |
| `C` | `float` | `1.0` | 正则化强度 |

- 文档：[LBE 方法卡](../../research/method_cards/LBE.md)

### 深度分类器

#### `SelfPUClassifier`（注册名 `self_pu`）

Self-PU 分类器（概率重加权 / 蒸馏 / 高置信度标记）。

```python
SelfPUClassifier(class_prior, *, backbone=None, hidden_dim=128, warmup_epochs=10,
                 self_paced_start=10, self_paced_end=50, distill_start=50, max_epochs=200,
                 max_trust_ratio=0.25, pace_1=0.2, pace_2=0.3, meta_step_size=1e-3,
                 reweight_gamma=1/16, distillation_alpha=10.0, ema_decay=0.99,
                 student_loss_weight=1.0, teacher_loss_weight=1.0, batch_size=256,
                 learning_rate=1e-3, weight_decay=0.0, threshold=0.5,
                 require_validation=False, random_state=None, device=None)
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `class_prior` | `float` | 必填 | 类先验 |
| `backbone` | `Any \| None` | `None`（内置 MLP） | 特征骨干（表格 MLP；数据画像特征维度由 fit 推断） |
| `hidden_dim` | `int` | `128` | 隐藏层维度 |
| `warmup_epochs` | `int` | `10` | 预热轮数 |
| `self_paced_start` / `self_paced_end` | `int` | `10` / `50` | 自步长标定轮数区间 |
| `distill_start` | `int` | `50` | 蒸馏开始轮数 |
| `max_epochs` | `int` | `200` | 最大训练轮数 |
| `max_trust_ratio` | `float` | `0.25` | 最大可信比例 |
| `pace_1` / `pace_2` | `float` | `0.20` / `0.30` | 自步进阈值 |
| `meta_step_size` | `float` | `1e-3` | 元学习步长 |
| `reweight_gamma` | `float` | `1/16` | 重加权折扣 |
| `distillation_alpha` | `float` | `10.0` | 蒸馏损失权重 |
| `ema_decay` | `float` | `0.99` | 教师模型 EMA 衰减 |
| `student_loss_weight` / `teacher_loss_weight` | `float` | `1.0` / `1.0` | 学生/教师损失权重 |
| `batch_size` / `learning_rate` / `weight_decay` | `int` / `float` / `float` | `256` / `1e-3` / `0.0` | 训练设置 |
| `threshold` | `float` | `0.5` | 决策阈值（未调谐，见方法卡） |
| `require_validation` | `bool` | `False` | 是否要求 `fit` 传 `validation_data`（教师选择需要） |
| `random_state` / `device` | `int \| None` / `str \| None` | `None` / `None` | 种子与设备 |

- `fit(..., validation_data=...)` 提供 clean validation 时启用元重加权与验证基教师选择。
- 文档：[Self-PU 方法卡](../../research/method_cards/Self-PU.md) · 示例：[10_self_pu.py](../../../examples/minimal/10_self_pu.py)

#### `InfoMaxPUClassifier`（注册名 `infomax_pu`）

深度 PU 分类器（PURL 表示学习 → 类先验估计 → nnPU 分类）；构造参数较多，
需要细粒度控制时直接传实例给 `PUPipeline`。

```python
InfoMaxPUClassifier(*, class_prior=None, representation_dim=20, hidden_dim=60,
                    representation_epochs=200, classifier_epochs=200, learning_rate=1e-3,
                    representation_ratio_steps=4, representation_encoder_steps=1,
                    representation_weight_decay=5e-4, representation_batch_norm=False,
                    representation_activation=False, representation_batch_size=None,
                    representation_gradient_noise=0.0, classifier_hidden_dims=(),
                    classifier_batch_norm=False, classifier_optimizer="adam",
                    classifier_learning_rate=1e-3, classifier_weight_decay=0.0,
                    classifier_batch_size=256, prior_estimator=None,
                    random_state=None, encoder=None, device=None)
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `class_prior` | `float \| None` | `None` | 类先验；`None` 时用 `prior_estimator` 估计 |
| `representation_*` | 组合 | 见签名 | 表示学习阶段：`representation_dim`（表示维度）/ `representation_epochs` / `representation_ratio_steps` / `representation_encoder_steps` / `representation_weight_decay` / `representation_batch_norm` / `representation_activation` / `representation_batch_size` / `representation_gradient_noise` |
| `classifier_*` | 组合 | 见签名 | nnPU 分类器阶段：`classifier_hidden_dims` / `classifier_batch_norm` / `classifier_optimizer`（`"adam"` / `"adagrad"`）/ `classifier_learning_rate` / `classifier_weight_decay` / `classifier_batch_size` |
| `prior_estimator` | `BaseEstimator \| None` | `None` | `class_prior=None` 时的先验估计器 |
| `encoder` | `torch.nn.Module \| None` | `None` | 外置编码器（如 `build_encoder("cnn", ...)`）；`None` → 内置 MLP 编码器（向后兼容） |
| `device` | `str \| None` | `None` | torch 设备（`None`/`"auto"` 自动检测） |
| `random_state` / `hidden_dim` / `learning_rate` | — | — | 共用种子、隐藏维度与学习率 |

- 传入外置 `encoder` 时替代内部 `nn.Sequential(Linear...)` 编码部分，`ratio_head_` 接在编码器特征之后；`fit` 放行 4-D NCHW 图像输入。
- 文档：[InfoMax-PU 方法卡](../../research/method_cards/InfoMax-PU.md)

#### `WeightedContrastivePUClassifier`（注册名 `weighted_contrastive_pu`，别名 `wconpu`）

加权对比学习 PU 分类器（对比损失 + 分布匹配）。

```python
WeightedContrastivePUClassifier(class_prior, *, encoder=None, hidden_dim=128, embedding_dim=128,
                                queue_size=8192, temperature=0.07, momentum=0.999,
                                pseudo_label_momentum=0.9, contrastive_weight=0.1,
                                distribution_weight=0.1, hard_negative_quantile=0.25,
                                weak_augmentation=None, strong_augmentation=None, batch_size=256,
                                max_epochs=100, learning_rate=1e-2, optimizer_momentum=0.9,
                                scheduler="none", random_state=None, device=None)
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `class_prior` | `float` | 必填 | 类先验 |
| `encoder` | `torch.nn.Module \| None` | `None` | 特征编码器（图像用 `build_encoder("cnn", ...)`） |
| `hidden_dim` / `embedding_dim` | `int` | `128` / `128` | 隐藏层与嵌入维度 |
| `queue_size` / `temperature` / `momentum` | `int` / `float` / `float` | `8192` / `0.07` / `0.999` | 对比队列、温度与队列动量 |
| `pseudo_label_momentum` | `float` | `0.9` | 伪标签动量 |
| `contrastive_weight` / `distribution_weight` | `float` | `0.1` / `0.1` | 对比/分布损失权重 |
| `hard_negative_quantile` | `float` | `0.25` | 难负例分位数 |
| `weak_augmentation` / `strong_augmentation` | callable \| None | `None` | 弱/强增强变换（None 用默认） |
| `batch_size` / `max_epochs` / `learning_rate` / `optimizer_momentum` | — | `256` / `100` / `1e-2` / `0.9` | 训练设置 |
| `scheduler` | `"none"` / `"cosine_annealing"` | `"none"` | 学习率调度 |
| `random_state` / `device` | — | `None` / `None` | 种子与设备 |

- 文档：[WConPU 方法卡](../../research/method_cards/WConPU.md)

#### `DGPUClassifier`（注册名 `dgpu`）

深度生成式 PU 分类器（GAN 判别器 + 伪标签迭代）。

```python
DGPUClassifier(class_prior, generator, *, model=None, hidden_dim=128, rounds=3,
               initialization_epochs=100, annotation_epochs=100, generated_samples=5000,
               pseudo_label_fraction=0.1, confidence_threshold=0.95, debias_strength=0.8,
               distribution_momentum=0.999, batch_size=256, learning_rate=1e-4,
               weak_augmentation=None, strong_augmentation=None, random_state=None, device=None)
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `class_prior` | `float` | 必填 | 类先验 |
| `generator` | torch 模型 | 必填 | 数据生成器（如 GAN generator） |
| `model` | `torch.nn.Module \| None` | `None` | 判别器/分类器模型（None 内置） |
| `hidden_dim` | `int` | `128` | 隐藏维度 |
| `rounds` | `int` | `3` | 伪标签迭代轮数 |
| `initialization_epochs` / `annotation_epochs` | `int` | `100` / `100` | 初始化与标注轮数 |
| `generated_samples` | `int` | `5000` | 生成样本数 |
| `pseudo_label_fraction` | `float` | `0.1` | 伪标签比例 |
| `confidence_threshold` | `float` | `0.95` | 伪标签置信阈值 |
| `debias_strength` | `float` | `0.8` | 去偏强度 |
| `distribution_momentum` | `float` | `0.999` | 分布动量 |
| `batch_size` / `learning_rate` | — | `256` / `1e-4` | 训练设置 |
| `weak_augmentation` / `strong_augmentation` | callable \| None | `None` | 弱/强增强 |
| `random_state` / `device` | — | `None` / `None` | 种子与设备 |

- 文档：[DGPU 方法卡](../../research/method_cards/DGPU.md) · 示例：[10_self_pu.py](../../../examples/minimal/10_self_pu.py)

## PUPipeline

用法见 [howto/pipeline.md](../howto/pipeline.md)；运行示例见
[05_recpe_pipeline.py](../../../examples/minimal/05_recpe_pipeline.py)（先验估计 + 全流程）。

```python
pipe = PUPipeline(
    classifier="auto",          # 注册方法名 / "auto"（推荐器选算法）/ 分类器实例
    classifier_params=None,      # 显式注册名的构造参数；auto/实例模式不可用
    prior_estimator="pen_l1",   # "pen_l1"/"recpe"/"km1"/"km2"（后两者映射到
                                # KernelMeanPriorEstimator）/ 估计器实例 / None
    cv=5,                       # PU 分层 CV 折数，或自定义 splitter
                                # （默认 cv=None → 解析为 5 折 PUStratifiedKFold）
    metrics=DEFAULT_METRICS,    # 指标名元组，见下方指标表
                                # （默认 metrics=None → DEFAULT_METRICS）
    random_state=42,
    architecture="mlp",         # 深度算法架构："mlp"（表格）/ "cnn"（4-D NCHW 图像，需显式 wconpu/infomax_pu/nnpu）
    backbone="cnn13",           # CNN 骨架：cnn13/resnet18/resnet50（仅 cnn 有效）
    device=None,                # 深度分类器 torch 设备：None/"auto" 自动检测（有 GPU 用 CUDA）
)
report = pipe.fit_evaluate(
    X,
    y_pu,
    y_true=None,
    class_prior=None,
    sample_weight=None,         # 可选；逐 CV 训练折切片并传给最终 refit
    refit=True,
)
```

`refit=False` 只计算交叉验证指标，跳过全量模型重训与模型诊断；此时
`report.final_model` 和 `report.diagnostic` 为 `None`。该模式主要供参数搜索使用。

`sample_weight` 必须是一维、有限、非负且至少有一个正值。非 `None` 时，pipeline 在模型
解析后检查 `SampleWeightSupport`：只有 `supported` 能继续，`ignored` 与
`not_implemented` 均抛 `PipelineError`。报告的
`provenance["sample_weight"]` 保存是否提供、范围、均值和 ESS，不保存整列权重。

`provenance["architecture"]` 为 `"native_mlp"` / `"native_cnn"`；
`provenance["backbone"]` 为 CNN 骨架名（MLP 为 `None`）；`provenance["device"]`
保存 `{"requested", "resolved"}` 两键；`provenance["encoder"]` 为注入 encoder 的
构造摘要（`{"backbone", "in_channels"}`，MLP 无注入时为 `None`）。

### 类先验解析优先级

| 优先级 | 来源 | 报告中的 source |
|---|---|---|
| 1 | `fit_evaluate(class_prior=...)` 显式传入 | `user` |
| 2 | 分类器实例构造参数 `class_prior=...` | `constructor` |
| 3 | `prior_estimator` 自动估计（默认 `"pen_l1"`） | `estimated` |
| 4 | 方法不需要先验 | `none` |

- `prior_estimator=None` 且方法需要先验且未提供 → 抛出 `PipelineError`。
- 画像模块审计"先验 < 标注正样本比例"的不一致（`inconsistent_class_prior`）。
- auto 模式先验估计失败不中断流程：降级为无先验推荐（`report.prior.degraded`）。

### 指标

`DEFAULT_METRICS = ("pu_zero_one_risk", "pu_recall", "pu_estimated_precision", "pu_auc_roc")`。
别名：`pu_risk`、`auc`/`roc_auc`、`recall`、`precision`、`accuracy`、`f1`、
`negative_rate`、`ap`、`bacc`、`brier`、`ece`。

| 指标 | 需要 | basis |
|---|---|---|
| `pu_recall` / `pu_negative_rate` | 仅 `y_pu` + 预测 | `pu_observed` |
| `pu_zero_one_risk` / `pu_estimated_precision` | 预测 + 类先验 | `class_prior_dependent` |
| `pu_auc_roc` / `average_precision` | `y_true` + 连续分数 | `supervised_oracle` |
| `pu_accuracy` / `pu_f1` / `balanced_accuracy` | `y_true` + 预测 | `supervised_oracle` |
| `brier_score` / `expected_calibration_error` | `y_true` + 真实 `predict_proba` | `probability_calibration` |

这些指标函数（含 `calibration_bucket_stats`）均可从 `pu_toolbox.metrics` 直接导入。
缺失输入（无 `y_true`、无连续分数、无 `predict_proba`、无先验）时对应指标跳过并记录原因
（`CVMetric.available=False`），不中断流程；`CVMetric.n_computed` 给出已计算折数
（序列化进 `to_json()`）。

### classifier 选择语义

| `classifier=` | 行为 |
|---|---|
| `"auto"`（默认） | 先估先验 → `recommend_from_profile` 推荐 → 按 rank 扫描选中第一个**可自动实例化**的候选 |
| `"nnpu"` / `"upu"` 等注册名 | 直接使用（大小写不敏感，支持别名），使用 `classifier_params` 配置构造参数，并自动注入 `class_prior` 与 `random_state` |
| `UPUClassifier(...)` 实例 | 原样使用（`clone` 到每折），不注入任何参数 |

构造器有必填非 `class_prior` 参数的方法（如 `"ldce"` 需要 `flip_probability`）可在
显式名称模式下用 `classifier_params` 补齐；`"auto"` 会跳过它们并在
`report.provenance["skipped_candidates"]` 记录原因。`class_prior`、`random_state`、
`encoder` 为流水线管理参数，不能通过 `classifier_params` 覆盖。

### 深度算法与架构选择

`architecture` / `backbone` / `device` 参数契约：

| 参数 | 默认 | 取值 | 语义 |
|---|---|---|---|
| `architecture` | `"mlp"` | `"mlp"` / `"cnn"` | `"cnn"` 需显式深度分类器且其构造签名声明 `encoder` 参数（当前 `wconpu` / `infomax_pu` / `nnpu`）；未声明（如 `self_pu`）、`auto` 或非深度方法配 cnn 抛 `PipelineError` |
| `backbone` | `"cnn13"` | `"cnn13"` / `"resnet18"` / `"resnet50"` | 仅 `architecture="cnn"` 有效；非法值抛 `ValueError` |
| `device` | `None`（auto） | `None`/`"auto"`/`"cpu"`/`"cuda"` 等 | 透传给深度分类器（`_fresh_estimator` 按签名注入）；`None`/`"auto"` 自动检测：torch + CUDA 可用则 `"cuda"`，否则 `"cpu"` |

- 深度算法接入契约：要获得 `architecture="cnn"` 支持，分类器构造签名必须声明
  `encoder` 参数（特征提取器形态，pipeline 注入 `build_encoder` 产物，
  `_fresh_estimator` 按签名守卫注入）；未声明时配 cnn 在构造期即被拒绝
- 显式 `wconpu` / `infomax_pu` / `nnpu`：放行必填参数检查，`class_prior` 按
  「显式 > 估计」顺序注入；`architecture="cnn"` 时 encoder 由 pipeline 在
  `fit_evaluate` 内懒构建（`build_encoder("cnn", backbone=..., in_channels=...)`）
  并注入
- 输入维度：4-D NCHW + 显式深度分类器 + cnn → 正常（prior 估计与数据画像在
  展平视图上进行，CV splitter 按索引切分）；4-D + mlp 或非深度分类器 →
  `PipelineError`；2-D + cnn → `PipelineError`
- deep + `cv>1` 时打印训练成本警告（n_splits+1 次训练），建议减少折数（`cv` 最小为 2）
- `auto` 行为：深度方法无 GPU（`device=None`/`"auto"` 解析为 CPU）或小数据时评分低，不会被实际选中；解析为 CUDA 且数据量大时可能被推荐（其适用场景），选中后 torch 播种与训练成本警告照常生效

### 错误场景

| 场景 | 异常 |
|---|---|
| 无效 classifier / prior 名 | 构造时 `PipelineError`（fail-fast） |
| `"ldce"` 等不可自动实例化 | 构造时 `PipelineError`（提示传实例） |
| 无正样本 / 正样本 < `MIN_POSITIVE_SAMPLES` | `ValidationError` |
| 正样本 < CV 折数 | `ValidationError`（提示减折数） |
| 需要先验且最终缺失 | `PipelineError` |
| 用户传入 `class_prior` ∉ (0, 1) | `ValueError`（与 cv/metrics/architecture 等构造参数一致） |
| 先验估计值 ∉ (0, 1) | auto：降级为无先验推荐；显式：`PipelineError` |
| 先验估计器异常 | auto：降级（`prior.degraded` 记录）；显式：`PipelineError` |
| 未知指标名 / 非法 CV | 构造时 `ValueError` / `TypeError` |
| `sample_weight` 形状/数值非法 | `ValueError` |
| 分类器忽略或未实现 `sample_weight` | `PipelineError`（不会静默训练） |

## 分布漂移 API

用法与解释见[分布漂移指南](../howto/distribution_shift.md)；示例脚本见
[11_distribution_shift.py](../../../examples/minimal/11_distribution_shift.py)、
[12_shift_decision_tools.py](../../../examples/minimal/12_shift_decision_tools.py) 与
[13_dynamic_joint_shift.py](../../../examples/minimal/13_dynamic_joint_shift.py)。

### `analyze_pu_shift`

```python
report = analyze_pu_shift(
    X_source,
    y_source_pu,
    X_target,
    y_target_pu=None,
    alpha=0.1,
    probability_clip=1e-6,
    cv=5,
    random_state=42,
    moderate_auc=0.60,
    high_auc=0.75,
    min_effective_sample_fraction=0.50,
    max_boundary_fraction=0.05,
    min_relative_mass=0.10,
)
```

返回 `PUShiftReport`：

| 字段 | 契约 |
|---|---|
| `domain_auc` | 分层 OOF 域分类 ROC AUC，方向对称化到 `[0.5, 1]` |
| `severity` | `low` / `moderate` / `high`，默认分界 0.60/0.75 |
| `sample_summary` | 两域样本量、展平特征数、已标记正例率和目标 PU 可用性 |
| `weight_summary` | 归一化权重分位数、ESS、概率裁剪率、边界率与归一化前相对质量 |
| `adaptation_ready` | 目标 PU 已提供，且 ESS、边界率、相对质量均通过默认覆盖门槛 |
| `source_importance_weights` | 与源域行对应、均值为 1 的边际相对权重 |
| `issues` | `ProfileIssue` 元组，含问题码、级别、解释和行动建议 |

`report.to_dict()` / `to_json()` 不内嵌全量权重；`save(.csv)` 单独导出权重。
`save(.json/.md)` 保存结构化或人可读报告。权重估计范围固定为
`marginal_covariate`，不代表 `p_target(x,y)/p_source(x,y)` 联合权重。

### `ShiftAwarePUPipeline`

```python
workflow = ShiftAwarePUPipeline(
    pipeline=None,               # 可传配置好的 PUPipeline
    classifier="elkan_noto",    # pipeline=None 时生效
    alpha=0.1,
    shift_cv=5,
    allow_unstable=False,
    cv=5,                        # 其余关键字传给 PUPipeline
    random_state=42,
)
result = workflow.fit_evaluate(
    X_source,
    y_source_pu,
    X_target,
    y_target_pu=y_target_pu,
    y_true_source=None,
    y_true_target=None,
    class_prior=None,
    target_class_prior=None,
    adapt=True,
    refit=True,
)
```

`adapt=True` 必须提供目标 PU 标签，并把漂移报告的源域权重传给 `PUPipeline`；覆盖门禁
失败时默认抛 `PipelineError`。`adapt=False` 执行审计和未加权源域基线。
`ShiftAwarePipelineReport` 分开保存 `shift`、`source_pipeline` 与 `target_metrics`；目标
真值指标仍标为 `supervised_oracle`。其 `guarantee` 固定为
`covariate_shift_only`，目标 PU 标签当前作为适配安全门和目标评估输入，不参与边际域密度比。

### `ShiftAwarePUPipeline.compare`

`compare(...)` 在同一目标集运行未加权和加权两臂并返回 `ShiftComparisonReport`。报告的
`metric_deltas[*].improvement` 已统一方向：正数总表示加权臂更好（risk 会反号）。自动
`recommendation` 只允许使用目标真值 oracle 指标或目标类先验依赖指标；仅有 PU-observed
指标时为 `audit_only`，覆盖门禁失败且未 override 时为 `collect_target_data`。

### `PUShiftMonitor`

```python
monitor = PUShiftMonitor(
    X_reference,
    y_reference_pu,
    alpha=0.1,
    cv=5,
    auc_jump_threshold=0.10,
    label_rate_jump_threshold=0.05,
)
window, shift = monitor.update(
    X_window, y_window_pu=y_window_pu, window_id="2026-08", timestamp=None
)
monitor.save_history("history.json")
```

`ShiftWindow` 保存当前值、相邻窗口 delta、`alert_level` 和 `alert_codes`。窗口 ID 不允许
重复；`load_history` 要求持久化配置与当前监控器完全一致。

### `analyze_domain_assumptions`

分别接收源/目标特征和 PU 标签，以及可选的两个域类先验。先验缺失时每个域独立估计。
返回 `DomainAssumptionReport`，结论为 `stable`、`class_prior_shift`、
`labeling_mechanism_shift`、`both_shift` 或 `inconclusive`。平均标记倾向不识别 SCAR/SAR。
设置 `bootstrap_replicates>=2` 后，会分别对两个域做行级非参数重采样，每轮重新运行先验
估计器，并把 percentile 区间传播到类先验、标记率、平均倾向及三类域差值。`uncertainty`
记录请求、成功和失败 replicate；它反映采样和估计器变化，不覆盖识别假设偏差。

### `analyze_pu_uncertainty`

对已拟合模型构建选择性预测与主动人工复核计划（`PUUncertaintyReport`）。

```python
report = analyze_pu_uncertainty(
    estimator,                        # 已拟合模型（调用 predict/predict_proba）
    X,
    *,
    y_pu=None,                        # 提供后，标记正例不进入查询候选
    y_true=None,                      # 提供后启用真实标签监督指标
    min_confidence=0.5,               # 选择性预测的置信度阈值
    query_budget=0,                   # 查询数量上限（0 = 预算内全部候选）
    query_strategy="uncertainty",     # "uncertainty" / "shift_weighted" / "diverse_uncertainty"
    importance_weight=None,           # shift_weighted 必需（与行对齐的权重）
    random_state=42,
)
```

返回 `PUUncertaintyReport`（`pu_toolbox` 公共导出，dataclass）：

| 字段 | 内容 |
|---|---|
| `positive_probability` | 每样本的正类概率（与行对齐） |
| `uncertainty` | 每样本的不确定性度量（概率边际） |
| `selective_predictions` | 选择性预测集合（低于 `min_confidence` 的样本） |
| `query_indices` | 推荐主动复核的样本索引（按 `query_strategy` 排序） |
| `summary` | 样本数、选择策略、预算、拒绝预测统计的摘要键值 |
| `provenance` | 调用参数与配置记录 |

与行对齐的逐项数值不内嵌 `to_json()`（只存 `summary`）；CSV 序列化保留逐行
概率、不确定性、选择性预测与查询标记。

### `JointShiftPUClassifier`（research）

从 `pu_toolbox.estimators.research` 导入。`fit` 除源域 `X/y_pu` 外还必须显式传入
`X_target`、`y_target_pu`、`class_prior` 和 `target_class_prior`。它不在稳定注册表和
`PUPipeline` 自动选型中；`get_pu_metadata()["guarantee"]` 固定为
`research_joint_shift_approximation`。

### `DynamicJointShiftPUClassifier`（research）

从 `pu_toolbox.estimators.research` 导入。Torch clean-room 路径实现 Kumagai 等人 AISTATS
2025 的式 (13)、(19)–(23) 和 Algorithm 1：权重步骤固定共享特征，仅更新有界权重头；
分类步骤固定当前权重，更新共享特征和分类头。`training_mode="two_step"`、两个 correction
开关用于论文式消融。作者未公开源码，因此元数据是 `clean_room_paper_objective`，不是
`official_exact`；该方法不进入稳定注册表和 `auto`。

`build_joint_shift_estimator(...)` 可构造 `dynamic`、`trpu`、`tepu`、`fine_tune`、`mmd`、
`two_step` 及三种 correction 消融。所有神经对照共用特征/分类头规模和绝对值修正 PU risk。

## PUTuner

`PUTuner(classifier=..., param_grid=..., scoring=..., higher_is_better=None,
metrics=None, **pipeline_params)` 对 `sklearn.model_selection.ParameterGrid` 展开的每个组合
运行完整 `PUPipeline`。`classifier` 必须是显式注册名；`pipeline_params` 可包含 `cv`、
`prior_estimator`、`random_state`、`architecture` 等流水线参数。

`fit(X, y_pu, y_true=None, class_prior=None)` 返回 `TuningResult`：

| 字段 | 含义 |
|---|---|
| `best_params` / `best_score` | 最佳有效组合与 CV 均值 |
| `best_report` | 最佳组合的完整 `PipelineReport`，含全量重训模型 |
| `trials` | 全部 `TuningTrial`；状态为 `ok` / `unavailable` / `failed` |
| `scoring` | 规范化后的指标名 |
| `higher_is_better` | 选择方向；默认仅 `pu_zero_one_risk` 取最小 |

所有 trial 都无法计算选择指标时抛 `PipelineError`。完整示例见
[模型调整指南](../howto/model_tuning.md)。

## PUModelComparator

`PUModelComparator(classifiers=..., classifier_params=None, scoring=...,
higher_is_better=None, metrics=None, **pipeline_params)` 在相同的 PU-aware CV 设置下比较
至少两个显式注册名。`fit(X, y_pu, y_true=None, class_prior=None)` 返回
`ModelComparisonResult`，包含 `best_classifier`、`best_score`、逐模型 `trials` 和已全量
拟合的 `best_report`。失败模型被隔离记录，非最佳模型只执行 CV。

## 进度与取消

`PUPipeline.fit_evaluate`、`PUTuner.fit` 和 `PUModelComparator.fit` 均接受
`progress_callback=` 与 `cancellation_token=`。回调收到 `ProgressUpdate`（`stage`、
`completed`、`total`、`message`、`fraction`）；`CancellationToken.cancel()` 发出线程
安全的协作式取消信号，并在下一个安全边界抛出 `RunCancelledError`。正在执行的单次模型
`fit` 不会被强制终止。

## build_encoder

深度分类器的统一编码器构建入口（`pu_toolbox/estimators/deep/vision.py`，
亦从包根 `from pu_toolbox import build_encoder` 公共导出），PUPipeline 在
`architecture="cnn"` 时内部调用；也可手动传给分类器。

```python
encoder = build_encoder(
    architecture,             # "mlp" | "cnn"
    *,
    backbone="cnn13",         # "cnn13" / "resnet18" / "resnet50"（仅 cnn）
    in_channels,              # 图像通道数（如 RGB=3）
    normalization_mean=None,  # 每通道均值；默认 0.5
    normalization_std=None,   # 每通道标准差；默认 0.5
)
```

- `"mlp"` → 返回 `None`（分类器内置 MLP 路径，表格数据）
- `"cnn"` → 返回 `build_wconpu_backbone(...)` 图像骨干（4-D NCHW 输入，
  内嵌通道标准化）
- 非法 `architecture` → `ValueError`

## profile_pu_data

用法见 [howto/data_profiling.md](../howto/data_profiling.md)；示例：
[07_data_profiling.py](../../../examples/minimal/07_data_profiling.py)。

```python
report = profile_pu_data(
    X,
    y_pu,
    y_true=None,               # 提供后启用可识别的审计模式
    class_prior=None,
    min_labeled_positives=30,
    max_unlabeled_to_positive=100.0,
    low_variance_threshold=1e-12,
    scar_auc_threshold=0.65,
    cv=5,
    random_state=42,
)
```

所有 `y_pu == 1` 的样本必须在 `y_true` 中也是正类，否则接口拒绝输入。

### 返回 PUDataProfile

| 字段 | 内容 |
|---|---|
| `summary` | 样本数、特征数、已标正例数、未标记数、PU 比例、稀疏性、类先验及隐含标记频率 |
| `feature_statistics` | 缺失/无穷值数量、常数列和低方差列索引 |
| `selection_diagnostic` | AUC、状态、证据来源、可识别性、实际折数和评估样本数 |
| `issues` | 带稳定 `code`、严重级别、解释和行动建议的问题列表 |
| `assumption_hints` | 对 SCAR/SAR、类先验依赖和结果边界的说明 |

辅助接口：`report.has_errors` / `report.has_warnings` / `report.format_text()` / `report.to_dict()`。

`selection_diagnostic["status"]` 取值：`plausible`（CV AUC 不高于阈值）、`at_risk`
（AUC 高于阈值）、`inconclusive`（某一组少于两个样本，或存在非有限特征值）。

### 问题代码

| Code | 级别 | 建议 |
|---|---|---|
| `no_labeled_positives` | error | 检查标签编码或收集可信正例 |
| `no_unlabeled_samples` | error | 改用监督学习流程或补充未标记总体 |
| `missing_features` / `infinite_features` | error | 在仅由训练折拟合的 pipeline 中清理或插补 |
| `few_labeled_positives` | warning | 使用重复验证、置信区间并尽量补充正例 |
| `extreme_pu_imbalance` | warning | 使用 PU 分层切分和抗不平衡指标 |
| `constant_features` | warning | 在训练 pipeline 内删除常数列 |
| `high_dimensional_data` | warning | 使用正则化模型，避免在全数据上预处理 |
| `inconsistent_class_prior` | warning | 复核类先验、采样总体和标签定义 |
| `sar_signal` | warning | 审计正例支持 SAR；优先评估 PUSB/LBE 并做敏感性分析 |
| `observed_selection_signal` | info | 只有非识别性信号；补充审计或标记策略信息 |
| `low_variance_features` | info | 复核特征缩放；低方差列可能不携带信号 |
| `selection_diagnostic_inconclusive` | info | 评估组样本不足或特征非有限；补充审计或可信正例标签 |

## recommend_methods / recommend_from_profile

用法与决策原理见 [concepts/method_selection.md](../concepts/method_selection.md)。

```python
result = recommend_methods(
    X, y_pu,
    scenario=None,       # Scenario 或字符串，如 "case_control"
    assumption=None,     # Assumption 或字符串，如 "scar"
    class_prior=None,    # 已知 P(Y=1)；None 时排除需先验的方法
    has_gpu=False,
    top_k=5,
    random_state=42,
    config=None,         # ScoringConfig；默认 DEFAULT_CONFIG（从 pu_toolbox.advisor 导出）
)

result = recommend_from_profile(
    profile,             # 已有 PUDataProfile，跳过重复 profiling
    scenario=None, assumption=None, class_prior=None,
    class_prior_source=None,  # 先验来源说明（如 "user"/"estimated"），只影响 global-warning 措辞，不写入 provenance
    has_gpu=False, top_k=5, config=None,
)
```

返回 `RecommendationResult`：`candidates`（`MethodCandidate`：name/score/rank/reasons/warnings/metadata）、
`filters_applied`、`global_warnings`、`provenance`。导出：`result.to_json()` / `to_markdown()` / `save(path)`。

## 数据生成

用法见 [howto/sar_simulation.md](../howto/sar_simulation.md)；示例：
[06_sar_simulation.py](../../../examples/minimal/06_sar_simulation.py)。

```python
propensity = make_sar_propensity(X, y_true, mechanism="linear",
                                 label_frequency=0.4, strength=1.5)
y_pu, propensity = make_sar_labels(X, y_true, mechanism="nonlinear",
                                   label_frequency=0.4, random_state=42,
                                   return_propensity=True)
X, y_pu, y_true, propensity = make_sar_dataset(
    n_samples=1000, n_features=8, class_prior=0.3, separation=2.0,
    mechanism="linear", label_frequency=0.4, strength=1.5, random_state=42)
```

- `mechanism`：`"scar"` / `"linear"` / `"nonlinear"`（定义见 [concepts/scar_sar.md](../concepts/scar_sar.md)）。
  不传时默认 `"linear"`（SAR，标记依赖特征）并发出 `UserWarning`；类先验估计器假设 SCAR，
  SCAR 场景需显式传 `mechanism="scar"`。
- `label_frequency`：正类 propensity 的目标均值（校准），不是抽样后的精确比例。
- `make_sar_labels` 默认 `ensure_labeled=True`：小样本抽样未选中任何正类时选择
  propensity 最高的真实正类，保证下游可训练。
- 返回值 `propensity` 表示 `P(S=1|Y,X)`，真实负类位置固定为零。
- CLI 演示数据 `make-demo-data`（`--n` 每类样本数、`--c` 标注概率、`--separation`
  默认 1.0）内部使用 `make_scar_dataset`（`make_scar_dataset(n, c, n_features=5,
  separation=1.0, random_state=None)` → `(X, y_pu, class_prior)`，机制固定为
  SCAR；默认分离度避免强分离下类先验估计系统性低估）。

## analyze_pu_sensitivity

用法见 [howto/sensitivity_analysis.md](../howto/sensitivity_analysis.md)；示例：
[09_sensitivity_analysis.py](../../../examples/minimal/09_sensitivity_analysis.py)。

```python
analysis = analyze_pu_sensitivity(
    y_valid,
    y_pred,                              # 必须 {0,1}
    scores=None,                         # 有限一维数组；0 为阈值计算 PU zero-one risk
    class_priors=None,                   # 每项在 (0, 1)
    label_propensities=None,             # 每项在 (0, 1]
)
```

`class_priors` 与 `label_propensities` 至少提供一个；`y_pu` 必须同时包含
labeled-positive 和 unlabeled 组。返回 `PUSensitivityAnalysis`，每点含：

| 字段 | 含义 |
|---|---|
| `axis`, `value` | 当前扫动的参数及其假定值 |
| `class_prior` | 假定或由 `q/c̄` 推出的类先验 |
| `label_propensity` | 假定或由 `q/π` 推出的平均 propensity |
| `is_consistent` | 是否满足概率边界和观测恒等式 |
| `consistency_reason` | 不一致时的直接原因 |
| `pu_estimated_precision` | `min(π·Recall̂_P/P̂(Ŷ=1), 1)` |
| `pu_zero_one_risk` | `2π·FNR̂_P + FPR̂_U − π` |

`metric_ranges` 仅汇总 `is_consistent=True` 且指标可用的点。导出：
`analysis.to_frame(axis=...)` / `analysis.save(path)`（JSON 严格模式，无 NaN/Infinity）。

## build_diagnostic_report

用法见 [howto/diagnostic_reports.md](../howto/diagnostic_reports.md)；示例：
[08_diagnostic_report.py](../../../examples/minimal/08_diagnostic_report.py)。三种模式：

```python
# 纯数据（只运行 profile_pu_data；预测类指标 basis=unavailable）
report = build_diagnostic_report(X, y_pu, class_prior=0.30, random_state=42)

# 已拟合 estimator（调用 predict，优先 decision_function，回退 predict_proba[:, 1]；
# 不调用 fit；未拟合 estimator 抛 ValueError）
report = build_diagnostic_report(X_valid, y_valid, estimator=classifier,
                                 class_prior=0.30)

# 显式输出（estimator 与 y_pred/scores 互斥）
report = build_diagnostic_report(X_valid, y_valid, y_pred=predictions,
                                 scores=decision_scores, class_prior=0.30)
```

`y_true`（可选）：启用可识别的 selection-dependence 检查并计算
`accuracy`/`f1`/`roc_auc`（标为 `supervised_oracle`）；不会传给 estimator。

### 指标证据级别

| `basis` | 指标来源 | 是否需要额外假设 |
|---|---|---|
| `pu_observed` | PU 标签和模型预测 | 不需要真实标签 |
| `class_prior_dependent` | PU 标签、预测和类先验 | 数值随类先验改变 |
| `supervised_oracle` | 真实标签和预测/分数 | 仅用于有真值评价 |
| `unavailable` | 输入不足或数值无效 | `reason` 解释缺失原因 |

固定指标：`labeled_positive_recall` / `unlabeled_negative_rate` / `predicted_positive_rate`
（`pu_observed`）；`pu_estimated_precision` / `pu_zero_one_risk`
（`class_prior_dependent`）；`accuracy` / `f1` / `roc_auc`（`supervised_oracle`）。

### 报告问题代码（模型输出检查）

| Code | 级别 | 说明 |
|---|---|---|
| `constant_predictions` | warning | 全部样本被预测为同一类；复核阈值、先验和收敛状态 |
| `nonfinite_scores` | error | 分数包含 `NaN/inf`；监督 AUC 被标为不可用 |
| `constant_scores` | warning | 模型没有提供排序信息；检查训练和特征变化 |

`PUDiagnosticReport` 顶层 `schema_version` 当前为 `1.0`；`save()` 按 `.json`/`.md` 后缀推断格式，
JSON 严格编码（未定义值转 `null`）。

## 错误与异常

**所有权**：所有工具箱异常都继承自 `PULearningError`（`pu_toolbox.core.exceptions`），
调用方用 `except PULearningError` 即可统一捕获。异常消息的权威文本在代码中定义
（单一真相源）；本表只回答"什么场景抛什么异常、怎么处置"。

| 异常 | 定义位置 | 触发场景（示例） |
|---|---|---|
| `PULearningError` | `core/exceptions.py` | 基类，所有工具箱异常父类；不作一次性抛出的单个类型使用 |
| `ValidationError` | `core/exceptions.py` | 输入校验失败：`y` 编码非法、无正样本、正样本 < `MIN_POSITIVE_SAMPLES`、正样本少于 CV 折数、`sample_weight` 形状/数值非法 |
| `NotFittedError` | `core/exceptions.py` | 未 `fit` 即调用 `predict`/`decision_function`/`score_samples`（与 sklearn 兼容） |
| `RegistryError` | `core/exceptions.py` | 注册名或别名不存在、重复注册、参数校验失败（如先验语义不一致） |
| `PipelineError` | `core/exceptions.py` | 流水线层组合错误：无效 classifier/prior 名、方法不可自动实例化、需要先验但最终缺失、`sample_weight` 被 `ignored`/`not_implemented`、`architecture="cnn"` 与方法不兼容等 |
| `RunCancelledError` | `core/exceptions.py` | 协作式取消请求在下一个安全边界被抛出 |

> 全部异常定义统一于 `core/exceptions.py`（单一真相源）；`PipelineError` ／`RunCancelledError`
> 在 `pu_toolbox.workflows`／`pu_toolbox.progress` 经再导出保留公共路径，对调用方无行为差异
> ——统一捕获 `PULearningError` 即可。

**各 API 的错误与问题码索引**（详细表在各节内，此处导航）：

| API | 错误场景表 | 问题码表 |
|---|---|---|
| `PUPipeline` | [错误场景](#错误场景) | — |
| `profile_pu_data` | 接口拒绝（如 `y_true` 不包含标记正例） | [问题代码](#问题代码)（`no_labeled_positives` / `sar_signal` / …） |
| `build_diagnostic_report` | 未拟合 estimator 抛 `ValueError`；`estimator` 与 `y_pred`/`scores` 互斥 | 报告问题代码表（本节内上方，`constant_predictions` / …） |
| `PUTuner` / 漂移管线 | 无法计算任何选择指标时抛 `PipelineError`；覆盖门禁失败抛 `PipelineError` | — |

> 环境/安装类问题（版本矩阵、CI 职责、Python 支持）见 [dev/compatibility.md](../../dev/compatibility.md)。
