# Method Card: PNU Semi-Supervised Classification

## 1. 待办与注意

### 1.1 待办

- 实现前先定稿 P/N/U 数据协议：PNU 必须同时使用正、负、未标记样本；当前项目通用 `y_pu` 主要覆盖 P/U，不能无说明地复用为 P/N/U。
- 实现 `PNUClassifier`：组合 PN、PU、NU 的**经验风险**训练二分类器；建议公共 estimator 放在 `pu_toolbox/estimators/risk/pnu.py`。
- 先实现凸版本（平方损失）；非凸 ramp-loss + CCCP 作为可选扩展。
- 将 `class_prior` 设为必填，或显式接入已有类先验估计器；禁止静默把它设为 `0.5`。
- 实现 `eta` 的交叉验证选择；论文的方差公式仅在特定条件下给出理论依据，不能作为无验证数据时的通用默认值。
- 写测试：端点退化、经验风险等价、输入校验、adapter smoke、合成数据上的端到端训练。

### 1.2 注意

- PNU 不是“仅有 P 与 U”的 PU 分类器：训练必须同时有正、负、未标记样本。
- 论文假设三个集合分别独立同分布于 $`p(x\mid y=+1)`$、$`p(x\mid y=-1)`$、$`p(x)`$。若项目的 `y_pu` 只有 `+1/0`，不能直接使用；必须额外提供带负标签数据或先扩展数据协议。
- 该方法需要正类先验 $`\theta_P`$，论文实验中将其视为已知或先估计；先验估计误差不在本文保证范围内。
- 无偏风险在灵活模型下可出现负风险/过拟合。论文结论指出未来可结合 nnPU；当前 toolbox 已有 nnPU，建议把“PNU + 非负校正”留为后续扩展，不能宣称本文已给出该公式。**[项目适配]**
- 项目资源清单将 PNU 标为 `official_exact`，官方源码位于 `pywsl`；实现时应以论文公式为数学权威，以 `pywsl` 为源码级复现参考。不得再将 PNU 记为仅论文依据的方法。

---

## 2. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Semi-Supervised Classification Based on Classification from Positive and Unlabeled Data |
| Authors | Tomoya Sakai, Marthinus Christoffel du Plessis, Gang Niu, Masashi Sugiyama |
| Venue | ICML / PMLR 70 |
| Year | 2017 |
| Family | `risk_estimation` / semi-supervised PU extension |
| Setting | 有标注正集 $`X_P`$、负集 $`X_N`$ 与未标记集 $`X_U`$ 的二分类 |
| Requires class prior | `True`：$`\theta_P`$（$`\theta_N=1-\theta_P`$） |
| Requires propensity | `False` |
| Requires negative samples | `True` |
| GPU required | `False`（native PyTorch 实现可支持 GPU） |

### Assumptions

```math
X_P\overset{i.i.d.}{\sim}p_P(x)=p(x\mid y=+1),\quad
X_N\overset{i.i.d.}{\sim}p_N(x)=p(x\mid y=-1),\quad
X_U\overset{i.i.d.}{\sim}p(x)=\theta_Pp_P(x)+\theta_Np_N(x).
```

不要求 cluster / manifold / low-density separation 等传统半监督分布假设。

---

## 3. 符号与记号

| 论文符号 | 含义 | 开发侧对应 |
|---|---|---|
| $`g(x)`$ | 实值决策函数，按 $`\mathrm{sign}(g(x))`$ 分类 | base estimator 的 score function |
| $`\ell(m)`$ | margin loss，$`m=yg(x)`$ | loss callable |
| $`R_P,R_N`$ | P/N 条件风险 | P/N sample mean |
| $`R_{U,P},R_{U,N}`$ | U 上取 $`\ell(g)`$ / $`\ell(-g)`$ 的风险 | U sample mean |
| $`R_{PN},R_{PU},R_{NU}`$ | PN、PU、NU 总风险 | risk components |
| $`\widetilde\ell(m)`$ | composite loss：$`\ell(m)-\ell(-m)`$ | signed composite loss |
| $`\theta_P,\theta_N`$ | 正/负类先验，和为 1 | `class_prior`, `1-class_prior` |
| $`\eta\in[-1,1]`$ | PNU 取舍参数 | `eta` |
| $`\gamma\in[0,1]`$ | PNPU / PNNU 中的权重 | `abs(eta)` |

---

## 4. 核心公式

### 4.1 基础 PN、PU、NU 风险

```math
R_{PN}(g)=\theta_PR_P(g)+\theta_NR_N(g).
```

定义复合损失 $`\widetilde\ell(m)=\ell(m)-\ell(-m)`$。

```math
R_{PU}(g)=\theta_P\mathbb E_P[\widetilde\ell(g(x))]+\mathbb E_U[\ell(-g(x))].
```

```math
R_{NU}(g)=\theta_N\mathbb E_N[\widetilde\ell(-g(x))]+\mathbb E_U[\ell(g(x))].
```

它们与 $`R_{PN}`$ 有相同的总体风险；实现时将每个期望替换为对应样本均值。注意 composite loss 落在 P/N 条件项上，不是在 U 项上。

### 4.2 PNU 风险

```math
R_{PNPU}^{\gamma}(g)=(1-\gamma)R_{PN}(g)+\gamma R_{PU}(g),
```

```math
R_{PNNU}^{\gamma}(g)=(1-\gamma)R_{PN}(g)+\gamma R_{NU}(g).
```

```math
R_{PNU}^{\eta}(g)=
\begin{cases}
R_{PNPU}^{\eta}(g), & \eta\ge0,\\
R_{PNNU}^{-\eta}(g), & \eta<0.
\end{cases}
```

端点是关键回归测试：$`\eta=-1,0,+1`$ 分别退化为 NU、PN、PU。

### 4.3 首选：凸实现（平方损失）

选择满足 $`\ell(m)-\ell(-m)=-m`$ 的凸 surrogate；论文实验训练使用平方损失 $`\ell_S(m)=(1-m)^2/4`$。此时：

```math
R_{C\text{-}PU}(g)=\theta_PR_P^L(g)+R_{U,N}(g),\qquad
R_P^L(g)=\mathbb E_P[-g(x)],
```

```math
R_{C\text{-}NU}(g)=\theta_NR_N^L(g)+R_{U,P}(g),\qquad
R_N^L(g)=\mathbb E_N[g(x)].
```

将上式代入 §4.2，即可得到可微、凸（在线性模型 + $`\ell_2`$ 正则下）的 PNU 目标。训练目标：

```math
\min_w\ \widehat R_{PNU}^{\eta}(g_w)+\lambda\lVert w\rVert_2^2.
```

### 4.4 非凸实现（可选）

若损失满足 $`\ell(m)+\ell(-m)=1`$，可用 ramp loss：

```math
\ell_R(m)=\tfrac12\max(0,\min(2,1-m)).
```

对应 PU/NU 风险可改写为无偏非凸目标，需 CCCP 求局部解。v1 不实现：优化复杂、可重复性和维护成本高，且平方损失已覆盖论文主实验路径。**[项目适配]**

### 4.5 $`\eta`$ 的理论启发

设 $`\psi_P=\theta_P^2\sigma_P^2(g)/n_P`$、$`\psi_N=\theta_N^2\sigma_N^2(g)/n_N`$，在 $`n_U\to\infty`$、固定 $`g`$ 下：

```math
\gamma_{N\text{-}PNPU}^*=\frac{\psi_N-\psi_P}{\psi_P+\psi_N},\qquad
\gamma_{N\text{-}PNNU}^*=\frac{\psi_P-\psi_N}{\psi_P+\psi_N}.
```

仅在相应 $`\gamma^*\in[0,1]`$ 的分支使用；论文的大型实验取 $`\sigma_P(g)=\sigma_N(g)`$ 作为近似，再以五折 CV 的 PNU 零一风险选择超参数。对实现而言，优先直接在验证折上搜索 `eta`，上式只可作为候选网格中心或初始化。

---

## 5. 算法概要

1. 校验 $`X_P,X_N,X_U`$ 均非空，$`0<\theta_P<1`$。
2. 选择 `eta_grid`（至少含 `[-1, 0, 1]`）和模型正则化参数；对每组参数做分层交叉验证。
3. 对每个 $`\eta`$，构造 $`\widehat R_{PNPU}^{\eta}`$ 或 $`\widehat R_{PNNU}^{-\eta}`$，并最小化加正则项的经验风险。
4. 用独立验证集/折的同一 PNU 风险选最优参数；重训最终模型。
5. 输出 classifier；同时保存 `class_prior_`、`eta_`、各风险分量，便于诊断。

论文的 Gaussian-kernel 实验设定可作为 benchmark，不应写死到通用实现：中心为 $`X_P\cup X_N`$，带宽候选为 `{1/8, 1/4, 1/2, 1, 3/2, 2} × median_pairwise_distance`。

---

## 6. 源码状态

| 字段 | 内容 |
|---|---|
| Source status | `official_exact` |
| Official code | `pywsl` (`https://github.com/t-sakai-kure/pywsl`) |
| License | `MIT` |
| Registry status | `implementation_status=API_ONLY`, `backend=TORCH`, `source_status=OFFICIAL_EXACT` |
| Integration basis | adapter (`pywsl`) + native PyTorch；论文 §2–§5 的风险定义、优化目标与实验 protocol 仍是数学权威 |

### 6.1 集成边界

- `pywsl` 可用于复现实验流程、超参数搜索、风险组合细节和 sanity check。
- native PyTorch 部分应服务项目统一 estimator API 与后续 GPU 支持。
- adapter 不应把上游脚本式数据协议泄漏到 `BasePUClassifier` 公共 API。
- 许可证虽为 MIT，集成时仍需在依赖/复制源码前复核上游仓库 LICENSE 文件。
- 外部资料还列出 MATLAB 版 PNU 分类源码；项目当前资源清单和 registry 已选择 `pywsl` 作为集成依据，本方法卡按项目权威源处理。

### 6.2 Registry 元数据

```python
name = "pnu"
aliases = ["pnu_classifier", "pn-pu-nu"]
family = AlgorithmFamily.RISK_ESTIMATION
scenario = (Scenario.CASE_CONTROL,)
assumption = (Assumption.SCAR,)
requires_class_prior = True
backend = Backend.TORCH
implementation_status = ImplementationStatus.API_ONLY
source_status = SourceStatus.OFFICIAL_EXACT
upstream_url = "https://github.com/t-sakai-kure/pywsl"
license = "MIT"
```

---

## 7. API 接口与项目落点

以下为项目适配建议，不是论文原始 API。当前 registry 将 PNU 注册为 `API_ONLY`，表示元数据存在但训练逻辑尚未落地；本卡的目标是让后续实现任务无需重新判断源码状态、数学公式和测试边界。

### 7.1 公共 API 与 P/N/U 数据协议

| API / 决策点 | 约定 |
|---|---|
| `fit(X, y_pu, *, class_prior=None, sample_weight=None)` | 当前 `BasePUClassifier` 公共契约；所有 native classifier 仍需遵守该 sklearn 风格入口。 |
| PNU 训练输入 | 必须能区分 P、N、U 三组样本；当前 `normalize_pu_labels()` 只把 `+1/-1/0` 归一到 P/U，不足以直接表达 N。 |
| 协议前置任务 | 实现 `PNUClassifier` 前必须先定稿 P/N/U 数据编码或附加 keyword（如 `sample_role` / `negative_mask` / `X_negative`），并写入 contract tests。 |
| adapter 输入 | 若临时 adapter 只能接受 `pywsl` 风格输入，应封装在 adapter 内部，不作为项目长期公共 API。 |
| `predict(X)` / `decision_function(X)` | 标准二分类输出；`decision_function` 返回越大越偏正的 score。 |
| `empirical_risk(...)` | 可作为开发侧诊断函数返回 PN、PU、NU、PNU 分量；公共签名需等 P/N/U 协议定稿后再冻结。 |
| `get_params()` / `set_params()` | 由 sklearn `BaseEstimator` 提供，至少覆盖构造函数中的公开超参数。 |

建议初版只支持：线性可微模型 + 平方损失；不在 `fit` 内隐式估计类先验，也不在无验证数据时自动选择 `eta`。

### 7.2 构造参数建议

| 参数 | 含义 |
|---|---|
| `class_prior` | 正类先验；可在构造函数或 `fit(..., class_prior=...)` 显式传入，但不得静默默认 `0.5`。 |
| `eta` | 固定 PNU 取舍参数，默认可取 `0.0` 表示 PN 端点。 |
| `eta_grid` | 交叉验证候选；若启用自动选择，必须至少包含 `[-1, 0, 1]`。 |
| `loss` | v1 推荐仅支持 `"squared"`，对应论文主实验路径；ramp loss + CCCP 不进入初版。 |
| `reg_lambda` | $`\ell_2`$ 正则强度，建议默认 `1e-3`，并通过验证集/交叉验证选择。 |
| `max_iter` / `tol` | 优化器迭代上限与收敛阈值。 |
| `random_state` | 控制交叉验证划分、初始化和可复现实验。 |
| `backend` | 建议默认 `"torch"`，与 registry 的 `Backend.TORCH` 对齐。 |

### 7.3 拟合属性建议

| 属性 | 含义 |
|---|---|
| `class_prior_` | 拟合时实际使用的 $`\theta_P`$。 |
| `eta_` | 拟合时实际使用或交叉验证选出的 $`\eta`$。 |
| `risk_components_` | 最近一次训练/验证的 PN、PU、NU、PNU 风险分量。 |
| `n_features_in_` | 输入特征维度，遵循 sklearn 约定。 |
| `_is_fitted` | 项目基类已有拟合状态标记；必要时也可暴露 sklearn 风格 fitted attributes。 |
| 模型参数属性 | 线性模型至少保存权重和截距；命名需与项目现有 estimator 风格一致。 |

### 7.4 模块落点

| 模块 | 责任 |
|---|---|
| `pu_toolbox/estimators/risk/pnu.py` | `PNUClassifier` 公共 estimator 与训练流程。 |
| `pu_toolbox/losses/pnu.py` | PN/PU/NU/PNU 风险组合函数，复用 uPU/nnPU 已有 loss 和风险分解风格。 |
| `pu_toolbox/source_adapters/pywsl.py` | 封装 `pywsl` 复现入口；如另有统一 adapter 命名，应遵守 `source_adapters` 现有模式。 |
| `pu_toolbox/registry/builtin_methods.py` | 当前已有 PNU 元数据；仅在 native 实现完成后再更新 `implementation_status`。 |

---

## 8. 测试参考

### 8.1 MATH tests

- 用手工 logits/小数组验证 PN、PU、NU、PNU 风险组合。
- `eta=0` 时 PNU objective 等于 PN objective。
- `eta=1` 时 PNU objective 等于 PU branch。
- `eta=-1` 时 PNU objective 等于 NU branch。
- 对平方损失路径，验证 $`\widetilde\ell(m)=\ell(m)-\ell(-m)=-m`$，且 composite loss 用在 P/N 条件项上。

### 8.2 PROPERTY tests

- P/N/U 三组样本内部顺序打乱不改变经验风险。
- 若支持 `sample_weight`，经验均值必须在 P、N、U 组内分别归一化，不得除以拼接后的总权重。
- `0 < class_prior < 1`，且 $`\theta_N=1-\theta_P`$。
- P/N/U 任一组为空应抛出明确异常。
- `eta` 必须落在 `[-1, 1]`；分支选择在 `eta=0` 附近行为明确且可测试。

### 8.3 CONTRACT tests

- 与 `tests/contract/test_classifier_api.py` 对齐：`fit` 返回 `self`，`predict(X)` 输出 `{0, 1}`，`decision_function(X)` 返回一维 score，`get_params()` / `set_params()` 可用。
- PNU 专用 P/N/U 数据协议定稿后，必须把协议加入 contract tests；在定稿前不要把 `fit(X, y, unlabeled_X)` 宣称为最终公共 API。
- 拟合后应暴露 `class_prior_`、`eta_`、`risk_components_` 和特征维度属性。

### 8.4 ADAPTER smoke tests

- 若接入 `pywsl`，用最小 toy dataset 验证 adapter 能加载并运行，或在依赖不可用时明确 skip。
- smoke test 不应依赖网络下载；上游源码或依赖应通过项目依赖/fixtures 明确提供。
- adapter 测试只验证桥接边界，不把上游脚本参数固化为项目公共 API。

### 8.5 PAPER-like regression

- 使用两类高斯分布，分别控制 $`n_P,n_N,n_U`$ 与 $`\theta_P`$，验证训练可运行、风险分量有限、端点行为正确。
- 验证交叉验证能从 `eta_grid` 选择端点或内部值；不强制“PNU 总优于 PN”，因为改进依赖样本量、先验和方差。
- 可复现论文式核模型实验：特征缩放、五折 CV、平方损失；将结果视为回归基准而非精确复现实验表。

### 8.6 非目标

- 初版不强制 PNU 在所有合成数据上优于 PN/uPU/nnPU。
- 初版不实现或测试 ramp loss + CCCP。
- 初版不把 `pywsl` 的训练脚本参数视作项目公共 API。
