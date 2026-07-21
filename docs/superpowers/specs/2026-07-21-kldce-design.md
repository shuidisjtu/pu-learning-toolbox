# KLDCE Design Spec

> 2026-07-21 | 基于 `docs/research/method_cards/KLDCE.md` + 线上补充附录
> 修订 2: 纳入 problems.md 论文对照审查 — 盒约束、等式常数 C、Taylor 展开点、RBF 参数、ridge 语义、MoM 符号 等 7 项修正

## 1. 背景与前置条件

- **LDCE**（线性版）已完整实现，2026-07-21 修复了 MoM 符号（`-X_U`）。
- **KLDCE**（核化版）的补充附录已取得：给出了 Algorithm 1、α/γ SMO 更新式 (21)–(26)、RBF Taylor 质心更新式 (33)–(35)、偏置四项平均式 (37)–(40)。
- **首版交付策略**：ACS 外循环 + `scipy` QP oracle 解固定 μ 的对偶 QP（数学上与 SMO 等价）；附录原生 SMO 留待后续 PR 替换。
- **首版仅支持 RBF kernel**：附录的 μ 更新明确依赖 Gaussian Taylor 展开。

## 2. 模块拆分

### 2.1 共享原语提取

`_mom_centroid` 和 `_centroid_covariance` 从 `ldce.py` 移至新文件。

**协方差 ridge 语义**（v2 修正）：

```python
S_hat_raw = _centroid_covariance(X_U)           # 论文椭球约束用（无 ridge）
S_solve   = S_hat_raw + covariance_ridge * I     # 线性求解用
# 约束检查始终使用 S_hat_raw；Eq.35 求解使用 S_solve
# 在文档中注明 ridge 为数值稳定化变体，非论文原式
```

**MoM 符号**（v2 修正 — 同时适用于 LDCE 和 KLDCE）：

```python
# 论文 MoM 对象是 {ỹ_i · x_i | ỹ_i = -1} = {-x_i}
m_hat = _mom_centroid(-X_U, g, rng)
# 协方差因符号相消可直接用 X_U 计算
S_hat_raw = _centroid_covariance(X_U)
```

**`pu_toolbox/utils/centroid.py`**（NEW）

```
_mom_centroid(X, g, rng)               → centroid   (Algorithm 1; 调用者负责传入 -X_U)
_centroid_covariance(X)                → Ŝ_raw      (Eq. 10, 无 ridge)
```

`ldce.py` 改为 `from ...utils.centroid import _mom_centroid, _centroid_covariance`，删掉本地定义。

### 2.2 KLDCE 新文件

**`pu_toolbox/estimators/risk/kldce.py`**（NEW）

```
KLDCEClassifier(BasePUClassifier)           # 主类
_find_feasible_init(Aeq, beq, lb, ub)       # Phase-I LP 可行初值
_build_dual_qp(mu, K, y_tilde, lambda, n, k) # 附录式 (24) → Q, d, Aeq, beq, lb, ub
_solve_qp_oracle(Q, d, Aeq, beq, lb, ub, z0) # scipy.optimize 包装
_rbf_centroid_delta(alpha, gamma, X, ...)    # 附录式 (33) (μ=0 Taylor 展开)
_update_centroid(m_hat, S_solve, delta, b)   # 附录式 (35) (RBF only)
_recover_bias_from_kkt(alpha, gamma, ...)    # QP oracle 版: KKT 自由支持向量
# 占位 — 后续 PR
_smo_alpha_pair(...)
_smo_gamma_pair(...)
_recover_bias_smo_incremental(...)           # 附录式 (37)–(40)
_smo_solve(...)
```

## 3. API 设计

```python
class KLDCEClassifier(BasePUClassifier):
    family = AlgorithmFamily.RISK_ESTIMATION
    assumption = (Assumption.SCAR,)
    scenario = (Scenario.SINGLE_TRAINING_SET,)
    requires_class_prior = False
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.OFFICIAL_RELATED
    backend = Backend.NUMPY
    maturity = Maturity.RESEARCH

    def __init__(
        self,
        flip_probability: float,            # 必填
        *,
        sigma: float | str = "scale",       # 论文带宽 σ; gamma = 1/(2σ²)
        reg_strength: float = 1.0,          # λ
        centroid_radius: float = 1.0,       # b (椭球半径)
        mom_groups: int = 10,
        covariance_ridge: float = 1e-4,     # 仅求解用，不改变约束；标注为数值变体
        max_acs_iter: int = 50,
        max_dual_variables: int = 1000,     # 规模保护: max |z| = n + n_U
        tol: float = 1e-6,
        random_state: int | None = None,
    ):
```

**与 v1 的差异**：
- 移除 `C_alpha`、`C_gamma`（改为训练时按 n 计算）
- 移除 `max_smo_iter`（QP oracle 不需要）
- `gamma` → `sigma`（论文原生参数，避免 sklearn `gamma` 混淆）
- 移除 `max_train_samples`，只保留 `max_dual_variables`

**σ 的默认值**：

```python
if sigma == "scale":
    sigma = 1.0 / np.sqrt(X.shape[1])  # 1/sqrt(d) heuristic
# 始终: gamma = 1/(2 * sigma**2)
```

## 4. fit() 流程

```text
fit(X, y_pu, *, class_prior=None):
  1. validate_pu_X_y → {+1, 0}, split P (k samples) / U (n_U = n-k samples)
  2. 校验 flip_probability ∈ (0,1), n_dual = n + n_U ≤ max_dual_variables
  3. p = k/[n(1-h)], 检查 |1-2ph| 近零 (≤ 1e-12)
  4. ỹ = +1 (P), -1 (U)
  5. MoM: m̂ = _mom_centroid(-X_U, g, rng)   (论文: {ỹ_i · x_i | ỹ=-1} = {-x_i})
     协方差: Ŝ_raw = _centroid_covariance(X_U)  (符号相消)
     S_solve = Ŝ_raw + covariance_ridge · I
  6. sigma = "scale" → 1/sqrt(d); gamma = 1/(2*sigma²)
     K = kernel_matrix(X, X, sigma)    # 直接实现 K=exp(-||x-z||²/(2σ²))
  7. 计算论文常数:
     C_alpha = 1/n           # α 盒约束上界 (不可调)
     C_gamma = 1/(2*n)       # γ 盒约束上界 (不可调)
     c = -(n-k)/(2*n)
     C_eq = c/(1-2*p*h)      # 等式约束右端 (显式公式, 非从变量反推)
     lb = [0]*(n+n_U), ub = [C_alpha]*n + [C_gamma]*n_U
  8. 构造初始 Aeq (不含 μ), beq=C_eq
     调用 _find_feasible_init(Aeq, beq, lb, ub) → z₀
     验证 Aeq@z₀ == beq 且 lb ≤ z₀ ≤ ub
     初始化 μ = m̂

  ACS 外循环 (t=1..max_acs_iter):
    a. 固定 μ: _build_dual_qp(μ, K, ỹ, λ, n, k) → Q, d(μ), Aeq, beq, lb, ub
       _solve_qp_oracle(Q, d, Aeq, beq, lb, ub, z₀) → z=[α;γ], diagnostics
    b. 记录 fixed_mu_dual_objective, eq_residual, box_violation
    c. 固定 z: 按附录式(33) 从 α,γ 计算 Δ
       (Taylor 展开点 μ=0, 非 μ=m̂ — 论文原式)
    d. 按附录式(35) 更新 μ:
       解 S_solve · u = Δᵀ, d = Δ@u
       若 d ≤ ε: μ=m̂, 标记 degenerate_centroid_step
       否则: μ = m̂ - u · √(b/d)
    e. 验证: (μ-m̂)ᵀ Ŝ_raw (μ-m̂) ≤ b + tol
       记录 centroid_constraint_residual
    f. _recover_bias_from_kkt(α, γ, K, ỹ, μ, λ, C_alpha) → b₀
    g. 收敛判断: max(相对目标变化, ||Δμ||, max_kkt_violation) < tol → break
    h. z₀ ← z (warm start)

  9. 存储 fitted attributes
  10. _is_fitted = True
```

## 5. 决策函数（附录式 25）

```text
f(x) = [ Σᵢ αᵢ ỹᵢ K(x, xᵢ) - Σ_{i=k+1}ⁿ γᵢ ỹᵢ K(x, xᵢ) - C_eq · K(x, μ) ] / (2λ) + b₀
```

三点均与 v1 保持一致（已核实与论文一致）：
- γ 项为减号
- 质心项 `- C_eq · K(x, μ) / (2λ)` 不可省略
- 全局缩放 `1/(2λ)`

## 6. 关键算法子函数

### 6.1 `_find_feasible_init(Aeq, beq, lb, ub)` → z₀

当 C_eq ≠ 0 时 α=γ=0 违反约束。求解 Phase-I LP：

```text
min 0  s.t.  Aeq @ z = beq,  lb ≤ z ≤ ub
```

用 `scipy.optimize.linprog`。验收: `abs(Aeq@z₀ - beq) ≤ 1e-10` 且盒约束满足。

### 6.2 `_build_dual_qp(mu, K, y_tilde, lambda, n, k)` → (Q, d, Aeq, beq, lb, ub)

以附录式 (24) 为权威。

- **Q**: `(n+n_U) × (n+n_U)`，仅来自 Gram 矩阵 + 标签 + λ。**固定 μ 时 Q 不含 μ**。
- **d(μ)**: 线性项，μ 进入此处——含 `K(xᵢ, μ)` 和标签的组合。
- **Aeq**: `1 × (n+n_U)`，元素为 `[ỹ₁,...,ỹₙ, -ỹ_{k+1},...,-ỹₙ]`。
- **beq**: `C_eq = -(n-k)/(2*n*(1-2*p*h))`（显式计算）。
- **lb**: `[0] * (n+n_U)`
- **ub**: `[1/n]*n + [1/(2n)]*n_U`

### 6.3 `_solve_qp_oracle(Q, d, Aeq, beq, lb, ub, z0)` → (z, diagnostics)

求解 `max dᵀz - ½ zᵀQz  s.t. Aeq@z=beq, lb≤z≤ub`。

用 `scipy.optimize.minimize(method="SLSQP")`。返回 z=[α;γ] 及 diagnostics（目标、等式残差、盒约束违反度、KKT 残差、状态）。

### 6.4 `_rbf_centroid_delta(alpha, gamma, X, y_tilde, lambda, sigma)`

附录式 (33)：**在 μ=0 处 Taylor 展开**（不围绕 m̂）。

```text
K(x, μ) = exp(-||x-μ||²/(2σ²))
∂K/∂μ|_{μ=0} = K(x,0) · x/σ² = exp(-||x||²/(2σ²)) · x/σ²

Δ = -1/(2λσ²) · Σᵢ αᵢ ỹᵢ exp(-||xᵢ||²/(2σ²)) xᵢ
    + 1/(2λσ²) · Σ_{i=k+1}ⁿ γᵢ ỹᵢ exp(-||xᵢ||²/(2σ²)) xᵢ
```

### 6.5 `_update_centroid(m_hat, S_solve, delta, centroid_radius, tol)`

- 解 `u = S_solve⁻¹ · Δᵀ`（用 `scipy.linalg.solve`，设 `assume_a="sym"`）
- `d = Δ @ u`
- 若 `d ≤ tol`: `μ = m_hat`，标记 `degenerate_centroid_step`
- 否则: `μ = m_hat - u · √(centroid_radius / d)`
- 验证 `(μ-m̂)ᵀ Ŝ_raw (μ-m̂) ≤ centroid_radius + tol`

### 6.6 `_recover_bias_from_kkt(alpha, gamma, K, y_tilde, mu, lambda, C_alpha)`

QP oracle 版：收集 `0 < αᵢ < C_alpha` 的自由支持向量，由 KKT margin 条件逐个反推 bᵢ，取中位数。无自由变量时从上下界构成的可行区间取中点，标记 `bias_recovery="bounded_interval"`。

附录式 (37)–(40) 的四项平均增量更新留给 `_recover_bias_smo_incremental`（SMO 版 PR）。

### 6.7 RBF 核函数实现

不直接使用 `sklearn.metrics.pairwise.rbf_kernel`（它的参数是 gamma）。

```python
def _rbf_kernel(X, Z, sigma):
    sqdist = scipy.spatial.distance.cdist(X, Z, 'sqeuclidean')
    return np.exp(-sqdist / (2 * sigma**2))
```

## 7. 拟合属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `alpha_full_` | `(n,)` | α 对偶乘子（含零值） |
| `gamma_unlabeled_` | `(n_U,)` | γ 对偶乘子（含零值） |
| `unlabeled_indices_` | `(n_U,)` | U 样本在 X 中的索引 |
| `support_indices_` | `(n_sv,)` | α≠0 或 γ≠0 的样本索引 |
| `bias_` | `float` | b₀ |
| `class_prior_` | `float` | p = k/[n(1−h)] |
| `flip_probability_` | `float` | h |
| `centroid_hat_` | `(d,)` | MoM 质心 m̂ |
| `centroid_opt_` | `(d,)` | 优化后质心 μ |
| `centroid_covariance_raw_` | `(d,d)` | Ŝ（论文约束用，无 ridge） |
| `C_eq_` | `float` | 等式约束右端常数 |
| `n_acs_iter_` | `int` | ACS 收敛轮数 |
| `acs_history_` | `list[dict]` | 每轮 {dual_obj, eq_residual, box_violation, centroid_constraint_residual} |
| `converged_` | `bool` | 收敛标志 |
| `classes_` | `(2,)` | [0, 1] |

## 8. 文件变更清单

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 1 | `pu_toolbox/utils/centroid.py` | **新建** | MoM + 协方差原语（无 ridge 参数） |
| 2 | `pu_toolbox/estimators/risk/ldce.py` | 修改 | import 替换 + MoM 传入 `-X_U` + 自行加 ridge |
| 3 | `pu_toolbox/estimators/risk/kldce.py` | **新建** | KLDCEClassifier (~600 行) |
| 4 | `pu_toolbox/estimators/risk/__init__.py` | 修改 | 导出 KLDCEClassifier |
| 5 | `pu_toolbox/registry/builtin_methods.py` | 修改 | 绑定 kldce native class |
| 6 | `tests/estimators/risk/test_ldce_math.py` | 修改 | import 路径适配 |
| 7 | `tests/estimators/risk/test_kldce_math.py` | **新建** | 公式级 MATH tests |
| 8 | `tests/estimators/risk/test_kldce_oracle.py` | **新建** | QP oracle 对照 |
| 9 | `tests/estimators/risk/test_kldce_property.py` | **新建** | 约束/h 失配/ACS 日志 |
| 10 | `tests/contract/test_classifier_api.py` | 修改 | 添加 KLDCE factory |
| 11 | `tests/test_builtin_methods.py` | 修改 | native 6→7, api_only 9→8 |
| 12 | `docs/research/method_cards/KLDCE.md` | 修改 | 更新 Registry 引用、σ 参数 |

## 9. 测试策略

| 类别 | 文件 | 验收标准 |
|------|------|----------|
| MATH | `test_kldce_math.py` | 4–8 样本手工验证 Q/d/Aeq/beq/lb/ub vs 附录式 (24)；`C_eq` 显式公式；可行初始化 `Aeq@z₀==beq` + 盒约束；决策函数逐项（`-γ`、`-C_eq·K(x,μ)/(2λ)`、`1/(2λ)`）；`_rbf_centroid_delta` vs 手算 Δ（μ=0 展开） |
| ORACLE | `test_kldce_oracle.py` | 固定 μ: QP oracle 等式残差 ≤ 1e-8，盒约束满足；与未来 SMO 同一 μ 下目标差 ≤ 1e-6、margin 差 ≤ 1e-5 |
| PROPERTY | `test_kldce_property.py` | 椭球约束消融；h 失配 `ĥ∈{0.6h,…,1.4h}` 不崩溃；随机种子可复现；Taylor 近似前后目标记录（不默认 ACS 单调） |
| API | `test_classifier_api.py` | sklearn fit/predict/clone/pipeline 全契约 |
| Registry | `test_builtin_methods.py` | native 计数 + metadata 断言 |

## 10. v2 修正摘要（problems.md 对照）

| # | 优先级 | 问题 | 修正 |
|---|--------|------|------|
| 1 | 严重 | 盒约束 `C_alpha=1.0` | 改为 `C_alpha=1/n`, `C_gamma=1/(2n)`，不可调 |
| 2 | 严重 | 等式常数 C 来源错误 | 显式公式 `C_eq = -(n-k)/(2*n*(1-2*p*h))` |
| 3 | 严重 | Taylor 展开点 μ=m̂ | 改为 μ=0（论文原式） |
| 4 | 严重 | RBF 参数传递 | 直接实现 `exp(-sqdist/(2σ²))`，参数改 `sigma` |
| 5 | 高 | Ridge 改变质心更新 | 标注为数值稳定化变体；约束检查用 `Ŝ_raw` |
| 6 | 高 | MoM 输入符号 | `_mom_centroid(-X_U)` — 同时修复 LDCE |
| 7 | 中 | QP oracle ≠ 论文 SMO | 标注为等价求解器替代；SMO 后续 PR |

## 11. 不纳入本 PR

- 附录原生 SMO（`_smo_*` 占位，后续 PR 替换 QP oracle）
- 非 RBF 核（附录 μ 更新是 RBF Taylor 专用）
- `max_dual_variables` 以上规模的近似核（Nyström/RFF）
- `predict_proba`（继承 `NotImplementedError`）
