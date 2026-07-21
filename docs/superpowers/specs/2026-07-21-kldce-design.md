# KLDCE Design Spec

> 2026-07-21 | 基于 `docs/research/method_cards/KLDCE.md` + 线上补充附录
> 修订 2: 纳入 problems.md 论文对照审查 — 盒约束、等式常数 C、Taylor 展开点、RBF 参数、ridge 语义、MoM 符号 等 7 项修正
> 修订 3: problems.md 第三轮 — d(μ) 符号、bounded_interval 公式、pinv 回退、输入校验

## 1. 背景与前置条件

- **LDCE**（线性版）已完整实现，2026-07-21 修复了 MoM 符号（`-X_U`）。
- **KLDCE**（核化版）的补充附录已取得：给出了 Algorithm 1、α/γ SMO 更新式 (21)–(26)、RBF Taylor 质心更新式 (33)–(35)、偏置四项平均式 (37)–(40)。
- **首版交付策略**：ACS 外循环 + `scipy` QP oracle 解固定 μ 的对偶 QP（数学上与 SMO 等价）；附录原生 SMO 留待后续 PR 替换。
- **首版仅支持 RBF kernel**：附录的 μ 更新明确依赖 Gaussian Taylor 展开。

## 2. 模块拆分

### 2.1 共享原语提取

`_mom_centroid` 和 `_centroid_covariance` 从 `ldce.py` 移至新文件。

**协方差 ridge 语义**（v3 修正）：

```python
S_hat_raw = _centroid_covariance(X_U)           # 论文椭球约束用（无 ridge）
# 默认 ridge=0.0 — 论文原式。对 S_raw 使用稳定的线性求解/伪逆。
# ridge > 0 为显式 opt-in 数值变体，运行时标注在结果中。
if covariance_ridge > 0:
    S_solve = S_hat_raw + covariance_ridge * I
else:
    S_solve = S_hat_raw
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
        covariance_ridge: float = 0.0,     # 默认论文原式; >0 时标记为数值变体
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
  2. 校验 flip_probability ∈ (0,1)
     校验 k > 0 (至少一个正例), n_U > 0 (至少一个无标签样本)
     校验 mom_groups ≥ 1 且 mom_groups ≤ n_U
     校验 n_dual = n + n_U ≤ max_dual_variables
  3. p = k/[n(1-h)], **显式检查 0 < p ≤ 1**（p 是类别先验，违反时拒绝）
     检查 |1-2ph| 近零 (≤ 1e-12)
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
       **注**: 论文 Algorithm 1 先更新 μ 再 SMO α/γ；QP oracle 版先固定 μ 解联合 QP 再更新 μ。
       两者均为 ACS 的合理块坐标顺序变化，但 QP oracle 版不是论文 Algorithm 1 的逐行实现。
    b. 记录 fixed_mu_dual_objective, eq_residual, box_violation
    c. 固定 z: 按附录式(33) 从 α,γ 计算 Δ
       (Taylor 展开点 μ=0, 非 μ=m̂ — 论文原式)
    d. 按附录式(35) 更新 μ:
       解 S_solve · u = Δᵀ         (S_solve = S_raw + ridge·I)
       计算 q = uᵀ · S_raw · u      (缩放基准始终用 S_raw)
       若 q ≤ ε: μ=m̂, 标记 degenerate_centroid_step
       否则: μ = m̂ - u · √(b/q)     (保证 (μ-m̂)ᵀS_raw(μ-m̂) = b)
    e. 验证: (μ-m̂)ᵀ Ŝ_raw (μ-m̂) ≤ b + tol
       若违反且 ridge>0: 标记 constraint_violated, 触发椭球投影回退
       记录 centroid_constraint_residual
    f. _recover_bias_from_kkt(α, γ, K, ỹ, μ, λ, C_eq, C_alpha, C_gamma) → b₀
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

以附录式 (24) 为权威。令 `N = n + n_U`（总对偶变量数）。

**Q 的逐块结构**（`N × N`，对称）：

```text
z = [α₁…αₙ | γ_{k+1}…γₙ]
Q = 1/(2λ) · [  Q_αα    Q_αγ  ]
              [  Q_γα    Q_γγ  ]

Q_αα[i][j] = ỹᵢ ỹⱼ K(xᵢ, xⱼ)           (i,j = 1…n)
Q_αγ[i][j] = -ỹᵢ ỹ_{k+j} K(xᵢ, x_{k+j})  (i=1…n, j=1…n_U)
Q_γα = Q_αγᵀ
Q_γγ[i][j] = ỹ_{k+i} ỹ_{k+j} K(x_{k+i}, x_{k+j})  (i,j = 1…n_U)
```

**线性项 d(μ)**（`N` 维）：

```text
d(μ)_i = 1 + C_eq·ỹᵢ·K(xᵢ, μ)/(2λ)               (i=1…n, 对应 α)
d(μ)_{n+i} = 1 - C_eq·ỹ_{k+i}·K(x_{k+i}, μ)/(2λ)   (i=1…n_U, 对应 γ)
```

**约束**：

```text
Aeq = [ỹ₁…ỹₙ | -ỹ_{k+1}…-ỹₙ]     (1 × N)
beq = C_eq = -(n-k) / (2·n·(1-2·p·h))
lb  = [0] × N
ub  = [1/n]×n + [1/(2n)]×n_U
```

验收：MATH 测试用 4 样本手工计算每个块，与独立实现的期望值比较（atol=1e-14）。
另增单样本交叉项测试：从 §5 的 RKHS 系数 `r = αỹ - γỹ_U - C_eq·φ(μ)` 展开
`λ‖r‖²`，逐项核对 Q_αα/Q_αγ/Q_γγ 的符号和 d(μ) 的符号——确保 QP 公式
不是用同一错误公式互相验证。

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

### 6.5 `_update_centroid(m_hat, S_raw, S_solve, delta, centroid_radius, tol)`

- 解 `u = S_solve⁻¹ · Δᵀ`（用 `scipy.linalg.solve`）
- **ridge=0 严格模式**：先检查 `S_raw` 条件数；若 `cond(S_raw) > 1e12` 或 Cholesky 失败，
  报 `LinAlgError("S_raw is near-singular; use covariance_ridge > 0 for numerical stabilization")`
- **ridge>0 变体模式**：`S_solve = S_raw + ridge·I` 保证可逆，运行后标记 `centroid_solver="ridge_stabilized"`
- **约束缩放基准始终为 `S_raw`**：`q = u @ S_raw @ u`
- 若 `q ≤ tol`: `μ = m_hat`，标记 `degenerate_centroid_step`
- 否则: `μ = m_hat - u · √(centroid_radius / q)`（**保证 `(μ-m̂)ᵀS_raw(μ-m̂) = b`**）
- 验证 `(μ-m̂)ᵀ S_raw (μ-m̂) ≤ centroid_radius + tol`
  若违反且 ridge > 0: 标记 `constraint_violated`，触发显式椭球投影 `μ ← m̂ + (μ-m̂)·√(b/constraint)`

### 6.6 `_recover_bias_from_kkt(alpha, gamma, K, y_tilde, mu, lambda, C_eq, C_alpha, C_gamma)` → b₀

QP oracle 版：收集 `0 < αᵢ < C_alpha` **或** `0 < γᵢ < C_gamma` 的自由支持向量。

决策函数 `f(xᵢ)` 由 §5 公式计算（含 `-C_eq·K(x,μ)/(2λ)` 项）。对每个自由变量由 KKT margin 条件反推：

```text
自由 αᵢ (ỹᵢ=+1): bᵢ = 1 - gᵢ
自由 γᵢ (ỹᵢ=-1): bᵢ = 1 - gᵢ  (ỹ=-1, KKT: -(g+b)≥1, 自由时取等号)
其中 gᵢ = f(xᵢ) - b₀ (不含 bias 的决策分数)
b₀ = median({bᵢ})
```

无自由变量时（α 和 γ 全部在边界），令 `gᵢ = f(xᵢ) - b₀`。
由 KKT margin 条件：

```text
L = max({1 - gᵢ  | αᵢ=0, ỹᵢ=+1} ∪
        {-1 - gᵢ | αᵢ=C_alpha, ỹᵢ=-1} ∪
        {1 - gᵢ  | γᵢ=0, ỹᵢ=-1})
U = min({1 - gᵢ  | αᵢ=C_alpha, ỹᵢ=+1} ∪
        {-1 - gᵢ | αᵢ=0, ỹᵢ=-1} ∪
        {1 - gᵢ  | γᵢ=C_gamma, ỹᵢ=-1})
```

若 `L ≤ U`: b₀ = (L+U)/2，标记 `bias_recovery="bounded_interval"`
否则标记 `bias_recovery="indeterminate"`，b₀ = 0

验收：为 α=0/C、γ=0/C 各组合写单测（覆盖全部六种边界情况）。

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

## 10. v2 修正摘要（problems.md 第一轮对照）

| # | 优先级 | 问题 | 修正 |
|---|--------|------|------|
| 1 | 严重 | 盒约束 `C_alpha=1.0` | 改为 `C_alpha=1/n`, `C_gamma=1/(2n)`，不可调 |
| 2 | 严重 | 等式常数 C 来源错误 | 显式公式 `C_eq = -(n-k)/(2*n*(1-2*p*h))` |
| 3 | 严重 | Taylor 展开点 μ=m̂ | 改为 μ=0（论文原式） |
| 4 | 严重 | RBF 参数传递 | 直接实现 `exp(-sqdist/(2σ²))`，参数改 `sigma` |
| 5 | 高 | Ridge 改变质心更新 | 标注为数值稳定化变体；约束检查用 `Ŝ_raw` |
| 6 | 高 | MoM 输入符号 | `_mom_centroid(-X_U)` — 同时修复 LDCE |
| 7 | 中 | QP oracle ≠ 论文 SMO | 标注为等价求解器替代；SMO 后续 PR |

## 11. v3 修正摘要（problems.md 第三轮）

| # | 优先级 | 问题 | 修正 |
|---|--------|------|------|
| 1 | 严重 | d(μ) 符号与 §5 决策函数不一致 | α: `1+C_eq·ỹK/(2λ)`, γ: `1-C_eq·ỹK/(2λ)`；增单样本交叉项测试 |
| 2 | 高 | bounded_interval L/U 公式缺 margin 常数 1 | 用 `gᵢ=f(xᵢ)-b₀` 重写；四种 KKT 情况各写单测 |
| 3 | 高 | ridge=0 时 pinv 回退不等价 | 严格模式报 `LinAlgError` 提示加 ridge；>0 标记变体状态 |
| 4 | 中 | 输入校验缺 k>0/n_U>0/p≤1/mom_groups≤n_U | 在步骤 2 添加显式校验，说明违反的统计前提 |

## 12. 不纳入本 PR

- 附录原生 SMO（`_smo_*` 占位，后续 PR 替换 QP oracle）
- 非 RBF 核（附录 μ 更新是 RBF Taylor 专用）
- `max_dual_variables` 以上规模的近似核（Nyström/RFF）
- `predict_proba`（继承 `NotImplementedError`）
