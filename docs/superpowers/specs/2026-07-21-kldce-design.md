# KLDCE Design Spec

> 2026-07-21 | 基于 `docs/research/method_cards/KLDCE.md` + 线上补充附录
> 修订 1: 纳入队友审查 — 修正等式约束、决策函数符号、可行初始化等 12 项

## 1. 背景与前置条件

- **LDCE**（线性版）已完整实现（`pu_toolbox/estimators/risk/ldce.py`），MoM 质心 + 协方差估计 + 椭球约束交替优化。
- **KLDCE**（核化版）的补充附录已取得：给出了 Algorithm 1、α/γ SMO 更新式 (21)–(26)、RBF Taylor 质心更新式 (33)–(35)、偏置四项平均式 (37)–(40)。
- **首版交付策略**：ACS 外循环 + `scipy` QP oracle 解固定 m 的对偶 QP；附录 SMO 留待后续 PR 替换。
- **首版仅支持 RBF kernel**：附录的 μ 更新明确依赖 Gaussian Taylor 展开。`linear` 退化解属于研究扩展，不与论文 KLDCE 一同标为原生实现。

## 2. 模块拆分

### 2.1 共享原语提取

`_mom_centroid` 和 `_centroid_covariance` 从 `ldce.py` 移至新文件。

**`_centroid_covariance` 的 ridge 语义修正**：

```python
# 原 ldce.py 行为：ridge 写入 S_hat 内部 → 约束本身被改
# 修正后：
S_hat_raw = _centroid_covariance(X_U, ridge=0.0)      # 论文椭球约束用
S_solve   = S_hat_raw + covariance_ridge * np.eye(d)   # 线性求解用
```

**`pu_toolbox/utils/centroid.py`**（NEW）

```
_mom_centroid(X_U, g, rng)              → m̂    (Algorithm 1)
_centroid_covariance(X_U)               → Ŝ_raw (Eq. 10, 无 ridge)
```

`ldce.py` 改为 `from ...utils.centroid import _mom_centroid, _centroid_covariance`，删掉本地定义；内部自行加 ridge。

### 2.2 KLDCE 新文件

**`pu_toolbox/estimators/risk/kldce.py`**（NEW）

```
KLDCEClassifier(BasePUClassifier)           # 主类
_find_feasible_init(Aeq, beq, lb, ub)       # Phase-I 可行初值
_build_dual_qp(m, K, y_tilde, ...)          # 附录式 (24) → Q, d, Aeq, beq, lb, ub
_solve_qp_oracle(Q, d, Aeq, beq, lb, ub)    # scipy.optimize 包装
_rbf_centroid_delta(alpha, gamma, X, ...)   # 附录式 (33)
_update_centroid(m_hat, S_hat_raw, delta, ...) # 附录式 (35) (RBF only)
_recover_bias_from_kkt(alpha, gamma, ...)   # QP oracle 版：自由支持向量 KKT
# 以下为占位 — 后续 PR 启用
_smo_alpha_pair(...)                        # 附录式 (21)–(23)
_smo_gamma_pair(...)                        # 附录式 (24)–(26)
_recover_bias_smo_incremental(...)          # 附录式 (37)–(40) 四项平均
_smo_solve(...)                             # SMO 主循环
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
        gamma: float | str = "scale",       # RBF γ; σ = sqrt(1/(2γ))
        reg_strength: float = 1.0,          # λ
        centroid_radius: float = 1.0,       # b (椭球半径)
        C_alpha: float = 1.0,               # C₁ — α 盒约束上界
        C_gamma: float = 1.0,               # C₂ — γ 盒约束上界
        mom_groups: int = 10,
        covariance_ridge: float = 1e-4,     # 仅用于线性求解，不改变约束
        max_acs_iter: int = 50,
        max_dual_variables: int = 1000,     # 规模保护：max |z|
        tol: float = 1e-6,
        random_state: int | None = None,
    ):
```

**与 LDCE 的差异**：去掉 `learning_rate`、`n_inner_iter`（不需要 GD）；新增 `gamma`、`C_alpha`、`C_gamma`、`max_dual_variables`；无 `kernel`（首版仅 rbf）。

**`gamma` ↔ `σ` 映射**（必须固定）：

```python
if gamma == "scale":
    sigma = np.sqrt(1.0 / (2.0 * (1.0 / X.shape[1])))  # 1/(n_features) heuristic
else:
    sigma = np.sqrt(1.0 / (2.0 * gamma))
```

## 4. fit() 流程

```text
fit(X, y_pu, *, class_prior=None):
  1. validate_pu_X_y → {+1, 0}, split P (k samples) / U (n_U = n-k samples)
  2. 校验 flip_probability ∈ (0,1), n_dual = n + n_U ≤ max_dual_variables
  3. p = k/[n(1-h)], 检查 |1-2ph| 近零
  4. ỹ = +1 (P), -1 (U)
  5. MoM → m̂, 协方差 → Ŝ_raw  (utils/centroid.py)
     S_solve = Ŝ_raw + covariance_ridge · I   (仅求解)
  6. Gram 矩阵 K = rbf_kernel(X, X, sigma)
  7. 构造 lb = [0]*len(z), ub = [C₁]*n + [C₂]*n_U
     调用 _find_feasible_init(Aeq, beq, lb, ub) → z₀
     初始化 m = m̂

  ACS 外循环 (t=1..max_acs_iter):
    a. 固定 m: _build_dual_qp(m, K, ỹ, λ, C₁, C₂) → Q, d, Aeq, beq, lb, ub
       _solve_qp_oracle(Q, d, Aeq, beq, lb, ub, z₀) → z=[α;γ], diagnostics
    b. 记录 fixed_m_dual_objective, eq_residual, box_violation
    c. 固定 z: 按附录式(33) 从 α,γ 计算 Δ
    d. 按附录式(35) 更新 μ → m
       解 S_solve · u = Δᵀ, 计算 d = Δ @ u
       若 d ≤ ε_tol: 保持 m=m̂, 标记 degenerate_centroid_step
       否则: m = m̂ - u · √(b/d)
    e. 验证椭球约束: (m-m̂)ᵀ Ŝ_raw (m-m̂) ≤ b + tol
       记录 centroid_constraint_residual
    f. QP oracle 版：从自由 α 支持向量的 KKT 条件恢复 b
       _recover_bias_from_kkt(alpha, gamma, ...) → b
    g. 构造 warm start: z₀ ← z (下轮 QP 热启动)
    h. 收敛判断: 相对目标变化 + ||Δm|| + KKT 均 < tol → break

  8. 存储 fitted attributes
  9. _is_fitted = True
```

## 5. 决策函数（附录式 25）

**唯一的正确公式**：

```text
f(x) = [ Σᵢ αᵢ ỹᵢ K(x, xᵢ) - Σ_{i=k+1}^n γᵢ ỹᵢ K(x, xᵢ) - C · K(x, μ) ] / (2λ) + b
```

关键点：
- **γ 项为减号**：`- Σ_U γᵢ ỹᵢ K(x, xᵢ)`
- **质心项**：`- C · K(x, μ) / (2λ)`，不可省略
- **全局缩放**：`1/(2λ)` 作用于整个括号内

`C` 来自附录式 (8) 的等式约束右端：`C = Σαᵢỹᵢ - Σ_U γᵢỹᵢ`（固定值）。

## 6. 关键算法子函数

### 6.1 `_find_feasible_init(Aeq, beq, lb, ub)` → z₀

当 `C ≠ 0` 时 `α=γ=0` 违反 `Aeq @ z = beq`。求解小 LP：

```text
min 0   s.t.   Aeq @ z = beq,  lb ≤ z ≤ ub
```

用 `scipy.optimize.linprog` 找一个可行点。验收测试验证 `Aeq @ z₀ == beq` 且盒约束满足。

### 6.2 `_build_dual_qp(m, K, y_tilde, lambda, C1, C2)` → (Q, d, Aeq, beq, lb, ub)

以附录式 (24) 为权威。

- **Q**: `(n+n_U) × (n+n_U)`，仅来自 Gram 矩阵 + 标签 + λ。**固定 m 时 Q 不含 m**。
- **d(m)**: 线性项，m 进入此处——含 `K(xᵢ, m)` 和标签的组合。
- **Aeq**: `1 × (n+n_U)` 向量，元素为 `ỹᵢ`（α 部分）和 `-ỹᵢ`（γ 部分）。
- **beq**: 标量 `C`（附录式 8 的常数）。
- **lb**: `[0] * (n+n_U)`
- **ub**: `[C₁]*n + [C₂]*n_U`

### 6.3 `_solve_qp_oracle(Q, d, Aeq, beq, lb, ub, z0)` → (z, diagnostics)

用 `scipy.optimize.minimize`（`method="SLSQP"` 或 `"trust-constr"`）求解：

```text
max  dᵀz - ½ zᵀQz   s.t.   Aeq @ z = beq,  lb ≤ z ≤ ub
```

返回 `z = [α; γ]` 以及 `diagnostics`：目标值、等式残差、盒约束违反度、KKT 残差、状态。

### 6.4 `_rbf_centroid_delta(alpha, gamma, X, y_tilde, lambda, sigma)`

附录式 (33)：RBF 下对 `G(xᵢ, μ) = exp(-||xᵢ - μ||²/(2σ²))` 在 `μ = m̂` 处 Taylor 展开，对 μ 求导：

```text
Δ = -1/(2λσ²) · Σᵢ αᵢ ỹᵢ exp(-||xᵢ||²/(2σ²)) xᵢ
    + 1/(2λσ²) · Σ_{i=k+1}ⁿ γᵢ ỹᵢ exp(-||xᵢ||²/(2σ²)) xᵢ
```

### 6.5 `_update_centroid(m_hat, S_solve, delta, centroid_radius, tol)`

- 解 `u = S_solve⁻¹ · Δᵀ`
- `d = Δ @ u`
- 若 `d ≤ tol`: 保持 `m = m_hat`，标记 `degenerate_centroid_step`
- 否则: `m = m_hat - u · √(centroid_radius / d)`
- 更新后检查 `(m-m̂)ᵀ Ŝ_raw (m-m̂) ≤ b + tol`

### 6.6 `_recover_bias_from_kkt(alpha, gamma, K, y_tilde, m, lambda, C_alpha)`

QP oracle 版：收集 `0 < αᵢ < C₁` 的自由支持向量，由 KKT margin 条件逐个反推 `bᵢ`，取中位数。无自由变量时从上下 KKT 界构成的可行区间取中点，标记 `bias_recovery="bounded_interval"`。

附录式 (37)–(40) 的四项平均更新留给 `_recover_bias_smo_incremental`（SMO 版 PR 启用）。

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
| `centroid_opt_` | `(d,)` | 优化后质心 m |
| `centroid_covariance_raw_` | `(d,d)` | Ŝ（论文约束用，无 ridge） |
| `n_acs_iter_` | `int` | ACS 收敛轮数 |
| `acs_history_` | `list[dict]` | 每轮 {dual_obj, eq_residual, box_violation, centroid_constraint_residual} |
| `converged_` | `bool` | 收敛标志 |
| `classes_` | `(2,)` | [0, 1] |

## 8. 文件变更清单

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 1 | `pu_toolbox/utils/centroid.py` | **新建** | MoM + 协方差原语（ridge 移除） |
| 2 | `pu_toolbox/estimators/risk/ldce.py` | 修改 | import 替换 + 自行加 ridge |
| 3 | `pu_toolbox/estimators/risk/kldce.py` | **新建** | KLDCEClassifier (~600 行) |
| 4 | `pu_toolbox/estimators/risk/__init__.py` | 修改 | 导出 KLDCEClassifier |
| 5 | `pu_toolbox/registry/builtin_methods.py` | 修改 | 绑定 kldce native class |
| 6 | `tests/estimators/risk/test_ldce_math.py` | 修改 | import 路径适配 |
| 7 | `tests/estimators/risk/test_kldce_math.py` | **新建** | 公式级 MATH tests |
| 8 | `tests/estimators/risk/test_kldce_oracle.py` | **新建** | QP oracle 对照 |
| 9 | `tests/estimators/risk/test_kldce_property.py` | **新建** | 收敛/约束/h 失配 |
| 10 | `tests/contract/test_classifier_api.py` | 修改 | 添加 KLDCE factory |
| 11 | `tests/test_builtin_methods.py` | 修改 | native 6→7, api_only 9→8 |
| 12 | `docs/research/method_cards/KLDCE.md` | 修改 | 更新 Registry 引用、γ/σ 映射 |

## 9. 测试策略

| 类别 | 文件 | 验收标准 |
|------|------|----------|
| MATH | `test_kldce_math.py` | 4–8 样本手工验证 Q/d/Aeq/beq/lb/ub vs 附录式 (24)；`_rbf_centroid_delta` vs 手算 Δ；决策函数逐项核对（覆盖 `-γ`、`-C·K(x,μ)/(2λ)`、`1/(2λ)`）；可行初始化 `Aeq@z₀==beq` + 盒约束 |
| ORACLE | `test_kldce_oracle.py` | 固定 m: QP oracle 等式残差 ≤ 1e-8，盒约束满足；RBF KLDCE QP oracle vs 未来 SMO 同一 m 下目标差 ≤ 1e-6、margin 差 ≤ 1e-5 |
| PROPERTY | `test_kldce_property.py` | 椭球约束消融（有/无约束目标对比）；h 失配 `ĥ∈{0.6h,…,1.4h}` 不崩溃；随机种子可复现；Taylor 近似前后目标记录（不默认 ACS 单调） |
| API | `test_classifier_api.py` | sklearn fit/predict/clone/pipeline 全契约 |
| Registry | `test_builtin_methods.py` | 计数 + metadata 断言 |

## 10. 不纳入本 PR

- 附录广义 SMO（`_smo_*` 占位，后续 PR 替换 QP oracle）
- `kernel="linear"` 或其他非 RBF 核（附录 μ 更新是 RBF Taylor 专用）
- `max_dual_variables` 以上规模的近似核（Nyström/RFF）
- `predict_proba`（与 LDCE 一致抛出 `NotImplementedError`）
