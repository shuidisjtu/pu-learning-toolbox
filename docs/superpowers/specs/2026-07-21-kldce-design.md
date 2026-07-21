# KLDCE Design Spec

> 2026-07-21 | 基于 `docs/research/method_cards/KLDCE.md` + 线上补充附录

## 1. 背景与前置条件

- **LDCE**（线性版）已完整实现（`pu_toolbox/estimators/risk/ldce.py`），MoM 质心 + 协方差估计 + 椭球约束交替优化。
- **KLDCE**（核化版）的补充附录已取得：给出了 Algorithm 1、α/γ SMO 更新式 (21)–(26)、RBF Taylor 质心更新式 (33)–(35)、偏置四项平均式 (37)–(40)。
- **首版交付策略**：ACS 外循环 + `scipy` QP oracle 解固定 m 的对偶 QP（`n ≤ 500` 规模保护）；附录 SMO 留待后续 PR 替换。

## 2. 模块拆分

### 2.1 共享原语提取

`_mom_centroid` 和 `_centroid_covariance` 从 `ldce.py` 移至新文件：

**`pu_toolbox/utils/centroid.py`**（NEW）

```
_mom_centroid(X_U, g, rng)       → m̂   (Algorithm 1)
_centroid_covariance(X_U, ridge)  → Ŝ    (Eq. 10)
```

`ldce.py` 改为 `from ...utils.centroid import _mom_centroid, _centroid_covariance`，删掉本地定义。

### 2.2 KLDCE 新文件

**`pu_toolbox/estimators/risk/kldce.py`**（NEW）

```
KLDCEClassifier(BasePUClassifier)     # 主类
_build_dual_qp(...)                   # 式 (24) → Q, d, a, U
_solve_qp_oracle(Q, d, a, U)          # scipy.optimize 包装
_rbf_centroid_delta(alpha, gamma, ...) # 附录式 (33)
_update_centroid(m_hat, S_hat, delta, ...) # 附录式 (35) / linear 退化
_recover_bias(alpha, gamma, ...)       # 式 (37)–(40) 四项平均
# 以下为占位 — 后续 PR 启用
_smo_alpha_pair(...)                   # 式 (21)–(23)
_smo_gamma_pair(...)                   # 式 (24)–(26)
_smo_solve(...)                        # SMO 主循环
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
        flip_probability: float,         # 必填
        *,
        kernel: str = "rbf",             # "rbf" | "linear"
        gamma: float | str = "scale",
        reg_strength: float = 1.0,
        centroid_radius: float = 1.0,
        mom_groups: int = 10,
        covariance_ridge: float = 1e-4,
        max_acs_iter: int = 50,
        max_smo_iter: int = 1000,        # QP oracle 版本仅占位
        tol: float = 1e-6,
        max_train_samples: int = 500,
        random_state: int | None = None,
    ):
```

**与 LDCE 的参数差异**：去掉了 `learning_rate`、`n_inner_iter`（SMO 不需要梯度下降）；新增 `kernel`、`gamma`、`max_smo_iter`、`max_train_samples`。

## 4. fit() 流程

```text
fit(X, y_pu, *, class_prior=None):
  1. validate_pu_X_y → {+1, 0}, split P/U
  2. 校验 flip_probability ∈ (0,1), n ≤ max_train_samples
  3. p = k/[n(1-h)], 检查 |1-2ph| 近零
  4. ỹ = +1 (P), -1 (U)
  5. MoM → m̂, 协方差 → Ŝ  (utils/centroid.py)
  6. Gram 矩阵 K = kernel(X, X)
  7. 初始化 α=0, γ=0, b=0, m=m̂

  ACS 外循环 (t=1..max_acs_iter):
    a. 固定 m: _build_dual_qp → Q, d, a, U
       _solve_qp_oracle(Q, d, a, U) → z=[α;γ]
    b. 从 α,γ 按式(33) 计算 Δ
    c. 固定 z: 按式(35) 更新 m (rbf) / LDCE 退化 (linear)
    d. 按式(37-40) 恢复 b
    e. 记录 dual_obj, KKT, ||Δm||
    f. 收敛判断 → break

  8. 存储 fitted attributes
  9. _is_fitted = True
```

## 5. 决策函数

- **rbf**: `f(x) = Σᵢ αᵢỹᵢ K(x,xᵢ) + Σᵢ γᵢỹᵢ K(x,xᵢ) + b`
- **linear**: `f(x) = wᵀx + b`, `w = Σᵢ (αᵢ+γᵢ) ỹᵢ xᵢ`

## 6. 关键算法子函数

### 6.1 `_build_dual_qp(m, K, y_tilde, ...)` → (Q, d, a, U)

以附录式 (24) 为权威，构造：
- `Q`: `(n+n_U) × (n+n_U)` — 由 Gram 矩阵 + 标签 + 质心项组成
- `d`: 全 1 向量
- `a`: 等式系数向量
- `U`: [C₁,...,C₁, C₂,...,C₂]

### 6.2 `_rbf_centroid_delta(alpha, gamma, X, y_tilde, lambda, sigma)`

附录式 (33)：RBF 下对 `G(xᵢ, μ)` 做 Taylor 展开，对 μ 求导得 Δ。

### 6.3 `_update_centroid(m_hat, S_hat, delta, centroid_radius, ridge, kernel)`

- rbf: 解 `A = Ŝ + ridge·I`, `u = A⁻¹Δᵀ` → `m = m̂ - u √(b/Δu)`
- linear: `v` 由线性判别权重直接构造 → 退化为 LDCE 的闭式更新

### 6.4 `_recover_bias(alpha, gamma, K, y_tilde, C1, C2)`

附录式 (37)–(40)：对每对更新后的 α 对和 γ 对求 b₁,b₂,b_{k+1},b_{k+2}，取四项平均。

## 7. 拟合属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `support_vectors_` | `(n_sv, d)` | 支持向量 |
| `dual_alpha_` | `(n_sv,)` | α ≠ 0 的乘子 |
| `dual_gamma_` | `(n_sv,)` | γ ≠ 0 的乘子 |
| `bias_` | `float` | b₀ |
| `class_prior_` | `float` | p = k/[n(1-h)] |
| `centroid_hat_` | `(d,)` | MoM 质心 m̂ |
| `centroid_opt_` | `(d,)` | 优化后质心 m |
| `centroid_covariance_` | `(d,d)` | Ŝ |
| `n_acs_iter_` | `int` | ACS 收敛轮数 |
| `objective_history_` | `list[float]` | 每轮对偶目标 |
| `converged_` | `bool` | 收敛标志 |
| `classes_` | `(2,)` | [0, 1] |

## 8. 文件变更清单

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 1 | `pu_toolbox/utils/centroid.py` | **新建** | MoM + 协方差原语 |
| 2 | `pu_toolbox/estimators/risk/ldce.py` | 修改 | 替换本地定义为 import |
| 3 | `pu_toolbox/estimators/risk/kldce.py` | **新建** | KLDCEClassifier (~500 行) |
| 4 | `pu_toolbox/estimators/risk/__init__.py` | 修改 | 导出 KLDCEClassifier |
| 5 | `pu_toolbox/registry/builtin_methods.py` | 修改 | 绑定 kldce native class |
| 6 | `tests/estimators/risk/test_ldce_math.py` | 修改 | import 路径适配 |
| 7 | `tests/estimators/risk/test_kldce_math.py` | **新建** | 公式级 MATH tests |
| 8 | `tests/estimators/risk/test_kldce_oracle.py` | **新建** | QP oracle 对照 |
| 9 | `tests/estimators/risk/test_kldce_property.py` | **新建** | linear vs LDCE 交叉验证 |
| 10 | `tests/contract/test_classifier_api.py` | 修改 | 添加 KLDCE factory |
| 11 | `tests/test_builtin_methods.py` | 修改 | native 6→7, api_only 9→8 |
| 12 | `docs/research/method_cards/KLDCE.md` | 修改 | 更新 Registry 引用 |

## 9. 测试策略

| 类别 | 文件 | 验收标准 |
|------|------|----------|
| MATH | `test_kldce_math.py` | 4-8 样本手工验证 Q/d/a/U；_rbf_centroid_delta vs 手算；pair 更新后 aᵀz=0 |
| ORACLE | `test_kldce_oracle.py` | QP oracle 目标与 KKT 残差 ≤ 1e-8；SMO vs oracle 目标差 ≤ 1e-6 |
| PROPERTY | `test_kldce_property.py` | linear KLDCE vs LDCE 决策方向一致；椭球约束消融；h 失配稳健性 |
| API | `test_classifier_api.py` | sklearn fit/predict/clone/pipeline 全契约 |
| Registry | `test_builtin_methods.py` | 计数 + metadata 断言 |

**特别关注**：linear KLDCE 的 `_decision_function` 输出应与同参数 LDCE 的 `decision_function` 在符号和相对排序上一致——这是用已实现代码交叉验证新实现的关键回归测试。

## 10. 不纳入本 PR

- 附录广义 SMO（`_smo_*` 占位，后续 PR 替换 QP oracle）
- `kernel="poly"` 或其他非 RBF 核
- `max_train_samples` 以上规模的近似核（Nyström/RFF）
- `predict_proba`（与 LDCE 一致抛出 `NotImplementedError`）
