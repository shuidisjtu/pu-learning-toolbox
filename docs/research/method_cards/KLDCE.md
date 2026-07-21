# KLDCE 求解器设计说明：ACS + 广义 SMO

> **Implementation status (2026-07-21):** QP oracle 版本已实现。
> 类: `pu_toolbox.estimators.risk.kldce.KLDCEClassifier`
> 首版仅支持 RBF kernel (参数 `sigma`)，使用 scipy SLSQP 作为 QP oracle。
> 附录原生 SMO 留待后续 PR。

## 1. 结论与实施建议

KLDCE（Kernelized LDCE）不是“给线性 LDCE 换一个核函数”即可完成的模型。其训练问题是由以下两个子问题交替构成的非联合凸优化：

1. **固定质心 `m`**：求带两类对偶变量、不同盒约束和一个线性等式约束的核二次规划（QP）。
2. **固定对偶变量/判别函数**：在 LDCE 同款椭球约束内更新质心 `m`。

论文正文只给出原始问题、对偶问题和最终决策函数；其在线补充附录则给出了对偶推导，以及 KLDCE 的 Algorithm 1、`α/γ` 两组 SMO 更新、质心更新和偏置更新。用户提供的 *Loss Decomposition and Centroid Estimation for Positive and Unlabeled Learning (Appendix)* 正是该附录。

**推荐交付路径**：以附录 Algorithm 1 为算法规格，同时先完成 `ACS + 通用 QP oracle`，以其输出作为小数据金标准；只有在目标值、KKT 残差、预测均一致后，才以附录的 SMO 更新替换 QP oracle。不要把现有 `SVC`/LIBSVM 直接宣称为 KLDCE。

---

## 2. 论文实际提供的内容

### 2.1 已给出的数学规格

论文第 6.2 节给出：

- 含松弛变量的核化原始问题（式 23）；
- 引入 $`\alpha=(\alpha_1,\ldots,\alpha_n)`$ 和 $`\gamma=(\gamma_{k+1},\ldots,\gamma_n)`$ 后的对偶 QP（式 24）；
- 对偶的椭球质心约束、线性等式约束及盒约束；
- 由 $`\alpha,\gamma,m^*`$ 组成的判别函数（式 25）；
- 固定 `m` 时对 $`\alpha,\gamma`$ 使用 SMO，固定对偶变量时更新 `m`；
- `α` 与 `γ` 的二变量解析更新、可行区间裁剪、以误差差选择第二变量的规则；
- 高斯核下用 Taylor 展开得到的 `m` 更新；
- 四个边界样本偏置估计的平均更新规则；
- 完整 KLDCE Algorithm 1。

论文还说明：式 (24) 的“详细对偶推导和优化过程”在 online supplemental material 的 appendix 中。附录补齐了主要推导，但仍未给出 KKT 违反度的精确定义、`I/A` 与 `I'/A'` 可行边界的展开式、缓存策略、容差、外层停止判据及实现代码。

### 2.2 正文未规定、实现必须自行决定的内容

| 缺口 | 为什么重要 | 建议 |
|---|---|---|
| 工作集选择 | 附录规定第一变量违反 KKT、第二变量取最大误差差，但未定义违反度 | 用标准化 KKT residual 实现第一变量；第二变量按附录取最大 $`|E_i-E_j|`$ |
| 可行区间 | 附录仅以 `I/A`、`I'/A'` 指代上下界 | 由盒约束和守恒关系逐项求交集，并单测所有标签组合 |
| 系数/索引 | 附录的 `C₁/C₂`、`β` 命名与 Algorithm 1 的边界表述存在歧义 | 以原始拉格朗日式 (6)、式 (9)–(10) 为权威，禁止只按伪代码猜上界 |
| 停止准则 | Algorithm 1 只有 `MaxIterTime` 与抽象的 `converge` | 用最大 KKT violation + 相对目标变化 + 最大 ACS 轮数 |
| 核化质心 | 附录通过高斯核的 Taylor 展开求 `m` | 仅将该推导用于 Gaussian/RBF；其他核必须另推导，不能复用 |

---

## 3. 变量、目标与接口边界

令前 `k` 个样本为干净标注正例，后 `n-k` 个为被统一标成 `-1` 的无标签/污染负例；$`\tilde y_i\in\{-1,+1\}`$ 为观测标签。

| 变量 | 范围 | 作用 |
|---|---|---|
| $`\alpha_i`$ | 全部样本 | 第一组对偶乘子 |
| $`\gamma_i`$ | 仅无标签样本 | 第二组对偶乘子，源自无标签样本的额外松弛约束 |
| $`m`$ | 质心变量 | 在椭球内修正的污染负集质心 |
| $`\hat m,\hat S`$ | 常量 | MoM 初始质心与经验协方差 |
| $`h`$ | `(0,1)` | `+1→-1` 翻转率 |
| $`p=k/[n(1-h)]`$ | `(0,1]` | censoring PU 下的正类先验 |
| $`\lambda`$ | `>0` | 正则化系数 |
| $`b`$ | `>0` | 质心椭球半径 |
| $`C_1`$ | `1/n` | α 盒约束上界（由对偶推导固定，不可调） |
| $`C_2`$ | `1/(2n)` | γ 盒约束上界（由对偶推导固定，不可调） |
| $`C_\text{eq}`$ | `-(n-k)/(2n(1-2ph))` | 等式约束右端常数（显式计算，非从变量反推） |
| $`\sigma`$ | `>0` | RBF 带宽；$`K(x,z)=\exp(-\|x-z\|^2/(2\sigma^2))`$ |

论文式 (24) 具有下面的结构（实现应以论文原式为准）：

```math
\max_{\alpha,\gamma,m}\quad
\mathbf 1^\top\alpha+\mathbf 1^\top\gamma
-\frac12 z^\top Q(m)z+q(m)^\top z+r(m),
```

其中 $`z=[\alpha;\gamma]`$，$`Q(m)`$ 由 Gram 矩阵和标签组成，且约束为：

```math
A_\text{eq} z=C_\text{eq},\qquad 0\le z_j\le U_j,\qquad
(m-\hat m)^\top\hat S(m-\hat m)\le b.
```

其中 $`A_\text{eq}=[\tilde y_1,\ldots,\tilde y_n, -\tilde y_{k+1},\ldots,-\tilde y_n]`$，
$`U=[1/n,\ldots,1/n,\,1/(2n),\ldots,1/(2n)]`$，
$`C_\text{eq}=-(n-k)/[2n(1-2ph)]`$。
实现时不要手工猜测这些常数：将论文公式逐项编码为可测试的 `dual_objective(z, m)`、`dual_gradient(z, m)` 与 `equality_coefficients`。

---

## 4. ACS 外层设计

### 4.1 附录 Algorithm 1 的准确流程

附录第 4 节的 Algorithm 1 规定如下：

```text
输入：污染样本、η、λ、椭球半径 β、高斯核带宽 σ
1. 用正文 Algorithm 1（MoM）求初始质心 μ̂。
2. 计算质心协方差 Σ̂。
3. 初始化 α、γ、偏置 b、β 与 MaxIterTime。
4. 重复直至 ACS 收敛：
   a. 由当前 α、γ 计算 Δ；按附录式 (35) 更新 μ。
   b. 重复直至 t > MaxIterTime：
      i.  选择违反 KKT 的 α₁；选择使 |E₁-E₂| 最大的 α₂；
          以附录式 (21)–(23) 更新并裁剪 α 对。
      ii. 选择违反 KKT 的 γ_{k+1}；选择使 |E_{k+1}-E_{k+2}| 最大的 γ_{k+2}；
          以附录式 (24)–(26) 更新并裁剪 γ 对。
      iii.以附录式 (37)–(40) 更新偏置 b。
      iv. t ← t+1。
5. 返回式 (25)/附录 Algorithm 1 最后一行的 f(x)。
```

其中 $`E_i=f(x_i)-\tilde y_i`$；$`Q_i=f(x_i)+\tilde y_i`$。Algorithm 1 的 `β` 是质心椭球半径，但附录第 3 节又用 $`\beta_i`$ 表示拉格朗日乘子；代码中必须改名，例如 `centroid_radius` 与 `xi_nonnegative_multiplier`，避免覆盖。

### 4.2 高斯核质心更新：附录特有前提

附录不是直接使用线性 LDCE 的 $`m\leftarrow\hat m+\hat S^{-1}w\cdots`$。它明确选择 Gaussian kernel，在 **$`\mu=0`$ 处** 对 $`G(x_i,\mu)`$ 作 Taylor 展开（注意：不是围绕当前质心 $`\hat\mu`$ 展开），利用 $`G(\mu,\mu)=1`$，然后定义：

```math
\Delta=-\frac{1}{2\lambda\sigma^2}\sum_{i=1}^{n}
\alpha_i\tilde y_i e^{-\|x_i\|^2/(2\sigma^2)}x_i
+\frac{1}{2\lambda\sigma^2}\sum_{i=k+1}^{n}
\gamma_i\tilde y_i e^{-\|x_i\|^2/(2\sigma^2)}x_i.
```

由此得到附录式 (35)：

```math
\mu\leftarrow\hat\mu-\Delta\hat\Sigma^{-1}
\sqrt{\frac{\beta}{\Delta\hat\Sigma^{-1}\Delta^\top}}.
```

这是 **RBF/Gaussian 专用且依赖 Taylor 近似** 的子问题解，不是任意 Mercer kernel 的通用闭式更新。第一版 KLDCE 应只接受 `kernel="rbf"`；若支持 linear/poly 等核，必须重新推导 `G(x,\mu)` 的 `\mu` 子问题并新增 oracle 对照。

### 4.3 流程

```text
输入：X、y_pu、RBF 带宽 σ、h、λ、b、MoM 分组数 g
1. 验证 censoring PU 输入 (k>0, n_U>0, mom_groups≤n_U)；转换为 ỹ∈{-1,+1}；
   计算 p=k/[n(1-h)]，**显式检查 0<p≤1**（p 是类别先验，违反时拒绝而非静默继续）；
   检查 |1-2ph| 近零（等价于 |1-2k/n| 近零，病态时拒绝）。
2. 对污染负集运行 MoM：**论文对象是 {ỹ_i·x_i | ỹ=-1} = {-x_i}**，因此传入 `-X_U` 得到 m̂。
   按论文式 (10) 计算协方差 Ŝ_raw = _centroid_covariance(X_U)（符号相消，可用 X_U 直接计算）。
3. ridge 默认为 0（论文原式）；若 Ŝ_raw 奇异，严格模式报错，变体模式加 ridge>0。
4. 初始化 μ←m̂。
   构造初始 A_eq=[ỹ; -ỹ_U], C_eq=-(n-k)/(2n(1-2ph))。
   调用 Phase-I LP 找可行初值 z₀（α=γ=0 违反 A_eq·z=C_eq 当 C_eq≠0）。
5. 对 outer_iter=1..max_outer_iter：
   a. 固定 μ：构造 Q、d(μ)（详见 §6.2 逐块公式）、A_eq、lb、ub（C_alpha=1/n, C_gamma=1/(2n)）；
      调用 QP oracle 得 z=[α;γ]；记录 dual_obj、eq_residual、box_violation。
      **注**：论文 Algorithm 1 是先更新 μ 再 SMO 更新 α/γ；QP oracle 版先固定 μ 解联合 QP 再更新 μ。
      两者均为 ACS 的合理块坐标顺序变化，但 QP oracle 版不是论文 Algorithm 1 的逐行实现。
   b. 从 α、γ 按附录式 (33) 计算 Δ（**μ=0 Taylor 展开**）。
   c. 固定 α、γ：解 Ŝ_solve·u=Δᵀ；令 q=uᵀ·Ŝ_raw·u（**缩放基准始终为 Ŝ_raw**）；
      若 q≤ε：保持 μ=m̂；否则 μ=m̂-u·√(b/q)。验证 (μ-m̂)ᵀŜ_raw(μ-m̂)≤b。
   d. **QP oracle 版**：由自由 α 支持向量的 KKT margin 条件恢复 b₀（中位数）；
      无自由变量时用 bounded_interval（含 margin 常数 1 的 L/U 公式）。
      **附录式 (37)–(40) 的四项平均增量更新留给 SMO 版。**
   e. 若 KKT、目标相对变化和 μ 变化均低于 tol，停止。
6. 固化支持向量、对偶变量、μ* 与核参数。
输出：训练后的 KLDCE 模型与 solver diagnostics。
```

### 4.4 数值化质心更新

令 $`A=\hat\Sigma+\text{ridge}\cdot I`$（默认 `ridge=0` 即论文原式），解 $`u=A^{-1}\Delta^\top`$。
**缩放基准始终为原始 $`\hat\Sigma`$**：计算 $`q=u^\top\hat\Sigma u`$（非 $`\Delta u`$）。
若 $`q\le\varepsilon`$，保持 `μ=μ̂` 并标记 `degenerate_centroid_step`；否则令：

```math
\mu\leftarrow\hat\mu-u\sqrt{\beta/q}.
```

这保证 $`(\mu-\hat\mu)^\top\hat\Sigma(\mu-\hat\mu)=\beta`$ 严格满足（非仅事后验证）。
约束检查仍使用论文定义的 $`\hat\Sigma`$（不含 ridge）。
ridge>0 时标记 `centroid_solver="ridge_stabilized"`。

### 4.5 外层收敛与单调性

ACS 通常只保证收敛到驻点，不能保证全局最优。每次更新都应至少记录：

- `dual_objective_fixed_m`；
- 质心约束残差；
- 最大 KKT violation；
- `relative_objective_change`；
- `m_change_norm`；
- 失败原因（不可行、NaN、达到迭代上限、核矩阵异常）。

若某次更新使可验证目标劣化超过容差，回滚到上一可行状态并降低步长/改用阻尼质心更新。该阻尼是工程保护，不是论文声明的算法保证。

---

## 5. 固定 `m` 的 QP：先实现 oracle

### 5.1 为什么需要 oracle

广义 SMO 的错误往往表现为“看似收敛但分类边界错误”。小规模通用 QP oracle 可独立验证：

- 对偶变量是否可行；
- 对偶目标是否正确；
- KKT 残差是否足够小；
- 自研 SMO 是否得到相同的目标和预测。

### 5.2 oracle 的最小契约

固定 `m` 后，oracle 必须返回：

```python
DualSolveResult(
    z,                    # 拼接后的 [alpha, gamma]
    objective,
    equality_residual,
    max_box_violation,
    max_kkt_violation,
    status,
)
```

验收时，对小样本（例如 `n≤100`）要求：

```text
abs(A_eq @ z - C_eq) <= 1e-8           # 等式约束（非 a@z=0）
max(-z, z - [1/n, …, 1/(2n)]) <= 1e-8  # 盒约束 (α≤1/n, γ≤1/(2n))
SMO 目标与 oracle 的相对差 <= 1e-6
SMO 与 oracle 在固定测试点上的 margin 最大差 <= 1e-5
```

通用凸 QP 求解器只能作为开发与测试依赖；它不是 KLDCE 的最终可扩展后端。

---

## 6. 广义 SMO 设计

### 6.1 与标准 C-SVC SMO 的差异

标准 SMO 处理单组变量、统一上界和典型 $`y^\top\alpha=0`$。KLDCE 固定 `m` 的对偶有两组变量（`α`、`γ`），每组上界与线性项不同，并共享一个线性等式约束。因此：

- **可以复用**：核缓存、误差缓存、二变量 QP、KKT 选择和停止框架；
- **不能直接复用**：LIBSVM 的 C-SVC 变量定义、成对上下界公式和偏置恢复；
- **必须重写**：`z` 的统一索引、每一维 `U_j`、等式系数 `a_j`、对偶梯度以及 pair feasible interval。

### 6.2 统一变量形式

拼接：

```text
z = [α_1, …, α_n, γ_{k+1}, …, γ_n]
U = [1/n, …, 1/n, 1/(2n), …, 1/(2n)]
```

令固定 `m` 后的最大化目标为：

```math
D(z)=d^\top z-\frac12z^\top Qz+const,
\qquad A_\text{eq} z=C_\text{eq},\quad 0\le z\le U.
```

梯度：$`g=\nabla D(z)=d-Qz`$。所有符号、`d` 与 `Q` 必须通过对论文式 (24) 的逐项单元测试确认。

### 6.3 二变量可行更新

选择 `i,j` 后，在保持等式约束时用方向：

```math
z_i\leftarrow z_i+a_jt,
\qquad z_j\leftarrow z_j-a_it.
```

这保证 $`a_i\Delta z_i+a_j\Delta z_j=0`$。由盒约束计算 `t∈[L,H]`：

```math
0\le z_i+a_jt\le U_i,
\qquad 0\le z_j-a_it\le U_j.
```

沿该方向，目标是单变量凹二次函数。定义 $`s=a_je_i-a_ie_j`$，则：

```math
D(z+ts)=D(z)+t(s^\top g)-\frac12t^2(s^\top Qs).
```

若 $`\eta=s^\top Qs>\varepsilon`$：

```math
t^*=clip((s^\top g)/\eta, L, H).
```

若 $`\eta\le\varepsilon`$，分别比较 `t=L`、`t=H` 与 `t=0` 的目标值，选择最大的可行值。这是核矩阵半正定/数值退化时必要的保护。

更新后增量维护梯度：

```math
g\leftarrow g-t^*Qs.
```

该统一形式将两种变量的差异限定在 `a、U、d、Q` 中；它是从论文式 (24) 到可实现 SMO 的关键桥梁。

### 6.4 工作集与停止规则

对最大化问题，先根据 KKT 找出最违反的可上升坐标 `i`，再选令预期目标增益最大的 `j`。实现可采用两阶段策略：

1. **正确性优先**：遍历所有合法 pair，选择最大预测增益；复杂度高但适合 oracle 对照的小数据测试。
2. **性能优先**：最大 KKT violation 选 `i`，再从活动集按二次近似增益选 `j`；使用核行缓存。

停止必须同时满足：

```text
max_kkt_violation <= kkt_tol
and equality_residual <= feasibility_tol
and box_violation <= feasibility_tol
```

仅用“本轮没有变量更新”或“参数变化很小”不足以证明 QP 已解。

### 6.5 偏置恢复

论文式 (25) 含偏置 $`b_0`$，恢复策略因求解器而异：

**QP oracle 版（首版交付）**：
1. 收集**自由 α 和自由 γ**：$`0<\alpha_i<C_1`$ 或 $`0<\gamma_i<C_2`$。
2. 由 KKT margin 条件：
   - 对自由 α（$`\tilde y_i=+1`$）：$`b_i=1-g_i`$
   - 对自由 γ（$`\tilde y_i=-1`$）：$`b_i=1-g_i`$（无标签样本标签为 -1，KKT 条件为 $`-1\cdot(g_i+b)\ge 1`$，自由时取等号得 $`b=1-g_i`$）
   其中 $`g_i=f(x_i)-b_0`$（决策分数不含 bias），$`f(x_i)`$ 按式 (25) 计算。对所有自由变量得到的 $`b_i`$ 取中位数。
3. **无自由变量时**（α 和 γ 全部在边界）：由六种 KKT 边界构造可行区间：
```math
\begin{aligned}
L=\max(&\{1-g_i\mid\alpha_i=0,\tilde y_i=+1\}\cup\{-1-g_i\mid\alpha_i=C_1,\tilde y_i=-1\}\;\cup\\
       &\{1-g_i\mid\gamma_i=0,\tilde y_i=-1\}\\
U=\min(&\{1-g_i\mid\alpha_i=C_1,\tilde y_i=+1\}\cup\{-1-g_i\mid\alpha_i=0,\tilde y_i=-1\}\;\cup\\
       &\{1-g_i\mid\gamma_i=C_2,\tilde y_i=-1\}
\end{aligned}
```
若 $`L\le U`$ 取中点；否则标记 `indeterminate`，$`b_0=0`$。

**SMO 版（后续 PR）**：附录式 (37)–(40) 的四项平均增量更新。原生复现优先采用论文的四项平均，并用 KKT oracle 检查可行性。

禁止照搬 `sklearn.svm.SVC` 内部逻辑——其偏置恢复基于标准 C-SVC 对偶，KLDCE 的决策函数和 KKT 条件不同。

### 6.6 附录给出的两类 SMO 更新

附录已经指定两组变量**分别成对更新**，而非任意从拼接向量中混合选择 `α` 与 `γ`：

| 对 | 第一变量 | 第二变量选择 | 更新与投影 |
|---|---|---|---|
| `α` 对 | 选择违反 KKT 的 $`\alpha_1`$ | $`\arg\max_j|E_1-E_j|`$ | 用式 (21) 的解析步更新 $`\alpha_1`$，投影到 $`[I,A]`$；再由式 (23) 更新 $`\alpha_2`$ |
| `γ` 对 | 选择违反 KKT 的 $`\gamma_{k+1}`$ | $`\arg\max_j|E_{k+1}-E_j|`$ | 用式 (24) 的解析步更新 $`\gamma_{k+1}`$，投影到 $`[I',A']`$；再由式 (26) 更新 $`\gamma_{k+2}`$ |

对 `α` 对，附录式 (21) 为：

```math
\alpha_1^{t+1}=\alpha_1^t-
\frac{2\lambda\tilde y_1(E_1-E_2)}
{G_{11}-2G_{12}+G_{22}},
```

随后裁剪到 $`[I,A]`$，并令：

```math
\alpha_2^{t+1}=\alpha_2^t+
\tilde y_1\tilde y_2(\alpha_1^t-\alpha_1^{t+1}).
```

对 `γ` 对，附录式 (24) 为：

```math
\gamma_{k+1}^{t+1}=\gamma_{k+1}^t+
\frac{2\lambda\tilde y_{k+1}(Q_{k+1}-Q_{k+2})}
{G_{k+1,k+1}-2G_{k+1,k+2}+G_{k+2,k+2}},
```

再裁剪到 $`[I',A']`$ 并由式 (26) 更新配对变量。附录没有展开 `I/A` 与 `I'/A'` 的标签组合；实现时应以各变量的上界和配对守恒关系求区间，并用枚举标签组合的单元测试固定其行为。

附录的偏置更新与标准 SMO 的误差缓存形式一致：先从每个 pair 得到 $`b_1,b_2,b_{k+1},b_{k+2}`$（式 37–39），再取四者平均（式 40）。这与“只从一个自由支持向量恢复偏置”的通用建议不同；原生复现应优先采用论文的四项平均，并用 primal/KKT oracle 检查其可行性。

---

## 7. 核函数与复杂度

### 7.1 核函数接口

```python
kernel(X_left, X_right) -> ndarray[float64, shape=(n_left, n_right)]
```

按附录推导，首批只支持：`rbf`。每个核必须满足近似对称性：

```text
max_abs(K(X, X) - K(X, X).T) <= kernel_symmetry_tol
```

RBF 采用 $`K(x,z)=\exp(-\|x-z\|^2/(2\sigma^2))`$（论文原生参数 $`\sigma`$）。
不使用 `sklearn.metrics.pairwise.rbf_kernel`（其参数是 $`\gamma=1/(2\sigma^2)`$，直接传入 $`\sigma`$ 会得到错误核矩阵）。
默认 $`\sigma=1/\sqrt{d}`$（`d` 为特征维数），对应 $`\gamma=d/2`$。

### 7.2 规模限制

精确 Gram 矩阵为 $`O(n^2)`$ 内存，SMO 训练通常至少需要频繁的核行访问。论文自身也指出 KLDCE 的核计算较慢。因此：

- 第一版应声明 `max_train_samples`，超过阈值显式拒绝；
- 核缓存使用 LRU 行缓存；
- 大样本仅提供线性 LDCE，或另开近似核（Nyström/RFF）研究任务；
- 不能在未验证近似目标后把近似核称为论文原始 KLDCE。

---

## 8. 测试与验收门槛

### 8.1 公式级单测

- 对手工构造的 4–8 个样本，逐项核对 `Q、d(μ)、A_eq、C_eq、lb、ub` 与式 (24)。Q 的逐块公式见 design spec §6.2。
- **单样本交叉项测试**：从 §5 决策函数的 RKHS 系数 $`r=\alpha\tilde y-\gamma\tilde y_U-C_\text{eq}\,\varphi(\mu)`$ 展开 $`\lambda\|r\|^2`$，逐项核对 Q 三块和 d(μ) 的符号——防止用同一错误公式互相验证。
- **可行初始化测试**：`A_eq @ z₀ == C_eq` 且盒约束满足（C_eq ≠ 0 时 α=γ=0 不可行）。
- **决策函数逐项核对**：分别验证 `-γ` 项、`-C_eq·K(x,μ)/(2λ)` 项、`1/(2λ)` 缩放。
- 任意 pair update 后，验证盒约束和 $`A_\text{eq} z=C_\text{eq}`$。
- 与显式二次函数重新计算相比，增量梯度误差低于 `1e-10`（float64、小样本）。
- 椭球 `μ` 更新后验证 $`(\mu-\hat\mu)^\top\hat\Sigma(\mu-\hat\mu)\le b`$；奇异协方差（ridge=0 严格模式）清晰报错而非静默 pinv。
- **bounded_interval 四种 KKT 情况**各写一个单测（α=0/C × ỹ=+1/-1）。
- 默认 `covariance_ridge=0` 即论文原式；>0 时标记 `centroid_solver="ridge_stabilized"`。

### 8.2 oracle 对照

| 测试 | 要求 |
|---|---|
| 固定 `m` | SMO 与通用 QP 的目标、可行性、KKT 残差一致 |
| 单次 ACS | 给定同一初值时，两者 `m` 更新一致 |
| 多次 ACS | 两者目标轨迹接近，测试 margin 与预测一致 |
| 退化核 | 重复样本/近重复样本不崩溃，状态明确为收敛、受限或数值失败 |

### 8.3 端到端基准

使用已知真标签的合成数据，先做 train/test 切分，再仅在训练正例中随机隐藏比例 $`h`$。比较 LDCE、KLDCE（RBF）、uPU/nnPU 与”U 全当负类”基线。非线性同心圆等数据上 RBF KLDCE 应显示增益。

必须额外做：

- `h` 失配：$`\hat h\in\{0.6h,0.8h,h,1.2h,1.4h\}`$；
- 质心约束消融：有/无椭球约束；
- 随机种子重复；
- 训练时间、峰值内存与支持向量数量记录。

---

## 9. 推荐参考材料

1. **Gong et al., TPAMI 2021 + Appendix** — KLDCE 的唯一原始数学与 SMO 规格；附录已取得，应与正文成对使用。[IEEE 记录](https://ieeexplore.ieee.org/document/8839365/)
2. **Platt (1998), SMO** — 二变量 QP 解析更新、KKT 与启发式选择的原始权威来源。[Microsoft Research PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/1998/04/sequential-minimal-optimization.pdf)
3. **Chang & Lin, LIBSVM** — 可生产实现的 SMO、核缓存、收敛与数值细节。只复用框架，不复用其标准 C-SVC 对偶公式。[官方论文](https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf)；[官方实现](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
4. **Gao et al. (AAAI 2016), LICS** — LDCE 质心平滑的直接来源，给出了 MoM、椭球约束与交替优化的完整线性伪代码。[AAAI 论文](https://cdn.aaai.org/ojs/10293/10293-13-13821-1-2-20201228.pdf)
5. **Shi et al. (IJCAI 2018)** — LDCE 前身版本，可核对线性推导；它不含 KLDCE，不能替代补充材料。[IJCAI 论文](https://www.ijcai.org/proceedings/2018/0373.pdf)

---

## 10. 实现决策清单

- [x] 已取得并归档 KLDCE supplemental appendix；逐式核对 `C/C₁/C₂`、`μ` 子问题和偏置。
- [ ] 将式 (24) 写为纯函数，并用数值差分检查梯度。
- [ ] 实现固定 `m` 的 QP oracle 与 KKT 检查器。
- [ ] 实现 ACS，先通过 oracle 端到端验证。
- [ ] 实现统一变量的广义 SMO，并完成 oracle 回归测试。
- [ ] 加入核缓存、warm start、失败诊断与规模保护。
- [ ] 在明确数据满足 censoring PU 且 `h` 可用时，再将 KLDCE 注册为原生实现。
