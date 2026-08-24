# Method Card: KLDCE（Kernelized LDCE）

## 1. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Loss Decomposition and Centroid Estimation for Positive and Unlabeled Learning |
| Authors | Chen Gong, Hong Shi, Tongliang Liu, Chuang Zhang, Jian Yang, Dacheng Tao |
| Venue | IEEE TPAMI |
| Year | 2021（online 2019） |
| Setting | censoring PU，单一 i.i.d. 训练样本；核化（KLDCE），首版仅支持 RBF kernel（参数 $`\sigma`$） |
| 数学规格 | 论文 §6.2 式 (23)–(25)（核化原始问题、对偶 QP、决策函数）+ 在线补充附录（对偶推导、Algorithm 1、$`\alpha/\gamma`$ 两组 SMO 更新、质心更新、偏置更新） |
| Requires class prior | `False`（由 `h` 与观测正例比例估计） |
| Requires propensity | `True`（翻转率 `h`；可调参或外部估计） |
| Requires negative samples | `False` |
| GPU required | `False` |

KLDCE（Kernelized LDCE）是线性 LDCE 的核化版，线性版见 [`LDCE.md`](LDCE.md)。其训练问题是两个子问题交替构成的**非联合凸**优化：固定质心 `m` 时求带两类对偶变量、不同盒约束和一个线性等式约束的核二次规划（QP）；固定对偶变量/判别函数时在椭球约束内更新质心 `m`。

### Assumptions

与 LDCE 相同，令真实标签 $`Y\in\{-1,+1\}`$，观测/污染标签为 $`\tilde Y`$：

```math
P(\tilde Y=-1\mid Y=+1)=h,\qquad
P(\tilde Y=+1\mid Y=-1)=0.
```

即正例以常数概率 `h` 被翻转为观测负类（SCAR 型 censoring），负例绝不被观测为正例。

---

## 2. 问题设定与符号

前 `k` 个样本为干净标注正例，后 `n-k` 个为被统一标成 `-1` 的无标签/污染负例；$`\tilde y_i\in\{-1,+1\}`$ 为观测标签。

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

---

## 3. 核心公式

### 3.1 对偶 QP 结构（论文式 24）

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

### 3.2 高斯核质心更新（附录式 35）

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

该质心更新是 RBF/Gaussian 专用且依赖 $`\mu=0`$ Taylor 近似的子问题解，不是任意 Mercer kernel 的通用闭式更新。

### 3.3 两类 SMO 更新式（附录式 21–26）

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

再裁剪到 $`[I',A']`$ 并由式 (26) 更新配对变量。其中 $`E_i=f(x_i)-\tilde y_i`$；$`Q_i=f(x_i)+\tilde y_i`$。

### 3.4 偏置更新（附录式 37–40）

附录的偏置更新与标准 SMO 的误差缓存形式一致：先从每个 pair 得到 $`b_1,b_2,b_{k+1},b_{k+2}`$（式 37–39），再取四者平均（式 40）。这与“只从一个自由支持向量恢复偏置”的通用建议不同。

---

## 4. 算法概要

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

注意：Algorithm 1 的 `β` 是质心椭球半径，但附录第 3 节又用 $`\beta_i`$ 表示拉格朗日乘子；代码中必须改名（例如 `centroid_radius` 与 `xi_nonnegative_multiplier`），避免覆盖。

### 4.2 两类 SMO 更新对的选择规则

附录指定两组变量**分别成对更新**，而非任意从拼接向量中混合选择 `α` 与 `γ`：

| 对 | 第一变量 | 第二变量选择 | 更新与投影 |
|---|---|---|---|
| `α` 对 | 选择违反 KKT 的 $`\alpha_1`$ | $`\arg\max_j|E_1-E_j|`$ | 用式 (21) 的解析步更新 $`\alpha_1`$，投影到 $`[I,A]`$；再由式 (23) 更新 $`\alpha_2`$ |
| `γ` 对 | 选择违反 KKT 的 $`\gamma_{k+1}`$ | $`\arg\max_j|E_{k+1}-E_j|`$ | 用式 (24) 的解析步更新 $`\gamma_{k+1}`$，投影到 $`[I',A']`$；再由式 (26) 更新 $`\gamma_{k+2}`$ |

---

## 5. 论文边界

- **正文未规定、需自行决定的内容**：附录规定第一变量违反 KKT、第二变量取最大误差差，但**未定义违反度**；`I/A`、`I'/A'` 可行边界的展开式未给出；`C₁/C₂`、`β` 命名与 Algorithm 1 的边界表述存在歧义；Algorithm 1 只有 `MaxIterTime` 与抽象的 `converge` 停止判据；核化质心仅给出 Gaussian 核 Taylor 展开推导。完整可执行实现仍需从原始拉格朗日式 (6)、式 (9)–(10) 出发逐项编码。
- **ACS 收敛性**：外层交替优化（ACS）通常只保证收敛到驻点，不能保证全局最优。
- **核计算复杂度**：精确 Gram 矩阵为 $`O(n^2)`$ 内存；论文自身也指出 KLDCE 的核计算较慢，大样本（如 `Skin` 约 245k）应使用线性 LDCE 而非 KLDCE。

---

## 6. 实现注记

> 见 ADR-0014 #5（RBF 采用论文原生参数 $`\sigma`$，勿直接传 `sklearn.metrics.pairwise.rbf_kernel` 的 $`\gamma=1/(2\sigma^2)`$；默认 $`\sigma=1/\sqrt{d}`$，对应 $`\gamma=d/2`$）

> 见 ADR-0014 #6（registry `kldce` 曾被误注册为 `centroid_pu` 别名导致静默解析到线性 LDCE，2026-08-05 已修复；`KLDCEClassifier` 构造要求 `flip_probability`，auto 模式无法自动实例化，需显式传实例）

- **QP oracle 版不是论文 Algorithm 1 的逐行实现**：当前实现先固定 `μ` 解联合 QP 再更新 `μ`（论文是先更新 `μ` 再 SMO 更新 `α/γ`）。两者均为 ACS 的合理块坐标顺序变化，但不可宣称逐行复现。
- **质心缩放基准始终为原始 $`\hat\Sigma`$**：计算 $`q=u^\top\hat\Sigma u`$（非 $`\Delta u`$），更新后 $`(\mu-\hat\mu)^\top\hat\Sigma(\mu-\hat\mu)=b`$ 严格满足（非仅事后验证）；约束检查不使用 ridge 后的矩阵，`ridge>0` 时标记 `centroid_solver="ridge_stabilized"`。
- **质心更新为 RBF 专用 $`\mu=0`$ Taylor 近似**：若支持 linear/poly 等核，必须重新推导 $`G(x,\mu)`$ 的 `\mu` 子问题并新增 oracle 对照。
- **数值校验**：显式检查 $`0<p\le1`$（违反时拒绝而非静默继续）与 $`|1-2ph|`$ 近零（等价于 $`|1-2k/n|`$ 近零，病态时拒绝）；`ridge` 默认为 0（论文原式），若 $`\hat\Sigma`$ 奇异，严格模式报错，变体模式加 `ridge>0`。
- **禁止照搬 `sklearn.svm.SVC` 内部逻辑**：其偏置恢复基于标准 C-SVC 对偶，KLDCE 的决策函数和 KKT 条件不同。
- **偏置恢复优先采用论文式 (37)–(40) 的四项平均**，并用 primal/KKT oracle 检查其可行性。
- 对偶常数不要手工猜测：将论文公式逐项编码为可测试的 `dual_objective(z, m)`、`dual_gradient(z, m)` 与 `equality_coefficients`。
- **收敛判据（2026-08-24 修正）**：QP oracle 版曾以 SLSQP 解处目标梯度范数
  作为 `kkt_residual`；约束 QP 最优解处该梯度由乘子平衡、不为零，导致 ACS
  停止判据永不触发（出现"QP 最优但 300 轮不收敛"）。判据现为"目标/μ 相对
  变化 + eq/box 可行性"组合，严格 KKT（乘子恢复）经 `_true_kkt_residual`
  进入诊断与测试（见实现注记与工具箱 benchmark findings）。

---

## 7. 论文实验参考

KLDCE 与 LDCE 共享同一论文实验协议与数据集（5 折、$`h\in\{0.2,0.3,0.4\}`$、配对 t 检验，数据集含 UCI/USPS/HockeyFight/NBA）——见 [`LDCE.md`](LDCE.md) §7。KLDCE 特有设置：

| 项目 | 论文设置 |
|---|---|
| 核带宽搜索 | $`s\in\{2^0,2^1,2^2,2^3\}`$ |
| 对照方法 | WSVM、uPU、nnPU、RP、LDCE、KLDCE |
| 消融 | 椭球约束有/无版本对比（去掉约束时准确率显著下降）；$`\hat h\in\{0.6h,0.8h,h,1.2h,1.4h\}`$ 翻转率失配评估 |
| 规模边界 | 论文指出核计算较慢；未采用近似核时不应承诺 KLDCE 在大规模（如 Skin）运行 |

---

## 8. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `official_related`：无官方代码，论文正文 + 在线补充附录为唯一权威 |
| 实现状态（2026-08-24） | QP oracle 版已实现（scipy SLSQP 作为 QP oracle、RBF-only、$`\sigma`$ 默认 `1/\sqrt{d}`）；2026-08-24 修正 ACS 收敛判据（生产判据=目标/μ 相对变化 + eq/box 可行性，严格 KKT 经 `_true_kkt_residual` 进入诊断与测试，见 §6 注记）；附录原生 SMO 留待后续 PR |
| Registry | `kldce` 独立注册（2026-08-05），详见 ADR-0014 #6 |
| 复现风险 | 正文未规定 KKT 违反度定义、可行区间展开式、缓存策略、容差与停止判据（见 §5），SMO 需从附录式 (21)–(26)、(37)–(40) 自行实现并用 QP oracle 对照验证；精确 Gram 矩阵 $`O(n^2)`$ 内存 |

### 参考资料

1. **Gong et al., TPAMI 2021 + Appendix** — KLDCE 的唯一原始数学与 SMO 规格；附录已取得，应与正文成对使用。[IEEE 记录](https://ieeexplore.ieee.org/document/8839365/)
2. **Platt (1998), SMO** — 二变量 QP 解析更新、KKT 与启发式选择的原始权威来源。[Microsoft Research PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/1998/04/sequential-minimal-optimization.pdf)
3. **Chang & Lin, LIBSVM** — 可生产实现的 SMO、核缓存、收敛与数值细节。只复用框架，不复用其标准 C-SVC 对偶公式。[官方论文](https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf)；[官方实现](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
4. **Gao et al. (AAAI 2016), LICS** — LDCE 质心平滑的直接来源，给出了 MoM、椭球约束与交替优化的完整线性伪代码。[AAAI 论文](https://cdn.aaai.org/ojs/10293/10293-13-13821-1-2-20201228.pdf)
5. **Shi et al. (IJCAI 2018)** — LDCE 前身版本，可核对线性推导；它不含 KLDCE，不能替代补充材料。[IJCAI 论文](https://www.ijcai.org/proceedings/2018/0373.pdf)
