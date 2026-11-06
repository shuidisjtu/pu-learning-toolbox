# Method Card: KLDCE（Kernelized LDCE）

> 参数契约（签名、参数表、返回结构）以 [API 参考](../../user/reference/api.md) 为权威；本文档只记论文研究内容、实现边界与复现状态。

2026-11-04 崩溃路径修复：原 `_rbf_kernel` 一次构造全尺寸 GEMM，在本机 NumPy
OpenBLAS 0.3.23.dev / 2 线程环境曾出现 native 崩溃；本轮旧 CIFAR 调用在 90 秒
上限内也未返回。改为每 2048 行调用同一 `build_rbf_basis`，保留所有样本、所有核中心、
完整稠密结果与原 dtype，不改核公式、QP、ACS/SMO 预算或不可行域判定。
三个真实缓存的完整核构造通过：CIFAR 45000×512、MNIST/Fashion-MNIST 各 54000×512，
耗时约 46/226/40 秒，进程峰值 RSS 最大约 14.21 GiB。浮点乘法分块会改变舍入；
抽样对照逐行原公式/float64 直接距离的核值最大绝对差约 9.78e-4，不宣称逐位一致。
这只证明已定位的核构造路径可运行，不等于完整 QP/优化收敛或全部失败任务已恢复；
其他估计器的共享 RBF 默认行为未修改，原结果不回填。

同轮继续定位：核构造通过后，SciPy 1.15.1 的 `_highs_wrapper.py` 在逐列提取
`basis.col_status` / `solution.col_dual` 时反复复制 Python 列表，81,000 个变量的
Phase-I 在 240 秒诊断上限内未返回。现在利用本问题仅有一条等式和有限盒约束，
从等式最小值顶点按索引顺序填充容量，线性时间构造可行点；仍核验原 1e-10 等式残差
及盒约束，超界拒绝。零目标的 Phase-I 中任一可行点都是最优点，但新顶点不保证
等于 HiGHS 原顶点，有限 ACS/SMO 轨迹可能变化，不声称数值重放。随机小问题经独立
HiGHS LP 对照、大规模 81,000 变量及现有公式/KKT/SMO/性质测试验证；真实 CIFAR
该阶段约 0.002 秒完成。正式结果须用新代码身份重跑验收，不改写旧结果。
对偶 QP 构造同时改为按 1024 行直接填入原四个稠密块，避免并存完整 `yy`、`Q_aa`、
`Q_ag`、`Q_gg` 临时矩阵；完整 Q 的形状、float64、约束与目标不变，四块逐元素公式
对照通过。该修改降低临时内存，不把稠密 QP 变成低秩/抽样算法，最终 Q 本身仍可能很大。
资源计划与候选预检的命名 payload 下界随之下调为仅 `Q`（构造期分块临时矩阵不计入），
旧五项临时块不再计入；该下界不是峰值内存或正式准入。
真实 CIFAR 45000 行受限拟合（1 ACS / 2 SMO）已完整返回，包含 KKT bias 恢复；
在同样 `NUMPY_MADVISE_HUGEPAGE=0`、BLAS 2 线程下，QP 分块前后进程峰值
RSS 约 109.03 → 57.48 GiB。该环境设置与既有生产队列一致；探针仅验证阶段可运行和
临时内存，不作为完整迭代收敛、正式预算或分类指标验收。

资源预检只计算已命名稠密数组的 payload 下界，不等于峰值内存或拟合批准。

2026-11-01：独立 `run_classical_candidate_experiment.py --method kldce` 已接 SCAR 四角色、
PA/OA 与选中预测规则回放、逐次成功/失败收据。训练参数须全部显式给出，`flip_probability`
由训练生成器的 `1-c_realized` 绑定（复用 v3 已定义的机制口径，不是 `1-c_requested`）；
固定数量取整后两者可能不同。`prior_policy=derived` 保留原生 `k/[n(1-h)]`；
`population_override` 显式转发选择侧总体先验，二者不得混称。
`KLDCECandidateTrainer` 在拟合前检查命名 QP/协方差及全矩阵预测的 payload 下界；
这是拒绝屏障，不保证完整峰值内存。成功收据记真实 ACS 历史、原生收敛状态，
历史只保留每轮最后一个 QP，故 rollback 总次数/总 SMO 成本保持未知，不伪造 neural epoch。
失败不复用旧 ACS 历史，记录已开始/完成的整次 fit 与耗时；核心求解器文件未改，
浮点 32/64、两种先验策略的原生 fit/预测/pickle 有逐值回归，不代表论文数值复现。
预检的 h 与原 `fit` 一样先转 Python float；已用当前 NumPy 默认与弱标量提升模式
核对 NumPy float32/64 参数，避免预检先验提前舍入（测试钩子状态在结束后恢复）。

新增可行性预检直接对应当前核心 Phase-I LP：`C_eq=-n_U/[2n(1-2ph)]`，
`Aeq·z` 在盒约束内可达区间为 `[-n_U/n, n_P/n+n_U/(2n)]`。
等式右端超区间则不启动 fit；已用原始 LP 交叉核对正/负分母、内点与边界。
例如 `p=0.5, h=0.7` 时右端低于可达下限，当前实现无解，不是加迭代次数能修复的预算问题。
不调整 h/先验/约束来凑可行；部分低标记率格子的正式适用性仍待协议裁决，冻结单元不自动放行。
h 与先验须和实际 SCAR 生成过程一致；可行性不证明完整内存、求解收敛或论文数值复现。

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

**退化语义（论文未覆盖，2026-08-25 实现决策）**：式 (35) 的推导假设 $`\hat\Sigma`$ 正定（$`q=\Delta\hat\Sigma^{-1}\Delta^\top>0`$）。真实数据上 $`\hat\Sigma=X_U^\top X_U/|U|^2`$ 因样本噪声常带小的负特征值，使 $`q<0`$ 或 Taylor 近似在 $`\|\mu-\hat\mu\|`$ 大时失真，两类都会让一次 $`\mu`$ 更新**降低**对偶目标，把 ACS 推入 period-2 极限环（$`\mu`$ 在边界点与 $`\hat\mu`$ 间交替，$`\mathrm{rel\_mu}`$ 恒大、永不收敛）。实现的单调回滚（§6）：仅当下一轮对偶目标不减时接受 $`\mu`$ 更新，否则回滚到上一轮 QP 求解点并冻结 $`\mu`$，ACS 收敛于所见最佳对偶值。

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

### 2026-11-02 运行适用性与恢复

论文第6页式(23)–(24)保留等式与盒约束，见
[作者公开稿](https://gcatnjust.github.io/ChenGong/paper/gong_pami2019.pdf)。
当前实现的等式左端可达范围为 `[-U/n, P/n + U/(2n)]`，
右端为 `-U/[2n(1-2*pi*h)]`。若右端超界，增加迭代或更换 GPU 不能得到可行解。
例如当前配方 `pi=0.5, h=0.9, P/n=0.05` 时，右端 `-4.75` 超出 `[-0.95,0.525]`。
论文实验的 `h∈{0.2,0.3,0.4}` 对应 `c∈{0.6,0.7,0.8}`（`h=1-c`）；本 Survey 的
`c=0.1/0.3/0.5` 对应 `h=0.9/0.7/0.5`，只有 `c=0.5` 在边界附近可行，低 c 属于论文未覆盖的
外推区。Survey 的固定数量 SCAR 用 `c_realized` 计算 `h`，与论文 i.i.d. Bernoulli 翻转有
有限样本差异；判定按实际计数执行。
本轮逐项预检实际 SCAR 生成计数，按标记率分组恢复；不再用一个 `c` 的不可行性
推断其他 `c` 也不可行。**未改论文/实现的目标、约束或冻结配方**，不做裁剪、改先验、
抽样或线性替代。预检与既有 Phase-I/adapter 约束对照测试支持工程判定，不支持论文数值复现。
对仍不可行的90项保留明确阻断；要改变其可行域需另行批准并版本化，不自动放宽。

> 见 ADR-0014 #5（RBF 采用论文原生参数 $`\sigma`$，勿直接传 `sklearn.metrics.pairwise.rbf_kernel` 的 $`\gamma=1/(2\sigma^2)`$；默认 $`\sigma=1/\sqrt{d}`$，对应 $`\gamma=d/2`$）

> 见 ADR-0014 #6（registry `kldce` 曾被误注册为 `centroid_pu` 别名导致静默解析到线性 LDCE，2026-08-05 已修复；`KLDCEClassifier` 构造要求 `flip_probability`，auto 模式无法自动实例化，需显式传实例）

- **QP oracle 版不是论文 Algorithm 1 的逐行实现**：当前实现先固定 `μ` 解联合 QP 再更新 `μ`（论文是先更新 `μ` 再 SMO 更新 `α/γ`）。两者均为 ACS 的合理块坐标顺序变化，但不可宣称逐行复现。
- **质心缩放基准始终为原始 $`\hat\Sigma`$**：计算 $`q=u^\top\hat\Sigma u`$（非 $`\Delta u`$），更新后 $`(\mu-\hat\mu)^\top\hat\Sigma(\mu-\hat\mu)=b`$ 严格满足（非仅事后验证）；约束检查不使用 ridge 后的矩阵，`ridge>0` 时标记 `centroid_solver="ridge_stabilized"`。
- **质心更新为 RBF 专用 $`\mu=0`$ Taylor 近似**：若支持 linear/poly 等核，必须重新推导 $`G(x,\mu)`$ 的 `\mu` 子问题并新增 oracle 对照。
- **数值校验**：显式检查 $`0<p\le1`$（违反时拒绝而非静默继续）与 $`|1-2ph|`$ 近零（等价于 $`|1-2k/n|`$ 近零，病态时拒绝）；`ridge` 默认为 0（论文原式），若 $`\hat\Sigma`$ 奇异，严格模式报错，变体模式加 `ridge>0`。
- **禁止照搬 `sklearn.svm.SVC` 内部逻辑**：其偏置恢复基于标准 C-SVC 对偶，KLDCE 的决策函数和 KKT 条件不同。
- **偏置恢复优先采用论文式 (37)–(40) 的四项平均**，并用 primal/KKT oracle 检查其可行性。
- **b₀ 恢复的类对称纪律（2026-08-27 实现决策，低先验全负修复）**：QP 版
  一次性 KKT 恢复中，自由 SV 的余量估计形成两个紧簇（正类 $`b\approx1-g`$、
  负类 $`b\approx-1-g`$，$`g=O(1/n)`$ 量级）。全体自由 SV 取中位数会坍缩到
  多数类簇——低先验时负类自由 SV 占绝对多数，$`b_0\approx-1`$，决策函数
  $`f(x)\approx b_0+g/(2\lambda)`$ 对一切点输出负值（全负预测；排序判别力
  AUC 完好，问题仅在 0 阈值）。修复：**类对称中位数**——正/负类自由 SV
  各取中位数再平均（与式 (37)–(40) 四项平均的类平衡意图一致）；仅一类有
  自由 SV 时回退 bounded-interval（L/U 界来自两类边界 SV 的 KKT 不等式），
  无交集时 $`b_0=0`$。回归锁定：`tests/estimators/risk/test_kldce_kkt.py`
  `TestBiasRecoveryClassSymmetry` 与 `test_kldce_math.py` 类对称数值用例。
- 对偶常数不要手工猜测：将论文公式逐项编码为可测试的 `dual_objective(z, m)`、`dual_gradient(z, m)` 与 `equality_coefficients`。
- **收敛判据（2026-08-24 修正）**：QP oracle 版曾以 SLSQP 解处目标梯度范数
  作为 `kkt_residual`；约束 QP 最优解处该梯度由乘子平衡、不为零，导致 ACS
  停止判据永不触发（出现"QP 最优但 300 轮不收敛"）。判据现为"目标/μ 相对
  变化 + eq/box 可行性"组合，严格 KKT（乘子恢复）经 `_true_kkt_residual`
  进入诊断与测试（见实现注记与工具箱 benchmark findings）。
- **质心单调回滚（2026-08-25）**：原生 SMO 版实测暴露 $`\hat\Sigma`$ 负特征值
  引发的 period-2 极限环（见 §3.2 退化语义）。实现采用"单调接受 + 回滚冻结"：
  ACS 每轮先解 QP，若下一轮对偶目标 < 上一轮，则回滚 $`\mu`$ 到本轮 QP 求解
  点并置 `mu_locked`；回滚后从回滚点的 $`z`$ 重解（非极限环上的 $`z`$），后续轮
  $`\mu`$ 冻结、`rel_mu=0`，收敛判据在所见最佳对偶处触发。诊断键
  `centroid_solver="locked"` 标记冻结轮；`mu_change=0` 判定（见测试
  `test_period2_limit_cycle_rolls_back_and_converges`）。

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
| Source status | `not_found`：未找到可确认的 KLDCE 作者实现；论文正文 + 在线补充附录为唯一权威 |
| 实现状态（2026-10-19 核查） | 原生广义 pair-SMO 已接入 `fit`，SLSQP oracle 保留作数值对照；内层对拼接的 `[α;γ]` 联合变量全局选 pair，**不是**附录 Algorithm 1 按 α/γ 两组分别选对的逐步复现；ACS 块更新顺序也与附录不同。实现和边界见来源/行为审计。质心回滚、`inner_tol=1e-6` 与 ACS 判据见 §6。 |
| Registry | `kldce` 独立注册（2026-08-05），详见 ADR-0014 #6 |
| 复现风险 | 正文未规定 KKT 违反度定义、可行区间展开式、缓存策略、容差与停止判据（见 §5）；当前 generalized pair-SMO 使用附录解析更新思想，但在拼接 `[α;γ]` 上全局选对，非 Algorithm 1 分组式逐步复现；SLSQP oracle 对照的是固定质心联合 QP，不能证明逐步行为相等。质心单调回滚（§3.2/§6）也是论文未覆盖的实现决策；精确 Gram 矩阵 $`O(n^2)`$ 内存 |

### SMO 内层性能探针与默认值定格（2026-08-25，修复后复测）

探针网格：small（n=400）/ mid（n=2000）× π∈{0.1,0.3,0.5} × seeds 0..4，`h=0.1`、`sigma="scale"`、`reg_strength=1.0`、`class_prior=π`（SCAR 简化构造 `y[:int(n·π)]=1`）。首轮（1e-8）与修复后（单调回滚 + 1e-6）均在同一网格复测。

| 档位 | 首轮结果（1e-8，2026-08-25） | 修复后（单调回滚 + `inner_tol=1e-6`） |
|---|---|---|
| small（n=400）15 cells | 15/15 收敛，`n_acs_iter ≤ 3`，`max(inner_n_iter)=496`，2.9–38.2s/cell | 15/15 收敛，1–2 轮，0.7–1.4s/cell（增量梯度 + 单调回滚提速） |
| mid（n=2000）15 cells | 14/15 收敛但 2–4 轮、46–138s，第 2 轮 inner 常撞 2000 上限；**seed=4（π=0.1）period-2 极限环：50 轮打满、1520s 不收敛** | **15/15 收敛，2 轮，13.6–54.8s/cell**，`inner=[337–866, 0]`（回滚冻结后第二轮 inner=0，μ 冻结） |

**默认值定格结论（修复后）**：`max_inner_iter=2000` / `inner_tol=1e-6`。依据：`inner_tol=1e-6` 与 ACS `tol=1e-6`、oracle 一致性验收（dual 差 ≤1e-6）对齐；1e-6→1e-8 是亚线性逼近尾巴（O(n) 对更新），无消费方需要多余精度（trace：viol 1e-3→1e-5 约 800 对，1e-5→1e-8 再 ~1100 对，即旧默认在 mid 档必然撞 2000 上限）。质心单调回滚（§3.2）消除 period-2 极限环，全网格 30/30 收敛、总耗时 415.7s（原 >5000s）、最慢 cell 54.8s（原 1520s）。

**mid 档边界（2026-08-25 撤销）**：修复前 KLDCE 从 SCAR 主网格 mid 档（n=2000）排除（首轮探针超预算）；修复后 mid 档 15/15 收敛、最慢 54.8s，**mid 档可运行**，不再需要排除。

### 参考资料

1. **Gong et al., TPAMI 2021 + Appendix** — KLDCE 的唯一原始数学与 SMO 规格；附录已取得，应与正文成对使用。[IEEE 记录](https://ieeexplore.ieee.org/document/8839365/)
2. **Platt (1998), SMO** — 二变量 QP 解析更新、KKT 与启发式选择的原始权威来源。[Microsoft Research PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/1998/04/sequential-minimal-optimization.pdf)
3. **Chang & Lin, LIBSVM** — 可生产实现的 SMO、核缓存、收敛与数值细节。只复用框架，不复用其标准 C-SVC 对偶公式。[官方论文](https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf)；[官方实现](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
4. **Gao et al. (AAAI 2016), LICS** — LDCE 质心平滑的直接来源，给出了 MoM、椭球约束与交替优化的完整线性伪代码。[AAAI 论文](https://cdn.aaai.org/ojs/10293/10293-13-13821-1-2-20201228.pdf)
5. **Shi et al. (IJCAI 2018)** — LDCE 前身版本，可核对线性推导；它不含 KLDCE，不能替代补充材料。[IJCAI 论文](https://www.ijcai.org/proceedings/2018/0373.pdf)

来源区分（2026-10-19 更正）：作者页上的 `CEGE_PAMI20.rar` 是 CEGE 会议早期版本，历史上用于说明线性 LDCE 家族的来源差异（见 [`LDCE.md`](LDCE.md)），不是 KLDCE 的官方实现。作者维护的[出版物目录](https://gcatnjust.github.io/ChenGong/publications.html)对 TPAMI 2019 论文只列 paper、未列 code；核查细节见来源与行为复核。KLDCE registry/ledger 不再将该包登记为其 `upstream_url`；这只表示本次未找到可确认的 KLDCE 作者代码，不推断代码在其他位置不存在。
