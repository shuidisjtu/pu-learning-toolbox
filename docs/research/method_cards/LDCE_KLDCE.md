# Method Card: LDCE / KLDCE PU Learning

## 1. 待办与注意

### 1.1 待办

- 实现线性 **LDCE**：以无标签集为观测负类（含假负例），通过 hinge-loss 分解和质心估计构造鲁棒经验风险。
- 实现 **KLDCE** 前先确认核化优化所需的 ACS + SMO 求解器设计；论文未给出可直接复用的工程代码，不能把线性 LDCE 的梯度下降直接套到核模型。
- `h`（真实正例被翻为观测负例的概率）和椭球半径 `b` 都应作为显式超参数；`b` 用交叉验证选择。`h` 可交叉验证或由外部类先验估计器提供。
- 用广义 median-of-means（MoM）初始化观测负集质心，并按式 (10) 计算其经验协方差；协方差求逆必须使用带正则化的稳定求解，不能显式裸求逆。
- 在交替优化中同时更新参数 `w` 与受椭球约束的真实无标签质心 `m`；记录收敛轮数、目标值和数值失败原因。
- **[项目现状]** Phase 1 全部完成（Elkan-Noto、uPU、nnPU、ReCPE、PNU、LDCE），KLDCE 待实现（需先确认 ACS/SMO 求解器设计）。
- **[Registry 已修正]** `builtin_methods.py` 中 `scenario=SINGLE_TRAINING_SET`、`assumption=[SCAR]`、`implementation_status=NATIVE`。

### 1.2 注意

- 论文只适用于 **censoring PU**：一个 i.i.d. 总样本中，正例以常数概率被观察，负例绝不被观察为正例。独立抽取的 case-control `P/U` 数据不满足其 `p=k/[n(1-h)]` 先验公式。
- `h` 不是无关紧要的调参项：它同时决定类别先验、无偏质心修正和目标函数系数。错设会系统性移动分类边界；应保存实际使用值及其来源。
- 论文所谓"unbiased"针对真实无标签集质心的估计；实际目标还使用 hinge loss 的上界，以及有限样本的 MoM/椭球约束，不能表述为无条件的真实风险精确估计。
- 若 `1-2ph` 接近 0，质心项会病态；由于 `p=k/[n(1-h)]`，它等价于 `1-2k/n`。需在拟合前检查并拒绝/警告近奇异设定。
- 论文实验以二值 `{-1,+1}` 标签和线性/核判别函数为前提；不要把概率当作原生输出。若 Toolbox API 需要 `predict_proba`，应明确这是后处理校准，而非论文算法保证。

---

## 2. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Loss Decomposition and Centroid Estimation for Positive and Unlabeled Learning |
| Authors | Chen Gong, Hong Shi, Tongliang Liu, Chuang Zhang, Jian Yang, Dacheng Tao |
| Venue | IEEE TPAMI |
| Year | 2021（online 2019） |
| Family | `risk_estimation` |
| Setting | censoring PU，单一 i.i.d. 训练样本（`Scenario.SINGLE_TRAINING_SET`） |
| Assumption | SCAR（常数翻转率 `h`）（`Assumption.SCAR`） |
| Requires class prior | `False`（由 `h` 与观测正例比例估计） |
| Requires propensity | `True`（翻转率 `h`；可调参或外部估计） |
| Requires negative samples | `False` |
| GPU required | `False` |

> **Registry 状态**：已修正为 `scenario=[SINGLE_TRAINING_SET]`、`assumption=[SCAR]`、`implementation_status=NATIVE`。

### Assumptions

令真实标签 $`Y\in\{-1,+1\}`$，观测/污染标签为 $`\tilde Y`$。标注正例集合 $`S_P`$ 干净；其余无标签样本被统一写成观测负例集合 $`\tilde S_N`$：

```math
P(\tilde Y=-1\mid Y=+1)=h,\qquad
P(\tilde Y=+1\mid Y=-1)=0.
```

这要求遗漏机制对正例是同质的常数 `h`（SCAR 型 censoring），并且训练样本先从总体随机抽取、再发生标注遗漏。

---

## 3. 符号与记号

| 论文符号 | 含义 | 开发侧对应（建议） |
|---|---|---|
| $`S=S_P\cup S_U`$ | 原始 PU 样本，总数 `n` | `X`, `y_pu` |
| $`k`$ | 标注正例数 | `n_labeled` |
| $`\tilde S_N`$ | 将 `S_U` 全标为 `-1` 的污染负集 | `corrupted_negative_set` |
| $`Y,\tilde Y`$ | 真实、观测标签 | `y_true`（不可见）、`y_corrupted` |
| $`h`$ | 正例 `+1→-1` 翻转率 | `flip_probability` |
| $`p=P(Y=+1)`$ | 真实正类先验 | `class_prior_` |
| $`h_w(x)=w^\top x`$ | 线性判别函数 | `decision_function` |
| $`m(S_U)`$ | 真实无标签集的 $`YX`$ 质心 | `true_unlabeled_centroid_` |
| $`\hat m(\tilde S_N)`$ | MoM 初始质心 | `corrupted_centroid_` |
| $`\hat S`$ | 式 (10) 的质心经验协方差 | `centroid_covariance_` |
| $`\lambda,b,g`$ | L2 正则、椭球半径、MoM 分组数 | `reg_strength`, `centroid_radius`, `mom_groups` |

---

## 4. 核心公式

### 4.1 类先验与可识别条件

观测正例比例近似为 $`k/n=P(\tilde Y=+1)`$。在单一样本 censoring 假设下：

```math
p=P(Y=+1)=\frac{P(\tilde Y=+1)}{1-h}
\approx \frac{k}{n(1-h)}.
```

实现必须验证 $`0<h<1`$、$`0<p\le1`$；否则数据设定或 `h` 不可用。

### 4.2 损失分解

使用 hinge loss $`\ell(z)=[1-z]_+`$，其上界可写成：

```math
\ell(z)\le \frac12\big([1-z]_+ + [1+z]_+\big)+\frac12(1-z).
```

第一项是关于 $`z`$ 的偶函数，不受标签翻转影响；标签噪声只进入线性项，从而将未知标签风险转化为对 $`m(S_U)=|S_U|^{-1}\sum_{i\in U}y_ix_i`$ 的估计。

### 4.3 污染质心与协方差

对观测负集（其标签均为 $`\tilde y_i=-1`$）：

```math
m(\tilde S_N)=\frac{1}{n-k}\sum_{i=k+1}^{n}\tilde y_i x_i,
\qquad
\mathbb E[m(\tilde S_N)]=(1-2ph)m(S_U).
```

因此 $`m(\tilde S_N)/(1-2ph)`$ 是真实无标签质心的无偏估计。论文用 MoM 得到更稳健的 $`\hat m(\tilde S_N)`$，再用：

```math
\hat S=\frac{\sum_{i\in U}x_i^\top x_i}{(n-k)^2}
-\frac{(\sum_{i\in U}x_i\tilde y_i)^\top(\sum_{i\in U}x_i\tilde y_i)}{(n-k)^2}.
```

### 4.4 LDCE 优化目标

令 $`\phi(z)=[1-z]_+ + [1+z]_+`$，$`c=-(n-k)/(2n)`$。论文的线性模型为：

```math
\begin{aligned}
\min_{w,m}\quad &\frac1n\sum_{i=1}^{k}\ell(\tilde y_i w^\top x_i)
+\frac1{2n}\sum_{i=k+1}^{n}\phi(\tilde y_i w^\top x_i)
+\frac{c}{1-2ph}w^\top m+\lambda\|w\|_2^2\\
\text{s.t.}\quad &(m-\hat m)^\top\hat S(m-\hat m)\le b.
\end{aligned}
```

固定 `w` 时，约束子问题闭式更新：

```math
m\leftarrow\hat m+\hat S^{-1}w\sqrt{\frac{b}{w^\top\hat S^{-1}w}}.
```

固定 `m` 时，用次梯度/梯度法最小化 `w` 子问题。实际实现须给 `\hat S` 加 ridge 后用线性方程求解。

---

## 5. 算法概要

### 5.1 线性 LDCE

1. 校验 `y_pu`，将标注正例编码为 `+1`，无标签编码为观测 `-1`；计算 `n,k,p`。
2. 将 `\tilde S_N` 随机近等分为 `g` 组；求各组均值，选取到其余组均值距离中位数最小者，得到 MoM 质心 `\hat m`。
3. 按式 (10) 得 `\hat S`，加 ridge，初始化 `w`。
4. 交替执行闭式 `m` 更新和 `w` 的凸优化，直到目标相对变化/参数变化满足容差或达到 `max_iter`。
5. 输出 $`\operatorname{sign}(w^\top x)`$；同时暴露先验、`h`、质心及收敛诊断。

#### 论文 Algorithm 1：污染负集 MoM 质心

```text
输入：污染负集 Ṡ_N，分组数 g ≥ 1
1. 将 Ṡ_N 随机划分为 g 个样本数尽量相等的子集 Ṡ_N[1], ..., Ṡ_N[g]
2. 对每组 i 计算均值 m_i = mean(Ṡ_N[i])
3. 对每组 i 计算 r_i = median_j ||m_i - m_j||₂
4. 取 i* = argmin_i r_i
输出：m̂ = m_i*
```

#### 论文 Algorithm 2：LDCE 交替优化

```text
输入：污染样本 Ṡ={(x_i, ỹ_i)}，翻转率 h，正则 λ，椭球半径 b
1. 调用 Algorithm 1，得到初始质心 m̂
2. 按式 (10) 计算质心协方差 Ŝ
3. 初始化 w，令 t = 0
4. 重复直至收敛：
   a. 固定 w，更新质心：
      m ← m̂ + Ŝ⁻¹w · sqrt(b / (wᵀŜ⁻¹w))
   b. 固定 m，通过梯度下降求解：
      w ← argmin_w [
            (1/n) Σ_{i=1}^k ℓ(ỹ_i wᵀx_i)
          + (1/(2n)) Σ_{i=k+1}^n φ(ỹ_i wᵀx_i)
          + c/(1-2ph) · wᵀm + λ||w||²
      ]
      其中 c=-(n-k)/(2n)，φ(z)=[1-z]₊+[1+z]₊
   c. t ← t + 1
5. 返回收敛后的 w
```

**工程化替换**：上述 `Ŝ⁻¹` 应实现为解线性方程 `(Ŝ + ridge·I)v=w`；当 $`w^\top v`$ 近零时令 `m=m̂` 或停止并报告退化，不能直接除零。

### 5.2 KLDCE

以核展开 $`f(x)=\sum_i\alpha_i K(x_i,x)+b_0`$ 替代线性函数。论文将带松弛变量的核化问题分解为两个凸子问题，以 **ACS** 交替更新，并用 **SMO** 解核 SVM 型子问题。

#### KLDCE ACS/SMO 求解流程

> 论文正文给出了 KLDCE 的原始问题、对偶问题和"ACS + SMO"求解说明，但没有像 LDCE Algorithm 2 那样列出独立编号的完整 KLDCE 伪代码；以下是依据式 (23)–(25) 整理的实现流程，不能视为论文逐字伪代码。

```text
输入：污染样本 Ṡ，核函数 K，h，λ，b
1. 用 Algorithm 1 初始化 m̂，并由式 (10) 得 Ŝ；初始化可行 m←m̂。
2. 重复直至 ACS 收敛：
   a. 固定 m：构造式 (24) 的二次规划对偶。
      使用 SMO 迭代更新 α、γ，满足：
      - 盒约束：0 ≤ α_i ≤ C₁，0 ≤ γ_i ≤ C₂
      - 等式约束：C₁Σ_i α_i ỹ_i + C₂Σ_{i∈U}γ_i ỹ_i = 0
      得到当前 α、γ 和偏置 b₀。
   b. 固定 α、γ（等价于固定当前判别函数）：
      按与 LDCE 相同的椭球约束子问题更新 m，
      并验证 (m-m̂)ᵀŜ(m-m̂) ≤ b。
   c. 计算原始/对偶目标及约束残差；若相对变化小于 tol 则停止。
3. 用式 (25) 组装决策函数：
   f(x) = (1/(2λ))Σ_i α_i ỹ_i K(x,x_i)
        - (1/(2λ))Σ_{i∈U} γ_i ỹ_i K(x,x_i)
        - (C/(2λ))K(x,m*) + b₀。
输出：α、γ、m*、b₀ 与 f(x)
```

实现前须从论文补充材料或可信实现核对式 (24) 的 `C/C₁/C₂` 定义、SMO 工作集选择、偏置恢复和停止准则；正文未充分规定这些工程细节。

**开发边界**：KLDCE 的时间/内存均受 Gram 矩阵约束，适合中小样本；大样本需明确不支持、使用近似核，或另行实现预算策略。论文未提供将其转成通用 sklearn 核分类器的等价接口。

---

## 6. 源码状态

| 字段 | 内容 |
|---|---|
| Source status | `official_related` |
| Official code | `https://gcatnjust.github.io/ChenGong/code/CEGE_PAMI20.rar` |
| License | `needs_review` |
| Registry status | `implementation_status=API_ONLY`, `backend=NUMPY`, `source_status=OFFICIAL_RELATED` |
| Integration basis | clean-room（方法卡先行），官方源码仅作算法参考 |

### 6.1 源码内容（已审计 2026-07-21）

`CEGE_PAMI20.rar` 解压后共 5 个文件：

| 文件 | 说明 |
|---|---|
| `main.m` | 主脚本：10-fold CV，GermanCredit 数据集 |
| `SemiLinearTraining.m` | 核心训练函数（151 行 MATLAB） |
| `GermanCredit.mat` | 数据集 |
| `Idx0.05.mat` / `Idx0.1.mat` | 两种 labeled ratio（5%/10%）的 CV 索引 |

### 6.2 代码与论文差异（关键）

实际代码实现的是 **CEGE 早期会议版本**，与 PAMI 终稿 LDCE 存在以下重大差异：

| 维度 | 论文 LDCE | 实际代码 |
|---|---|---|
| **噪音模型** | 单向 censoring：P(Ỹ=-1\|Y=+1)=h, P(Ỹ=+1\|Y=-1)=0 | **双向噪音**：同时计算 `Yita_N` 和 `Yita_P`，U 分两次被当作 noisy N 和 noisy P |
| **质心估计** | MoM 稳健初始化（Algorithm 1）→ 单一无偏质心校正 | 两个方向各算一个无偏质心估计，再以 `Beta` 加权组合；无 MoM |
| **椭球约束** | 式 (13) 的 `(m-m̂)ᵀŜ(m-m̂) ≤ b` 约束 + 闭式更新 | **不存在**；质心仅为梯度中的常数项（`Mu_S`），不参与优化 |
| **优化方式** | 交替优化：固定 w→闭式更新 m，固定 m→梯度更新 w（Algorithm 2） | 简单梯度下降（800 iter，自适应步长）；无交替结构 |
| **损失函数** | hinge loss 偶/奇分解（§4.2 式 (11)-(12)） | 支持 squared / hinge / squared-hinge 三种 loss，但 hinge 梯度未做偶/奇分离 |
| **正则化** | L2 正则 ‖w‖² | `J'*J*w` 形式的正则（`J` 为去掉 bias 行的单位阵） |

关键代码片段——同时计算两个方向的质心：

```matlab
% treat U as noisy N
Mu_tilte_NoisyN = (sum(labeled.*y) - sum(unlabeled)) / n;
% treat U as noisy P
Mu_tilte_NoisyP = (sum(labeled.*y) + sum(unlabeled)) / n;
% 加权组合两个无偏估计
Mu_S = Beta*Tau_P*Mu_tilte_NoisyN + (1-Beta)*Tau_N*Mu_tilte_NoisyP;
```

核心优化——简单梯度下降，无交替、无椭球约束：

```matlab
w = -1+2*rand(DataDim,1);
for Iter = 1 : MaxIter
    Grad = WeakGrad + Gamma_1*StrongGrad + 2*Gamma_2*(J'*J)*w;
    w = w - StepSize*Grad;
end
```

### 6.3 集成边界

- `CEGE_PAMI20.rar` 为 `.rar` 压缩包，无明确 license 声明，仅作算法参考，不直接复用代码。
- clean-room 实现以论文公式为数学权威；URL 中 "CEGE" 为论文方法的另一简称（Centroid Estimation with Generalized Eigenvalue），但压缩包内代码对应的是 **会议早期版本**（双向噪音 + 无约束 GD），**并非 PAMI 终稿的 LDCE 算法**（单向 censoring + MoM + 椭球约束交替优化）。
- 这份代码不可直接用作 LDCE 实现参考；论文 Algorithm 1（MoM 质心）和 Algorithm 2（交替优化）均需从零实现。
- KLDCE 的 ACS/SMO 细节论文正文未充分规定，实现前需从补充材料或可信实现核对工程参数定义。
- 许可证：`.rar` 压缩包无 license 文件，默认只作参考，不直接再分发。

### 6.4 Registry 元数据

```python
# builtin_methods.py:164-183 — 已修正的注册状态
name = "centroid_pu"
aliases = ["ldce", "kldce", "centroid_estimation"]
family = AlgorithmFamily.RISK_ESTIMATION
scenario = (Scenario.SINGLE_TRAINING_SET,)       # ✅ 已修正
assumption = (Assumption.SCAR,)                   # ✅ 已修正
requires_class_prior = False
backend = Backend.NUMPY
maturity = Maturity.RESEARCH
implementation_status = ImplementationStatus.NATIVE  # ✅ 已实现
source_status = SourceStatus.OFFICIAL_RELATED
upstream_url = "https://gcatnjust.github.io/ChenGong/code/CEGE_PAMI20.rar"
license = "needs_review"
```

---

## 7. API 接口与项目落点

> 以下为基于 `BasePUClassifier` 契约（`core/base.py`）和 `docs/project_management/process_checklist.md` 的项目建议，非论文原文。

### 7.1 公共 API

| API / 决策点 | 约定 |
|---|---|
| `fit(X, y_pu, *, class_prior=None, sample_weight=None)` | 遵守 `BasePUClassifier` 公共契约；sklearn 风格入口。LDCE 不需要外部传入 `class_prior`（由 `h` 和 `k/n` 推导），传入时作为覆盖值使用。 |
| PU 训练输入 | `y_pu` 接受 `{+1,0}` / `{+1,-1}` / `{1,0}` / `{1,-1}`（base class 标准），通过 `normalize_pu_labels()` 转换为内部 `{+1,0}`，再在算法内部映射为 `{+1,-1}`。 |
| 稀疏支持 | `accept_sparse=False`（涉及协方差矩阵运算，稀疏无意义）。 |
| `predict(X)` | 返回项目标准 `{0, 1}` 标签（margin >= 0 为 1，否则 0）。 |
| `decision_function(X)` | 返回 $`w^\top x`$ margin，shape `(n_samples,)`。 |
| `predict_proba(X)` | 继承 base class 默认行为，抛出 `NotImplementedError`；论文不提供后验概率。 |
| `get_params()` / `set_params()` | 由 sklearn `BaseEstimator` 提供，覆盖构造函数中的公开超参数。 |
| `get_pu_metadata()` | 返回 LDCE 特有诊断：flip_probability、class_prior、centroid_radius、convergence 信息。 |

### 7.2 构造参数

```python
class LDCEClassifier(BasePUClassifier):
    def __init__(
        self,
        flip_probability,          # h；必填，或显式允许 prior_estimator 提供
        reg_strength=1.0,          # λ
        centroid_radius=1.0,       # b；CV 选择
        mom_groups=10,             # g
        covariance_ridge=1e-4,
        max_iter=100,
        tol=1e-6,
        random_state=None,
    ):
```

| 参数 | 类型 | 默认值 | 约束 |
|---|---|---|---|
| `flip_probability` | `float` | **必填** | `0 < h < 1` |
| `reg_strength` | `float` | `1.0` | `> 0` |
| `centroid_radius` | `float` | `1.0` | `> 0`；CV 选择 |
| `mom_groups` | `int` | `10` | `≥ 1`；`g=1` 退化为普通均值 |
| `covariance_ridge` | `float` | `1e-4` | `> 0` |
| `max_iter` | `int` | `100` | `≥ 1` |
| `tol` | `float` | `1e-6` | `> 0` |
| `random_state` | `int \| None` | `None` | MoM 分组随机种子 |

### 7.3 拟合属性

| 属性 | 类型 | 含义 |
|---|---|---|
| `coef_` | `np.ndarray (d,)` | 线性判别权重 w |
| `class_prior_` | `float` | 估计的正类先验 p = k/[n(1-h)] |
| `flip_probability_` | `float` | 实际使用的翻转率 h |
| `corrupted_centroid_` | `np.ndarray (d,)` | MoM 污染质心 m̂ |
| `true_unlabeled_centroid_` | `np.ndarray (d,)` | 优化后的真实无标签质心 m |
| `centroid_covariance_` | `np.ndarray (d,d)` | 质心经验协方差 Ŝ（含 ridge） |
| `n_labeled_` | `int` | 标注正例数 k |
| `n_unlabeled_` | `int` | 无标签样本数 n-k |
| `n_iter_` | `int` | 交替优化实际轮数 |
| `objective_history_` | `list[float]` | 每轮目标值 |
| `converged_` | `bool` | 是否收敛 |
| `classes_` | `np.ndarray` | `np.array([0, 1])` |
| `_is_fitted` | `bool` | 拟合状态 |

### 7.4 模块落点

| 模块 | 责任 | 状态 |
|---|---|---|
| `pu_toolbox/estimators/risk/ldce.py` | `LDCEClassifier` — 线性 LDCE 交替优化 + sklearn API | ✅ 已实现 (NATIVE) |
| `pu_toolbox/estimators/risk/kldce.py` | `KLDCEClassifier` — 核化 KLDCE（ACS + SMO） | 待设计 |
| `pu_toolbox/registry/builtin_methods.py` | `centroid_pu` 元数据，`implementation_status=NATIVE` | ✅ 已修正 scenario/assumption |

---

## 8. 测试参考

### 8.1 MATH tests

用固定小数组手工计算，逐项验证：

| 模块 | 必测行为 | 通过标准 |
|---|---|---|
| 先验计算 | $`p=k/[n(1-h)]`$ | 与手算一致；`p>1`、`h\notin(0,1)` 抛出参数错误 |
| MoM 质心 | 固定随机种子下分组、组均值与中心组选择可复现 | `g=1` 退化为普通均值；`g>|U|` 拒绝或按明确规则处理 |
| 协方差 | 式 (10) 的 shape、对称性和 ridge 后可解性 | 小型手算样本一致；奇异输入不产生 NaN/Inf |
| `m` 更新 | 椭球边界闭式解 | 更新后满足约束（允许数值容差）；`w=0` 有稳定退化处理 |
| `w` 子问题 | 固定 `m` 时目标下降 | 次梯度/优化器返回有限参数；目标与梯度维度正确 |

### 8.2 PROPERTY tests

- **质心无偏性（统计测试）**：从已知分布反复生成 censoring PU 数据，比较 $`m(\tilde S_N)/(1-2ph)`$ 与真实 $`m(S_U)`$ 的重复均值；样本量增大时偏差应趋近 0。
- **约束必要性回归**：同一固定合成数据上，对比启用/禁用椭球约束。禁用仅用于消融测试，不能成为默认实现路径；检查启用约束后目标与泛化性能不劣于无约束的合理范围。
- **`h` 敏感性**：真实 `h=0.3`，运行 $`\hat h\in\{0.6h,0.8h,h,1.2h,1.4h\}`$；验证轻度误差不会触发数值异常，并记录性能退化曲线。
- 标签转换：`1/0` PU 标签正确转为内部 `+1/-1`；`S_P` 维持干净正例，无标签不得进入正例分支。
- 交替优化：停止条件、`max_iter`、诊断属性正确记录；参数/目标收敛或报告未收敛，不静默失败。

### 8.3 CONTRACT tests

与 `tests/contract/test_classifier_api.py` 对齐：

- `fit` 返回 `self`
- `predict(X)` 输出 `{0, 1}`，shape `(n_samples,)`，dtype `int`
- `decision_function(X)` 返回一维 score，shape `(n_samples,)`
- `get_params()` / `set_params()` 可用
- Pipeline / clone 兼容
- 拟合后暴露 `class_prior_`、`flip_probability_`、`converged_` 等属性
- `predict_proba` 抛出 `NotImplementedError`

### 8.4 PAPER-like regression

- **线性可分合成数据**：复现论文图 2 的精神：先生成正负两类，再随机隐藏部分正例；LDCE 应优于"把 U 全当负类"的朴素线性分类器，且边界可视化/准确率可回归。
- **交叉实现对照**：在线性核、小样本、固定 `h` 下，KLDCE 的预测应与 LDCE 同方向高度一致；这只是 smoke test，不等价于证明 ACS/SMO 正确。
- 每个配置至少多随机种子重复，报告均值和标准差。

### 8.5 测试数据与指标

- 训练侧必须由完整二分类数据**先划分 train/test，再只在训练正例中按 `h` 随机隐藏**；测试标签始终保持真实标签，避免标签泄漏。
- 主指标采用论文使用的 test accuracy；Toolbox 同时应报告 `ROC-AUC`、`F1`、balanced accuracy，以应对 USPS 等类别不平衡数据。不要用未校准的 margin 伪装概率指标。
- 每个配置至少多随机种子重复，报告均值和标准差；论文使用五次试验，并以配对 t 检验（显著性 0.05）比较方法。

---

## 9. 论文实验参考

### 9.1 可复现的基准协议

| 项目 | 论文设置 |
|---|---|
| 数据切分 | 5 折：每轮 80% 训练、20% 测试；所有方法共享同一切分和同一 PU 隐藏结果 |
| PU 构造 | 训练集中所有原负例进入 `U`；随机将 $`h\in\{0.2,0.3,0.4\}`$ 比例的原正例移入 `U` |
| 预处理 | 特征归一化到 $`[-1,1]`$ |
| 重复与检验 | 5 次试验的平均测试准确率；配对 t 检验，$`\alpha=0.05`$ |
| 对照方法 | WSVM、uPU、nnPU、RP，以及 LDCE、KLDCE |
| LDCE/KLDCE 搜索 | $`\lambda\in\{2^{-4},\ldots,2^4\}`$；$`b\in\{0.1,0.2,\ldots,0.9\}`$；KLDCE 核带宽 $`s\in\{2^0,2^1,2^2,2^3\}`$ |

注意：论文把 `h` 当作已知实验控制量；真实项目中应区分"以真值构造的 benchmark"与"以估计 $`\hat h`$ 训练的现实 benchmark"。

### 9.2 论文数据集

| 类别 | 数据集 / 规模 | 任务转换 |
|---|---|---|
| UCI | Vote (435×16)、Balance (625×4)、Breast (683×10)、Australian (690×14)、Banknote (1372×4)、Mushroom (8124×112)、PhishingWebsites (11055×30)、Connect-4 (67557×42)、Skin (245057×3) | 二分类；Connect-4 的第一类为正，其余为负 |
| USPS | 9298 个 16×16 图像，256 维；"0"为正（1553 正、7745 负） | 手写数字 PU |
| HockeyFight | 1000 视频（500 fight / 500 non-fight），100 维 BoW 特征 | 暴力行为 PU |
| NBA | 1340 名新秀，22 属性 | 生涯超过 5 年为正 |

### 9.3 关键消融与验收结论

- **椭球约束消融**：论文在 USPS、HockeyFight、NBA 的每个 `h` 下均比较了有/无约束版本；去掉约束时 LDCE/KLDCE 准确率显著下降。这是实现验收的必要消融，而不是可选优化。
- **参数敏感性**：在真实数据和 $`h=0.2\sim0.4`$ 下，论文考察 `b=0.1\sim0.9` 与 $`\lambda=2^{-4}\sim2^4`$；结论是小幅参数变化通常不致严重恶化，但不表示可以跳过交叉验证。
- **翻转率失配**：真实 $`h=0.3` 时，用 $`\hat h\in\{0.6h,0.8h,h,1.2h,1.4h\}`$ 评估；所有方法都会受影响，轻度偏差未造成灾难性下降。应将此作为 `h` 输入接口的稳健性回归。
- **规模边界**：论文明确指出 KLDCE 的核计算较慢。`Skin`（约 245k 样本）应优先用于线性 LDCE 的可扩展性测试；未采用近似核时不应承诺 KLDCE 可在该规模运行。

---

## 10. 论文边界与决策

- Theorem 1 给出质心缩放关系；Theorem 2 说明标签噪声增加协方差；Theorem 3 给出所用经验协方差。这三者共同解释 MoM + 椭球约束，而非可随意省略的预处理。
- Remark 1：hinge loss 本身不满足 linear-odd 性质；论文优化的是其紧上界。Remark 2：该上界与原 hinge loss 的最大差距为 1。
- 本文不提供 SCAR/SAR 诊断、`h` 的置信区间、自动可靠的 `h` 估计或概率校准。将这些作为 Toolbox 附加能力时，应与论文算法本体分开标注。
- 若当前优先目标是稳定的通用 PU 风险估计，已有 uPU/nnPU 更直接；LDCE 的增量价值在于 one-sided label-noise + 质心不确定性建模，适合明确满足 censoring 机制且可获得可靠 `h` 的数据。

---

## 11. 实现验收清单

- [x] `flip_probability` 校验 `0 < h < 1` 并持久化为 `flip_probability_`
- [x] 类先验 `p = k/[n(1-h)]` 验证 `0 < p ≤ 1`
- [x] `1-2ph` 近零检测与拒绝/警告
- [x] MoM 质心在固定 `random_state` 下可复现
- [x] `g=1` 退化为普通均值
- [x] 协方差矩阵加 ridge 后用线性方程求解，不裸求逆
- [x] 交替优化记录收敛轮数（`n_iter_`）和目标值（`objective_history_`）
- [x] `converged_` 属性正确反映收敛状态
- [x] 标签通过 `normalize_pu_labels()` 转换
- [x] `predict` 返回 `{0, 1}`，不返回 `{-1, +1}`
- [x] `predict_proba` 继承 base class `NotImplementedError`
- [x] `decision_function` 返回一维 margin，shape `(n_samples,)`
- [x] Pipeline / clone / `get_params` / `set_params` 兼容
- [x] 所有项目路径、枚举和 registry 属性已与仓库实际代码对齐
