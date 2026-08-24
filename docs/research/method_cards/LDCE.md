# Method Card: LDCE PU Learning

## 1. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Loss Decomposition and Centroid Estimation for Positive and Unlabeled Learning |
| Authors | Chen Gong, Hong Shi, Tongliang Liu, Chuang Zhang, Jian Yang, Dacheng Tao |
| Venue | IEEE TPAMI |
| Year | 2021（online 2019） |
| Setting | censoring PU，单一 i.i.d. 训练样本（`Scenario.SINGLE_TRAINING_SET`） |
| Assumption | SCAR（常数翻转率 `h`）（`Assumption.SCAR`） |
| Requires class prior | `False`（由 `h` 与观测正例比例估计） |
| Requires propensity | `True`（翻转率 `h`；可调参或外部估计） |
| Requires negative samples | `False` |
| GPU required | `False` |

### Assumptions

令真实标签 $`Y\in\{-1,+1\}`$，观测/污染标签为 $`\tilde Y`$。标注正例集合 $`S_P`$ 干净；其余无标签样本被统一写成观测负例集合 $`\tilde S_N`$：

```math
P(\tilde Y=-1\mid Y=+1)=h,\qquad
P(\tilde Y=+1\mid Y=-1)=0.
```

这要求遗漏机制对正例是同质的常数 `h`（SCAR 型 censoring），并且训练样本先从总体随机抽取、再发生标注遗漏。

---

## 2. 问题设定与符号

censoring PU 设定：一个 i.i.d. 总样本中，正例以常数概率 `h` 被翻转为观测负类，负例绝不被观测为正例；观测到的干净正例构成 $`S_P`$，其余样本统一视为污染负集 $`\tilde S_N`$。目标是在未知真实标签下，通过 hinge loss 的偶/奇分解与真实无标签质心的无偏估计构造鲁棒经验风险。

| 论文符号 | 含义 |
|---|---|
| $`S=S_P\cup S_U`$ | 原始 PU 样本，总数 `n` |
| $`k`$ | 标注正例数 |
| $`\tilde S_N`$ | 将 `S_U` 全标为 `-1` 的污染负集 |
| $`Y,\tilde Y`$ | 真实、观测标签 |
| $`h`$ | 正例 `+1→-1` 翻转率 |
| $`p=P(Y=+1)`$ | 真实正类先验 |
| $`h_w(x)=w^\top x`$ | 线性判别函数 |
| $`m(S_U)`$ | 真实无标签集的 $`YX`$ 质心 |
| $`\hat m(\tilde S_N)`$ | MoM 初始质心 |
| $`\hat S`$ | 式 (10) 的质心经验协方差 |
| $`\lambda,b,g`$ | L2 正则、椭球半径、MoM 分组数 |

---

## 3. 核心公式

### 3.1 类先验与可识别条件

观测正例比例近似为 $`k/n=P(\tilde Y=+1)`$。在单一样本 censoring 假设下：

```math
p=P(Y=+1)=\frac{P(\tilde Y=+1)}{1-h}
\approx \frac{k}{n(1-h)}.
```

实现必须验证 $`0<h<1`$、$`0<p\le1`$；否则数据设定或 `h` 不可用。

### 3.2 损失分解

使用 hinge loss $`\ell(z)=[1-z]_+`$，其上界可写成：

```math
\ell(z)\le \frac12\big([1-z]_+ + [1+z]_+\big)+\frac12(1-z).
```

第一项是关于 $`z`$ 的偶函数，不受标签翻转影响；标签噪声只进入线性项，从而将未知标签风险转化为对 $`m(S_U)=|S_U|^{-1}\sum_{i\in U}y_ix_i`$ 的估计。

### 3.3 污染质心与协方差

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

### 3.4 LDCE 优化目标

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

固定 `m` 时，用次梯度/梯度法最小化 `w` 子问题。

---

## 4. 算法概要

### 4.1 线性 LDCE

1. 校验 `y_pu`，将标注正例编码为 `+1`，无标签编码为观测 `-1`；计算 `n,k,p`。
2. 将 `\tilde S_N` 随机近等分为 `g` 组；求各组均值，选取到其余组均值距离中位数最小者，得到 MoM 质心 `\hat m`。
3. 按式 (10) 得 `\hat S`，加 ridge，初始化 `w`。
4. 交替执行闭式 `m` 更新和 `w` 的凸优化，直到目标相对变化/参数变化满足容差或达到 `max_iter`。
5. 输出 $`\mathrm{sign}(w^\top x)`$；同时暴露先验、`h`、质心及收敛诊断。

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

### 4.2 KLDCE（核化版）

核化版 KLDCE 详见 [`KLDCE.md`](KLDCE.md) — ACS 外循环 + QP oracle + RBF kernel。

---

## 5. 论文边界

- 论文只适用于 **censoring PU**：一个 i.i.d. 总样本中，正例以常数概率被观察，负例绝不被观察为正例。独立抽取的 case-control `P/U` 数据不满足其 `p=k/[n(1-h)]` 先验公式。
- `h` 不是无关紧要的调参项：它同时决定类别先验、无偏质心修正和目标函数系数。错设会系统性移动分类边界；应保存实际使用值及其来源。
- 论文所谓"unbiased"针对真实无标签集质心的估计；实际目标还使用 hinge loss 的上界，以及有限样本的 MoM/椭球约束，不能表述为无条件的真实风险精确估计。
- 若 `1-2ph` 接近 0，质心项会病态；由于 `p=k/[n(1-h)]`，它等价于 `1-2k/n`。需在拟合前检查并拒绝/警告近奇异设定。
- 论文实验以二值 `{-1,+1}` 标签和线性/核判别函数为前提；不要把概率当作原生输出。若 API 需要 `predict_proba`，应明确这是后处理校准，而非论文算法保证。
- **理论依据**：Theorem 1 给出质心缩放关系；Theorem 2 说明标签噪声增加协方差；Theorem 3 给出所用经验协方差。这三者共同解释 MoM + 椭球约束，而非可随意省略的预处理。
- **Remark 1**：hinge loss 本身不满足 linear-odd 性质；论文优化的是其紧上界。**Remark 2**：该上界与原 hinge loss 的最大差距为 1。
- 本文不提供 SCAR/SAR 诊断、`h` 的置信区间、自动可靠的 `h` 估计或概率校准。将这些作为附加能力时，应与论文算法本体分开标注。
- 若当前优先目标是稳定的通用 PU 风险估计，已有 uPU/nnPU 更直接；LDCE 的增量价值在于 one-sided label-noise + 质心不确定性建模，适合明确满足 censoring 机制且可获得可靠 `h` 的数据。
- KLDCE 的 ACS/SMO 细节论文正文未充分规定，实现前需从补充材料或可信实现核对工程参数定义。

---

## 6. 实现注记

> 见 ADR-0014 #7（`CEGE_PAMI20.rar` 实为 CEGE 会议早期版：双向噪音 + 无椭球约束 + 无 MoM + 简单梯度下降，非 PAMI 终稿 LDCE，不可作实现参考；含 rar 5 文件审计与代码-论文差异大表）

- **工程化替换**（对应 Algorithm 2 步骤 4a）：`Ŝ⁻¹` 应实现为解线性方程 `(Ŝ + ridge·I)v=w`，不裸求逆；当 $`w^\top v`$ 近零时令 `m=m̂` 或停止并报告退化，不能直接除零。
- **默认 `max_iter`=10000**（偏离论文实现的 100 轮上限）：100 轮交替在普通规模的 censoring PU 数据上常不收敛（工具箱 SCAR 基准确认）；放宽上限只延伸收敛预算、不改变解语义（可实现单元结果与 100 轮版一致），故源码默认采用 10000。

---

## 7. 论文实验参考

### 7.1 可复现的基准协议

| 项目 | 论文设置 |
|---|---|
| 数据切分 | 5 折：每轮 80% 训练、20% 测试；所有方法共享同一切分和同一 PU 隐藏结果 |
| PU 构造 | 训练集中所有原负例进入 `U`；随机将 $`h\in\{0.2,0.3,0.4\}`$ 比例的原正例移入 `U` |
| 预处理 | 特征归一化到 $`[-1,1]`$ |
| 重复与检验 | 5 次试验的平均测试准确率；配对 t 检验，$`\alpha=0.05`$ |
| 对照方法 | WSVM、uPU、nnPU、RP，以及 LDCE、KLDCE |
| LDCE/KLDCE 搜索 | $`\lambda\in\{2^{-4},\ldots,2^4\}`$；$`b\in\{0.1,0.2,\ldots,0.9\}`$；KLDCE 核带宽 $`s\in\{2^0,2^1,2^2,2^3\}`$ |

注意：论文把 `h` 当作已知实验控制量；真实项目中应区分"以真值构造的 benchmark"与"以估计 $`\hat h`$ 训练的现实 benchmark"。

### 7.2 论文数据集

| 类别 | 数据集 / 规模 | 任务转换 |
|---|---|---|
| UCI | Vote (435×16)、Balance (625×4)、Breast (683×10)、Australian (690×14)、Banknote (1372×4)、Mushroom (8124×112)、PhishingWebsites (11055×30)、Connect-4 (67557×42；不同预处理口径见 benchmarks/assigned_methods/configs/pusb_table2_datasets.json)、Skin (245057×3) | 二分类；Connect-4 的第一类为正，其余为负 |
| USPS | 9298 个 16×16 图像，256 维；"0"为正（1553 正、7745 负） | 手写数字 PU |
| HockeyFight | 1000 视频（500 fight / 500 non-fight），100 维 BoW 特征 | 暴力行为 PU |
| NBA | 1340 名新秀，22 属性 | 生涯超过 5 年为正 |

### 7.3 关键消融结论

- **椭球约束消融**：论文在 USPS、HockeyFight、NBA 的每个 `h` 下均比较了有/无约束版本；去掉约束时 LDCE/KLDCE 准确率显著下降。这是实现验收的必要消融，而不是可选优化。
- **参数敏感性**：在真实数据和 $`h=0.2\sim0.4`$ 下，论文考察 `b=0.1\sim0.9` 与 $`\lambda=2^{-4}\sim2^4`$；结论是小幅参数变化通常不致严重恶化，但不表示可以跳过交叉验证。
- **翻转率失配**：真实 $`h=0.3`$ 时，用 $`\hat h\in\{0.6h,0.8h,h,1.2h,1.4h\}`$ 评估；所有方法都会受影响，轻度偏差未造成灾难性下降。
- **规模边界**：论文明确指出 KLDCE 的核计算较慢。`Skin`（约 245k 样本）应优先用于线性 LDCE 的可扩展性测试；未采用近似核时不应承诺 KLDCE 可在该规模运行。

---

## 8. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `official_related` |
| Official code | `https://gcatnjust.github.io/ChenGong/code/CEGE_PAMI20.rar` |
| License | `needs_review` |
| Integration basis | clean-room（方法卡先行），官方源码仅作算法参考 |

- `.rar` 压缩包无 license 声明，仅作算法参考，不直接复用代码；代码对应的是 **CEGE 会议早期版本**，并非 PAMI 终稿的 LDCE 算法（详见 ADR-0014 #7），论文 Algorithm 1（MoM 质心）和 Algorithm 2（交替优化）均需从零实现。
- 线性 LDCE 已实现（NATIVE）；核化版 KLDCE 见 [`KLDCE.md`](KLDCE.md)。
