# Method Card: Self-PU

## 1. 待办与注意

### 1.1 待办

- [x] 确认 Self-PU 的数据协议：可靠正样本 `D_P`、边缘分布未标记样本 `D_U`、已知正类先验 `pi_p`，以及用于 loss reweighting 的干净验证集。
- [x] 梳理 self-paced PU：动态扩大 trusted set，同时允许已选样本因当前模型置信度下降而回到 U 集。
- [x] 梳理 self-calibrated loss：用 clean validation batch 对未信任样本的 CE/nnPU 两种损失进行元梯度加权。
- [x] 梳理 self-distillation：两种不同学习速度的 student 互相蒸馏，并分别与 EMA teacher 保持一致。
- [x] 记录论文 benchmark、模型、训练阶段和消融实验协议。
- [x] 将三阶段训练流程实现为项目统一的 PyTorch estimator。
- [x] 设计 `validation_data=(X_val, y_val)` 公共 API，并明确它与普通 PU validation data 的区别。
- [x] 实现动态 trusted-set manager、meta reweighting 和 student/teacher checkpoint 管理。
- [ ] 复现 MNIST、CIFAR-10 与 ADNI paper-like benchmark。

### 1.2 注意

- Self-PU 是深度 PU 训练框架，不是独立的 class-prior estimator；论文沿用 nnPU 的已知 `pi_p` 设定。
- self-calibrated reweighting 需要含真实正负标签的 clean validation batch。只有 P/U 标签而没有 clean validation labels 时，不能严格复现该模块。
- trusted set 的正负伪标签来自当前模型置信度，不能把被选样本当作永久正确标签；论文明确使用 in-and-out 机制重新审查历史选择。
- 动态采样、soft labels、meta weights、student consistency、EMA teacher 是相互关联的组件，不能把 Self-PU 简化成单次 pseudo-labeling。
- 论文的最终测试模型是验证集表现更好的 teacher；不能默认使用最后一个 student 或两个 teacher 的简单平均。
- 当前仓库已提供 Self-PU clean-room 核心并注册为 `native`；官方图像 benchmark 仍未完成，不能把小型 smoke 结果写成论文复现。

## 2. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Self-PU: Self Boosted and Calibrated Positive-Unlabeled Training |
| Authors | Xuxi Chen, Wuyang Chen, Tianlong Chen, Ye Yuan, Chen Gong, Kewei Chen, Zhangyang Wang |
| Venue | ICML 2020, PMLR 119 |
| Pages | 1510–1519 |
| Year | 2020 |
| Family | `deep_pu` |
| Scenario | `case_control` |
| Requires class prior | `True`，论文沿用 `pi_p` 已知设定 |
| Requires propensity | `False` |
| Requires negative samples | 训练集不需要；clean validation batch 需要真实正负标签 |
| Backend | PyTorch |
| Official paper | [PMLR paper page](https://proceedings.mlr.press/v119/chen20b.html) |
| Official PDF | [Self-PU PDF](https://proceedings.mlr.press/v119/chen20b/chen20b.pdf) |
| Official code | [TAMU-VITA/Self-PU](https://github.com/TAMU-VITA/Self-PU) |

论文提出三个 “self” 模块：

1. **Self-paced learning**：逐步从 U 中选择高置信正/负样本，形成 trusted set；
2. **Self-calibrated loss reweighting**：用 clean validation loss 学习每个不可信样本的 CE/nnPU 权重；
3. **Self-supervised distillation**：让不同 pace 的 student 以及其 moving-average teacher 互相提供一致性监督。

## 3. 问题设定与假设

令 `Y in {+1,-1}`，正类先验为：

```math
\pi_p=P(Y=+1),
\qquad
\pi_n=1-\pi_p.
```

训练数据：

```math
D=D_P\cup D_U,
```

```math
D_P=\{x_i^p\}_{i=1}^{n_p},
\qquad x_i^p\sim p(x\mid Y=+1),
```

```math
D_U=\{x_i^u\}_{i=1}^{n_u},
\qquad x_i^u\sim p(x).
```

其中：

```math
p(x)=\pi_p p(x\mid Y=+1)+\pi_n p(x\mid Y=-1).
```

Self-PU 还使用一个 clean validation set：

```math
D_V=\{(x_j^v,y_j^v)\}_{j=1}^{m},
\qquad y_j^v\in\{+1,-1\},
```

该集合只用于 self-calibrated weight 的元更新和模型选择，不应与 U 伪标签混淆。

### 3.1 项目输入协议建议

| 数据 | 标签 | 用途 |
|---|---|---|
| `X_P` | 可靠正类 | 初始 P 监督、nnPU 风险 |
| `X_U` | 无标签 | self-paced 候选、nnPU 风险、蒸馏 |
| `X_val`, `y_val` | 真实 `+1/-1` | meta reweighting、early stopping、teacher 选择 |
| `class_prior` | `pi_p` | nnPU 风险分解 |

如果没有 `D_V`：

- 可以实现 self-paced + distillation 的消融版本；
- 不能声称完成论文的 self-calibrated loss；
- 不应使用测试集标签替代 `D_V`。

## 4. 符号与记号

| 论文符号 | 含义 | 项目侧对应 |
|---|---|---|
| `g(x)` | 实值分类器输出 | `student(x)` / `teacher(x)` logit |
| `f(g(x))` | logit 到概率的单调映射 | `sigmoid(logit)` |
| `D_P` | labeled positive set | `X[y_pu == 1]` |
| `D_U` | unlabeled marginal set | `X[y_pu == 0]` |
| `D_trust` | 当前可信伪标签集合 | `trusted_indices`, `trusted_labels` |
| `D-D_trust` | 尚未信任的样本 | U 中未进入 trusted set 的样本 |
| `L_CE` | hard/soft target cross entropy | trusted / untrusted CE |
| `L_nnPU` | sigmoid nnPU loss | 已有 `NonNegativePU` 风险组件的深度版本 |
| `L_SP` | self-paced hybrid loss | trusted CE + remaining nnPU |
| `w_i` | 第 i 个不可信样本的两维 loss 权重 | `w[i, 0:2]` |
| `gamma` | CE 权重总量约束因子 | `reweight_gamma` |
| `delta` | meta inner-step 学习率 | `meta_step_size` |
| `alpha` | hard sample mining 阈值 | `distillation_alpha` |
| `beta` | EMA teacher 衰减系数 | `ema_decay` |
| `g_1,g_2` | 两个不同 pace 的 student | `student_1`, `student_2` |
| `G_1,G_2` | 对应 EMA teacher | `teacher_1`, `teacher_2` |
| `r` | 最终 trusted set 占 U 的比例 | `max_trust_ratio` |

## 5. 与 nnPU 的关系

Self-PU 的基础风险使用 nnPU 的 sigmoid loss。令：

```math
\ell_{sig}(g(x),y)=
\frac{1}{1+\exp(y g(x))}.
```

无偏 PU 经验风险为：

```math
\widehat R_{uPU}(g)=
\pi_p\widehat R_P^+(g)
+\widehat R_U^-(g)
-\pi_p\widehat R_P^-(g).
```

nnPU 版本为：

```math
\widehat R_{nnPU}(g)=
\pi_p\widehat R_P^+(g)
+\max\left(0,
\widehat R_U^-(g)-\pi_p\widehat R_P^-(g)
\right).
```

Self-PU 并没有删除 nnPU，而是让：

- trusted set 使用更强的 supervised CE；
- 未信任样本继续使用 nnPU 或经校准的 CE/nnPU 混合；
- student/teacher consistency 作为额外正则。

因此 Self-PU 的实现应复用 nnPU 的风险定义，但不能把 Self-PU 注册成 nnPU 的别名。

## 6. Self-Paced PU Learning

### 6.1 置信度排序与双向选择

当前模型 `g` 输出概率：

```math
p(x)=f(g(x))=P(Y=+1\mid x).
```

对 U 按 `p(x)` 降序排列：

- 前 `n_t` 个样本作为高置信正类；
- 后 `n_t` 个样本作为高置信负类；
- 两组样本从当前 U pool 移入 `D_trust`。

Self-PU 同时选择正、负两端，使 trusted set 在每次扩展时保持正负数量平衡；不能只选择高置信正例。

### 6.2 Dynamic Rate Sampling

论文不固定每轮选择数量，而是在 warm-up 后把 trusted set 大小从 0 线性增加到 `r|D_U|`：

```math
|D_{trust}(t)|
=
\left\lfloor
r|D_U|\cdot
\frac{t-t_{warmup}}{t_{end}-t_{warmup}}
\right\rfloor.
```

论文实验中：

- warm-up 约 10 epochs；
- `r` 在 10%–40% 范围研究；
- 默认实验最终 trusted ratio 常设为 25%。

项目实现必须保存每轮的 `target_trust_size`，否则无法诊断采样 pace。

### 6.3 In-and-Out Trusted Set

被选过的样本不是永久可信。每次扩大 trusted set 时，重新检查已有 `D_trust`：

```text
1. 对当前 D_trust 重新计算概率和置信度；
2. 删除当前模型已不再高置信的样本；
3. 被删除样本回到 D_U 或 untrusted pool；
4. 从更新后的 U 中补充新的高置信正/负样本。
```

该机制用于避免早期模型错误把样本加入 trusted set 后造成错误累积。实现中不能用一次性 `set.add()` 代替。

### 6.4 Soft Labels

对已选样本，论文使用当前模型概率作为 soft label：

```math
\tilde y(x)=
\left[1-f(g(x)),f(g(x))\right]
```

对应负类/正类概率。这样可以减轻伪标签错误的硬边界影响。hard positive/hard negative 只用于 trusted set 的选择方向，训练 CE 应保留 soft target。

### 6.5 Self-Paced Hybrid Loss

最初 self-paced 阶段的混合损失为：

```math
L_{SP}=
\sum_{(x,y)\in D_{trust}}L_{CE}(x,y)
+\sum_{x\in D-D_{trust}}L_{nnPU}(x).
```

这里的 `D-D_trust` 包含原始 P 和剩余 U；正样本 P 仍需进入 nnPU 风险计算，不能仅把剩余 U 当作负类。

## 7. Self-Calibrated Loss Reweighting

### 7.1 未信任样本的两项损失

对 `x_i in D_U-D_trust`，定义：

```math
L_{CE}^{S}(x_i)=
-\tilde y_i\log f(g(x_i))
-(1-\tilde y_i)\log(1-f(g(x_i))),
```

其中 `tilde y_i=f(g(x_i))` 是 soft target。将它与样本的 nnPU loss 组合：

```math
l(x_i,w_i)=
w_{i,1}L_{CE}^{S}(x_i)
+w_{i,2}L_{nnPU}(x_i).
```

`w_i` 是每个样本独立的二维权重，不是全 batch 一个标量。

### 7.2 Meta inner step

先用当前权重对训练模型做一个小步更新：

```math
\theta^*=
\theta-delta\nabla_\theta
\sum_{i=1}^{n}l(x_i,w_i).
```

其中 `delta` 是 meta inner-step 学习率，`w_i` 先取接近 0 的扰动值，使得验证集梯度可以反映每项损失对泛化的贡献。

### 7.3 Clean validation meta-gradient

用更新后的 `theta*` 在 clean validation batch 上计算 CE：

```math
u_i=
-\left.
\frac{\partial}{\partial w_i}
\frac{1}{m}\sum_{j=1}^{m}
L_{CE}(g_{\theta^*}(x_j^v),y_j^v)
\right|_{w_i=0}.
```

只保留正的 meta gradient：

```math
\tilde w_{i,k}=\max(u_{i,k},0),
```

再分别归一化两列：

```math
w_{i,k}=
\frac{\tilde w_{i,k}}
{\sum_j\tilde w_{j,k}}.
```

如果某一列权重和为 0，实现必须定义稳定回退策略，例如均匀权重或只使用另一项 loss，并记录诊断标记。

### 7.4 CE 权重平衡约束

论文使用 `gamma` 限制 CE 权重总量。按 `w_{i,2}` 的累计量定义：

```math
T=\sup\left\{k:\sum_{i=1}^{k}w_{i,2}<\gamma n\right\}.
```

然后：

```math
w^*_{i,1}=w_{i,1}\mathbf 1\{i<T\},
```

```math
w^*_{i,2}=w_{i,2}\mathbf 1\{i<T\}+\mathbf 1\{i\ge T\}.
```

直觉是：当 self-calibrated CE 不应继续增加时，后续样本退回 nnPU 项，而不是让不可靠的 soft CE 主导训练。

### 7.5 Reweighted Hybrid Loss

```math
L_{SP+Reweight}=
\sum_{(x,y)\in D_{trust}}L_{CE}(x,y)
+\sum_{x\in D_U-D_{trust}}l^*(x)
+\sum_{x\in D_P}L_{nnPU}(x),
```

其中：

```math
l^*(x)=
\frac{1}{n}\sum_{i=1}^{n}l(x_i,w_i^*).
```

论文在两个 student 上分别维护对应的 trusted set 和 reweighting 过程；项目实现不能只为一个 student 计算 weights 后无说明地复制给另一个 pace。

## 8. Self-Supervised Distillation

### 8.1 两个不同 pace 的 student

初始化两个相同结构的 student：

```math
g_1\sim g_2\quad\text{architecture 相同},
```

但设置不同的 trusted-set 最终比例/采样速度，因此形成 `D_trust,1` 与 `D_trust,2`。不同 pace 产生的模型差异被用作互相蒸馏的监督来源。

### 8.2 Student-to-student consistency

对不在对应 trusted set 的样本计算预测 MSE：

```math
L_{MSE}(g_1,g_2,x)=
\left\|f(g_1(x))-f(g_2(x))\right\|_2^2,
```

并对方向相反的集合分别计算：

```math
L_{students}=
\sum_{x\in D-D_{trust,1}}l_{stu}(g_1,g_2,x)
+\sum_{x\in D-D_{trust,2}}l_{stu}(g_2,g_1,x).
```

### 8.3 Hard sample mining

论文不是对所有 untrusted 样本无条件施加 MSE，而是只对 nnPU 风险足够大的挑战样本启用：

```math
l_{stu}(g_i,g_j,x)=
\begin{cases}
L_{MSE}(g_i,g_j,x),
&L_{nnPU}(x)>\alpha L_{MSE}(g_i,g_j,x),\\
0,&L_{nnPU}(x)\le\alpha L_{MSE}(g_i,g_j,x).
\end{cases}
```

`alpha` 越小，触发蒸馏的样本越多；实现必须记录每轮的 active hard-sample fraction。

### 8.4 EMA teacher

为每个 student 建立同结构 teacher `G_i`，其参数使用 moving average：

```math
\Theta_{i,t}=
\beta\Theta_{i,t-1}+(1-\beta)\theta_{i,t}.
```

其中 `theta_i,t` 是当前 student 参数，`Theta_i,t` 是 teacher 参数。项目实现必须使用 teacher 的 `no_grad` 更新，不让 teacher 参数直接参与反向传播。

Teacher consistency 为：

```math
L_{teachers}=
\sum_{x\in D}
\left\|f(G_1(x))-f(g_1(x))\right\|_2^2
+\sum_{x\in D}
\left\|f(G_2(x))-f(g_2(x))\right\|_2^2.
```

### 8.5 总目标

论文整体目标为：

```math
L=L_{SP+Reweight}+L_{students}+L_{teachers}.
```

训练阶段按论文顺序：

1. warm-up：只使用基础 PU 训练；
2. 约第 10–50 epoch：self-paced + self-calibrated reweighting；
3. 约第 50–200 epoch：加入 student/teacher distillation；
4. 最后比较两个 teacher 在 validation set 上的表现，选择更好的 teacher 测试。

## 9. 完整算法流程

```text
输入：D_P、D_U、clean D_V、class prior pi_p
输入：两个 pace 配置 r1/r2、模型 g1/g2、teacher G1/G2

1. 初始化两个 student 和两个 EMA teacher；warm-up 若干 epoch。
2. 对每个 student 独立执行 self-paced：
   a. 对 U 计算 sigmoid 概率；
   b. 按概率排序，选择两端等量样本加入 trusted set；
   c. 重新检查历史 trusted set，低置信样本移出；
   d. trusted set 用 soft-label CE，剩余数据用 nnPU。
3. 对两个 student 的 untrusted batch：
   a. 计算 soft CE 和 nnPU 两项 loss；
   b. 用 clean validation batch 做 one-step meta update；
   c. 得到非负、归一化、受 gamma 约束的 per-instance weights；
   d. 更新 student 参数。
4. trusted set 达到预定阶段后：
   a. 计算两个 student 在各自 untrusted 集上的挑战样本；
   b. 加入不同 pace 的 student MSE；
   c. 加入 student 与 EMA teacher 的 MSE；
   d. 更新 student，再更新 teacher EMA。
5. 在 clean validation set 上比较 teacher_1 和 teacher_2。
6. 选择验证性能更好的 teacher 作为最终模型，输出 decision score/probability。
```

## 10. 超参数与论文实验协议

| 参数 | 含义 | 论文/项目建议 |
|---|---|---|
| `class_prior` | `pi_p` | 必须在训练 fold 内确定；论文默认已知 |
| `warmup_epochs` | warm-up 时长 | 论文约 10 epochs |
| `max_trust_ratio` | 最终 trusted/U 比例 | 论文研究 10%–40%，常用 25% |
| `pace_1`, `pace_2` | 两个 student 的 sampling pace | 建议不同，例如 20%/30% |
| `meta_step_size` | `delta` | reweighting inner step |
| `reweight_gamma` | CE 权重总量约束 | 论文使用小于 1 的约束因子；需与源码核对 |
| `distillation_alpha` | hard mining 阈值 | 控制 student MSE 触发范围 |
| `ema_decay` | `beta` | teacher moving-average 衰减 |
| `student_loss_weight` | student consistency 权重 | 论文整体 loss 可先按 1 实现 |
| `teacher_loss_weight` | teacher consistency 权重 | 论文整体 loss 可先按 1 实现 |
| `optimizer` | Adam | 论文使用 Adam |
| `scheduler` | cosine annealing | 论文实验使用 |
| `batch_size` | 训练/验证 batch | MNIST/CIFAR 256，ADNI 64 |

### 10.1 论文 benchmark

| 数据集 | 正类 | 负类/未标记来源 | `pi_p` | 模型 |
|---|---|---|---:|---|
| MNIST | 奇数 1/3/5/7/9 | 偶数 0/2/4/6/8 | 0.49 | 6-layer ReLU MLP |
| CIFAR-10 | airplane/automobile/ship/truck | bird/cat/deer/dog/frog/horse | 0.40 | 13-layer ReLU CNN |
| ADNI | clinical AD 或 SUVR≥1.08 | 剩余训练样本，含 MCI 潜在正例 | 0.43 | multi-scale 3-branch CNN |

论文使用 MNIST/CIFAR 的 `n_p=1000`，ADNI 的 `n_p=113`；U 集为剩余训练样本。正式复现还需记录官方预处理、数据划分和五次运行的均值/标准差。

## 11. API 设计要求（集成前规格）

### 11.1 推荐公共接口

```python
class SelfPUClassifier(BasePUClassifier):
    def __init__(
        self,
        class_prior,
        *,
        backbone=None,
        warmup_epochs=10,
        self_paced_start=10,
        self_paced_end=50,
        distill_start=50,
        max_epochs=200,
        max_trust_ratio=0.25,
        pace_1=0.20,
        pace_2=0.30,
        meta_step_size=1e-3,
        reweight_gamma=1 / 16,
        distillation_alpha=10.0,
        ema_decay=0.99,
        batch_size=256,
        learning_rate=1e-3,
        random_state=None,
        device="cpu",
    ):
        ...
```

### 11.2 fit 数据协议

```python
fit(
    X,
    y_pu,
    *,
    class_prior=None,
    validation_data=None,
    sample_weight=None,
)
```

| 参数 | 约定 |
|---|---|
| `X` | P/U 数据；图像输入可为 NCHW 或项目统一 tensor protocol |
| `y_pu` | `1` 表示 labeled positive，`0` 表示 U |
| `class_prior` | 覆盖构造参数的 `pi_p`，必须在 `(0,1)` |
| `validation_data` | 必须是 `(X_val, y_val)` 且 `y_val in {+1,-1}`；Self-calibration 和 teacher 选择使用 |
| `sample_weight` | 需要明确是否同时作用于 nnPU、trusted CE 和 meta batch；不能静默只作用一部分 |

### 11.3 拟合属性

| 属性 | 含义 |
|---|---|
| `student_1_`, `student_2_` | 两个不同 pace 的最终 student |
| `teacher_1_`, `teacher_2_` | EMA teacher |
| `best_teacher_` | validation 选择的最终 teacher |
| `trusted_indices_` | 最终 trusted set 及其 soft labels |
| `trusted_history_` | 每 epoch 的进入/移出/规模 |
| `reweight_history_` | 每轮 per-instance weight 统计、零权重比例、gamma 截断比例 |
| `distillation_history_` | active hard sample 比例、student/teacher MSE |
| `training_history_` | nnPU、CE、student、teacher、总损失 |
| `class_prior_` | 实际使用的 `pi_p` |
| `best_teacher_index_` | 1 或 2 |

### 11.4 预测语义

- `decision_function(X)` 返回最终 teacher 的 logit；
- `predict_proba(X)` 返回 sigmoid 后验形式的工程概率；
- `predict(X)` 使用 0 logit 或显式 threshold；
- 不把 trusted pseudo-label 当作最终模型输出；
- 保存 checkpoint 时必须包含两个 student、两个 teacher、trusted-set state 和 optimizer/scheduler state。

## 12. Toolbox 集成映射

| 模块 | 责任 | 当前状态 |
|---|---|---|
| `pu_toolbox/estimators/deep/self_pu.py` | Self-PU estimator、三阶段训练循环 | ✅ native core |
| `pu_toolbox/losses/nnpu.py` | 复用 sigmoid nnPU 风险与非负校正 | ✅ 已有基础 |
| `pu_toolbox/core/validation.py` | P/U 标签和 `class_prior` 校验 | ✅ 可复用 |
| `pu_toolbox/model_selection/` | clean validation split、无测试泄漏协议 | ⏳ 需扩展 |
| `pu_toolbox/registry/builtin_methods.py` | `self_pu` metadata 和 lazy binding | ✅ native |
| `tests/unit/estimators/test_self_pu.py` | trusted set、reweight、EMA、训练 smoke | ✅ |
| `benchmarks/paper_like/self_pu/` | MNIST/CIFAR/ADNI 配置 | ⏳ |

建议把以下组件拆开测试，而不是只写一个端到端训练测试：

1. `TrustedSetManager`：排序、动态规模、in-and-out、soft labels；
2. `MetaLossReweighter`：inner update、validation gradient、非负化、gamma 截断；
3. `EMATeacher`：无梯度更新和参数递推；
4. `StudentDistillationLoss`：hard sample mining 和 mask；
5. `SelfPUClassifier`：三阶段调度和最终 teacher 选择。

## 13. 测试与验收标准

### 13.1 Trusted set

- 第一个 self-paced epoch 的 trusted set 规模符合 dynamic-rate 公式。
- 每次选择正、负样本数量相同或符合显式配置。
- 已选样本在当前模型低置信时能被移出。
- soft label 等于当前模型概率，且不被错误地硬编码成 0/1。
- trusted set 不包含重复索引，P 原始样本不被误当作 U 伪标签。

### 13.2 Meta reweighting

- 没有 `validation_data` 时拒绝启用 self-calibration，或明确降级为消融模式并发出 warning。
- 权重非负，归一化后每一列和为 1；零梯度列有稳定回退。
- `gamma` 截断后 CE 权重总量不超过约束。
- inner-step 使用可微的临时参数，不能原地污染正式 student 参数。
- validation labels 不参与最终测试指标以外的训练数据泄漏。

### 13.3 Distillation

- 两个 student 结构一致但 pace 不同。
- student MSE 只作用于对应 untrusted mask。
- hard sample mining 的 active mask 与 `L_nnPU > alpha * L_MSE` 一致。
- teacher 参数更新满足 EMA 公式且不产生梯度。
- teacher 的 MSE 使用全数据或论文指定数据范围，并记录具体 mask。

### 13.4 训练调度

- warm-up、self-paced/reweight、distillation 三阶段边界可通过参数配置。
- 每轮记录 trusted size、loss components、active ratio 和 validation metric。
- 训练结束选择 validation 更好的 teacher，而不是依赖固定 teacher 编号。
- 固定 seed 时合成小数据训练不会产生 NaN/Inf。

### 13.5 Paper-like benchmark

- MNIST、CIFAR-10、ADNI 采用论文正负类别定义和预处理；
- `n_p`、`n_u`、`pi_p` 与论文设定一致；
- batch size、Adam、cosine scheduler、warm-up 和阶段 epoch 对齐；
- 运行五个随机种子，报告 mean/std；
- 做 ablation：无 dynamic rate、无 in-and-out、无 soft labels、无 reweight、无 students、无 teachers；
- 报告 accuracy 之外的 AUC、precision、recall、trusted-set precision 和训练稳定性。

## 14. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `official_exact` |
| Implementation status | `NATIVE` clean-room core；paper-like benchmark 待完成 |
| Official code | `TAMU-VITA/Self-PU` |
| 依赖 | PyTorch、数据增强/图像 backbone、clean validation protocol |
| 最大风险 | self-calibration 依赖真实验证标签；trusted set 错误会造成确认偏差；meta gradient 增加显存和实现复杂度 |
| 复现风险 | 官方实验使用图像 backbone、特定数据协议、五次运行；不能用线性模型 smoke test 宣称复现论文结果 |
| 集成顺序 | 先实现 TrustedSetManager，再实现 nnPU/CE 混合，再实现 reweight，最后加入 student/teacher distillation |

## 15. 参考资料

1. Chen et al., *Self-PU: Self Boosted and Calibrated Positive-Unlabeled Training*, ICML 2020, PMLR 119:1510–1519。
2. 官方论文页面：[PMLR](https://proceedings.mlr.press/v119/chen20b.html)。
3. 官方代码：[TAMU-VITA/Self-PU](https://github.com/TAMU-VITA/Self-PU)。
4. nnPU 风险实现与理论：[Kiryo et al., 2017](https://arxiv.org/abs/1703.00593)。
