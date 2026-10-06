# VPU 采样假设审计（Gate 0，P3.1 局部）

> 审计日期：2026-09-28
> 审计人：shuidisjtu（**单方技术审计，合作者复核未获得**）
> 上游锁定：`https://github.com/HC-Feynman/vpu` @ `603bcc1e628795f57a5ac87e5b0b977273b7cf91`（MIT）
> 结论转抄去向：决策 D21；台账条目 `method_ledger.json` 的 `vpu` 在 P3.1 后续切片中**转抄 D21**，不转抄本文件
> **引用格式**：作者仓库一律写作 `HC-Feynman/vpu @ 603bcc1: <文件>:<行>`，PU-Bench 对照一律写作 `PU-Bench: <文件>:<行>`——两者均可由公开仓库与固定提交复核。本地核对路径（见 §3）只是审计环境说明，**不构成引用**。

本记录是 VPU 接入公共训练视图前的 Go/No-Go 门禁产物。它回答一个问题：**VPU 的变分目标所要求的边缘样本池，在工具箱的 OS 分区上应当由哪些行构成**。结论用于冻结台账的 `native_sampling_assumption`，并决定 estimator 的 `os_or_ts` 语义。

**本审计不构成 P3.1 验收。** VPU 的完整 P3.1 验收（方法卡、原文可追溯、冒烟、公开行为对照、共享规格）仍未完成，且 VPU 不进入冻结执行矩阵。

## 1. 结论

```text
native_sampling_assumption = ts
run_view                    = ts-compatible
calibration_applied         = true
```

即：**VPU 的边缘池必须是 $D_U\cup D_P$**。工具箱现有实现用传入 `fit` 的完整 `X` 作为边缘池，与作者实现的池**逐集合等价**，等价于始终运行校准后的 TS 视图——只是此前没有显式记录。

## 2. 论文事实：边缘项要求总体边缘分布

VPU 用非负函数 $\Phi_\theta(x)$ 近似正类后验，变分目标（论文式 (6)）为

```math
L_{var}=\log\mathbb E_{x\sim f}[\Phi_\theta(x)]-\mathbb E_{x\sim f_P}[\log\Phi_\theta(x)].
```

其中 $f$ 是**训练总体的边缘分布**，$f_P$ 是正类条件分布。这是**论文事实**：$f$ 既不是类条件分布，也不是「仅未标记行」的分布。方法卡 [`method_cards/VPU.md`](../../method_cards/VPU.md) 第 21–22 行记录了同一式子。

## 3. 作者实现（本次亲验）

2026-09-28 拉取锁定提交并逐行核对。本节引用一律用 `HC-Feynman/vpu @ 603bcc1: <文件>:<行>` 的形式。（审计环境说明：克隆落在项目树之外的 `F:/Temp/refs/HC-Feynman-vpu`，仓库不随提交入库。）

**3.1 作者自己写明边缘 loader 覆盖正例。** 两处 docstring：

- `vpu.py:79` —— `x_loader: loader for training data (including positive and unlabeled)`
- `vpu.py:158` —— `x_loader: loader for the whole training set (positive and unlabeled)`

**3.2 池结构是包含关系，不是互斥划分。** `dataset/dataset_cifar.py:32-40` 对每个类取 `idxs[0:-500]` 进未标记池；正例类额外把 `idxs[:n_labeled_per_class]` 放进 `train_labeled_idxs`。`idxs` 在之前已 shuffle，而 cifar10 / `num_labeled=3000`（`run.py:10`）/ 4 个正类（`run.py:22`）下 `n_labeled_per_class=750`，远小于每类保留量 4500，故**被标记的正例仍然留在 X 池里**，只是标签被抹成 `-1`（`dataset/dataset_cifar.py:102`）。

规模：训练分区 50000 行，每类留 500 行作验证集（`dataset_cifar.py:40`），故 X 池 = 45000 行；其中 3000 行是已标记正例（`p_loader`，`dataset_cifar.py:131`）。

**3.3 映射到工具箱词汇是精确的。** 作者的 X 池 = 训练分区中除验证集以外的全部行 = 工具箱 OS 分区里的 $D_P\cup D_U$。因此工具箱现有的「完整 `X` 作边缘池」与作者实现**逐集合等价**。

**3.4 训练与预测的其余环节逐项对上**（均为视图无关，列出以固定核对范围）：

| 环节 | 作者 | 工具箱 | 一致 |
|---|---|---|---|
| 变分项 | `vpu.py:118` `logsumexp(log_phi_x) - log(len) - mean(log_phi_p)` | `vpu_objective` | ✅ 逐项相同 |
| 优化器 | `vpu.py:34`/`:41` `Adam(betas=(0.5, 0.99))` | `VPUClassifier` | ✅ |
| MixUp 采样 | `vpu.py:126` `Beta(mix_alpha, mix_alpha)`，`run.py:13-14` 默认 `0.3` / `lam=0.03` | `mixup_alpha=0.3` / `regularization_weight=0.03` | ✅ |
| MixUp 目标梯度 | `vpu.py:121` `target_x = output_phi_x[:, 1].exp()`（**未 detach**） | `vpu_objective` 保留梯度 | ✅ |
| 归一化 | `vpu.py:175-179` 取 X 池上最大 $\log\Phi$ | `max_log_phi_` 取边缘池最大 $\log\Phi$ | ✅ |
| 判决阈值 | `vpu.py:194` `log_phi > log(0.5)` | `decision_function` 的 0 阈值 | ✅（边界 `>` 与 `>=` 之别） |

**3.5 MixUp 写法看似不同、实则分布等价，必须主动说明。** 作者写 `data = lam * data_x + (1 - lam) * data_p_perm`（`vpu.py:128`），工具箱写 `mixed = mixing * p_batch + (1 - mixing) * x_batch`。令 `mixing = 1 - lam` 即逐项相同，而 $\mathrm{Beta}(\alpha,\alpha)$ 对称、`lam` 与 `1-lam` 同分布；作者对 P 端做了 `randperm`（`vpu.py:124-125`）而工具箱不置换，但两侧样本本就独立抽取，配对联合分布一致。已数值核验：`data`、`log_target` 与正则损失逐项 `allclose`（float32 量级约 `3e-8`）。**这是参数化差异，不是语义差异**；并排读代码时不得判为缺陷。

## 4. 对照实现（PU-Bench）

PU-Bench 的 VPU 套件（本地对照库，见 §3 环境说明）由未参与本项目实现的人写给另一个基准套件，构成独立旁证：

- `train/vpu/losses.py:12-14` 记录源实现「trains a log-softmax model Phi and uses column 1 as `log phi`」，目标式为 `logmeanexp(log_phi_x) - mean(log_phi_p) + lam * mixup_regularizer`——与工具箱逐项一致；
- `config/methods/vpu.yaml:16` 把 **「separate labeled-positive P loader and all-training-data X loader」列为从源实现 retained 的组件**，而非 `benchmark_adaptations`（该清单始于 `:25`）；
- `train/vpu/trainer.py:90-98` 的训练侧 `vpu_x_loader` 直接取整个训练 dataset。

**引用强度**：该套件自声明 `source_reproduction: false`（`vpu.yaml:7`），故不能替代作者仓库本身；本次作者侧已亲验（§3），故它只作旁证。

## 5. 数据事实：OS 分区下 $D_U$ 不是 $p(x)$ 样本

协议 §2.3 第 1 条锁定「基础数据始终是 OS」——不在 `ts` 时重新独立抽样 TS 数据。在 OS 生成下，标记只作用于一部分正例（SCAR，$n_L=\mathrm{round}(c\,n_+)$），其余行成为未标记行。因此**工具箱持有的 $D_U$ 排除了已标记正例**，其经验分布是 $p(x\mid s=0)$ 而不是 $p(x)$；$D_U\cup D_P$ 才还原 $p(x)$ 样本。

这条论证不是本审计新立的：**同一条论证已为 `dist_pu` 的 label-distribution alignment 项冻结于决策 D18②**（「OS 下 $X_U$ 排除了已标记正例、不再代表完整 marginal，约束落在错的值上；并集把它搬回正确落点」）。

**定性强度高于 D18③**。D18③ 明写 `dist_pu` 的 entropy 项「只有协议一致性依据，不是数学必然」；VPU 不同——**论文公式直接要求总体边缘**（§2），叠加数据事实（本节），是本项目内少见的「论文事实 + 数据事实」双重依据。两处措辞**不得互抄**。

## 6. OS 对照的定性：有效，但不是原生视图

协议为原生 TS 方法保留 OS 对照。对 VPU 而言：

> OS 对照**存在且可解释**，它是**协议定义的消融**——把边缘池限制到 $p(x\mid s=0)$，从而在变分目标里引入一个可观测的错配。它**不是**「VPU 在其原生假设下的运行」，任何报告都不得把它读作 VPU 的原生结果。

该定性与 D18③ 对 `dist_pu` entropy 项的定性同类，也保证了 OS/TS 仍是单变量对照。

## 7. 为何不是 `both`

- 字面上 VPU 需要的是「代表 $p(x)$ 的池」。在 genuine OS 数据集定义下（$X\sim p(x)$、标注是其后的独立一步），`X` 本身也是 $p(x)$ 样本，故 `both` 可辩。
- 但 registry 对 VPU 只登记 `Scenario.CASE_CONTROL`（`get_metadata("vpu").scenario`），而 `tests/contract/test_ledger_registry_consistency.py` 的 `test_native_sampling_assumption_matches_registry_scenario` 要求台账值与该 scenario **严格等价**。选 `both` 就必须改 registry 的 scenario，属方法学定位变更，超出本切片。
- 且 `resolve_training_view` 对 `ts` 与 `both` **行为完全相同**（两者同以 `native not in {"ts", "both"}` 判定），`both` 不带来额外能力。
- **故登记 `ts`**，并在此记录 `both` 的可辩性已审议。

**词汇澄清**：本项目的 `ts` / `CASE_CONTROL` 指的是协议 §2.3 的**视图语义**（边缘池包含正例、即 TS-OS 校准），而**不是**要求数据本身按两样本独立抽样生成——协议 §2.3 第 1 条明写基础数据恒为 OS。VPU 属于前者。

## 8. 验证路径：当前与上游一致，且存在一个未决问题

**当前事实（本审计核对的代码状态，不是设计意图）**：上游与本工具箱**都把验证变分风险对整份验证分区连同其正例子集求值**，即验证边缘池为 $U_{val}\cup P_{val}$。

- 作者侧：`vpu.py:210` 把 `val_x_loader` 描述为「the whole validation set (including positive and unlabeled data)」，`dataset/dataset_cifar.py:114`/`:133-134` 取整个验证分区；`cal_val_var`（`vpu.py:204-237`）同时消费 `val_x_loader` 与 `val_p_loader`。
- PU-Bench 侧：`PU-Bench: train/vpu/trainer.py:112-138`、`:165-189` 同构。
- **工具箱侧：同样如此**——`pu_toolbox/estimators/risk/vpu.py:152-158` 把整个 `val_X` 作为验证边缘池，`:222-231` 对它求变分风险。

**故当前不存在「与上游的验证口径分歧」**；本审计早先版本曾如此断言，属把尚未发生的设计决策写成既成事实，已更正。

**不得引用 D17②/D19⑥ 来支持此处。** 决策 D17② 明文「只冻结 `pusb_kernel`，不推广到其他方法」，D19⑥ 只冻结 `self_pu`——两者**都不覆盖 VPU**。把它们的「验证保持 OS」口径搬到这里，是对 D 记录的误用。

**裁决（2026-09-28，决策 D22）：走方向 1——保持与上游一致。**

- VPU 的验证变分风险在本项目内定位为 **source diagnostic**：允许按上游定义（合并边缘池）计算、记录并作 os/ts 对照；**不得**充当 PA/OA 选模准则。
- 协议 §2.3 据此新增「适用范围」段，明确该条约束的是**参与选参或裁决**的验证视图——**规则本身未被修改**，不存在对 VPU 的例外。
- 若将来 VPU 进入执行矩阵且需要 PA 选模，**必须另行裁决**：或改用符合 §2.3 的 OS 验证视图（此时即成为对上游的显式分歧，须按 D17② 的论证结构自行成文），或为该行指定其它 PA 机制。
- `survey_protocol_v1.json` **不改**：改它会变更协议摘要并触发 D12 式的绑定重绑级联，而 VPU 不在执行矩阵内、无执行影响。
- 本裁决**不改任何代码**——当前行为即目标行为。

## 9. 与视图无关的既有适配缺口（本切片不改，仅登记）

以下缺口在本次审计中暴露，**与 OS/TS 视图无关**，不属本切片范围；登记以免被误读为接线引入的偏差：

1. **无学习率衰减**：作者每 20 epoch 将学习率减半并重建优化器（`vpu.py:39-41`），工具箱实现中不存在。
2. **学习率默认值差一个数量级**：作者 `run.py:11` 为 `3e-5`，工具箱 `VPUClassifier` 为 `3e-4`。
3. **训练预算不同构**：`epochs` 100 vs 50、`batch_size` 128 vs 500（`run.py:9`/`:12`），且每 epoch 迭代数由数据量推导，而上游固定 `val_iterations=30`（`vpu.py:92`、`run.py:8`）。
4. 网络为工具箱小型 MLP，非论文 UCI 七层 MLP300 或 CIFAR CNN；未完成共享 backbone、图像路径与公开数值对照。

## 10. 残留不确定性

- 上述 §9 全部条目；
- **§8 的验证路径**：已由决策 D22 裁决为「保持与上游一致 + 定位 source diagnostic，不作 PA/OA 选模准则」，协议 §2.3 的适用范围随之写明。**残留条件**：将来 VPU 进入执行矩阵且需要 PA 选模时，必须另行裁决（D22 ③(c)）；
- 本审计为**单方技术审计**，无合作者复核；
- 作者仓库亲验限于 cifar10 路径；`dataset_avila.py` / `dataset_grid.py` / `dataset_pageblocks.py` 的池构造未逐一核验（结构同源，但未亲验逐字）；
- VPU 输出是归一化分数，不是独立概率校准后验；
- OS 对照的行为变化（2026-09-28 已随接线发生）：显式 `os` 请求对 VPU 的语义由「静默跑 TS」变为「真跑 OS」，方法卡已记录；同一变更修好了路由转发（决策 D23）。

## 11. 证据清单

| 层 | 载体 | 位置 |
|---|---|---|
| 论文事实 | 论文式 (6)；方法卡 | `method_cards/VPU.md:21-22` |
| 作者实现 | `HC-Feynman/vpu @ 603bcc1`（亲验） | `vpu.py:79`、`:158`、`:21`、`:110-118`、`:34`/`:41`、`:121`、`:124-129`、`:175-179`、`:194`、`:204-237`；`dataset/dataset_cifar.py:32-40`、`:102`、`:114`、`:131-134`；`run.py:8-14`、`:22`、`:67`、`:71-75` |
| 对照实现 | `PU-Bench`（本地对照库） | `train/vpu/losses.py:12-14`；`train/vpu/trainer.py:13`、`:90-98`、`:112-138`、`:165-189`；`config/methods/vpu.yaml:6-12`、`:14`、`:16`、`:23` |
| 数据事实 | 协议 §2.3；决策 D18② | `pu_survey_protocol.md` §2.3 第 1 条；`survey_execution_plan.md` 决策 D18② |
| 工具箱现状 | 训练侧 | `pu_toolbox/estimators/risk/vpu.py:128`、`:129-130`、`:170`、`:182-183` |
| 工具箱现状 | 验证侧（§8 未决问题的证据） | `pu_toolbox/estimators/risk/vpu.py:152-158`、`:222-231` |
