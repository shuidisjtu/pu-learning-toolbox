# P2.0e TS-OS 校准接入训练执行链（交付记录）

> **2026-10-04 技术复核**：五方法校准/路由回归通过，交付 Excel 的逐行视图与校准标志一致；见 [技术复核](../reviews/pilot_independent_review_20261004.md)及[表格对账](p21_workbook_review_20261004.md)。这验证接线与记录，不代替方法学签署或原始制品重放。

> 定位：本文件是 P2.0e 的交付记录，**随各方法接线逐次追加**。P2.0e 的整体状态与完成口径
> 仍以 [`survey_execution_plan.md`](../survey_execution_plan.md) 的 P2.0e 行与决策 D16 为准；
> 数值与状态以方法台账 `pu_toolbox/experiment/method_ledger.json` 的
> `run_view` / `calibration_applied` 与各 run 的 manifest 为准，本文件只作解释。
>
> 当前进度：`nnpu`（#68）、`upu`（#74）、`pusb_kernel`（#76）、`dist_pu`（#77）、`self_pu`（本切片）
> 已接线，五个适用方法**无剩余待接线项**；线性 `pusb` 经 2026-09-27 裁决**不适用校准**（其训练信号即
> 「U ≡ 负类」，不存在可替换的未标记损失输入，见 D16 ①），本切片不含其工作。逐方法口径见 D16 ③ / D17 / D18 / D19。

## 1. `ts` 视图的语义

协议 §2.3 的 `ts` 视图：**无标签风险项的经验分布由 `D_U` 换成 `D_U ∪ D_P`**，即 PU-Bench
的 case-control 把已标注正例放回 U 的等价形式（决策 D10 已认定两者等价）。逐 run 的实际
视图记在 manifest 的 `run_view` / `calibration_applied`——是**实际视图**，不是台账的
`native_sampling_assumption`；它同时是聚合分组维度（D13）与续跑判定维度（D14）。

## 2. 各方法的校准形态（调查结果，供后续切片使用）

下表是 P2.0e 接线前对逐个方法的注入点调查。**它只记录工程形态，不构成任何方法的数学结论**
（见 D16 ②）：每个方法轮到实施时必须单独形成规格、红灯测试与验收证据。

| 方法 | 求解形态 | 注入点 | 已知障碍 | 在 Pilot 矩阵内 | 状态 |
|---|---|---|---|---|---|
| `nnpu` | 逐 mini-batch SGD | 每个训练批次把正例批 `cat` 进无标签损失输入 | — | 是 | 已接线（#68） |
| `upu` | 全量凸求解（squared 闭式 / logistic L-BFGS / double-hinge SLSQP） | 唯一构造点：无标签角色集合与 RBF 候选池 | — | 是 | 已接线（#74） |
| `pusb_kernel` | 全量 BFGS + (σ,λ) 网格 CV | `_pu_objective_and_gradient`（**2 个调用点 / 3 个角色**，训练折与终拟合共用 `_fit_coefficients`） | ① CV **验证折**按协议须保持 OS，标志只能作用于训练折与终拟合；② 冻结训练先验分位数阈值会随视图漂移（属训练产物，非选择项） | 是 | 已接线（本切片 §5） |
| `dist_pu` | 全量梯度下降（`batch_size` 为兼容性死参数） | 分布对齐项 | 对齐目标里的 `-π` 是"U ~ p(x)"的**显式编码**，并入 P 后须**重新推导目标**，不能机械拼接——属方法学判断 | 是 | 已接线（本切片 §6） |
| `self_pu` | 每 epoch 各抽一次批的 SGD（无 DataLoader） | **只有 1 处**：负 PU 项的角色集合（未信任 U 行） | 调查时判断为"最硬"（trusted-set 按分数取最低分为负半）——接线时的结论是**该风险在结构上不存在**：manager 只吃 U-local 概率、形状 fail-loud，已知正例根本不在其人口内（§7.2） | 是 | 已接线（本切片 §7） |
| 线性 `pusb` | sklearn `LogisticRegression` 一次 `fit` | **无** | 训练信号即「U ≡ 负类」，不存在与风险估计器同形的"未标记损失输入" | 否（附加工程基线不入榜） | **不适用校准**（2026-09-27 裁决，D16 ①）|

## 3. 接线范式（工程经验，不构成其他方法的数学先例）

**采纳：估计器内就地接线。** `fit` 增加 keyword-only `os_or_ts`，由估计器按自身风险公式替换
无标签经验项。理由：上游 `route_training_view` 按 `fit` 签名判定（`protocols.py`），闸门天然
生效；未接线方法**显式**请求 `ts` 仍 fail-loud；实验层不必理解各方法的损失、内部 CV、
trusted set 或全量/mini-batch 差异。上游 `resolve_training_view` 按台账原生假设解析默认视图，
无需改动。

**否决：实验层统一增广数据后喂给估计器。** 它会绕过"训练接口允许替换未标记损失输入"这条
门禁，把不同方法压成一套语义，且与 nnpu 的逐 mini-batch 语义冲突。

**否决：抽共享的并集 helper。** nnpu 的逐批拼接无法复用它，对全量求解类也仅省几行掩码，
收益薄。

> **2026-09-28 修订：部分重开，边界改小（见 §10）。** 上面否决的是「抽共享的**并集 helper**」，
> 收益口径是省代码行数。后续方案换掉了这个口径：核心层只做**角色与来源**——三个角色的身份、
> 行序、所有权契约与索引留痕——不替算法做并集，也**仍然不复用** nnpu 的逐批拼接（§10 与方案
> §5.2 明确要求设备内消费者维持内联构造）。因此本条的否决理由（nnpu 无法复用、全量类只省几行）
> **没有被推翻**，改的只是收益定位：面向尚未接入的方法，以及把「哪一行为何可写」的契约固定下来。
> 抽象的规模上限也一并记录在案——若参考消费证明核心层被迫理解算法公式，按方案 §6 阶段 D 的
> kill criterion 回退。

## 4. `upu` 切片（PR #74，squash 提交 `aeca474`）

### 4.1 实现

- **无标签风险集合**：OS 用 `X_U`，`ts` 用全训练集（即 `X_U ∪ X_P`），分母随之取
  `n_P + n_U`。正例项 `-(π/n_P)Σ_P g`、class prior `π`、正则项**不变**。全量求解下并集由
  "换池子 + 换分母"实现，不复制行、不改索引。
- **RBF**：中心**候选池**跟随视图；中心**数量**一律截断到**校准前**的 `n_U`（默认值与显式
  `n_centers` 同受此上限约束，且上限**不**取自候选池大小）。理由是 OS/TS 对照不得混入模型
  容量变化。
- **不加 `ts` × `sample_weight` 互斥门**（与 `nnpu` 不同）：uPU 的
  `sample_weight_support = IGNORED`，并集不产生"无权重定义的追加行"。该差异已写入
  `fit` docstring 与台账，否则复核人会问为何 nnpu 有门而 upu 没有。

### 4.2 证据（按归属分层）

| 层级 | 内容 |
|---|---|
| 独立数学 golden | squared 闭式解：OS 应得 `33/28`、TS 应得 `6/11`。期望值由测试用原始数组**独立重推**，不调用生产 `_phi` / `_fit_squared`；两个手算有理数是真正的独立锚点 |
| 结构测试 | 三条 solver 路径 × 两视图的池大小与分母；`n_U = 1` 边界；RBF **池身份**（monkeypatch `subsample_centers` 直接观测其收到的池参数，**种子无关**）；中心数不变性（`n_centers = 4` 跨在 `n_U` 与 `n_total` 之间才具判别力）；默认 == 显式 `os`；非法视图值 fail-loud |
| 真实跑批 | spambase `split_0`、`c = 0.1`、`π = 0.39404477287546186`；产物在仓库外 `F:/Temp/lab/P2.0e/upu_os`、`upu_ts` |

真实跑批实测（同单元两跑）：

| 指标 | OS | TS |
|---|---|---|
| `run_view` / `calibration_applied` | `os-compatible` / `False` | `ts-compatible` / `True` |
| `formal_blockers` | 不含 ts 阻断位 | **含** `ts_view_collaborator_review` |
| test PA accuracy | 0.8382193268186754 | 0.8686210640608035 |
| test PA AUC | 0.9215468467667881 | 0.9245534524126899 |
| test OA accuracy | 0.8545059717698155 | 0.8599348534201955 |

**边界**：真实指标不同只证明实现路径非空转；**数值正确性以独立 golden 为准**，不以真实跑批为准。

一处已知巧合：TS 跑的 PA `val_proxy_accuracy` = `1.4674373718378804` 与早先 `nnpu_os` 跑逐位
相同。已验算为离散网格撞值（恰等于 `2π + 125/184`，184 为验证集大小），**非别名、非产物复用**：
两次产物的 `execution_unit.method` 均为 `upu`、`estimator_parameters` 一致，但 `val_score_min`
与 `val_accuracy` 不同。

### 4.3 过程中修正的"记录与实现不符"

- **中心数不变量曾经为假。** 原 `min(200, n_U)` 只在**默认分支**生效，显式 `n_centers > n_U`
  时中心数随视图变化（实测 `n_U = 3`、`n_total = 5` 下，`n_centers = 4` 得 OS 3 / TS 4），
  而方法卡、台账与 docstring 已声称"不随视图变化"。处置：**改代码让记录成真**（显式值同受
  `n_U` 上限），并以可判别测试锁定。
- **决策 D16 ② 曾经过度推广。** 原以"统一规则"措辞把 uPU 的处置写成对 `ts` 视图的**全局规则**，
  与协议/spec 的"仅约束 uPU、不自动推广到其他方法"边界冲突。危险在于载体不对等：设计稿在
  `docs/superpowers/` 是会被删除的工作稿，D 表是版本控制的权威文档——宽措辞会存活并静默压过
  窄范围。已收窄为「对 `upu` …该结论不自动推广到其他方法」。
- **通用工具层曾掺入 Survey 语义。** `pu_toolbox/utils/basis.py` 的 `subsample_centers` /
  `resolve_basis_fn` 参数文档一度以 OS/TS 视图定义 `X_pool`；该模块是各估计器共用的通用实现。
  已恢复通用描述，视图解读留在 `UPUClassifier.fit`。

## 5. `pusb_kernel` 切片（PR #76，squash 提交 `0b16815`）

### 5.1 实现

- **校准范围**：`ts` 只作用于**内部 CV 训练折**与**最终 refit**，无标签风险项的经验分布由 `D_U`
  换成 `D_U ∪ D_P`（分母随之取 `n_P + n_U`）。正例项、class prior、正则项不变。
- **验证折保持 OS**：内部 CV 的验证折两个角色由该折自己的 P/U 就地构造，**不经**视图开关。这是
  协议驱动的**跨目标选参**决策（D17）——用 OS 目标的验证值去选 TS 训练目标的超参数；被否决的
  替代（验证折并入该折自己的 P，即 `U_val ∪ P_val`，同样 held-out 且在「选参准则与训练风险
  同形」上更自洽）与否决理由见 D17。
- **容量与池子不变**：RBF 中心**候选池**本已取自完整 `X`（不是 `X_U`），故**不**照搬 upu 的
  「中心池跟随视图」——照搬会人为抬高 P 被选为中心的权重。中心数量、CV fold、先验分位数阈值池
  一概不随视图变化。
- **接口**：`fit(..., os_or_ts="os")`；角色构造 helper `_risk_role_designs` 只接收显式布尔
  `include_positive_in_unlabeled`，**不接收** run 级视图字符串，且验证折**不经**它——「验证保持
  OS」在代码结构上 fail-closed，而非靠调用者记得传 `"os"`。

### 5.2 证据（按归属分层）

| 层级 | 内容 |
|---|---|
| 独立 golden | OS/TS objective 与 gradient 的字面量期望值，由独立脚本用原始数组重推（**不调用**生产 helper）；TS 侧显式体现 P 同时进正项与无标签项 |
| 公式无关锚 | 两视图各做有限差分梯度检查；另有一条 mean-invariance 性质测试（把分母写死会在重复行上被抓住） |
| OS 冻结 golden | 改造前从 HEAD（`7c716f9`）取样 `coef_`/`cv_scores_`/`threshold_`/中心索引/fold 并固化：重构后逐位未变。离散量（`sigma_`/`reg_lambda_`/中心索引/fold）精确相等，连续量给容差（BFGS 末位可随 BLAS/SciPy 版本漂移，取样环境 numpy 2.4.6 / scipy 1.17.1 已记入测试） |
| 结构测试 | 角色 spy（布尔序列、design 行数）、每 fold × 每 `(σ, λ)` 的验证折反向测试、阈值池大小、中心/fold/容量不变、**角色重合但物理样本不复制** |
| 变异检验 | 对四个不变量各注入一处缺陷，确认确有测试翻红（复制行入并集 / 验证折注入并集 / 阈值池扩容 / 中心池跟随视图），每次运行后按哈希还原生产文件 |
| 真实跑批 | 已完成，见 §5.3（视图/标志/阻断位/容量不变 + 路径非空转；不含也无法含估计器内部选择） |

### 5.3 真实跑批

Spambase `split_0`、`c=0.1`、`π=0.39404477287546186`（与 nnpu/upu 审计参数一致）。命令：

```bash
uv run python scripts/run_survey_experiment.py data/splits/spambase/split_0 --protocol survey-v1.2 --dataset spambase --method pusb_kernel --os-or-ts os --c 0.1 --seeds 0 --class-prior 0.39404477287546186 --out-dir F:/Temp/lab/P2.0e/pusb_kernel_os
uv run python scripts/run_survey_experiment.py data/splits/spambase/split_0 --protocol survey-v1.2 --dataset spambase --method pusb_kernel --os-or-ts ts --c 0.1 --seeds 0 --class-prior 0.39404477287546186 --out-dir F:/Temp/lab/P2.0e/pusb_kernel_ts
```

产物：`F:/Temp/lab/P2.0e/pusb_kernel_os/os/c_0.1/seed_0/` 与
`pusb_kernel_ts/ts/c_0.1/seed_0/`（层级与前两切片一致）。跑批前用**同一串 flag 跑 upu** 做过
命令形状核对：产物层级逐层相同，且 upu 的 PA `0.8382193268186754` / OA `0.8545059717698155`
与 §4.2 所记逐位相同——两次切片的命令逐字可比。

实测（同单元两跑，`execution_mode = versioned_pilot`）：

| 指标 | OS | TS |
|---|---|---|
| `run_view` / `calibration_applied` | `os-compatible` / `False` | `ts-compatible` / `True` |
| `formal_blockers` | 不含 ts 阻断位 | **含** `ts_view_collaborator_review` |
| `estimator_parameters` | `n_basis=300, cv=5, max_iter=200, tol=1e-5, random_state=0` + 官方 σ/λ 网格 | **逐字相同** |
| test PA accuracy | 0.8219326818675353 | 0.8827361563517915 |
| test PA AUC | 0.9538024428053754 | 0.9545676708433306 |
| test OA accuracy | 0.8805646036916395 | 0.8892508143322475 |
| test OA AUC | 0.9538024428053754 | 0.9545676708433306 |
| `selection.OA` 的 `val_score_min` / `val_score_scale` | -4.2203822834906415 / 9.672616291594961 | -4.038635261279397 / 8.604982562375582 |
| `elapsed` | 57.30 s | 41.61 s |

**产物级可证的两件事**：① 视图与校准标志被正确记录，且 TS 侧确实多挂 `ts_view_collaborator_review`
（OS 侧没有）；② 两视图的 `estimator_parameters` 逐字一致而 scores 分布统计
（`val_score_min`/`val_score_scale`）不同——即**校准改的是风险项、不是模型容量**，且路径非空转。

**本表核对项之外的三点边界**（避免被读成已验证）：

1. **真实指标不同只证明路径非空转**，数值正确性以 §5.2 的独立 golden 与变异检验为准。
2. **manifest 不承载估计器内部的 `(σ, λ)`、中心数与冻结 `threshold_`**：`candidate_runs[].params`
   为空对象、`selection.threshold` 是**选择协议**用于把 scores 转标签的阈值候选（PA 侧 OS 0.6 /
   TS 0.4、OA 侧两者均 0.5），**不是**估计器 `fit` 内冻结的训练先验分位数阈值。设计稿 §9 把
   "核对中心数、选中超参数、threshold"列为跑批核对项，**超出了产物的承载能力**——要核对它们须
   另外输出审计字段。本地重拟合**不**作为替代：跑批的 PU 标签由 runner 用 SCAR 生成器按 `c=0.1`
   现场生成（切分制品存的是真实标签），复现它需逐字重放生成器与种子推导，差一点就会给出看不出
   错的数字。
3. `selection.*.metrics.val_proxy_accuracy` 大于 1（OS 1.3184 / TS 1.3316）是该 proxy 指标的
   已知形态（离散网格撞值，语义见 §4.2 对 upu 同类值的验算），**不得**当作通过率读。

**边界**：真实跑批只能证明「实际视图与校准标志被正确记录」和「路径非空转」；它**证明不了**内部
验证折未被校准——后者由结构测试与变异检验负责（§5.2 前四层），不得写进跑批核对项。

## 6. `dist_pu` 切片（PR #77，squash 提交 `8e1efbe`）

### 6.1 实现

`fit` 增加 keyword-only `os_or_ts`，并把两个分布正则项抽成模块级
`_distribution_regularizers(probs, unlabeled_mask, class_prior, *, include_positive_in_unlabeled)`：

- **角色 helper 只接收显式布尔**，不接收 run 级视图字符串——与 `pusb_kernel` 同一条 fail-closed
  原则（D17 ④）。但本方法**没有内部验证折**：`fit` 无 `validation_data`，验证/测试由 experiment 层
  在训练后消费，因此不需要「验证折就地构造」那一半，也**不存在** `pusb_kernel` 的「跨目标选参」问题。
- helper **不在模块导入期引入 `torch`**（可选依赖的边界保持原样），只用 tensor 方法与掩码索引。
- 数值表达式**逐字符复刻**原目标，含 `log` 内的 `1e-6`。它**不是**数值守卫——真正的兜底是 `fit` 里
  对 logits 的 `clamp(-10, 10)`（概率因此被夹在 `[4.5e-5, 0.99995]`）。方法卡里把它称作 epsilon
  守卫的旧措辞已一并修正，免得后来者「顺手清理掉没用的 epsilon」。
- `os_or_ts` 的合法性校验放在 `torch.manual_seed` **之前**：非法视图不得扰动全局随机状态。
- 校准只换角色集合：两个正则项消费的是**同一个前向输出**，正例 BCE、class prior、Mixup 池、
  epoch、callback 与 checkpoint 契约一概不动。

### 6.2 证据（按归属分层）

| 层级 | 内容 |
|---|---|
| OS 冻结基线 | 改造前（`b7d42f2`）在固定 CPU / seed / 小模型下采集 `loss_history_` 全序列、4 个参数张量、探针 `decision_function` 与 callback epoch 序列，两组（`mixup_weight=0` 与 `0.3`）。采集环境 Python 3.11.15 / torch 2.14.0+cu126 / numpy 2.4.6，**两次运行逐字节一致** |
| 改造等价性 | 重构后 OS 路径的 `loss_history_` 与改造前**逐位相等**——比容差断言更硬，且独立于冻结基线的容差松紧 |
| 独立数学期望 | alignment/entropy 在固定概率与掩码下的期望值由**独立 numpy 算式**算出后写成字面量；测试**不用生产 helper** 生成期望值 |
| 结构不变量 | 角色集合在 **`fit` 实际使用它的那一层**被钉住（正则项输入张量的行数 == `len(X)`，而非 `len(X)+n_P`）；两个正则权重置零时两视图**逐位同轨**；Mixup 池长度、RNG 调用顺序与形状、callback 序列、训练矩阵形状、`sample_weight` 被忽略 |
| 变异检验 | 五处注入**全部被抓住**：TS alignment 仍旧只用 U／TS entropy 仍旧只用 U／目标改由观测 P/U 比例推／把 P 重复两次计入 marginal／OS 误用完整 X。每次运行后按哈希还原生产文件 |
| 真实跑批 | 见 §6.3 |

**一处必须记录的观测**：变异检验**第一轮漏掉了第 4 项**（把 P 重复两次计入 marginal）。原因是那个
缺陷注入在 `fit` 的**调用点**，而当时的数学测试直接调用 helper，看不见它。结构不变量那一层因此
才被补出来，并写进了测试文件的 docstring——**没有变异检验，「测试齐全」只是自评**。

### 6.3 真实跑批

Spambase `split_0`、`c=0.1`、`π=0.39404477287546186`（与前三切片同参数）。命令：

```bash
uv run python scripts/run_survey_experiment.py data/splits/spambase/split_0 --protocol survey-v1.2 --dataset spambase --method dist_pu --os-or-ts os --c 0.1 --seeds 0 --class-prior 0.39404477287546186 --out-dir F:/Temp/lab/P2.0e/dist_pu_os
uv run python scripts/run_survey_experiment.py data/splits/spambase/split_0 --protocol survey-v1.2 --dataset spambase --method dist_pu --os-or-ts ts --c 0.1 --seeds 0 --class-prior 0.39404477287546186 --out-dir F:/Temp/lab/P2.0e/dist_pu_ts
```

产物：`F:/Temp/lab/P2.0e/dist_pu_os/os/c_0.1/seed_0/` 与 `dist_pu_ts/ts/c_0.1/seed_0/`
（层级与前两切片逐层相同）。

| 指标 | OS | TS |
|---|---|---|
| `run_view` / `calibration_applied` | `os-compatible` / `False` | `ts-compatible` / `True` |
| `estimator_parameters` | `alignment_weight=1.0`、`entropy_weight=0.05`、`mixup_weight=0.1`、`epochs=200`、`hidden_dim=128`、`lr=1e-3`、`random_state=0`、`device=cpu` | **逐字相同** |
| `formal_blockers` | 3 项（linux 环境偏离、PA 准则待复核、协议偏离） | **4 项**，多出 `ts_view_collaborator_review` |
| PA accuracy / AUC | 0.7937024972855592 / 0.9146301726946888 | 0.7937024972855592 / 0.9145314335930171 |
| OA accuracy / AUC | 0.8990228013029316 / 0.9542048046446874 | 0.8990228013029316 / 0.9545948240962904 |

**「路径非空转」需要独立证据**：单看指标差异只有 AUC 第 4 位，很容易被读成「视图根本没生效」。用
manifest 里的**同一份 `estimator_parameters`** 直接把两个视图各跑一次，`loss_history_` 第 1 轮即
分叉、末轮相对差 **5.80%**（0.08046819 vs 0.08513858，序列非逐位相同）。故指标差异小是 AUC 与
阈值决策稳健所致，**不是**视图未生效。

**边界（不得读作结论）**：单 seed 的 PA/OA 差**不构成**「校准更优」或「校准更差」的证据。本表只
证明四件事：①实际视图与校准标志被正确记录；②blocker 按协议挂上；③估计器内部参数与模型容量
不随视图变化；④两视图产物确实不同。内部角色集合的正确性由 §6.2 的结构不变量与变异检验负责——
manifest **不承载**估计器内部的 `loss_history_` 与选择细节（`candidate_runs[].params` 为空对象）。

## 7. `self_pu` 切片（本 PR）

### 7.1 实现

- `pu_toolbox/estimators/deep/self_pu.py`：新增模块级
  `_marginal_negative_risk(unlabeled_negative, positive_negative_losses, *, n_unlabeled_role, include_positive_in_unlabeled)`
  ——**唯一**的角色咽喉点，只接收显式布尔（与 D17 ④ 同形）；OS 时**原样返回输入**（逐位不变）。
- `fit` 增加 keyword-only `os_or_ts: str = "os"`；非法值在 `torch.manual_seed` **之前** fail-loud
  （与 `dist_pu` 同址：非法视图不得扰动全局随机状态）；`sample_weight` 的整体拒绝不变。
- 正例批的 negative losses 由已算出的 `positive_logits` 导出一次，同时供校正项 `R_p^-` 与并集侧
  → **0 次额外前向、0 次额外 RNG**（由计数 backbone 与 `RandomState` 记录子类锁定）。
- 校准范围**只有一处**：负 PU 项的角色集合由「未信任 U 行」换成「未信任 U 行 ∪ 正例批」，按两批
  **实际行数**混合。trusted 人口与 pace、伪标签、meta 矩阵两列维度、consistency（student 用未信任
  子集、teacher 用整批 U）、验证/选模（含无 validation 时的 ablation 分支）与 checkpoint 一概不动。

### 7.2 调查结论的修正：当初判定的「最硬障碍」在结构上不存在

§2 表在接线前把 `self_pu` 记为最硬——trusted-set 按分数取「最低分为负半」，并集带进的已知正例会
拿到伪负标签。实施结论是**该风险不成立**：`TrustedSetManager` 只接收形状严格为 `(n_U,)` 的概率
（不匹配即 fail-loud），`manager.indices` 恒为 U-local，映射回全局后与 `positive_global` 由
`flatnonzero` 天然互斥。故身份不变量是**结构性保证**，本切片加的是**回归锁**（含「已知正例概率
全局最低」的反例构造）而非补漏；也**没有**在生产热循环里加运行期断言（200 epoch × 2 student ×
2 视图会跑 400 次），只由形状校验与测试钉住。

### 7.3 证据（按归属分层）

| 层级 | 内容 |
|---|---|
| OS 冻结基线（实现方自动复现） | 改造前在固定 CPU/seed/小模型下采集**两套配置**（A 消融＝Pilot 口径、B clean-meta＝库口径）的 student/teacher state_dict、四类 history、best teacher 与 decision scores，写成 `test_self_pu_ts_view.py` 的 `_GOLDEN`；常量由一次性脚本生成，**无手抄** |
| 改造等价性（实现方自动复现） | 改造后四组跑（A/B × 默认/显式 `os`）与 golden 逐值比对，**最大绝对偏差 0.0**（逐位未变）；测试本身按 `rtol=1e-6` 断言，给未来无关的浮点重排留余量 |
| 独立数学期望 | 混合语义由**手写表达式**给出（含 `np.r_[np.full(n_u_role, u), positives].mean()` 这一显式并集数组），**不调用生产 helper**：均匀权重下混合 == 并集经验均值（Gate A）；非均匀下 == 质量混合且**不等于**并集均值；OS 返回输入本身 |
| 结构不变量 | 负角色行数**精确** == `n_U − 该 epoch trusted 规模`（配置取整批 U，使该式不含 RNG 假设）；校准标志只出现在训练装配处，且 helper 调用次数恰为 2×epochs（含无 validation 的消融分支）；meta 两列行数 == 未信任行数（**只有 clean-meta 分支可达**）；trusted 人口与 pace 不随视图变化；RNG 抽样序列两视图逐位相同；前向调用序列两视图相同；`α_U` 施加位置不改变 meta 权重 |
| 变异检验 | §7.4 |
| 真实跑批 | §7.5 |
| **复核人待办** | 以上全部为**实现方自证**；`ts` 路径的方法学复核**尚未获得**（D16 ④ / D19 ⑨） |

**第一轮 CI 暴露的「冻结方式」错误（必须记录）**：CI 报
`golden.valid.val_teacher[1]`（第 2 epoch 选中的 teacher）本地为 1、CI 为 2。根因**不是实现漂移**，
而是第一版 golden 冻结了一个**由噪声级裕度决定的选择**：实测该配置四个 epoch 的两个 teacher
PU 验证风险差为 **0 / 0 / 2.4e-7 / 6e-8**（float32 噪声量级），`argmin` 取谁随环境的浮点末位而变——
冻结它的取值等于冻结环境。修正：

- 两个**排序导出**的选择（`val_teacher` 序列、`best` 的 teacher 索引与 epoch）改为
  「**按本 run 自身的风险走规则** + 冻结裕度决定性时才要求相等」，并列容差 `_TIE_TOL = 1e-4`
  （高于噪声三个数量级、低于 trusted 边界裕度一个数量级）；`best` 的两条选模依据（PU 验证风险
  argmin / clean 验证指标 argmax）各自的规则也被显式钉住；
- **trusted 成员仍逐值冻结**：同一 run 上的边界裕度为 **1.4e-3 ~ 1.6e-2**，比噪声高四个数量级，
  第一轮 CI 也确实通过了它们；
- 新增 `test_param_a_tie_broken_the_other_way_is_still_accepted` 复现该 CI 观测（把第 2、4 epoch
  的并列破向另一边）：**把容差置 0 该用例即失败**，证明容差是承重的，不是为了让测试变绿。

### 7.4 变异检验（隔离方案，不改写生产文件）

九处注入**全部被抓住**；收尾三项校验通过（`git diff --check` 空、`git status --short` 空、
生产文件 sha256 与开始前一致）。**没有**改写工作树里的生产文件：补丁打在**内存副本**上、写入
临时文件，再由一次性 pytest 插件在收集前替换模块。脚本在**工作区不干净时 fail-closed 拒跑**
（第一次运行正是被这条挡住——这条守卫比「事后还原」强，因为它不依赖 `finally` 被执行）。

| # | 注入的缺陷 | 抓住它的测试（红灯数） |
|---|---|---|
| M1 | 已知正例进入 manager 人口（概率与索引空间一起改为 P∪U） | 身份反例 `test_edge_a_known_positive_ranked_lowest_never_becomes_trusted`（9） |
| M2 | `unlabeled_global` 直接含正例 | 同上身份反例（13） |
| M3 | TS 仍只用 U 的负项（调用点把标志写死 `False`） | `test_basic_the_ts_view_fits_and_moves_the_solution` + 校准标志守卫（4） |
| M4 | 正例行进入 meta 两列 | 28 条（含新增的 meta 行数测试，但见下方边界 2） |
| M5 | 混合改用固定 1/2 | 数学字面量 4 条（`!= fixed_half` 那条）（4） |
| M6 | 混合分母误用整批 U 行数 | **仅**精确角色行数不变量 2 条（2） |
| M7 | 验证风险也被校准（PU 验证的 marginal 含正例） | `test_basic_pu_validation_tracks_and_restores_best_teacher`（3） |
| M8 | teacher consistency 扩进正例批 | OS 冻结基线 4 条（训练历史逐值变化）（4） |
| M9 | OS 默认路径被校准（调用点写死 `True`） | OS 冻结基线 4 条 + TS 行为 2 条（7） |

**两处必须记录的边界**：

1. **M6 只有 2 条测试能抓**——正是为它写的精确角色行数不变量。若没有这一条，缺陷会**静默存活**
   （它只改变混合的质量权重，不动任何形状或标志）。这与 `dist_pu` 切片 M4 的教训同源：
   **没有变异检验，「测试齐全」只是自评**。
2. **M4 的抓住方式是下游形状报错，而非那条专门的 meta 行数断言**：把正例行塞进两列后，先炸的是
   `hard_distillation_loss` 的对齐检查。该断言目前是**结构性锁**（鉴别力由「未信任行数 24–28 <
   批大小 30」体现，与 M6 同源），还没有被一个隔离变异体单独证明过——不得在文档里写成「由变异
   检验证明」。

### 7.5 真实跑批

Spambase `split_0`、`c=0.1`、`π=0.39404477287546186`（与前四切片同参数）。命令（两视图各一次）：

```bash
uv run python scripts/run_survey_experiment.py data/splits/spambase/split_0 --method self_pu --dataset spambase --training-path native_2d --protocol survey-v1.2 --c 0.1 --seeds 0 --class-prior 0.39404477287546186 --os-or-ts os --out-dir F:/Temp/lab/P2.0e/self_pu_os
uv run python scripts/run_survey_experiment.py data/splits/spambase/split_0 --method self_pu --dataset spambase --training-path native_2d --protocol survey-v1.2 --c 0.1 --seeds 0 --class-prior 0.39404477287546186 --os-or-ts ts --out-dir F:/Temp/lab/P2.0e/self_pu_ts
```

产物：`self_pu_os/os/c_0.1/seed_0/` 与 `self_pu_ts/ts/c_0.1/seed_0/`（层级同前三切片），
各 **400 个 checkpoint**（200 epoch × `teacher_1` / `teacher_2`；组件名与 `epoch_components` 一致）。

| 指标 | OS | TS |
|---|---|---|
| `run_view` / `calibration_applied` | `os-compatible` / `False` | `ts-compatible` / `True` |
| `estimator_parameters` | 22 键（`hidden_dim=128`、`max_epochs=200`、`batch_size=256`、`warmup=10`、`self_paced 10→50`、`distill_start=50`、`pace=0.2/0.3`、`random_state=0`…） | **逐字相同** |
| `formal_blockers` | 4 项 | **5 项**，多出 `ts_view_collaborator_review` |
| PA accuracy / AUC | 0.9022801302931596 / 0.9650512949633184 | 0.9131378935939196 / 0.9642564451948615 |
| OA accuracy / AUC | 0.9131378935939196 / 0.9624248348588524 | 0.9022801302931596 / 0.9635356497526586 |

OS 与 TS 的 PA/OA **accuracy 恰好互换了取值**（0.9022… 与 0.9131… 在两视图间换位）。这是一个巧合，
**不得**读成「视图把 PA/OA 调换」——两条跑的选模与报告路径都相同，换位的只是两个数。

**两个视图确实各自训练出了不同的模型**：两份制品最后一个 epoch 的 `teacher_1` 权重最大绝对差
**0.08757619559764862**；重放审计里 trusted 集合含有的真实正例数也分别为 398 / 405。

**两个视图都落在消融分支**（D19 ⑧ 的制品级证据）：两条跑批日志各含一条
`UserWarning: validation_data was not supplied; running the explicit Self-PU ablation without meta
reweighting or clean-validation teacher selection.`；重放审计中 `meta_influence_calls = 0`。故
**Pilot 的 `self_pu` 不含 self-calibrated 元重加权**，含 meta 的分支只由 §7.3 的单元测试覆盖。
协议侧的 `formal_blockers` 本就带 `SelfPU_clean_validation_meta_reweighting_OA_integration`，与这条
边界一致。

> **补记（2026-09-28，决策 D24）**：该阻断位的运行时追加**已移除**，「阻断正式资格」不再生效；
> 依据是当日裁决「现有消融变体纳入 pilot 范围」。**放行不等于合作者已复核**——上方关于消融边界的
> 陈述仍然有效，manifest 的 `method_variant` 仍如实写入
> `without_clean_validation_meta_reweighting`。本句为补记，上方原文按历史记录保留、**不改写**。

#### 7.5.1 重放身份审计

制品**不落盘** trusted 成员、meta 行数或 consistency 行数，所以身份证据来自**重放审计**：从
manifest 取参数/seed/视图/split 引用 → 重建视图并校验摘要 → 走同一 trainer 调用路径重放 →
**逐位比对制品最后一个 epoch 的两个 teacher checkpoint** → 通过后才写诊断。产物
`F:/Temp/lab/P2.0e/self_pu_identity_audit.json`（脚本一次性，不入库；JSON 与本节摘要才是留存物）。

**重现校验（审计成立的前提）**：两视图 × 两 teacher 全部 `bitwise = true`、`max_abs_delta = 0.0`
（对 `epoch_0200_teacher_{1,2}.pt`）。输入同源：四个角色的 `feature_sha256` 全部匹配，
`split_sha256` 重算一致。

| 审计项 | OS | TS |
|---|---|---|
| 已揭示正例 / 真实正例 / U 行数 | 130 / 1305 / 3182 | 同 |
| U 中的隐藏正例 | 1175 | 同 |
| `manager` 人口（两次构造） | `[3182, 3182]` | `[3182, 3182]` |
| **已揭示正例 ∩ trusted** | **0** | **0** |
| trusted 中的真实正例数（观测，**非**不变量） | 398 | 405 |
| trusted 最终规模（student 1 / 2） | 636 / 794 | 636 / 794 |
| `meta_influence_calls` / rows | 0 / `[]` | 0 / `[]` |
| consistency（student）调用数 / 每对行数 | 302 / 两列逐对相等 | 302 / 两列逐对相等 |
| teacher consistency 行数（由 `min(batch_size, n_U)` 推得，非观测） | 256 | 256 |
| `calibration_mode_` / 选模依据 | `ablation` / `pu_validation_nnpu_risk` | 同 |
| Gate C（审计进程内构造的反例） | trusted 规模 6、交集 **0** | 同 |

**一处必须写明的边界**：审计初版把「正例」取成 split 里的**真实标签**，于是与 1305 个真实正例求交
得到 398/405 的「交集」——那是**隐藏正例被自我步进机制伪标注**，是 PU 学习在正常工作，**不是**身份
泄漏。守卫对象是**已揭示正例**（估计器被告知的那 130 个），它与 trusted 的交集必须为 0。两个数字都
留在审计 JSON 里并各自标注含义，不得混用。

**边界（不得读作结论）**：与前三切片相同——单 seed 的 PA/OA 差**不构成**「校准更优/更差」的证据。
本节只证明：①实际视图与校准标志被正确记录；②blocker 按协议挂上；③估计器内部参数与模型容量不随
视图变化；④两视图确实训练出不同模型且都在消融分支；⑤身份不变量在真实制品上成立（经逐位重现的重放）。

#### 7.5.2 CIFAR adapter 技术 smoke

`cifar10/split_0`（`π=0.4`、`c=0.1`）、`--training-path cnn_feature_adapter`、两视图各一次：

| 检查项 | OS | TS |
|---|---|---|
| `run_view` / `calibration_applied` | `os-compatible` / `False` | `ts-compatible` / `True` |
| `training_path` / `budget` | `cnn_feature_adapter` / `two_student_sampled` | 同 |
| `estimator_parameters` | — | **与 OS 逐字相同** |
| `formal_blockers` | 4 项 | **5 项**（多 `ts_view_collaborator_review`） |
| checkpoint 数 / 组件 | 400（`teacher_1`、`teacher_2`） | 400（同） |

**adapter 路径的 hook 转发成立**：两视图都完成了 200 epoch × 2 student 的完整训练并写出 400 个
双 teacher checkpoint，说明 `os_or_ts` 经 `EpochCheckpointTrainer` → `DeepFitTrainer` 一路到达
`self_pu`（与 §7.3 的 native_2d 路径同一套转发，但这条走 4D 图像 → 适配器特征）。

**smoke 指标无意义，不得引用**：adapter 的 encoder 是**随机初始化**的
（`backbone.specs.image.weights = None`），故 PA/OA accuracy ≈ 0.5997、AUC ≈ 0.53（近随机）是
构造使然，只说明管线跑通，不说明该方法在此数据上的表现。adapter 特征缓存按
「dataset/split/seed」共享（与 method/c/mechanism 无关，见 manifest 的
`adapter_shared_scope`），故第二次跑复用同一条缓存（两次跑后缓存仍为 1 条）。

## 8. 操作后果（供 P2.1 排期与 P2.2 聚合）

- 本切片后 `upu` 的 pilot **默认视图为 `ts`**，故其默认行挂 `ts_view_collaborator_review`、
  正式不合格——与 `nnpu` 自 #68 起的处境相同。且 `--os-or-ts` 是**整矩阵**参数（D14），
  无法只给 `upu` 单开 OS。
- **RBF 那半改动对 pilot 结果零影响**：pilot 的 `upu` profile 是
  `linear` / `squared` / `fit_intercept = false`（`survey_protocol_v1.json`），不走 RBF 分支。
- 本切片后 `pusb_kernel` 的 pilot **默认视图同样翻为 `ts`**（其台账原生假设本就是 `ts`，接线后
  resolver 即返回 `ts`），默认行从此挂 `ts_view_collaborator_review`、正式不合格——与 `upu`/
  `nnpu` 同处境，且同样因 `--os-or-ts` 是整矩阵参数（D14）而无法只给它单开 OS。
- 与 upu 那半改动不同，**本切片会实际改变 `pusb_kernel` 的 pilot 结果**：它本身就在 Pilot 矩阵内，
  且 profile 走的就是 RBF 分支（`n_basis=300`、`cv=5`、官方 σ/λ 网格）。
- 本切片后 `dist_pu` 的 pilot **默认视图同样翻为 `ts`**，默认行挂 `ts_view_collaborator_review`、
  正式不合格——与 `nnpu`/`upu`/`pusb_kernel` 同处境，同样因 `--os-or-ts` 是整矩阵参数（D14）而
  无法只给它单开 OS。它**在 Pilot 矩阵内**，故本切片同样会实际改变其 pilot 结果。
- 现有 benchmark 产物（`benchmarks/assigned_methods/results/clean_room_multiseed/`）中的 `dist_pu`
  **全部是校准前的 OS 视图**（`official/dist_pu.json` 状态为 `locked_not_executed`）；引用那些数字
  时必须标注视图，不得与接线后的 ts 跑混算。
- 本切片后 `self_pu` 的 pilot **默认视图同样翻为 `ts`**（台账原生假设本就是 `ts`），默认行挂
  `ts_view_collaborator_review`、正式不合格——与 `nnpu`/`upu`/`pusb_kernel`/`dist_pu` 同处境，
  同样因 `--os-or-ts` 是整矩阵参数（D14）而无法只给它单开 OS。它**在 Pilot 矩阵内**，故本切片会
  实际改变其 pilot 结果。**但改的是消融分支**：Pilot 不向 PU 方法传 clean validation（协议 §2.4），
  故 `self_pu` 的 pilot 行恒为 `calibration_mode_="ablation"`（含 meta 的分支只由单元测试覆盖，
  见 §7.1/§7.3 与 D19 ⑧）。
- 因此引用 `self_pu` 的 pilot 结果时必须同时标注**视图**与**分支**：它既不是「论文完整 Self-PU」，
  也不是「无校准的 Self-PU」。
- **D20 之后本节前几条的措辞需要按时间读**：上面若干条写「默认行挂
  `ts_view_collaborator_review`、正式不合格」，那是**放行前**的状态。D20 ① 已移除该阻断位的
  运行时追加，故此后新产生的 `ts` run 不再挂它；**本切片之前已产出的制品仍如实记录它们当时挂过**
  （§4–§7 的逐 run 表格不改写，它们描述的是那些 run 的 manifest 原样）。放行只解除「阻断正式
  资格」，**不构成**合作者对 `ts` 路径的方法学复核。
- 线性 `pusb` 经 2026-09-27 裁决为**不适用校准**（D16 ①），本切片未改其代码；D16 ① 完成口径中
  的「+ `pusb` 明确处置」由 **D20 ②** 落地。
  其台账 `uncertainty` 里仍留着「TS-OS 校准尚未接入训练执行链…P2 前须确认训练接口」的措辞
  （`method_ledger.json` 的 `pusb` 条目）——该措辞就 `pusb` 而言仍属实，但它读起来像「待办」，
  而裁决已把它定为「不适用」。是否把它改写为「不适用校准」，随 D16 ① 措辞一并另行裁决。

## 9. 未决项

- **`ts` 路径的合作者复核仍未获得**（D16 ④）：按 **D20 ①** 的单方技术验收放行，manifest 已
  **不再**挂 `ts_view_collaborator_review`（该阻断位的运行时追加已移除）；本记录**不构成**对 `ts`
  路径的方法学复核，**放行也不等于合作者已签署**——该口径保留在方法台账 `uncertainty`
  （`nnpu`/`upu`）与三张方法卡的视图条目中。
- **`self_pu` 的两处覆盖边界**（见 §7.4）：① M4 那类「正例行进入 meta」目前只被下游形状报错抓住，
  专门的 meta 行数断言还是结构性锁；② 无 validation 时的 ablation teacher selection **不经过**
  角色 helper，故只能用「helper 调用次数恰为 2×epochs」间接守卫，无法直接 spy 它的输入行数。
  两者都不影响本切片的结论，但若要更强的保证需要给该分支加可观测钩子——属额外工程量，未做。
- 建议（终审提出，尚未实施）：新增"台账 `calibration_applied` ↔ 代码钩子
  （`accepts_training_view`）"的**派生一致性契约测试**。今天该槽只是人工断言，而
  `native_sampling_assumption` 已与 registry scenario 耦合；这条守卫可机械地消灭本项目反复
  出现的"记录与实现不符"那一类问题。**本切片只走了半步**：`test_training_view_routing.py::TestResolutionForARealMethod`
  把 `pusb_kernel` 与 `dist_pu` 的「台账声明 ↔ 真实估计器类 ↔ resolver 默认值」三者手工配对钉住（参数化，不重复），仍**不是**
  覆盖全部方法的机械守卫。
- 本切片新增的 `resolve_training_view` 断言顺带暴露一个事实：台账的 `run_view` /
  `calibration_applied` 是**描述性**字段，resolver 只读 `native_sampling_assumption`；因此
  显式视图请求的正确性不受台账这两个字段影响，能守住「记录与实现一致」的只有上述手工配对的测试。
- `survey_execution_plan.md` 的 P2.0e 验收标准格仍写"`ts` 视图**逐训练 mini-batch** 执行
  `D_U^k ← D_U^k ∪ D_P^k`"——那是忠实复述协议 §2.3 原文（受摘要绑定的锁定要求文档），
  全量求解是该规则的**退化单批**情形。故**不改写**该格（改写会制造新的不一致）；
  "全量求解 vs 逐 mini-batch"的说明落在本文件 §2 与 §4.1。
- **台账的行号引用会静默腐烂，且本轮实测规模比预期大一个量级**：接线 `dist_pu` 前顺手核对，
  发现全台账 122 处 `卡.md:行号` / `registry:行号` 引用中有 **36 处失效**——`upu` 条目 10 处引用
  了一个**从未存在**的 `upu.md`（真名是按论文命名的 `Convex_Formulation_for_PU_DATA_Learning.md`）；
  `pusb`/`pusb_kernel`/`lbe`/`self_pu` 的 22 处 `registry:` 整体偏移 +38（2026-09-19 的 vpu 与 pulda
  两块插到 `pusb` 之上），其中数处指向**别的方法**的注册块、两处因「值碰巧相同」而肉眼不可见；
  `PUSB.md` 4 处偏移 +3。**没有任何门禁校验这些引用**，CI 全绿也没拦住；36 处已在
  `fix(ledger)` 提交中修正并复验为 0 失效（详见该提交）。
- **建议（仍未实施）**：给引用加机械门禁——卡文件必须存在且行号在范围内、`registry:NN` 必须落在
  **本条目的** registry 块内。这两条能永久拦住上述 36 处中的 32 处，实现成本约 40 行
  （本轮审计脚本的核心逻辑）。剩 4 处「行号在范围内但内容已漂移」只有把引用从行号换成 **§ 锚点**
  才能根治，要动全部 17 个条目的 122 处引用，属独立任务。**本轮决定暂不加门禁**，故记为遗留。
- **「原生 TS 但未接线」的测试槽位已改用测试内注册的合成方法**（方案 §5.7.1）：原先钉在 `dist_pu`
  上，接线后必然翻红（实测正是 `test_survey_pilot_driver.py` 的 2 个测试）。改用合成方法而非
  「挪到 `self_pu`」，是因为 P2.0e 的终态是 `self_pu` 也接线，届时真实矩阵里再无未接线方法，
  该槽位会**第二次**腐烂。合成槽位为永久保证，改完不再依赖任何真实方法的接线状态。

## 10. 训练视图分层：核心层与实验层（本 PR）

**背景。** §1-§7 为五个方法逐个接线。后续方法若继续各写各的角色构造，语义会漂移；但直接抽
一个共用 helper 已被 §3 否决过一次，且那条理由（nnpu 逐批拼接无法复用、全量类只省几行掩码）
成立。本 PR 采取的是**缩小边界**的做法：不抽并集 helper，而是把「角色与来源」下沉到一个中立层，
把「政策与历史」留在实验层。

**改了什么。**

| 层 | 文件 | 职责 |
|---|---|---|
| 核心层（新增） | `pu_toolbox/core/training_views.py` | `TrainingView` + `build_training_view`：从一次训练分区识别 `positive` / `native_unlabeled` / `loss_unlabeled` 三个角色与来源索引。不认识台账、方法名、manifest |
| 实验层（改） | `pu_toolbox/experiment/training_views.py` | 台账门禁（`_validate_options` 原样留在本层）、路由裁决 `resolve_training_view`、历史 `run_view` 词表在出口现算、manifest 构造；`calibrate_ts_os_batch` 改为边界适配器 |

**为什么必须分层。** estimator 是通用算法，不得静态依赖 survey 协议。实测：`import
pu_toolbox.experiment.training_views` 会经实验包 `__init__` 拉起 **28 个 estimator 模块**，而
`import pu_toolbox.estimators.risk.upu` 拉起的 experiment 模块数为 **0**。让 estimator 反向导入
实验层即破坏这条边界（倒置、污染与成环三者都不是想要的，按高风险处理）。依赖方向固定为
`estimators → core`，由 `tests/unit/core/test_training_view_contract.py` 的**子进程**导入边界测试
守住——必须子进程，同进程内只要先导入过 experiment 就再也测不出来。

**兼容面。** `calibrate_ts_os_batch` / `TSOSBatchView` 的公开字段、异常类型、错误信息与 manifest
字段**逐字保持不变**：`tests/unit/experiment/test_training_views.py` 的 6 条测试未作改动即通过。
值得注意的是这两个符号在生产代码中**没有调用方**，所以真正的兼容面是公开导出 + 这 6 条测试 +
`api.md` 条目，而不是调用点。

**所有权契约（本次新增）。** `frozen=True` 只禁止重绑属性，不阻止原地改写数组内容，故逐项写明：
三组 positions 与 `source_indices` **自有并冻结**（构造时复制后 `setflags(write=False)`）；
`source_features` / `source_labels` **借用**不复制（features 可能是整个训练集），调用方不得在视图
存活期间改写。冻结的是自有副本，**不得**作用于调用方传入的数组。

**未做（有意）。** 不迁移任何生产 estimator，`upu` 也不迁。核心 API 的可消费性由测试专用
reference consumer 证明；生产迁移推迟到第一个真实待接入算法——理由是本轮实测的角色构造重复量
每个方法仅一行到数行，拿已验收算法做迁移无法证明收益，只会引入重构风险。

## 11. 首个真实生产消费者：`vpu`（P3.1 局部前置）

§10 把生产迁移推迟到第一个真实待接入算法。**VPU 就是那一个**（P3.1 技术预集成中的方法，
采样假设先经审计裁决为原生 `ts`，见 [`vpu_sampling_audit.md`](../admission/vpu_sampling_audit.md) 与决策 D21）。
本节只登记这次接入改变了什么；VPU 的 P3.1 完整验收（共享 backbone、图像路径、公开数值对照、
多 seed GPU/资源记录、双人复核）**仍未完成**，VPU 也**不进入**冻结执行矩阵。

**接入形态。**

| 项 | 取值 | 说明 |
|---|---|---|
| `fit(os_or_ts=...)` | `"ts"`（默认） | VPU 的声明视图就是 ts，故默认值取 ts 而非其余五方法的 `"os"`——裸 `fit` 保持论文目标，不构成行为破坏 |
| positive 池 | `view.source_features[view.positive_positions]` | 两个视图下都是同一批标注正例 |
| marginal 池 | `view.source_features[np.sort(view.loss_unlabeled_positions)]` | ts 下还原为完整分区**自身的源顺序**，故固定 seed 下抽样轨迹与接线前逐位相同 |
| 新增审计字段 | `n_loss_unlabeled_` | 实际进入边缘池的行数；`n_positive_` / `n_unlabeled_` 语义不变 |
| 验证路径 | **不动** | 见 D22：验证变分风险定位为 source diagnostic，估计器只向核心层请求 `train` 角色 |

**两个池都由角色位置产生，不由源数组直接端走。** 这是本次接入的验证价值所在：若绕过
`*_positions`，公共构造器漏掉或多加入样本时估计器的测试仍会通过。以**隔离变异**验证（全程未改写
工作树里的生产文件，变异经 `-p` 插件在测试导入前生效）：把核心构造器的 ts 分支分别改成
「丢弃 P」与「重复 P」后，`tests/unit/estimators/test_vpu_ts_view.py` 分别有 2 条与 1 条测试转红。
**证据边界**：断言是行数级的——等长换行不在覆盖范围内（那一层由核心层契约测试负责），已在测试
模块 docstring 写明。

**冻结基线（隔离采集，全程未改写工作树）。** 把**接线前**的 `vpu.py`（`git show` 取出、以私有模块名
载入，其余模块仍取当前树）与接线后的版本在同一份数据、同一 seed、CPU 上各跑一遍，两个配置
（24 行 / 8 隐层 1 层 2 epoch；130 行 / 16 隐层 2 层 6 epoch）逐项对比：

| 量 | 裸默认 | 显式 `ts` |
|---|---|---|
| `decision_function` | 逐位相同 | 逐位相同 |
| `history_`（全序列） | 逐位相同 | 逐位相同 |
| `max_log_phi_` | 逐位相同 | 逐位相同 |

即**接线没有改变 ts 路径的任何输出**，裸 `fit` 的历史行为得以保持。同一脚本另确认显式 `os` 的结果
与基线**不同**——视图开关确实换了池，而不是只改了 manifest。

**路由规则变更（附带，决策 D23）。** `route_training_view` 原先只转发 `ts`（依据「只有校准请求会
改变训练」）。这条对默认值为 `"os"` 的方法成立，但 VPU 的默认值是 `"ts"`，于是显式 `os` 请求会被
丢弃、**静默跑成 ts**——正是该模块要防的错配，方向相反。新规则：`ts` 照旧（未声明则 fail-loud），
显式 `os` 在目标**具名**声明 `os_or_ts` 时转发、否则不动（这类目标本就是 OS）。非对称是刻意的：
未声明的 `ts` 是静默**降级**故拒绝，未声明的 `os` 是**无操作**。`None` 一律不转发，交由估计器自身
默认值决定。对既有五个方法行为等价（实测全 registry 只有 5 个估计器声明 `os_or_ts`、默认一律
`"os"`、无一在 `fit` 上收 `**kwargs`）；`**kwargs` 不视为 `os` 的声明。
