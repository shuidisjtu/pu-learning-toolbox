# P2.0e TS-OS 校准接入训练执行链（交付记录）

> 定位：本文件是 P2.0e 的交付记录，**随各方法接线逐次追加**。P2.0e 的整体状态与完成口径
> 仍以 [`survey_execution_plan.md`](survey_execution_plan.md) 的 P2.0e 行与决策 D16 为准；
> 数值与状态以方法台账 `pu_toolbox/experiment/method_ledger.json` 的
> `run_view` / `calibration_applied` 与各 run 的 manifest 为准，本文件只作解释。
>
> 当前进度：`nnpu`（#68）、`upu`（#74）已接线。剩余 `pusb_kernel`、`dist_pu`、`self_pu`
> 待各自独立设计；线性 `pusb` 待适用性裁决（见 D16 ①）。

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
| `pusb_kernel` | 全量 BFGS + (σ,λ) 网格 CV | `_pu_objective_and_gradient`（3 处调用） | ① CV **验证折**按协议须保持 OS，标志只能作用于训练折与终拟合；② 冻结训练先验分位数阈值会随视图漂移（属训练产物，非选择项） | 是 | 待独立设计 |
| `dist_pu` | 全量梯度下降（`batch_size` 为兼容性死参数） | 分布对齐项 | 对齐目标里的 `-π` 是"U ~ p(x)"的**显式编码**，并入 P 后须**重新推导目标**，不能机械拼接——属方法学判断 | 是 | 待独立设计 |
| `self_pu` | 每 epoch 各抽一次批的 SGD（无 DataLoader） | 5–6 处（未标记损失、缓存概率张量形状、`TrustedSetManager` 人口数） | 最硬：trusted-set 按分数取"最低分为负半"，并入的**已知正例会拿到伪负标签**；须先冻结"已知正例永不进入负半"的身份约束 | 是 | 待独立设计 |
| 线性 `pusb` | sklearn `LogisticRegression` 一次 `fit` | **无** | 训练信号即「U ≡ 负类」，不存在与风险估计器同形的"未标记损失输入" | 否（附加工程基线不入榜） | 待适用性裁决 |

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

## 5. 操作后果（供 P2.1 排期与 P2.2 聚合）

- 本切片后 `upu` 的 pilot **默认视图为 `ts`**，故其默认行挂 `ts_view_collaborator_review`、
  正式不合格——与 `nnpu` 自 #68 起的处境相同。且 `--os-or-ts` 是**整矩阵**参数（D14），
  无法只给 `upu` 单开 OS。
- **RBF 那半改动对 pilot 结果零影响**：pilot 的 `upu` profile 是
  `linear` / `squared` / `fit_intercept = false`（`survey_protocol_v1.json`），不走 RBF 分支。

## 6. 未决项

- **`ts` 路径的合作者复核仍未获得**（D16 ④）：每个 `ts` run 的 manifest 继续挂
  `ts_view_collaborator_review`，本记录**不构成**对 `ts` 路径的方法学复核。
- 建议（终审提出，尚未实施）：新增"台账 `calibration_applied` ↔ 代码钩子
  （`accepts_training_view`）"的**派生一致性契约测试**。今天该槽只是人工断言，而
  `native_sampling_assumption` 已与 registry scenario 耦合；这条守卫可机械地消灭本项目反复
  出现的"记录与实现不符"那一类问题。
- `survey_execution_plan.md` 的 P2.0e 验收标准格仍写"`ts` 视图**逐训练 mini-batch** 执行
  `D_U^k ← D_U^k ∪ D_P^k`"——那是忠实复述协议 §2.3 原文（受摘要绑定的锁定要求文档），
  全量求解是该规则的**退化单批**情形。故**不改写**该格（改写会制造新的不一致）；
  "全量求解 vs 逐 mini-batch"的说明落在本文件 §2 与 §4.1。
