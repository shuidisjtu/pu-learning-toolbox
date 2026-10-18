# P3.1 技术验收复核（2026-10-16）

## 结论与范围

**P3.1 暂不整体验收通过。** P3MIX 仍未交付完整 estimator；首轮另确认 PUET 的
TS + bootstrap 风险归一化缺陷。**2026-10-17 复核：#114 修复已独立验证通过，PUET
实现层面转为“有条件接受工程证据”，正式准入仍未完成。** 其余条目按下表列出技术证据及待决条件。
这是代理技术审核与验收建议；除明确标注“shuidisjtu 授权代录”的 PULDA 小节外，
不代表 shuidisjtu/HENG958 本人签署，不修改正式准入位。

审核基线：首轮 `92efb4dfbdf8b7a776ea87989aa1c54ed5fea8b6`；2026-10-17 同步至
`e84a76c`（`92efb4d..e84a76c` 共 8 个提交），后续复核以新基线为准，下列首轮结论保留为历史。
范围包括执行计划 P3.1、台账 12 条 pending_owner_review_methods、P3MIX，
以及与准入相关的既有 VPU 适配。八方法草稿和后增五方法草稿共同覆盖 13 个已注册方法，
其中 VPU 不在该 12 条待审列表中；不能把五方法证据包当作完整交付清单。

本轮读取交付记录、方法卡、registry/台账、真实实现及相关测试，并运行 CPU 专项。
未重新取得全部原论文及锁定上游源码，来源结论限于仓库现有证据与实现的一致性，
不宣称已完成原表读数、上游源码或论文数值的独立复核。
CNN 正式适配、资源 profile、配方冻结分别留给 P3.2、P3.3、P4.1。

## 2026-10-17 状态同步

`git fetch` 后当前分支快进到 `e84a76c`，未创建合并提交；工作区未跟踪的首轮审核文档保留。
冻结文件 `survey_protocol_v1.json`、`survey_comparison_v3.json`、`uv.lock`、`pyproject.toml`
在 `92efb4d..e84a76c` 内无差异。台账、方法卡、执行计划与 runner 有新增，PULDA estimator
及其证据文件未改。

新增 8 个提交：`53f0223` #114 PUET bootstrap 风险池归一化；`4bf9e04` #115 LaGAM
发布采样定位改为 OS；`84627d8` #116/#117 PULNS 额外 support 与 CVIR 先验/路径；
`0fcf77d` PUET 源绑定与 CVIR 真实先验校验；`7891e0e` 整改验收记录；`2a89175` LBE
分支对称/收敛诊断；`3efe1ad` Self-PU/LaGAM 独立 support 接线；`e84a76c` 算法收尾验收
与 fail-closed 预算测试。

独立复核进度：已在 `e84a76c` 重跑 #114 新增回归
`test_edge_bootstrap_root_masses_and_constant_leaf`，JUnit 记录 36 tests / 0 failures /
0 errors；其代码改动与首轮复现方向一致。`#115`–`#117`、PULNS/CVIR/Self-PU 新适配器及
队友自报的 4030/4056 项全量通过尚未由本轮独立重跑。P3MIX 仍未交付完整 estimator，
P3.1 仍未整体验收。后续 PULDA 审核在新基线上继续；旧行号需按 `92595ca` 重新定位。
2026-10-17：PUET、LaGAM、CVIR 与 PULNS 完整复核已完成并计入下文；**#114、#115、#117
已按用户指示关闭；#116 已按用户指示关闭（接口/审计交付，实际 support 制品/预算前置保留）**。

2026-10-17 main 合并后增量：`origin/main` 已到 `92595ca`（`e84a76c..92595ca`
共 7 个提交）。本分支合并后解决 `CVIR.md` 与 `algorithm_completion_20261017.md`
两处同段文档冲突，代码无冲突；新增内容为 PUET 证据/适配补项、独立 clean-support
制品与 CPU 入口、BBE 与 TEDⁿ 独立工程路径。对本文件结论的影响见
[main 合并后增量复核](#2026-10-17-main-合并后增量复核)。

## 确认的实现问题：PUET TS bootstrap

位置：`pu_toolbox/estimators/risk/puet.py:173`、`:191`、`:255`。

- TS 分支把训练 P 复制到 U 风险池，U 池大小成为 `n_U+n_P`，每行权重为其倒数。
- bootstrap 分支却仍只抽原 `n_U` 行，导致 U 总质量成为 `n_U/(n_U+n_P)`，不是 1。
- P 总质量仍为总体 prior；因此根风险、分裂增益和叶节点投票均可能变化。
- 默认 `bootstrap=False` 不受这个分支错误影响；现有 bootstrap 确定性测试只覆盖 OS。

独立复现输入为 8 行全零特征、4 个 P/4 个 U、prior=0.4、单树、seed=0。
四个组合的根投票分别为 OS/非bootstrap=-1、OS/bootstrap=-1、
TS/非bootstrap=-1、**TS/bootstrap=+1**。恒定特征没有分裂，P/U 数量也固定，
正确归一化下两种 bootstrap 设置均应满足 `2*0.4 < 1`，投票 -1。

复现命令的核心代码（本轮临时脚本 `tmp/p3_review_probe.py`，退出码 1）：

```python
import numpy as np
from pu_toolbox.estimators.risk.puet import PUExtraTreesClassifier

X = np.zeros((8, 1))
y = np.r_[np.ones(4, dtype=int), np.zeros(4, dtype=int)]
model = PUExtraTreesClassifier(
    class_prior=0.4, n_estimators=1, bootstrap=True, random_state=0
).fit(X, y, os_or_ts="ts")
assert model.trees_[0].vote == -1
```

建议整改：先加入 TS bootstrap 的风险质量/恒定叶回归测试，再使抽样数量与权重分母一致；
保留 OS 和默认非 bootstrap 路径的行为。本轮不修改算法代码。
已登记 [Issue #114](https://github.com/shuidisjtu/pu-learning-toolbox/issues/114)，
用户负责告知队友；本轮不重复提报。

## 2026-10-17 PUET 修复复核

**结论：#114 修复通过；PUET 实现层面建议改为“有条件接受工程证据”，正式准入仍未完成。**
审核基线 `e84a76c`。本轮独立核对锁定作者源码、论文、实现、测试与证据包；未修改源码。

### 修复内容与不变量

`pu_toolbox/estimators/risk/puet.py:193` 将 bootstrap 的 U 抽样数由 `n_unlabeled`
改为实际 U 风险池大小 `len(unlabeled_rows)`。TS 模式下该池为“原 U + P 副本”，
权重分母同为 `len(unlabeled_rows)`，因此每棵树保持：

- P 总质量 = `prior`；
- U 总质量 = `1`。

原复现（8 行全零特征、4 P / 4 U、`prior=0.4`、单树、`seed=0`）修复后四个组合
OS/非bootstrap、OS/bootstrap、TS/非bootstrap、TS/bootstrap 的根投票均为 **-1**；
旧代码 TS/bootstrap 为 **+1**。旧实现 4P/4U TS/bootstrap 的根节点 U 质量为
`4/(4+4)=0.5`、根风险 `0.32`；修复后 U 质量 `1.0`、根风险 `0.96`。

### 回归与行为漂移

- 新增 `test_edge_bootstrap_root_masses_and_constant_leaf`：2 视图 × 2 bootstrap ×
  3 组 P/U 数量 × 3 seeds，共 **36 组**；独立运行 **36 passed / 0 failed / 0 errors**。
  该测试在旧代码上会因 U 质量不为 1 而失败，是有效回归测试。
- 以 `92efb4d` 旧实现逐值对照：OS 非bootstrap、OS bootstrap、TS 非bootstrap 的
  预测、重要度、叶数、深度完全一致；仅 TS bootstrap 按预期改变，说明修复范围最小。

### 算法与来源核对

- 论文 Proposition 2(a) 的 nnPU 二次风险 `4(W_P+W_N)v(1-v)` 与实现
  `4·W_P·(W_U-W_P)/W_U` 代数等价；`W_U<=W_P` 的归零边界与论文 `v>1` 一致。
- 论文 §4 明确对 P/U 分别做 bootstrap；作者发布代码无 `bootstrap` 参数，
  本实现按论文扩展并在证据包登记，不称作者源码逐行复现。
- 锁定源码 `5cb15d7e` 的 `tree.py`、`trees.py`、`LICENSE` SHA256 与证据包一致；
  论文 PDF SHA256 `e3866f4…` 与公开对照证据一致。
- `check_p3_admission_evidence.py --source-dir puet=...`：`ok=true`，核验 3 个源文件。
- 作者源码的叶 tie 随机、阈值裁边、无正增益保护、父节点 `min_samples_leaf`
  与本实现的固定负类、完整区间、严格正增益、子节点约束，均已在证据包登记为适配。

### 本轮测试

PUET 定向测试 **76 passed / 0 failed**（含 source behavior、budget、runner smoke）；
台账/证据/路由 **79 passed**；契约 **22 passed**；格式、测试质量、基线配置门禁通过。
未独立重跑队友自报的 4030/4056 项全量测试。

### 保留条件

1. **证据同步（已补齐，2026-10-17）**：`32db08c` 已将 36 组回归纳入
   `methods.puet.source_behavior_tests`，并把 `bootstrap_risk_contract` 同步到
   证据包/台账/预算草稿；`c4210fd` 另补来源与适配验收记录。
2. **适配测试缺口（已补齐，2026-10-17）**：`test_puet_adaptation_contract.py`
   独立覆盖子节点 `min_samples_leaf`、完整阈值区间、严格负增益及越界
   `max_features`；不再只依赖 tie/零增益测试。
3. **API 小差异（已登记）**：作者源码对 `max_features > n_features` 打印提示并
   返回 `None`，本实现保留正整数截断且不修改构造参数；差异已写入证据包，未为
   对齐作者而改变既有配置。
4. **正式准入前置**：候选池、CPU 树预算/存储、PA/OA 阶段资格、公开数值对照、
   frozen-lock/资源验收与负责人签署仍待 P3.2/P3.3/P4.1。本轮不据此宣布 P3.1 完成。

2026-10-17 main 合并后复核：上述 1–3 已由 `32db08c`/`c4210fd` 补齐；未改变
PUET 训练代码、默认配置或正式准入。

## 2026-10-17 LaGAM #115 修复复核

**结论：#115 修复通过；采样身份裁决为 OS。方法层面仍暂缓/有条件，不是正式准入。**
审核基线 `e84a76c`。

- 作者锁定 commit `362a7f41…` 的 `utils/utils_data.py:182-193`：先移除 validation，
  再将 labeled P 从 U 移除，发布实验为 OS/single-training-set；9 个源码文件 SHA256
  与来源收据一致。
- 修复同步：`lagam.py` 的 `scenario` 改为 `SINGLE_TRAINING_SET`；台账
  `native_sampling_assumption=os`、`sampling_review.registry_view=os`；准入草案快照
  与来源收据 `resolution` 同步。
- 行为：`fit`/`predict` 未改；默认 OS 原本就是实际路径，显式 TS 继续拒绝；
  未来新增 `os_or_ts` 接口也不会因 ledger 为 os 而自动启用 TS（测试覆盖）。
- 回归：LaGAM 定向测试 **194 passed / 0 failed / 23 expected skips / 1 GPU deselected**；
  候选准入契约 **3 passed**；文档链接门禁通过。
- 未发现历史 LaGAM run；无需重核 manifest。
- 保留条件：clean support/额外标签预算、标准 PU-only PA 不适用、weight-only 等适配、
  来源许可/全文、公开数值、候选/资源与负责人审阅均未完成。
- 文档同步：方法卡、README、执行计划、来源复核正文中过时的“采样定位待审”已修正；
  Issue #115 已按用户指示关闭。

## 2026-10-17 CVIR #117 复核

**结论：#117 修复通过；本 Survey 的 CVIR 走 OS adaptation + 显式 α_U。OS→TS 校准
不属于本实验，不是运行前置。正式准入仍未完成。**

- 来源：锁定 `acmi-lab/PU_learning` commit `5e5e350dc0588de95a36eb952e3cf5382e786aec`；
  7 个文件 SHA256 与来源收据一致；论文 §3 与 Algorithm 2 已核对，α 是 U 内正类比例。
- 协议口径：协议 §2.3 的基础数据始终是 OS；`os_or_ts` 是训练视图标记，不是重新
  抽样 TS。因此本实验不需要、也不使用 OS→TS 校准。
- 运行路径：
  - OS：`CVIRKnownPriorTrainer` + `input_sampling="os"`，显式 α_U/来源，仅 SCAR；
    population π 留在 PA，不转发给 `fit`；拒绝 frozen Survey。
  - 直接原生 TS：`input_sampling="ts"` 只面向调用方提供的独立 P/边缘 U，
    不是本 Survey 的 OS 数据。
  - OS→TS：不实现、不属于本实验；`os_or_ts="ts"` 显式拒绝，`calibration_hooked=false`。
- provenance：`known_prior`、α_U、`prior_source`、`source_verified=false`、
  `empirical_unlabeled_positive_fraction=null`、`n_unlabeled_candidates`、
  `positive_rows_added_to_candidates=0`、`unlabeled_roles`、`calibration_applied=false`。
- 测试：CVIR 定向 **37 passed / 0 failed / 1 GPU deselected**；候选准入契约 **3 passed**。
- 非阻断项：来源收据/准入草案中的 `explicit_ts_unsupported` 已移除并改为范围说明；
  OS adapter 测试建议补 retained negative count；台账 uncertainty 第一句偏历史，
  本轮未改台账。
- 2026-10-17 main 合并后：新增独立 BBE/TEDⁿ 工程路径（`a68a241`、`92595ca`），
  二者均为 native TS/显式 α_U，不接入本 CVIR、OS Survey 或 CLI/UI；不改变本
  Survey 的 OS adaptation 结论，也不把 OS→TS 校准重新列为前置。
- 正式条件：α_U 来源/有限样本口径、候选/预算/选模、公开数值、资源与负责人审阅。
- Issue #117 已按用户指示关闭。

## 2026-10-17 PULNS #116 复核

**结论：#116 的接口/审计/披露层已交付；共享 support 准备/CPU 入口已补，但实际
逐数据集 support 规模与正式预算仍待批准。**
审核基线 `e84a76c`；2026-10-17 main 合并后补充共享工作流复核。

- 官方源码：独立检索未确认 PULNS 作者仓库/发布包；PULSNAR 是不同算法；
  按论文独立适配，`not_found` 登记合理。
- 适配器：`PULNSExtraSupportTrainer` 仅 SCAR OS + 终点 OA；核六角色稳定 IDs
  与组隔离；ADNI 无组证据阻断；拒绝 PA、frozen Survey、SAR 和 TS 校准。
- 披露：manifest 顶层 `extra_supervision`、逐候选 provenance、额外正负标签数、
  用途、总标签预算、`comparison_group=extra_clean_support`、`pa_eligible=false`。
- 新增记录：`support_origin`（external / carved_from_original_pool）、逐角色
  original/remaining/reduced 池效应；carved 的减少量之和必须等于 support 规模。
- 来源：`support_source_verified` / receipt / `source_verification_scope`；
  默认是未核实的调用方声明。
- 2026-10-17 main 合并后：新增共享 `support_data.py` 与
  `prepare_clean_support.py` / `run_clean_support_experiment.py`，支持
  Spambase/Connect-4/MNIST/F-MNIST/CIFAR-10 的 train 内 support 切分、五角色收据、
  剩余 train-only 预处理和 CPU 单候选 OA 回放。制品层记录 `support_origin`、
  完整原角色计数和未核验来源；adapter 签名完整时转发本 PR 的四个审计参数，
  部分契约拒绝。该路径仍是探索性工程入口，不批准实际 support 大小/标签预算，
  ADNI/文本/正式 CNN 未开放。
- 测试：PULNS/共享 support adapter 及 runner 定向 **74 passed / 0 failed /
  8 deselected**（7 个文件；排除 resnet18 与 GPU）；新增 carved 训练减少量必须
  与 `support_removed_from_train` 一致的反例。1 个 resnet18 用例因本机
  缺 `torchvision` 未运行，属环境缺失，不是 #116 逻辑缺陷。
- 保留：八集实际 support 来源/规模/制品、正式预算、资源与负责人审阅仍未完成；
  共享入口只提供准备/回放能力，不构成预算批准；标准 PU-only PA 不接受。
- Issue #116 已按用户指示关闭；关闭仅指接口/审计交付，不解除正式 support 制品/
  预算与准入前置。

## 2026-10-18 PULDA 复核（论文已获取）

**结论：实现身份为 PULDA（oracle/known-prior），不是 PULDA\*（BBE 估计先验）。
代码级未发现新 bug；方法身份与适配已按用户授权代录，正式准入与 stage/budget 仍未签署。**

- 论文：TPAMI 2023, Vol. 45, No. 12, pp. 15345–15363；本地 19 页 PDF，
  SHA-256 `a2cffd8e0b3275aebf8ab89a29bea11a07b598f252f228df479c8db0e16c4070`。
  已核对 Eq. (19)–(20)、(31) 与 Algorithm 1。
- 协议 §3 第 2 条：主实验可向需要总体类先验的算法传入真实 `π_population`，结果标注
  `known-prior`（oracle-prior）。因此固定先验 PULDA 符合本 Survey 主实验；
  PULDA\* 的 BBE 先验估计属于单独变体/敏感性实验，不在本轮实现范围。
- 来源：锁定 `jiangyangby/PULDA` @ `7b3dcad95bd7caa0a9477af37a05764fbe6e27bc`；
  7 个来源文件 SHA-256 与证据包一致，`check_p3_admission_evidence.py` 返回 `ok=true`。
- 损失与 EMA：工具箱 `LDA`、two-way margin、EMA 与作者源码在 4 个随机批次上逐值一致
  （差值 0.0）；PULDA 及 P3 来源/预算/准入定向测试 **164 passed / 0 failed /
  7 GPU deselected**。
- MixUp 的 paper/source 差异：论文 Eq. (31) 使用当前模型预测
  `σ(f(x1))`、`σ(f(x2))` 作为目标；作者发布代码使用持久伪标签并强制 P 目标为 1；
  工具箱跟随作者代码。按用户 2026-10-18 决定，尊重队友实现，仅在复核文档登记，
  等训练结果出现明显错误时再处理。
- U/P 与 TS 适配：工具箱风险损失逐批使用 `U∪P`，MixUp/伪标签仍使用原始 U；
  作者代码则把 P 副本物理并入 U。按用户决定，尊重当前实现与协议适配，
  不声称作者数据池/MixUp/RNG 完整重放。
- 证据补全项（不阻断本次方法身份签署）：来源证据未纳入
  `customized/mixup.py` 与 `dataTools/mixupDataset.py` 的 SHA-256；
  `source_behavior_tests` 尚未列入公式、TS 角色和 CNN 两阶段测试；
  `p3_public_comparison` 的 PULDA 条目仍标 `full_text_unavailable`。正式证据更新时补齐。
- **shuidisjtu 签署（授权代录，2026-10-18）**：确认 PULDA 方法身份为
  known-prior/oracle-prior PULDA；接受作者代码的 MixUp、U/P 池与 TS 校准适配。
  本签署不包含 PULDA\* 实现、warmup/pu_mixup 阶段资格、60+60 预算、候选/资源规格
  或正式 Survey 准入；这些仍按后续单独讨论处理。

### 2026-10-18 PULDA stage/budget 负责人决定

以下四项由 shuidisjtu 于 2026-10-18 交互确认，覆盖 PULDA 的两个
`owner_questions`，并登记候选池与阈值选择；不改变上一小节方法身份签署的历史范围，
也不构成正式准入：

| 项 | 决定 | 边界 |
|---|---|---|
| PA/OA 阶段资格 | 仅 `pu_mixup`；`warmup` 只作诊断保留，不进入 PA/OA 候选 | 其他阶段方法仍 pending，需按同一全局规则统一 |
| 训练预算 | 原生 `warmup_epochs=60` + `pu_epochs=60`；P/U batch 16/128；两阶段 cosine 均 `T_max=pu_epochs=60`；120 个快照；不早停；`pu_mixup` 内 60 个 epoch 全部可选 | 若全局 epoch cap < 120，另立等半缩放版本并标 `benchmark-adapted` |
| 候选池 | 单一默认配置（draft `constructor_defaults`）；`class_prior=π_population` 固定，不扫描先验 | 共享表示、存储/资源、公开数值仍待定 |
| 分类阈值 | 固定 logit 0（当前 `_predict` 默认，等价于 sigmoid 0.5）；阈值候选集 `{0}`；PA/OA 只选 checkpoint，不调阈值 | 其他方法是否调阈值仍待全局阈值政策；本项不改变训练梯度 |

图像表征同日决定：按协议 §2.5 使用 ResNet-18；PULDA 采用**端到端
ResNet-18**（注入 `encoder` 可训练），不走 feature-adapter，结果标为
`end-to-end CNN` / `benchmark-adapted`，不声称作者 CNN 或论文数值复现。
共享初始化、BN、每 epoch 增强、输入/归一化、表格 MLP 共享规格与资源
profile 由 [#119](https://github.com/shuidisjtu/pu-learning-toolbox/issues/119)
与队友确认；`approved_representation_ref` 仍为 `null`。

机器草案 `p3_selection_budget_spec_20261007_draft.json` 的
`formal_decisions` 仍为 `null`，`admitted=false`；未生成或签署新版本
protocol/registry，未修改台账、冻结协议或 v3 矩阵。该决定不等于 P3.1 整体完成。

## 2026-10-18 Grad-PU 复核（论文已获取，无官方代码）

**结论：本轮未发现新的公式/实现缺陷；有条件接受工程证据。图像路径按方案 B
使用 `cnn13_no_bn`（method-private），不进入标准 ResNet-18 主榜；正式准入仍受
调度/配方、阶段预算、资源证据和 paper-only 来源边界阻断。**

- 论文：AAAI 2023, 37(6), 7296–7303；本地 8 页 PDF，
  SHA-256 `c9ec9499389007546c5e3020edd98a8e17f221511b380663fc65efedeeca9244`。
  用户确认未找到作者代码；`source_status=not_found`，本轮按 paper-only 复核，
  不称源码复现。
- 公式：Eq. (5)–(7) 与实现一致：tanh 前 raw score 的输入梯度惩罚、未归一化正例权重
  `1 - β log((1+g)/2)`、P/U 绝对损失、β 从 0 线性升至 `beta_max`；`class_prior`
  不参与目标；阈值固定 0。
- TS：`fit(os_or_ts="ts")` 的 U 风险与插值边缘角色逐批取 `U∪P`；作者实验是
  remaining U，TS 不是原实验重放；该差异已登记。
- 测试：GradPU estimator 25 项（23 passed / 2 CUDA skipped）；P3 source/checkpoint/
  feature-adapter 24 passed；contract/evidence 78 passed；0 failed。
- 图像路径决定（2026-10-18）：采用方案 B，GradPU 使用现有 `cnn13_no_bn`
  作为 method-private 图像路径，不进入标准 ResNet-18 主榜；详见
  [#120](https://github.com/shuidisjtu/pu-learning-toolbox/issues/120)。
  论文明确移除 BatchNorm（印刷页 7300），实现继续 BN fail-closed；共享
  BN-free ResNet-18 不作为当前方案。
- paper-only 身份：用户 2026-10-18 接受；不要求作者代码，不称源码复现或
  source-faithful。学习率调度、batch 语义、网络/epoch、TS/SAR 声明与证据
  补全见 [#121](https://github.com/shuidisjtu/pu-learning-toolbox/issues/121)。
- 其他适配差异：论文 MLP 4×300 vs 本实现 MLP128×1；CNN13 no-BN vs
  `cnn13_no_bn`；batch250 vs 256；学习率逐渐降到 0.1× vs 固定学习率；
  100/200 epoch vs 默认 200；论文 500 validation vs 外部 PA/OA。均为
  `benchmark-adapted`，不得称论文配方复现。
- 阶段/预算：`checkpoint_stages=("pu",)`，默认 200 个快照；PA/OA 资格、候选池、
  存储/资源、公开数值未批准；`admitted=false`。

**本轮不签署正式准入。**

## 2026-10-18 Robust-PU 复核（论文与官方源码已获取）

**结论：SPL 权重、阈值调度和加权 BCE 与官方默认实现一致；本轮未发现新的公式
缺陷。正式 CLI 默认按 ledger 解析为 `ts`，population π 与 `U∪P` 风险池一致，
正式协议路径没有先验问题；裸 estimator 或显式 `os` 覆盖才需要 π_U。有条件接受
适配子集；正式准入仍受来源证据、阶段/资源/公开数值边界阻断。**

- 论文/源码：KDD 2023，`arXiv:2308.00279`；本地 11 页 PDF，
  SHA-256 `844eb4d3033aa55201a2b972063b4354061182a321533e785695e85b5c7c6bf4`。
  锁定 `woriazzc/Robust-PU` @ `34d950f2c6e56510855a922acb5f84b6459773ef`。
- 默认 SPL：Welsch 权重、logistic hardness、linear scheduler、`alpha_p/n=0.1`、
  `max_thresh=2.0`、`grow_steps=10`、`phi=0`、batch 64、pre-lr 1e-3、lr 1e-4；
  与当前 constructor 默认值一致，hard/linear 子集也已对照。
- `class_prior`：官方 `utils.py:66-119` 定义为 **U 内正类比例**；方法卡采用该语义。
  正式 CLI（`run_survey_experiment.py`）的 `--os-or-ts` 默认 None，按 ledger
  `native_sampling_assumption=ts` 解析为 `ts`；runner 传 population π，U 风险取
  `U∪P`，因此正式协议路径一致。裸 `fit(os_or_ts="os")` 或显式
  `--os-or-ts os` 覆盖时，U 风险只取原始 U，应传 π_U；当前 estimator 不自动
  换算，这是 OS 调用边界，不是正式 TS 路径的缺陷。
- 预训练 batching：官方 `pre_train` 用拼接 P+U 的 DataLoader（batch 64）；
  本实现分开抽取 P/U 批并取 `steps=max(ceil(P/B),ceil(U/B))`，更新次数和批次组成
  与官方不同。episode 内用 `array_split` 而非 DataLoader 尾批，也有轻微差异。
  按 2026-10-18 用户决定，尊重当前实现，保留为 `benchmark-adapted`，不要求逐行
  对齐官方；evidence 仍应补记。
- 预算：论文是 pre-train 100 epoch / 每次迭代 20 epoch；官方 parser 默认 400/100；
  本实现默认 10/20。按用户决定保留当前 10/20，正式预算表登记为
  `benchmark-adapted`，不称论文配方。
- moving weights：官方全局缓存随每轮 shuffle 重排时可能错位；本实现绑定原始行索引，
  属于有意的修复。默认 `phi=0` 时官方无跨轮缓存，差异不影响默认路径；`phi>0`
  时保留为适配披露，不声称官方脚本逐行等价。
- 测试：estimator 25 项（23 passed / 2 CUDA skipped）；P3 source/checkpoint
  13 项 passed；0 failed。
- 来源证据：证据包只列 `main.py`、`spl_utills.py`、`lossFunc.py`；方法卡依赖的
  `utils.py`（先验语义）和 `models.py`（架构）未纳入摘要；`paper_review_ref` 仍为
  `null`；`source_behavior_tests` 只有两项；源码换行哈希口径也需统一。许可证仍未确认。
- 其他小边界：`pretrain_epochs=0` 时仍强制要求 `class_prior`（实际不再使用）；
  `phi=1` 被拒绝；`hardness`/`scheduler`/`spl_type`/`restart` 只实现子集。
  这些是 API/适配边界，不改变公式；当前正式流程有 prior，故不阻断。
- 阶段/预算/选模/资源/公开数值仍待批准；`admitted=false`。

## 2026-10-17 main 合并后增量复核

合并基线 `origin/main` `92595ca`；解决 `CVIR.md`、`algorithm_completion_20261017.md`
两处同段冲突，`survey_execution_plan.md` 与 `api.md` 自动合并。代码无冲突。

- **PUET**：`32db08c` 补齐 36 组 bootstrap 证据、子节点/阈值区间/负增益适配回归和
  越界 `max_features` 登记；`c4210fd` 补验收记录。本文件前述保留条件 1–3 关闭，
  正式候选/预算/选模/公开数值/负责人审阅仍阻断。
- **PULNS / clean support**：`8bf8d54`/`ab239fd`/`f996126` 提供五数据集 support
  制品准备与 CPU 单候选 OA 入口；制品记录 `support_origin`、完整角色池效应和未核实
  来源，并在 adapter 签名完整时转发四个新参数，部分契约拒绝。它是探索性工程入口，
  没有批准实际 support 大小/标签预算，也没有执行真实数据或正式 CNN/GPU 验收；
  #116 的“接口/审计交付”结论保留，正式制品/预算前置不变。
- **BBE/TEDⁿ**：`a68a241`/`92595ca` 新增独立 native-TS 工程路径，分别估计 α_U 和
  执行 transform-estimate-discard；不接入 CVIR、OS Survey、总体 π 或 CLI/UI，
  不属于原 P3.1 13 方法表，不改变 P3.1 结论。
- **Windows 编码**：合并后发现 BBE/TEDn 两个契约测试用默认 `read_text()` 读中文
  台账，在 GBK 环境失败；已改为显式 `encoding="utf-8"`，断言与生产代码未变。

合并后定向测试（CPU，JUnit 本机收据）：

| 组 | 结果 |
|---|---|
| PUET 证据/适配/预算/准入 | 138 passed / 0 failed / 0 skipped |
| BBE + TEDⁿ | 140 passed / 0 failed / 0 skipped |
| clean-support 制品/入口 | 56 passed / 0 failed / 0 skipped |
| PULNS/共享 adapter/runner | 74 passed / 0 failed / 8 deselected |

全仓本地快速回归仍受本机缺 `torchvision` 与 Windows symlink 权限限制，未作为
合并通过证据；以 PR CI 复跑为准。P3MIX 仍未交付完整 estimator，P3.1 仍未整体验收。

## 逐方法验收建议

下表“有条件接受”仅指当前披露的工程证据可继续进入后续审阅，不是正式方法准入。
共同条件是来源/适配身份确认、训练与标签预算批准，以及选模与候选范围登记；
不要求为了通过 P3.1 而先执行完整主榜。

| 方法 | 技术验收建议 | 需要保留或解决的条件 |
|---|---|---|
| P3MIX | 退回补交完整交付 | 目前是 slides 对应的 batch/pool/loss 组件；完整论文/源码流程、estimator、训练时序、注册和端到端测试未齐。组件测试通过不能算方法完成 |
| PUET | 修复已验证；有条件接受工程证据（2026-10-17） | #114 已关闭；36 组回归、证据同步、子节点/阈值区间/负增益与 `max_features` 越界登记已由 `32db08c`/`c4210fd` 补齐；CPU 树预算、候选/选模/公开数值与负责人审阅另定 |
| PULDA | 论文已核对；实现身份为 known-prior PULDA；方法身份/适配已授权代录（2026-10-18） | 非 PULDA\*；损失/EMA 与作者源码一致；paper Eq. (31) 与作者持久伪标签 MixUp 的差异已登记；接受作者代码的 MixUp/U/TS 适配；stage/budget、候选/资源与正式准入另议；不得用 test-best 选模 |
| Grad-PU | 有条件接受工程证据 | 官方源码未确认；保留论文目标独立实现身份，不称逐行复现；梯度惩罚/二阶图及无 BN 约束、TS 插值范围须作为适配规格 |
| Robust-PU | 有条件接受适配子集 | nnPU warmup 需要总体 prior，组合不能称 prior-free；TS 只作用 warmup；内部固定预算/末轮与源 clean-val best/restart/scheduler 不同 |
| Split-PU | 有条件接受适配子集 | TS 只作用 teacher；Gaussian/余弦替代图像增强/SimSiam、easy 损失归一化等差异保留；论文 teacher50/代码20/本实现10不混为统一配方 |
| PAN | 有条件接受工程证据 | 原论文目标的 C/D 独立实现，官方源码未确认；OS-only、拒绝 TS；C 推断快照不代表 D/优化器续训恢复 |
| RP | 有条件接受 PU 特例证据 | 固定 rho0=0，不称一般双向噪声算法；OS-only；手工 frac_pos2neg 须登记来源或固定默认估计，禁止由 test 真值提供 |
| PULNS | 独立 support 变体接口/审计已交付；共享准备/CPU 入口已补，实际 support 制品待批准（2026-10-17） | #116：adapter 已接 OA-only SCAR OS，六角色 ID/组隔离、support_origin/角色池效应、source 验证范围与额外标签预算已记录；`8bf8d54`/`ab239fd`/`f996126` 已补五数据集分拆、收据和 CPU 单候选回放，但实际八集 support 规模/正式预算/准入仍待批准；标准 PU-only PA 不接受 |
| GEN-PU | 论文目标适配有条件接受 | prior-weighted minimax 与作者 demo 的非饱和/未加权 Du 明确不同；须选择正式目标身份，登记 prior 来源、TS 池和 PN 阶段候选资格 |
| Holistic-PU | 工程变体有条件接受 | 论文 signed 全时序/variance 与发布邻差/Jenks 不等价；默认 fixed+continue、新增 LZO 正例 CE/reinitialize 的正式 recipe 仍须选择，不称完整作者重放 |
| CVIR | OS adaptation 已接好；α_U 语义明确；正式准入仍待定（2026-10-17） | 本 Survey 走 `CVIRKnownPriorTrainer` + 显式 α_U/来源，population π 只给 PA；OS→TS 校准不属于本实验，不列为 blocker；独立 BBE/TEDⁿ 另行交付但不接入 CVIR；有限样本转换、候选/预算、公开数值与资源另定 |
| LaGAM | 采样身份已裁决为 OS；方法准入仍暂缓/有条件（2026-10-17） | #115 已修复：registry/ledger/类 scenario 与锁定源码一致；默认 OS、显式 TS 拒绝。clean support 仍是额外训练监督，标准 PU-only PA 不适用；support/标签预算、weight-only 等适配、来源许可/全文、公开数值与资源另定 |
| VPU（既有） | 沿用 D21/D22，不新增准入 | 原生 TS、OS 消融、验证变分风险仅作 diagnostic；源调度选项不解除 D22 的 PA 选模待决项或正式 recipe 前置 |

BBE/TEDⁿ 是 main 合并后新增的独立 native-TS 扩展，不在上述 13 方法表；
不纳入 P3.1 方法准入计数，正式预算/数值/负责人审阅另列。

以上多数限制已在交付中披露，不作为“隐瞒缺陷”。首轮确认的新实现缺陷仅为上述 PUET 路径，
该路径已由 #114 修复并在 2026-10-17 复核通过；未发现其它缺陷不等于证明不存在。

补充核对：`pu_toolbox/estimators/deep/split_pu.py:509` 的 hard consistency
置信度掩码仅使用 `weak_prob >= 0.95`。已从锁定作者 commit
`82fb9730597a65a12156736419d4588675bc24d5` 的 `utils.py:128-140` 核对，
作者 `hard_loss` 也是 `p_label.ge(0.95)`，因此这里是源行为，不是当前实现缺陷。
工具箱的其它视图、增强和损失归一化差异仍按方法卡披露，不能据此声称完整源码复现。

## 首轮实际验证（2026-10-16，历史）

> 本节记录首轮结果。PUET 复现失败已由 #114 修复并在 2026-10-17 复核通过；
> 下列 359/104 项与 463 项合计是首轮口径，不代表本轮 #115–#117/PULNS 复核。

- 项目 `.venv` 原无 pytest/torch；本轮安装 pytest 9.1.1、torch 2.13.0+cpu。
  PyTorch 本体约 466 MiB；仅 CPU 环境，不是正式 Windows CUDA frozen-lock 验收。
  未修改 `pyproject.toml` 或 `uv.lock`。
- `uv run --no-sync python scripts/check_p3_admission_evidence.py`：通过，5 方法；
  未传 `--source-dir`，不包含上游文件核验。
- `uv run --no-sync python scripts/check_p3_public_comparison.py`：通过，12 读数/15 格；
  未传 paper/source 目录，不包含原文独立核查；所有 numeric 资格仍为 false。
- 27 个文件的专项测试：**359 passed / 0 skipped / 0 failed / 0 errors**，
  **11 个 GPU 用例 deselected**，约 38 秒；XML：本机 `tmp/p3-review-tests.xml`。
  含 13 方法基础实现、P3MIX 组件、五方法公式/预算、VPU 视图/调度、
  LaGAM 虚拟更新、CVIR 来源/编码器、证据检查器、准入/台账契约及四角色 runner smoke。
- 7 条 warning：6 条 pipeline 高成本提示、1 条合成 runner 无 checkpoint 磁盘估算提示。
  本轮未跑全仓、完整 CNN 专项、真实数据训练或 GPU；不借用历史通过数。
- PUET 独立复现断言失败（首轮；已由 #114 修复，见上文 2026-10-17 复核）。
- 后续 7 个测试文件：**104 passed / 0 skipped / 0 failed / 0 errors**，
  **8 个 GPU 用例 deselected**，约 16 秒；XML：本机 `tmp/p3-review-followup.xml`。
  覆盖 LaGAM/VPU encoder-stream、Holistic-PU LZO/初始化、CVIR 来源/编码器、
  PULNS support CNN 及共享特征 smoke。这里只核对与 P3.1 有关的接口和行为边界，
  不据此宣布 P3.2 完成。两轮共 463 项测试通过（首轮口径）；PUET 复现失败已修复。

测试入口使用 `uv run --no-sync python -m pytest`，指定上述文件、
`-m "not gpu" -o addopts= --tb=short -q --basetemp tmp/p3-review-tests
--junitxml=tmp/p3-review-tests.xml`。机器本地 XML/临时脚本未作为正式实验制品入库。

## 下一步

本轮技术审核已覆盖上述交付范围；审核工作完成不等于 P3.1 整体验收通过。
尚未独立重核的论文全文、上游源码和数值结论保留前述限制。

1. PUET #114 已修复并关闭；P3MIX 仍需补交完整流程。
2. LaGAM #115、CVIR #117、PULNS #116 已按用户指示关闭；仍需裁决/批准
   PULNS 实际 support 制品与标签预算、CVIR α_U 来源、LaGAM support/选模互斥，
   以及各方法变体身份与正式预算。
3. 逐项确认后再登记接受/条件/整改状态；保留 P3.2/P3.3/P4.1 的独立验收边界。

依据：[执行计划](../survey_execution_plan.md)、[五方法证据](../p3_admission_evidence_20261005.md)、
[原八方法交接](../admission/p3_preintegration_handoff.md)、
[后增方法交接](independent_progress_20261005.md)。

## GEN-PU 后续复核：采用论文目标

用户已明确选择以论文作为主要实现依据，尊重当前独立实现；作者 demo 为差异对照，
不要求将论文 minimax 目标改为 demo 的非饱和 BCE。此决定不等于批准全部实验 recipe。

本轮从 IJCAI 官网下载原论文（DOI 10.24963/ijcai.2018/312）独立核对：

- PDF 第 2 页（印刷 2256）§2.1：P~pp、U~p，原生理论 TS；
  PDF 第 3 页（2257）式 (3)–(11)：先验权重、Du 混合项、Gn 对 Dn 的 anti-GAN 符号，
  与 `gen_pu.py` 两个损失函数及训练更新方向一致，未发现新的公式实现错误。
- PDF 第 5 页（2259）实验设置明确全部方法使用已知真实先验，支持 known-prior 比较。
  同页 §4.1 又描述 5000 P + 5000 N 划为 500 labeled P 和 9500 U，属于 OS；
  理论 TS 与实验 OS 的差异也存在于论文自身，不能只归因于 demo。
- 当前 `os_or_ts="ts"` 只将 Du 的真实池改为 U∪P，Dp/Dn 仍用 P；
  最终 PN 使用新生成流及总体先验加权。此 PN risk 是已披露的工程选择，
  不声称论文唯一规定或作者数值重放。GAN/PN 六优化器成本和 PN-only 快照均有测试覆盖。
- 论文 PDF 第 4 页（2258）Theorem 2 的分布恢复论证以正负分布 well-separated 等条件为前提；
  有限步训练和通用重叠数据上的恢复不能据此保证。

专项验证：`uv run --no-sync python -m pytest tests/unit/estimators/test_gen_pu.py
tests/unit/estimators/test_gen_pu_cnn.py -m "not gpu" -o addopts= -q
--basetemp tmp/genpu-review-pytest-detail-20261016 --junitxml=tmp/genpu-review-20261016.xml`。
退出码 0；XML 核验 **34 passed、0 failures/errors/skipped，2 个 GPU 用例 deselected**。
未进行真实数据数值复现、GPU 验证或超参数搜索。

结论：论文目标差异已由用户裁决；本轮未发现新的实现 bug。
正式 recipe（包括归一化、生成器输出域、训练阶段与选模预算）及公开数值准入仍另行验收。

## 2026-10-18 PAN 复核（官方源码已定位）

**结论：原 `source_status=not_found` 是检索遗漏；作者相关源码为
`morning-dews/PAN@5be703639e4c1eb2c63d869ba150d84ebf0c116e`，`source_status` 改为
`official_related`。论文 Eq.(7) 与作者 `main.py` 的 PAN 目标/采样不一致；按用户决定，
保留当前 paper-faithful 实现并登记源码差异，正式 Survey 采样跟随论文的
single-training-set/OS。正式准入仍未完成。**

- 来源：仓库 README 写“Code for paper Predictive Adversarial Learning from Positive
  and Unlabeled Data”；论文正文未直接给该链接；仓库无 LICENSE；`main.py` SHA-256
  `fde2d2b01d4a04cbafbbde82d3e18765f3f8f86b15cef3aeda7a43be346e8ba9`。
- 本机 benchmark：`F:\Temp\PU-Bench` @ `2d95a19`（`train/pan/core.py`、
  `config/methods/pan.yaml`、fixture `scenarios: [case-control]`）；
  `F:\Temp\PUBench` @ `a9a62b0`（`core/algorithms.py:1694`、`hparams_registry.py:82`）。
- 论文 vs 作者 `main.py:351-423`：论文 U 项是 `log(1-D(u))`；源码是
  `max(log(1-D(u_h))-log(1-mean(D(u_h))), 0)`，只取前 `n_P` 个 U；源码用
  `eps=1e-2`、对 P/U_h/U 分别求和，C 更新复用同一 batch；论文 Algorithm 1 写重新
  采样 U，Eq.(7) 用期望。源码默认 `n_epochs=200`、Adam `betas=(0.5,0.999)`、
  `weight_decay=1e-3`；当前实现为 100 epochs、默认 betas、无 wd。源码
  `PuDataIterator` 按数据集 P/U 比例组 batch，论文写 1:1，当前实现 1:1。
- 采样：源码 `cfg/*/uPU` 全部 `overlap=True`，U 为完整训练池（含被标注正例），
  P 为标注正例副本，属 case-control/overlap；PU-Bench fixture 也是 `case-control`。
  论文 §5.1 Step 2 / 脚注 5 是 single-training-set、U 为剩余正负样本。当前实现跟随
  论文。
- 实现：`pan.py` 的 Eq.(7) 符号/梯度隔离、prior 忽略、显式 ts 拒绝、C/D 独立
  encoder、fold 隔离和 checkpoint 均有测试覆盖；本轮不改 `pan.py`。
- 测试：PAN + pipeline/fold/contract/storage 定向组 **79 passed / 0 failed /
  0 skipped / 4 deselected**（2 GPU + 2 resnet18，`tmp/pan-review-clean.xml`）；
  PAN estimator 两文件 **16 tests / 0 failed / 2 CUDA skipped**（`tmp/pan-gpu.xml`）。
  首次未排除 resnet18 的 2 个失败均为 GenPU/Holistic 用例缺 `torchvision`，与 PAN 无关。
- 2026-10-18 决定（shuidisjtu）：`official_related`；保留 paper-faithful 实现；
  正式采样与论文一致；更新方法卡/扩展草稿/台账/review；新建
  [Issue #123](https://github.com/shuidisjtu/pu-learning-toolbox/issues/123)。
- 剩余阻断：`adversarial_budget_and_inference_only_snapshot_spec`（C-only snapshot、
  D/optimizer 预算、当前源码资源重测）、PA/OA 阶段资格、公开数值、负责人审阅；
  `admitted=false`，P3.1 仍未整体验收。
