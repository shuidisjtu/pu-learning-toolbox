# 逐 epoch checkpoint 容量治理方案

状态日期：2026-09-30。状态：**已实施**（方案与合入时机已由实验主责批准，实现于同日落地，见 §9）。

定位：本文件是 [artifact_storage_architecture.md](artifact_storage_architecture.md) 的**阶段 1（本次实施）**。
架构文档给出制品分级、规模推演与面向 8 数据集 × 22 方法的迁移路径；本文件只描述本次的回收实施。
上游交付记录见 [epoch_checkpoint_delivery.md](epoch_checkpoint_delivery.md)：它描述机制如何工作，
本文件描述在 B4 容量阻塞下如何在不改变选模结果的前提下回收权重文件。

## 1. 问题

B4（`nnpu` / `cifar10` / `native_cnn`，35 runs）每个候选都要按 epoch 落盘完整权重快照。
按 `epoch_checkpoint_delivery.md:44-45` 的实测口径（ResNet-18 约 44 MB/权重、200 epoch 约 8–9 GB/候选/seed），
B4 的 checkpoint 预算是**百 GB 量级**（项目内部估算约 280 GB，非租用容量保证）。

而可用空间是：

- AutoDL 数据盘 50 G，已用 26 G（含 B1/B2 结果 6.4 G 与 `backup-stage` 5.8 G），**可用约 25 G**；
- 文件存储默认每用户上限 200 GB（`autodl_official_basis.md`），且**当前尚未开通挂载**。

`pu_toolbox/experiment/pilot_plan.py:805-811` 明确记载了这一设计的代价：

> ``total_bytes`` … is what the pilot actually occupies … **Nothing deletes them** and offline
> selection needs them afterwards, so this is the figure a host holding the whole pilot is sized
> against. ``peak_bytes_per_run`` … is enough **if checkpoints are reclaimed as runs finish**.

即：峰值门禁（`guard_*`）按「跑完即回收」设想过，但**实现里没有任何回收**，因此 B4 的容量需求是全量而非峰值。

**这套取舍的初衷是可审计与可复现，而当前的占用量已经反噬该目标** —— 存不下就无从审计。
这是本方案立项的直接理由。

## 2. 事实基线（代码现状）

| 事实 | 位置 | 结论 |
|---|---|---|
| 两种保存模式 | `checkpoints.py:191-198`、`:256` | `checkpoint_dir` 有值 → `checkpoints/attempt-*` 持久化（`persistent=True`）；为 `None` → `TemporaryDirectory`，由 `RunTrajectory.checkpoint_owner`（`tracking.py:44`）持有，**run 结束自动清理** |
| 正式跑批无法走临时分支 | `runner.py:826-829` | `_checkpoint_root` 在 `manifest_path` 存在时强制回退到 manifest 旁的 `checkpoints/`；**没有配置开关能让正式跑批不持久化** |
| 关闭捕获会阻断验收 | `runner.py:838-842` | `config['capture_epoch_checkpoints']=False` 会退回旧单点路径，`per_epoch_independent_PA_OA_checkpoint_selection` 阻断保留（`api.md:1438-1439`） |
| manifest 记录完整快照清单 | `runner.py:542-547`、`checkpoints.py:96-110` | `candidate_runs[].epoch_checkpoints` 与 `selection.PA/OA.checkpoint` 均为**引用**（含 epoch、component、sha256、逐 epoch 验证指标） |
| 选择只消费分数 | `strategies.py:317-336` | `ProtocolOA.select` 在每个 epoch 上只取 `model.decision_function(x_val)` 的结果做阈值扫描；PA 同构。**模型在取到分数后不再被使用** |

最后一条是本方案的依据：选模的**决策依据**是「逐 epoch 的验证侧分数与指标」，而**权重文件**只在恢复模型评分时被需要。

## 3. 参考实现的对照（外部证据）

两篇参考文献的官方仓库均**不保存逐 epoch 快照**，因此不存在本问题：

| | PU-Bench（Chen et al.） | PUBench（Wu et al.） |
|---|---|---|
| 保存时机 | 指标改善时（`if improved`） | 默认仅训练结束 |
| 保存范围 | 单文件**原地覆盖**，`torch.save(model.state_dict(), self.save_path)` | 默认 `model.pkl`；逐 step 快照需显式 `--save_model_every_checkpoint`，**默认关闭** |
| 选模 | 训练中在线，monitor `val_proxy_acc` | 训练中在线 |
| 出处 | `train/utils/checkpointing.py:174-197` | `train.py:133-143`、`:296-303` |

**差异的根源是选模哲学**：它们在线按单一 monitor 定 best，只需一个 checkpoint；
本项目要求 PA / OA 两套准则在**同一批训练轨迹**上、分别于 `pu_val` / `clean_val` 独立选择，
且选择可事后复核，因此必须保留整条轨迹。这条差异**不应因为容量压力而消除** —— 消除它等于改变预注册的选模口径。

## 4. 方案：保留选中，回收其余

**不改变任何选择逻辑，只改变「跑完之后盘上留什么」。**

- **保留集**：`selections` 中每个协议（PA、OA）的 `(run_index, checkpoint_index)` 指向的权重文件。
  两套协议可能选中不同的候选轨迹与不同 epoch，因此保留集是**至多两组**，同一文件被两套协议同时选中时只保留一份。
- **回收对象**：该 run 的其余 `EpochCheckpoint` 权重文件（该 attempt 目录下的 `epoch_XXXX_<component>.pt`）。
- **回收时机**：test 评测完成、manifest 构造**之前** —— 使 manifest 如实反映最终盘上状态。
- **回收粒度**：逐个文件 `unlink`，不是删除整个 `attempt-*` 目录（保留目录内被选中的文件）。
- **元数据不受影响**：`EpochCheckpoint` 对象（epoch、component、sha256、逐 epoch PA/OA 验证指标）
  在内存中保留，manifest 的 `candidate_runs[].epoch_checkpoints` 仍然完整记录**哪些 epoch 被评估过、各自什么指标**。

**manifest 语义变更（必须落地，不得留悬空引用）**：

- `EpochCheckpoint` 增加可变的 `reclaimed` 状态；
- `reference()`（`checkpoints.py:96-110`）保持输出原 `path` 与 `sha256`，并输出 `reclaimed: true/false`；
- 被回收项**不写成 `path=null`** —— 那会被读成「从未持久化」（临时目录语义，见 `epoch_checkpoint_delivery.md:38-39`），
  与事实不符。保留原路径并标记 `reclaimed`，审计者可据此区分「曾经存在、已被回收」与「从未落盘」。

**语义边界**：`reclaimed` 专指「**曾落盘、后回收**」，因而**必须**伴随原 `path` 与 `sha256`。
若将来某个模式从源头就不落盘某些 epoch 权重（架构文档 §6.2 的阶段 2），那是**第三种状态**，
**不得复用本字段、更不得补造路径或摘要** —— 见
[`artifact_storage_architecture.md` §6.3](artifact_storage_architecture.md)。

**收益**：每 run 由 `epochs × candidates × components` 个文件降到至多 2 个（OA/PA 各一）。
按 B4 估算，占用由百 GB 量级降到**个位数 GB**（保留的即是可做模型级复核的选中权重）。

**代价**：非选中 epoch 的模型级复核能力（异常样本分析、用新准则在这条轨迹上重选、重新生成预测）不再可用。
选中权重的复核能力**保留**：重算阈值、查看 selected epoch、生成新预测仍可做。
这是原取舍的**收窄**，不是放弃。

## 5. 协议口径影响（须在实施前确认）

`epoch_checkpoint_delivery.md:64-66` 规定：

> 实际训练满声明 epoch 且**完整快照可供 PA/OA 独立选择**，才解除本单元的
> `per_epoch_independent_PA_OA_checkpoint_selection` 阻断；缩短训练、关闭 capture、
> 旧自定义 trainer 或**缺少持久化**，均不能获得正式可复现资格。

本方案与这段文字的**关系必须说清，不能含糊**：

1. 门禁判定发生在 **run 执行期**。回收发生在 PA/OA 选择与 test 评测**全部完成之后**，
   因此「完整快照可供独立选择」在执行期**完全成立** —— 这不是「关闭 capture」，也不是「缺少持久化」。
2. 但门禁文字若被读作「选择完成后完整快照仍须在盘上」，则本方案与该读法冲突。
   **该读法本次不予采纳**，理由是它把「选择能力」偷换成了「存储义务」，而后者正是本方案要治理的对象。
3. 采用本方案意味着**表述层的收窄**：此后解除阻断的单元，
   其可持久复现的范围是「选中权重 + 全部逐 epoch 选择记录」，**不是**「全部 epoch 权重」。

**处置**：该收窄须以决策记录形式登记（`survey_execution_plan.md` 附录 A 决策账本），并在
`epoch_checkpoint_delivery.md` 增补一节说明回收语义。**未登记前不合入。**

## 6. 可比性影响

- 回收发生在选择之后，**不改变任何 run 的选模结果、阈值、selected epoch 或 test 指标**。
  因此**不构成口径变更，不破坏组内可比性**。
- **B3a 无需重跑**。B3a 已完成的部分保留全量快照；B3b 起只保留选中权重。
  两者的选择结果仍是同一套逻辑的产物。
- 必须**如实披露**的差异：B3a 为「全量快照留存」，B3b/B4 为「选中权重留存」。
  该披露写入批次验收记录与 P2.2 交接清单。

## 7. 实现设计（待编码）

改动面：

| 文件 | 改动 |
|---|---|
| `pu_toolbox/experiment/checkpoints.py` | `EpochCheckpoint` 增加 `reclaimed` 状态；`reference()` 输出该字段 |
| `pu_toolbox/experiment/runner.py` | 在 manifest 构造前执行回收：由 `selections` 汇总保留集，回收其余权重文件并置 `reclaimed` |
| `scripts/run_survey_experiment.py` | 新增 CLI 开关（默认**关闭**，保持库与 probe 行为不变） |
| `scripts/run_survey_pilot.py` | 向单元脚本传递该开关（正式跑批启用） |

设计约束：

- 开关默认值必须为**关闭**，使既有测试与 `-m paper` / probe 路径行为不变；
- 回收失败（文件缺失、权限）须**报错而非静默继续** —— 与 `checkpoints.py:116-117` 的摘要校验失败同等对待；
- 回收不得触碰 `candidate_runs`/`selection` 的元数据写入路径。

## 8. 测试计划

新增测试（`tests/unit/experiment/`）：

1. 回收后**只存在选中权重**：非选中文件被删、OA/PA 各自选中的文件保留；
2. **选择结果不变**：同一合成数据下，开启与关闭回收的 `test_results` 与 `SelectionArtifact` 逐值相同；
3. **manifest 语义**：被回收项 `reclaimed=true` 且保留原 `path`/`sha256`；未回收项 `reclaimed=false`；
4. **OA 与 PA 选中不同 epoch** 时两组权重都被保留；
5. 开关关闭时**行为与现状完全一致**（回归保护）。

既有回归：`test_epoch_checkpoints.py`、`test_checkpoint_selection.py`、`test_checkpoint_coverage.py`、
`test_checkpoint_storage_profiles.py`、`test_checkpoint_disk_preflight.py` 必须全绿。

## 9. 合入与验收

- **批准**：实验主责于 2026-09-30 批准「保留选中、回收其余」，并要求**赶在 B3b 启动前合入**。
- **流程**：`feature/` 分支 → 实现 + 测试 → PR → 合并 `main`。B3a 运行中的进程使用已加载的旧代码，不受影响。
- **前置**：§5 的决策记录登记完成（决策 D25 已登记于 `survey_execution_plan.md` 附录 A）。
- **实施（2026-09-30）**：`feature/checkpoint-reclaim` 分支落地，改动面与 §7 一致——
  `checkpoints.py` 的 `reclaimed` 字段与 `reference()` 输出、`runner.py` 的回收函数与调用点、
  单元脚本与 Pilot 的两级开关（默认关闭）、专项测试与 `epoch_checkpoint_delivery.md` §4 的
  回收语义增补。§8 的新增测试与既有 `checkpoint*` 回归全部通过。
- **验收标准**：
  1. §8 全部新增与既有测试通过；
  2. 门禁与格式检查（`check_format.py`、`check_test_quality.py`、`check_doc_links.py`）通过；
  3. B3b 首个 run 产出后，实测抽查：盘上仅存选中权重、manifest 标记正确、test 指标与预期一致。

## 10. 未决事项

- **B4 分批粒度**：本方案把单 run 占用降到峰值量级，但 B4 仍须先跑 dry-run 取准确的
  `guard_*` 数值，再决定单批 run 数；本方案不替代该测量。
- **非选中快照的归档**：若后续认为某些 epoch 的权重仍有归档价值，需另行定义筛选规则；
  本方案不预设，按 YAGNI 留待实际需要。
- **`epoch_checkpoint_delivery.md` 的增补措辞**：待 §5 决策记录定稿后同步。
