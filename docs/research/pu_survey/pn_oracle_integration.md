# PN oracle 接入（决策与口径记录）

> 定位：PN oracle 对照路径（协议 §2.4 第 10 条）的**接入决策**、与 PU-Bench 的**口径差异**，
> 以及 P2 跑批的接口约定。缺陷修复的过程证据见 git log。
>
> P0 曾声称 PN oracle 已实现，实际接入后训练用的是 PU 视图而非真实标签——缺陷本质是 runner
> 恒传 PU 视图、`SupervisedTrainer` 只转调不替换标签。经 `CleanLabelGenerator` +
> `Generator.output_view` 修复（决策 D-A），标签语义双向 fail-loud 由 P2.0b 的 `label_semantics`
> 声明补上。
>
> 上游依据：[pu_survey_protocol.md](pu_survey_protocol.md) §2.4/§5、
> [experiment_layer.md](../../dev/experiment_layer.md) §1 D4（视图语义与声明守卫）、
> [survey_execution_plan.md](survey_execution_plan.md) P2；
> 参考文献 2 = PU-Bench `2d95a19`（与本项目其他锁定值同一 commit）。

## 1. 接入决策（2026-09-11）

### D-A 接入形态：`CleanLabelGenerator` + `Generator.output_view`

新增 `CleanLabelGenerator`（恒等策略：真实标签原样作为 view 返回），`Generator` 增加
`output_view` 类属性（默认 `"pu"`），runner 尊重之。复用 `Generator` 变化点，PA 防护免费继承
（若误配 PA，`view!="pu"` 直接 fail-loud）。备选 A2（runner 内 config 分支，改动分散）、
A3（独立 `fit_oracle()`，重复整条管线违反 DRY）均否决。

### D-B 深度路径（图像）分阶段

协议 §2.5 要求图像统一 ResNet-18，CIFAR-10 的 oracle 需端到端 CNN；`DeepFitTrainer` 以
`history_["val_risk"]` argmin 选 epoch，oracle 须改为按 clean validation 真实 accuracy 取 argmax。
分两阶段：Phase 1 交付 MLP 路径（Spambase/IMDB，不阻塞 P2 启动），Phase 2 补 CNN 路径与
oracle 专用深度 trainer。

### D-C 跑批次数：每 (dataset, seed) 一次

oracle 用真实标签训练，结果对 c 恒定（c 只影响标记）。每 (dataset, seed) 跑 1 次共 **15 次**，
聚合时广播到各 c 列并标 `c_independent: true`；不按 c 重复跑（省 3 倍 GPU 时间）。

### D-D 与 PU-Bench 数值并列展示

榜单中 PN oracle 单独一组（协议 §5.1），注明选模口径差异，不与 PU-Bench 论文表直接对比。

## 2. 与 PU-Bench `2d95a19` 的口径对照

已 clone 源码核实（`train/pn_trainer.py`、`config/methods/pn.yaml`、`train/base/` 等）。
PU-Bench 的全监督基线叫 `pn`（`PNTrainer`），README 称 "fully supervised PN oracle baseline"。

| 维度 | PU-Bench `pn` | 本协议 §2.4 第 10 条 | 判定 |
|---|---|---|---|
| 训练数据 | 同一底层 train 分区（`split_source_train_val` → PU 采样），训练循环取 `true_labels`（`pn_trainer.py:64`），损失 `BCEWithLogitsLoss` | 同 | ✅ 一致 |
| **选模** | `monitor: val_proxy_acc`（**PU-only proxy accuracy**，`config/methods/pn.yaml:19`，patience 5） | **仅以 clean_val 的真实 Accuracy 选模** | ⚠️ **实质分歧** |
| 分类器 bias 初始化 | `pn` 不在 `SOURCE_FAITHFUL_NO_BIAS_INIT` 内 → 末层 bias 用 PU 先验 `pi_unlabeled` 初始化（`train/base/data_model.py:71-97`） | 无规定 | ⚠️ 我方未覆盖 |
| 候选池/调参 | `pn.yaml` 为扁平 fallback 配方，**不参与任何超参搜索** | 同一候选预算 | ⚠️ 我方更严 |
| backbone | PN 不覆写 `create_model`，走 `select_public_model`；但 6 个 PU 方法用私有模型族，**代码无"强制同 backbone"断言** | 同一表征/backbone | ⚠️ 我方更严 |
| seed 数 | 10 个（`datasets_typical/*.yaml:3`） | 5 个 | 差异（我方按协议） |
| 报告口径 | 无 "OA only" 字面；但 **test split 只出 oracle 指标**（proxy 的 Test 列打印 `"--"`，`epoch_loop.py:168-177`），PN 在 train/val 仍算 proxy | "标为 `PN oracle (OA only)`，不得伪造 PA 结果" | ✅ 语义一致，我方措辞更显式 |
| 资源计量 | 同 `finalize()` 路径：`duration_seconds`/`time_to_best_seconds`/`max_gpu_memory_bytes` | 协议 §5.4 | ✅ 一致 |
| 结果聚合 | **仓库内无聚合代码**（无 mean±std 脚本，聚合在论文侧） | 协议 §5.2 要求五次均值±标准差 | 我方需自建 |

**实质分歧：选模口径。** PU-Bench 的 `pn` 严格说是"用真标签训练、但用 PU proxy 选 checkpoint"
的半监督式参照；本协议要求纯监督上界（真实 Accuracy 选模）。据此
[执行计划 D6](survey_execution_plan.md) 决定 PN 行降级为背景参考，不参与「交叉验证对照」的
数值裁决；oracle 的 manifest 写死 `selection_metric: "clean_val_accuracy"` 使口径可审计
（PU-Bench 侧对应 `val_proxy_acc`）。

**bias 初始化（我方未覆盖）。** 本实验沿用工具箱默认（不做 PU 先验 bias 初始化），并在
台账/manifest 注明——oracle 定位是纯监督上界，引入 PU 先验会使其不再是上界。

## 3. 设计约束（接入须遵守的既有不变量）

- 四路必须 `view="clean"`、索引两两不重叠、`test.for_selection=False`（`validate_bundle`）
- `ProtocolPA.select` 在 `view != "pu"` 时 raise——PA **结构性**拿不到真实标签
- `Trainer.trains_on_real_labels` 声明：clean 视图要求 `True`，PU 视图要求 `False`，两个方向都在训练前 fail-loud
- **推论**：oracle"用真实标签训练"必须显式声明（`CleanLabelGenerator.output_view="clean"` +
  `SupervisedTrainer.trains_on_real_labels=True`），不能靠隐式复用 PU 通道——否则视图语义不变量失效
- `partition_fair_leaderboard_runs` 已能强制同数据集内 split/seed/epoch/batch/tuning 预算一致，
  oracle 只须正确构造 `LeaderboardRunSpec` 即可纳入公平门禁

（视图语义与模块职责的权威描述见 experiment_layer.md §1 D4。）

## 4. 与 P2 跑批的接口

- 交付物：`--oracle` 入口 + 跑批清单（dataset × seed 共 15 条）+ 结果目录约定
- 执行方：HENG958（GPU 窗口）；榜单聚合由 shuidisjtu 的聚合脚本消费 oracle 结果
- 公平性：聚合时构造 `LeaderboardRunSpec(method="pn_oracle", ...)` 并过
  `partition_fair_leaderboard_runs`，与同数据集同训练路径的 PU 方法同组校验

## 5. 开放问题

1. 图像数据集上 oracle 的训练路径分组（端到端 vs feature-adapter）需与协议 §5.1 的
   "训练路径分层"对齐——oracle 是否需要在两组各出一条？（Phase 2 实施前决策）
2. 并列展示的口径差异说明是否要写进最终榜单脚注（面向论文读者，P4.2）
3. **oracle 的 backbone 尚未与 PU 方法对齐**（Phase 1 遗留）：当前 `OracleMLP` 用 sklearn
   `MLPClassifier` 的默认结构（100 单元隐层），而 PU 深度方法默认 `nn.Linear(d, 1)`、经典方法
   根本不含网络——按协议 §2.5 第 4 条本应"同一表征/backbone"。根因是"数据集内共享 MLP 规格"
   尚未确定，该规格经复核属 **P3 前置**（非 P4 中心注册表）；规格落定后 `OracleMLP` 须改按规格
   构造，在此之前 pilot 的 oracle 行单列，不与 PU 行混排。
