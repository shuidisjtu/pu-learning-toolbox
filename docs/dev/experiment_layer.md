# 实验层（experiment）设计文档

> 对应需求：PU 调研实验协议 [pu_survey_protocol.md](../research/pu_survey/pu_survey_protocol.md)
> §2.4（四份数据角色与 PA/OA 双协议）；执行状态见
> [survey_execution_plan.md](../research/pu_survey/survey_execution_plan.md) 任务分工表。
> 本文件描述 **pilot（P0）阶段的实验层设计蒸馏**：架构、关键决策、边界与已知局限。

## 1. 为什么有这个层

协议 §2.4 要求四份数据各带明确职责（`train` 只训练、`pu_val` 仅 PA、`clean_val` 仅 OA、
`test` 只评测），而工具箱既有 `PUPipeline` 是"单份数据 + 内部 PU 分层 CV + 平均分选模"，
没有数据角色声明与模型选择协议两个概念。实验层补齐该缺口，作为
工具箱面向实验/研究者用户的公共 API 层。

这一演进不是早期设计错误：`PUPipeline` 面向非专家用户（一键分析 + 推荐 + 报告），
"一份数据、内部划分"对他们是合理简化，此前也从未有用户需要数据角色；分类器层的
sklearn 式 `fit(X, y)` 契约是应当保留的底层设计。"数据带角色"是本次实验首次出现的
用户需求，属正常架构演进——补上缺口即可，无需推翻现有部分。

## 2. 关键架构决策

| # | 决策 | 选择与理由 |
|---|---|---|
| D1 | 编排 vs 策略 | `ExperimentRunner` 固定编排骨架（Template Method）；数据生成、训练、选模、留痕均为**可注入策略接口**（`Generator`/`Trainer`/`SelectionProtocol`）。不采用 Bridge 双层次——变化点各自成轴，研究者 DIY = 实现策略 ABC 并注入，不继承 runner（ADR-0018） |
| D2 | 现有层改动 | 与 `PUPipeline`/分类器签名零改动；仅 nnPU `history_` 内部补记 `val_risk`（供深度轨迹读取），SA 语义与早停逻辑不变 |
| D3 | 公共 API | `ExperimentRunner.fit(model, train, pu_val, clean_val, test)`（协议 §2.4 第 7 条字面形态）+ `DatasetBundle`/`DatasetPart` 数据合约 |
| D4 | 视图语义 | bundle 输入全部为 clean 视图（真实标签）；Generate 阶段产 PU 视图，Trainer/PA 路径**结构性接收不到真实标签**（`DatasetPart.view` + PA 入口 `view=="pu"` 校验） |
| D5 | 轨迹语义 | `RunTrajectory.best_epoch` 为 **1-based position in `traj.epochs`**（`epochs[best_epoch-1]` 为最优记录）；`EpochRecord.epoch` 为来源标签（展示性，可能 0 基）。深路径借用分类器内部验证（nnPU early-stop best state），经典算法单点 |

## 3. 模块地图（pu_toolbox/experiment/）

| 模块 | 职责 |
|---|---|
| `bundle.py` | `DatasetPart`/`DatasetBundle`/`validate_bundle`（索引两两不重叠、clean 视图、`test.for_selection=False` 强制） |
| `datasets.py` | survey 八数据集锁定映射与确定性四路切分；官方 test 显式传入，无官方 test 时先分层留出 20% |
| `image.py` | 图像 train-only 统计、输入缩放、ResNet-18/首层/增强配置与哈希留痕；验证/test 禁用增强 |
| `feature_adapter.py` | 固定/仅 train 拟合 CNN 的二维特征提取、权重/特征哈希及跨方法公平性分组门禁 |
| `text.py` | 协议固定 `all-MiniLM-L6-v2` 的 384 维文本向量、revision 留痕与内容寻址 SHA-256 缓存 |
| `training_views.py` | mini-batch 级 OS/TS-compatible 损失视图；TS 方法把 P 同时保留在正例损失并入 U 损失，且仅限 train |
| `tracking.py` | 纯数据类：`EpochRecord`/`RunTrajectory`/`SelectionArtifact`/`RunResult` |
| `protocols.py` | 策略 ABC：`Generator.generate(X, y_true, c, seed)` + `output_view` 声明（PU 生成器 `"pu"`，oracle 生成器 `"clean"`）；`Trainer.fit(estimator, X, y, *, class_prior, val_pu)` + `trains_on_real_labels` 声明（PU trainer 默认 `False`，oracle trainer 置 `True`，runner 据此要求声明与生成视图一致）；`SelectionProtocol.select(trajectories, val_part, threshold_candidates, *, class_prior)`（需要 π 的协议必须 fail-loud，不需要的必须忽略） |
| `strategies.py` | `SCARGenerator`（fixed-count `round(c·n₊)` 无放回）、`SARLBEAGenerator`/`SARLBEBGenerator`（PU-Bench `2d95a19`：k=10/shrink 1.0、辅助模型 lbfgs(100) 拟合真实标签、**抽样池限定正例集** S=1⟹Y=1、输入任意 ndim——4-D NCHW 展平后 fit/predict 用同一视图）、`CleanLabelGenerator`（PN oracle 视图：真实标签透传、`output_view="clean"`）、`ProtocolPA`（proxy accuracy，Wang et al. 2026 Def. 1 的 OS 分支，π 必传）/`ProtocolOA` + `proxy_accuracy`/`select_threshold`、`FitTrainer`/`DeepFitTrainer`/`SupervisedTrainer`；三个 PU 生成器的元数据共享审计词汇（`c_requested`/`n_labeled_requested` 未夹紧值 vs `c_realized`/`n_labeled` 夹紧后实际值、`generation_seed`、`label_view_sha256` 标签视图摘要） |
| `manifest.py` | 留痕写入/加载 + 8 必填键校验（seed/split_ref/generation/selection/test_results/elapsed/failures/resources） |
| `runner.py` | `ExperimentRunner`：校验→生成→候选训练→PA/OA 离线选择→独立 test 评测→留痕 |

## 4. 边界与已知局限

- **正式资格**：PA 正式准则（proxy accuracy，R9）与逐 epoch checkpoint 独立恢复已实现；
  但 P2.0b/P2.0c 验收、合作者签署、完整 Self-PU OA meta-reweighting、CNN/full-batch oracle
  仍未完成，各单元保持 `formal_eligible=false`。阻断与验收见
  [p2_0a_review](../research/pu_survey/p2_0a_review.md)、
  [epoch_checkpoint_delivery](../research/pu_survey/epoch_checkpoint_delivery.md)。
- **PN oracle**：MLP 路径已接入（Phase 1）；CNN oracle 的 clean-val checkpoint 选择与
  backbone 对齐列为 Phase 2。见 [pn_oracle_integration](../research/pu_survey/pn_oracle_integration.md)。

## 5. 文档与代码的分工（实施载体约定）

类/模块 docstring 中 `Design notes` 仅记"为什么"（≤6 行）并链接本文件（§2 决策表）；
长效规格文本只存在于本文件与协议（不重复记入代码注释）；被否决/被替代的决策（如
Bridge 方案）不写入注释，决策记录归 `docs/` ADR 体系。
