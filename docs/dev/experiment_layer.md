# 实验层（experiment）设计文档

> 对应需求：PU 调研实验协议 [pu_survey_protocol.md](../research/pu_survey/pu_survey_protocol.md)
> §2.4（四份数据角色与 PA/OA 双协议）；实现状态与现状差距见
> [implementation_plan.md](../research/pu_survey/implementation_plan.md) §1/§7。
> 本文件描述 **pilot（P0）阶段的实验层设计蒸馏**：架构、关键决策、边界与已知局限。

## 1. 为什么有这个层

协议 §2.4 要求四份数据各带明确职责（`train` 只训练、`pu_val` 仅 PA、`clean_val` 仅 OA、
`test` 只评测），而工具箱既有 `PUPipeline` 是"单份数据 + 内部 PU 分层 CV + 平均分选模"，
没有数据角色声明与模型选择协议两个概念（实现计划 §1.2）。实验层补齐该缺口，作为
工具箱面向实验/研究者用户的公共 API 层。

## 2. 关键架构决策

| # | 决策 | 选择与理由 |
|---|---|---|
| D1 | 编排 vs 策略 | `ExperimentRunner` 固定编排骨架（Template Method）；数据生成、训练、选模、留痕均为**可注入策略接口**（`Generator`/`Trainer`/`SelectionProtocol`）。不采用 Bridge 双层次——变化点各自成轴，研究者 DIY = 实现策略 ABC 并注入，不继承 runner（§1.4） |
| D2 | 现有层改动 | 与 `PUPipeline`/分类器签名零改动；仅 nnPU `history_` 内部补记 `val_risk`（供深度轨迹读取），SA 语义与早停逻辑不变 |
| D3 | 公共 API | `ExperimentRunner.fit(model, train, pu_val, clean_val, test)`（协议 §2.4 第 7 条字面形态）+ `DatasetBundle`/`DatasetPart` 数据合约 |
| D4 | 视图语义 | bundle 输入全部为 clean 视图（真实标签）；Generate 阶段产 PU 视图，Trainer/PA 路径**结构性接收不到真实标签**（`DatasetPart.view` + PA 入口 `view=="pu"` 校验） |
| D5 | 轨迹语义 | `RunTrajectory.best_epoch` 为 **1-based position in `traj.epochs`**（`epochs[best_epoch-1]` 为最优记录）；`EpochRecord.epoch` 为来源标签（展示性，可能 0 基）。深路径借用分类器内部验证（nnPU early-stop best state），经典算法单点 |

## 3. 模块地图（pu_toolbox/experiment/）

| 模块 | 职责 |
|---|---|
| `bundle.py` | `DatasetPart`/`DatasetBundle`/`validate_bundle`（索引两两不重叠、clean 视图、`test.for_selection=False` 强制） |
| `tracking.py` | 纯数据类：`EpochRecord`/`RunTrajectory`/`SelectionArtifact`/`RunResult` |
| `protocols.py` | 策略 ABC：`Generator.generate(X, y_true, c, seed)`；`Trainer.fit(estimator, X, y, *, class_prior, val_pu)`；`SelectionProtocol.select(trajectories, val_part, threshold_candidates)` |
| `strategies.py` | `SCARGenerator`（fixed-count `round(c·n₊)` 无放回）、`SARLBEAGenerator`/`SARLBEBGenerator`（PU-Bench `2d95a19`：k=10/shrink 1.0、辅助模型 lbfgs(100) 拟合真实标签、**抽样池限定正例集** S=1⟹Y=1）、`ProtocolPA`/`ProtocolOA` + `select_threshold`、`FitTrainer`/`DeepFitTrainer`/`SupervisedTrainer` |
| `manifest.py` | 留痕写入/加载 + 7 必填键校验（协议 §5.6 最小版：seed/split_ref/generation/selection/test_results/elapsed/failures） |
| `runner.py` | `ExperimentRunner`：校验→生成→候选训练→PA/OA 离线选择→独立 test 评测→留痕 |

## 4. 边界与已知局限（P0）

- **P0 已实现**（§7 验收清单全项 + smoke）：四路合约、SCAR/SAR 生成与 manifest、PA/OA 独立
  artifact、PN oracle（`SupervisedTrainer` + `protocols=[ProtocolOA()]`，调用方显式传递）、阈值
  选择、资源/失败最小留痕（elapsed + failures 字段）、二维（uPU）与 CNN（nnPU）端到端 smoke
- **后续跟进项**（正式 survey 数据生成前处理）：
  1. 训练失败记录与同 seed 重试（协议 §5.5）——当前 `failures` 恒为空、训练异常直接冒泡
  2. runner 前置能力门禁（`native_architectures`/`input_ndims` 训练前 fail-loud）
  3. SelfPU 补记 val 指标（当前仅 nnPU；SelfPU 走 DeepFitTrainer 裸 fit 退化路径）
  4. 完整资源计量三口径（协议 §5.3-4 / 实现计划 §6）不在 pilot 面
  5. OA 阈值评测已用 val 侧固定变换（F1 修复）；`_auc` 的裸 `except Exception` 可收窄
- **测试**：`tests/unit/experiment/`（`-m unit`；PARTIAL_COVERAGE 登记的 dataclass/ABC
  纯定义条目随覆盖自动移除）；自动验证要求（索引不重叠、PA 不读真实标签、test 不进训练/
  选择）由 `test_bundle.py`/`test_runner.py` 的断言保证。

## 5. 文档与代码的分工（实施载体约定）

类/模块 docstring 中 `Design notes` 仅记"为什么"（≤6 行）并链接本文件或
`implementation_plan.md` §1.4；长效规格文本只存在于本文件与协议/实现计划
（不重复记入代码注释）；被否决/被替代的决策（如 Bridge 方案）不写入注释，决策记录
归 `docs/` ADR 体系或实现计划 §4.1。
