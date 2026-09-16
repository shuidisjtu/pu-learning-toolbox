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
| `datasets.py` | survey 八数据集锁定映射与确定性四路切分；官方 test 显式传入，无官方 test 时先分层留出 20% |
| `image.py` | 图像 train-only 统计、输入缩放、ResNet-18/首层/增强配置与哈希留痕；验证/test 禁用增强 |
| `feature_adapter.py` | 固定/仅 train 拟合 CNN 的二维特征提取、权重/特征哈希及跨方法公平性分组门禁 |
| `text.py` | 协议固定 `all-MiniLM-L6-v2` 的 384 维文本向量、revision 留痕与内容寻址 SHA-256 缓存 |
| `training_views.py` | mini-batch 级 OS/TS-compatible 损失视图；TS 方法把 P 同时保留在正例损失并入 U 损失，且仅限 train |
| `tracking.py` | 纯数据类：`EpochRecord`/`RunTrajectory`/`SelectionArtifact`/`RunResult` |
| `protocols.py` | 策略 ABC：`Generator.generate(X, y_true, c, seed)` + `output_view` 声明（PU 生成器 `"pu"`，oracle 生成器 `"clean"`）；`Trainer.fit(estimator, X, y, *, class_prior, val_pu)` + `trains_on_real_labels` 声明（PU trainer 默认 `False`，oracle trainer 置 `True`，runner 据此要求声明与生成视图一致）；`SelectionProtocol.select(trajectories, val_part, threshold_candidates)` |
| `strategies.py` | `SCARGenerator`（fixed-count `round(c·n₊)` 无放回）、`SARLBEAGenerator`/`SARLBEBGenerator`（PU-Bench `2d95a19`：k=10/shrink 1.0、辅助模型 lbfgs(100) 拟合真实标签、**抽样池限定正例集** S=1⟹Y=1、输入任意 ndim——4-D NCHW 展平后 fit/predict 用同一视图）、`CleanLabelGenerator`（PN oracle 视图：真实标签透传、`output_view="clean"`）、`ProtocolPA`/`ProtocolOA` + `select_threshold`、`FitTrainer`/`DeepFitTrainer`/`SupervisedTrainer`；三个 PU 生成器的元数据共享审计词汇（`c_requested`/`n_labeled_requested` 未夹紧值 vs `c_realized`/`n_labeled` 夹紧后实际值、`generation_seed`、`label_view_sha256` 标签视图摘要） |
| `manifest.py` | 留痕写入/加载 + 8 必填键校验（seed/split_ref/generation/selection/test_results/elapsed/failures/resources） |
| `runner.py` | `ExperimentRunner`：校验→生成→候选训练→PA/OA 离线选择→独立 test 评测→留痕 |

## 4. 边界与已知局限（P0）

- **P2.0a（2026-09-16）**：`survey_protocol.py` 与 `survey_execution.py` 接入版本化
  `survey_protocol_v1.json`，脚本 `--protocol survey-v1` 选取 dataset/method/training_path 单元，
  runner 再次校验参数锁与真实表征，manifest 保存协议、预算、比较组和正式阻断信息。
  CIFAR 随机冻结 adapter 缓存与原生 nnPU 接线已有 CPU/GPU smoke；无协议入口为技术验证。
  详细决策、oracle/增强/全 epoch checkpoint 未完成边界见
  [P2.0a 交付](../research/pu_survey/p2_0a_delivery.md)。

- **P0 已实现**（§7 验收清单全项 + smoke）：四路合约、SCAR/SAR 生成与 manifest、PA/OA 独立
  artifact、阈值选择、资源/失败最小留痕（elapsed + failures 字段）、二维（uPU）与 CNN（nnPU）
  端到端 smoke
- **PN oracle 接入（2026-09-11，详见 [pn_oracle_integration.md](../research/pu_survey/pn_oracle_integration.md)）**：
  `CleanLabelGenerator`（真实标签透传）+ `SupervisedTrainer` + `protocols=[ProtocolOA()]`，
  脚本入口 `run_survey_experiment.py --oracle`。PA 因 clean 视图校验结构性拒绝；
  **视图与 trainer 的标签语义不一致在 runner 内双向 fail-loud**（PU 视图 + 真实标签 trainer；
  clean 视图 + 未声明 `trains_on_real_labels` 的 PU trainer）——此前两种误配都会静默把标记
  当作真实标签（或反之）训练出错误的"上界"。判定以声明为准：未声明 `True` 的监督 trainer 会被
  当作 PU trainer。守卫不覆盖估计器自身的优化目标（见 pn_oracle_integration.md §8）。深度
  （CNN）oracle 的 clean-val checkpoint 选择列为 Phase 2
- **SAR LBE-A/LBE-B OA-only 执行路径（2026-09-15，issue #43）**：官方脚本新增
  `--labeling-mechanism {scar, sar_lbe_a, sar_lbe_b}`（默认 `scar`，与 `--method` 正交——机制是
  实验自变量，SAR 行可跑任意 survey 方法）。SAR 分支显式注入 `[ProtocolOA()]`（协议 §2.3：SAR 下
  PA 仅可作诊断，正式选模与结论只用 OA），c 仅接受协议 token `{0.05, 0.5}`（PU-Bench vary-e），
  运行落在 `<out-dir>/<mechanism>/c_<token>/seed_<seed>/`；SCAR 保持 runner 默认 PA+OA，目录为
  `c_<token>`——两侧均按**用户输入 token** 命名（不再经 `:.1f` 归一，故 `--c 0.05` 不会落进
  `c_0.1`；同一数值的不同拼写会被拒绝而非写成两个目录）。请求门禁（c 词法五重校验、机制×oracle
  组合、SAR c 取值）在读取数据、创建目录与构造估计器之前 fail-loud；`c_requested_token`
  （用户输入的 c 拼写）在运行成功后由脚本回写 manifest（runner 的 manifest schema 是固定白名单，
  不序列化任意 config）。生成器审计字段见 api.md。
  **边界**：脚本暂无 `--architecture` 入口，4-D 图像 bundle 上 SAR 标记可正常生成，但端到端训练
  需经 `ExperimentRunner` API 显式声明 `architecture`（CLI 侧的图像执行路径归 P2.0a 执行矩阵）。
- **后续跟进项**（正式 survey 数据生成前处理）：
  - 数据前处理 P1：八数据集目录/四路切分、SBERT 文本缓存、图像 train-only 统计及
    ResNet-18/增强留痕、TS-OS batch 校准、CNN feature adapter 与公平性分组门禁均已实现
    （2026-09-06）；真实八数据集产物仍待正式实验环境生成。
  1. ~~训练失败记录与同 seed 重试~~：候选从 fresh clone 以同 seed 自动重试一次；恢复与最终
     失败均写入 `failures`，最终失败候选从排名排除，全失败时先写 manifest 再终止
     （协议 §5.5，2026-09-06 完成）
  2. ~~runner 前置能力门禁~~：已按 `native_architectures`/`input_ndims` 在 PU 生成与训练前
     fail-loud；`config["architecture"]` 可显式声明 `"mlp"`/`"cnn"`（2026-09-06 完成）
  3. ~~SelfPU 补记 val 指标~~：`pu_validation_data` 与论文所需 clean `validation_data` 分离，
     每 epoch 记录两个 teacher 的 PU nnPU-risk、恢复最优 teacher checkpoint，`DeepFitTrainer`
     输出可选模轨迹且不会重复裸 fit（2026-09-06 完成）
  4. ~~完整资源计量三口径~~：逐候选记录每次尝试/成功尝试成本，记录本 seed 全候选调参成本与
     全过程峰值 GPU allocated memory；PU 生成、runner 总时间、环境及不属于 runner 的共享预处理
     口径分列（协议 §5.3-4 / 实现计划 §6，2026-09-06 完成）
  5. OA 阈值评测已用 val 侧固定变换（F1 修复）；~~`_auc` 裸异常捕获~~已改为仅将
     单类别测试集标记为不可用并记录原因，模型评分/指标错误保持 fail-loud（2026-09-06 完成）
- **测试**：`tests/unit/experiment/`（`-m unit`；PARTIAL_COVERAGE 登记的 dataclass/ABC
  纯定义条目随覆盖自动移除）；自动验证要求（索引不重叠、PA 不读真实标签、test 不进训练/
  选择）由 `test_bundle.py`/`test_runner.py` 的断言保证。

## 5. 文档与代码的分工（实施载体约定）

类/模块 docstring 中 `Design notes` 仅记"为什么"（≤6 行）并链接本文件或
`implementation_plan.md` §1.4；长效规格文本只存在于本文件与协议/实现计划
（不重复记入代码注释）；被否决/被替代的决策（如 Bridge 方案）不写入注释，决策记录
归 `docs/` ADR 体系或实现计划 §4.1。
