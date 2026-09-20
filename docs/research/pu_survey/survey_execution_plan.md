# PU 调研实验执行计划（pilot → 主榜）

> 定位：本文件是**执行路线与状态**，与协议的承接关系——
> [pu_survey_protocol.md](pu_survey_protocol.md) 是要求纲要，
> [implementation_plan.md](implementation_plan.md) 是现状差距与技术实现维度；
> 状态日期：2026-09-18。

## 1. 现状
- **实验层基础功能就绪（不等于正式协议全绿）**：`pu_toolbox/experiment/` 34 个公共API，四路数据合约、PA/OA 独立选模、策略化接口、
  数据准备链（datasets/image/text/feature_adapter/training_views）、资源计量与失败语义、公平性门禁均已实现并入门禁覆盖。
- **22 目标方法**：7 个已通过现有可训练接口接入——uPU、nnPU、KLDCE、Dist-PU、PUSB、LBE、Self-PU（均有方法卡）；
  另有 Grad-PU（二维 MLP）、PUET（CPU Extra Trees）、VPU（二维 MLP）及 PULDA（二维 MLP）完成实验性组件、注册、方法卡与 smoke，但未入 Survey 台账/执行矩阵，**不计入 P3.1/P3.2 验收或正式榜**；
  其余 10 个**未出现**（无注册/无占位/无方法卡）：A 类 PAN、GEN-PU、PULNS、RP、CVIR、Holistic-PU、P3MIX，
  C 类 Robust-PU、Split-PU、LAGAM；第 22 个即 PN oracle（下述）；
  `api_only` 0 个。
- **PN oracle（2026-09-11 接入 MLP 路径）**：无注册项；经 `CleanLabelGenerator`（真实标签透传、
  `output_view="clean"`）+ `SupervisedTrainer` + 仅 OA 协议接入，脚本入口
  `run_survey_experiment.py --oracle`。CNN（图像）路径的 clean-val checkpoint 选择留待 Phase 2。
  详见 [pn_oracle_integration.md](pn_oracle_integration.md)。
- **SAR-OA 执行路径（2026-09-15 接通，issue #43）**：官方脚本新增
  `--labeling-mechanism {scar, sar_lbe_a, sar_lbe_b}`（默认 `scar`，与 `--method` 正交）。
  SAR 行 OA-only、c 仅接受协议 token `{0.05, 0.5}`（PU-Bench vary-e），输出落
  `<out-dir>/<mechanism>/c_<token>/seed_<seed>/`；生成器审计字段（请求值/夹紧值/seed/标记摘要）
  入 manifest。**可运行边界**：SAR 行仍需 P2.0b 的标签语义门禁与 P2.0a 的执行矩阵才可进入正式
  pilot 混排；在此之前 SAR 产物按技术验证单列（见 §2.0 三种门槛）。
- **能力声明现状**：架构声明为 `native_architectures`/`input_ndims`/`encoder_parameter`/`trains_encoder`
  四字段；P2.0b 新增独立的 `label_semantics` 主输入语义声明（每个注册分类器都须显式声明，
  由契约测试强制，不写死数量）。
  当前 7 个 Survey 方法中仅 nnPU 支持原生 CNN（mlp/cnn、{2,4}、encoder 注入；
  工具箱整体另有 infomax_pu、weighted_contrastive_pu 为 mlp/cnn 双架构，不在 Survey 22 方法范围）；
  Self-PU 为 mlp/{2,4}（非原生 CNN——由 native_architectures={"mlp"} 承载；input_ndims 按模板
  "支持输入维度"口径如实声明，2026-09-14 审阅 P1#1 修正 issue #38 的收窄）；
  Dist-PU 声明 {mlp}/{2}（非
  tabular-only）；uPU/KLDCE/PUSB/LBE 为 tabular-only（{2}）——图像数据集上这些方法必须走
  `cnn_feature_adapter`（benchmark-adapted）。
- **协议要求但还未实现/未声明的项**：
  1. 台账 6 槽（`native_sampling_assumption`/`run_view`/`calibration_applied`/`prior_semantics`/
     `adaptation_level`/模态与 backbone；另有 `paper`/`code_version`/`implementation_status` 身份与来源
     字段）在代码与方法卡中**均未声明**（仅协议文字 + 台账 JSON + 实验层 per-call 机制）
  2. ~~官方示例脚本（协议 §2.4 第 9 条）未实现~~ ✅ 2026-09-08 已实现（见 §2.2 的 1.3 先行小工作）
  3. 中心超参数注册表（实现计划 §6，参考 PU-Bench `core/hparams_registry.py`）未实现；
     当前候选池仅为 runner `config["candidates"]` 的运行态配置
  4. ~~标签语义无声明字段、runner 错配静默~~：P2.0b 阶段 A 的声明位、registry 同步与 experiment
     层守卫已完成工程实现，仍待 HENG958 复核/签署；pipeline/comparison 入口检查、CLI/UI 展示与
     ADR 收口属于阶段 B，尚未完成。PU `{1,0}` 与监督 PN `{0,1}` 数值同形，无法用取值代替
     声明，见 [标签语义方案](../../dev/label_semantics_plan.md) 与 [交付记录](p2_0b_delivery.md)。

## 2. 推进路线

### 2.0 分工、依赖与验收（v2）

实施主体：**shuidisjtu**（数据、实验编排、结果留痕与文档）和 **HENG958**（算法接入、深度训练与 GPU 执行）。每项只有一名**主责**；协作者须在交付前完成复核。任务完成必须有测试、manifest、运行记录或 PR 链接等可复核证据，不能只以口头或代码存在判定完成。

| 编号 | 任务 | 前置 | 验收标准 | 状态 / 主责 |
|---|---|---|---|---|
| P1.1 | 环境与 GPU 验证 | — | `uv.lock` 可复现；目标环境完成 GPU smoke；版本、设备与验证记录可追溯 | ✅ 已完成 / shuidisjtu；HENG958 GPU 能力复核完成，正式跑批需 frozen-lock 环境（见独立复核记录） |
| P1.2 | 数据获取与版本审计 | — | 数据来源、版本、标签映射与许可记录入 manifest；ADNI 的准入状态明确 | 🚧 manifest 侧已补齐（2026-09-19，三 pilot 数据集；许可按官方页面原文逐条记录——UCI 为 CC BY 4.0，另两个来源未声明，见 §4 第 0f 条）；ADNI 不在 pilot 范围，准入路线见 D1。本项 🚧 的口径是**范围**而非遗漏：协议 §2.1 列 8 个数据集，pilot 阶段先取 3 个易得且分属三种模态的（CIFAR-10 / Spambase / IMDB，见 §1.2），故 pilot 范围内已完成、按协议全量仍待办 / shuidisjtu |
| P1.3a | 方法台账 | — | 7 个已实现方法的 6 槽（另有 paper/code_version/implementation_status 身份与来源字段）已填写，evidence 覆盖其中 3 槽；每项经对应方法负责人复核 | ✅ 已完成 / shuidisjtu；HENG958 已复核 nnPU、Self-PU（2026-09-16） |
| P1.3b | 官方 Survey 脚本 | — | 四路输入、PA/OA、结果归档与 oracle 入口均有脚本级测试 | ✅ 已完成 / shuidisjtu |
| P1.4 | Pilot 数据产物 | P1.1、P1.2 | CIFAR-10、IMDB、Spambase 各 5 个 seed 的四路 split、预处理与 manifest 均通过合同验证 | ✅ shuidisjtu 侧已完成（三数据集统一重建 2026-09-19，见 §4 第 0f 条）；传输/校验两端工具已就位（`scripts/survey_splits_archive.py`，见第 0g 条）；**载体已定（网盘带外传），三个归档已于 2026-09-20 发送，逐文件索引与归档摘要先于发送入库**（`docs/research/pu_survey/data/split_artifacts_index.json`）；⏸ HENG958 可执行性复核等待其取件校验 |
| P2.0a | Pilot 共享规格与 oracle 对齐决策（阶段 A） | P1.3a、P1.3b | 版本化执行矩阵（`survey_protocol_v1.json`：预算定义表 + 执行单元行）、runner 强制消费、manifest 扩展（protocol_version/backbone/budget/representation/comparability_group + adapter manifest 合并）、CIFAR adapter 接线、训练路径分组与 PN oracle 对齐方式书面锁定 | ✅ 已签署验收（2026-09-17）/ HENG958 交付；shuidisjtu 复核签署；**不放行正式 P2.1**（R5/R8 记为 P2.1 前置条件，两者已于 2026-09-19 工程兑现）；见 [交付记录](p2_0a_delivery.md)、[复核包](p2_0a_review.md) |
| P2.0b | 标签语义门禁（阶段 A） | P1.3a、P1.3b | `label_semantics_plan` P1+P2 提前完成：声明位 + registry 同步 + experiment 层检查，错误组合 fail-loud；pipeline 层检查属阶段 B | 🚧 工程实现与回归完成，HENG958 独立复核/签署待办；见 [交付记录](p2_0b_delivery.md) |
| P2.0c | 交叉验证对照预注册（阶段 A） | P2.0a | 对照矩阵与判定规则冻结入本文档「交叉验证对照」节（§2 末）；锚点数值预注册 | 🚧 技术审计修订完成，36 锚点/7 行映射待审；HENG958 正式复核未签署，见 [复核包](p2_0c_review.md) |
| P2.0d | SAR-OA 执行路径（issue #43） | P1.3b | 官方脚本可选标记机制（SCAR / SAR LBE-A / SAR LBE-B）；SAR 仅 `{0.05,0.5}` 且强制 OA-only；生成器审计字段入 manifest；脚本级端到端测试 | ✅ 已完成 / shuidisjtu；HENG958 已复核（2026-09-16） |
| P2.1 | Pilot 跑批与运行制品 | P1.4、P2.0a、P2.0b、P2.0c | 每个计划单元产生完整 manifest、选择 artifact、资源/失败记录；oracle 按 `(dataset, seed)` 去重 | ⏳ 待办 / HENG958；编排载体与磁盘预算已就位（`scripts/run_survey_pilot.py`，见第 0h 条），跑批主机与环境路线、GPU 窗口排定仍待定 |
| P2.2 | Pilot 聚合与审计 | P2.1 | 发布 `pilot / partial benchmark` 分层结果；检查路径隔离、复现字段和异常单元；不得生成跨数据集总排名 | ⏳ 待办 / shuidisjtu；HENG958 复核深度结果 |
| P3.1 | 缺失方法接入（经典/B 类） | P2.0a、P2.0b | 每方法完成实现、方法卡、台账、原文可追溯、冒烟与公开行为对照；使用已锁定的共享规格 | 🚧 技术预集成 / shuidisjtu：VPU、PULDA 已完成独立组件，台账/矩阵与正式验收未做；其余 PAN、RP、CVIR、PULNS 待办 |
| P3.2 | 缺失方法接入（深度/C 类） | P2.0a、P2.0b | 同 P3.1，另需 GPU smoke、设备/随机性与保存加载验证 | ⏳ 待办 / HENG958：PUET、Grad-PU、Robust-PU、Split-PU、LAGAM、GEN-PU、Holistic-PU、P3MIX；Grad-PU/PUET 独立组件已完成，台账/矩阵及正式验收仍待前置项；PUET 为 CPU 树方法，分组/GPU 条款须复核 |
| P3.3 | 深度 GPU 验证与调度 | P3.2 | GPU 预约、显存预算、失败/OOM 重试及结果路径均有记录；不与 P2.1 竞争同一窗口 | ⏳ 待办 / HENG958 |
| P4.1 | 中心超参数注册表 | 各方法候选参数已确定 | 候选池预注册、版本化；版本写入 artifact 并受 manifest 校验 | ⏳ 待办 / shuidisjtu |
| P4.2 | 主榜聚合与分析 | P3.1、P3.2、P3.3、P4.1 | 22 项全部通过门禁后，按四组结果和训练路径分层；结论区分文献事实、实验观测与推断 | ⏳ 待办 / shuidisjtu；HENG958 复核 C/A 深度结论 |

**依赖与升级规则（三种门槛）**：① **技术 smoke**（单方法链路验证）仅需基础设施可用，可随时执行；② **正式 pilot 跑批**须 P2.0a/P2.0b/P2.0c 全绿；③ **与 oracle 或跨方法结果混排**须阶段 A 验收全部满足。未达门槛而提前执行的结果，产物必须标记为”需按 P2.0 规格重跑”。P2.1 与 P3.3 共享单卡时，HENG958 负责排定并记录 GPU 窗口；数据许可、共享规格、标签语义或资源不足造成阻塞时，主责须在计划的“开放问题与风险”中记录影响与下一步，并由两位实施主体共同决定升级、拆分或降级。Survey 分工在其范围内覆盖 ADR-0008 中较早的论文分配。


### P1 执行准备（pilot 前完成）

- **1.1 环境 ✅（2026-09-08）**：T600（驱动 596.52）上 pytest -m gpu 真实验证通过；
  `uv sync --extra research --extra dev`（torch 2.14.0+cu126 / torchvision 0.29.0+cu126 / lightning /
  sentence-transformers 5.7.0 / densratio / pytest / ruff）+ SBERT 模型落地
  （`all-MiniLM-L6-v2`，revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` 锁定、内容寻址缓存生效）；
  依赖由 `uv.lock` 固化（见 D3）。
- **1.2 数据获取**：种子子集先行 = `CIFAR-10 + Spambase + IMDB`（1 图 + 1 表 + 1 文）；各数据集获取渠道——MNIST/F-MNIST/CIFAR-10 自动下载，
  20News `sklearn.fetch_20newsgroups`，Spambase/Connect-4 UCI，IMDB HF/原始文本，ADNI 特殊（见 D1）。
- **1.3 先行小工作**（pilot 前，工作量小收益大）：
  1. **方法台账 JSON ✅（2026-09-08）**（`pu_toolbox/experiment/method_ledger.json`，程序真相源）：7 个已实现方法 × 6 槽
     （另有 `paper`/`code_version`/`implementation_status` 身份与来源字段），值按方法卡/论文
     填写，存疑处显式标注；实验脚本读取（TS-OS 判定/结果标注"原生适用 vs 假设违背鲁棒性"的依据）。
     【采用独立 JSON 方案；方法卡节、注册表字段扩展后期可做。经典 5 方法（uPU/KLDCE/Dist-PU/PUSB/LBE）
     已由 shuidisjtu 填写；nnPU/Self-PU 已由 HENG958 于 2026-09-16 对照论文、官方实现、
     registry 与 estimator 完成复核，机器可查状态写入台账 schema 1.1。】
     【台账与 registry 边界（2026-09-14，issue #42）】registry（类属性权威 + `_SYNC_FIELDS` 镜像）
     是代码真相源，负责训练正确性门禁（`requires_class_prior` 等）；本台账是 Survey 范围的实验标注源
     （当前 8 键，含 `pusb_kernel`），负责 TS-OS/原生适用等标注与结果留痕；两者一致性由
     `tests/contract/test_ledger_registry_consistency.py` 锁定。台账 evidence 当前覆盖 6 槽中的 3 个
     （原生假设/先验语义/适配级别），其余槽证据待补（不阻塞 pilot）。
  2. **官方示例脚本 ✅（2026-09-08）**（协议 §2.4 第 9 条交付物 + pilot 执行载体；
     `scripts/run_survey_experiment.py`）：
     读取四份数据 → c/candidates 配置 → `ExperimentRunner` → PA/OA → 结果归档；先验必传由 registry
     `requires_class_prior`（类属性权威）驱动门禁、台账 `prior_semantics` 只做结果标注，未入台账的方法
     经本脚本运行 fail-loud；PN oracle 已于 2026-09-11 接入（`--oracle`，见 §1）。
- **1.4 pilot 数据产物 ✅（2026-09-08）**：种子 3 集 × 5 seeds 四路切分完成
  （`data/splits/<dataset>/split_<seed>/` 四份 npz + split_manifest.json；
  `scripts/prepare_survey_splits.py` 生成，产物经 15 套合同断言 + 示例脚本端到端冒烟验证）。
  模态预处理：Spambase z-score(float32)、IMDB SBERT 384-d(revision 锁定)、
  CIFAR-10 uint8 NCHW + train-only 通道统计（统计均冻结入 manifest）。切分执行口径见 D2。

### P2 pilot 实验（结果一律标注 `pilot / partial benchmark`）

- 7 方法 × 3 数据集 × c∈{0.1,0.3,0.5} × 5 seed × PA/OA 双协议；PN oracle 对照
  （**跑批去重**：oracle 结果对 c 恒定，每 (dataset, seed) 跑 1 次共 15 次，
  聚合时广播到各 c 列并标注 `c_independent`；脚本用 `--oracle`）
- 模态-方法矩阵：表格/文本上 7 法均原生（文本＝SBERT 384 维特征 + MLP）；
  图像端到端仅 nnPU 原生（CNN）；其余 6 法（含 Self-PU，mlp-only、无 encoder，issue #38）
  图像走 `cnn_feature_adapter`（基准-适配、与原生路径**强制分组**，不混合排名）
- **PUSB 行采用 `pusb_kernel`**（official-aligned RBF 核实现，需 π，由 registry `requires_class_prior`
  门禁强制每 run 传入）；linear baseline `pusb` 为附加工程基线，不入榜（2026-09-14，issue #42）
- 候选池：pilot 阶段用论文默认参数 + 少量合手候选（中心注册表到 P4 引入）
- **执行矩阵（阶段 A，P2.0a）**：冻结 `pu_toolbox/experiment/survey_protocol_v1.json`（预算定义表 +
  执行单元行），runner 强制消费；矩阵锁定字段（representation/backbone/budget/training_path）
  不可被 CLI 覆盖，允许的覆盖（--c/--seeds/--split-ref/--candidates）偏离协议须标记
  `protocol_deviation` 并排除正式榜
- **CIFAR adapter 接线（阶段 A，P2.0a）**：feature-adapter 原语已受测（`feature_adapter.py`），
  ResNet-18 encoder factory、weights/seed 锁定、encoder state 共享范围与缓存复用、
  `run_survey_experiment.py --protocol survey-v1` 装配已于 2026-09-16 接通并完成 CPU smoke。
  当前 adapter 是随机冻结 encoder 工程基线，不是训练后的 CNN（R4：接受并已写明边界）；
  后续 `survey-v1.1` 补齐逐 epoch 权重保存与独立恢复；正式跑批仍受阶段 A、
  PA 正式准则及未完成路径门禁阻断，见 [P2.0a 交付](p2_0a_delivery.md)。
  规格与报告范围已由合作者签署确认（2026-09-17），逐条签署记录见
  [P2.0a 复核包 §4](p2_0a_review.md#4-签署记录)。
- **交叉验证（阶段 A，P2.0c）**：对照矩阵与判定规则预注册（见本节末「交叉验证对照」），
  随 pilot 执行并写入聚合报告
- **依赖**：oracle 与各 PU 方法须在同一 backbone 规格下比较（协议 §2.5 第 4 条）——该规格
  由阶段 A 的执行矩阵锁定。pilot 若在其确定前启动，oracle 行须标注自身结构并单列，待规格落定后重跑
- 产出：pilot 榜单（SCAR-PA / SCAR-OA），交付前提=验证全链路（环境、数据、runner、manifest、资源计量）正常

### P3 算法接入（与 P2 并行推进；分工作如下：表 2.0）

- **前置：确定数据集内共享的 backbone 规格**（表格/文本 MLP：层数、宽度、激活、优化器、
  epoch 预算；图像 ResNet-18 已在协议 §2.5 锁定）。这是**方法接入的接口约定**——每个方法接入
  时都要按它实现，所以归属 P3，**不由 P4 的中心注册表承担**（注册表管的是超参候选池）。
  已实现的 7 个方法需回溯对齐：其表格路径默认各为 `nn.Linear(d, 1)`（深度类）或非网络
  实现（经典类），尚未共享同一规格
- **前置：标签语义声明的收口（阶段 B）**：P1+P2（声明位 + registry 同步 + experiment 层检查）
  已于 2026-09-14 提前至 P2 阶段 A 实施（P2.0b）；P3 前置保留 P3+P4（pipeline/comparison 入口
  检查 + CLI 展示 + ADR/文档收口）。方案、检查点与参考文献 2 的对照见
  [label_semantics_plan.md](../../dev/label_semantics_plan.md)
- 原始 14 个缺失方法的接入顺序：B 类（VPU、PULDA，风格接近已有 B 类）→ A 类（7 个，依赖论文及其源码复现）
  → C 类（5 个，深度/优化设计，需 GPU 验证）；具体主责以表 2.0 为唯一真相源。
- 每方法 = 实现 + 方法卡 + 台账登记（方法台账 JSON 同步更新）+ 门禁（原文可追溯、冒烟、
  公开结果对照，协议 §5）；**接入验收须确认使用共享 backbone 规格**，方法私有网络只能标
  `benchmark-adapted` 单列报告（协议 §2.5 第 4 条）；
  TS 原生方法在其训练循环内接入 `calibrate_ts_os_batch`

#### P3.1 技术预集成进度（2026-09-18；非正式验收）

[VPU](../method_cards/VPU.md) 已完成二维特征版的变分风险与 MixUp 正则实现、
注册/API、方法卡、公式及接口测试、PU 验证风险、训练轨迹与权重恢复。与
`HC-Feynman/vpu` 锁定提交 `603bcc1e628795f57a5ac87e5b0b977273b7cf91`
对照了核心目标与归一化方式；默认网络是工具箱适配，**不是**论文的图像或七层表格网络。
[PULDA](../method_cards/PULDA.md) 已完成分布对齐、双向 margin、EMA 与伪标签 MixUp
两阶段二维 MLP 适配，并对照 `jiangyangby/PULDA` 锁定提交
`7b3dcad95bd7caa0a9477af37a05764fbe6e27bc`；作者仓库仅提供 CIFAR 路径且存在缺失模块导入，
故不宣称官方端到端复现。两项均于 2026-09-19 在 RTX A6000 完成单次 CUDA smoke。
这两项只是可独立使用的实验性组件，**未修改已冻结的 Survey 台账/执行矩阵**；正式 P3.1 仍需
P2.0b 签署、共享 backbone/执行规格对齐、台账与矩阵登记、公开结果对照、多 seed
GPU/资源记录及合作者复核。下一项按表 2.0 推进 PAN。

#### P3.2 技术预集成进度（2026-09-18；非正式验收）

以下两项已作为独立组件推送至 `main`，可供 API/CLI 调用，但**尚未进入 Survey 台账或执行矩阵，
不计入 P3.2 完成项，也不进入正式榜**。远程 Actions 结果待核验；本次推送时 GitHub CLI 未登录，
不能据此宣称远程门禁通过。

| 方法 | 已完成的技术工作 | 验收前仍需完成 |
|---|---|---|
| [Grad-PU](../method_cards/GradPU.md)（`9059fcf`） | 论文式 (5)–(7) 的二维特征 MLP 实现、注册/API、方法卡、公式及接口测试、CPU/CUDA smoke、训练轨迹与权重恢复 | P2.0b 后接入台账/矩阵；对齐共享 backbone 与协议，补多 seed GPU/资源记录、公开结果对照及双人复核；如宣称图像论文复现，还需 CNN 路径 |
| [PUET](../method_cards/PUET.md)（`86b56de`） | CPU Extra Trees 的 nnPU/quadratic 分支、注册/API、方法卡、节点风险及随机性测试、pickle/pipeline 验证 | P2.0b 后接入台账/矩阵；公开结果对照及双人复核；确认树模型的分组、共享规格适用方式及 GPU smoke 是否豁免；若宣称完整作者实现，还需 uPU/logistic 分支 |

**下一步独立工作**：在不触碰待审核协议的前提下，继续按 P3.2 名单逐项预集成缺失方法，
每项单独提交并标清论文/实现边界；优先评估 Robust-PU，随后视官方实现与依赖情况推进
Split-PU、LAGAM、GEN-PU、Holistic-PU、P3MIX。正式验收仍受 P2.0b、共享规格、Survey
台账/矩阵、公开结果对照和合作者复核约束；此顺序只是技术预集成建议，不改变表 2.0 的主责与门禁。

### P4 完整主榜

- 中心超参数注册表（参考 PU-Bench `core/hparams_registry.py`，注册表版本入 artifact）：
  管理各方法的**超参候选池**；backbone 规格由 P3 的共享规格确定，不在注册表内
- 22 方法全部通过四项门禁 → 四组榜单（SCAR-PA 主榜 / SCAR-OA / SAR-OA / PN oracle）→ §5.7 分析
- 发布条件（协议 §4 原文）：完整主榜以 22 个目标方法全部通过相应门禁为发布条件；
  在此之前只可发布明确标为 `pilot / partial benchmark` 的部分结果。

### 交叉验证对照（预注册，2026-09-14）

试验结果须与两篇参考文献（Wang et al., ICLR 2026 = PUBench；Chen et al., 2026 = PU-Bench）
及各方法原论文交叉验证。本节为**预注册**：锚点数值与判定规则在 pilot 启动前冻结，执行后按
规则裁决，禁止事后调整（阈值、锚点、对照来源均以本节为准；确需修订须记录理由并重发
预注册版本）。

**锚点来源三类**：① PUBench（PA/PAUC/OA 三准则并列；我方 OA↔其 OA 列、我方 PA↔其 PA 列）；
② PU-Bench（其选模用真实验证标签的 macro-F1 = Wang 定义的 OA，**无 PA 机制**——我方 PA 结果
与其数值无直接可比性）；③ 各方法原论文（以方法卡 `paper` 字段与官方实现为准）。

**PU-Bench Table 1（p6）：cc/SCAR、c=0.1、10 seeds，Accuracy% ± std**

| 方法 | CIFAR-10 | Spambase | IMDB |
|---|---|---|---|
| nnPU | 85.30±2.63 | 81.66±2.73 | 77.37±4.82 |
| PUSB | 87.68±2.85 | 81.56±2.95 | 77.86±3.82 |
| Dist-PU | 88.09±3.85 | 85.71±3.95 | 77.88±5.02 |
| LBE-PU | 83.98±3.62 | 68.53±3.71 | 76.25±4.73 |
| Self-PU | 76.73±2.43 | 72.10±2.92 | 74.05±3.23 |
| PN oracle | 94.88±0.57 | 91.03±0.66 | 79.89±0.83 |

> 注：uPU、KLDCE 未评估。其 "PUSB" 行对应仓库 `nnpusb` 实现，与**本项目 PUSB 同属
> Kato PUSB/nnPUSB 来源家族**（Kato/Teshima/Honda，ICLR 2019，官方上游
> MasaKat0/PUlearning），但 estimator 目标、模型族、先验语义与训练协议不同，**不能直接数值
> 等价比较**。**PN 行不参与数值裁决**（降级为背景参考：其 `pn` 用 PU-only proxy 选模
> （`val_proxy_acc`），我方 oracle 以 `clean_val` 真实 Accuracy 选模，口径不同——
> `pn_oracle_integration.md` 既有决策；PN ≥ 各 PU 方法仅作协议内诊断预期）。主要协议差异：
> 其验证比例 0.01（我方 5%+5%）、其 CIFAR-10 仅 /255.0（我方 train-only 通道归一化）、
> 其 SBERT 特征 L2 归一化（我方口径待核对）。

**PUBench Table 1/2（p8-9）：CIFAR-10，正例率 30%，Accuracy%，PA / PAUC / OA 三列**；
Case 1 正类 {0,1,2,8,9}、Case 2 {2,3,5,7,9}——与我方 {0,1,8,9} 划分不同，仅判量级+趋势。
uPU 80.24/76.07/82.04（Case 1）、66.21/69.03/70.46（Case 2）；nnPU 82.03/75.56/82.40、
74.27/62.67/77.62；PUSB 81.53/82.49/82.91、75.74/78.80/78.35；Dist-PU 81.64/79.31/83.56、
73.46/74.83/74.69；LBE 82.71/73.60/85.03、72.47/63.54/75.96（"-c" 校准版各行从略，
见论文原表）。无 PN 基线行、无 Self-PU。

**原论文锚点（方法卡转录）**：nnPU→CIFAR-10（可行，backbone 差异登记）；PUSB→Spambase
（Kato et al. ICLR 2019 Table 2，扫 π∈{0.2,0.4,0.6,0.8} 与我方 c 轴不同，协议轴差异登记）；
Dist-PU→CIFAR-10（clean-room 档，仅量级）；Self-PU→CIFAR-10（13 层 CNN，我方 mlp-only +
adapter，仅量级）；uPU/KLDCE 无 pilot 对照点；LBE 论文为深度 Adam/EM 形态、我方线性近似，
无直接数值对照——无锚点单元显式记录"无直接对照点"，不得静默跳过。

**判定规则**：

1. **数值裁决仅用于"协议高度一致"的锚点**（同选模口径、同 backbone 家族、同 c 档、同指标），
   标准误感知：`SE_pooled = √(SD_anchor²/n_anchor + SD_ours²/n_ours)`；量级一致 ⟺
   `|Δ mean| ≤ max(3pp, 2·SE_pooled)`。超限 → 人工排查（实现级 `-m paper` 复现测试/官方源码
   对照/台账证据 → 协议级 c 抽样/划分/验证比例/backbone 预处理 → 数据口径版本/标签映射/类先验），
   结论分三档：**实现错误 / 协议差异可解释 / 无解释**；仅"无解释"升级为正确性警报。
2. **协议差异不可消除的锚点**（backbone 不同、正类划分不同等）：只记录方向与量级摘要，
   不做数值裁决。
3. **诊断预期**（非正确性硬门禁；违反 → 人工审阅并记录，不自动判实现错误）：PN oracle ≥
   各 PU 方法；同方法 c 增大时 Accuracy 不明显下降；组内相对排序与锚点排序秩相关（软提示）。

对照结论写入聚合报告，按协议 §5.7 区分"文献事实 / 本实验观测 / 推断"；对照矩阵版本与
resolved 单元写入 manifest（manifest 侧 2026-09-19 已接线：runner 按选择协议分列写入，
入口脚本开跑前校验覆盖；聚合报告侧仍属 P2.2）。

## 3. 决策记录（需讨论后确定）

| # | 决策 | 内容 | 日期 |
|---|---|---|---|
| D1 | **ADNI 数据获取** | **该数据集的获取似乎比较麻烦**，我会咨询一下学长。我目前的查证结论是（2026-09-08）：需通过 ADNI LONI 官网（adni.loni.usc.edu）在线申请——科研机构身份 + 接受数据使用协议（DUA）+ 研究用途描述，由 ADNI 数据共享与出版委员会（DPC）评审约 1-2 周，批准后经 LONI IDA 下载；限制：不得商用/重新分发、年度更新。决定后若申请通过，ADNI 加入后续实验矩阵；届时矩阵按"7+1"处理 | 2026-09-08 |
| D2 | 协议分工执行口径 | 数据准备协议 §2.4"工具箱不负责切分"字面与参考实现并存，产生了一个矛盾点——我会修改/补充协议的说法，并和学长说一声 | 2026-09-08 |
| D3 | 依赖锁与 CUDA 环境落地 | **uv.lock 已入库**（2026-09-08，chore(deps)）：替代 requirements.txt，PR 快层 CI 用 lock 确定性、nightly `--no-lock` 重新解析验证"最新可解析"（ADR-0012 修订）。**torch CUDA 配置**：pyproject `[tool.uv.index] pytorch-cu(cu126)` + `[tool.uv.sources]` 仅 `sys_platform=='win32'` 生效（CI Linux/macOS 保持 PyPI CPU 版；win 上 torch 2.14.0+cu126）；torchvision 并入 torch extra。多环境（T600/HENG958 主力机）由此保持一致 | 2026-09-08 |
| D4 | issue #41 阶段 A/B 拆分 | 阶段 A（P2 pilot 前置，P2.0a/b/c）：版本化执行矩阵 + runner 强制消费 + manifest 扩展 + CIFAR adapter 接线 + label_semantics_plan P1+P2 提前 + 对照矩阵预注册；阶段 B（P3 前置）：label_semantics P3+P4（pipeline 层检查 + 文档收口）。issue #41 于阶段 B 完成后关闭 | 2026-09-14 |
| D5 | Self-PU `input_ndims` 恢复 `{2,4}` | 审阅 P1#1：模板定义该字段为"支持输入维度"，4D 展平是 fit 的实际公共行为；"非原生 CNN"由 `native_architectures={"mlp"}` 承载（修正 issue #38 的收窄，代码随 fix 分支 PR） | 2026-09-14 |
| D6 | PU-Bench PN 行降级为背景参考 | 其 `pn` 用 `val_proxy_acc` 选模、我方 oracle 用 `clean_val_accuracy`，数值不可直接对比（`pn_oracle_integration.md` 既有决策）；不参与「交叉验证对照」判定规则第 1 条的数值裁决 | 2026-09-14 |
| D7 | SAR 标记频率口径修正 | 复核 PU-Bench 论文与锁定代码 `2d95a19`：`config/datasets_vary_e/*.yaml` 均使用 `c_values: [0.05, 0.5]`；原协议 `{0.1,0.5}` 与参考实现不一致，修正为 `{0.05,0.5}`。SCAR 主实验 `{0.1,0.3,0.5}` 不变；issue #43 按修正后口径实施 | 2026-09-15 |
| D8 | SAR 执行路径设计（issue #43） | `--labeling-mechanism` 与 `--method` **正交**（机制是实验自变量，SAR 行可跑任意 survey 方法）；SAR 强制 OA-only（协议 §2.3 下 PA 仅可诊断，v1 不产出 PA 日志）；SAR c 只接受**规范 token** `{0.05,0.5}`，目录按用户输入 token 命名（`c_0.05` 不得被格式化为 `c_0.1`），同值异拼写（`0.05`/`5e-2`）拒绝；`c_requested_token` 由脚本在运行成功后回写 manifest（runner manifest schema 为固定白名单，不改 runner，降低与 P2.0a 冲突） | 2026-09-15 |

## 4. 存在的开放问题与风险

0. **P2.0a 后续补充（2026-09-16）**：`survey-v1.1` 已保存逐 epoch 权重、双 teacher，
   支持 PA/OA 独立选择与恢复；实际完整预算/快照覆盖才解除相应 checkpoint 门禁。
   PA 仍是 PU 分离度代理，正式 Accuracy/阈值准则未实现，新增专用 `formal_blockers`；
   （**2026-09-20 更新**：准则本体已由第 0j 条实现，残留为合作者签署与 P2.0c 裁决。）
   CNN oracle 仍未接入，full-batch/经典路径没有同预算/backbone oracle。
   签署材料与逐条决定见 [复核包](p2_0a_review.md)，存储/加载/测试见
   [checkpoint 交付](epoch_checkpoint_delivery.md)。不得只删除阻断字段升级结果。

0a. **HENG958 独立复核（2026-09-16）**：P1.1 GPU 能力与 P1.3a nnPU/Self-PU 台账已复核；
    当前服务器全局 PyTorch 与 Linux lock 版本不同；Linux lock 的 CUDA 13 组件还高于
    前次记录驱动 550.54.14 支持范围。P2.1 前须共同决定环境兼容路线，并完成 frozen-lock
    GPU 验证，不能把旧全局环境 smoke 视为替代。
    P1.4 所需 split 产物未在当前工作树中；CIFAR 历史 manifest 的通道统计审计错误已由
    issue #52 的重建与验收关闭（5 个 seed 已核验并迁入标准路径），但制品不在版本库中，
    HENG958 可执行性复核仍等待产物同步。
    证据、命令和
    解除条件见 [独立复核记录](heng958_independent_review.md)。

0b. **P2.0d 负责人复核（2026-09-16）**：SAR-LBE-A/B 与方法正交、规范 c token、
    OA-only、生成审计和确定性均通过脚本级测试；新增 `survey-v1.1` 版本化端到端回归。
    该复核不解除 P2.0a/b/c 对正式跑批的门禁，详见 [独立复核记录](heng958_independent_review.md#4-p20d-sar-oa-执行路径复核)。

0c. **P2.0a 审核修改（2026-09-16）**：当前矩阵升为 `survey-v1.2`，KLDCE 三行在缺少
    `flip_probability` 安全绑定前不可运行；Self-PU 按每 epoch 两次抽样优化更新单列预算和
    比较组；矩阵关键字段严格校验，`selection_spec` 明确为描述性文档。IMDB split manifest
    记录真实向量是否 L2 归一化及来源，但历史 split 不自动更新。详情见 [复核包](p2_0a_review.md)。
    上述修改已由合作者完成定向复验并签署（见下条 0d）。

0d. **P2.0a 签署验收（2026-09-17）**：`review_status` 由 `pending_collaborator_review`
    改为 `accepted`；`collaborator_review` 移出 `formal_blockers`，新增
    `linux_frozen_lock_environment_deviation`（Linux 分支按方案 3 记录环境偏差并保持
    正式阻断）。协议摘要 `faa9084c…a42ff16` → `b5b6b5f4…ed2ff20`。签署对象为**交付提交**
    `83236d8`，**验证 HEAD** `0c90f27`（两者不同是预期的：交付对象不因后续修复前移）。
    R1–R10 逐条结论与 overall decision 见 [复核包 §4](p2_0a_review.md#4-签署记录)。
    接受 P2.0a 工程交付，**不放行正式 P2.1**；R5（P2.1 聚合入口须实现并强制调用分榜门禁）
    与 R8（跑批前置磁盘检查、checkpoint 覆盖不变量改强制并补测试）记为 P2.1 前置条件。
    继续保持阻断：R9、P2.0b、P2.0c、缺失的 CNN/full-batch oracle、完整 Self-PU OA
    meta-reweighting、Linux frozen-lock 环境偏差、P1.4 制品统一重建（IMDB/Spambase 待建；
    issue #52 的 CIFAR-10 部分已关闭）。
    （**2026-09-20 更新**：本段所列的 R9 准则本体已由第 0j 条实现，残留仅剩合作者签署；
    P1.4 制品重建与跨机交付见第 0f、0g 条。）
    IMDB 制品层（`data/splits/imdb/`）不含返工新增的有效口径字段，并入 P1.4 三数据集
    统一重建，验收按代码与测试层进行。

0e. **R5/R8 前置条件兑现（2026-09-19，工程层）**：R8 由 PR #59（`0f42adf`）兑现；
    R5 由 `scripts/aggregate_survey_runs.py` 兑现——按 `comparability_group` 分组、
    按 `(seed, c)` 细分单元，对每个单元强制调用 comparability 门禁与分榜门禁，
    `--diagnostic` 只放松正式资格。兑现过程修复四类缺陷（每个 manifest 造一个 spec
    导致的重复方法、`fullbatch` 的描述性 `batch_size`、`classical` 组跨四个预算族、
    分榜门禁比较 `split_sha256` 因而只能在单元粒度调用），并补组级跨单元一致性检查；
    候选全部失败、只留空 `selection` 的失败记录不再被当作结果聚合。
    本记录不改变 2026-09-17 的签署结论，0d 所列其余阻断项继续生效；
    见 [交付记录](p2_0a_delivery.md) §6。

0f. **P1.2 字段补齐与 P1.4 三数据集统一重建（2026-09-19）**：split manifest 新增
    `provenance` 块——来源 URL、版本、引用、许可状态，以及本地下载记录的
    `sha256 / bytes / downloaded_at`。P1.2 四项要求里的标签映射未重复记录：manifest 顶层的
    `positive_classes` / `negative_classes` 就是它，另记 `label_semantics` 说明这些 id 的含义。
    许可按官方页面原文逐条记录：**UCI 明确声明 CC BY 4.0**（含署名要求），Stanford sentiment 与
    Toronto CIFAR 两页确无许可条款（后者的第三方标注互相矛盾，不转引）。三条都记下查证日期；
    无许可者留空许可名并保留日期，以区别于「没查」。`source_url`/`version`/`citation`/`license`
    四项均入 manifest。可复核的发布事实入 `datasets.py` 目录（进版本库），机器本地的下载摘要仍来自
    `data/raw/*/provenance.json`，随制品走（含字节实际来自哪个 URL——IMDB 走的是镜像，
    制品里只印落地页会把读者引到错的主机去复算摘要）。
    随后三数据集 × 5 seed 统一重建：`uv run python scripts/prepare_survey_splits.py
    --datasets spambase,cifar10,imdb --seeds 0,1,2,3,4`。重建前先跑到临时目录做对照，
    **60 个 npz 逐字节相同**（管线确定性），差异只在 manifest：IMDB 的 `preprocessing`
    补齐 `effective_output_normalization` / `normalization_source`（即 0d 所记缺口），
    其余除新增 `provenance` 块外逐键相同。取代前的 15 份 manifest、摘要对照与验证口径见
    `data/archive/split-manifests-pre-p1.2-20260919/`。

0g. **split 制品跨机传输与接收端校验（2026-09-19）**：`data/` 按设计不进版本库
    （`.gitignore` 注明 never distributed），制品一律带外传。接收端此前校验的是制品的**形状**——
    `run_survey_experiment.py` 会走 `validate_bundle`（四路结构、索引不重叠、`test` 不得用于选择）
    并比对 manifest 的 `dataset`/`seed` 与请求——但**没有任何完整性校验**：
    `indices_sha256` 被原样抄进 run manifest 从不与 `.npz` 重算比对，`--split-ref` 传入的 JSON
    连 schema 都不校验（既有测试甚至断言可塞任意 `{"note": ...}`），因此截断、损坏或与数据不符的
    制品会被静默接受，只在结果异常时才暴露。
    新增 `scripts/survey_splits_archive.py`（逻辑在 `pu_toolbox/experiment/split_archive.py`）：
    `pack` 产出逐文件 `sha256`/大小索引与**确定性** tar（tar 成员元数据归零、成员按序，
    同一棵树在不同时间、不同机器打包逐字节相同——归档摘要因此可对外公布）；
    `verify` 在落地端**双向**校验（索引描述而树上没有、树上多出而索引没描述，后者含改名/复制出来的
    多余 split 或整个数据集），逐文件比大小与摘要，并**从 `.npz` 重算索引摘要**与 manifest 比对，
    报告**全部**问题而非第一个，对截断、同尺寸损坏、不可读文件一律报告而不抛异常。
    索引与归档摘要须记入仓库后再发送：随字节同行的摘要只能证明传输无损，不能证明发出去的是对的。
    两条边界必须一并说明：`X` 只由文件摘要兜底，**打包之前**就存在的损坏会被 `pack` 背书而非被查出；
    本工具**没有接进跑批路径**，是操作者手动跑的，没人跑 verify 的交付仍是没人验证过的交付。
    **载体与交付（2026-09-20）**：本项当初只交付与载体无关的两端工具、载体留待决策；现已定为
    **网盘带外传**（交大网盘，需 jaccount 登录），三个归档已发送（cifar10 813 MB / imdb 343 MB /
    spambase 1.6 MB，合计 1.2 GB）。逐文件索引与归档摘要已于发送**之前**入库
    （`docs/research/pu_survey/data/split_artifacts_index.json`，含各归档摘要与产出 commit），
    接收端据此校验，比对基准不随字节同行；跑批主机路线仍待决策。
    接收端复核仍按 `heng958_independent_review.md` 的解除条件执行（合同测试 + 每模态至少一个
    `run_survey_experiment.py` smoke），本项只解除「制品怎么过去、怎么证明没坏」这一段。

0h. **全 pilot 跑批驱动与 checkpoint 磁盘预算（2026-09-19）**：`run_survey_experiment.py`
    单次调用在**第一个失败处即中止**（`run_survey_experiment.py:656` 的 `return 1`），跑全 pilot 会让
    一次失败带走后面全部排队运行；`scripts/` 也一直只有单单元入口，全 pilot 无编排载体。
    新增 `scripts/run_survey_pilot.py`（计划逻辑在 `pu_toolbox/experiment/pilot_plan.py`）：
    从协议枚举 **645 次运行**（18 个非 oracle 可运行单元 × 5 seed × 7 个 c token
    ——scar 3 + LBE-A 2 + LBE-B 2；加 3 个 oracle 单元 × 5 seed。oracle 由 `--oracle` 定义于 clean
    视图，**不参与 c 网格**），按单元与机制分批，**批次只覆盖待跑的运行**：
    脚本执行的是 seeds × c 的笛卡尔积且自身没有逐格完成判定，所以只有待跑集恰好等于完整网格时
    才合并成一次调用，半完成的单元按 seed 拆开，否则会把已完成格子再跑一遍。
    **已完成按 manifest 判定，不按目录存在判定**：预检失败会写 `rejected_versioned_pilot`；
    而「候选全部 excluded」（每次 attempt 都失败）**仍写 `versioned_pilot`**，只是 `selection` 为空——
    只看 mode 会把这一格记成已完成且永不重跑，故判定同时要求 `selection` 非空。
    识别不了的记录一律算未完成（宁可重跑，不留空洞）。此外**要求 manifest 记录的
    `split_sha256` 仍等于磁盘上该 (dataset, seed) split 的 `indices_sha256`**：
    一次运行只对它所跑的那份数据构成证据，重建 split（P1.2/P1.4 刚做过）之后旧 manifest
    描述的是本 pilot 已不再持有的数据，算作完成会让 pilot 自称跑完却握着一批不属于任何
    split 的结果。反之，若改用「预测结果目录」判定，
    一旦实现与脚本漂移就会朝**跳过工作**的方向静默出错，与 §4 第 0e 条聚合入口被真实 pilot
    目录树打穿是同一类失效。默认一批失败即停（`--keep-going` 继续），收尾按 manifest 重扫并报数。
    驱动**不替协议取值，也不越过协议去补记录**。π 是多数方法的硬门禁
    （registry `requires_class_prior`），协议 §3.1 已把它定义为**数据生成 metadata**：
    分层划分前完整二元化池的正例经验比例、所有 seed 共享的常量，并**明文禁止从任何子集反推**。
    但此前**两个制品层都没记它**：split manifest 只有各子集的 `role_positive_rates`，
    run manifest 的 `generation` 只记 `c_realized` 等、三个 π 一个都没有——而 §3.1 要求
    「每次运行的 metadata 中记录三者及实际实现的 c」。本轮补上能补的那一半：
    `prepare_survey_dataset` 在**唯一能看到完整池的地方**记录
    `class_prior.{population, population_basis, train}`；驱动**从 split manifest 读取** π，
    `--class-prior dataset=value` 降级为覆盖（与记录值冲突时告警，因为协议把它定为每数据集常量），
    两者都没有时在**开跑前**一次性检查全部计划运行并拒绝启动，而不是跑到第一百个才失败。
    π_U 故意不入 split manifest：它是某次运行标签视图的性质，记在这里会被当成数据集常量读。
    `--device`（脚本默认 CPU，而 cifar10 行需要 GPU）同理透传。
    **run manifest 侧的 π**：按 §3.1 它应记录 π_population / π_train / π_U 三者。第 0j 条随
    `split_ref` 内联与 selection 块的 `class_prior{population, source}` 补上了 **π_population** 这一半；
    另两个仍缺，属 runner 的 manifest 白名单（P2.0a 绑定范围），本项不改，作为协议问题上报。
    **磁盘预算**（`uv run python scripts/run_survey_pilot.py --dry-run`，与 runner 跑前门禁同源——
    同一个 `unit_checkpoint_bytes` + `checkpoint_disk_requirement`）。645 次运行中 **330 次写 checkpoint**，
    另 315 次不写（`lbe` / `pusb_kernel` / `upu` 三个闭式单元 × 3 数据集 × 35 次；闭式与核方法不留
    逐 epoch 状态）。三个数必须分开看，混用会朝相反方向错：

    - **累积 1280.6 GiB**（cifar10 1274.4 / imdb 5.3 / spambase 0.8），单次峰值 **17.58 GiB**
      （`cifar10/self_pu/cnn_feature_adapter`，两份 teacher）。这是「一次成功即一次 attempt」的占用：
      `checkpoints.py` 没有任何清理逻辑，离线 selection 之后仍需这些文件，故逐次累加成立。
    - **跑前门禁要求 35.16 GiB 空闲**（单次峰值 × 2）。`DEFAULT_CHECKPOINT_ATTEMPTS = 2` 是给重试的
      预留，而重试确实写进独立的 `checkpoints/attempt-*` 目录（同一格重跑就多一份，且永不复用），
      所以预留不是虚的；但它是**上限**，不是常态占用。

    **这两个数对 adapter 行是上界，不能当真实占用用。** `unit_checkpoint_bytes`
    （`survey_protocol.py:60-61`）对任何 `backbone` 以 `resnet18` 开头的行一律按 45 MiB/epoch 计，
    但 6 个 `cnn_feature_adapter` 行**从不保存 ResNet**——adapter 只一次性提取冻结特征，
    之后训练的是 MLP head（CLI 对 adapter 行传 `encoder=None`）。一次独立复核用**真实写入器**
    （`EpochCheckpointTrainer` + 真实组装的 adapter 行模型 + 512 维特征）实测出倍数：

    | 行 | 写入组件 | 每文件 | 单次成功运行 |
    |---|---|---|---|
    | `cifar10/self_pu/cnn_feature_adapter` | teacher_1, teacher_2 | 265,695 B | 101.4 MiB |
    | `cifar10/dist_pu` / `pn_oracle`（adapter） | model | 265,655 B | 50.7 MiB |
    | `cifar10/nnpu/native_cnn` | `Sequential(encoder_, head)` | 44,784,523 B（11,177,025 参数） | 8.34 GiB |

    即该常量对 adapter 行**高估 ≈178×**，对 native_cnn 行只高约 5%（原设计如此）。
    据实测修正后：**累积 ≈319 GiB**（cifar10 ≈313）、**单次峰值行是 `cifar10/nnpu/native_cnn`**
    （8.34 GiB），门禁预留 ≈17.58 GiB。**向队友报磁盘需求时应以 ≈319 GiB 为准**——
    下表头条的 1280.6 GiB 大 4 倍，会直接影响主机决策。
    单组件 native CNN 行按常量 8.79 GiB/次，与 `epoch_checkpoint_delivery.md:44-45` 的
    「8–9 GB/候选/seed」一致。口径与前提：候选数取协议现值 1 个、组件数取各方法类声明的
    `epoch_components`；`unit_checkpoint_bytes` 声称是下界（忽略文件系统开销），
    但对 adapter 行它同时是上界——这两个方向相反的偏差都源于同一个常量。
    该常量属 P2.0a 已签署的绑定且被既有测试锁住，**本项不单方面修改**，作为协议问题上报：
    需要协议负责人决定是按 `training_path` 区分，还是接受保守值。
    队友侧主机已在 [P2.0a 交付记录](p2_0a_delivery.md) 记录（8 张 RTX A6000、每张约 48 GB，
    **与其他任务共享**，当时仅暴露物理 0 号卡），但**该服务器的可用磁盘容量全仓文档均无记录**，
    能否容纳需 HENG958 侧确认；GPU 独占窗口同样属其侧决策。

0i. **切分流水线的两处效率修复与一处一致性缺陷（2026-09-20）**：

    1. **重复加载**：`scripts/prepare_survey_splits.py` 的三个 `load_*` 原本写在
       `for seed in seeds` 循环**内部**，而加载与 seed 无关。IMDB 因此每 seed 重解一次
       `aclImdb_v1.tar.gz` 并读五万个成员（实测约 2.5 分钟/遍），CIFAR-10 每 seed 重读 813 MB
       pickle（约 5 秒/遍）。现外提到 seed 循环之前（`_load_dataset`）。
       **口径要说准**：大头是 IMDB 的 tar，CIFAR-10 的重复加载并不显著——上一轮 15 份重建的
       IMDB 阶段耗时约 15 分钟，而那**全部是缓存命中下的读 tar 时间**，与编码无关。
    2. **缓存键含顺序**：`encode_survey_texts` 的键取的是**整个有序文本列表**的摘要，而
       `prepare_text` 传给它的顺序由 seed 决定（各 role 依次拼接），于是「同一批文本、不同排列」
       成了不同的键，每个 seed 都重编全部语料。缓存里 5 条各 76,800,128 字节的 npy，
       正是「每 seed 一份完整编码」。改为按**排序去重后的语料**作键、命中时按索引装配回调用方顺序；
       schema 升至 1.1（键语义变了，旧键失效，一次性重编）。
    3. **顺带查出的真缺陷**：旧方案下**同一条文本在不同 seed 的嵌入值不同**。实测 IMDB test 集
       （indices 跨 seed 完全相同）的嵌入跨 seed 最大差 **1.583e-07**，即文本特征取决于它被编进
       哪个 seed 的 batch；新方案下同一文本在任何 seed 下**逐值相同（0.000e+00）**。
       这不是提速的副产品，而是本次更重要的结果：嵌入现在是「文本 + 模型 + revision」的纯函数。

    4. **又一处可复现性缺陷（由 2 暴露，非 2 引入）**：`_atomic_write_json` 以 `sort_keys=True`
       写缓存元数据，但**未命中**路径返回的是内存里插入序的 dict、**命中**路径返回的是从文件读回的
       排序序 dict。同一内容、不同键序，而 `prepare_text` 把它拼进 manifest 的 `preprocessing`，
       于是**同一个 split 准备两次，manifest 字节取决于那次编码是否命中缓存**——在传输索引按摘要
       比对之后，这会让内容相同的两份制品摘要对不上。已让两条路径都返回「缓存文件里存的那份」，
       并以单测钉住键序一致（该测试在修复前会失败）。

    制品影响：本批重建新增 `class_prior`，且 **IMDB 的 13 个 npz 字节改变**
    （max|ΔX| ≈ 1.7e-07，量级参照 |X| ≈ 0.27，即 float32 舍入级别；`indices`/`role_sizes`/
    `role_positive_rates` 全部不变，切分本身无改动）。CIFAR-10 与 Spambase 的 40 个 npz
    **逐字节不变**。取代前的 manifest 与前后对照见
    `data/archive/split-manifests-pre-p1.2b-20260920/`。
    代价与收益：缓存占用由 5×73 MiB 降为 73 MiB；旧键的 5 个文件（367 MB）成为死重，可删。

0j. **PA 正式选模准则实现（2026-09-20）**：R9 的**准则本体**落地。此前 PA 用
    `pu_val_separation`（标记正例组均值 − 未标注组均值）选模且不选阈值，与预注册不符。
    现按参考文献 1（Wang et al. 2026）Definition 1 的 **OS 分支**（协议 §2.3 统一采用 OS 数据生成）：

        PA(θ) = (2π/n'_P)·Σ_{D'_P} 1[f(x) ≥ θ] + (1/(n'_P+n'_U))·Σ_{D'_P∪D'_U} 1[f(x) < θ]

    第二项遍历**全部**验证样本（含标记正例），这是论文定义。π 是第一项的权重，因此它改变
    **argmax 而非仅尺度**——这正是 π 必须 fail-loud 而不能取默认值的原因。代入完美分类器得
    PA = ACC + π（常数平移），命题 1（PA 的排序与 ACC 一致）由此成立。
    **π 的取值链**：run 自己的 `class_prior`（与训练同一常数，使 `--allow-prior-override` 对选模
    同样生效）→ `split_ref.class_prior.population`（§3.1 的数据生成 metadata，split 制品是唯一
    能记录它的地方）→ 都没有则**在训练前 fail-loud**（runner 预检，`ProtocolPA.select` 内另有第二道）。
    预检显式跳过 clean view：那里 PA 由自己的视图守卫拒绝，报错才指向真正的缺陷（generator/protocol
    错配）；预检也排在协议/配置/能力检查之后，以免用"缺 π"遮蔽更根本的缺陷。
    **与 OA 同构**：每 checkpoint 在各自 val 侧 min-max，共用同一阈值网格与 val 侧仿射常数（test
    阶段复用），tie-break 遵循 `selection_spec.tie_breaking` 的"最早"。`split_ref` 因此新增内联
    `class_prior`，run manifest 的 selection 块新增 `class_prior{population, source}`。
    **摘要变更**：`pa_criterion` 与 decisions 第 7 条改写，canonical digest
    `b5b6b5f4…ed2ff20` → `c15b0c9e…eaff529`，`survey_comparison_v1.json` 的绑定同步重绑
    （`survey_comparison.py` 双向校验版本与摘要，两者必须同一 commit 落地）。`protocol_version`
    保持 `survey-v1.2`：准则是预注册的，本次是让实现符合它，不是改协议。
    **残留阻断**：合作者尚未签署；P2.0c 的 54 条 `blocked_pending_pa_criterion` 映射**本次不动**
    （`p2_0c_delivery.md` 明确"不得因其他项通过而被顺手删除"），留待 P2.0c 复核统一裁决。
    代码里的阻断串由 `PA_separation_proxy_not_preregistered_accuracy_threshold` 换为
    `PA_criterion_pending_collaborator_acceptance`。
    **这是行为变化而非纯签名变化**：PA 选出的 candidate/epoch/threshold 会与旧实现不同。已知一例
    是候选池测试——`multiplier` 只缩放分数（`m·(10x₀−5)` 的符号与 m 无关，两候选 predict 相同），
    旧实现奖励分数范围因而选它，归一化后两者完全相同，于是平手取最早候选，与 OA 的行为一致。

1. **GPU 算力/显存**：shuidisjtu 本机 T600（4GB）不够强，所以主要进行轻量批与开发验证的工作，
   显存不足时（批大小/并行）需在实验记录中说明资源限制；
2. **数据集获取确认**：20News/IMDB/Connect-4/Spambase 的版本与标签编码需与协议锁定映射核对；
   数据版本或标签编码变化时必须提供映射转换审计记录（协议 §2.2 要求）
3. **结果可解释性**：pilot 结果必须区分"文献事实/本实验观测/推断"（协议 §5.7 要求，P2 起执行）

## 5. 留痕约定

- 每个实验产物：manifest 固化（协议 §5.6：代码 commit、依赖锁文件、Python/PyTorch/CUDA/GPU、
  数据与方法配置、注册表版本、seed、split/label manifest、选择 artifact、schema 版本）
- 结果按四组存储（SCAR-PA / SCAR-OA / SAR-OA / PN oracle），跨数据集只比较趋势，不生成总排名
- 交叉验证对照结论随聚合报告归档（§2 末「交叉验证对照」三档结论），区分文献事实/本实验观测/推断
- 本计划随执行更新：每完成一个阶段、每产生一个决策，更新 §1/§2/§3 相应条目
