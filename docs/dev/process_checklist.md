# 进度清单

> 总体策略：**framework-first**——先完成稳定框架与 API 契约，用 mock estimator 跑通链路，再逐个集成论文算法。当前 21 个注册方法均已完成 clean-room 核心实现（NATIVE）；另有一个隔离的联合漂移 research 求解器，后续重点是官方数据、历史环境和 paper-like benchmark。新接入的 GradPU、PUET、VPU、PULDA 仍属实验性子集，未完成 Survey P3.1/P3.2 正式验收。
> 实际执行顺序与原始路线图有调整：优先实现 PU 特有的风险估计方法（工具箱核心差异化能力），经典分类器包装器后移。
> **Method Card 为可选文档**，新算法接入不要求必写。

## 阶段历史（已闭环）

- Phase 0 ✅ 项目骨架：pyproject + Core 基类 + Registry（初始 15 个 api_only 占位，现已按实现状态升级）+ 测试框架
- Phase 1 ✅ 核心 PU 风险估计：Elkan-Noto / uPU / nnPU / ReCPE / PNU / PU splitters / metrics / minimal examples
- Phase 2 ✅ 部分：penL1 类先验与算法推荐器完成；三经典包装器与 TIcE/AlphaMax 列 v1 范围外
- Phase 3 ✅ 机制就绪：PUSB benchmark 全链路（来源锁/manifest/shard 聚合/断点续跑/审计器）；官方数据全量运行依赖外部，见下
- Phase 4 ✅ 推荐与诊断：Data Profiler / SCAR-SAR 提示 / 推荐器 / 诊断报告 / 敏感性分析
- Phase 5 ✅ SAR：数据模拟器、PUSB/LBE/Centroid/LLSVM 接口与 SCAR vs SAR 对比 benchmark
- Phase 6 ✅ 深度 PU 大部分：Self-PU / Dist-PU / InfoMax PU / WConPU / DGPU 全链路（clean-room 多 seed、Fashion-MNIST 3-seed smoke、InfoMax 暂定协议 20 seeds）；剩余见下
- Phase 7 ✅ 分布漂移感知 PU 第一版：OOF 域审计、有界边际相对权重、ESS/覆盖门禁、
  `ShiftAwarePUPipeline`、`shift-audit` CLI 与三类产物；联合漂移动态训练列后续研究
- Phase 8 ✅ 分布漂移决策扩展：配对加权对照与 `shift-run`、窗口监控、双域先验/标记
  机制分解、不确定性拒绝/主动复核；新增研究级类别条件联合漂移近似求解器
- Phase 9 ✅ 联合漂移研究执行层：论文式动态共享特征目标、四类对照与五类消融、公开表格
  数据多 seed/CI 协议、双域先验 bootstrap，以及部署监控 CLI/UI

> 逐条明细与批次历史见 git log(压缩前旧路径:git log --all -- docs/project_management/process_checklist.md)。

## 未完成项

- [ ] Phase 3 官方数据/历史环境全量运行（依赖外部官方数据与历史环境提供，非工具箱缺口）
- [ ] Phase 6 WConPU 官方视觉 + DGPU EDM paper-like 全量（依赖 CUDA/授权数据）
- [ ] InfoMax 未公开类别分组、batch size 与 KM 变体核对
- ⚠️ v1 范围外：Phase 2 三经典包装器 + TIcE/AlphaMax 类先验估计

## 发布状态 (v1.11.0)

- Survey P2.0a 工程交付（未发布，2026-09-17 已签署验收）：版本化矩阵、runner 强制消费与参数锁、
  协议/预算/表征 manifest、真实 split/seed 对应、CIFAR 原生 nnPU 与随机冻结 adapter 接线、
  摘要校验缓存、共享 MLP oracle、路径/预算/标签摘要比较门禁；46 项 CPU 专项通过，
  沙箱外 4 项 GPU smoke 通过（含既有 CNN13 测试），新增 3 项 GPU 测试。
  后续 `survey-v1.1` 补齐逐 epoch 权重保存、双 teacher 保留、PA/OA 独立选择和推理恢复；
  新增 checkpoint 专项 24 项、GPU 回归 8 项及完整快层 1532 项通过；
  正式结果仍受 P2.0b/c、PA Accuracy/阈值准则及未完成路径阻断；
  R5/R8 两项 P2.1 前置条件已于 2026-09-19 工程兑现（聚合入口强制分榜门禁、
  跑批前磁盘检查、checkpoint 覆盖不变量改按声明校验），见交付记录 §6；
  CNN oracle 等缺少对齐对照的单元明确不可冒充上界。见
  [交付记录](../research/pu_survey/p2_0a_delivery.md)。

- Survey P2.0b 标签语义门禁 + P2.0c 交叉验证对照预注册（未发布，工程完成、合作者签署待办）：
  P2.0b 新增分类器 `label_semantics` 声明位、registry 同步与 runner 训练前按视图强制检查
  （PU 视图须 `"pu"`、clean 视图须 `"pn"`；第三方未声明估计器按 `"pu"` 保守处理），
  堵住 `SupervisedTrainer` 配 PU 风险估计器静默产出假 `pn_oracle` 的路径；
  P2.0c 为外部对照矩阵预注册，来源技术审计后部分锚点与行级映射退回待审。
  两者均不放行正式 P2.1。见
  [P2.0b 交付记录](../research/pu_survey/p2_0b_delivery.md)、
  [P2.0c 交付记录](../research/pu_survey/p2_0c_delivery.md)、
  [P2.0c 复核包](../research/pu_survey/p2_0c_review.md)。

- PN oracle 接入（未发布，随下一版本发布）：`CleanLabelGenerator` +
  `Generator.output_view` 视图声明 + `Trainer.trains_on_real_labels` 声明与
  runner 双向守卫（视图与 trainer 标签语义不一致即 fail-loud）+
  脚本 `--oracle` 入口与 `oracle_integration.json` 口径留痕。
  修复动机：原路径会把 SCAR 标记当作真实标签训练，静默产出错误的
  "全监督上界"（详见 docs/research/pu_survey/pn_oracle_integration.md）
- 双架构阶段 0 能力契约（未发布，随下一版本发布）：Registry 4 能力字段 +
  Pipeline 并行校验 + list-methods 能力列 + encoder 输出校验 helper +
  契约测试（详见 docs/dev/dual_architecture_plan.md §5）
- 双架构阶段 1 整理现有双架构实现（未发布，随下一版本发布）：build_encoder
  公共导出 + 报告 provenance 4 字段 + UI CNN 候选集元数据驱动 + CV fold
  训练隔离测试（详见 docs/dev/dual_architecture_plan.md §5）
- 双架构阶段 2 nnpu encoder 试点（未发布，随下一版本发布）：nnpu
  新增 encoder 参数（model 复用为 head，fit 内 Sequential 组合）、
  MLP/CNN 双架构声明、CV fold 隔离与 pipeline 端到端测试、
  gpu marker + CUDA 执行级测试（详见
  docs/dev/dual_architecture_plan.md §5）
- CNN 提示文案回归修复（未发布，随下一版本发布）：Pipeline cnn 报错与 CLI
  `--architecture` help 的候选方法提示改为由 registry 能力声明动态生成
  （原硬编码 wconpu/infomax_pu 遗漏 nnPU），并补错误文案/动态更新/排除
  api_only 的回归测试（issue #45 低优先级 UX 子项闭环）
- 双架构契约路线 B 收口（未发布，随下一版本发布）：issue #45 决策——
  `adapter_architectures` 不落地为元数据字段（从计划 §4.2/§9 与模板删除）；
  `encoder_parameter` 语义收窄为声明性元数据（注入依构造函数签名，不以
  该字段驱动）；计划 §5 阶段 4 补现状注记（实验层已实现
  `cnn_feature_adapter`，主链路不集成）；契约测试补字段集钉子与
  声明性语义钉子（详见 issue #45 与 docs/dev/dual_architecture_plan.md）
- Survey 语义统一（未发布，随下一版本发布）：issue #42 审计闭环——PUSB
  方法身份拆分：台账拆为 `pusb`（linear baseline，附加工程基线不入榜）与
  `pusb_kernel`（official-aligned RBF，pilot 行，需 π）；`run_survey_experiment.py`
  先验门禁改由 registry `requires_class_prior` 驱动（台账只做结果标注），
  未入台账方法 fail-loud；新增台账↔registry 一致性契约测试 6 条不变量
  （tests/contract/test_ledger_registry_consistency.py）；survey 文档双架构
  表述限缩（7 个 Survey 方法中仅 nnPU 原生 CNN）与 implementation_plan
  状态横幅（详见 issue #42 与 docs/research/pu_survey/survey_execution_plan.md）
- Self-PU CNN 声明收口（未发布，随下一版本发布）：issue #38 决策——
  Self-PU 不支持 native CNN（mlp-only）；`input_ndims` 收窄为 `{2}`
  （4D 展平仅为估计器层容忍，非声明能力），契约 pin 与台账同步；
  survey 图像行经 `cnn_feature_adapter` 与其余 5 法同组；4-D+mlp 的
  Pipeline 报错提示改为 registry 动态候选（修复 #45 修复时遗漏的
  第三处硬编码 wconpu/infomax_pu），补回归测试（详见 issue #38 与
  docs/dev/dual_architecture_plan.md 阶段 3）
- Self-PU `input_ndims` 契约修正（未发布，随下一版本发布）：审阅 P1#1——
  `input_ndims` 恢复 `{2,4}`（模板定义该字段为"支持输入维度"，4D 展平是
  fit 实际公共行为，声明 {2} 与行为矛盾且 runner 会拒绝本可运行的输入）；
  "非原生 CNN"语义由 `native_architectures={"mlp"}` 承载；契约 pin、台账
  `code_capability` 与 dual_architecture_plan 阶段 3 注记同步（详见
  docs/dev/dual_architecture_plan.md 阶段 3）
- SAR-OA 执行路径（未发布，随下一版本发布）：issue #43——官方脚本新增
  `--labeling-mechanism {scar, sar_lbe_a, sar_lbe_b}`（默认 scar，与 --method
  正交）；SAR 分支强制 OA-only（runner 默认是 `protocols or [PA, OA]`，
  故须显式注入 `[ProtocolOA()]` 而非空列表）；SAR c 仅接受协议 token
  `{0.05,0.5}`（PU-Bench vary-e，D7），目录按用户 token 命名并落
  `<mechanism>/c_<token>/seed_<seed>`；c 词法五重校验与机制×oracle 组合门禁
  在读数据/建目录/建模型前 fail-loud；生成器元数据统一审计词汇
  （`c_requested`/`n_labeled_requested` 未夹紧 vs `c_realized`/`n_labeled`
  夹紧后、`generation_seed`、`label_view_sha256`），posterior 输入支持任意
  ndim（4-D NCHW 展平后 fit/predict 同视图）；`c_requested_token` 由脚本在
  运行成功后回写 manifest（不改 runner）；测试抽取
  `tests/unit/experiment/_survey_script_helpers.py` 共享夹具并新增 SAR 脚本
  测试文件（basic/param/edge/determ 四类，RED→GREEN）
- **版本**: `1.11.0`（2026-08-29：pu-workflow skill 更新——新增可选扩展场景
  （漂移迁移 `shift-audit`/`shift-run`、部署监控 `shift-monitor`/`review`、基准审计
  `audit-benchmark`，各带强制检查点）、输入契约补充 NaN/Inf 拒绝、技能最低版本
  要求升至 `pu-toolbox >= 1.10.0`）
- **版本**: `1.10.0`（2026-08-29：传统 PU 第一次调优收尾——KLDCE b₀ 类对称修复
  （低先验全负根因）、契约 v2 基线重跑与 KLDCE 调优轮重跑（r3，默认参数即有效
  工作点）、Elkan-Noto 调优轮重跑（r2，`mode=weighted_retraining` 12/12 全单元
  confirmed）；第 6 步写回三轮全部落地——LDCE 组合默认（v4）、uPU squared（v5）、
  elkan_noto weighted_retraining（v6），每轮重锁基线 + 确认种子重跑 +
  companion 逐单元审计，当前对齐基线 baseline_v6，ADR-0016 闭环）
- **版本**: `1.9.0`（2026-08-27：新增七方法传统 PU 可复现 benchmark、锁定基线、
  数据泄露预检、断点续跑、配对统计比较与七轮调优证据（ADR-0016 verdict 留档）；
  新增 AP、balanced accuracy、Brier score、ECE 指标及概率可用性契约；KLDCE
  改为原生 SMO 内层求解并修复收敛诊断，LDCE 默认迭代上限提升；修正 Elkan–Noto
  等非零原生阈值模型的 PU 零一风险语义）
- **版本**: `1.8.0`（2026-08-21：新增 AISTATS 2025 联合漂移 PU clean-room 动态目标、
  对照/消融与公开数据 benchmark、双域 bootstrap 区间，以及 `shift-monitor`/`review` CLI
  和 UI 部署面板）
- **版本**: `1.7.0`（2026-08-21：新增配对漂移适配比较、窗口告警历史、双域 PU
  假设分析、不确定性/主动复核，以及明确标为 research 的联合漂移近似求解器）
- **版本**: `1.6.0`（2026-08-21：新增分布漂移审计、协变量加权 PU 工作流、
  `PUPipeline.sample_weight` 严格传递契约和 `shift-audit` CLI）
- **版本**: `1.5.1`（2026-08-16：验收修复——CNN 模型序列化、PUTuner 坏参数隔离、
  UI 运行历史持久化）
- **版本**: `1.5.0`（2026-08-15：新增 `classifier_params` 与 CLI
  `--classifier-param`，支持按注册名调整模型；新增 PU-aware `PUTuner`，搜索阶段仅做
  CV 并只重训最佳候选；新增 Streamlit 图形界面，支持数据上传、模型配置、参数搜索、
  指标与诊断展示，以及报告、预测和模型下载）
- **算法**: 21 个已注册方法，全部 native 实现
- **质量门禁**: 8 道（test_quality / doc_links / project_metadata / math_rendering / api_docs / skill_sync / baseline_configs / format）
- **v1 范围外**: Phase 2 三个经典包装器与 TIcE/AlphaMax 类先验估计
- **依赖外部**: Phase 3 官方历史环境，以及 WConPU CUDA/授权数据和 DGPU EDM/CelebA
  全量运行；InfoMax 暂定 Fashion-MNIST 20-seed 协议已执行

历史执行记录见 git log；关键决策见 [`docs/adr/`](../adr/)。
