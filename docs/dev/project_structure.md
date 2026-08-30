# Project Directory Structure

> 本文档是项目目录结构的权威定义。已实现/存在的文件如实列出，规划文件标注 `(planned)`。

## 1. 项目根目录

```text
pu-learning-toolbox/
  pyproject.toml
  README.md
  CONTRIBUTING.md
  LICENSE
  .gitignore
```

## 2. Python 包（`pu_toolbox/`）

```text
pu_toolbox/
  core/
    __init__.py
    base.py
    validation.py
    labels.py
    config.py
    exceptions.py
    random.py
    tags.py
    device.py                             (shared: resolve_device/resolve_device_name CUDA 自动检测单源)
  preprocessing/
    __init__.py
    pu_labeling.py
    selection_bias.py                     (SCAR/SAR propensity、标签与合成数据)
    profiling.py                          (兼容 summary 与 SCAR 筛查接口)
    data_profiler.py                      (结构化 PU 画像、问题与假设提示)
  prior/
    __init__.py
    recpe.py                              (native)
    pen_l1.py                             (native)
    kernel_mean.py                        (native: KM1/KM2 kernel-mean class-prior estimation)
  utils/
    __init__.py
    basis.py                              (shared)
    centroid.py                           (shared: MoM + 协方差原语, LDCE/KLDCE 共用)
    activations.py                        (shared: sigmoid_stable 数值稳定实现单源)
    serialization.py                      (shared: JSON 严格模式/Markdown 序列化助手单源)
  losses/
    __init__.py
    upu.py                                (native)
    nnpu.py                               (native)
    pnu.py                                (native)
    llsvm.py                              (native)
  estimators/
    classic/                              (2/2 native)
      __init__.py
      elkan_noto.py                       (native)
      llsvm.py                            (native)
    risk/
      __init__.py
      _class_prior.py                     (shared: 类先验推导与 1−2ph 稳定性检查, KLDCE/LDCE 共用)
      ldce.py                             (native)
      kldce.py                            (native: ACS + 原生 SMO + RBF kernel)
      kldce_smo.py                        (原生配对 SMO 求解器: 解析更新 + KKT 选择)
      dist_pu.py                          (native)
      upu.py                              (native)
      nnpu.py                             (native)
      pnu.py                              (native)
    bias_aware/
      __init__.py
      pusb.py                             (native: linear source-classifier baseline)
      pusb_kernel.py                      (native: official-aligned RBF PUSB adapter)
      lbe.py                              (native)
    deep/
      __init__.py
      self_pu.py                          (native: self-paced + meta reweight + distillation)
      infomax_pu.py                       (native: PURL + nnPU pipeline)
      weighted_contrastive_pu.py          (native core)
      vision.py                           (WConPU CNN13/ResNet 与 tensor augmentation adapters)
      dgpu.py                             (native orchestration + generator protocol)
      _validation.py                      (encoder 输出校验: 2-D/有限/feature_dim>=1)
    research/
      __init__.py                         (研究级估计器公开入口)
      joint_shift.py                      (软类别条件域比与交替 PU 更新的联合漂移近似)
      dynamic_joint_shift.py              (论文式联合权重/分类目标与共享特征动态训练)
      joint_shift_baselines.py            (trPU/tePU/fine-tune/MMD 对照与消融工厂)
    __init__.py
  metrics/
    __init__.py
    classification.py                     (已实现: PU risk/recall/precision + supervised wrappers)
  diagnostics/
    __init__.py
    benchmark.py                          (benchmark 产物、provenance 与 PU split 审计)
    report.py                             (数据/模型/指标诊断，JSON/Markdown 报告)
    sensitivity.py                        (类先验/平均标记倾向假设敏感性)
    shift.py                              (源/目标域 OOF 漂移审计、相对权重与 ESS 报告)
    domain_assumptions.py                 (双域类先验、平均标记倾向与敏感性差异报告)
    shift_monitor.py                      (固定参考域的窗口历史、变化量与分级告警)
    uncertainty.py                        (概率边际、拒绝预测与主动人工复核策略)
  model_selection/
    __init__.py
    comparison.py                         (PUModelComparator 多模型 CV 与最佳模型重训)
    split.py                              (已实现: PUStratifiedKFold + PUStratifiedShuffleSplit)
    tuning.py                             (PUTuner: PU-aware 参数网格、trial 与最佳报告)
  registry/
    __init__.py
    registry.py                           (含别名解析逻辑)
    metadata.py
    builtin_methods.py                    (17 论文方法元数据 + native 绑定)
  advisor/
    __init__.py
    recommender.py                        (算法推荐管线: recommend_methods / recommend_from_profile)
    rules.py                              (评分规则引擎: ScoringConfig + 评分/警告函数)
    _types.py                             (数据类: MethodCandidate / RecommendationResult)
  workflows/
    __init__.py
    pipeline.py                           (PUPipeline 顶层编排与类先验处理)
    _errors.py                            (工作流共享异常类型)
    _evaluation.py                        (指标解析、PU-aware CV 执行与聚合)
    _inputs.py                            (输入校验、splitter 准备与 CV provenance)
    _models.py                            (模型解析、参数校验与 estimator 构造)
    _reporting.py                         (报告组装与参数 provenance)
    report.py                             (报告数据类: PriorInfo/CVMetric/PipelineReport)
    shift.py                              (ShiftAwarePUPipeline 协变量加权编排与组合报告)
  cli/                                    (CLI 入口: argparse 子命令与工作流薄封装)
    __init__.py
    run.py                                (run 子命令: 双 CSV 输入、目录三件套输出、退出码 0/1/2)
    info.py                               (list-methods / list-priors 子命令, registry 实时读取)
    demo.py                               (make-demo-data 子命令: SCAR 演示数据)
    profile.py                            (profile 子命令: 数据画像 + SCAR/SAR 诊断, 写 profile.json)
    recommend.py                          (recommend 子命令: 算法推荐 + 类先验估计, 写 recommendation.json)
    sensitivity.py                        (sensitivity 子命令: 假设敏感性分析, 写 sensitivity.json)
    audit_benchmark.py                    (audit-benchmark 子命令: 实验产物与 provenance 审计)
    skill.py                              (skill 子命令: 安装内置 pu-workflow 技能到用户级 agent 目录)
    shift.py                              (shift-audit 子命令与三产物导出)
    deployment.py                         (shift-monitor/review 部署子命令与产物导出)
  ui/                                     (可选 Streamlit 图形界面，核心安装不导入 streamlit)
    __init__.py                           (数据/配置辅助函数导出)
    app.py                                (Streamlit 页面流程协调)
    configuration.py                      (运行配置与 Streamlit session state 适配)
    data.py                               (CSV/NPY 上传解析与校验)
    execution.py                          (普通/调参/比较三种运行模式调度)
    history.py                            (进程级运行历史:刷新保留、重启清空)
    parameters.py                         (构造器参数元数据与类型化控件)
    results.py                            (指标、诊断与下载结果渲染)
    runtime.py                            (后台线程、进度状态与取消控制)
    launcher.py                           (pu-toolbox-ui 启动入口)
    deployment.py                         (窗口漂移、coverage 与主动复核 Streamlit 面板)
  __init__.py
  run_config.py                           (UI/CLI 共用的可移植 JSON 运行配置契约)
  progress.py                             (线程安全进度快照、协作式取消 token)
```

## 3. 测试（`tests/`）

```text
tests/
  contract/                             # 契约测试 — 写一次，所有 NATIVE 分类器复用
    test_classifier_baseline.py         # fit/predict/decision_function/get_params 等
    test_capability_declarations.py     # 能力声明 4 组不变量契约测试
    test_build_encoder_export.py        # build_encoder 双层导出契约(mlp→None/ValueError/结构一致)
  estimators/                           # 按方法的测试（MATH/PROPERTY/API）
    risk/
      test_ldce_math.py                 # LDCE 算法正确性 (MATH: MoM, 协方差, m-更新, 梯度)
      test_ldce_core.py                 # LDCE fit/predict/属性/标签 (unit)
      test_ldce_api.py                  # LDCE 收敛/约束/API/错误/回归 (unit)
      test_kldce_math.py                # KLDCE 公式验证 + QP oracle + bias 恢复 (MATH)
      test_kldce_property.py            # KLDCE 约束/鲁棒性 (PROPERTY)
      test_kldce_kkt.py                 # KLDCE 真 KKT 残差(乘子恢复)+ ACS 收敛 (unit)
      test_kldce_smo.py                 # KLDCE 原生 SMO 求解器/等价/增量梯度 (unit)
      test_class_prior.py               # 类先验推导与质心分母共享助手 (unit)
      __init__.py
    research/
      test_dynamic_joint_shift_math.py  # 论文目标手算金标准、修正边界与动态训练
    __init__.py
  unit/                                 # 算法特有逻辑测试
    diagnostics/
      test_report.py                    # 诊断报告指标、输入契约与序列化
      test_sensitivity.py               # 假设扫描公式、边界与导出
      test_benchmark_audit.py           # 持久化 benchmark 产物审计测试
      test_shift.py                     # 漂移审计、相对权重、ESS 与序列化测试
      test_domain_assumptions.py        # 双域先验/标记机制分解、可行性与序列化
      test_shift_monitor.py             # 窗口 delta、告警、历史恢复与配置门禁
      test_uncertainty.py               # 拒绝覆盖、三类查询策略与逐行产物
    estimators/
      test_elkan_noto.py                # Elkan-Noto 特有逻辑
      test_upu.py                       # uPU 特有逻辑
      test_nnpu.py                      # nnPU 特有逻辑（含训练动态/早停）
      test_nnpu_encoder.py              # nnPU encoder 注入(Sequential 组合/4-D 边界/默认回归)
      test_bias_aware.py                # PUSB / LBE 特有逻辑
      test_pusb_kernel.py               # official-aligned PUSB 公式、CV 与确定性
      test_dist_pu.py                   # Dist-PU 特有逻辑
      test_self_pu.py                   # Self-PU pace/meta/EMA/三阶段训练
      test_deep_pu.py                   # InfoMax PU/WConPU/DGPU 接口与 registry
      test_deep_pu_vision.py            # WConPU 视觉骨干与张量增强
      test_vision.py                    # 统一深度编码器入口 build_encoder
      test_deep_vision_pickle.py        # 深度视觉模块 pickle 往返(E2/E3 回归;importorskip torch)
      test_llsvm.py                     # LLSVM 特有逻辑
      test_encoder_validation.py        # validate_encoder_features 单元测试(2-D/有限/维度边界)
      test_nnpu_gpu.py                  # nnPU CNN encoder GPU 执行级测试(无 CUDA 自动 skip)
    losses/
      test_nnpu_loss.py                 # nnPU golden tests (MATH + PROPERTY)
      test_upu_loss.py                  # uPU golden tests (MATH + PROPERTY)
      test_llsvm_loss.py                # LLSVM loss golden tests (MATH)
      test_pnu_loss.py                  # PNU loss golden tests (MATH)
    metrics/
      test_classification.py            # PU 指标测试
    model_selection/
      test_comparison.py                # 多模型比较、失败隔离与选择方向
      test_split.py                     # PU 切分器测试
      test_tuning.py                    # PUTuner 网格、选择方向与不可用指标
    ui/
      test_history.py                   # 进程级运行历史与收尾条目写入(D9 回归)
      test_ui_helpers.py                # 上传解析、类型化参数、配置状态与模型目录
      test_runtime.py                   # 后台运行、进度快照与协作式取消
      test_deployment.py                # UI 部署分析辅助函数与错误路径
      test_cnn_candidates.py            # CNN 候选集与骨架清单元数据驱动(registry 能力声明推导)
    prior/
      test_recpe.py                     # ReCPE 特有逻辑
      test_pen_l1.py                    # penL1 特有逻辑
      test_kernel_mean.py               # KM1/KM2 kernel-mean 类先验估计
    preprocessing/
      test_pu_labeling.py               # PU/PNU 标签生成
      test_selection_bias.py            # SCAR/SAR propensity、标记率校准与合成数据
      test_profiling.py                 # 数据画像统计
      test_data_profiler.py             # 结构化报告、质量规则与审计诊断
      __init__.py
    scripts/
      test_check_baseline_configs.py    # 基线配置一致性门禁脚本测试
      test_check_doc_links.py           # 文档链接门禁(orphan/md-link)脚本测试
      test_check_format.py              # 格式门禁(ruff lint+format)脚本测试
      test_check_math_rendering.py      # MathJax 渲染门禁脚本测试
      test_check_skill_sync.py          # skill 双份一致性门禁脚本测试
      test_check_test_quality_exemptions.py # 测试质量门禁豁免审查测试
      test_generate_structure.py        # 结构文档生成器(--check/--update)单元测试
    advisor/
      test_recommender.py               # 算法推荐器过滤、评分与输出
      test_scoring_rules.py             # 推荐评分规则与推荐器边界
      __init__.py
    cli/
      test_classifier_params.py         # classifier-param JSON 解析与错误路径
      test_cli_main.py                  # CLI 入口冒烟
      test_run_deep.py                  # run 子命令深度架构路径（.npy 输入 + 参数）
      test_info.py                      # list-methods / list-priors 子命令
      test_demo.py                      # make-demo-data 子命令
      test_skill.py                     # skill install 子命令（内置技能安装）
      test_audit_benchmark.py           # audit-benchmark 子命令单元测试
      test_profile.py                   # profile 子命令测试(pu-workflow 步骤 1)
      test_recommend.py                 # recommend 子命令测试(pu-workflow 步骤 2)
      test_sensitivity_cmd.py           # sensitivity 子命令测试(pu-workflow 步骤 4)
      test_shift_audit.py               # shift-audit 参数、错误与产物测试
      test_shift_run.py                 # shift-run 参数、错误与配对产物测试
      test_deployment_commands.py       # shift-monitor/review CLI 参数与产物旅程
    core/
      test_device.py                    # resolve_device 设备解析共享助手测试
    workflows/
      test_pipeline_report.py           # PipelineReport.summary() 先验可靠性上下文测试
      test_metric_availability.py       # 指标可用性条件(compute_metric + proba gate)
      test_architecture_capability.py   # 架构能力校验 gate(能力/签名漂移 fail-loud)
      test_report_provenance.py         # 报告 provenance 架构能力 4 字段(mlp 裸配/native_cnn 全配)
    test_basis_single_source.py         # 单一数据源 RBF kernel 公式一致性
    test_run_config.py                  # UI/CLI 可移植运行配置 schema 与序列化
  integration/                          # 跨组件集成（CLI + PUPipeline + registry + estimators）
    test_model_configuration.py         # 命名参数、必填参数与 PUTuner 确定性
    test_pipeline.py                    # PUPipeline 全流程/先验解析/错误/可用性/确定性
    test_pipeline_deep.py               # PUPipeline 深度算法集成（架构选择;importorskip torch）
    test_run.py                         # run 子命令进程内端到端（真实 CSV IO + 真实训练）
    test_cli_deep_save_model.py         # run --save-model CNN 保存回归（E2/E3;importorskip torch）
    test_ui_app.py                      # Streamlit 页面渲染（ui extra；无依赖时 skip）
    test_ui_history_flow.py             # UI 运行写入进程级历史（D9 接线;importorskip streamlit）
    test_pipeline_device.py             # PUPipeline/深度估计器默认值(设备自动检测、epochs)
    test_prior_params.py                # prior_params 转发集成测试(CLI --prior-param 后端)
    test_run_errors.py                  # run 子命令错误路径集成测试(用户输入友好失败)
    test_pipeline_sample_weight.py      # PUPipeline 逐折权重传递与语义拒绝测试
    test_shift_workflow.py              # ShiftAwarePUPipeline 审计/适配/目标评估集成
    test_joint_shift_classifier.py      # 联合漂移合成协议、有界性与确定性
    test_shift_comparison.py            # 配对加权对照、证据门禁与报告集成
    test_joint_shift_baselines.py       # 联合漂移四基线、消融、边界与确定性
    test_cv_fold_isolation.py           # CV 折间训练隔离(折内权重在变/模板不被训练/折间不泄漏)
    test_nnpu_pipeline_cnn.py           # nnPU 端到端 provenance 映射(cnn/mlp)+ encoder pickle 往返
  e2e/                                  # 真实子进程端到端用户旅程（CI nightly 运行）
    test_profile_script.py              # pu-workflow profile 步骤脚本（子进程）
    test_recommend_script.py            # pu-workflow recommend 步骤脚本（子进程,含 profile→recommend 链）
    test_sensitivity_script.py          # pu-workflow sensitivity 步骤脚本（子进程）
    test_e2e_quickstart.py              # quickstart 旅程（demo→auto run→report）+ CLI 错误旅程
    test_e2e_workflow_journey.py        # profile→recommend 链 + demo→profile→recommend→run 全链
    test_e2e_subcommands.py             # 真实子进程 e2e: profile/recommend/sensitivity
  benchmarks/                           # Benchmark runner 测试
    test_assigned_benchmark_runner.py   # 前五篇 benchmark runner 测试
    test_assigned_preflight.py          # 前五篇 paper-run 就绪审计测试
    test_deep_pu_benchmark_runner.py    # 深度 PU runner 测试
    test_deep_pu_model_selection.py     # 深度 PU runner 的 clean-validation 模型选择测试
    test_deep_pu_official_data.py       # 官方数据加载与 split 测试
    test_pusb_official_data.py          # PUSB 官方数据构造、runner 与 provenance
    test_pusb_table2_benchmark.py       # claim-safe PUSB Table 2 benchmark runner 测试
    test_pusb_table2_data.py            # 锁定 PUSB Table 2 数据集 loader 与采样审计测试
    test_joint_shift_public_benchmark.py # 公开数据多 seed/CI/样本重叠 benchmark
    test_traditional_pu_benchmark_runner.py # 传统 PU runner 状态机/resume/产物/失败隔离
    test_traditional_pu_data.py         # 数据协议: SCAR/SAR/PNU 形状、h、病态性
    test_traditional_pu_statistics.py   # 统计原语: 成功率/CI/配对差值
    test_traditional_pu_configs.py      # <<< 新文件,补注释
    test_traditional_pu_protocol.py     # <<< 新文件,补注释
    test_traditional_pu_compare.py      # <<< 新文件,补注释
    test_traditional_pu_resume.py       # <<< 新文件,补注释
    test_traditional_pu_leakage_audit.py # 泄露审计: 黑名单/重复样本/guard/preflight 负向测试 + CLI 门禁
    test_traditional_pu_tuning_round.py # 调优轮工具: 候选配置生成/退化率核对
    test_traditional_pu_tuning_rank.py  # 调优轮排名: §4 筛选链/scar-sar 混合网格
    tuning_helpers.py                   # 调优轮测试共享构造器(非测试模块)
    test_traditional_pu_tuning_round_methods.py # 调优轮逐方法注册 smoke 测试(nnPU/uPU/LLSVM/Elkan-Noto)
  __init__.py                           # tests 包声明(支持 tests.helpers 导入)
  conftest.py                           # 共享 pytest fixtures(种子/rng/数据 fixture)
  helpers.py                            # 数据工厂等普通函数(测试直接 import,不依赖 pytest)
  test_import.py                        # 导入冒烟
  test_labels.py                        # 标签规范化
  test_validation.py                    # 输入校验
  test_registry.py                      # 注册机制
  test_builtin_methods.py               # 注册表元数据
```

测试权威级别（pytest markers）：`unit`（算法特有逻辑）、`math`（手工计算 → 失败=代码bug）、`property`（数学不变量 → 失败=代码bug）、`contract`（API 契约）、`integration`（跨组件集成）、`e2e`（真实子进程用户旅程）、`slow`（慢速）、`paper`（论文复现）。

测试金字塔分层与 CI 映射：`unit` + `integration` 为 PR 快层（`-m "not slow and not e2e"`）；`e2e` + `slow` 为 nightly 顶层（`-m "slow or e2e"`）。

## 4. 示例（`examples/`）

```text
examples/
  minimal/
    01_elkan_noto.py          (Elkan-Noto 概率校正 + 加权重训)
    02_upu.py                 (uPU 凸风险最小化)
    03_nnpu.py                (nnPU 非负风险估计)
    04_pnu.py                 (PNU 半监督扩展)
    05_recpe_pipeline.py      (ReCPE 类先验估计 + uPU 联合流程)
    06_sar_simulation.py      (SCAR/SAR 标记机制与 propensity)
    07_data_profiling.py      (PU 数据画像与 SCAR/SAR 审计提示)
    08_diagnostic_report.py   (已拟合 PUSB 的结构化诊断报告)
    09_sensitivity_analysis.py (固定模型输出的假设敏感性审计)
    10_self_pu.py            (clean validation 下的 Self-PU 三阶段训练)
    11_distribution_shift.py (源/目标漂移审计与受保护的协变量加权适配)
    12_shift_decision_tools.py (配对决策、窗口监控、双域假设与主动复核)
    13_dynamic_joint_shift.py (论文式动态联合漂移 research 求解器)
```

## 4.1 Benchmark（`benchmarks/`）

```text
benchmarks/
  _common.py                   (共享原语: canonical_hash 等单源)
  __init__.py
  assigned_methods/            (前五篇 + SCAR/SAR 已执行 benchmark)
    __init__.py
    runner.py                  (统一 JSON benchmark runner)
    run.py                     (CLI)
    README.md
    preflight_paper.py         (源码/数据/历史环境与 toolbox 差距分轴审计)
    pusb_official_data.py      (IJCNN1 仓库扩展校验、官方抽样、kernel/uLSIF 与 provenance)
    pusb_table2_data.py        (Table 2 六数据集锁定 loader 与官方采样可行性审计)
    pusb_table2_benchmark.py   (Table 2 严格/兼容计划、checkpoint/resume 与 provenance)
    pusb_table2_parallel.py    (Table 2 有界并行、失败重试和自动聚合)
    pusb_table2_aggregate.py   (跨 shard 完整 key、配置和 provenance 验收)
    pusb_table2_report.py      (严格结果验收、配对置信区间与 Markdown/JSON 报告)
    configs/official/          (五份 locked_not_executed paper-like 配置)
    results/clean_room_multiseed/ (3 methods × 3 seeds 实际产物)
    results/official_preflight/ (当前节点前五篇 blocker 报告)
    results/pusb_official_data_smoke/ (PUSB 官方数据缩小网格 smoke)
    results/pusb_official_data_feasible_multiseed/ (IJCNN1 仓库扩展：完整网格 3 seeds × 3 U + uLSIF)
    results/pusb_table2_data_audit/ (Table 2 数据完整性与官方采样可行性报告)
    results/pusb_table2_strict_plan/ (45 单元/4500 trials 的 claim-safe dry-run 计划)
    results/pusb_table2_strict_full/ (4500 trials 聚合结果、统计摘要与声明安全报告)
    results/scar_sar_comparison/ (SCAR vs SAR 对比 benchmark: 3 mechanisms × 10 seeds)
  deep_pu/
    runner.py                  (InfoMax PU/WConPU/DGPU 统一 runner)
    run.py                     (CLI)
    official_data.py           (公开数据、确定性 PU split、clean validation/grid selection、resume 与 provenance)
    run_official_data.py       (official-data CLI)
    preflight_paper.py         (GPU/EDM/授权数据/实现差距审计)
    configs/
      clean_room_multiseed.json
      official_data_smoke_fashion_mnist.json
      official_data_infomax_fashion_protocol_pi05.json
      official_data_wconpu_cifar10_protocol.json
      official_sources.lock.json
      official/               (三份 paper-like 配置；InfoMax provisional matrix 已执行，其余待执行)
    results/clean_room_multiseed/ (3 methods × 3 seeds 实际产物)
    results/official_data_smoke_fashion_mnist/ (真实数据 3-seed smoke)
    results/infomax_fashion_protocol_preflight/ (InfoMax Fashion-MNIST 完整协议执行前审计)
    results/official_preflight/ (当前节点完整配置 blocker 报告)
    results/wconpu_cifar10_protocol_preflight/ (WConPU CIFAR-10 执行前审计)
  joint_shift/
    __init__.py
    runner.py                  (公开 Wisconsin 多 seed/CI 联合漂移 smoke)
    README.md                  (协议、运行命令、产物与声明边界)
  traditional_pu/
    __init__.py
    data.py                    (SCAR/SAR/PNU 合成数据、h 传递与病态性检查)
    leakage_audit.py           (泄露审计: 特征黑名单/重复样本/y_true 路径守卫/trial 列门禁/preflight 报告)
    statistics.py              (汇总原语: 均值/CI/配对差值/成功率)
    runner.py                  (统一 runner: SCAR 主网格 + SAR 诊断线 + PNU 三元网格、状态机/resume/超时)
    run.py                     (CLI: --seed-set/--results-dir/--timeout-profile)
    tuning_round.py            (调优轮工具: 候选配置生成/§4 筛选排名/§5 退化率核对)
    README.md                  (协议、命令、超时冻结流程与声明边界)
    configs/
      seven_methods_pu_baseline_v1.json
      pnu_baseline_v1.json
      pnu_baseline_v2.json      (PNU 显式锁定默认参数基线, 调优轮 companion 来源)
      seven_methods_pu_baseline_v2.json (契约 v1 口径历史快照, 勿在当前源码下重跑)
      seven_methods_pu_baseline_v3.json (契约 v2 口径正式基线: risk 跟随原生 predict 阈值)
      kldce_tuning_r1/         (KLDCE 调优第 1 轮 10 候选配置, 由 tuning_round 生成)
      kldce_tuning_r3/         (KLDCE 调优第 1 轮重跑: b₀ 类对称修复后, 同参数表)
      ldce_tuning_r1/          (LDCE 调优第 1 轮 10 候选配置, 由 tuning_round 生成)
      nnpu_tuning_r1/          (nnPU 调优第 1 轮 10 候选配置, 由 tuning_round 生成)
      upu_tuning_r1/           (uPU 调优第 1 轮 10 候选配置, 由 tuning_round 生成)
      llsvm_tuning_r1/         (LLSVM 调优第 1 轮 10 候选配置, 由 tuning_round 生成)
      elkan_noto_tuning_r1/    (Elkan-Noto 调优第 1 轮 10 候选配置, 由 tuning_round 生成)
      elkan_noto_tuning_r2/    (Elkan-Noto 调优第 1 轮重跑: 契约 v2 口径, 同参数表)
      pnu_tuning_r1/           (PNU 调优第 1 轮 10 候选配置, 由 tuning_round 生成)
    results/                   (运行产物: <run-name>/ 下四件套 + report.md)
```

启用 clean-validation 模型选择的 official-data 运行还会生成 `model_selection.csv`，逐 seed
记录每个参数候选、验证指标、分数与耗时；`trials.csv` 记录最终选中参数和 refit 结果。

## 5. 文档（`docs/`）

```text
docs/
  README.md                    # 导航首页（用户 / 开发者 / 项目过程分栏）
  adr/                         # 架构与流程决策记录(ADR 索引 + 编号决策)

  user/                        # 用户文档：旅程式（快速开始 → 概念 → 操作 → 参考）
    README.md                  # 用户旅程图
    quickstart.md              # 5 分钟快速开始
    concepts/
      pu_problem.md            # PU 问题设定、符号表与 π 的角色
      scar_sar.md              # SCAR/SAR 机制与识别边界
      method_selection.md      # 选型决策原理（推荐器 + 决策表）
    howto/
      pipeline.md              # PUPipeline 端到端工作流
      cli.md                   # 命令行接口
      data_profiling.md        # 数据画像与假设提示
      diagnostic_reports.md    # 生成诊断报告
      sensitivity_analysis.md  # 类先验/标记倾向敏感性分析
      sar_simulation.md        # SCAR/SAR 数据模拟
      self_pu.md               # Self-PU 训练
    reference/
      api.md                   # 核心 API 精确契约

  dev/                         # 开发者文档（贡献前必读）
    architecture.md            # 当前架构:模块分层、数据流、注册表
    project_structure.md       # 目录结构（本文档，权威来源）
    compatibility.md           # Python/依赖支持矩阵、CI 职责与构建策略
    architecture_audit.md       # 审计发现快照、复跑指南与治理机制（ADR-0001）
    process_checklist.md        # 进度清单与发布状态（权威来源）
    release_process.md          # 发布流程：版本策略、预检清单、上传、回滚与维护

  research/
    method_cards/
      class_prior_estimation.md
      Kernel_Mean_Class_Prior.md
      Elkan_Noto.md
      Convex_Formulation_for_PU_DATA_Learning.md
      KLDCE.md
      LDCE.md
      nnpu.md
      PNU.md
      ReCPE.md
      LLSVM.md
      Dist-PU.md
      PUSB.md
      LBE.md
      Self-PU.md
      InfoMax-PU.md
      WConPU.md
      DGPU.md
      Importance_Weighted_PU_Shift.md

```

## 6. 脚本（`scripts/`）

```text
scripts/
  check_test_quality.py      (测试质量门禁：方法数/marker/覆盖率)
  check_doc_links.py          (文档一致性检查：4 条规则)
  check_project_metadata.py   (Python/CI/extras/Hatchling 跨文件一致性)
  check_math_rendering.py     (方法卡 MathJax 渲染检查：缺上下标参数/括号配对/$ 配对)
  check_skill_sync.py         (Skill 同步检查：skills/ 定义与脚本枚举一致，第 5 道门禁)
  check_baseline_configs.py   (基线配置一致性：锁定配置 vs 源码构造器默认参数，第 6 道门禁)
  check_format.py             (格式门禁：ruff check + format --check 全目录，第 7 道门禁)
  pu_workflow/                (pu-workflow skill 环节脚本——兼容薄包装, 委托给 CLI 子命令)
    profile.py                (委托 `pu-toolbox profile`: profile.json 契约)
    recommend.py              (委托 `pu-toolbox recommend`: recommendation.json 契约)
    sensitivity.py            (委托 `pu-toolbox sensitivity`: sensitivity.json 契约)
```

## 7. CI/CD（`.github/`）

```text
.github/
  pull_request_template.md
  workflows/
    tests.yml                (PR 快层: Python matrix unit+integration + 静态门禁 + wheel 安装冒烟)
    nightly.yml              (每周一 03:23 UTC: slow + e2e 顶层全量, 3 × 3 matrix)
```
