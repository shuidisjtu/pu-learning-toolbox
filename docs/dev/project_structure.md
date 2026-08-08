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
  __init__.py

  core/
    __init__.py
    base.py
    validation.py
    labels.py
    config.py
    exceptions.py
    random.py
    tags.py

  preprocessing/
    __init__.py
    pu_labeling.py
    selection_bias.py         (SCAR/SAR propensity、标签与合成数据)
    profiling.py              (兼容 summary 与 SCAR 筛查接口)
    data_profiler.py          (结构化 PU 画像、问题与假设提示)

  prior/
    __init__.py
    recpe.py                  (native)
    pen_l1.py                (native)
    kernel_mean.py           (native: KM1/KM2 kernel-mean class-prior estimation)

  utils/
    __init__.py
    basis.py                  (shared)
    centroid.py               (shared: MoM + 协方差原语, LDCE/KLDCE 共用)

  losses/
    __init__.py
    upu.py                    (native)
    nnpu.py                  (native)
    pnu.py                   (native)
    llsvm.py                 (native)

  estimators/
    __init__.py
    classic/                 (2/2 native)
      __init__.py
      elkan_noto.py          (native)
      llsvm.py               (native)
    risk/
      __init__.py
      ldce.py                   (native)
      kldce.py                  (native: ACS + QP oracle + RBF kernel)
      dist_pu.py                (native)
      upu.py                  (native)
      nnpu.py                  (native)
      pnu.py                   (native)
    bias_aware/
      __init__.py
      pusb.py                (native: linear source-classifier baseline)
      pusb_kernel.py         (native: official-aligned RBF PUSB adapter)
      lbe.py                 (native)
    deep/
      __init__.py
      self_pu.py             (native: self-paced + meta reweight + distillation)
      infomax_pu.py          (native: PURL + nnPU pipeline)
      weighted_contrastive_pu.py (native core)
      vision.py               (WConPU CNN13/ResNet 与 tensor augmentation adapters)
      dgpu.py                (native orchestration + generator protocol)

  metrics/
    __init__.py
    classification.py          (已实现: PU risk/recall/precision + supervised wrappers)

  diagnostics/
    __init__.py
    report.py                  (数据/模型/指标诊断，JSON/Markdown 报告)
    sensitivity.py             (类先验/平均标记倾向假设敏感性)

  model_selection/
    __init__.py
    split.py                   (已实现: PUStratifiedKFold + PUStratifiedShuffleSplit)

  registry/
    __init__.py
    registry.py                (含别名解析逻辑)
    metadata.py
    builtin_methods.py         (16 论文方法元数据 + native 绑定)

  advisor/
    __init__.py
    recommender.py             (算法推荐管线: recommend_methods / recommend_from_profile)
    rules.py                   (评分规则引擎: ScoringConfig + 评分/警告函数)
    _types.py                  (数据类: MethodCandidate / RecommendationResult)

  workflows/
    __init__.py
    pipeline.py                (PUPipeline 编排: 画像→先验→训练→CV→评估)
    report.py                  (报告数据类: PriorInfo/CVMetric/PipelineReport)

  cli/                         (CLI 入口: argparse 子命令 run / list-methods / list-priors / make-demo-data, PUPipeline 的薄封装)
    __init__.py
    run.py                     (run 子命令: 双 CSV 输入、目录三件套输出、退出码 0/1/2)
    info.py                    (list-methods / list-priors 子命令, registry 实时读取)
    demo.py                    (make-demo-data 子命令: SCAR 演示数据)
```

## 3. 测试（`tests/`）

```text
tests/
  conftest.py                         # 共享 fixtures + PU 数据工厂函数
  test_import.py                      # 导入冒烟
  test_labels.py                      # 标签规范化
  test_validation.py                  # 输入校验
  test_registry.py                    # 注册机制
  test_builtin_methods.py             # 注册表元数据

  contract/                           # 契约测试 — 写一次，所有 NATIVE 分类器复用
    test_classifier_baseline.py            # fit/predict/decision_function/get_params 等

  estimators/                          # 按方法的测试（MATH/PROPERTY/API）
    risk/
      test_ldce_math.py               # LDCE 算法正确性 (MATH: MoM, 协方差, m-更新, 梯度)
      test_ldce_core.py               # LDCE fit/predict/属性/标签 (unit)
      test_ldce_api.py                # LDCE 收敛/约束/API/错误/回归 (unit)
      test_kldce_math.py              # KLDCE 公式验证 + QP oracle + bias 恢复 (MATH)
      test_kldce_property.py          # KLDCE 约束/鲁棒性 (PROPERTY)

  unit/                               # 算法特有逻辑测试
    diagnostics/
      test_report.py                  # 诊断报告指标、输入契约与序列化
      test_sensitivity.py             # 假设扫描公式、边界与导出
    estimators/
      test_elkan_noto.py              # Elkan-Noto 特有逻辑
      test_upu.py                     # uPU 特有逻辑
      test_nnpu.py                    # nnPU 特有逻辑（含训练动态/早停）
      test_bias_aware.py              # PUSB / LBE 特有逻辑
      test_pusb_kernel.py             # official-aligned PUSB 公式、CV 与确定性
      test_dist_pu.py                 # Dist-PU 特有逻辑
      test_self_pu.py                 # Self-PU pace/meta/EMA/三阶段训练
      test_deep_pu.py                 # InfoMax PU/WConPU/DGPU 接口与 registry
      test_llsvm.py                   # LLSVM 特有逻辑
    losses/
      test_nnpu_loss.py              # nnPU golden tests (MATH + PROPERTY)
      test_upu_loss.py               # uPU golden tests (MATH + PROPERTY)
      test_llsvm_loss.py             # LLSVM loss golden tests (MATH)
    metrics/
      test_classification.py          # PU 指标测试
    model_selection/
      test_split.py                   # PU 切分器测试
    prior/
      test_recpe.py                   # ReCPE 特有逻辑
      test_pen_l1.py                  # penL1 特有逻辑
      test_kernel_mean.py             # KM1/KM2 kernel-mean 类先验估计
    preprocessing/
      test_pu_labeling.py             # PU/PNU 标签生成
      test_selection_bias.py          # SCAR/SAR propensity、标记率校准与合成数据
      test_profiling.py               # 数据画像统计
      test_data_profiler.py           # 结构化报告、质量规则与审计诊断
    registry/
    advisor/
      test_recommender.py             # 算法推荐器过滤、评分与输出
    workflows/
      test_pipeline.py                # PUPipeline 全流程/先验解析/错误/可用性/确定性
    cli/
      test_cli_main.py                # CLI 入口冒烟
      test_run.py                     # run 子命令
      test_info.py                    # list-methods / list-priors 子命令
      test_demo.py                    # make-demo-data 子命令

  benchmarks/                        # Benchmark runner 测试
    test_assigned_benchmark_runner.py     # 前五篇 benchmark runner 测试
    test_pusb_official_data.py             # PUSB 官方数据构造、runner 与 provenance
    test_deep_pu_benchmark_runner.py      # 深度 PU runner 测试
    test_deep_pu_official_data.py         # 官方数据加载与 split 测试

```

测试权威级别（pytest markers）：`math`（手工计算 → 失败=代码bug）、`property`（数学不变量 → 失败=代码bug）、`contract`（API 契约）、`slow`（慢速）、`paper`（论文复现）。

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
```

## 4.1 Benchmark（`benchmarks/`）

```text
benchmarks/
  assigned_methods/            (前五篇 + SCAR/SAR 已执行 benchmark)
    preflight_paper.py         (源码/数据/历史环境与 toolbox 差距分轴审计)
    pusb_official_data.py      (IJCNN1 校验、官方抽样、kernel smoke 与 provenance)
    configs/official/          (五份 locked_not_executed paper-like 配置)
    results/official_preflight/ (当前节点前五篇 blocker 报告)
    results/pusb_official_data_smoke/ (PUSB 官方数据缩小网格 smoke)
    results/pusb_official_data_feasible_multiseed/ (完整网格 3 seeds × 3 U + uLSIF)
  deep_pu/
    runner.py                  (InfoMax PU/WConPU/DGPU 统一 runner)
    run.py                     (CLI)
    official_data.py           (公开数据、确定性 PU split、clean validation/grid selection、resume 与 provenance)
    run_official_data.py       (official-data CLI)
    preflight_paper.py         (GPU/EDM/授权数据/实现差距审计)
    configs/
      clean_room_multiseed.json
      official_data_smoke_fashion_mnist.json
      official_data_infomax_fashion_protocol.json
      official_data_wconpu_cifar10_protocol.json
      official_sources.lock.json
      official/               (三份 locked_not_executed paper-like 配置)
    results/clean_room_multiseed/ (3 methods × 3 seeds 实际产物)
    results/official_data_smoke_fashion_mnist/ (真实数据 3-seed smoke)
    results/official_preflight/ (当前节点完整配置 blocker 报告)
    results/wconpu_cifar10_protocol_preflight/ (WConPU CIFAR-10 执行前审计)
```

启用 clean-validation 模型选择的 official-data 运行还会生成 `model_selection.csv`，逐 seed
记录每个参数候选、验证指标、分数与耗时；`trials.csv` 记录最终选中参数和 refit 结果。

## 5. 文档（`docs/`）

```text
docs/
  README.md                    # 导航首页（用户 / 开发者 / 项目过程分栏）

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
    architecture.md            # 设计决策与代价、模块分层、数据流、注册表
    project_structure.md       # 目录结构（本文档，权威来源）
    roadmap.md                 # 版本路线与阶段叙事
    compatibility.md           # Python/依赖支持矩阵、CI 职责与构建策略
    resources.md               # 论文源码状态与集成策略

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
  project_management/
    decision_log.md
    process_checklist.md
    cli_design.md

```

## 6. 脚本（`scripts/`）

```text
scripts/
  check_test_quality.py      (测试质量门禁：方法数/marker/覆盖率)
  check_doc_links.py          (文档一致性检查：4 条规则)
  check_project_metadata.py   (Python/CI/extras/Hatchling 跨文件一致性)
  check_math_rendering.py     (方法卡 MathJax 渲染检查：缺上下标参数/括号配对/$ 配对)
  check_skill_sync.py         (Skill 同步检查：skills/ 定义与脚本枚举一致，第 5 道门禁)
```

## 7. CI/CD（`.github/`）

```text
.github/
  pull_request_template.md
  workflows/
    tests.yml                (Python matrix + 静态门禁 + wheel 安装冒烟)
```
