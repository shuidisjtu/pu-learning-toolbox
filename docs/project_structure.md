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
      pusb.py                (native)
      lbe.py                 (native)
    deep/
      __init__.py
      self_pu.py             (native: self-paced + meta reweight + distillation)
      infomax_pu.py          (native: PURL + nnPU pipeline)
      weighted_contrastive_pu.py (native core)
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
    builtin_methods.py         (15 论文方法元数据 + native 绑定)
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
      test_kldce_math.py              # KLDCE 公式验证 (MATH: Q/d/Aeq/bias/delta)
      test_kldce_oracle.py            # KLDCE QP oracle + bias 恢复 (MATH)
      test_kldce_property.py          # KLDCE 约束/鲁棒性/可复现性 (PROPERTY)

  unit/                               # 算法特有逻辑测试
    diagnostics/
      test_report.py                  # 诊断报告指标、输入契约与序列化
      test_sensitivity.py             # 假设扫描公式、边界与导出
    estimators/
      test_elkan_noto.py              # Elkan-Noto 特有逻辑
      test_upu.py                     # uPU 特有逻辑
      test_nnpu.py                    # nnPU 特有逻辑（含训练动态/早停）
      test_bias_aware.py              # PUSB / LBE 特有逻辑
      test_dist_pu.py                 # Dist-PU 特有逻辑
      test_self_pu.py                 # Self-PU pace/meta/EMA/三阶段训练
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
    preprocessing/
      test_pu_labeling.py             # PU/PNU 标签生成
      test_selection_bias.py          # SCAR/SAR propensity、标记率校准与合成数据
      test_profiling.py               # 数据画像统计
      test_data_profiler.py           # 结构化报告、质量规则与审计诊断

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
  deep_pu/
    runner.py                  (InfoMax PU/WConPU/DGPU 统一 runner)
    run.py                     (CLI)
    official_data.py           (公开数据、确定性 PU split、resume 与 provenance)
    run_official_data.py       (official-data CLI)
    preflight_paper.py         (GPU/EDM/授权数据/实现差距审计)
    configs/
      clean_room_multiseed.json
      official_data_smoke_fashion_mnist.json
      official_sources.lock.json
      official/               (三份 locked_not_executed paper-like 配置)
    results/clean_room_multiseed/ (3 methods × 3 seeds 实际产物)
    results/official_data_smoke_fashion_mnist/ (真实数据 3-seed smoke)
    results/official_preflight/ (当前节点完整配置 blocker 报告)
```

## 5. 文档（`docs/`）

```text
docs/
  README.md
  architecture.md
  project_structure.md
  method_selection.md
  development_roadmap.md
  development_compatibility.md
  resources_optimized.md

  user/
    sar_simulation.md             # SCAR/SAR 数据模拟与 benchmark 指南
    data_profiling.md             # 数据画像、可识别性边界与行动建议
    diagnostic_reports.md         # 模型输出、证据级别与可审计报告
    sensitivity_analysis.md       # 类先验/标记倾向敏感性与解释边界
    self_pu.md                    # Self-PU 数据协议、训练状态与消融边界

  research/
    method_cards/
      class_prior_estimation.md
      Elkan_Noto.md
      Convex_Formulation_for_PU_DATA_Learning.md
      KLDCE.md
      LDCE.md
      nnpu.md
      PNU.md
      ReCPE.md
      LLSVM.md
  project_management/
    decision_log.md
    process_checklist.md
    division.txt

```

## 6. 脚本（`scripts/`）

```text
scripts/
  check_test_quality.py      (测试质量门禁：方法数/marker/覆盖率)
  check_doc_links.py          (文档一致性检查：4 条规则)
  check_project_metadata.py   (Python/CI/extras/Hatchling 跨文件一致性)
```

## 7. CI/CD（`.github/`）

```text
.github/
  pull_request_template.md
  workflows/
    tests.yml                (Python matrix + 静态门禁 + wheel 安装冒烟)
```
