# Project Directory Structure

> 本文档是项目目录结构的权威定义。已实现/存在的文件如实列出，规划文件标注 `(planned)`。

## 1. 项目根目录

```text
pu-learning-toolbox/
  pyproject.toml
  README.md
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

  datasets/                  (planned)
    __init__.py
    synthetic.py            (planned)
    loaders.py              (planned)

  preprocessing/
    __init__.py
    pu_labeling.py
    profiling.py

  prior/
    __init__.py
    recpe.py                  (native)
    base.py                  (planned)
    pen_l1.py                (planned)
    tice.py                  (planned)
    alphamax.py              (planned)
    wrappers.py              (planned)

  propensity/                (planned)
    __init__.py
    base.py                  (planned)

  utils/
    __init__.py
    basis.py                  (shared)

  losses/
    __init__.py
    upu.py                    (native)
    nnpu.py                  (native)
    pnu.py                   (native)

  estimators/
    __init__.py
    classic/                 (1/4 native)
      elkan_noto.py          (native)
      pu_bagging.py           (planned)
      biased_svm.py           (planned)
      weighted_lr.py          (planned)
    risk/
      __init__.py
      upu.py                  (native)
      nnpu.py                  (native)
      pnu.py                   (native)
    bias_aware/              (planned)
      __init__.py
      pusb.py                (planned)
      lbe.py                 (planned)
    deep/                    (planned)
      __init__.py
      self_pu.py             (planned)
      dist_pu.py             (planned)

  source_adapters/
    __init__.py
    base.py
    registry.py              (planned)

  metrics/
    __init__.py
    classification.py          (已实现: PU risk + supervised wrappers)

  model_selection/
    __init__.py
    split.py                   (已实现: PUStratifiedKFold + PUStratifiedShuffleSplit)

  advisor/                   (planned)
    __init__.py

  registry/
    __init__.py
    registry.py
    metadata.py
    aliases.py
    builtin_methods.py         (15 论文方法元数据 + native 绑定)
    source_metadata.py
    source_policy.py
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
    test_classifier_api.py            # fit/predict/decision_function/get_params 等

  unit/                               # 算法特有逻辑测试
    estimators/
      test_elkan_noto.py              # Elkan-Noto 特有逻辑
      test_upu.py                     # uPU 特有逻辑
      test_nnpu.py                    # nnPU 特有逻辑（含训练动态/早停）
    losses/
      test_nnpu_loss.py              # nnPU golden tests (MATH + PROPERTY)
      test_upu_loss.py               # uPU golden tests (MATH + PROPERTY)
    metrics/
      test_classification.py          # PU 指标测试
    model_selection/
      test_split.py                   # PU 切分器测试
    prior/
      test_recpe.py                   # ReCPE 特有逻辑
    preprocessing/
      test_pu_labeling.py             # PU/PNU 标签生成
      test_profiling.py               # 数据画像统计

  integration/                        # 跨模块集成（待扩展）
  regression/                         # 慢速/论文复现（待扩展）
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
```

## 5. Benchmark（`benchmarks/`）

```text
benchmarks/                  (骨架就位，目录已创建)
  configs/
    smoke/
    synthetic/
  runners/
  paper_like/
  regression_tests/
  reports/
```

## 6. 文档（`docs/`）

```text
docs/
  README.md
  architecture.md
  project_structure.md
  method_selection.md
  development_roadmap.md
  resources_optimized.md

  research/
    method_cards/
      class_prior_estimation.md
      Elkan_Noto.md
      Convex_Formulation_for_PU_DATA_Learning.md
      nnpu.md
      PNU.md
      ReCPE.md

  project_management/
    decision_log.md
    process_checklist.md
    division.txt
```

## 7. 脚本（`scripts/`）

```text
scripts/
  check_test_quality.py      (测试质量门禁：方法数/marker/覆盖率)
  check_doc_links.py          (文档一致性检查：6 条规则)
```

## 8. CI/CD（`.github/`）

```text
.github/
  workflows/
    tests.yml                (planned)
```
