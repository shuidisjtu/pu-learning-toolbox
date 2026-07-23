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

  preprocessing/
    __init__.py
    pu_labeling.py
    profiling.py

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

  estimators/
    __init__.py
    classic/                 (1/1 native)
      elkan_noto.py          (native)
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
    deep/                    (planned)
      __init__.py
      self_pu.py             (planned)

  source_adapters/
    __init__.py
    base.py
    registry.py              (planned)

  metrics/
    __init__.py
    classification.py          (已实现: PU risk/recall/precision + supervised wrappers)

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
    estimators/
      test_elkan_noto.py              # Elkan-Noto 特有逻辑
      test_upu.py                     # uPU 特有逻辑
      test_nnpu.py                    # nnPU 特有逻辑（含训练动态/早停）
      test_bias_aware.py              # PUSB / LBE 特有逻辑
      test_dist_pu.py                 # Dist-PU 特有逻辑
    losses/
      test_nnpu_loss.py              # nnPU golden tests (MATH + PROPERTY)
      test_upu_loss.py               # uPU golden tests (MATH + PROPERTY)
    metrics/
      test_classification.py          # PU 指标测试
    model_selection/
      test_split.py                   # PU 切分器测试
    prior/
      test_recpe.py                   # ReCPE 特有逻辑
      test_pen_l1.py                  # penL1 特有逻辑
    preprocessing/
      test_pu_labeling.py             # PU/PNU 标签生成
      test_profiling.py               # 数据画像统计

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

## 5. 文档（`docs/`）

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
      KLDCE.md
      LDCE.md
      nnpu.md
      PNU.md
      ReCPE.md
  project_management/
    decision_log.md
    process_checklist.md
    division.txt

  user/                       (planned)
```

## 6. 脚本（`scripts/`）

```text
scripts/
  check_test_quality.py      (测试质量门禁：方法数/marker/覆盖率)
  check_doc_links.py          (文档一致性检查：4 条规则)
```

## 7. CI/CD（`.github/`）

```text
.github/
  workflows/
    tests.yml                (push/PR: 测试 + lint + 质量门禁 + 文档检查)
```
