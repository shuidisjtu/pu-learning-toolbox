# Project Directory Structure

> 本文档是项目目录结构的权威定义。已实现/存在的文件如实列出，规划文件标注 `(planned)`。

## 1. 项目根目录

```text
pu-learning-toolbox/
  pyproject.toml
  README.md
  .gitignore
  CLAUDE.md
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
    synthetic.py
    loaders.py

  preprocessing/             (planned)
    __init__.py
    input_adapter.py

  prior/
    __init__.py
    base.py
    pen_l.py                 (planned)
    tice.py                  (planned)
    alphamax.py              (planned)
    wrappers.py              (planned)

  propensity/                (planned)
    __init__.py
    base.py

  losses/                    (planned)
    __init__.py
    base.py
    upu.py
    nnpu.py
    pnu.py

  estimators/
    __init__.py
    classic/                 (planned)
      elkan_noto.py
      pu_bagging.py
      biased_svm.py
      weighted_lr.py
    risk/                    (planned)
      upu.py
      nnpu.py
      pnu.py
    bias_aware/              (planned)
      pusb.py
      lbe.py
    deep/                    (planned)
      self_pu.py
      dist_pu.py

  source_adapters/
    __init__.py
    base.py
    registry.py              (planned)

  metrics/                   (planned)
    __init__.py
    supervised.py
    pu_estimated.py

  model_selection/           (planned)
    __init__.py
    split.py

  advisor/                   (planned)
    __init__.py

  registry/
    __init__.py
    registry.py
    metadata.py
    aliases.py
    source_metadata.py
    source_policy.py
```

## 3. 测试（`tests/`）

```text
tests/
  conftest.py
  test_import.py
  test_labels.py
  test_validation.py
  test_registry.py
  test_builtin_methods.py
  test_pen_l.py             (planned)
  test_losses.py             (planned)
  test_estimators_classic.py (planned)
```

## 4. 示例（`examples/`）

```text
examples/                    (planned)
  minimal/
    run_elkan_noto.py
    run_prior_estimation.py
```

## 5. Benchmark（`benchmarks/`）

```text
benchmarks/                  (planned)
  configs/
    smoke/
      classic_pu.yaml
  runners/
    run_smoke.py
  paper_like/
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
    METHOD_CARD_TEMPLATE.md
    method_cards/
      class_prior_estimation.md

  project_management/
    decision_log.md
    process_checklist.md
```

## 7. 外部资源（`external/`）

```text
external/                    (planned)
  README.md
  official_sources/
```

## 8. 脚本（`scripts/`）

```text
scripts/                     (planned)
  check_method_cards.py
```

## 9. CI/CD（`.github/`）

```text
.github/
  workflows/
    tests.yml                (planned)
```
