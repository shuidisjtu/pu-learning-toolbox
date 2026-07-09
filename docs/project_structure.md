# Project Directory Structure

> **权威来源**：本文档是项目目录结构的唯一权威定义。
> - Python 包内结构（`pu_toolbox/`）以 [`architecture.md`](architecture.md) §2 为准；
> - 根目录文件、`tests/`、`benchmarks/`、`examples/`、`docs/`、`scripts/` 等以本文档为准；
> - `A.md` §5 和 `architecture.md` §2 中不再维护独立版本，改为引用本文档。

---

## 1. 项目根目录

```text
pu-learning-toolbox/
  pyproject.toml
  README.md
  LICENSE
  .gitignore
  CLAUDE.md
```

---

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

  datasets/
    __init__.py
    synthetic.py
    loaders.py
    scar_simulator.py
    sar_simulator.py
    selection_bias_simulator.py
    benchmark_catalog.py

  preprocessing/
    __init__.py
    input_adapter.py
    feature_checker.py
    sparse_utils.py
    representation.py

  prior/
    __init__.py
    base.py
    tice.py
    alphamax.py
    recpe.py
    pen_l.py
    wrappers.py

  propensity/
    __init__.py
    base.py
    scar.py
    sar.py
    elkan_noto.py
    labeling_bias.py

  losses/
    __init__.py
    base.py
    upu.py
    nnpu.py
    pnu.py
    convex_pu.py
    distribution_alignment.py

  estimators/
    __init__.py
    classic/
      __init__.py
      elkan_noto.py
      pu_bagging.py
      biased_svm.py
      weighted_lr.py
      spy.py
      rocchio.py
      one_dnf.py

    risk/
      __init__.py
      upu.py
      nnpu.py
      pnu.py
      centroid.py
      llsvm.py
      dist_pu.py

    bias_aware/
      __init__.py
      pusb.py
      lbe.py
      propensity_weighted.py

    deep/
      __init__.py
      self_pu.py
      infomax_pu.py
      weighted_contrastive_pu.py
      dgpu.py

  source_adapters/
    __init__.py
    base.py
    registry.py
    license.py
    official_source.py
    external_runner.py
    matlab_adapter.py
    torch_repo_adapter.py
    chainer_adapter.py

  metrics/
    __init__.py
    supervised.py
    pu_estimated.py
    diagnostics.py
    calibration.py
    assumption_checks.py

  model_selection/
    __init__.py
    split.py
    scorer.py
    threshold.py
    validation_curve.py
    prior_sensitivity.py
    propensity_sensitivity.py

  advisor/
    __init__.py
    data_profiler.py
    method_selector.py
    complexity_estimator.py

  registry/
    __init__.py
    registry.py
    metadata.py
    aliases.py
    source_metadata.py
    source_policy.py
```

---

## 3. 测试（`tests/`）

```text
tests/
  test_import.py
  test_labels.py
  test_validation.py
  test_registry.py
  test_estimators_classic.py
  test_prior.py
  test_losses.py
  test_split.py
  test_metrics.py
  test_advisor.py
  test_source_adapters.py
```

---

## 4. 示例（`examples/`）

```text
examples/
  minimal/
    run_elkan_noto.py
    run_pu_bagging.py
    run_prior_estimation.py

  paper_like/
    reproduce_nnpu_mnist.py
    reproduce_pnu.py
    reproduce_pusb.py
    reproduce_dist_pu.py

  sar_cases/
    run_labeling_bias_demo.py
    compare_scar_vs_sar.py
    run_selection_bias_demo.py

  source_adapter_demo/
    check_official_nnpu.py
    check_pusb_repo.py
    check_dist_pu_repo.py
```

---

## 5. Benchmark（`benchmarks/`）

```text
benchmarks/
  configs/
    smoke/
      classic_pu.yaml
      prior_estimation.yaml
      risk_losses.yaml
    synthetic/
      scar_baselines.yaml
      prior_sensitivity.yaml
      scar_vs_sar.yaml
      propensity_sensitivity.yaml
    nnpu.yaml
    pnu.yaml
    pusb.yaml
    dist_pu.yaml

  runners/
    run_smoke.py
    run_synthetic.py
    run_paper_like.py

  paper_like/
    elkan_noto/
    nnpu/
    pnu/
    pusb/
    dist_pu/
    self_pu/

  regression_tests/
    baseline_v0_1.yaml
    risk_v0_2.yaml

  reports/
    README.md
```

---

## 6. 文档（`docs/`）

```text
docs/
  README.md
  CLAUDE.md
  architecture.md
  method_selection.md
  development_roadmap.md
  resources_optimized.md
  reading_path.md
  teamwork_division.md
  project_structure.md

  user/
    getting_started.md
    method_selection.md
    assumptions.md
    evaluation.md
    api_reference.md
    contributing.md

  architecture/
    architecture_overview.md
    api_contract.md
    registry_schema.md
    source_adapter_spec.md
    benchmark_protocol.md
    paper_to_module_mapping.md

  research/
    README.md
    METHOD_CARD_TEMPLATE.md
    method_cards/
      elkan_noto.md
      class_prior_estimation.md
      recpe.md
      upu_convex_pu.md
      nnpu.md
      pnu.md
      pusb.md
      lbe.md
      dist_pu.md
      self_pu.md
      llsvm.md
      centroid_ldce.md
      infomax_pu.md
      weighted_contrastive_pu.md
      dgpu.md
    reproduction_notes/
      recpe_source.md
      nnpu_source.md
      pnu_source.md
      pusb_source.md
      lbe_source.md
      dist_pu_source.md
      self_pu_source.md
    source_notes/
      source_status_table.md
      license_review.md
      dependency_review.md

  project_management/
    A.md
    teamwork_division.md
    milestone_checklist.md
    handoff_tracker.md
    decision_log.md
    risk_register.md
    month_1_summary.md
```

---

## 7. 外部资源（`external/`）

```text
external/
  README.md
  official_sources/
    .gitkeep
  legacy_adapters/
    .gitkeep
```

---

## 8. 脚本（`scripts/`）

```text
scripts/
  update_registry_from_metadata.py
  check_method_cards.py
  run_all_smoke_tests.py
```

---

## 9. CI/CD（`.github/`）

```text
.github/
  workflows/
    tests.yml
```

---

## 10. Agent Skill

```text
agent_skill/
  skill.json
  README.md
```
