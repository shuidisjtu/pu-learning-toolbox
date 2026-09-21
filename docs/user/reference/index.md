# API 参考索引

> 公共 API 的入口索引，按模块分组快速定位。每个符号的**签名、参数、返回契约**见
> [api.md](api.md)（本页只做导航，不重复契约）；行为真相源是 docstring（ADR-0013）。

## 分类器与估计器

→ [api.md §分类器与估计器](api.md#分类器与估计器)

- **类先验估计**：`class_prior_estimation`、`recpe`、`km1`/`km2`（`KernelMeanPriorEstimator`）
- **经典**：`elkan_noto`
- **风险估计**：`upu`、`nnpu`、`pnu`、`puet`、`centroid_pu`、`kldce`、`llsvm`、`dist_pu`、`vpu`、`pulda`
- **Bias-Aware**：`pusb`、`pusb_kernel`、`lbe`
- **深度**：`gradpu`、`self_pu`、`infomax_pu`、`weighted_contrastive_pu`、`dgpu`

## 流水线与编排

- [`PUPipeline`](api.md#pupipeline) — 一键式端到端 PU 分析
- [`PUTuner`](api.md#putuner) — PU-aware 网格调参
- [`PUModelComparator`](api.md#pumodelcomparator) — 多模型比较
- [进度与取消](api.md#进度与取消) — `ProgressUpdate` / `CancellationToken`

## 分布漂移

→ [api.md §分布漂移 API](api.md#分布漂移-api)

- `analyze_pu_shift`、`ShiftAwarePUPipeline`、`PUShiftMonitor`
- `analyze_domain_assumptions`、`analyze_pu_uncertainty`

## 数据画像与推荐

- [`profile_pu_data`](api.md#profile_pu_data) — 数据画像
- [`recommend_methods` / `recommend_from_profile`](api.md#recommend_methods--recommend_from_profile) — 算法推荐

## 诊断与评估

- [`analyze_pu_sensitivity`](api.md#analyze_pu_sensitivity) — 先验/标记倾向敏感性
- [`build_diagnostic_report`](api.md#build_diagnostic_report) — 结构化诊断报告

## 实验层（survey 研究者）

→ [api.md §实验层](api.md#实验层experiment)

- `ExperimentRunner`、`DatasetBundle`/`DatasetPart`、SCAR/SAR 生成器、PA/OA 选模协议、
  训练器、`prepare_survey_dataset` 等 34+ 符号

## 数据生成

→ [api.md §数据生成](api.md#数据生成)

- `make_sar_propensity`、`make_sar_labels`、`make_sar_dataset`、`make_scar_dataset`

## 错误与异常

→ [api.md §错误与异常](api.md#错误与异常)

- `PULearningError`、`ValidationError`、`NotFittedError`、`RegistryError`、
  `PipelineError`、`RunCancelledError`
