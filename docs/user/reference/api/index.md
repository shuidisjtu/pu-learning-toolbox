# API reference

> **Scope:** this directory is the authoritative public API contract reference. Each entry contains its
> signature, parameter table, return structure, and minimal example. For concepts and complete
> walkthroughs, follow links to [concepts](../../concepts/), task-oriented [guides](../../howto/),
> and [examples/minimal](../../../../examples/minimal/). Looking up a parameter contract does not
> require following those links.
>
> **Version:** **v2.0.0 (unreleased)**, matching `pyproject.toml`; updated with package releases. The
> [release process](../../../dev/release_process.md) is optional Chinese maintenance background.
> **Coverage:** `scripts/check_api_docs.py` checks that this directory covers every public export
> and registered algorithm name. Document new public APIs in the corresponding contract page.

## Reading guide

| Task | Location |
|---|---|
| Run a first experiment in five minutes | [Quickstart](../../quickstart.md) |
| Understand PU settings, SCAR/SAR, and method selection | [Concepts](../../concepts/) |
| Follow a complete task workflow | [Guides](../../howto/), starting from the [user index](../../README.md) |
| **Look up an API signature, parameters, or return contract** | [Classifiers](classifiers.md), [tools](tools.md), and [errors](errors.md) |
| Run an example script | [examples/minimal/](../../../../examples/minimal/) |
| Read algorithm derivations, formulas, and reproduction status | [Method cards](../../../research/method_cards/) (optional Chinese research background) |

> Parameter contracts live here; docstrings are the source of truth for behavioral details
> under ADR-0013. Walkthroughs live in guides/examples, and algorithm research lives in method
> cards. Each information category has one authoritative location, connected by links.

## Classifiers and prior estimators

See [classifiers and estimators](classifiers.md#classifiers-and-estimators) for method-specific
contracts, including standalone research interfaces. Find registered names and aliases in the
[registry index](#registry-index) or run `pu-toolbox list-methods`.

- **Class-prior estimation:** `class_prior_estimation`, `recpe`, and the `km1` / `km2`
  variants of `KernelMeanPriorEstimator`.
- **Classifiers:** classic, risk-estimation, bias-aware, and deep methods, with explicit
  input, prior, support-data, and architecture requirements.
- **Standalone mixture estimation:** `BBEMixtureEstimator` and `best_bin_estimate`.

## Encoder construction
- [`build_encoder`](tools.md#build_encoder): shared image encoder construction.

- [`build_diagnostic_report`](tools.md#build_diagnostic_report): independent structured diagnostics.

## Distribution shift

See the [distribution-shift APIs](tools.md#distribution-shift-api):

- `analyze_pu_shift` and `PUShiftMonitor`.
- `analyze_domain_assumptions` and `analyze_pu_uncertainty`.
- Research-only joint-shift classifiers and their interpretation limits.

## Data profiling and method discovery

- [`profile_pu_data`](tools.md#profile_pu_data): data-quality and assumption evidence.
- [Explicit method discovery](tools.md#explicit-method-discovery-and-selection): registry capabilities and candidate selection.

## Diagnostics and evaluation

- [`analyze_pu_sensitivity`](tools.md#analyze_pu_sensitivity): class-prior and labeling-propensity
  sensitivity.

## Research experiments

See the [experiment layer](tools.md#experiment-layer-experiment) for `ExperimentRunner`,
`DatasetBundle` / `DatasetPart`, SCAR/SAR generators, PA/OA selection, trainers, support-data
adapters, checkpoints, and result/reporting interfaces. Start with the
[custom experiment guide](../../howto/custom_experiments.md) for a complete workflow.

## Data generation

See [data generation](tools.md#data-generation):

- `make_sar_propensity`, `make_sar_labels`, `make_sar_dataset`, and `make_scar_dataset`.
- `make_fixed_count_labels`, `make_pu_labels`, and `make_gaussian_case_control_data`
  for same-pool labeling and independent TS sampling.

## Errors and exceptions

See [errors and exceptions](errors.md#errors-and-exceptions) for `PULearningError`,
`ValidationError`, `NotFittedError`, `RegistryError`, `PipelineError`, and
`RunCancelledError`.

## Registry index

This table provides quick navigation and aliases. Complete signatures, parameter tables, and
return behavior are documented in [Classifiers and estimators](classifiers.md). Method cards
provide optional Chinese research background.

| Registered name (aliases) | Class | Family | Main constructor parameters | Method card (Chinese) |
|---|---|---|---|---|
| [`class_prior_estimation`](classifiers.md#classpriorestimator-class_prior_estimation-aliases-cpe--pen_l1) (`cpe`, `pen_l1`) | [`ClassPriorEstimator`](classifiers.md#classpriorestimator-class_prior_estimation-aliases-cpe--pen_l1) | class-prior | `sigma` / `reg_lambda` / `theta_grid` / `n_centers` | [class_prior_estimation](../../../research/method_cards/class_prior_estimation.md) |
| [`recpe`](classifiers.md#recpeestimator-recpe-alias-re_cpe) (`re_cpe`) | [`ReCPEEstimator`](classifiers.md#recpeestimator-recpe-alias-re_cpe) | class-prior | `copy_fraction` / `base_estimator` | [ReCPE](../../../research/method_cards/ReCPE.md) |
| [`elkan_noto`](classifiers.md#elkannotoclassifier-elkan_noto-alias-en) (`en`) | [`ElkanNotoClassifier`](classifiers.md#elkannotoclassifier-elkan_noto-alias-en) | classic | `base_estimator` / `calibration_method` / `internal_cv_folds` / `eps` | [Elkan_Noto](../../../research/method_cards/Elkan_Noto.md) |
| [`upu`](classifiers.md#upuclassifier-upu-alias-convex_pu) (`convex_pu`) | [`UPUClassifier`](classifiers.md#upuclassifier-upu-alias-convex_pu) | risk | `class_prior` / `loss` / `reg_lambda` | [Convex uPU](../../../research/method_cards/Convex_Formulation_for_PU_DATA_Learning.md) |
| [`nnpu`](classifiers.md#nonnegativepuclassifier-nnpu-alias-nn-pu) (`nn-pu`) | [`NonNegativePUClassifier`](classifiers.md#nonnegativepuclassifier-nnpu-alias-nn-pu) | risk | `model` / `encoder` / `class_prior` / `loss` / `optimizer` | [nnPU](../../../research/method_cards/nnpu.md) |
| [`pnu`](classifiers.md#pnuclassifier-pnu) | [`PNUClassifier`](classifiers.md#pnuclassifier-pnu) | risk | `class_prior` / `eta` / `reg_lambda` | [PNU](../../../research/method_cards/PNU.md) |
| [`puet`](classifiers.md#puextratreesclassifier-puet-alias-pu_extra_trees) (`pu_extra_trees`) | [`PUExtraTreesClassifier`](classifiers.md#puextratreesclassifier-puet-alias-pu_extra_trees) | risk | `class_prior` / `n_estimators` / `max_features` / `max_candidates` | [PUET](../../../research/method_cards/PUET.md) |
| [`centroid_pu`](classifiers.md#ldceclassifier-centroid_pu-alias-ldce) (`ldce`) | [`LDCEClassifier`](classifiers.md#ldceclassifier-centroid_pu-alias-ldce) | risk | `flip_probability` / `reg_strength` / `centroid_radius` | [LDCE](../../../research/method_cards/LDCE.md) |
| [`kldce`](classifiers.md#kldceclassifier-kldce-alias-kernelized_ldce) (`kernelized_ldce`) | [`KLDCEClassifier`](classifiers.md#kldceclassifier-kldce-alias-kernelized_ldce) | risk | `flip_probability` / `sigma` / `reg_strength` | [KLDCE](../../../research/method_cards/KLDCE.md) |
| [`llsvm`](classifiers.md#llsvmclassifier-llsvm) | [`LLSVMClassifier`](classifiers.md#llsvmclassifier-llsvm) | risk | `alpha` / `beta` / `gamma` / `reg_lambda` / `max_epochs` | [LLSVM](../../../research/method_cards/LLSVM.md) |
| [`dist_pu`](classifiers.md#distpuclassifier-dist_pu-alias-distpu) (`distpu`) | [`DistPUClassifier`](classifiers.md#distpuclassifier-dist_pu-alias-distpu) | risk | `class_prior` / `hidden_dim` / `epochs` / `learning_rate` / `distribution_loss` | [Dist-PU](../../../research/method_cards/Dist-PU.md) |
| [`vpu`](classifiers.md#vpuclassifier-vpu-alias-variational_pu) (`variational_pu`) | [`VPUClassifier`](classifiers.md#vpuclassifier-vpu-alias-variational_pu) | risk | `hidden_dim` / `max_epochs` / `regularization_weight` / `mixup_alpha` | [VPU](../../../research/method_cards/VPU.md) |
| [`pulda`](classifiers.md#puldaclassifier-pulda-alias-label_distribution_alignment) (`label_distribution_alignment`) | [`PULDAClassifier`](classifiers.md#puldaclassifier-pulda-alias-label_distribution_alignment) | risk | `class_prior` / `warmup_epochs` / `pu_epochs` / `margin` / `mixup_weight` | [PULDA](../../../research/method_cards/PULDA.md) |
| [`cvir`](classifiers.md#cvirclassifier-cvir-alias-conditional_value_ignoring_risk) (`conditional_value_ignoring_risk`) | [`CVIRClassifier`](classifiers.md#cvirclassifier-cvir-alias-conditional_value_ignoring_risk) | risk | `unlabeled_positive_prior` (U proportion) / `hidden_dim` / `max_epochs` / `warm_start_epochs` | [CVIR](../../../research/method_cards/CVIR.md) |
| [`pusb`](classifiers.md#pusbclassifier-registered-as-pusb-alias-biased_pu) (`biased_pu`) | [`PUSBClassifier`](classifiers.md#pusbclassifier-registered-as-pusb-alias-biased_pu) | bias-aware | `threshold` / `C` / `max_iter` | [PUSB](../../../research/method_cards/PUSB.md) |
| [`pusb_kernel`](classifiers.md#pusbkernelclassifier-registered-as-pusb_kernel-alias-kernelized_pusb) (`kernelized_pusb`) | [`PUSBKernelClassifier`](classifiers.md#pusbkernelclassifier-registered-as-pusb_kernel-alias-kernelized_pusb) | bias-aware | `n_basis` / `cv` / `sigma_grid` / `reg_grid` | [PUSB §7.3](../../../research/method_cards/PUSB.md) |
| [`lbe`](classifiers.md#lbeclassifier-registered-as-lbe) | [`LBEClassifier`](classifiers.md#lbeclassifier-registered-as-lbe) | bias-aware | `max_iter` / `n_em_iter` / `C` | [LBE](../../../research/method_cards/LBE.md) |
| [`lbe_neural`](classifiers.md#neurallbeclassifier-registered-as-lbe_neural-alias-neural_lbe) (`neural_lbe`) | [`NeuralLBEClassifier`](classifiers.md#neurallbeclassifier-registered-as-lbe_neural-alias-neural_lbe) | bias-aware / torch | `pretrain_steps` / `n_em_iter` / `m_steps` / `encoder` | [LBE §10](../../../research/method_cards/LBE.md#10-2026-10-14-独立神经网络变体) |
| [`gradpu`](classifiers.md#gradpuclassifier-registered-as-gradpu-alias-grad_pu) (`grad_pu`) | [`GradPUClassifier`](classifiers.md#gradpuclassifier-registered-as-gradpu-alias-grad_pu) | deep | `alpha` / `beta_max` / `model` / `max_epochs` | [GradPU](../../../research/method_cards/GradPU.md) |
| [`robust_pu`](classifiers.md#robustpuclassifier-registered-as-robust_pu-alias-robust-pu) (`robust-pu`) | [`RobustPUClassifier`](classifiers.md#robustpuclassifier-registered-as-robust_pu-alias-robust-pu) | deep | `class_prior` / `pretrain_epochs` / `episodes` / `spl_type` | [Robust-PU](../../../research/method_cards/Robust-PU.md) |
| [`split_pu`](classifiers.md#splitpuclassifier-registered-as-split_pu-alias-split-pu) (`split-pu`) | [`SplitPUClassifier`](classifiers.md#splitpuclassifier-registered-as-split_pu-alias-split-pu) | deep | `class_prior` / `teacher_epochs` / `split_epochs` / `student_epochs` | [Split-PU](../../../research/method_cards/Split-PU.md) |
| [`tedn`](classifiers.md#tednclassifier-standalone-experimental-registration-native-ts) (`transform_estimate_discard`) | [`TEDnClassifier`](classifiers.md#tednclassifier-standalone-experimental-registration-native-ts) | risk / torch | `input_sampling="ts"` (required) / `holdout_fraction` / `warm_start_epochs` / `max_epochs` | [TEDⁿ](../../../research/method_cards/TEDn.md) |
| [`lagam`](classifiers.md#lagamclassifier-registered-as-lagam-alias-la_gam) (`la_gam`) | [`LaGAMClassifier`](classifiers.md#lagamclassifier-registered-as-lagam-alias-la_gam) | deep | `support_data` (required by fit) / `warmup_epochs` / `num_clusters` | [LaGAM](../../../research/method_cards/LaGAM.md) |
| [`pan`](classifiers.md#panclassifier-pan) (`predictive_adversarial_pu`) | [`PANClassifier`](classifiers.md#panclassifier-pan) | deep | `encoder` / `model` / `discriminator` / `adversarial_weight` / `max_epochs` | [PAN](../../../research/method_cards/PAN.md) |
| [`genpu`](classifiers.md#genpuclassifier-registered-as-genpu-alias-gen_pu) (`gen_pu`) | [`GenPUClassifier`](classifiers.md#genpuclassifier-registered-as-genpu-alias-gen_pu) | deep | `class_prior` / `latent_dim` / `max_epochs` / `classifier_epochs` | [GEN-PU](../../../research/method_cards/GEN-PU.md) |
| [`holistic_pu`](classifiers.md#holisticpuclassifier-holistic_pu) (`holisticpu`) | [`HolisticPUClassifier`](classifiers.md#holisticpuclassifier-holistic_pu) | deep | `encoder` / `warmup_epochs` / `trend_scale` / `max_epochs` | [Holistic-PU](../../../research/method_cards/Holistic-PU.md) |
| [`rp`](classifiers.md#rankpruningclassifier-rp) (`rank_pruning`) | [`RankPruningClassifier`](classifiers.md#rankpruningclassifier-rp) | classic | `base_estimator` / `n_cv_folds` / `frac_pos2neg` / `min_retained` | [RP](../../../research/method_cards/RP.md) |
| [`pulns`](classifiers.md#pulnsclassifier-pulns) (`negative_selector_pu`) | [`PULNSClassifier`](classifiers.md#pulnsclassifier-pulns) | deep | `support_data` (required by fit) / `episodes` / `discount` / `terminal_weight` | [PULNS](../../../research/method_cards/PULNS.md) |
| [`self_pu`](classifiers.md#selfpuclassifier-registered-as-self_pu) | [`SelfPUClassifier`](classifiers.md#selfpuclassifier-registered-as-self_pu) | deep | `class_prior` / `backbone` / `warmup_epochs` / `self_paced_start` | [Self-PU](../../../research/method_cards/Self-PU.md) |
| [`infomax_pu`](classifiers.md#infomaxpuclassifier-registered-as-infomax_pu) | [`InfoMaxPUClassifier`](classifiers.md#infomaxpuclassifier-registered-as-infomax_pu) | deep | `class_prior` / `representation_*` / `classifier_*` (see the linked contract) | [InfoMax-PU](../../../research/method_cards/InfoMax-PU.md) |
| [`weighted_contrastive_pu`](classifiers.md#weightedcontrastivepuclassifier-registered-as-weighted_contrastive_pu-alias-wconpu) (`wconpu`) | [`WeightedContrastivePUClassifier`](classifiers.md#weightedcontrastivepuclassifier-registered-as-weighted_contrastive_pu-alias-wconpu) | deep | `class_prior` / `encoder` / `hidden_dim` / `embedding_dim` | [WConPU](../../../research/method_cards/WConPU.md) |
| [`p3mix`](classifiers.md#p3mixclassifier-experimental-registration-native-os) | [`P3MixClassifier`](classifiers.md#p3mixclassifier-experimental-registration-native-os) | deep | `variant` / `beta` / `gamma` / `pool_size` / `encoder` | [P3MIX](../../../research/method_cards/P3MIX.md) |
| [`dgpu`](classifiers.md#dgpuclassifier-registered-as-dgpu) | [`DGPUClassifier`](classifiers.md#dgpuclassifier-registered-as-dgpu) | deep | `class_prior` / `generator` / `model` / `hidden_dim` | [DGPU](../../../research/method_cards/DGPU.md) |
| — (`km1` / `km2` variant) | [`KernelMeanPriorEstimator`](classifiers.md#kernelmeanpriorestimator-km1--km2-variants) | class-prior | `variant="km1"/"km2"` and other kernel-mean settings | [Kernel_Mean](../../../research/method_cards/Kernel_Mean_Class_Prior.md) |

> Registered classifier names are discoverable through the registry and `list-methods`.
> `get_algorithm(name)` and `get_metadata(name)` discover built-ins on first use without manual
> initialization. Importing the registry alone does not load the built-in catalog.
> `clear_registry()` resets the registry; the next public use reloads the built-ins.
> Custom `register_method(metadata, estimator_cls)` requires a concrete implementation class;
> unimplemented placeholders cannot be registered. Runtime registration does not run tests.
> The registry stores lowercase IDs and resolves names and aliases case-insensitively, so `nnPU`
> resolves to `nnpu` without a separate alias. Custom registrations with duplicate or conflicting
> names or aliases raise `RegistryError` without changing the registry. The family column uses
> friendly abbreviations. CLI and JSON use
> enum values `class_prior_estimation` / `classic_calibration` / `risk_estimation` / `bias_aware` /
> `deep_pu`. Use `classifier_params` for required constructor arguments other than `class_prior`;
> a configured instance can supply Python-object protocol dependencies. The old prior alias `pe`
> still resolves but emits `FutureWarning`; migrate to `class_prior_estimation` or `cpe`.
