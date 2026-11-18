# Project Directory Structure

> This document is the authoritative directory inventory. Existing files are listed as such;
> proposed files carry `(planned)`. `generate_structure.py` generates the tree blocks for
> `pu_toolbox/`, `tests/`, `scripts/`, and `docs/`, checking file names in both directions.
> Sections 1, 4, 4.1, and 7 are maintained manually. Section 5 explains suffix coverage,
> grouped subtrees, and the limits of the generator's checks.

## 1. Repository root

```text
pu-learning-toolbox/
  pyproject.toml
  README.md
  CONTRIBUTING.md
  LICENSE
  .gitignore
  uv.lock
```

### 1.1 Separate source code, experiment assets, and deliverables

| Category | Location | Management |
|---|---|---|
| Package source | `pu_toolbox/` | Tracked in Git; workspace cleanup does not move public import paths. |
| Automated tests / examples | `tests/`, `examples/` | Tracked in Git; keep real datasets and training outputs elsewhere. |
| Experiment / maintenance entry points | `scripts/`, `benchmarks/` | Tracked in Git; retain public execution and reporting entry points, and retire scripts specific to obsolete experiment processes. |
| User / developer / research documentation | `docs/user/`, `docs/dev/`, `docs/research/` | Tracked in Git; retain methods, data, and usage documentation. Consult Git history for experiment process records. |
| Original user uploads | `data/incoming/user-uploads/` (created locally) | Categorize under `papers/`, `results/`, `datasets/`, `protocols/`, `tasks/`, or `diagnostics/`; retain original names and SHA256 hashes. |
| Incoming external deliveries | `data/incoming/` | Keep separate from accepted data until downloads, verification, and reconciliation finish. A `.part` file alone does not justify deleting an older file. |
| Raw data / standard splits | `data/raw/`, `data/splits/` | Ignored locally; retain historical receipt paths and do not rewrite split manifests merely to standardize names. |
| Run / diagnostic evidence | `results/` | Ignored locally; retain manifests, logs, selection artifacts, queue state, and historical protocol identities. |
| Outgoing tables / documents | `data/outgoing/` | Organize by delivery date; do not overwrite delivered snapshots or track them in Git by default. |
| Old builds / development drafts | `data/archive/` (created locally) | Archive by category. Archiving neither constitutes formal acceptance nor grants permission to delete. |
| One-off local demos | `data/local-demos/` (created locally) | Keep separate from published examples; retain user files even when unrelated to PU experiments. |
| Temporary plans / run helpers | `.tmp/`, `tmp/` | Ignored locally; may contain background controllers, handoff lists, or rerun evidence. Do not empty entire directories. |
| Rebuildable development caches | `.pytest_cache/`, `.ruff_cache/`, `__pycache__/` | Clean only after checking for active writers, tracked files, and user files. |

These local directories need not exist in a clean clone. A server may generate a local
`LOCAL_WORKSPACE.html` index of categorized uploads, recent deliveries, and cleanup receipts.
Do not commit machine-specific absolute paths or private download URLs.
Historical absolute paths may be embedded in scripts, manifests, and delivery indexes.
When moving originals, verify SHA256 before and after the move and retain relative symlinks
at old locations when needed. Register those compatibility paths only in the local Git exclude;
do not hide tracked test fixtures or frozen evidence behind blanket `*.xlsx` or `*.json` rules.

Base cleanup on evidence, rather than file dates. Rebuildable caches may be removed;
before deleting undelivered render intermediates superseded by a complete newer version,
verify matching provenance and the absence of unique results, and record how to rebuild them.
Retain delivered results, uploads, raw data, splits, frozen protocols/comparison matrices,
and checkpoints by default. Use the experiment's checkpoint reclamation mechanism;
bulk directory deletion cannot replace an audit of selection artifacts and recovery state.
Virtual environments, external Git worktrees, and background services are outside routine cleanup.
Preview exact paths before execution; retain move/delete inventories, hashes, and recovery limits afterward.

## 2. Python package (`pu_toolbox/`)

```text
pu_toolbox/
  core/
    __init__.py                           (public re-exports)
    base.py                               (shared: BasePUClassifier/BasePriorEstimator/BasePULoss estimator and loss base classes)
    validation.py                         (shared: validate_pu_X_y/validate_pnu_X_y input and scalar validation; common fit entry point)
    labels.py                             (shared: normalize_pu_labels/normalize_pnu_labels PU/PNU label normalization)
    config.py                             (shared: label constants (+1/0/-1), probability clipping, and PU ratio thresholds)
    exceptions.py                         (shared: PULearningError family with Validation/NotFitted/Registry subclasses)
    random.py                             (shared: sklearn-compatible check_random_state seed normalization)
    tags.py                               (shared: typed Scenario/Assumption/SourceStatus/Maturity enums)
    device.py                             (shared: resolve_device/resolve_device_name CUDA autodetection)
    training_views.py                     (shared: neutral TrainingView/build_training_view construction of P/raw-U/loss-U roles and source indices; independent of ledgers, method names, and manifests)
    fit_state.py                          # opt-in state invalidation after failed fits
  preprocessing/
    __init__.py                           (public re-exports)
    _sampling.py                          # shared fixed positive quotas and uniform labeling without replacement
    pu_labeling.py                        (make_scar_labels/make_pu_labels PU/PNU labels and synthetic Gaussian data)
    selection_bias.py                     (make_sar_propensity/make_sar_labels SAR propensities and synthetic data)
    profiling.py                          (pu_data_summary lightweight summaries and scar_diagnostic SCAR diagnostics)
    data_profiler.py                      (profile_pu_data structured summaries, issues, and assumption warnings)
  prior/
    __init__.py                           (public re-exports)
    recpe.py                              (native: ReCPEEstimator regrouping prior estimation; default KM2 base estimator)
    pen_l1.py                             (native: ClassPriorEstimator penalized L1 closed-form prior with theta grid search)
    kernel_mean.py                        (native: KernelMeanPriorEstimator KM1/KM2 kernel-mean mixture proportion estimation)
    bbe.py                                (standalone: BBE α_U statistic and isolated held-out PvU scoring; not registered as a population prior)
    semantics.py                          (native-U and population-prior mass conversion for OS and independent TS)
  utils/
    __init__.py                           (public re-exports)
    basis.py                              (shared: build_linear_basis/build_rbf_basis basis construction and center subsampling)
    centroid.py                           (shared: MoM centroids and empirical covariance for LDCE/KLDCE)
    activations.py                        (shared: sigmoid_stable numerically stable sigmoid for uPU/nnPU losses)
    serialization.py                      (shared: json_safe/canonical_hash report serialization and Markdown escaping)
  losses/
    __init__.py                           (public re-exports)
    upu.py                                (native: UPULoss unbiased PU risk with logistic/squared margin losses)
    nnpu.py                               (native: NonNegativePULoss nonnegative risk evaluation and nnPU training step)
    pnu.py                                (native: PN/NU/PU risk components reused by risk/pnu.py)
    llsvm.py                              (native: llsvm_objective/calibration_loss LLSVM loss components from official formulas)
  estimators/
    classic/                              (native classical estimators)
      __init__.py                         (public re-exports)
      elkan_noto.py                       (native: Elkan–Noto SCAR probability correction/weighted refit; stratified K-fold OOF estimation of c)
      llsvm.py                            (native: linear large-margin LLSVM calibration; nonconvex SGD objective and early stopping aligned with official formulas)
      rank_pruning.py                     # PU-only Rank Pruning with OOF pruning and weighting
    risk/
      __init__.py                         (public re-exports)
      _class_prior.py                     (shared: class-prior derivation and 1−2ph stability checks for KLDCE/LDCE)
      ldce.py                             (native: linear LDCE for censoring PU, MoM centroid, ellipsoid constraint, and alternating m/subgradient-w updates)
      kldce.py                            (native: ACS + native SMO + RBF kernel)
      kldce_smo.py                        (native paired SMO solver: analytical updates and KKT selection)
      dist_pu.py                          (native: default clean-room expectation objective and optional author polar-histogram loss; single-stage MLP, optional torch)
      vpu.py                              (native: VPU variational risk/MixUp, trainable CNN, and CPU-resident data batching)
      pulda.py                            (native: PULDA distribution alignment, bidirectional margin, EMA pseudo-label MixUp, and two-stage 2-D MLP adaptation)
      upu.py                              (native: du Plessis'15 convex uPU estimator with double-hinge/logistic/squared losses)
      nnpu.py                             (native: nnPU nonnegative risk with correction-branch -γr gradient; mini-batch SGD + encoder)
      pnu.py                              (native: convex PNU closed-form semisupervised squared loss aligned with pywsl; η mixes P/NU)
      puet.py                             (native: CPU PU Extra Trees ensemble driven by nnPU/quadratic risk)
      cvir.py                             (CVIR paper-weighted objective, injected CNN/CPU batching, and two-stage cost snapshots)
    bias_aware/
      __init__.py                         (public re-exports)
      pusb.py                             (native: PUSB selection-bias scores; LR source classifier and configurable threshold preserve ordering rather than posterior probabilities)
      pusb_kernel.py                      (native: official-aligned RBF PUSB adapter)
      lbe.py                              (native: instance-dependent labeling bias; alternating weighted LR posterior/propensity estimation, sklearn)
    deep/
      __init__.py                         (public DGPU/GradPU/RobustPU/SplitPU/LaGAM/InfoMaxPU/SelfPU/WConPU classifier and vision-factory exports)
      grad_pu.py                          (native: GradPU input-gradient penalty and unnormalized positive weighting; 2-D MLP)
      robust_pu.py                        (native: Robust-PU nnPU pretraining and self-paced weighted pseudo-negatives; 2-D MLP)
      split_pu.py                         (native: Split-PU nnPU teacher, prediction-disagreement groups, and branch consistency; 2-D MLP)
      lagam.py                            (native: LaGAM independent-support meta-gradients, injected CNN/CPU batching, and stage costs)
      self_pu.py                          (native: SelfPUClassifier dual-student self-paced learning, meta-reweighting, and distillation aligned with official implementation)
      infomax_pu.py                       (native: InfoMaxPURepresentation/InfoMaxPUClassifier PURL+nnPU pipeline and PU-SMI objective)
      weighted_contrastive_pu.py          (native: WeightedContrastivePUClassifier prototypes, SAT, and momentum-queue weighted contrast under the WConPU paper protocol)
      vision.py                           (shared: public build_encoder/CNN_BACKBONES CNN encoder factory)
      dgpu.py                             (native: DGPUClassifier discriminative-generative cooperation and generator fit/sample/warm_start protocol)
      _validation.py                      (shared: validate_encoder_features checks 2-D, finite outputs, and feature_dim>=1)
      pan.py                              # PAN predictive-adversarial MLP, native OS
      pulns.py                            # PULNS RL negative selection with independent clean support
      gen_pu.py                           # GenPU two generators, three discriminators, and synthetic PN
      holistic_pu.py                      # Holistic-PU trends and natural-break pseudo-labels
      p3mix.py                            # P3Mix/P3Mix-E/P3Mix-C experimental classifiers and mixup components
      lbe_neural.py                       # dual-network Adam generalized EM, stable E/M objectives, and independent CNN adaptation
      tedn.py                             # TEDn alternating training, internal held-out BBE, unweighted objective, and training-error stopping
    research/
      __init__.py                         (public joint-shift classifiers, paper objectives, and JOINT_SHIFT_METHODS factory)
      joint_shift.py                      (joint-shift approximation using soft class-conditional domain ratios and alternating PU updates)
      dynamic_joint_shift.py              (paper-style joint weighting/classification objectives and dynamic shared-feature training)
      joint_shift_baselines.py            (JointShiftPUBaseline trPU/tePU/fine-tune/MMD controls and build_joint_shift_estimator ablation factory)
    __init__.py                           (package marker; classifiers re-exported through subpackages and package root, with no symbols here)
  metrics/
    __init__.py                           (public classification-metric exports)
    classification.py                     (implemented: PU-only pu_zero_one_risk/pu_recall metrics and supervised wrappers)
  diagnostics/
    __init__.py                           (public diagnostic-report classes and assumption-analysis functions)
    benchmark.py                          (audit_benchmark_results: result-directory completeness, provenance, and metric-column checks)
    report.py                             (implemented: build_diagnostic_report combines data profiles and model-metric diagnostics)
    sensitivity.py                        (sensitivity to class-prior/mean-labeling-propensity assumptions)
    shift.py                              (implemented: analyze_pu_shift covariate-shift audit → PUShiftReport)
    domain_assumptions.py                 (implemented: analyze_domain_assumptions cross-domain prior/labeling-mechanism shift diagnostics)
    shift_monitor.py                      (implemented: PUShiftMonitor windowed repeated shift audits, alerts, and persistent history)
    uncertainty.py                        (implemented: analyze_pu_uncertainty uncertainty, selective prediction, and query queues)
  model_selection/
    __init__.py                           (public splitters and lazily loaded tuning/comparison exports to avoid import cycles)
    comparison.py                         (implemented: PUModelComparator CV comparison/ranking of registered classifiers)
    split.py                              (implemented: PUStratifiedKFold PU-aware splitters retain labeled positives in each fold)
    tuning.py                             (implemented: PUTuner hyperparameter grid search reusing PUPipeline and PU metrics)
  registry/
    __init__.py                           (public re-exports; __getattr__ lazily forwards advisor symbols to avoid import cycles)
    registry.py                           (implemented: thread-safe register_method/get_algorithm central registry and alias resolution)
    metadata.py                           (public: AlgorithmMetadata contract shared by registry and documentation generation)
    builtin_methods.py                    (implemented: built-in metadata and concrete class paths; loaded on demand, with optional explicit prewarm)
  advisor/
    __init__.py                           (public recommendation engine and result data structures)
    recommender.py                        (implemented: recommend_methods/recommend_from_profile data-profile → registered-method recommendations)
    rules.py                              (implemented: score_method rules, method_warnings risk warnings, and ScoringConfig weights)
    _types.py                             (private advisor components: MethodCandidate/RecommendationResult data structures)
  workflows/
    __init__.py                           (public PUPipeline, PipelineReport, and ShiftAware reports)
    pipeline.py                           (implemented: PUPipeline profile→prior→training→CV→diagnostics; mlp/cnn and auto recommendations)
    _evaluation.py                        (metric resolution, PU-aware CV execution, and aggregation)
    _inputs.py                            (input validation, splitter preparation, and CV provenance)
    _models.py                            (private workflow classifier/prior resolution, instantiation, and parameter checks)
    _reporting.py                         (private build_pipeline_report assembly and provenance normalization)
    report.py                             (PipelineReport/PriorInfo/CVMetric data classes; to_dict/to_json/save contracts)
    shift.py                              (implemented: ShiftAwarePUPipeline paired baseline/weighted comparisons → ShiftComparisonReport)
  cli/                                    (argparse subcommands and thin workflow wrappers)
    __init__.py                           (build_parser/main assembles subcommands; user errors → exit 1)
    run.py                                (run: two CSV inputs, three standard output files, exit codes 0/1/2)
    info.py                               (list-methods/list-priors: registered classifiers and priors, with explicit km1/km2 entries)
    demo.py                               (make-demo-data: SCAR X/y_pu/y_true CSV files checked for five-fold usability)
    profile.py                            (profile: profiles and SCAR/SAR diagnostics; writes profile.json for recommend)
    recommend.py                          (recommend: reads profile.json, recommends methods/estimates priors, writes recommendation.json)
    sensitivity.py                        (sensitivity: prior/propensity scans with fixed predictions; writes sensitivity.json)
    audit_benchmark.py                    (audit-benchmark: result audits, PASS/FAIL summaries, optional JSON)
    skill.py                              (skill: installs the wheel's pu-workflow skill into user agent directories)
    shift.py                              (shift-audit/shift-run: shift audit and weighted-training comparison; writes shift_report/comparison artifacts)
    deployment.py                         (shift-monitor/review deployment commands and artifact exports)
  ui/                                     (optional Streamlit UI; core installation does not import streamlit)
    __init__.py                           (public data loading, run configuration, deployment analysis, and parameter catalogs)
    app.py                                (Streamlit main(): upload→configuration→background training→results)
    configuration.py                      (run configuration/session-state adapters: apply_run_configuration/parse_json_mapping)
    data.py                               (CSV/NPY parsing and validation: load_feature_data including 4-D NCHW, and load_label_data)
    execution.py                          (execute_analysis normal/tuning/comparison dispatch; finalize_run history completion)
    history.py                            (process-wide history: thread-safe append/snapshot; retained on refresh, cleared on restart)
    parameters.py                         (parameter metadata and typed widgets: classifier_catalog signature reflection/parameter_schema/render_parameter_form)
    results.py                            (render_results metrics, diagnostic hints, and JSON/MD/pkl downloads)
    runtime.py                            (submit_background/BackgroundRun thread pool, progress snapshots, and cooperative cancellation)
    launcher.py                           (pu-toolbox-ui main() wraps streamlit run app.py)
    deployment.py                         (UI-independent analyze_deployment_window and render_deployment_tools panel)
  experiment/                             # public four-role experiments, PA/OA, independent testing, read-only summaries/reports
    cvir_adapter.py                       # independent CVIR wiring for explicit OS alpha_U and population-pi/view-prior conversion
    pulns_adapter.py                      # extra clean-support role/group isolation, budget, and endpoint OA gates; ADNI isolation marked unverifiable
    __init__.py                           (public ExperimentRunner/DatasetBundle/strategies/resource summaries/select_threshold)
    bundle.py                             (four-role data contract: DatasetPart/DatasetBundle/validate_bundle)
    datasets.py                           (eight Survey datasets, official/derived test policies, deterministic four-role splits)
    feature_adapter.py                    (CNN-to-2-D features, encoder/feature hashes, and leaderboard fairness-group gates)
    image.py                              (train-only image statistics, scaling, ResNet-18/augmentation configuration, and hashes)
    candidate_image.py                    (PULDA shared ResNet-18 / GradPU private BN-free image construction; technical, without formal admission)
    text.py                               (fixed SBERT vectors, revision tracking, verified content-addressed cache keyed by deduplicated corpus rather than order; same text has equal features across seeds)
    training_views.py                     (Survey boundary adapter: ledger gates, routing, legacy run_view vocabulary, manifest records; delegates role construction to core.training_views)
    manifest.py                           (manifest writing/loading and required-key validation)
    protocols.py                          (Generator/Trainer/SelectionProtocol strategy ABCs)
    resources.py                          (candidate/tuning/GPU resource accounting, environment detection, cross-seed summaries)
    runner.py                             (ExperimentRunner validation, PU views, candidate training, PA/OA selection, independent testing)
    strategies.py                         (SCAR/SAR-LBE generators, PA/OA selection, three Trainer implementations, select_threshold)
    tracking.py                           (EpochRecord/RunTrajectory/SelectionArtifact/RunResult provenance data classes)
    survey_execution.py                   (P2.0a shared model/image assembly, random frozen-adapter validation cache, original-space SAR labeling)
    survey_protocol.py                    (single v3.2 execution configuration, parameter/budget locks, actual run identities; historical results load independently)
    checkpoints.py                        # per-epoch weight snapshots, digest checks, and independent inference restoration
    split_archive.py                      # cross-host artifacts: per-file digest index, deterministic tar, receiver verification
    pilot_plan.py                         # deterministic grids, completion checks, checkpoint space estimates, scope filters
    method_ledger.py                      # Survey method-ledger access; enumeration separated from the programmatic source of truth
    survey_summary.py                     (pure statistics: seed means/sample standard deviations, groups, deduplicated costs, missing metrics)
    survey_recipe_registry.py             # P4.1 recipe registry: closed sets/schema/digests and manifest binding checks (pure functions)
    survey_provenance.py                  # report identity: deterministic assembly of input roots and digests, plus explicit Git commit/status lookup
    batch_evidence.py                     # §5.6 content-addressed commit/uv.lock/protocol/environment evidence; verifies run/batch identity and builds manifest references
    survey_workbook.py                    # report → four protocol §5.1 sheets + diagnostic/coverage/readme; no ranking, incomplete seeds only in coverage (pure functions; openpyxl)
    survey_diagnostics.py                 # per-method observations and advisory flags from plan §1.4.5 signals; lists unavailable signals (pure functions)
    survey_recipe_binding.py              # v2 candidate-pool artifacts, candidate identities, PA/OA trajectory reconciliation
    clean_support_adapters.py             # Self-PU full-meta / LaGAM independent-support audits and endpoint OA wiring
    support_data.py                       # independent clean support: raw reconstruction/frozen-split carving, seed policy, five-role artifact validation
    support_checkpoints.py                # independent-support meta snapshots and PA+clean-support/OA selection tracks
    genpu_recipe.py                       # single-candidate GEN-PU engineering recipe, six-network cost, PN-stage checkpoint guards
    candidate_image_runner.py             # PULDA/GradPU image trajectories, actual stage costs, snapshot-byte guards
    survey_profile_identity.py            # explicit profile/algorithm mapping, manifest identity validation, and clean-support/CVIR execution guards
    survey_staged_budget.py               # explicit GEN-PU/PULDA stage budgets, native costs, snapshot-size verification
    survey_episodic_budget.py             # native Robust/Split/Holistic episodes, stopping rules, paid LZO cost checks
    classical_candidate.py                # non-epoch cost receipts for PUET trees and RP nested fits
    kernel_candidate.py                   KLDCE prior policy, dense-memory/feasibility preflight, and cost receipts
    survey_backlog.py                     # current execution adapters; prior/stage/actual-resource checks
    survey_conclusions.py                 # §1.4.5 signal attribution separates literature facts, observations, and inference; rejects unsupported conclusions (pure functions)
    reporting.py                          (summary text/JSON/CSV, four-group workbooks, diagnostics, and conclusions)
    results.py                            (read-only v3 manifests, identity/budget consistency, seed aggregation, provenance disclosure)
    research_settings.py                  # sampling, risk pools, prior inputs, supervision, representation, and readable summaries
  __init__.py
  run_config.py                           (implemented: portable RunConfiguration JSON shared by CLI/UI; schema_version validation)
  progress.py                             (CancellationToken/emit_progress cooperative cancellation and progress callbacks)
```

## 3. Tests (`tests/`)

```text
tests/
  contract/                             # reusable contracts for registered classifiers
    test_classifier_baseline.py         # fit/predict/decision_function/get_params
    test_capability_declarations.py     # four groups of capability invariants
    test_build_encoder_export.py        # build_encoder dual exports: mlp→None, ValueError, structural consistency
    test_ledger_registry_consistency.py # ledger/registry fields, calibration, sampling, HENG958 owner review, no line-number citations
    test_public_exports.py              # __all__ names must exist and star imports must work; complements check_api_docs
    test_layer_boundaries.py            # experiment dependency ratchet: AST includes local imports; subprocess graphs detect transitive imports
    test_registry_dynamic_import.py     # approved Core→Algorithms/Estimation importlib exception; registry observations cover edges invisible to static AST/import graphs
    conftest.py                         # shared ledger_methods fixture loads ledger and registers built-ins
    test_pulns_literature_anchors.py    # PULNS Table 3 anchors, protocol sources, dispersion-unit ambiguity
  estimators/                           # method-specific MATH/PROPERTY/API tests
    risk/
      test_ldce_math.py                 # LDCE MoM, covariance, m-update, gradients (MATH)
      test_ldce_core.py                 # LDCE fit/predict/attributes/labels (unit)
      test_ldce_api.py                  # LDCE convergence/constraints/API/errors/regressions (unit)
      test_kldce_math.py                # KLDCE formulas, QP oracle, bias restoration (MATH)
      test_kldce_property.py            # KLDCE constraints/robustness (PROPERTY)
      test_kldce_kkt.py                 # true KKT residuals with multiplier recovery, ACS convergence (unit)
      test_kldce_smo.py                 # native SMO solver, equivalence, incremental gradients (unit)
      test_class_prior.py               # shared prior derivation and centroid denominator helpers (unit)
      __init__.py
    research/
      test_dynamic_joint_shift_math.py  # hand-calculated paper objectives, correction boundaries, dynamic training
    __init__.py
  unit/                                 # component-specific logic
    diagnostics/
      test_report.py                    # report metrics, input contracts, serialization
      test_sensitivity.py               # assumption-scan formulas, boundaries, exports
      test_benchmark_audit.py           # persisted benchmark artifact audits
      test_shift.py                     # shift audits, relative weights, ESS, serialization
      test_domain_assumptions.py        # two-domain prior/mechanism decomposition, feasibility, serialization
      test_shift_monitor.py             # window deltas, alerts, history restoration, configuration gates
      test_uncertainty.py               # rejection coverage, three query policies, per-row artifacts
    estimators/
      test_elkan_noto.py                # OOF propensity near true c, weighted refit, priors, normalized probabilities, pipeline
      test_upu.py                       # separable boundaries, prior sensitivity, loss variants, PU-CV λ search, validation risk
      test_nnpu.py                      # nnPU training dynamics and early stopping
      test_nnpu_encoder.py              # encoder injection, Sequential composition, 4-D boundaries, default regression
      test_bias_aware.py                # LBE bounded propensity/EM, all-positive PUSB errors, parameter/prior validation
      test_pusb_kernel.py               # official-aligned formulas, CV, determinism
      test_dist_pu.py                   # optional torch, mixup boundaries, class-prior/epochs validation
      test_vpu.py                       # golden formulas, normalization, registry/pipeline, checkpoints, CUDA smoke
      test_vpu_ts_view.py               # core-role TS/OS pools, bare default equals TS, validation requests only train roles
      test_pulda.py                     # golden dual losses, two-stage trajectories, registry/pipeline, checkpoints, CUDA smoke
      test_grad_pu.py                   # golden formulas, API/registry, determinism, checkpoints, CUDA smoke
      test_robust_pu.py                 # self-paced weights, API/registry, determinism, checkpoints, CUDA smoke
      test_split_pu.py                  # JS formulas, stages, API, determinism, checkpoints, CUDA smoke
      test_lagam.py                     # second-order meta-labels, support gates, cluster contrast, CUDA smoke
      test_puet.py                      # golden node risks, ensembles, determinism, pickle, pipeline
      test_self_pu.py                   # pace/meta/EMA/three-stage training
      test_deep_pu.py                   # InfoMax PU/WConPU/DGPU API and registry
      test_deep_pu_vision.py            # WConPU vision backbones and tensor augmentation
      test_vision.py                    # shared build_encoder entry point
      test_deep_vision_pickle.py        # vision pickle round trips, E2/E3 regressions; optional torch
      test_llsvm.py                     # separable synthetic training, calibration against all-positive output, early stopping, explicit priors
      test_encoder_validation.py        # validate_encoder_features: 2-D, finite values, dimension boundaries
      test_nnpu_gpu.py                  # CNN encoder GPU execution; skips without CUDA
      test_nnpu_ts_view.py              # TS/OS union input, view gates, OS validation
      test_upu_ts_view.py               # independent golden TS/OS derivation, solver pools/denominators, RBF centers
      test_pusb_kernel_ts_view.py       # golden TS/OS roles/denominators, frozen OS baseline
      test_pusb_kernel_view_invariants.py # unchanged centers/folds/threshold pools; overlapping roles without physical duplication
      test_dist_pu_ts_view.py           # OS/TS views, frozen expectation baseline, independent author-histogram golden
      test_dist_pu_view_invariants.py   # fit-role sets, bitwise equivalence with zero regularizers, unchanged Mixup pools/RNG
      test_self_pu_ts_view.py           # view expectations, frozen OS baseline, A ablation/B clean-meta, independent mixture arithmetic, row-count alpha, bitwise equivalence
      test_self_pu_view_invariants.py   # negative-role count equals untrusted U, calibration only in training assembly, equal RNG/forward counts, unchanged trusted population/pace
      test_cvir.py                      # negative retention order, prior gates, explicit TS routing
      test_pan.py                       # formulas, gradient isolation, save/load
      test_pulns.py                     # rewards, support isolation, empty negatives
      test_rank_pruning.py              # OOF, pruning, noise weights
      test_gen_pu.py                    # anti-GAN direction and risk-pool calibration
      test_holistic_pu.py               # paper/code differences and training
      test_p3mix.py                     # training variants, checkpoints, engineering boundaries
      test_p3mix_math.py                # source formulas and pure candidate/MixUp helpers
      test_pan_cnn.py                   # dual-encoder CNN updates, isolation, checkpoint replay
      test_pulda_cnn.py                 # two-stage CNN, CPU-resident images, weight replay
      test_pulda_augmentation.py        # per-batch two-stage augmentation, MixUp order, rejection, template isolation
      test_grad_pu_cnn.py               # original-image second-order gradients, BN-free encoder, weight replay
      test_grad_pu_training_recipe.py   # actual Adam schedule, batch/update counts, run declarations
      test_grad_pu_memory.py            # offload equivalence of second-order gradients, RNG, full updates
      test_grad_pu_split_backward.py    # component/GP-chunk vs joint backward loss, parameter, RNG equivalence
      test_pulda_gradpu_inference_modes.py # per-module inference-mode restoration, errors, BN/RNG invariance
      test_split_pu_review_characterization.py # #122 JSD direction/denominator and mini-batch population-mixture counterexample
      test_robust_pu_cnn.py             # CNN stages, BN isolation, CPU weights/snapshot replay
      test_split_pu_cnn.py              # multilayer CNN distillation, stage isolation, snapshot replay
      test_p3_budget_selection_evidence.py # five-method budgets, source adaptation, selection-signature boundaries
      test_p3_source_behavior.py        # independent scalar source formulas, gradients, boundaries for five methods
      test_holistic_pu_cnn.py           # two-stage CNN, template/BN isolation, snapshot replay
      test_holistic_pu_initialization.py # later-stage reinitialization, Adam isolation, cumulative costs
      test_holistic_pu_lzo.py           # positive-mixup endpoint selection, state restoration, full-budget costs
      test_pulns_cnn.py                 # CNN/reward-probe isolation, state extraction, best model, full-budget costs
      test_gen_pu_cnn.py                # pixel generation, four-CNN isolation, G gradients, six-network costs, replay
      test_lbe_review.py                # source formulas, probability contracts, failed refit invalidation, provenance/calibration
      test_lbe_neural.py                # budgets, templates/randomness, serialization, OS-only boundaries
      test_lbe_neural_math.py           # joint posterior, full-likelihood gradients, independent extreme-logit references
      test_vpu_schedule.py              # source Adam resets, fixed steps, default compatibility
      test_vpu_encoder_stream.py        # injected CNN, global streamed diagnostics, device batching, snapshot compatibility
      test_cvir_source_encoder.py       # source differences/independent gradients, CNN batching, U ordering, stage snapshots
      test_lagam_encoder_stream.py      # CNN/support batching, BN/RNG isolation, snapshots, budget rejection
      test_lagam_virtual_scope.py       # independent finite differences, two-column perturbations, mathematical tie references
      test_puet_adaptation_contract.py  # child-leaf constraints, full thresholds, negative-gain rejection, out-of-range feature-count adaptation
      test_tedn.py                      # formula gradients, native TS contracts, real BBE synthetic training
      test_tedn_training.py             # pre-update estimates, CNN holdout isolation, convergence, checkpoint replay
      test_robust_pu_source_contract.py # independent formulas/actual gradients, per-row moving weights, source-schedule subset
      test_candidate_fit_failures.py    # failed refits/callbacks and unchanged success paths for five candidates
      test_extended_candidate_fit_failures.py # extended parameter binding/cancellation and successful-refit state guards
      test_pan_prediction_modes.py      # per-layer inference modes, BN/RNG isolation, error recovery
      test_pan_source_contract.py       # independent analytical gradients, actual alternating sampling, source binding
      test_rank_pruning_source_contract.py # independent sort/count, real OOF isolation, final reweighting costs
      test_nnpu_lifecycle.py            # failed-refit invalidation, per-layer scoring-mode restoration, successful-training equivalence
      test_self_pu_lifecycle.py         # failed-refit invalidation, teacher scoring isolation, OS/TS selection equivalence
      test_self_pu_parameters.py        # finite real/integer budgets, early-failure RNG isolation, NumPy scalars
      test_dist_pu_encoder.py           # two-stage/CPU-stream encoders, singleton BN, legacy MLP/snapshot compatibility
      test_p3mix_input_isolation.py     # in-place network-input isolation and error-mode restoration across three variants
      test_cvir_input_modes.py          # in-place input isolation, inference modes, early parameter failure
      test_lagam_paper_behavior.py      # independent BCE/contrast references, zero gradients, hard-label meta-step
      test_pulns_checkpoints.py         # episode callbacks, default compatibility, failed-state invalidation, internal-best restoration
      test_cvir_calibration.py          # training-risk-pool calibration, bitwise expanded-role references, failure/replay boundaries
      test_p3mix_batching.py            # combined batching, single-role tail batches, default compatibility
      test_kldce_feasible_start.py      # single-equality box feasibility, independent LP, large-scale regression
    losses/
      test_nnpu_loss.py                 # nnPU golden tests (MATH + PROPERTY)
      test_upu_loss.py                  # uPU golden tests (MATH + PROPERTY)
      test_llsvm_loss.py                # LLSVM loss golden tests (MATH)
      test_pnu_loss.py                  # PNU loss golden tests (MATH)
    metrics/
      test_classification.py            # hand-calculated average_precision/pu_recall/pu_estimated_precision/pu_negative_rate, sklearn references, empty groups, low priors
    model_selection/
      test_comparison.py                # model comparison, failure isolation, selection direction
      test_split.py                     # positives per fold, disjoint complete coverage, approximate PU ratios, MIN_POSITIVE_SAMPLES
      test_tuning.py                    # PUTuner grids, selection direction, unavailable metrics
    ui/
      test_history.py                   # process-wide history and final entries (D9 regression)
      test_ui_helpers.py                # upload parsing, typed parameters, configuration state, model catalogs
      test_runtime.py                   # background runs, progress snapshots, cooperative cancellation
      test_deployment.py                # deployment-analysis helpers and error paths
      test_cnn_candidates.py            # registry-capability-driven CNN candidate/backbone catalogs
    prior/
      test_recpe.py                     # copy-fraction boundaries, custom base classifier/estimator, default KM2, None confidence interval, density-ratio quantiles
      test_pen_l1.py                    # automatic median-bandwidth σ, separated/low-prior bands, bounded estimates
      test_kernel_mean.py               # KM1/KM2 kernel-mean prior estimation
      test_bbe.py                       # independent tail formulas, confidence criteria, repeated scores, train/holdout isolation
      test_prior_semantics.py           # prior targets, mass identities, invalid domains, identifiable mixtures
    preprocessing/
      test_pu_labeling.py               # six public PU/PNU label functions, dispatcher, validation, determinism
      test_selection_bias.py            # SCAR/SAR propensities, labeling-rate calibration, synthetic data
      test_profiling.py                 # pu_data_summary/pnu_data_summary counts/ratios, NaN/Inf flags, sparse input, label normalization
      test_data_profiler.py             # structured reports, quality rules, audit diagnostics
      __init__.py
      test_sampling_semantics.py        # same-pool quotas, independent TS, provenance, legacy-entry compatibility
    scripts/
      test_check_api_docs.py            # API reference public-symbol coverage gate
      test_check_baseline_configs.py    # baseline-configuration consistency gate
      test_check_doc_links.py           # orphan/Markdown-link gate
      test_check_format.py              # ruff lint/format gate
      test_check_math_rendering.py      # MathJax rendering gate
      test_check_skill_sync.py          # mirrored skill consistency gate
      test_check_test_quality_exemptions.py # test-quality exemption review
      test_generate_structure.py        # structure generator --check/--update
      test_check_comment_quality.py     # legacy markers, bare TBDs, trailing comments
      test_check_doc_links_sources.py   # rule-1 path references in .py comments
      test_generate_layer_deps.py       # layer dependency generator
      test_check_doc_links_docs_scope.py # expanded reverse rule-2 scope after docs generator integration
      test_generate_structure_docs_scope.py # per-root suffix coverage and grouped-subtree exemptions
      test_check_comment_quality_partitions.py # coverage assertions, empty-scan rejection, partition severity, monotone ratchet
      test_check_doc_links_rule3.py     # architecture-table/registry checks fail on missing extraction; separate file respects 15-test limit
      test_check_project_metadata.py    # synthetic-repository main() failure paths; each mutation exits nonzero
      test_check_test_quality_matcher.py # category-token boundaries and collisions; prefixes cannot be sole coverage credit
      test_check_survey_recipe_registry.py # main() rejects malformed data, missing profiles, bad schema
      test_collect_survey_public_data.py # verified downloads, reuse/no-overwrite, pinned ADNI-substitute digest
      test_recipe_production_gate.py    # formal-registry conditional CI, lock rejection, determinism
      _lbe_comparison_helpers.py        # synthetic four-role fixtures for exploratory LBE comparisons
      test_lbe_comparison_execution.py  # paired fits, selected-rule restoration, test isolation
      test_lbe_comparison_plan.py       # split/seed/budget identities and preflight rejection
      test_lbe_diagnostics.py           # branch symmetry, independent SCAR gradients, diagnostic plans
      test_source_provenance.py         # exploratory-script Git directory and actual loaded implementation identity
      test_clean_support_execution.py   # actual CPU training for three methods, explicit budgets, OA replay, warning rejection
      test_prepare_clean_support.py     # raw readers, default preview, artifact writing, source/output boundaries
      test_clean_support_checkpoint_execution.py # dual-track support CLI, plan identity, selected-model replay
      test_genpu_recipe_execution.py    # plan digests, actual dual-track CLI, post-selection reclamation, replay
      test_candidate_image_execution.py # image CLI plan binding, PA/OA replay, nonformal boundaries
      test_selfpu_ts_support_execution.py # full-meta TS CLI view binding, plan guards, dual-track replay
      test_survey_image_support_execution.py # support assembly through image paths: native-CNN scaling, adapter features, unchanged tables
      test_support_protocol_context.py  # support CLI version, c token, receipt provenance deviations
      test_candidate_image_adni.py      # ADNI-substitute grayscale images, dual-selection replay, nonclinical source boundaries
      test_pan_image_cli.py             # four-role OS image commands, dual-selection replay, rejection of nonformal budgets
      test_vpu_image_cli.py             # four-role TS commands, normalization replay, explicit parameters, role isolation
      test_p3mix_image_execution.py     # explicit image budgets, dual-protocol replay, reclamation
      test_classical_candidate_execution.py # PUET/RP explicit plans, four-role replay, failure evidence
      test_cvir_image_execution.py      # image priors, view locks, dual-protocol replay
      test_kldce_candidate_execution.py # actual SCAR binding, four-role replay, failure evidence
    advisor/
      test_recommender.py               # filtering, scoring, outputs
      test_scoring_rules.py             # recommendation scores and boundaries
      __init__.py
    cli/
      test_classifier_params.py         # classifier-param JSON parsing and errors
      test_cli_main.py                  # CLI entry smoke
      test_run_deep.py                  # deep run paths: .npy input and parameters
      test_info.py                      # list-methods/list-priors
      test_demo.py                      # make-demo-data
      test_skill.py                     # bundled skill installation
      test_audit_benchmark.py           # audit-benchmark
      test_profile.py                   # profile, pu-workflow step 1
      test_recommend.py                 # recommend, pu-workflow step 2
      test_sensitivity_cmd.py           # sensitivity, pu-workflow step 4
      test_shift_audit.py               # shift-audit parameters, errors, artifacts
      test_shift_run.py                 # shift-run parameters, errors, paired artifacts
      test_deployment_commands.py       # shift-monitor/review parameters and artifact journeys
    core/
      test_device.py                    # shared resolve_device helper
      test_training_view_contract.py    # neutral OS/TS roles/order, owned frozen/borrowed arrays, no estimator→experiment imports
      test_random.py                    # direct check_random_state tests including differences from inline construction
      test_fit_state.py                 # exceptions, cancellation, signatures for fit-state guards
      test_fit_guard_coverage.py        # candidate fits retain shared failure guards and do not declare clean roles; AST, no source hashes
    workflows/
      test_pipeline_report.py           # prior-reliability context in PipelineReport.summary()
      test_metric_availability.py       # compute_metric availability and probability gates
      test_architecture_capability.py   # fail-loud capability/signature drift
      test_report_provenance.py         # four architecture-capability fields for bare mlp/full native_cnn
      test_label_semantics_entrypoints.py # CLI/UI/tuning/recommendation label-semantics gates
      test_pipeline_label_semantics.py  # rejects mismatches in construction/run/clone/auto
      test_pipeline_prior_targets.py    # OS/TS estimation targets, per-fold consumption, real classifiers, population evaluation
    experiment/
      test_cvir_known_prior_adapter.py  # alpha_U/population pi, candidate roles, manifest/path rejection
      test_pulns_support_adapter.py     # all-role isolation, extra-label budgets, OA-only rejection
      test_bundle.py                    # valid bundles, view rejection, disjoint indices, labels
      test_datasets.py                  # eight-dataset mapping, official/derived tests, deterministic four-role splits
      test_feature_adapter.py           # four-role CNN features, hashes, fit scope, split/seed/budget fairness
      test_image.py                     # train-only statistics, channels/scaling, ResNet-18, no eval augmentation
      test_candidate_image.py           # seeded shared/private PULDA/GradPU image paths, BN, nonformal manifests
      test_candidate_image_receipt.py   # actual-fit receipts and input/template/parameter binding; pretraining mismatch rejection
      test_text.py                      # fixed SBERT dimension/revision/cache, tampering, permutation invariance, duplicate-text rows
      test_training_views.py            # TS/OS P/U loss sets, native assumptions, eval/test rejection, manifest hashes
      test_manifest.py                  # round trips and fail-loud required keys
      test_protocols.py                 # strategy ABC contracts and generate return shapes
      test_runner.py                    # small-PU/CNN end-to-end smoke and fail-loud errors
      test_prepare_survey_splits.py     # tabular/image/text split pipeline with synthetic fixtures
      test_survey_script.py             # run_survey_experiment.py smoke, mandatory priors, missing files
      test_survey_script_cvir.py        # explicit CLI prior conversion, views, snapshot capacity/reclamation, existing evidence
      _survey_script_helpers.py         # dynamic loading and base/SAR four-role split fixtures
      test_survey_script_sar.py         # OA-only/c-token gates, audit fields, determinism
      test_strategies_labeling.py       # fixed counts, posterior positive pools, seed determinism
      test_shared_sampling.py           # general/experiment sampling consistency, label hashes, subsequent RNG streams
      test_strategies_selection.py      # thresholds, PA/OA, true-label/non-PU rejection, empty trajectories
      test_tracking.py                  # trajectory/selection-artifact/result data-class fields
      test_trainers.py                  # Fit/DeepFit/Supervised strategies
      test_runner_oracle.py             # true-label forwarding, OA-only, manifest annotation, c independence
      test_runner_oracle_guards.py      # fail-loud view/generator/trainer/prior/mechanism mismatch checks
      _runner_oracle_helpers.py         # four-role bundles and recording trainers
      test_survey_execution.py          # shared MLP/random encoders, adapter cache, original-space SAR consistency
      test_survey_protocol.py           # execution matrix, parameter locks, fairness, formal-result blocking
      test_survey_protocol_validation.py # review fields, KLDCE feasibility, independent Self-PU budget gates
      test_survey_protocol_cnn.py       # real one-epoch CPU ResNet smoke and encoder provenance
      test_survey_protocol_review.py    # v3.x policy status, review_items, deterministic digests
      test_survey_script_protocol.py    # versioned scripts, split/seed matching, oracle deduplication, CIFAR cache hits
      test_checkpoint_selection.py      # different PA/OA epochs, leakage guards, independent-restoration counterexamples
      test_epoch_checkpoints.py         # callbacks, weights, CPU/CUDA save/load, deep-method coverage
      test_checkpoint_coverage.py       # declared coverage rather than saved set; missing components retain R9 block
      test_checkpoint_disk_preflight.py # pure capacity arithmetic, boundaries, execution-mode levels, size estimates
      test_dataset_provenance.py        # verified artifact facts, official license wording, download sources/digests
      test_split_archive.py             # index/data mismatch, truncated artifacts, deterministic packaging
      test_pilot_plan.py                # matrix expansion/batches; rejected, missing-c_token, or malformed manifests never complete
      test_pilot_plan_dimensions.py     # native_2d default dimensions; image rows do not require feature dimensions
      test_batch_evidence.py            # two identities, content addressing, tampering, same-batch validation, per-commit retention
      test_survey_pilot_driver_protocol.py # absent --protocol defaults to survey-v3.2; explicit selection respected; unknown files error
      test_survey_pilot_driver.py       # no batch without π, explicit conflicting-π override, argv forwarding
      _survey_pilot_helpers.py          # dynamic scripts, subprocess records, split-directory fixtures
      test_survey_pilot_request_scope.py # protocol controls plans/batches; unrelated bad splits do not block shards; snapshot failures and matrix-specific names
      test_survey_pilot_subset.py       # --datasets scope, unknown-name rejection, --plan-json partitions, execution integration
      test_survey_pilot_execution_scope.py # three-axis intersections; unknown/unrunnable/empty rejection; unrunnable rows retained outside plans
      test_survey_pilot_scope_wiring.py # shared plan/estimate/split/resume scope, full protocol digest/selection, scope-sensitive explicit TS domain
      test_survey_pilot_batch_contract.py # historical five-batch contract: 215/215/110/70/35, disjoint union 645, exclusions absent
      test_runner_class_prior.py        # PA π precedence: run over split; provenance; missing/incomplete rejection
      test_leaderboard_run_view.py      # distinct views cannot share a leaderboard group
      test_method_ledger.py             # enum/Chinese-note normalization, missing fields, invalid values
      test_runner_view_manifest.py      # success/failure run_view/calibration_applied provenance
      test_survey_script_view.py        # ledger defaults, CLI overrides, gates, artifact-path views
      test_training_view_routing.py     # signature-based forwarding; undeclared views fail; no OS parameter
      test_pilot_resume_view.py         # per-cell default views, separate OS/TS identities, invalid artifacts do not block scans
      test_checkpoint_storage_profiles.py # path/backbone/model_family dispatch, actual serialization bounds, exact pilot bytes
      test_survey_formal_eligibility.py # D24 canonical admission, genuine deviations blocked, ablation labels retained
      test_checkpoint_reclaim.py        # D25 selected-only weights, reclaimed semantics, disabled by default
      _survey_summary_helpers.py        # manifest trees, configs, audit/summary CLI fixtures
      test_survey_summary_boundaries.py # different views/mechanisms/paths/protocols never merge
      test_survey_summary_clean_support.py # independent support tracks, extra supervision isolation, diagnostics, formal gates
      test_survey_summary_states.py     # closed row-identity c_token/oracle and status/priority/reason vocabularies
      test_survey_summary_stats.py      # five-seed means/sample SD, missing seeds reported without filling, cost sums
      test_survey_summary_measures.py   # separate missing measurements, GPU peaks, secondary metrics, and selection identities
      _survey_recipe_registry_helpers.py # fixtures derived from frozen protocol rather than copied by hand
      test_survey_recipe_registry.py    # schema, candidate pools, digest rules
      test_survey_recipe_registry_manifest.py # fail-closed binding with legacy compatibility
      test_survey_recipe_registry_sources.py # source-evidence citation rules
      test_survey_workbook.py           # sheet assignment, separate clean support, LBE-A/B groups, missing seeds in coverage, unknown mechanisms error
      test_survey_diagnostics.py        # flag conditions, excluded incomplete seeds, unavailable signals, no paper-equivalence claims
      test_export_survey_results_cli.py # config/summary.json to workbook/cards/conclusions; malformed inputs do not leave partial output
      test_survey_provenance.py         # deterministic identity assembly, input roots, Git stamping, and missing-Git degradation
      test_p3_feature_adapter_smoke.py  # shared frozen-feature engineering smoke for preintegrated methods
      test_p3_pulda_puet_runner_smoke.py # four-role PULDA/PUET TS and PA/OA technical smoke
      test_survey_recipe_registry_draft.py # draft generation, exclusions, candidates, legacy manifest protection
      test_oracle_cnn.py                # CNN PN-oracle training, OA restoration, reclamation
      test_multistage_disk_preflight.py # two-stage peak-checkpoint budget precedence
      test_p3_new_methods_runner_smoke.py # new-method four-role PA/OA and support gates
      test_unimplemented_calibration_routing.py # rejecting an interface does not establish TS calibration capability
      test_multistage_candidate_accounting.py # all-stage Robust/Split update counts and early-stop snapshot bounds
      test_checkpoint_prediction_batching.py # inference-batch registration/restoration, legacy fallback, invalid metadata
      test_checkpoint_training_context.py # stages/epochs/updates, restoration, reclamation provenance
      test_prepare_survey_splits_adni.py # substitute-data ordering/junk/counts, no official test split, multi-archive receipts
      test_survey_image_spec.py         # v1 flat/per_dataset specifications, overrides/validation, modality-based dispatch
      test_survey_image_support.py      # support and four roles share train-fitted preprocessing and frozen encoder
      test_survey_pilot_support.py      # --support-root rejection/loading, seed placeholders, provenance-based resume
      test_survey_pilot_batch_evidence.py # one bundle per batch/mechanism, argv forwarding, dry-run and repeated-run behavior
      test_prepare_survey_splits_connect4.py # fixed one-hot encoding and standardization
      test_prepare_survey_splits_inputs.py # actual input identity and original receipt-byte digests
      test_prepare_survey_splits_mnist.py # input validation and official-test isolation
      test_prepare_survey_splits_news.py # seven-prefix mapping, original text, stable ordering
      test_prepare_survey_splits_wiring.py # new dataset CLI, source evidence, no-overwrite wiring
      test_pilot_recipe_resume.py       # legacy/missing/mismatched bindings never count as complete
      test_runner_recipe_binding.py     # pretraining locks, frozen candidates, selection identities
      test_survey_pilot_recipe_binding.py # recipe preflight, pinned subprocess digests, plan records
      test_survey_recipe_binding.py     # recipe lifecycle, scope, candidates, inner-CV binding
      test_survey_script_recipe_binding.py # single-run CLI rejection, isolated directories, old-artifact protection
      test_survey_recipe_pool.py        # pool digests, closed identities/indices, bounded parameter-search authority
      test_survey_recipe_pool_runtime.py # exclusions/retries across runner, CLI, pilot, resume
      test_neural_lbe_runner.py         # dual CNN, per-round inference snapshots, four-role/test isolation
      test_clean_support_adapters.py    # actual meta-updates, role isolation, supervision budgets, model replay
      test_support_data.py              # eight-dataset support sampling, split carve/seed policy, train-only preprocessing, artifacts
      test_p3mix_literature_anchors.py  # paper hashes, separation of literature/pilot values, pinned PDF/OpenReview identities
      _support_checkpoint_helpers.py    # synthetic CPU roles/methods for dual-track support selection
      test_support_checkpoint_guards.py # support roles, pure-PA rejection, capacity boundaries
      test_support_checkpoint_selection.py # actual meta-updates, dual candidate/snapshot/threshold selection, isolation/replay
      test_support_versioned_epochs.py  # versioned all-epoch candidates, warmup replay, strict coverage
      test_cvir_population_adapter.py   # view-prior conversion, four-role isolation, actual snapshot bounds
      test_genpu_recipe.py              # early rejection, actual six-network cost, bitwise default-training comparison
      test_candidate_image_runner.py    # image snapshots vs bare fit, role isolation, capacity rejection
      test_process_peak_rss.py          # RSS units, unsupported platforms, stable counts
      test_selfpu_ts_support.py         # explicit TS clean-meta four-network references, roles, failure guards
      _profile_identity_helpers.py      # draft identities/matrices; supplies no formal budgets
      test_profile_identity_execution.py # dual-track/native snapshots, actual subprocesses, full-provenance resume
      test_survey_profile_identity.py   # explicit mappings, legacy paths, erroneous/unauthorized binding rejection
      test_survey_staged_budget.py      # stage budgets/roles/capacity/native weights; rejects incorrect formal bindings
      _episodic_budget_helpers.py       # small draft budgets and deterministic synthetic fixtures for three methods
      test_survey_episodic_budget.py    # actual CLI/resume, native weights, stopping/LZO cost rejection
      test_staged_budget_comparability.py # actual stage projections, inner/GAN costs, rejection of cross-batch parameter comparisons
      test_survey_recipe_registry_locked.py # P4.1-B scope, recomputable snapshots, rejection of manual edits/unlocked bindings
      test_survey_canonical_version.py  # each protocol compared with its own canonical; actual subset/parameter deviations retained
      test_survey_full_meta_view.py     # default OS, CLI/resume consistency, ablation/explicit-TS compatibility
      test_pan_image_candidate.py       # dual-encoder isolation, per-epoch C/D costs, unchanged recorder values
      test_vpu_image_candidate.py       # training marginal pool, per-epoch normalization, recreated Adam, actual costs
      test_classical_candidate_cost.py  # actual tree/cross-fit costs and unchanged defaults
      test_p3mix_image_candidate.py     # native-OS image snapshots, teacher/role costs
      test_classical_candidate_receipts.py # per-fit cost receipts, retained failures, persistence rejection
      test_classical_selection_predictor.py # non-epoch thresholds and validation affine replay
      test_cvir_image_candidate.py      # image stage costs, observed priors, error rejection
      test_kldce_candidate.py           # elementwise adaptation equivalence, Phase-I feasibility, resource rejection
      test_cvir_bn_cardinality.py       # spatial BN cardinality, single-sample training, RNG isolation
      test_survey_backlog.py            # v3.1 historical isolation, real 2-D/CNN/SAR training
      test_gradpu_offload_execution.py  # explicit offload policies, frozen recipes, result-path isolation
      test_gradpu_split_execution.py    # split/chunk CLI records, output isolation, policy validation
      test_survey_classical_image.py    # PUET/RP image representations, version isolation, full tree/fold budgets
      test_survey_conclusions.py        # signal attribution, facts/observations/inference, required sources/observations, no unflagged conclusions, determinism
      test_experiment_dependencies.py   # DIY/results do not import Survey process modules
      test_result_recipe_identity.py    # recorded recipe/candidate/selection identity consistency
      test_result_validation.py         # selection evidence, actual budgets, reproducibility, single-seed boundaries
      test_results.py                   # read-only statistics, risk/missing/excluded states, report integration
      test_survey_export_dependencies.py # independent summary export and deterministic isolation
      test_result_oracle_identity.py    # per-seed PN split/feature/budget comparability evidence
      test_result_paths.py              # Windows long-path discovery/reading and provenance normalization
      test_split_identity.py            # identity/isolation from actual four-role indices, batch resume
      test_research_settings.py         # settings, prior provenance, risk views, labeling policies
      test_result_research_settings.py  # per-seed new/legacy settings, historical identities, report exports
    utils/
      test_activations.py               # stable float32/float64 sigmoid extremes, no overflow, saturation bounds
      test_json_scalars.py              # strict narrowing/unboxing, exact error messages, unchanged caller arrays
      test_serialization_hashes.py      # strict rejects nonfinite values, lenient writes NaN, equal finite-payload digests
      test_content_hashes.py            # array dtype/shape/bytes including 0-D; streamed file hashes independent of chunking
    test_basis_single_source.py         # shared RBF kernel formula consistency
    test_run_config.py                  # portable UI/CLI configuration schema and serialization
    test_stable_modules.py              # stable-module inventory completeness; shallow clones read index without tags
  integration/                          # CLI + PUPipeline + registry + estimators
    test_model_configuration.py         # named/required parameters and PUTuner determinism
    test_pipeline.py                    # complete workflow, priors, errors, availability, determinism
    test_pipeline_deep.py               # deep architecture selection; optional torch
    test_run.py                         # in-process end-to-end run, actual CSV IO/training
    test_cli_deep_save_model.py         # run --save-model CNN E2/E3 regressions; optional torch
    test_ui_app.py                      # Streamlit rendering; skips without ui extra
    test_ui_history_flow.py             # process-wide UI history wiring, D9; optional streamlit
    test_pipeline_device.py             # pipeline/deep-estimator device autodetection and epochs defaults
    test_prior_params.py                # prior_params forwarding behind CLI --prior-param
    test_run_errors.py                  # friendly user-input errors
    test_pipeline_sample_weight.py      # per-fold weight forwarding and semantic rejection
    test_shift_workflow.py              # shift audits, adaptation, target evaluation
    test_joint_shift_classifier.py      # synthetic protocol, boundedness, determinism
    test_shift_comparison.py            # paired weighted controls, evidence gates, reports
    test_joint_shift_baselines.py       # four controls, ablations, boundaries, determinism
    test_cv_fold_isolation.py           # varying fold weights, untrained templates, no cross-fold leakage
    test_nnpu_pipeline_cnn.py           # cnn/mlp provenance and encoder pickle round trips
    test_pan_pipeline_cnn.py            # PAN/PULDA native CNN pipelines, seed checks, architecture gates
    test_iris_smoke.py                  # native classifier trainability/prediction smoke without accuracy assertions
    test_pulns_cnn_support.py           # built-in CNN, independent reward set, no test-label use by public workflows
    test_dist_pu_pipeline_cnn.py        # explicit CNN/CV-copy isolation, frozen-feature adapter preservation
    test_p3mix_pipeline_cnn.py          # three variants: real convolutional CV and four-role checkpoint selection/reclamation
    test_survey_recipe_binding_v3.py    # CLI frozen-registry binding, complete fields, verifiable aggregation, tampering rejection
    test_classical_candidate_runner.py  # PUET/RP four-role PA/OA and costs without artificial epochs
    test_diy_experiment_example.py      # prediction/preprocessing/test-label isolation
    test_custom_encoder_example.py      # CNN/frozen features, initialization isolation, save/load
    test_sar_policy_example.py          # Bernoulli/fixed-quota policies, counts, provenance
    test_minimal_examples.py            # all minimal scripts, native-U priors, PNU and review decisions
  e2e/                                  # real-subprocess user journeys in nightly CI
    test_profile_script.py              # subprocess pu-workflow profile
    test_recommend_script.py            # subprocess recommend including profile→recommend
    test_sensitivity_script.py          # subprocess sensitivity
    test_e2e_quickstart.py              # demo→auto run→report and CLI error journeys
    test_e2e_workflow_journey.py        # profile→recommend and demo→profile→recommend→run
    test_e2e_subcommands.py             # subprocess profile/recommend/sensitivity
  benchmarks/                           # benchmark runners
    test_assigned_benchmark_runner.py   # first-five-paper runner
    test_assigned_preflight.py          # first-five-paper readiness audits
    test_deep_pu_benchmark_runner.py    # deep-PU runner
    test_deep_pu_model_selection.py     # clean-validation model selection
    test_deep_pu_official_data.py       # official data loading and splits
    test_pusb_official_data.py          # official-data construction, runner, provenance
    test_pusb_table2_benchmark.py       # claim-safe Table 2 runner
    test_pusb_table2_data.py            # locked loaders and sampling audits
    test_joint_shift_public_benchmark.py # public-data multiseed/CI/sample-overlap benchmark
    test_traditional_pu_benchmark_runner.py # state machine, resume, artifacts, failure isolation
    test_traditional_pu_data.py         # SCAR/SAR/PNU shapes, h, degeneracy
    test_traditional_pu_statistics.py   # success rates, CI, paired differences
    test_traditional_pu_configs.py      # SCAR-60/PNU-6 grids, determinism, disjoint seeds, changed-config resume rejection
    test_traditional_pu_protocol.py     # no oracle-label columns, contamination/resume gates, degenerate flags, all-failed NaN summaries
    test_traditional_pu_compare.py      # §6 pairing/direction/budget tolerance, oracle-only decisions, verdict CSV
    test_traditional_pu_resume.py       # incremental output, interrupted-row checks, completed cells still return 16 rows
    test_traditional_pu_leakage_audit.py # blacklist/duplicates/guard/preflight negative tests and CLI gates
    test_traditional_pu_tuning_round.py # candidate generation and degeneration-rate checks
    test_traditional_pu_tuning_rank.py  # §4 filtering and mixed SCAR/SAR grids
    tuning_helpers.py                   # shared tuning constructors; not a test module
    test_traditional_pu_tuning_round_methods.py # registered-method smoke for nnPU/uPU/LLSVM/Elkan-Noto
  __init__.py                           # package declaration for tests.helpers imports
  conftest.py                           # shared seeds/rng/data pytest fixtures
  helpers.py                            # ordinary data factories imported directly without pytest
  test_import.py                        # import smoke
  test_labels.py                        # label normalization
  test_validation.py                    # dimensions, required classes, sparse input
  test_validation_labels.py             # label_semantics declarations and binary truth labels
  test_registry.py                      # registration
  test_builtin_methods.py               # registry metadata
  estimator_factories.py                # zero-argument factories shared by contract/Iris tests; importable without torch
```

Pytest markers express test purpose: `unit` (component-specific logic), `math` (hand-calculated
references; failure indicates a code bug), `property` (mathematical invariants; failure indicates a
code bug), `contract` (API contracts), `integration` (cross-component behavior), `e2e` (real-subprocess
user journeys), `slow` (expensive tests), and `paper` (paper reproduction).

PR CI runs the fast layer with `-m "not slow and not e2e"`; nightly CI selects the top layer with
`-m "slow or e2e"`.

General and experiment tests are separated by **directory**: `tests/unit/experiment/` covers
Survey protocols as well as runner, protocols, bundle, manifest, tracking, trainers, and strategies.
This does not establish a separate **selection** boundary: these tests use `pytest.mark.unit`,
without a Survey partition marker, and CI does not filter them by path. Consult
[`compatibility.md`](compatibility.md) section 4 for the selection policy. Use the current tree
above for the file inventory rather than historical file-count snapshots.

## 4. Examples (`examples/`)

```text
examples/
  minimal/
    01_elkan_noto.py          (probability correction and weighted refit)
    02_upu.py                 (convex uPU risk minimization)
    03_nnpu.py                (nonnegative PU risk estimation)
    04_pnu.py                 (semisupervised PNU extension)
    05_recpe_pipeline.py      (ReCPE prior estimation with uPU)
    06_sar_simulation.py      (SCAR/SAR labeling mechanisms and propensities)
    07_data_profiling.py      (PU data profiles and SCAR/SAR audit hints)
    08_diagnostic_report.py   (structured diagnostics for a fitted PUSB model)
    09_sensitivity_analysis.py (assumption sensitivity with fixed model outputs)
    10_self_pu.py            (three-stage Self-PU with clean validation)
    11_distribution_shift.py (source/target audits and guarded covariate-weighted adaptation)
    12_shift_decision_tools.py (paired decisions, window monitoring, two-domain assumptions, active review)
    13_dynamic_joint_shift.py (paper-style dynamic joint-shift research solver)
    14_diy_experiment.py      (four-role experiments and three strategy extensions)
    15_pu_sampling.py         (paired OS/independent TS from one target population and prior counts)
    16_prior_targets.py       (known/estimated prior conversion in OS/TS reports and training)
    17_sar_policies.py        (same-pool Bernoulli/LBE-A/B quotas and score provenance)
    18_custom_encoders.py     (nnPU encoder injection, frozen-feature uPU, save/load)
```

## 4.1 Benchmarks (`benchmarks/`)

```text
benchmarks/
  _common.py                   (shared primitives including canonical_hash)
  __init__.py
  assigned_methods/            (first-five-paper and executed SCAR/SAR benchmarks)
    __init__.py
    runner.py                  (shared JSON benchmark runner)
    run.py                     (CLI)
    README.md
    preflight_paper.py         (separate source/data/historical-environment/toolbox gap audits)
    pusb_official_data.py      (IJCNN1 repository-extension checks, official sampling, kernel/uLSIF, provenance)
    pusb_table2_data.py        (locked six-dataset Table 2 loaders and official-sampling feasibility)
    pusb_table2_benchmark.py   (strict/compatible Table 2 plans, checkpoints/resume, provenance)
    pusb_table2_parallel.py    (bounded concurrency, retries, automatic aggregation)
    pusb_table2_aggregate.py   (cross-shard key/configuration/provenance acceptance)
    pusb_table2_report.py      (strict acceptance, paired confidence intervals, Markdown/JSON)
    configs/official/          (five locked_not_executed paper-like configurations)
    results/clean_room_multiseed/ (actual 3-method × 3-seed artifacts)
    results/official_preflight/ (first-five-paper blockers for the audited node)
    results/pusb_official_data_smoke/ (reduced-grid official-data smoke)
    results/pusb_official_data_feasible_multiseed/ (IJCNN1 repository extension: full 3-seed × 3-U grid + uLSIF)
    results/pusb_table2_data_audit/ (data integrity and official-sampling feasibility)
    results/pusb_table2_strict_plan/ (claim-safe 45-cell/4500-trial dry-run plan)
    results/pusb_table2_strict_full/ (4500-trial aggregates, statistics, claim-boundary report)
    results/scar_sar_comparison/ (3-mechanism × 10-seed SCAR/SAR comparison)
  deep_pu/
    runner.py                  (shared InfoMax PU/WConPU/DGPU runner)
    run.py                     (CLI)
    official_data.py           (public data, deterministic PU splits, clean-validation/grid selection, resume, provenance)
    run_official_data.py       (official-data CLI)
    preflight_paper.py         (GPU/EDM/licensed-data/implementation gap audits)
    configs/
      clean_room_multiseed.json
      official_data_smoke_fashion_mnist.json
      official_data_infomax_fashion_protocol_pi05.json
      official_data_wconpu_cifar10_protocol.json
      official_sources.lock.json
      official/               (three paper-like configurations; consult their evidence for execution status and InfoMax provisional choices)
    results/clean_room_multiseed/ (actual 3-method × 3-seed artifacts)
    results/official_data_smoke_fashion_mnist/ (three-seed real-data smoke)
    results/infomax_fashion_protocol_preflight/ (full-protocol Fashion-MNIST preflight)
    results/official_preflight/ (full-configuration blockers for the audited node)
    results/wconpu_cifar10_protocol_preflight/ (CIFAR-10 preflight)
  joint_shift/
    __init__.py
    runner.py                  (public Wisconsin multiseed/CI joint-shift smoke)
    README.md                  (protocol, commands, artifacts, claim boundaries)
  traditional_pu/
    __init__.py
    data.py                    (SCAR/SAR/PNU synthetic data, h forwarding, degeneracy checks)
    leakage_audit.py           (feature blacklist, duplicates, y_true path guards, trial-column gates, preflight reports)
    statistics.py              (means, confidence intervals, paired differences, success rates)
    runner.py                  (SCAR main grid, SAR diagnostics, PNU triplet grid, state machine/resume/timeouts)
    run.py                     (CLI: --seed-set/--results-dir/--timeout-profile)
    tuning_round.py            (candidate generation, §4 filtering/ranking, §5 degeneration checks)
    README.md                  (protocol, commands, timeout freezing, claim boundaries)
    configs/
      seven_methods_pu_baseline_v1.json
      pnu_baseline_v1.json
      pnu_baseline_v2.json      (explicit locked-default PNU baseline and tuning companion source)
      seven_methods_pu_baseline_v2.json (historical contract-v1 snapshot; do not rerun with current source)
      seven_methods_pu_baseline_v3.json (formal contract-v2 baseline; risk uses native predict threshold)
      kldce_tuning_r1/         (10 first-round candidates generated by tuning_round)
      kldce_tuning_r3/         (first-round rerun after b₀ class-symmetry fix with the same parameter table)
      ldce_tuning_r1/          (10 first-round candidates generated by tuning_round)
      nnpu_tuning_r1/          (10 first-round candidates generated by tuning_round)
      upu_tuning_r1/           (10 first-round candidates generated by tuning_round)
      llsvm_tuning_r1/         (10 first-round candidates generated by tuning_round)
      elkan_noto_tuning_r1/    (10 first-round candidates generated by tuning_round)
      elkan_noto_tuning_r2/    (first-round rerun under contract v2 with the same parameter table)
      pnu_tuning_r1/           (10 first-round candidates generated by tuning_round)
    results/                   (four standard artifacts + report.md under <run-name>/)
```

Official-data runs with clean-validation selection also produce `model_selection.csv`, recording
each candidate, validation metric, score, and elapsed time per seed. `trials.csv` records the
selected parameters and refit results.

## 5. Documentation (`docs/`)

> **Generator scope:** `generate_structure.py --check` compares the **file-name set** in both
> directions: repository files absent from the block are missing; listed files absent from disk
> are stale.
>
> - **Suffixes:** `docs/` covers `.md`, `.png`, and `.json` documentation, figures, and data
>   manifests. Other suffixes, such as `.csv` or `.ipynb`, need not be listed and are outside
>   verification. If listed, they produce an out-of-scope-suffix problem distinct from a missing
>   path. `--update` reports that problem as well as removing the unsupported entry.
> - **Grouped subtrees:** `adr/` and `research/pu_survey/` are registered by group through
>   [adr/README.md](../adr/README.md) and [pu_survey/README.md](../research/pu_survey/README.md).
>   Unlisted files inside these subtrees are intentionally exempt from missing-file checks;
>   this includes the two JSON files in `research/pu_survey/data/`. Explicitly listed entries,
>   including `pu_survey_protocol.md` and the two asset images, still require valid paths.
>   Other documentation paths are listed individually. Each registered subtree must retain its
>   directory line: `--check` names a missing registration anchor, and `--update` restores it
>   as an unannotated directory rather than silently dropping the subtree.
> - **Limits:** the generator checks names, never annotation content. It reads and preserves
>   annotations rather than deriving them from files. A valid path with an obsolete description
>   therefore passes path checks; human review must update descriptions when meaning changes.
>
> See [docs/README.md](../README.md) for the complete documentation index.

```text
docs/
  adr/                                    # architecture/process decisions: index and numbered ADRs
  user/                                   # user journey: quickstart → concepts → how-tos → reference
    concepts/
      pu_problem.md                       # problem setting, notation, role of π
      scar_sar.md                         # SCAR/SAR mechanisms and identification limits
      method_selection.md                 # method-choice principles, recommender, decision table
      class_prior_semantics.md            # population/U/risk-pool priors, conversion conditions, implementation limits
      pu_model_selection.md               # PA/OA, supervision roles, independent selection
      pu_sampling.md                      # OS/TS, finite-pool provenance, training risk views
      representations.md                  # comparison conditions for encoders, native CNNs, frozen features
    howto/
      pipeline.md                         # end-to-end PUPipeline workflow
      cli.md                              # command-line interface
      data_profiling.md                   # data profiles and assumption hints
      diagnostic_reports.md               # diagnostic report generation
      sensitivity_analysis.md             # class-prior/labeling-propensity sensitivity
      sar_simulation.md                   # SCAR/SAR simulation
      self_pu.md                          # Self-PU training
      distribution_shift.md               # shift audits and covariate-weighted adaptation
      model_tuning.md                     # model parameters and PU-aware grid search
      custom_experiments.md               # four-role experiment API and strategy extension
      custom_encoders.md                  # encoder injection, shared backbones, frozen features
      ui.md                               # UI installation and usage
      using_skill.md                      # enabling and using the pu-workflow skill
      pu_sampling.md                      # same-pool quotas, independent TS, metadata, entry-point migration
    reference/
      api/
        index.md                          # API navigation, version, and registry index
        classifiers.md                    # classifier and class-prior estimator contracts
        tools.md                          # pipeline, utilities, and experiment contracts
        errors.md                         # exception types and usage scenarios
    README.md                             # user journey map
    quickstart.md                         # five-minute quickstart
  dev/                                    # contributor guides and optional internal background
    architecture.md                       # current layers, dependencies, data flow, registry
    project_structure.md                  # authoritative directory inventory (this document)
    new_algorithm_template.md             # algorithm integration, capabilities, tests
    dual_architecture_plan.md             # dual-architecture upgrade phases/results (Chinese)
    label_semantics_plan.md               # label_semantics capability contract/checkpoints (Chinese)
    experiment_layer.md                   # injectable strategies and view/trajectory design (Chinese)
    compatibility.md                      # Python/dependency support, CI responsibilities, build policy
    architecture_principles.md            # architecture maintenance signals and audit history (Chinese)
    comment_governance.md                 # comment criteria, gates, accepted exceptions, migration
    single_source_map.md                  # sources, consumers, duplication criteria (Chinese)
    data_leakage_audit_design.md          # blacklist/duplicate/guard audit design (Chinese)
    distribution_shift_aware_pu.md        # OOF/covariate-weighting design (Chinese)
    distribution_shift_aware_pu_checklist.md # shift implementation checklist (Chinese)
    process_checklist.md                  # authoritative progress/release status (Chinese)
    release_process.md                    # versioning, preflight, upload, rollback, maintenance (Chinese)
    pu_research_semantics.md              # sampling/prior/consumption baseline and phase exits (Chinese)
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
      GradPU.md
      Robust-PU.md
      Split-PU.md
      LaGAM.md
      PUET.md
      PULDA.md
      VPU.md
      PUSB.md
      LBE.md
      Self-PU.md
      InfoMax-PU.md
      WConPU.md
      DGPU.md
      Importance_Weighted_PU_Shift.md
      CVIR.md                             # fixed-unlabeled-prior classical adapter; technical preintegration
      GEN-PU.md                           # three-discriminator/two-generator sources, views, adaptation limits
      Holistic-PU.md                      # trend grouping and pseudo-PN adaptation limits
      P3MIX.md                            # paper mapping, full estimator, admission boundaries
      PAN.md                              # dual-network risk and CNN integration boundaries
      PULNS.md                            # selector and independent clean-support label budget
      RP.md                               # PU-only rho0=0 sources and OOF calibration
      BBE.md                              # source differences, standalone α_U, holdout/sampling limits
      TEDn.md                             # alternating training order, source differences, independent-TS engineering limits
    traditional_pu/
      traditional_pu_metric_contract.md   # seven-method single-domain metrics, baselines, statistics
      traditional_pu_optimization_plan.md # tuning order, parameter groups, advancement rules, artifacts
    distribution_shift/
      distribution_shift_metric_contract.md # shift audits, cross-domain evaluation, deployment metrics
      joint_shift_research_protocol.md    # research evaluation and claim boundaries
      joint_shift_metadata_review_20261019.md # issue #69 domain-shift/PU-sampling metadata decisions
    pu_survey/
      assets/
        Dataset.png                       # eight-dataset class labels/indices (Table B.1)
        Dateset_PU.png                    # PU mappings/input sizes/counts (Table B.2)
      pu_survey_protocol.md               # eight-dataset protocol and independent PA/OA selection
  README.md                               # user/contributor/internal-background navigation
```

## 6. Scripts (`scripts/`)

The table groups scripts by purpose without moving them. Existing queues and documentation
depend on fixed paths; age or the presence of another runner does not establish obsolescence.
For commands, see the
[Survey runbook](../research/pu_survey/runbook.md)。

| Purpose | Common entry points (all under `scripts/`) |
|---|---|
| Data collection / splits | `collect_survey_public_data.py`, `prepare_survey_splits.py`, `survey_splits_archive.py`, `prepare_clean_support.py` |
| Single runs / pilot | `run_survey_experiment.py`, `run_survey_pilot.py`, `run_clean_support_experiment.py`, `run_genpu_recipe.py` |
| Background queues / dashboards | `run_survey_parallel_queue.py`, `run_survey_backlog_campaign.py`, `run_survey_queue_fit.py`, `monitor_survey_queue.py`, `monitor_survey_campaign.py` |
| Candidate paths / resource planning | `prepare_candidate_image_resource_plan.py`, `prepare_kldce_resource_plan.py`, `run_candidate_image_experiment.py`, `run_classical_candidate_experiment.py`, `run_candidate_readiness_campaign.py`, `run_candidate_five_seed_grid.py` |
| Audit / aggregation / delivery | `audit_survey_batches.py`, `audit_survey_queue.py`, `aggregate_survey_runs.py`, `summarize_survey_results.py`, `export_survey_results.py`, `export_survey_queue_handoff.py`, `compare_survey_results.py` |
| Diagnostics / rerun preparation | `run_lbe_comparison.py`, `run_lbe_diagnostics.py`, `diagnose_gradpu_encoder.py`, `prepare_survey_support_repair.py`, `prepare_kldce_recovery.py`, `recover_survey_backlog_canary.py` |
| Versioned protocols / recipes | `prepare_survey_recipe_registry.py`, `prepare_survey_backlog_revision.py`, `prepare_survey_classical_image_revision.py`; do not regenerate frozen assets merely to tidy directories. |
| Quality gates / structure generation | `check_*.py`, `generate_structure.py`, `generate_layer_deps.py` |
| Thin workflow wrappers / internal helpers | `pu_workflow/`, `_source_provenance.py`, `_survey_host.py`, `_lbe_diagnostic_reference.py`; underscore helpers are not general training entry points. |

```text
scripts/
  pu_workflow/                            (compatible thin skill wrappers delegating to CLI subcommands)
    profile.py                            (delegates to `pu-toolbox profile`: profile.json contract)
    recommend.py                          (delegates to `pu-toolbox recommend`: recommendation.json contract)
    sensitivity.py                        (delegates to `pu-toolbox sensitivity`: sensitivity.json contract)
  check_test_quality.py                   (test count/marker/coverage quality gate)
  check_doc_links.py                      (five documentation consistency rules)
  check_project_metadata.py               (Python/CI/extras/Hatchling cross-file consistency)
  check_math_rendering.py                 (method-card MathJax: missing sub/superscript arguments, brackets, dollar delimiters)
  check_skill_sync.py                     (complete mirrored resource trees, internal links, portable metadata)
  check_baseline_configs.py               (locked configurations vs constructor defaults)
  check_format.py                         (ruff check + format --check across the configured scope)
  check_api_docs.py                       (API reference public-symbol coverage: static __all__ extraction and registered names)
  generate_structure.py                   (--check validates / --update regenerates project_structure.md tree blocks)
  prepare_survey_splits.py                (data/raw → four-role train/pu_val/clean_val/test.npz and manifests)
  run_survey_experiment.py                (official example from protocol §2.4 item 9: single-method four-role PU, PA/OA, independent test)
  run_survey_pilot.py                     (batches, manifest completion, checkpoint budgets, --dry-run, --datasets/--methods/--training-paths subsets, --plan-json; shared --protocol for planning/execution)
  survey_splits_archive.py                (deterministic tar/index pack and bidirectional receiver verify)
  summarize_survey_results.py             # comparable-row means/sample SD; per-run costs avoid PA/OA double counting; formal/partial/diagnostic tables
  export_survey_results.py                # batch config/summary.json → survey_results.xlsx and diagnostic/conclusion md/json; no numeric recomputation
  check_survey_recipe_registry.py         # P4.1 registry gate: explicit skip when absent, mandatory checks when materialized
  check_comment_quality.py                (legacy markers, contextual TBDs, trailing comments; defaults to pu_toolbox/)
  generate_layer_deps.py                  # architecture.md §2.1 dependency-table generation
  prepare_survey_recipe_registry.py       (read-only P4.1 registry draft generation from frozen pilot protocol)
  collect_survey_public_data.py           (collect/verify missing public data; does not download official ADNI data)
  run_lbe_comparison.py                   # independent four-role paired linear/neural LBE exploration
  run_lbe_diagnostics.py                  # extended EM and fixed-SCAR-propensity diagnostics
  _lbe_diagnostic_reference.py            # unregistered SCAR-constrained analytical-gradient reference
  _source_provenance.py                   # separates exploratory-script repository identity from loaded source identity
  prepare_clean_support.py                # explicitly budgeted raw-data support artifacts; preview by default
  run_clean_support_experiment.py         # independent-support CPU single-candidate OA exploration/replay for three methods
  run_genpu_recipe.py                     # explicit single-candidate four-role engineering command and PA/OA manifests
  run_candidate_image_experiment.py       # PULDA/GradPU single-candidate four-role images and resource replay
  run_classical_candidate_experiment.py   independent CPU PUET/RP four-role planning/execution
```

## 7. CI/CD (`.github/`)

```text
.github/
  pull_request_template.md
  workflows/
    tests.yml                (PR fast layer: Python-matrix unit/integration, static gates, installed-wheel smoke)
    nightly.yml              (Monday 03:23 UTC: full slow/e2e top layer, 3 × 3 matrix)
```
