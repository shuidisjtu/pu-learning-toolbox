# Architecture

## 1. Design decisions

Design decisions and tradeoffs are recorded in [docs/adr/](../adr/README.md). This
document describes the current architecture.

**Relationship to `project_structure.md`**: this page explains module boundaries,
dependency directions, data flow, recommendation, and registration conventions.
[`project_structure.md`](project_structure.md) is authoritative for file inventories
and directory structure. **API contracts**: exact signatures and method semantics
belong in [the API reference](../user/reference/api.md), rather than being duplicated here.

**Research-setting semantics**: input sampling, labeling/selection policies, risk
roles, prior targets, validation supervision, and representations are separate
dimensions. Core owns role facts; estimators and the registry declare capabilities;
the experiment layer owns research policies. Start with [the PU problem
setting](../user/concepts/pu_problem.md). The [semantic baseline](pu_research_semantics.md)
and [ADR-0021](../adr/0021-pu-research-semantics.md) provide optional Chinese
implementation history and decision rationale.

`prior/semantics.py` supplies OS/TS probability-mass conversions. The pipeline resolves
an estimate to a population prior, then converts it per fold to the estimator's
default view target; `PriorInfo` records the values actually consumed. The experiment
layer's CVIR adapter reuses these conversions while retaining its research risk
views and parameter-binding policy. `preprocessing/_sampling.py` centralizes fixed
positive quotas and uniform sampling without replacement for labeling entry points
and experiment generators.

## 2. Module layers

Layers are conceptual groups. The module column identifies `pu_toolbox/` subpackages
and their responsibilities. File-level responsibilities and public entry points
are documented alongside the directory trees in
[`project_structure.md`](project_structure.md).

### Core — foundations used by the other layers

| Module | Responsibility | Details |
|---|---|---|
| `core/` | PU base classes, label semantics, input validation/normalization, device and random-state handling, exceptions, tags, and **neutral training-role views** (`core/training_views.py`) | [Exports](../../pu_toolbox/core/__init__.py) |
| `preprocessing/` | SCAR/SAR labeling and data generation; structured data profiling | [Exports](../../pu_toolbox/preprocessing/__init__.py), [profiling guide](../user/howto/data_profiling.md) |
| `registry/` | Algorithm registration, metadata, and built-in discovery | [Exports](../../pu_toolbox/registry/__init__.py), [built-in methods](../../pu_toolbox/registry/builtin_methods.py) |
| `advisor/` | Algorithm recommendations based on data profiles | [Exports](../../pu_toolbox/advisor/__init__.py), [method selection](../user/concepts/method_selection.md) |
| `utils/` | Shared utilities; private API that may change in minor releases | [Exports](../../pu_toolbox/utils/__init__.py) |

### Estimation — class priors and PU losses used by Algorithms

| Module | Responsibility | Details |
|---|---|---|
| `prior/` | Class-prior estimators (ReCPE / penL1 / KM) | [Exports](../../pu_toolbox/prior/__init__.py), [prior method card (Chinese)](../research/method_cards/class_prior_estimation.md) |
| `losses/` | PU risk/loss functions | [Exports](../../pu_toolbox/losses/__init__.py) |

### Algorithms — PU classifier implementations

| Module | Responsibility | Details |
|---|---|---|
| `estimators/` | PU classifiers grouped into `classic/`, `risk/`, `bias_aware/`, `deep/`, and `research/` | [Exports](../../pu_toolbox/estimators/__init__.py) |

The registry's [built-in table](../../pu_toolbox/registry/builtin_methods.py)
(`_BUILTINS`) is authoritative for algorithm-to-module mappings and implementation
status (NATIVE / api_only / `official_exact` adapter). [Method cards
(Chinese)](../research/method_cards/) explain individual algorithms.

### Evaluation — evaluation and diagnostics used by Orchestration

| Module | Responsibility | Details |
|---|---|---|
| `metrics/` | PU-only metrics and wrappers for standard supervised metrics | [Exports](../../pu_toolbox/metrics/__init__.py), [metric contract (Chinese)](../research/traditional_pu/traditional_pu_metric_contract.md) |
| `model_selection/` | PU-stratified splitting, tuning, and comparison | [Exports](../../pu_toolbox/model_selection/__init__.py), [tuning guide](../user/howto/model_tuning.md) |
| `diagnostics/` | Structured diagnostic reports and assumption sensitivity analysis | [Exports](../../pu_toolbox/diagnostics/__init__.py), [report guide](../user/howto/diagnostic_reports.md) |

### Orchestration — end-to-end workflows and CLI

| Module | Responsibility | Details |
|---|---|---|
| `workflows/` | PUPipeline orchestration and shift-aware workflows | [Exports](../../pu_toolbox/workflows/__init__.py), [pipeline guide](../user/howto/pipeline.md), [shift guide](../user/howto/distribution_shift.md) |
| `cli/` | Thin command-line wrappers | [CLI guide](../user/howto/cli.md), [exports](../../pu_toolbox/cli/__init__.py) |

### Experiment — experiment and research protocols

| Module | Responsibility | Details |
|---|---|---|
| `experiment/` | Protocol-based research orchestration: four data roles (`train/pu_val/clean_val/test`), independent PA/OA offline selection, independent test evaluation, and versioned provenance. `ExperimentRunner` provides a fixed skeleton with injectable generation, training, and selection strategies (`Generator`/`Trainer`/`SelectionProtocol`). | [Exports](../../pu_toolbox/experiment/__init__.py), [design background (Chinese)](experiment_layer.md), [ADR-0018 (Chinese)](../adr/0018-experiment-layer-public-api.md) |

### User Layer — tutorials, UI, reproduction tools, and agent wrappers

| Module | Responsibility | Details |
|---|---|---|
| `examples/` | Tutorials and minimal examples | [Examples](../../examples/) |
| `ui/` | Optional graphical interface; core installation does not import Streamlit | [Exports](../../pu_toolbox/ui/__init__.py), [UI guide](../user/howto/ui.md) |
| `benchmarks/` | Paper-reproduction runners, official-data integration, and artifact handling at the repository root; operators supply official data | [Benchmarks](../../benchmarks/) |
| `scripts/pu_workflow/` | Compatibility wrappers delegating to CLI subcommands | [Wrappers](../../scripts/pu_workflow/) |
| pu-workflow skill | Routes agent tasks to CLI/public Python APIs and loads analysis, sampling/prior, research, representation, and extension references as needed | [Skill entry](../../.claude/skills/pu-workflow/SKILL.md) |

### 2.1 Module dependencies

<!-- BEGIN: layer-deps (generated by scripts/generate_layer_deps.py; do not edit by hand) -->
**Cross-layer dependencies** (targets are dependencies; generated from code, do not edit):

| Layer | Uses/calls |
|---|---|
| Core (`core/`, `preprocessing/`, `registry/`, `advisor/`, `utils/`) | — |
| Estimation (`prior/`, `losses/`) | Core |
| Algorithms (`estimators/`) | Core, Estimation |
| Evaluation (`metrics/`, `model_selection/`, `diagnostics/`) | Core, Estimation (function-local only), Orchestration |
| Orchestration (`workflows/`, `cli/`) | Core, Estimation, Algorithms, Evaluation |
| Experiment (`experiment/`) | Core, Estimation, Algorithms (function-local only), Evaluation (function-local only) |
| User Layer (`ui/`, `benchmarks/`) | Core, Estimation, Algorithms, Evaluation, Orchestration |
<!-- END: layer-deps -->

The table is generated by `scripts/generate_layer_deps.py` using an AST scan of all
imports, **including function-local imports**. Edges marked “function-local only”
are absent from a module-load import graph. `--check` fails if the generated table
differs from the source. Its scope is **cross-layer edges**; dependencies and cycles
within a layer are described below.

**Dependencies and cycles within layers (maintained manually)**:
within Evaluation, `diagnostics/report.py` and `diagnostics/sensitivity.py` depend on
`metrics`, and `diagnostics/shift_monitor.py` depends on `diagnostics.shift`.
There are two within-layer cycles that involve function-local imports and therefore
do not appear in a module-load graph: `estimators/risk/kldce.py` ↔ `kldce_smo.py`,
and `experiment/survey_execution.py` ↔ `survey_protocol.py`. There are **no
cross-layer import cycles**. At the aggregated layer level, Evaluation and
Orchestration point to each other; Evaluation → Orchestration is the approved
upward exception described below, while the reverse direction is normal.

**Top-level files outside the layer map**:
`pu_toolbox/__init__.py` (the umbrella re-export facade), `pu_toolbox/run_config.py`,
and `pu_toolbox/progress.py` are excluded from the table. `run_config.py` does
depend upward on `estimators.deep.vision` and `workflows`; it is an unlayered
top-level convenience module. Although `benchmarks/` is outside the Python package,
it belongs to User Layer and is included in the scan.

**Leaf entry point and guards (maintained manually)**:
no layer depends on Experiment. `tests/contract/test_layer_boundaries.py` guards
this direction using both static AST scans of all lower-layer modules and
subprocess import graphs. The former catches function-local imports; the latter
catches transitive imports. Their individual blind spots are documented in the test.

**Approved dependency exceptions**:

1. **`registry` → `estimators` / `prior` (Core → Algorithms / Estimation)**:
   `registry/builtin_methods.py` loads algorithm classes from **strings** with
   `importlib.import_module` and `getattr`. This edge is invisible to both the
   static AST scan and the subprocess import graph;
   `tests/contract/test_registry_dynamic_import.py` guards its meaning.
   Name-based lookup decouples the experiment layer from algorithm families:
   adding an algorithm requires registration.
2. **`model_selection` → `workflows` (Evaluation → Orchestration, upward)**:
   `model_selection/comparison.py` and `tuning.py` statically depend on
   `PUPipeline`. `model_selection/__init__.py` lazily loads these modules using
   PEP 562 module-level `__getattr__`. Tuning and comparison built on a PU-aware
   workflow remain Evaluation responsibilities; moving them would cost more than
   it would improve the boundary.

**Survey attribution in general-layer docstrings**:
some general-layer docstrings mention Survey context. These references do not
affect behavior: they do not parse ledger fields, use Survey artifact paths, or
hard-code repository paths. They are retained because they explain why the code
was written that way.

**Representative module dependency chains**:

- **Label semantics**: `core/labels.py` (label-convention recognition and remapping)
  → `core/validation.py` (label normalization, X/y consistency, sample-size checks
  and warnings, returning canonical inputs) → estimator `fit` entry points.
- **Main-input intent gate**: the runner calls `core/validation.py`'s
  `validate_label_semantics` for generated views; PU workflows call it for PU main
  inputs. Pipeline resolution, execution, and actual-clone checks cover comparison,
  tuning, CLI, and UI; CLI sensitivity checks its direct-fit path separately.
  This gate does not replace value validation or object-identity leakage guards.
  See [ADR-0020 (Chinese)](../adr/0020-main-input-label-semantics.md) for rationale.
- **Data profiling**: `preprocessing/profiling.py` supplies statistical primitives
  (`pu_data_summary`/`pnu_data_summary`/`scar_diagnostic`, kept backward compatible)
  → `preprocessing/data_profiler.py` aggregates `PUDataProfile` and actionable
  issues → `workflows` (first pipeline step), `diagnostics` (reports), and
  `advisor` (recommendations).
- **Metadata semantics**: `core/tags.py` is authoritative for registry metadata
  fields and enums; registry and advisor use those definitions.
- **Devices and randomness**: `core/device.py`'s `resolve_device` is the single
  device-normalization entry point. `core/random.py`'s `check_random_state`
  normalizes seeds for preprocessing and experiments. Preprocessing declares
  `int | RandomState | None` and returns supplied instances unchanged to continue
  their random stream. Experiments consume the helper but declare `seed` as
  `int | None` in `experiment/protocols.py`. Estimators intentionally do not use
  this helper: their `random_state` contract is `int | None`. The optional
  [single-source map (Chinese)](single_source_map.md) records the rationale.
- **Training views**: `core/training_views.py` constructs neutral
  `positive` / `native_unlabeled` / `loss_unlabeled` roles and source indices
  from one training partition, without ledger, method-name, or manifest knowledge.
  `experiment/training_views.py` owns Survey ledger gates, routing decisions,
  derivation of the historical `run_view` vocabulary at the boundary, and manifest
  construction; it delegates role construction to core. Estimator `os_or_ts`
  consumers depend on core. **`estimator → core` is allowed;
  `estimator → experiment` is forbidden.** Static dependence on Survey protocols
  would reverse the general-algorithm boundary. The reverse edge is absent in
  measurement; estimator modules seen when importing lower modules come from the
  umbrella re-exports in `pu_toolbox/__init__.py`, which runs before any
  `pu_toolbox.*` import. `tests/contract/test_layer_boundaries.py` ratchets this
  boundary with static AST and subprocess import checks. Role membership is a
  fact owned by core; whether calibration is allowed is an experiment-layer policy.
- **Experiment injection**: `experiment/runner.py` supplies the fixed skeleton,
  with an injected `model` instance and strategy ABCs
  (`Generator`/`Trainer`/`SelectionProtocol` in `protocols.py`). Replaceable
  strategies handle generation, training, selection, and provenance.
  `ExperimentRunner` imports neither estimators nor registry; it accepts an
  instance through `fit(model, …)`. The **caller** obtains a class with
  `registry.get_algorithm(method)`, instantiates it, and injects it. Adding a
  registered algorithm requires no algorithm-specific runner, script, or
  selection-chain changes.

The layer list is an overview; consult the annotated
[`project_structure.md`](project_structure.md) trees for file-level detail.

## 3. Data flow

The main chain is `PUPipeline.fit_evaluate`, from input data to reports.
Brackets identify modules using the layer terminology above.

```text
User input (X, y_pu[, y_true])
  ↓ Validation + label normalization                [core/validation + labels]
Data profile                                        [preprocessing/data_profiler]
  ↓ Profile issues include an error → fail fast
Method recommendation (default auto path)            [advisor/recommend_from_profile]
  ↓ Skip for explicit classifier; record reasons in provenance.classifier
Class-prior estimation (when needed)                 [prior/*]
  ↑ Only for requires_class_prior=True; skip methods that do not require π
    (e.g. PUSB/LBE/Elkan-Noto/InfoMax PU)
Training (including CV splitting)                    [estimators/* + model_selection/split]
  ↓ predict / decision_function / predict_proba
Evaluation + diagnostics → report                   [metrics + diagnostics → workflows/report]
```

- **Input contract**: `X` / `y_pu` are PU data prepared by the user. Researchers
  choose samples and decide which positives receive labels outside the toolbox;
  the toolbox treats them as supplied inputs. Simulators
  (`make_sar_*`/`make_scar_labels`/demos) support synthetic research and do not
  replace real-world sampling. Optional `y_true` is used only for supervised
  metrics and auditing, never for training.
- **Profiling semantics**: `PUDataProfile` contains basic statistics, feature
  quality, issue severity, recommended actions, and labeling-mechanism evidence.
  Without audit `y_true`, SCAR/SAR hints are explicitly non-identifying screening.
  With `y_true`, selection dependence is evaluated only within true positives,
  avoiding confusion between class separability and SAR.
- **Two-domain path**, separate from the single-domain entry point:
  `analyze_pu_shift` uses an out-of-fold domain classifier to estimate observable
  marginal shift and relative density ratios. Once coverage gates pass,
  `ShiftAwarePUPipeline` passes source-domain weights per fold to `PUPipeline`.
  This path is restricted to covariate shift; marginal weights are not described
  as joint `p_target(x,y)/p_source(x,y)` adaptation.
- **Report assembly**: `build_diagnostic_report` in `diagnostics` reads profiles,
  fitted estimators, and metric interfaces without training models. Stable-schema
  JSON/Markdown reports distinguish observed-PU metrics, class-prior-dependent
  metrics, supervised-oracle metrics, and unavailable metrics.

### 3.1 Experiment-layer data flow

The experiment layer is a second entry point for researchers, using explicit data
roles and selection protocols. It runs alongside the PUPipeline path and bypasses
advisor/prior recommendations.

```text
Four clean-view roles: train/pu_val/clean_val/test    [bundle/DatasetPart + core/validation]
  ↓ Validate disjoint indices and views; exclude test from selection
Generate PU views (SCAR/SAR/oracle as needed)         [strategies/Generator]
  ↓ Trainer/PA cannot receive true labels (view=="pu" validation)
Candidate training → RunTrajectory                  [Trainer → injected estimator]
  (Training views: native OS; native TS methods use TS-OS calibration, train only)
  ↓
Independent PA/OA offline selection                  [strategies/SelectionProtocol]
  (Per-checkpoint threshold grid; π is required for PA)
  ↓
Independent test evaluation + manifest               [runner → manifest]
```

- **View semantics**: all four input bundle parts, including `pu_val`, store
  **true labels** in clean views, enforced by `validate_bundle`.
  `Generator.generate` produces PU labels from true labels under SCAR/SAR before
  training and selection; those PU labels are not stored in advance. Generated
  outputs have `view=="pu"`, structurally preventing the Trainer/PA path from
  receiving true labels. The names `pu_val` / `clean_val` describe the views used
  **at selection time** (PU for PA, clean for OA), rather than stored labels.
- **Sampling views (OS/TS)**: the base data remain OS (Single-Training-Set, one
  i.i.d. sample). Native TS (Two-Sample / Case-Control) methods do not independently
  resample. During training they apply TS-OS calibration per mini-batch,
  `D_U^k ∪ D_P^k`: the positive batch joins the unlabeled loss input. This applies
  only to train; validation and test remain OS. See
  [protocol §2.3](../research/pu_survey/pu_survey_protocol.md#23-pu-数据集生成).
- **Selection**: PA and OA are independent. PA uses proxy accuracy's OS branch
  with required π; OA uses true Accuracy on `clean_val`. Both search checkpoints
  and a threshold grid, breaking ties in favor of the earliest candidate as
  defined by `selection_spec`.
- **Versioned provenance**: manifests preserve seed / split_ref / generation /
  selection / test_results / elapsed / failures / resources for auditable runs.

## 4. Algorithm registration and recommendation

Algorithms register metadata in `registry`: name/aliases/family/scenario/assumption/
requires_class_prior/requires_clean_support/backend/maturity/source_status/
implementation_status, plus architecture capabilities
`native_architectures` / `input_ndims` / `encoder_parameter` / `trains_encoder`.
`pu_toolbox/core/tags.py` is authoritative for field semantics and enums;
`pu_toolbox/registry/builtin_methods.py` records built-ins, module mappings,
and implementation status.

Estimator class attributes are authoritative for declared capabilities.
`_sync_class_metadata_to_registry` in `registry/registry.py` synchronizes the fields
listed in `_SYNC_FIELDS`, including architecture capabilities, entry metadata,
`label_semantics`, and prior-target declarations. Only declarations on the concrete
class or non-abstract ancestors count; defaults inherited from `BasePUClassifier`
or `BasePriorEstimator` do not count as authoritative declarations.

Decide what belongs in each built-in entry **field by field**, rather than by whether
a class is bound:

- **Do not repeat class-declared fields in entry literals.** Synchronization would
  overwrite them, making them dead writes. The guard is
  `tests/test_builtin_methods.py::test_static_entries_do_not_redeclare_class_fields`.
- **Explicitly supply entry fields that the class does not declare.** Their literals
  are the sole source; missing or incorrect values can silently fall back to
  dataclass defaults. This applies to the eight entry fields `family`,
  `assumption`, `scenario`, `requires_class_prior`, `implementation_status`,
  `source_status`, `backend`, and `maturity`.
- Binding is not a reason to omit all literals. For example,
  `class_prior_estimation` is bound but its class declares none of those eight
  fields, so all eight must be supplied. `pusb` and `lbe` likewise need explicit
  `requires_class_prior` literals when they only inherit the base default.
  Unbound `api_only` entries must supply all eight.
- Architecture capabilities and `label_semantics` come from class attributes or
  defaults rather than entry literals. The current `_SYNC_FIELDS` definition is
  authoritative for the complete synchronization surface.

Methods with `requires_clean_support=True` need an additional clean labeled support
set. The ordinary PU-only advisor excludes them. Explicit training calls must supply
support data with an independent label budget as specified for the method.

### Registry versus the experiment method ledger

`registry` is the **code-side source of truth**: `get_algorithm(name)` returns an
algorithm **class** for callers to instantiate; `get_metadata(name)` returns
metadata for capability/prior gates and advisor recommendations.

`pu_toolbox/experiment/method_ledger.json` is the **experiment-side source of truth
for result annotations**. It records paper sources, native sampling assumptions,
OS/TS calibration, adaptation levels, and review findings. Scripts use it to enforce
Survey scope for `--method` (fail loudly outside the ledger), and to copy the entry
into the run directory as `method_ledger_entry.json`. Overlapping fields
(the four capabilities, `source_status`, and
`requires_class_prior` ↔ `prior_semantics`) must mirror the registry, enforced by
`tests/contract/test_ledger_registry_consistency.py`.
See [protocol §4](../research/pu_survey/pu_survey_protocol.md#4-需实现的算法)
for semantics and [the ledger](../../pu_toolbox/experiment/method_ledger.json)
for current fields.

`advisor` matches profiles to registry metadata: hard filtering
(trainability, scenario, sparsity, available class prior) → soft scoring
(assumption match, maturity, trust, scale, cost, GPU, and label availability)
→ risk hints. Weights are configurable through `ScoringConfig`.
The [API reference](../user/reference/api.md) defines recommendation structures and
serialization; [method selection](../user/concepts/method_selection.md) explains
user-facing decisions.

### Boundary with model_selection

The advisor performs **metadata reasoning**: filtering, scoring, and ranking profiles
against registry metadata without training models. `model_selection` performs
**empirical selection** after a method is chosen: `PUTuner` searches hyperparameters,
`PUModelComparator` compares configurations, and `PUStratifiedKFold` provides CV
folds, with actual model training. The advisor answers which algorithm is suitable
in seconds without a GPU; model selection evaluates which configuration works best
on the data.

By default, `PUPipeline(classifier="auto")` consumes the advisor's choice and uses
`model_selection` splitting for training. Tuning is an explicit addition and does
not run automatically. See [method selection](../user/concepts/method_selection.md).

## 5. Evaluation and splitting

| Capability | Architectural contract | Details |
|---|---|---|
| PU-stratified splitting | `PUStratifiedKFold` / `PUStratifiedShuffleSplit` preserve P/U proportions and ensure each training fold contains labeled positives | [Tuning guide](../user/howto/model_tuning.md) |
| PU metrics | PU-only `pu_zero_one_risk`/`pu_recall`/`pu_estimated_precision`/`pu_negative_rate` do not require true labels; standard supervised wrappers (AUC/F1/Accuracy, etc.) require truth | [Metric contract (Chinese)](../research/traditional_pu/traditional_pu_metric_contract.md) |
| SCAR/SAR evidence | Non-identifying screening without audit truth; selection-dependence checks within true positives when `y_true` is supplied | [Profiling guide](../user/howto/data_profiling.md) |
| Structured reports | `build_diagnostic_report` reads profiles, fitted estimators, and metrics without training; outputs stable-schema JSON/Markdown | [Report guide](../user/howto/diagnostic_reports.md) |
| Assumption sensitivity | `analyze_pu_sensitivity` holds model outputs fixed and checks compatibility over a class-prior × mean-labeling-propensity grid; it does not identify propensity | [Sensitivity guide](../user/howto/sensitivity_analysis.md) |
| Selection-bias simulation | `make_sar_*` supports constant/linear/nonlinear propensity; `y_true/propensity` are hidden from user-facing learners and reserved for benchmarks | [SAR guide](../user/howto/sar_simulation.md) |

## 6. Paper-to-implementation index

See the [project structure](project_structure.md) trees for each method's module
location. [Method cards (Chinese)](../research/method_cards/) cover paper formulas,
source status, and reproduction risks, including their source-status/risk section.
