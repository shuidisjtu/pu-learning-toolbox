# PU Learning Toolbox documentation

The index, external contributor guides, all user documentation, and public PU Survey
reproduction documents are in English. Other internal research and maintenance records may
continue in Chinese under the
[language policy](adr/0022-public-documentation-language.md).

## User documentation (docs/user/)

For library users, including developers integrating the public API into other repositories.
Concepts explain the research model, how-to guides cover tasks, and reference pages define
contracts. Start with the [quickstart](user/quickstart.md), concepts, and how-to guides, then
consult the [API index](user/reference/index.md) for exact contracts.

| Category | Document | Purpose |
|---|---|---|
| Entry | [user/README.md](user/README.md) | Reading order and task navigation |
| Quickstart | [user/quickstart.md](user/quickstart.md) | Installation, three CLI commands, and a minimal Python example |
| Concept | [user/concepts/pu_problem.md](user/concepts/pu_problem.md) | PU problem settings, notation, and the roles of π |
| Concept | [user/concepts/pu_sampling.md](user/concepts/pu_sampling.md) | OS/TS, finite-pool provenance, and training risk roles |
| Concept | [user/concepts/scar_sar.md](user/concepts/scar_sar.md) | SCAR/SAR mechanisms and identifiability limits |
| Concept | [user/concepts/class_prior_semantics.md](user/concepts/class_prior_semantics.md) | Population, U, risk-pool, and training-table priors; conversion conditions and limits |
| Concept | [user/concepts/pu_model_selection.md](user/concepts/pu_model_selection.md) | PA/OA, supervision budgets, and independent selection |
| Concept | [user/concepts/representations.md](user/concepts/representations.md) | Encoders, end-to-end CNNs, frozen features, and comparison conditions |
| Concept | [user/concepts/method_selection.md](user/concepts/method_selection.md) | Recommender logic and method selection tables |
| How-to | [user/howto/pipeline.md](user/howto/pipeline.md) | End-to-end training and evaluation with PUPipeline |
| How-to | [user/howto/cli.md](user/howto/cli.md) | Command-line interface |
| How-to | [user/howto/model_tuning.md](user/howto/model_tuning.md) | Model parameters and PU-aware grid search |
| How-to | [user/howto/custom_experiments.md](user/howto/custom_experiments.md) | Four-role experiment API, parameters, and strategy injection |
| How-to | [user/howto/custom_encoders.md](user/howto/custom_encoders.md) | Encoder injection, shared backbones, frozen features, and persistence |
| How-to | [user/howto/ui.md](user/howto/ui.md) | Graphical interface installation and use |
| How-to | [user/howto/data_profiling.md](user/howto/data_profiling.md) | Data profiling and assumption warnings |
| How-to | [user/howto/pu_sampling.md](user/howto/pu_sampling.md) | Same-pool fixed-count labeling, independent TS, metadata, and legacy API migration |
| How-to | [user/howto/diagnostic_reports.md](user/howto/diagnostic_reports.md) | Diagnostic report generation |
| How-to | [user/howto/sensitivity_analysis.md](user/howto/sensitivity_analysis.md) | Class-prior and propensity sensitivity |
| How-to | [user/howto/distribution_shift.md](user/howto/distribution_shift.md) | Distribution shift audits and covariate-weighted adaptation |
| How-to | [user/howto/sar_simulation.md](user/howto/sar_simulation.md) | SCAR/SAR data generation |
| How-to | [user/howto/self_pu.md](user/howto/self_pu.md) | Training Self-PU |
| How-to | [user/howto/using_skill.md](user/howto/using_skill.md) | Agent task routing, prior resolution, experiments, representations, and skill installation |
| Reference | [user/reference/index.md](user/reference/index.md) | Symbols grouped by module, linking to API contracts |
| Reference | [user/reference/api.md](user/reference/api.md) | Authoritative signatures, parameters, return structures, and minimal examples; coverage checked by `check_api_docs` |

## External contributor guides (docs/dev/)

Start with [CONTRIBUTING.md](../CONTRIBUTING.md) for environment setup, branches, tests, shared
helpers, and PR requirements. These five English guides contain the required extension and
architecture rules; ordinary contributions do not require reading Chinese internal plans.

| Document | Purpose |
|---|---|
| [dev/architecture.md](dev/architecture.md) | Module layers, dependency directions, data flow, and registry rules |
| [dev/project_structure.md](dev/project_structure.md) | Authoritative file layout and local asset boundaries |
| [dev/compatibility.md](dev/compatibility.md) | Python/dependency support, CI responsibilities, and packaging |
| [dev/new_algorithm_template.md](dev/new_algorithm_template.md) | Capability declarations, registration gates, and minimum CNN tests |
| [dev/comment_governance.md](dev/comment_governance.md) | Comment criteria, automated checks, retained cases, and migration policy |

## Internal maintenance and research (Chinese)

These records provide optional design and audit background. Maintainers own project status and
core architectural decisions. Required contribution rules are in the English guides above.

| Document | Purpose |
|---|---|
| [dev/architecture_principles.md](dev/architecture_principles.md) | Architecture maintenance, decay signals, and governance history |
| [dev/process_checklist.md](dev/process_checklist.md) | Authoritative task and release status |
| [dev/distribution_shift_aware_pu.md](dev/distribution_shift_aware_pu.md) | Shift-aware PU assumptions, implementation boundaries, and acceptance criteria |
| [dev/distribution_shift_aware_pu_checklist.md](dev/distribution_shift_aware_pu_checklist.md) | Shift feature tasks and evidence |
| [dev/release_process.md](dev/release_process.md) | Versioning, preflight checks, publishing, rollback, and ownership |
| [dev/data_leakage_audit_design.md](dev/data_leakage_audit_design.md) | Leakage checks, split isolation, and negative tests |
| [dev/dual_architecture_plan.md](dev/dual_architecture_plan.md) | MLP/CNN capability design and phased implementation |
| [dev/label_semantics_plan.md](dev/label_semantics_plan.md) | Label declarations, checkpoints, value-level ambiguity, and leakage guards |
| [dev/experiment_layer.md](dev/experiment_layer.md) | Injectable experiment strategies, view/trace semantics, and known limits |
| [dev/single_source_map.md](dev/single_source_map.md) | Shared concepts, authoritative implementations, consumers, and duplication audit |
| [dev/pu_research_semantics.md](dev/pu_research_semantics.md) | Sampling, prior, per-method risk consumption, reproduction, and integration baseline |
| [research/method_cards/](research/method_cards/) | Paper formulas, implementation/reproduction boundaries, and research solvers |
| [research/traditional_pu/](research/traditional_pu/) | Traditional PU metric/baseline contracts, tuning, promotion rules, and artifacts |
| [research/distribution_shift/](research/distribution_shift/) | Shift metrics, joint-shift protocol and claim boundaries, and metadata review |

## PU Survey reproduction

Start at
[research/pu_survey/README.md](research/pu_survey/README.md), then read the
[protocol](research/pu_survey/pu_survey_protocol.md), [runbook](research/pu_survey/runbook.md), and
[data sources](research/pu_survey/data/README.md). These describe execution capabilities, data
roles, versioned artifacts, reporting, provenance, and scientific limits. Implementation capability,
completed training, and verified paper numerical reproduction are separate claims.

## Architecture decisions (docs/adr/)

The [ADR archive](adr/) and its [index](adr/README.md) are primarily in Chinese. The English
[current language policy](adr/0022-public-documentation-language.md) supersedes the historical
language requirement. ADRs record decisions; version and task status belong to the internal
maintenance records above.

## Other entry points

- [Project README](../README.md) (English).
- [Contribution guide](../CONTRIBUTING.md) (English).
- [Minimal runnable examples](../examples/minimal/) (Python).
- [Deep PU benchmarks](../benchmarks/deep_pu/README.md) (Chinese): InfoMax PU, WConPU,
  DGPU runners, and multi-seed results.
