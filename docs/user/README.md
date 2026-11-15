# User documentation

This directory is for library users, including developers calling the toolbox API from other
repositories. It organizes PU Learning Toolbox documentation by task: concepts explain the
principles, how-to guides cover operations, and reference pages define contracts.

The entry pages and all concepts below are in English. How-to guides and the API reference
currently remain in Chinese and will be migrated at the same paths in later batches.

## Reading path

1. **[Quickstart](quickstart.md)**: installation, the demo CLI workflow, and a minimal Python
   example. **Start here.**
2. **Concepts** (`concepts/`), when you need the underlying principles:

   | Topic | Document |
   |---|---|
   | PU problem, notation, and the roles of π | [PU problem setting](concepts/pu_problem.md) |
   | OS/TS, finite pools, native U, and the actual loss pool | [Input sampling and training risk views](concepts/pu_sampling.md) |
   | Labeling mechanisms and identifiability | [SCAR/SAR](concepts/scar_sar.md) |
   | Population, U, risk-pool, and training-table priors | [Class-prior targets, sources, and conversions](concepts/class_prior_semantics.md) |
   | PA/OA conditions, independent selection, and supervision budgets | [PU model selection](concepts/pu_model_selection.md) |
   | Encoders, end-to-end CNNs, and frozen features | [Representations and training paths](concepts/representations.md) |
   | Method selection principles | [Choosing a PU method](concepts/method_selection.md) |

3. **How-to guides** (`howto/`, currently Chinese), ordered by experiment workflow:

   | Task | Guide |
   |---|---|
   | Describe analysis and research tasks in natural language | [Using the pu-workflow skill](howto/using_skill.md) |
   | Generate SCAR/SAR data | [SAR simulation](howto/sar_simulation.md) |
   | Construct same-pool OS and independent TS data | [PU sampling](howto/pu_sampling.md) |
   | Profile data and inspect assumption evidence | [Data profiling](howto/data_profiling.md) |
   | Train and evaluate with PUPipeline | [Pipeline](howto/pipeline.md) |
   | Use the command-line interface | [CLI](howto/cli.md) |
   | Configure models and search hyperparameters | [Model tuning](howto/model_tuning.md) |
   | Customize four-role experiments and extend strategies | [Custom experiments](howto/custom_experiments.md) |
   | Inject encoders and use frozen features | [Custom encoders](howto/custom_encoders.md) |
   | Use the graphical interface | [UI](howto/ui.md) |
   | Generate diagnostic reports | [Diagnostic reports](howto/diagnostic_reports.md) |
   | Analyze class-prior and propensity sensitivity | [Sensitivity analysis](howto/sensitivity_analysis.md) |
   | Audit and handle source-to-target distribution shift | [Distribution shift](howto/distribution_shift.md) |
   | Train a Self-PU classifier | [Self-PU](howto/self_pu.md) |

4. **[API reference](reference/api.md)** (currently Chinese), for exact parameters and contracts.

Start with the quickstart, read concepts as needed, and then choose a how-to guide for your task.
Each how-to guide includes prerequisites and next steps.

## Existing PU data

Identify the [sampling design](concepts/pu_sampling.md), then check the
[labeling assumptions](concepts/scar_sar.md) and
[prior target](concepts/class_prior_semantics.md). Continue with
[data profiling](howto/data_profiling.md) and the [pipeline](howto/pipeline.md).

## Synthetic and comparative studies

Start with the [independent dimensions of a research setting](concepts/pu_problem.md#6-describe-research-settings-with-independent-dimensions).
Read sampling, labeling policies, priors, [model selection](concepts/pu_model_selection.md), and
[representations](concepts/representations.md), then run the
[four-role DIY example](howto/custom_experiments.md).

The [OS/TS how-to guide](howto/pu_sampling.md) and
[example 15](../../examples/minimal/15_pu_sampling.py) cover sampling operations.
[Example 16](../../examples/minimal/16_prior_targets.py) and the
[pipeline guide](howto/pipeline.md) trace prior values into training.
[Example 17](../../examples/minimal/17_sar_policies.py) compares SAR policies.
The [encoder guide](howto/custom_encoders.md) and
[example 18](../../examples/minimal/18_custom_encoders.py) cover encoder injection and frozen features.

For repository development and contributions, use the contributor section of the
[documentation index](../README.md).
