# PU Survey

The toolbox provides reusable experiment execution and result reporting. Read the
[protocol](pu_survey_protocol.md) for scientific settings, the [runbook](runbook.md) for the
repository's Survey scripts, and the [custom experiment guide](../../user/howto/custom_experiments.md)
for direct Python calls and strategy extensions.

## Deliverables and acceptance boundaries

**Result reporting**

- Generate a workbook from existing v3 manifests or a summary, with four result groups:
  SCAR–PA, SCAR–OA, SAR–OA (LBE-A/B separately), and PN oracle. Keep training paths, datasets,
  risk views, and support settings distinct. Label PA with extra supervision as
  `PA+clean-support`; do not produce an overall ranking across datasets.
- Report Accuracy means and sample standard deviations over five seeds; AUC, F1, Precision,
  and Recall are supplementary metrics. Show missing seeds, undefined metrics, failures,
  numerical risks, and exclusions without filling in zeros or inventing observations.
- Retain single-configuration cost, complete tuning cost, and peak GPU memory. PA/OA share
  the same fit-cost record to avoid double counting.
- Write `survey_results.xlsx`, `diagnostic_cards.md/json`, and `conclusions.md/json`.
  Distinguish literature facts, observations from these experiments, and inference; identify
  diagnostic signals whose required evidence is missing.

**Execution capabilities**

- `ExperimentRunner.fit(model, train, pu_val, clean_val, test)` isolates four data roles,
  independently selects PA/OA models, and evaluates on independent test data. See
  [four-role preparation](../../user/howto/custom_experiments.md#2-prepare-four-data-roles).
- Inject generators, trainers, and selection strategies. Custom studies do not need built-in
  Survey configuration; see the [complete example](../../user/howto/custom_experiments.md#1-run-the-complete-example).
- Survey entry points read caller-prepared splits and record candidate trajectories,
  selection artifacts, selected weights, and metrics.
- The batch entry plans datasets, methods, and training paths; resumption checks view,
  profile, recipe, and split identities. Commands are in the [runbook](runbook.md).
- PN oracle shares splits, representations, and budgets with the PU comparison and uses OA only.
- Record seeds, effective configurations, split/view/prior semantics, environment, and
  code/dependency identities throughout the data-to-report command chain.
- New runs save `research_settings`. Summaries retain per-seed sampling, labeling, risk-pool,
  prior, supervision, and representation settings and display them in workbooks and diagnostic cards.

## Configuration and historical results

New runs use `survey-v3.2`, revision `2026-11-08-clean-support-coverage`
([execution configuration](../../../pu_toolbox/experiment/survey_protocol_v3_2.json)), and
`survey-recipes-v1.3` ([recipes](../../../pu_toolbox/experiment/survey_recipe_registry_v1_3.json)).
Locked recipes and runnable flags describe engineering execution capability. Clean-support
configuration covers all eight datasets. IMDB/20News use frozen-split SBERT features; the
`adni` slot uses raw frozen-split images and records unverifiable subject isolation through
`group_policy="unverifiable"`.

Read existing v3.0/v3.1/v3.2 manifests through the read-only result pipeline, retaining original
versions, blockers, failures, and selection identities. Existing summaries can be exported
directly without old execution configurations, comparison matrices, or approval documents.

Method capabilities and sources are recorded in the [registry](../../../pu_toolbox/registry/builtin_methods.py).
[Method cards](../method_cards/) provide optional Chinese research background. The
[method ledger](../../../pu_toolbox/experiment/method_ledger.json) records view, calibration,
and prior semantics. Supply literature facts explicitly through `--literature` and native
OS/TS diagnostic groupings through `--native-views`; reports explain missing information.

## Interpretation and provenance

Execution capability, completed training, and verified numerical reproduction are distinct
claims. Training artifacts and results are maintained separately from this repository.
Preserve numerical risks and exclusions recorded in each manifest when interpreting results.

Queues, server schedules, handovers, repair tasks, and complete approval histories are not
part of the toolbox. Historical records remain available at pre-cleanup commit `7d11f2c4`;
saved manifests are not relabeled as new versions. Read the [data notes](data/README.md) for
sources, licensing qualifications, and the Alzheimer MRI substitute used in the `adni` slot.
