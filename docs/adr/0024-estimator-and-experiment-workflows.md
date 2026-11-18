# ADR 0024: Estimator and experiment workflows

- Status: Accepted
- Date: 2026-11-18

## Context

[ADR 0023](0023-explicit-method-selection.md) retired automatic model selection
but retained explicit single-table orchestration. That orchestration also decided
how to split, estimate and route priors, score candidates, and refit a winner.
Those choices depend on the sampling design, loss, and available supervision;
a generic replacement would bake in policies that may not fit the user's study.

## Decision

Retire `PUPipeline`, `PUTuner`, `PUModelComparator`, `ShiftAwarePUPipeline`, their
reports and run configuration, CLI `run` and `shift-run`, and Streamlit. Remove
them directly without an alias, deprecation period, or replacement single-table
candidate evaluator. Historical reports keep their original interpretation.

Keep estimator APIs, registry discovery, split and metric primitives, prior
estimation/conversion, diagnostics, and independent CLI tools. Callers fitting
existing PU observations choose their actual risk pool and prior and own their
validation design. Keep the existing four-role `ExperimentRunner` contract for
research; do not treat PU labels as clean labels or reuse test truth for selection.
This decision supersedes the retention of `PUPipeline` in ADRs
[0018](0018-experiment-layer-public-api.md), [0021](0021-pu-research-semantics.md),
and [0023](0023-explicit-method-selection.md), not their estimator, sampling, or
research contracts.

## Consequences

Training moves to explicit Python calls; profiling, sensitivity, drift auditing,
monitoring, and review remain independently usable. See the
[migration guide](../user/howto/pipeline.md) for prior routing and model artifacts.
An audit and its weights alone do not show that weighted training improves target
performance.
