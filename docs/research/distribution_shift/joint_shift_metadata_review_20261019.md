# Joint-shift sampling metadata correction (#69, 2026-10-19)

## Finding

`JointShiftPUClassifier`, `DynamicJointShiftPUClassifier`, and `JointShiftPUBaseline`
declared `Assumption.SCAR` together with `Scenario.SELECTION_BIASED`. This conflated
between-domain joint-distribution shift with the PU label-selection mechanism.

The methods consume one pooled PU sample in each of the source and target domains.
Within each domain, labeled-positive selection is modeled as SCAR; the research
question is how to adapt across different source/target `P(X,Y)` distributions.
The [AISTATS 2025 paper](https://proceedings.mlr.press/v258/kumagai25a.html) likewise
describes using PU data from both training and test distributions without assuming
a particular shift type. Domain shift is not itself `Scenario.SELECTION_BIASED`.

## Correction

All three classes now declare:

- `assumption = (Assumption.SCAR,)`
- `scenario = (Scenario.SINGLE_TRAINING_SET,)`

The source/target domain-shift distinction remains expressed by the research
estimator inputs and this protocol, not by the PU sampling-scenario tag. No training
objective, prior semantics, registry membership, benchmark configuration, or
historical result changed. These classes remain research-only clean-room methods;
this metadata correction does not claim paper reproduction or formal admission.

## Verification

`tests/integration/test_joint_shift_baselines.py` parameterizes the metadata
contract over all three classes. The following focused checks passed on 2026-10-19:

- `tests/integration/test_joint_shift_baselines.py`
- `tests/integration/test_joint_shift_classifier.py`
- `tests/estimators/research/test_dynamic_joint_shift_math.py`
- `tests/benchmarks/test_joint_shift_public_benchmark.py`
- `tests/test_builtin_methods.py`
- Ruff check, format check, and `git diff --check`
