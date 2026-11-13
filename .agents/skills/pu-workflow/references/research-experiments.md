# Four-role research experiments

Read this reference for explicit validation/test splits, PA/OA selection,
custom generators or trainers, epoch checkpoints, and clean-support research.
Use [sampling-and-priors.md](sampling-and-priors.md) when defining input
sampling and training priors, and [representations.md](representations.md)
for images or frozen features.

## Prepare the experiment

Record the population, source identities, split seed, labeling policy,
population prior and its source, candidates, selection protocols, and budget.
Split by original sample identity, patient, or other application group before
constructing PU observations. Fit preprocessing within train.

`ExperimentRunner.fit` accepts four `DatasetPart(view="clean")` inputs with
true binary labels. The generator constructs PU views for train and pu_val.
For existing real PU observations, use the Pipeline/estimator route in
[analysis.md](analysis.md).

| Role | Supervision used | Purpose |
|---|---|---|
| train | Generated PU labels | Fit candidates and train-side preprocessing |
| pu_val | Generated PU labels | PA or a custom PU selector |
| clean_val | True labels | OA selector |
| test | True labels, `for_selection=False` | Evaluate selected predictions |

## Run a complete CPU experiment

Save this as a script in the working directory and execute it with the
environment containing `pu-toolbox`. It creates a new `research-output`
directory with a run manifest.

```python
from pathlib import Path

import numpy as np

from pu_toolbox.estimators.risk import UPUClassifier
from pu_toolbox.experiment import (
    DatasetBundle, DatasetPart, ExperimentRunner,
    FitTrainer, ProtocolOA, ProtocolPA, SCARGenerator,
)

rng = np.random.RandomState(7)
parts = []
offset = 0
for role, size in (("train", 240), ("pu_val", 80), ("clean_val", 80), ("test", 80)):
    truth = rng.permutation(np.arange(size) % 2)
    features = rng.normal(size=(size, 5)) + truth[:, None]
    parts.append(DatasetPart(
        X=features, labels=truth, view="clean",
        indices=np.arange(offset, offset + size),
        for_selection=role != "test",
    ))
    offset += size
bundle = DatasetBundle(*parts)
output = Path("research-output")
output.mkdir()
runner = ExperimentRunner(
    seed=7, generator=SCARGenerator(),
    protocols=[ProtocolPA(), ProtocolOA()], class_prior=0.5,
    threshold_candidates=np.linspace(0, 1, 11),
    manifest_path=str(output / "run.json"),
    config={
        "c": 0.4, "os_or_ts": "ts", "trainer": FitTrainer(),
        "candidates": [{"reg_lambda": value} for value in (0.01, 0.1)],
        "split_ref": {"dataset": "synthetic-balanced", "split_seed": 7},
    },
)
result = runner.fit(
    UPUClassifier(class_prior=0.5, fit_intercept=False, random_state=7),
    bundle.train, bundle.pu_val, bundle.clean_val, bundle.test,
)
for name, artifact in result.selections.items():
    selected = result.manifest["selection"][name]
    print(name, selected["candidate_index"], artifact.epoch,
          artifact.checkpoint_index, artifact.threshold, result.test_metrics[name])
```

This uses OS input with fixed-count SCAR labeling. `os_or_ts="ts"` requests
the estimator's complete OS mixture risk pool: native U plus labeled P.
The uPU loss in this configuration consumes population prior 0.5.
Record the input sampling design and risk view as separate settings.

## Match prior consumption to the trainer

Runner `class_prior` supplies the population prior used for PA and is handed
to the trainer. `FitTrainer` forwards it according to the estimator's fit
signature. For an estimator's native-U loss on OS data, a custom trainer computes
the training-partition prior with
`population_to_native_unlabeled(population_prior, mean(y_pu), input_sampling="os")`
and passes that value to the estimator. Record both population and consumed
values. Pipeline performs its own declared-target resolution per fold;
custom research trainers implement the chosen research contract.

## Interpret independent PA/OA selection

PA uses the PU validation view and the population prior. Its current built-in
criterion is the OS proxy accuracy:

```math
PA=2\pi\widehat P_L(\hat Y=1)+\widehat P_{L\cup U}(\hat Y=0).
```

Representative labeled positives and a correct population prior give the
fixed-rule expectation `Accuracy + pi`. OA uses representative clean
validation labels and computes `mean(predicted_class == true_class)`.
Select the protocols that match the available supervision and labeling
mechanism. The standard SAR comparison uses OA.

PA and OA each choose their candidate, checkpoint, and threshold from the
shared training trajectories, then evaluate the selected rule on test.
Read these result fields:

- `result.manifest["selection"][name]["candidate_index"]`: original candidate
  number. `artifact.run_index` indexes successful training trajectories.
- `artifact.epoch`: 1-based position within the selected trajectory's epoch
  list; `artifact.checkpoint_index`: selected weight snapshot. Classical
  single-point runs record both as `None`.
- `artifact.metrics`: validation criterion and `val_score_min` /
  `val_score_scale` used to transform scores before thresholding.
- `result.test_metrics[name]`: final metrics for that protocol's selection.
- `manifest["generation"]`, `manifest["research_settings"]`,
  `manifest["candidate_runs"]`, `failures`, and `resources`: labeling,
  effective training settings, candidate outcomes, and costs.

Continue prediction with the stored validation transform and threshold:

```python
def predict_selected(result, name, X_new):
    model = result.selected_models[name]
    artifact = result.selections[name]
    if artifact.threshold is None:
        return model.predict(X_new)
    scores = model.decision_function(X_new)
    minimum = artifact.metrics.get("val_score_min")
    scale = artifact.metrics.get("val_score_scale")
    if minimum is not None and scale:
        scores = (scores - minimum) / scale
    return (scores >= artifact.threshold).astype(int)
```

## Extend a research component

- **Generator:** implement `Generator.generate(X, y_true, c, seed)` returning
  `(y_pu, metadata)` and declare `output_view="pu"`. Include the mechanism,
  requested/actual labeling counts, randomness, and policy version.
  Built-in `SARLBEAGenerator` and `SARLBEBGenerator` are feature-dependent,
  weighted fixed-count policies. Use `generator=SARLBEAGenerator()` and
  `protocols=[ProtocolOA()]` for an LBE-A/OA comparison; retain its smoothing
  and exponent settings in the reported policy.
- **Trainer:** implement `Trainer.fit(estimator, X, y, *, class_prior=None,
  val_pu=None, os_or_ts=None)` and return `RunTrajectory`. Store fitted
  preprocessing together with the predictor. Inject a trainer instance in
  `config["trainer"]`.
- **Selector:** implement `SelectionProtocol.select(...)` returning
  `SelectionArtifact`. Give each selector a unique name that matches the
  returned `artifact.protocol`. PU selectors receive generated pu_val;
  selectors inheriting `ProtocolOA` receive clean_val. Preserve the
  validation score transform in the artifact for final evaluation.
- **Epoch checkpoints:** use `DeepFitTrainer` with a model exposing
  `epoch_callback`; the runner enables epoch capture through
  `config["capture_epoch_checkpoints"]=True`. Set `checkpoint_dir` and
  `checkpoint_bytes_per_component` in the config to record persistent
  snapshots and their storage budget. The runner constructs
  `pu_toolbox.experiment.checkpoints.EpochCheckpointTrainer` for this route.
  Budget disk space and candidates before training. Restore selected weights
  and the validation prediction rule with `load_selected_checkpoint(
  result.manifest["selection"][name], template)` from that module.

For clean-support methods, prepare a separate support set with original
sample identities, label availability, size, and training-stage use. Keep
support distinct from clean_val and test. Use the method's audited support
trainer and compatible protocol; support checkpoint trainers in
`pu_toolbox.experiment.support_checkpoints` support OA and the explicitly
named `ProtocolPACleanSupport` for eligible trajectories. Report
`PA+clean-support`, support supervision, and selected checkpoints together.

Finish with per-protocol selection/test results, actual candidate and epoch
budgets, split/label identities, prior consumption, representation settings,
and paths to manifests and model artifacts. In a source checkout,
`examples/minimal/14_diy_experiment.py` and `17_sar_policies.py` provide
larger component-injection examples; `docs/user/howto/custom_experiments.md`
contains the detailed public API tutorial.
