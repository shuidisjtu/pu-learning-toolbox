# Customizing Four-Role PU Experiments with Python

Prerequisites: complete the [quickstart](../quickstart.md) and prepare data with true binary
labels for a controlled PU experiment. This tutorial uses synthetic CPU data to call
`ExperimentRunner` directly and extend its strategies. For the official Survey script workflow,
see the [runbook](../../research/pu_survey/runbook.md); exact signatures
are in the [API reference](../reference/api/tools.md#four-role-execution-and-strategy-injection).

The conceptual dimensions are [input sampling and risk pools](../concepts/pu_sampling.md),
[labeling policies](../concepts/scar_sar.md), [prior targets](../concepts/class_prior_semantics.md),
[PA/OA](../concepts/pu_model_selection.md), and [representations](../concepts/representations.md).
This tutorial uses OS input and explicitly configures uPU with `os_or_ts="ts"`, estimating the
marginal risk term with the complete OS pool and consuming a population prior. Here the historical
parameter requests training-risk-pool adaptation; input provenance remains OS.

## 1. Run the complete example

From the repository root:

```powershell
uv sync --python 3.11
uv run python examples/minimal/14_diy_experiment.py
```

[Example source](../../../examples/minimal/14_diy_experiment.py) runs two experiments, each training
three uPU candidates:

| Experiment | Label generation | Training | Selection |
|---|---|---|---|
| `basic` | Built-in SCAR | Built-in `FitTrainer` | Independent PA and OA |
| `custom` | SCAR with a custom policy identifier | Fit scaling only on train | Fixed-threshold PU selection and independent OA |

The terminal reports each protocol's candidate number, threshold, and test Accuracy. Add
`--seed 11` to change the random seed. To save manifests, specify an output directory that does
not already exist:

```powershell
uv run python examples/minimal/14_diy_experiment.py --output-dir tmp/diy-experiment
```

The outputs `basic.json` and `custom.json` contain generation records, candidate parameters,
selections, test metrics, and costs. The example uses the general experiment interfaces directly,
omitting `survey_protocol` and `recipe_registry` from its configuration.

## 2. Prepare four data roles

All four input `DatasetPart` objects use true binary labels, with `1` positive and `0` negative,
and declare `view="clean"`. The runner separately generates PU views for train and pu_val,
where `1` means labeled positive and `0` means unlabeled.

| Role | Runtime purpose |
|---|---|
| `train` | Generate PU labels and train candidates; fit preprocessing statistics here |
| `pu_val` | Generate PU labels for PA or a custom PU selection strategy |
| `clean_val` | True-label validation for OA |
| `test` | Compute final metrics after candidate and threshold selection |

`indices` are global sample identities from one original data source. They must be unique
within and across all four roles. Preserve original identities when subsetting after splitting.
Set `for_selection=False` for test.

This independent snippet builds four inputs from a synthetic population with prior 0.5 and runs
one experiment:

```python
import numpy as np

from pu_toolbox.estimators.risk import UPUClassifier
from pu_toolbox.experiment import DatasetPart, ExperimentRunner, FitTrainer

rng = np.random.RandomState(7)
y_true = np.tile([0, 1], 300)
X = rng.normal(size=(600, 5)) + y_true[:, None]
order = rng.permutation(len(y_true))
parts = []
for start, stop in ((0, 300), (300, 400), (400, 500), (500, 600)):
    ids = order[start:stop]
    parts.append(DatasetPart(
        X=X[ids], labels=y_true[ids], view="clean", indices=ids,
        for_selection=start != 500,
    ))

runner = ExperimentRunner(
    seed=7,
    class_prior=0.5,
    config={"c": 0.4, "trainer": FitTrainer(), "os_or_ts": "ts"},
)
result = runner.fit(
    UPUClassifier(class_prior=0.5, fit_intercept=False),
    *parts,  # train, pu_val, clean_val, test
)
print(result.test_metrics)
```

These inputs are fully labeled benchmark data. For existing real P/U observations, use an
estimator directly; the four-role runner requires the complete-label
input views described above.

## 3. Configure parameters, candidates, and outputs

Each candidate dictionary is applied through the estimator's `set_params`. The runner clones
the model, trains each candidate, and lets PA and OA select candidates and thresholds independently.

```python
runner = ExperimentRunner(
    seed=7,
    class_prior=0.5,
    threshold_candidates=np.linspace(0, 1, 21),
    manifest_path="tmp/my-experiment.json",
    config={
        "c": 0.4,
        "os_or_ts": "ts",
        "trainer": FitTrainer(),
        "candidates": [{"reg_lambda": value} for value in (0.001, 0.01, 0.1)],
        "split_ref": {"dataset": "my_dataset", "split_seed": 7},
    },
)
```

Common configuration choices:

| Location | Purpose |
|---|---|
| `seed` | Label-generation seed and candidate seed when `random_state` is not explicitly set |
| `generator` | Label-generation strategy instance; defaults to `SCARGenerator()` |
| `protocols` | Selection strategy instances; defaults to `[ProtocolPA(), ProtocolOA()]` |
| `class_prior` | Population positive prior, required by PA; comes from generation settings or independent estimation |
| `threshold_candidates` | Validation-normalized threshold grid for built-in PA/OA; defaults to 11 points on 0–1 |
| `manifest_path` | Persist the run manifest; if omitted, read `result.manifest` |
| `config["c"]` | Positive-class labeling rate, default 0.1 |
| `config["candidates"]` | Candidate parameter dictionaries, default `[{}]` |
| `config["trainer"]` | Trainer instance; use `FitTrainer()` for classical single-point runs; the default is `DeepFitTrainer()` |
| `config["split_ref"]` | Data-source and split description; record actual data versions and index identities |
| `config["os_or_ts"]` | Request a training risk view supported by the estimator; omitting it preserves the estimator's default |
| `config["architecture"]` | Explicit `"mlp"` or `"cnn"` capability check |

The runner hands its population `class_prior` to the Trainer; built-in `FitTrainer` forwards it
according to the estimator's signature. The complete OS risk pool in this uPU example consumes
population value 0.5. For default native-U risk instead, a custom Trainer converts this train
partition's prior with `population_to_native_unlabeled(population_prior, mean(y_pu))` before
calling `fit`. [Example 18](../../../examples/minimal/18_custom_encoders.py) demonstrates this
conversion.

New models must support sklearn cloning and parameter interfaces, declare
`label_semantics="pu"` and their actual input capabilities, and return finite per-sample scores
from `decision_function(X)`. The [API reference](../reference/api/tools.md#experiment-layer-experiment)
defines complete data, model, and trajectory contracts. When a custom selector returns
`threshold=None`, the model must supply `predict(X)` returning 0/1 labels.

## 4. Extend the label generator

`Generator` constructs label views; the caller prepares feature arrays. Implement
`generate(X, y_true, c, seed)`, returning a label array and metadata dictionary. This example
reuses SCAR sampling and adds an application-specific policy version:

```python
from pu_toolbox.experiment import SCARGenerator
from pu_toolbox.experiment.protocols import Generator

class AuditedSCARGenerator(Generator):
    output_view = "pu"

    def generate(self, X, y_true, c, seed=None):
        labels, metadata = SCARGenerator().generate(X, y_true, c, seed)
        return labels, {**metadata, "policy_id": "diy-scar-v1"}
```

Inject it with `generator=AuditedSCARGenerator()`. Train and pu_val are passed separately, and
metadata enters `result.manifest["generation"]`. When replacing sampling logic, record the
realized rate, mechanism, seed, and label identity. Ensure every labeled row in a PU view is
truly positive. PA's pu_val requires both labeled positives and U. With SAR, choose the evaluation
criterion according to the mechanism; the official SAR comparison uses OA.

## 5. Extend the trainer

Implement `Trainer.fit` and return a `RunTrajectory`. It receives training PU labels and
optional `val_pu=(X_val, y_pu_val)`. This example fits a scaler within each candidate's train
partition and stores it with the fitted classifier in a prediction Pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pu_toolbox.experiment import FitTrainer
from pu_toolbox.experiment.protocols import Trainer

class ScaledFitTrainer(Trainer):
    trains_on_real_labels = False

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None, os_or_ts=None):
        scaler = StandardScaler()
        trajectory = FitTrainer().fit(
            estimator, scaler.fit_transform(X), y,
            class_prior=class_prior, os_or_ts=os_or_ts,
        )
        trajectory.model = Pipeline([("scale", scaler), ("classifier", trajectory.model)])
        return trajectory
```

Inject an **instance** through `config={"trainer": ScaledFitTrainer(), ...}`. Validation and test
`decision_function` calls reuse this fitted scaler. A custom classical single-point loop can
return:

```python
from pu_toolbox.experiment.tracking import EpochRecord, RunTrajectory

trajectory = RunTrajectory(
    epochs=[EpochRecord(epoch=1, metrics={})],
    model=fitted_model,
)
```

Deep models can use `DeepFitTrainer` and per-epoch checkpoints. Configuration includes
`capture_epoch_checkpoints`, `checkpoint_dir`, `checkpoint_bytes_per_component`, and
`reclaim_unselected_checkpoints`. Custom trainers produce their own per-epoch weights and
trajectories. `SelectionArtifact.epoch` is the 1-based position in the trajectory;
`checkpoint_index` identifies selected weights. The
[API reference](../reference/api/tools.md#experiment-layer-experiment) defines restoration
and storage-budget contracts.

## 6. Extend the selection protocol

Implement `SelectionProtocol.select`, returning a `SelectionArtifact`. This example selects
a candidate using built-in PU proxy accuracy at validation-normalized threshold 0.5:

```python
from dataclasses import replace

from pu_toolbox.experiment import ProtocolOA, ProtocolPA
from pu_toolbox.experiment.protocols import SelectionProtocol

class FixedThresholdPA(SelectionProtocol):
    name = "PA-fixed"

    def select(self, trajectories, val_part, threshold_candidates=None, *, class_prior=None):
        artifact = ProtocolPA().select(
            trajectories, val_part, np.array([0.5]), class_prior=class_prior,
        )
        return replace(artifact, protocol=self.name)
```

Inject it through `protocols=[FixedThresholdPA(), ProtocolOA()]`. The complete example injects
all three custom strategies. Each strategy needs a unique `name` matching its returned
`artifact.protocol`.

Ordinary `SelectionProtocol` subclasses receive generated pu_val. A custom OA selector requiring
clean validation labels should inherit `ProtocolOA`, which the runner uses to route clean_val.
Keep the artifact's protocol synchronized when renaming the selector. When defining custom
thresholds and score transformations, store validation-side `val_score_min` and `val_score_scale`
in `artifact.metrics`; the runner reuses them for final evaluation.

### 6.1 Work through independent PA/OA selection

```powershell
uv run python examples/minimal/14_diy_experiment.py --fixed-selection
```

This path reuses built-in protocols to compare two fixed single-point candidates: A scores the
first feature column, B the second. PU validation features are `[[1,0],[0,1]]` with PU labels
`[1,0]`; clean validation features are `[[0,1],[1,0]]` with true labels `[1,0]`.
The threshold grid is `[0,0.5,1]` and population prior is 0.5.

| Protocol | Selected candidate | Threshold | Validation criterion | Independent test Accuracy |
|---|---|---:|---:|---:|
| PA | A (index 0) | 0.5 | 1.5 | 0 |
| OA | B (index 1) | 0.5 | 1 | 1 |

Test has the same feature values as clean validation but independent sample identities; its
labels are read after selection. Fixed single-point models record `epoch` and `checkpoint_index`
as `None`. Actual per-epoch models use `EpochCheckpointTrainer` to save trajectories, then the
protocols independently select snapshots. See the
[API reference](../reference/api/tools.md#experiment-layer-experiment) for snapshot contracts.
PA includes a population-prior constant offset; see [model-selection concepts](../concepts/pu_model_selection.md)
for its meaning and formula.

## 7. Read results and continue prediction

```python
print(result.selections)      # Selected run, epoch, threshold and validation metrics
print(result.test_metrics)    # Independent test metrics
print(result.failures)        # Candidate failures and retries
print(result.resources)       # Training and tuning costs
print(result.manifest["selection"]["OA"]["candidate_index"])
```

`SelectionArtifact.run_index` indexes successful training trajectories. Read the original
candidate number from manifest `selection[name]["candidate_index"]`.

For continued classical-model prediction, retrieve both `result.selected_models[name]` and
`result.selections[name]`, reusing the validation score transform and selected threshold.
The complete example's `predict_selected(result, name, X_new)` implements this process so
new predictions follow the same classification rule as the report.

## 8. Replace research components one at a time

Start with fixed splits, seed, and known population prior. Replace one component at a time and
check the following outputs:

| Component | Minimal replacement | Output to inspect |
|---|---|---|
| Data and sampling | Replace the four `DatasetPart` objects; see example 15 for OS/independent TS | Source indices, roles, population and U priors |
| Generator | Replace SCAR with LBE-A/B; use OA for SAR | Formula version, requested/realized counts, label hashes; see example 17 |
| Prior and model | Fixed population value or independent estimate; pass the Trainer's risk target | Estimation provenance and actual training consumption; see examples 16 and 18 |
| Trainer | Inject `ScaledFitTrainer` or a custom loop | Train-fitted statistics, trajectories, cost, and checkpoints |
| SelectionProtocol | Replace standard PA with `FixedThresholdPA` | Independent candidates/snapshots/thresholds and validation transforms |
| Representation | Inject an encoder or extract frozen features first | Initialization, fit scope, and representation identity; see [encoders](custom_encoders.md) |

The complete example's `basic` and `custom` runs share the population, split, sampling policy,
training risk pool, and candidate budget, demonstrating generator, scaled trainer, and selector
injection. Manifests preserve split_ref, generation records, candidates, risk views, independent
selections, and test metrics for reconstructing the setting.

## 9. Next steps

- Replace the four `DatasetPart` objects with your data, retaining global identities and
  train-only preprocessing.
- Begin with candidate dictionaries, then replace the generator, trainer, or selector as needed.
- Consult the [API reference](../reference/api/tools.md#four-role-execution-and-strategy-injection) for exact
  contracts. The [experiment-layer design](../../dev/experiment_layer.md) is optional Chinese
  background on design rationale.
- Use the [Survey runbook](../../research/pu_survey/runbook.md) for official
  recipes and workbook generation.
