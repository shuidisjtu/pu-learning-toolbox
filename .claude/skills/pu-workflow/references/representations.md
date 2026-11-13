# Images, encoders, and frozen features

Read this reference for NCHW images, an injected encoder, or PU methods trained
on frozen features. Use [research-experiments.md](research-experiments.md)
when the task includes explicit validation/test roles and independent
selection, and [sampling-and-priors.md](sampling-and-priors.md) to resolve
the prior consumed by each loss.

## Select an executable training path

| Path | Model input and training | Record |
|---|---|---|
| `native_2d` | Train a method on existing two-dimensional features | Feature source, train-fitted preprocessing, dimension |
| `native_cnn` | Train an image encoder and prediction head | Backbone, initialization, normalization, trainable scope, budget |
| `cnn_feature_adapter` | Extract fixed encoder features, then train a PU method | Encoder state/source, fit scope, representation and split identities |

Query installed capabilities before choosing an image method:

```python
from pu_toolbox.registry import get_algorithm, register_all_builtin_methods

register_all_builtin_methods()
model_class = get_algorithm("nnpu")
print(model_class.native_architectures, model_class.input_ndims,
      model_class.encoder_parameter, model_class.trains_encoder)
```

The CNN Pipeline/CLI route uses an explicit classifier that declares encoder
injection and CNN input support. Example for headered PU labels and a finite
float NCHW array saved as `X.npy`, with a known population prior:

```bash
pu-toolbox run --data X.npy --labels y_pu.csv --classifier nnpu --architecture cnn --backbone cnn13_no_bn --device cpu --class-prior 0.4 --max-epochs 2 --cv 2 --out-dir image-output
```

Use `uv run pu-toolbox ...` in a uv-managed project. Choose device, epochs,
batch size, prior source, and candidate count from the task's budget.
Raw-image tasks use shape/finite-value and label-count checks; table profiling
uses two-dimensional data. An explicit feature-extraction route supplies
the two-dimensional features for table profiling and recommendations.

## Train with an encoder template

Torch is required for these paths. This complete CPU example uses a tiny
randomly initialized encoder, 32 synthetic images, and two training epochs.
It writes a JSON record plus the fitted predictor in `encoder-output`.

```python
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pu_toolbox.estimators.risk import NonNegativePUClassifier
from pu_toolbox.experiment.feature_adapter import encoder_state_sha256
from pu_toolbox.preprocessing import make_scar_labels
from pu_toolbox.prior import population_to_native_unlabeled

torch.manual_seed(7)
rng = np.random.RandomState(7)
template = nn.Sequential(
    nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(16, 4),
)
y_true = np.tile([0, 1], 16)
X = rng.uniform(size=(32, 1, 8, 8)).astype(np.float32)
y_pu = make_scar_labels(y_true, c=0.5, random_state=7)
pi_u = population_to_native_unlabeled(0.5, y_pu.mean(), input_sampling="os")
initial_hash = encoder_state_sha256(template)
model = NonNegativePUClassifier(
    encoder=template, class_prior=pi_u, batch_size=8,
    max_epochs=2, random_state=7, device="cpu",
).fit(X, y_pu)
assert model.encoder_ is not template
assert encoder_state_sha256(template) == initial_hash
output = Path("encoder-output")
output.mkdir()
(output / "native.pkl").write_bytes(pickle.dumps(model))
settings = {
    "input_sampling": "os", "population_class_prior": 0.5,
    "training_prior": {"target": "native_unlabeled", "value": pi_u},
    "training_path": "native_cnn", "pretraining": "none", "seed": 7,
    "initial_encoder_sha256": initial_hash,
    "fitted_encoder_sha256": encoder_state_sha256(model.encoder_),
    "budget": {"max_epochs": 2, "batch_size": 8, "n_candidates": 1},
}
(output / "settings.json").write_text(
    json.dumps(settings, indent=2, allow_nan=False), encoding="utf-8"
)
print(settings)
```

Direct nnPU `fit` with its default native-U loss consumes `pi_u` computed
from the current train partition. `PUPipeline` accepts the population prior
and resolves the declared consumer target for each fold and final fit.
nnPU deep-copies `template` into `encoder_` for each fit, giving candidates
and folds their own trainable encoder. The encoder returns finite
two-dimensional features; the model builds its prediction head from that
feature dimension.

For shared backbones, use
`build_encoder("cnn", backbone="cnn13_no_bn", in_channels=1)` from
`pu_toolbox.estimators.deep`. Discover accepted backbones through the installed
CLI or function signature. Shared encoders include channel normalization;
record the chosen means/stds and image range. Current ResNet builders use
random initialization (`weights=None`) and require torchvision.

## Freeze an encoder and train a two-dimensional method

The following continues the encoder example above. It creates disjoint
clean roles, extracts features with the fixed initialization template,
and trains uPU on the train PU observations.

```python
from pu_toolbox.estimators.risk import UPUClassifier
from pu_toolbox.experiment import DatasetBundle, DatasetPart, adapt_image_bundle_to_features

parts = []
offset = 0
for role, size in (("train", 32), ("pu_val", 12), ("clean_val", 12), ("test", 12)):
    parts.append(DatasetPart(
        X=rng.uniform(size=(size, 1, 8, 8)).astype(np.float32),
        labels=np.arange(size) % 2, view="clean",
        indices=np.arange(offset, offset + size), for_selection=role != "test",
    ))
    offset += size
bundle = DatasetBundle(*parts)
observed = make_scar_labels(bundle.train.labels, c=0.5, random_state=7)
training_prior = population_to_native_unlabeled(
    0.5, observed.mean(), input_sampling="os"
)
features, representation = adapt_image_bundle_to_features(
    bundle, template, feature_version="random-tiny-seed7-v1",
    backbone_manifest={"name": "tiny", "seed": 7, "pretraining": "none"},
    encoder_fit_scope="fixed_external", batch_size=8, device="cpu",
)
frozen_model = UPUClassifier(
    class_prior=training_prior, fit_intercept=False, random_state=7,
).fit(features.train.X, observed)
(output / "frozen.pkl").write_bytes(pickle.dumps((template, frozen_model)))
(output / "representation.json").write_text(
    json.dumps(representation, indent=2, allow_nan=False), encoding="utf-8"
)
print(features.train.X.shape, representation["representation_sha256"])
```

The adapter extracts all roles in eval/no-grad mode, verifies encoder state,
restores module modes, and preserves labels and sample identities.
`fixed_external` declares a fixed template's source. For an encoder fitted
on train, pass `encoder_fit_scope="train_partition_only"` with
`encoder_fit_indices=bundle.train.indices`; record the fitted weights,
training data and supervision in `backbone_manifest`.

Read `representation_sha256`, `encoder_state_sha256`, `split_sha256`,
`feature_sha256`, `encoder_fit_scope`, and `backbone_manifest` from the
representation record. Compare methods within a common representation and
split with shared candidate, seed, and epoch budgets. Group results by
`native_cnn` or `cnn_feature_adapter` and report each group's settings.

For an explicit PA/OA experiment, pass the adapted four-role bundle to
the runner described in [research-experiments.md](research-experiments.md)
and choose the trainer's consumed prior for its risk view. Fit normalization,
dimensionality reduction, and trainable representations on train.

## Save, reload, and predict

Load model files produced by the current trusted workflow. The native model
contains its fitted encoder and head. The frozen path stores the encoder and
the two-dimensional classifier as a pair. Apply the same preprocessing and
extraction before new predictions:

```python
from pu_toolbox.experiment.feature_adapter import extract_image_features

encoder, classifier = pickle.loads((output / "frozen.pkl").read_bytes())
new_features = extract_image_features(encoder, bundle.test.X, batch_size=8, device="cpu")
predictions = classifier.predict(new_features)
assert predictions.shape == bundle.test.labels.shape
```

Report the actual path, input shape, encoder initialization/pretraining,
trainable/frozen scope, prior consumption, training budget, and saved artifact
paths. Source-checkout examples `examples/minimal/18_custom_encoders.py` and
`docs/user/howto/custom_encoders.md` provide the larger demonstration and
backbone variants.
