# Injecting Encoders and Using Frozen Features

Prerequisites: install `uv sync --extra torch` and read
[representations and training paths](../concepts/representations.md) and
[prior targets](../concepts/class_prior_semantics.md). This tutorial uses synthetic CPU images
for end-to-end nnPU and traditional uPU trained on frozen encoder features.

## 1. Run the complete example

```powershell
uv run python examples/minimal/18_custom_encoders.py
uv run python examples/minimal/18_custom_encoders.py --output-dir tmp/encoder-demo
```

Use a new output directory; it contains `settings.json`, `native.pkl`, and `frozen.pkl`.
[Example 18](../../../examples/minimal/18_custom_encoders.py) uses role sizes 32/12/12/12,
images of shape `(N,1,8,8)` with values in `[0,1]`, and population prior 0.5.
Fixed-count SCAR labels half the train positives, giving labeled fraction 0.25 and converted
native-U prior 1/3. Both paths use fixed training budgets and the models' default thresholds;
test labels are read for final evaluation.

| Path | Encoder use | Training parameters and records |
|---|---|---|
| `native_cnn` | nnPU deep-copies the template at each fit and trains the copy and prediction head | Native-U prior, initial/fitted weight hashes, epochs, and save/load checks |
| `cnn_feature_adapter` | The same initial random template extracts all four roles in eval/no-grad mode | Native-U prior, fixed external provenance, and split/representation/feature hashes |

The random template records its seed and `pretraining="none"`. Interpret metrics under each
path's own training settings.

## 2. Construct a template and train nnPU

This snippet runs independently. Set the initialization seed before creating the template;
`random_state` controls fitting. The encoder produces finite two-dimensional features;
nnPU infers their dimension and appends a single-output linear head.

```python
import numpy as np
import torch
from torch import nn

from pu_toolbox.estimators.risk import NonNegativePUClassifier
from pu_toolbox.preprocessing import make_scar_labels
from pu_toolbox.prior import population_to_native_unlabeled

torch.manual_seed(7)
template = nn.Sequential(
    nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(16, 4),
)
rng = np.random.RandomState(7)
y_true = np.tile([0, 1], 16)
X = rng.uniform(size=(32, 1, 8, 8)).astype(np.float32)
y_pu = make_scar_labels(y_true, c=0.5, random_state=7)
pi_u = population_to_native_unlabeled(0.5, y_pu.mean(), input_sampling="os")
model = NonNegativePUClassifier(
    encoder=template, class_prior=pi_u, batch_size=8,
    max_epochs=2, random_state=7, device="cpu",
).fit(X, y_pu)
assert model.encoder_ is not template
assert model.decision_function(X).shape == (32,)
```

Direct nnPU `fit` with its default risk view consumes a native-U `class_prior`; the explicit
conversion uses this train partition's labeled fraction. `PUPipeline` instead accepts a
population prior and converts it using classifier declarations and per-fold counts; see the
[pipeline guide](pipeline.md).

The template retains initialization state; `model.encoder_` holds fitted state. Reuse the
initial template across candidates or folds, with each fit obtaining its own copy.
Use `model=nn.Linear(4,1)` to replace the score head.

## 3. Replace the shared backbone

Pass a built-in backbone through the same `encoder` parameter:

```python
from pu_toolbox.estimators.deep import build_encoder

torch.manual_seed(7)
shared = build_encoder("cnn", backbone="cnn13_no_bn", in_channels=1)
shared.eval()
with torch.no_grad():
    assert shared(torch.zeros(2, 1, 32, 32)).shape == (2, 512)
```

`build_encoder` includes channel normalization, defaulting to mean/std 0.5. The tiny template
uses input values directly; record parameters if adding custom normalization. The complete
example can optionally run shared backbones; both commands use `(N,1,32,32)`:

```powershell
uv run python examples/minimal/18_custom_encoders.py --backbone cnn13_no_bn --max-epochs 1
uv run python examples/minimal/18_custom_encoders.py --backbone resnet18 --max-epochs 1
```

ResNet requires torchvision and uses random initialization with `weights=None`. See the
[build_encoder contract](../reference/api.md#build_encoder) for exact parameters.

## 4. Train a traditional method with a frozen encoder

First assemble four clean roles into a `DatasetBundle`, then extract features with one encoder.
This snippet continues with `rng` and `template` above and generates train PU labels:

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
y_pu = make_scar_labels(bundle.train.labels, c=0.5, random_state=7)
pi_u = population_to_native_unlabeled(0.5, y_pu.mean(), input_sampling="os")

features, representation = adapt_image_bundle_to_features(
    bundle, template, feature_version="my-fixed-encoder-v1",
    backbone_manifest={"name": "tiny", "seed": 7, "pretraining": "none"},
    encoder_fit_scope="fixed_external", batch_size=8, device="cpu",
)
frozen_model = UPUClassifier(
    class_prior=pi_u, fit_intercept=False, random_state=7,
).fit(features.train.X, y_pu)
print(representation["representation_sha256"])
```

The adapter extracts in eval/no-grad mode, checks weight hashes, restores each module's mode,
and preserves role indices and labels. `fixed_external` declares the source of the fixed template.
To freeze an encoder fitted on train, use `encoder_fit_scope="train_partition_only"` and
`encoder_fit_indices=bundle.train.indices`; record training data, supervision type, and weight
version in `backbone_manifest`.

## 5. Save, reload, and continue prediction

The end-to-end model contains the fitted encoder and head. The frozen path stores the encoder
and classifier:

```python
import pickle

restored = pickle.loads(pickle.dumps(model))
np.testing.assert_array_equal(restored.decision_function(X), model.decision_function(X))
```

After loading your saved `native.pkl`, call `decision_function(X_new)` directly.
Loading `frozen.pkl` returns `(encoder, classifier)`. Apply the same preprocessing to new images,
extract features with the saved encoder in eval/no-grad mode, and call the classifier:

```python
encoder, classifier = pickle.loads(pickle.dumps((template, frozen_model)))
X_new = bundle.test.X
encoder.eval()
with torch.no_grad():
    new_features = encoder(torch.as_tensor(X_new, dtype=torch.float32)).cpu().numpy()
predictions = classifier.predict(new_features)
```

The saved CPU example has regression coverage for prediction consistency on both paths.
Settings preserve data splits, training budgets, and initialization provenance. See
[representation concepts](../concepts/representations.md) for comparisons and the
[four-role experiment tutorial](custom_experiments.md) for independent selection and custom trainers.
