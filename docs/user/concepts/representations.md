# Representations, Encoders, and Training Paths

## 1. Methods and representations jointly define the experiment

Running the same PU method on 2-D features, a trainable image encoder, or frozen
features changes both the learning task and computational budget. Record the input
shape, encoder/head, initialization or pretraining source, and training scope when
comparing results.

| Current training path | Input and training process | Information to record |
|---|---|---|
| `native_2d` | Train the method directly on 2-D features | Feature provenance, training-side preprocessing, and input dimensions |
| `native_cnn` | Train an image network end to end | Backbone, head, initialization, normalization, and update scope |
| `cnn_feature_adapter` | Extract 2-D features with a frozen encoder, then train the PU method | Encoder state, pretraining/fit scope, and feature/split identities |

[feature_adapter.py](../../../pu_toolbox/experiment/feature_adapter.py) defines the
training-path vocabulary and comparison gates. Group frozen-feature results
separately from end-to-end CNN results; check backbone, training scope, and compute
budget within each group.

## 2. Current capability declarations

Estimators and the registry describe capabilities with four fields:

| Field | Meaning |
|---|---|
| `native_architectures` | Native `mlp` / `cnn` architecture paths |
| `input_ndims` | Accepted array dimensionalities, such as 2-D features or 4-D NCHW images |
| `encoder_parameter` | Constructor parameter receiving an injected encoder, or `None` |
| `trains_encoder` | Whether the method trains that encoder |

`input_ndims` records accepted dimensionalities; `native_architectures` records
native architecture paths. The actual consumption path determines whether an image
passes through an encoder or is flattened. Authoritative declarations are in
[core/base.py](../../../pu_toolbox/core/base.py). The
[dual-architecture plan](../../dev/dual_architecture_plan.md) provides optional
Chinese implementation background.

## 3. Shared encoders and method lifecycles

`pu_toolbox.estimators.deep.build_encoder` supports `architecture="mlp"` (returns
`None` to use the method's built-in path) and `architecture="cnn"`. Available CNN
backbones are `cnn13`, `cnn13_no_bn`, `resnet18`, and `resnet50`. Inputs use NCHW;
normalization mean and standard deviation default to 0.5 per channel. ResNet
currently uses random initialization with `weights=None`.

nnPU accepts an injected `encoder`. Each `fit` deep-copies it into `encoder_`
before training, isolating candidate and CV-fold state. Consult individual
estimators for generator, multi-network, and head lifecycles. See
[nnpu.py](../../../pu_toolbox/estimators/risk/nnpu.py) and
[vision.py](../../../pu_toolbox/estimators/deep/vision.py).

Check architecture declarations without training or downloading weights. Built-in algorithms
are available on first lookup; no explicit registry initialization is needed:

```python
from pu_toolbox.registry import get_algorithm

model_class = get_algorithm("nnpu")
assert "cnn" in model_class.native_architectures
assert 4 in model_class.input_ndims
assert model_class.encoder_parameter == "encoder"
assert model_class.trains_encoder
```

See the [API reference](../reference/api/tools.md#build_encoder) for precise signatures, and the
[encoder tutorial](../howto/custom_encoders.md) and
[example 18](../../../examples/minimal/18_custom_encoders.py) for injection,
backbone replacement, and persistence.

## 4. Frozen features and data isolation

`adapt_image_bundle_to_features` uses the same encoder to extract features for
all four roles in evaluation mode without gradients. It compares encoder-state
hashes before and after extraction, preserves original indices and label views,
and returns representation-identity metadata. It does not train the encoder.

`encoder_fit_scope="fixed_external"` records the caller's declaration that the
encoder comes from a fixed external source. `"train_partition_only"` requires fit
indices exactly matching train. This is a caller assertion about fit scope;
the index check verifies that the declared scope matches train. Record pretraining
data, weight versions, and supervision scope in `backbone_manifest`.

Fit normalization, dimensionality reduction, and trainable encoders only on the
permitted training roles. Preserve split, encoder-state, and representation hashes
for comparisons, and place the PN oracle under the same representation and training
budget conditions. See [custom experiments](../howto/custom_experiments.md) for
composing a research workflow.

## 5. Sources

Chen et al. (2026), Appendix D.1 (within-dataset representation and backbone conventions):
[PU-Bench](https://openreview.net/forum?id=tb8DabMbMq).
For the reference engineering implementation, see the
[pinned model_factory.py](https://github.com/XiXiphus/PU-Bench/blob/2d95a19eefd72e66ff30128ec2f65e1d1d4cc077/train/utils/model_factory.py).

The paper's comparison conditions and the toolbox's encoder lifecycle are distinct
evidence; actual toolbox code determines the latter. See the
[Survey protocol](../../research/pu_survey/pu_survey_protocol.md) for the backbones
chosen by the official experiment configuration.
