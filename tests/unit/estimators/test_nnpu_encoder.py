# ruff: noqa: N802, N803, N806
"""nnPU encoder-injection tests (dual_architecture_plan.md §5 阶段 2).

Covers: Sequential composition, custom head via ``model``, 4-D input paths,
fail-fast without encoder, default-Linear regression, seed determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.core.exceptions import ValidationError  # noqa: E402
from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier  # noqa: E402

pytestmark = [pytest.mark.unit]


class _TinyMLPEncoder(torch.nn.Module):
    """2-D in → rep_dim out (no torchvision dependency)."""

    def __init__(self, in_features=5, rep_dim=8):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, 16), torch.nn.ReLU(), torch.nn.Linear(16, rep_dim)
        )

    def forward(self, x):
        return self.net(x)


class _TinyCNNEncoder(torch.nn.Module):
    """4-D (C,6,6) in → rep_dim out (no torchvision dependency)."""

    def __init__(self, in_channels=1, rep_dim=8):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, 4, 3, padding=1)
        self.fc = torch.nn.Linear(4 * 6 * 6, rep_dim)

    def forward(self, x):
        return self.fc(self.conv(x).relu().flatten(start_dim=1))


class _ScalarEncoder(torch.nn.Module):
    """Outputs 1-D per-sample scalars — invalid representation.

    ``validate_encoder_features`` rejects any output that is not 2-D
    after ``flatten(start_dim=1)``; a 1-D output survives flattening
    unchanged and triggers the ValueError (a 3-D output would be
    flattened into a legal 2-D shape, so 1-D is the right trigger).
    """

    def forward(self, x):
        return x.mean(dim=-1) if x.ndim == 2 else x.mean(dim=(-3, -2, -1))


def _table_data(n=40, d=5, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.normal(0.0, 1.0, size=(n, d)).astype(np.float32)
    y_pu = np.concatenate([np.ones(10, dtype=int), np.zeros(n - 10, dtype=int)])
    return X, y_pu


def _image_data(n=24, channels=1, size=6, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.normal(0.5, 0.3, size=(n, channels, size, size)).astype(np.float32)
    y_pu = np.concatenate([np.ones(6, dtype=int), np.zeros(n - 6, dtype=int)])
    return X, y_pu


def _snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _fit_encoder_clf(X, y_pu, *, encoder, model=None, seed=42, val=None):
    clf = NonNegativePUClassifier(
        model=model,
        encoder=encoder,
        class_prior=0.3,
        max_epochs=1,
        batch_size=16,
        random_state=seed,
        device="cpu",
    )
    clf.fit(X, y_pu, validation_data=val)
    return clf


def test_encoder_with_2d_input_composes_sequential_and_trains():
    X, y_pu = _table_data()
    encoder = _TinyMLPEncoder()
    initial = _snapshot(encoder)
    clf = _fit_encoder_clf(X, y_pu, encoder=encoder)
    assert isinstance(clf.model_, torch.nn.Sequential)
    assert clf.encoder_ is clf.model_[0]
    assert isinstance(clf.model_[-1], torch.nn.Linear)
    assert clf.model_[-1].in_features == 8  # rep_dim
    assert clf.model_[-1].out_features == 1
    # Training took effect on the encoder and the default head.
    assert not torch.equal(
        next(iter(_snapshot(clf.encoder_).values())), next(iter(initial.values()))
    )
    assert clf.decision_function(X[:8]).shape == (8,)


def test_custom_head_via_model_parameter_is_deepcopied_and_trained():
    X, y_pu = _table_data()
    head = torch.nn.Linear(8, 1)
    head_initial = {k: v.detach().clone() for k, v in head.state_dict().items()}
    clf = _fit_encoder_clf(X, y_pu, encoder=_TinyMLPEncoder(), model=head)
    assert clf.model_[1] is not head  # deepcopied, not the caller's instance
    after = _snapshot(clf.model_[1])
    assert not torch.equal(after["weight"], head_initial["weight"])


def test_invalid_encoder_output_raises_value_error():
    X, y_pu = _table_data()
    with pytest.raises(ValueError, match="encoder"):
        _fit_encoder_clf(X, y_pu, encoder=_ScalarEncoder())


def test_default_linear_model_preserved_without_encoder():
    X, y_pu = _table_data(d=5)
    clf = NonNegativePUClassifier(
        class_prior=0.3, max_epochs=1, batch_size=16, random_state=42, device="cpu"
    )
    clf.fit(X, y_pu)
    assert isinstance(clf.model_, torch.nn.Linear)
    assert clf.model_.in_features == 5


def test_encoder_with_4d_input_trains_and_predicts():
    X, y_pu = _image_data()
    clf = _fit_encoder_clf(X, y_pu, encoder=_TinyCNNEncoder())
    scores = clf.decision_function(X)
    assert scores.shape == (24,)


def test_4d_input_without_encoder_is_rejected():
    X, y_pu = _image_data()
    clf = NonNegativePUClassifier(
        class_prior=0.3, max_epochs=1, batch_size=16, random_state=42, device="cpu"
    )
    with pytest.raises((ValidationError, ValueError)):
        clf.fit(X, y_pu)


def test_4d_validation_data_accepted_with_encoder():
    X, y_pu = _image_data()
    X_val, y_val = _image_data(n=8, seed=2)
    clf = _fit_encoder_clf(X, y_pu, encoder=_TinyCNNEncoder(), val=(X_val, y_val))
    assert clf.decision_function(X_val).shape == (8,)


def test_seed_determinism_with_encoder():
    X, y_pu = _table_data()
    # The encoder is constructed by the caller before fit seeds the RNG, so
    # seed the construction too — otherwise its init draw differs between
    # the two fits and the assertion fails in most runs.
    torch.manual_seed(123)
    clf1 = _fit_encoder_clf(X, y_pu, encoder=_TinyMLPEncoder(), seed=42)
    torch.manual_seed(123)
    clf2 = _fit_encoder_clf(X, y_pu, encoder=_TinyMLPEncoder(), seed=42)
    for (k1, v1), (k2, v2) in zip(
        clf1.model_.state_dict().items(), clf2.model_.state_dict().items(), strict=True
    ):
        assert k1 == k2
        assert torch.equal(v1, v2)
