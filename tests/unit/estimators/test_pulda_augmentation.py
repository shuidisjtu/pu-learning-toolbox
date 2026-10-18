"""Train-only fresh image views, before MixUp and normalization, for issue 119."""

import copy
import pickle

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox import PULDAClassifier  # noqa: E402
from pu_toolbox.core.exceptions import NotFittedError  # noqa: E402

pytestmark = pytest.mark.unit


class RecordingTransform(torch.nn.Module):
    def __init__(self, output="valid"):
        super().__init__()
        self.output = output
        self.calls = []

    def forward(self, inputs):
        self.calls.append(inputs.detach().clone())
        if self.output == "shape":
            return inputs[:, :, :1]
        if self.output == "dtype":
            return inputs.double()
        if self.output == "nan":
            return inputs * float("nan")
        if self.output == "not_tensor":
            return inputs.cpu().numpy()
        return inputs.add_(0.125)


def inputs():
    images = np.linspace(0, 0.5, 12 * 3 * 4 * 4, dtype=np.float32).reshape(12, 3, 4, 4)
    return images, np.r_[np.ones(4, dtype=int), np.zeros(8, dtype=int)]


def classifier(transform):
    return PULDAClassifier(
        class_prior=0.4,
        encoder=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()),
        train_transform=transform,
        hidden_dim=4,
        depth=1,
        warmup_epochs=1,
        pu_epochs=1,
        positive_batch_size=2,
        unlabeled_batch_size=4,
        random_state=3,
        device="cpu",
    )


def test_basic_transform_is_fresh_each_batch_both_stages_and_before_mixup(monkeypatch):
    features, labels = inputs()
    original = features.copy()
    source = RecordingTransform()
    estimator = classifier(source)
    forwards = []
    original_train = estimator._train_epoch

    def capture(*args, **kwargs):
        handle = estimator.model_.register_forward_pre_hook(
            lambda module, args: forwards.append(args[0].detach().clone())
        )
        try:
            return original_train(*args, **kwargs)
        finally:
            handle.remove()

    monkeypatch.setattr(estimator, "_train_epoch", capture)
    estimator.fit(features, labels, os_or_ts="ts")
    calls = estimator.train_transform_.calls
    assert len(calls) == 4 and not source.calls
    assert estimator.history_["phase"] == ["warmup", "pu_mixup"]
    # Two warmup forwards; each PU step has original + mixed forwards.
    assert len(forwards) == 6
    for call, forward in zip(calls, [forwards[i] for i in (0, 1, 2, 4)], strict=True):
        torch.testing.assert_close(forward, call + 0.125)
    assert not torch.equal(calls[0], calls[1])
    np.testing.assert_array_equal(features, original)
    assert estimator.calibration_applied_ and estimator.n_loss_unlabeled_ == 12
    # Inference, validation/test and initial pseudo-label scanning never augment.
    estimator.decision_function(features)
    estimator.predict_proba(features[:3])
    assert len(estimator.train_transform_.calls) == 4


@pytest.mark.parametrize("output", ["shape", "dtype", "nan", "not_tensor"])
def test_param_invalid_transform_output_invalidates_previous_fit(output):
    features, labels = inputs()
    estimator = classifier(RecordingTransform()).fit(features, labels)
    estimator.train_transform = RecordingTransform(output)
    with pytest.raises(ValueError, match="train_transform must return"):
        estimator.fit(features, labels)
    with pytest.raises(NotFittedError):
        estimator.predict(features)


def test_edge_transform_rejects_tabular_or_nonmodule_and_does_not_mutate_template():
    features, labels = inputs()
    with pytest.raises(ValueError, match="4-D"):
        classifier(RecordingTransform()).fit(features.reshape(12, -1), labels)
    with pytest.raises(TypeError, match="train_transform"):
        classifier("not_module").fit(features, labels)


def test_determ_clone_pickle_and_disabled_transform_preserve_legacy_path():
    features, labels = inputs()
    template = classifier(RecordingTransform())
    first = clone(template).fit(features, labels)
    second = clone(template).fit(features, labels)
    np.testing.assert_array_equal(
        first.decision_function(features), second.decision_function(features)
    )
    assert not template.train_transform.calls
    assert first.train_transform_ is not second.train_transform_
    np.testing.assert_array_equal(
        pickle.loads(pickle.dumps(first)).decision_function(features),
        first.decision_function(features),
    )
    plain = classifier(None).fit(features, labels)
    assert plain.train_transform_ is None
    assert plain.optimizer_steps_ == first.optimizer_steps_ == 4
    assert np.isfinite(copy.deepcopy(plain).decision_function(features)).all()
