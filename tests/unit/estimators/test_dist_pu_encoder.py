"""Opt-in Dist-PU encoder training, bounded forwards and legacy MLP compatibility."""

# ruff: noqa: E402, N803, N806

import pickle

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox.core.exceptions import NotFittedError, ValidationError
from pu_toolbox.estimators.risk.dist_pu import DistPUClassifier

pytestmark = pytest.mark.unit


class RecordingEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.bn = torch.nn.BatchNorm1d(4)
        self.dropout = torch.nn.Dropout(0.2)
        self.calls = []
        self.first_training_weight = None

    def forward(self, inputs):
        self.calls.append(
            (
                len(inputs),
                inputs.device.type,
                self.training,
                self.bn.training,
                torch.is_grad_enabled(),
            )
        )
        if self.training and self.first_training_weight is None:
            self.first_training_weight = self.conv.weight.detach().clone()
        features = self.pool(self.conv(inputs)).flatten(1)
        return self.dropout(self.bn(features))


def data(n=13):
    features = np.random.RandomState(47).normal(size=(n, 3, 4, 4)).astype(np.float32)
    return features, np.r_[np.ones(4, dtype=int), np.zeros(n - 4, dtype=int)]


def estimator(encoder, **overrides):
    parameters = dict(
        hidden_dim=4,
        batch_size=4,
        distribution_loss="author_histogram",
        training_recipe="author_staged",
        warmup_epochs=1,
        pu_epochs=1,
        random_state=17,
        device="cpu",
        encoder=encoder,
    )
    parameters.update(overrides)
    return DistPUClassifier(0.4, **parameters)


@pytest.mark.parametrize("view", ["os", "ts"])
def test_basic_encoder_updates_both_stages_streams_and_retains_singleton_tail(view):
    features, labels = data()
    original_features, original_labels = features.copy(), labels.copy()
    template = RecordingEncoder().requires_grad_(False)
    before = {key: value.clone() for key, value in template.state_dict().items()}
    modes = [module.training for module in template.modules()]
    model = estimator(template).fit(features, labels, os_or_ts=view)

    assert model.encoder_ is model.model_[0] and model.encoder_ is not template
    assert all(parameter.requires_grad for parameter in model.encoder_.parameters())
    assert all(not parameter.requires_grad for parameter in template.parameters())
    assert model.optimizer_steps_ == 8 and model.epochs_completed_ == 2
    assert [record["stage"] for record in model.epoch_provenance_] == ["warmup", "dist_pu"]
    assert model.encoder_.bn.num_batches_tracked.item() == 9
    train_singletons = [c for c in model.encoder_.calls if c[0] == 1 and c[2] and c[4]]
    assert len(train_singletons) == 3 and all(not call[3] for call in train_singletons)
    assert not model.encoder_.calls[0][2]  # shape probe is eval-only
    assert model.encoder_.calls[0][0] == 1
    assert any(
        not torch.equal(parameter.detach(), before[name])
        for name, parameter in model.encoder_.named_parameters()
    )
    for name, value in template.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert [module.training for module in template.modules()] == modes
    np.testing.assert_array_equal(features, original_features)
    np.testing.assert_array_equal(labels, original_labels)
    assert model.pseudo_labels_.device.type == "cpu" and model.pseudo_labels_.shape == (13,)
    assert model.training_provenance_["pseudo_label_inference_batch_size"] == 4
    assert model.training_provenance_["maximum_rows_per_device_forward"] == 4
    assert model.training_provenance_["implementation_path"] == "author_staged_encoder_adapter"
    assert model.training_provenance_["calibration_applied"] is (view == "ts")
    probabilities = model.predict_proba(features)
    assert probabilities.shape == (13, 2) and np.isfinite(probabilities).all()
    assert max(call[0] for call in model.encoder_.calls) <= 4
    assert all(call[1] == "cpu" for call in model.encoder_.calls)


def test_determ_fits_clone_template_independently_and_reproduce_scores():
    features, labels = data()
    template = RecordingEncoder()
    first = estimator(template).fit(features, labels)
    second = clone(estimator(template)).fit(features, labels)
    assert first.encoder_ is not second.encoder_
    torch.testing.assert_close(
        first.encoder_.first_training_weight, second.encoder_.first_training_weight, rtol=0, atol=0
    )
    np.testing.assert_array_equal(first.predict_proba(features), second.predict_proba(features))
    assert first.loss_history_ == second.loss_history_
    original = second.encoder_.conv.weight.detach().clone()
    with torch.no_grad():
        first.encoder_.conv.weight.add_(1)
    torch.testing.assert_close(second.encoder_.conv.weight, original, rtol=0, atol=0)


def test_determ_ts_calibration_keeps_physical_batches_and_rng_calls_unchanged(monkeypatch):
    from pu_toolbox.estimators.risk import dist_pu as module

    features, labels = data()
    template = RecordingEncoder()
    original = module._author_histogram_batch_loss
    records = []

    def record(logits, observed, prior, **kwargs):
        records.append(
            (observed.detach().cpu().numpy().copy(), kwargs["include_positive_in_unlabeled"])
        )
        return original(logits, observed, prior, **kwargs)

    monkeypatch.setattr(module, "_author_histogram_batch_loss", record)
    os_model = estimator(template).fit(features, labels, os_or_ts="os")
    os_records, records = records, []
    os_rng = torch.get_rng_state().clone()
    ts_model = estimator(template).fit(features, labels, os_or_ts="ts")
    assert len(records) == len(os_records) == 8
    for (os_labels, os_flag), (ts_labels, ts_flag) in zip(os_records, records, strict=True):
        np.testing.assert_array_equal(os_labels, ts_labels)
        assert os_flag is False and ts_flag is True
    assert os_model.encoder_.calls == ts_model.encoder_.calls
    torch.testing.assert_close(torch.get_rng_state(), os_rng, rtol=0, atol=0)
    assert os_model.optimizer_steps_ == ts_model.optimizer_steps_
    assert not np.array_equal(os_model.predict_proba(features), ts_model.predict_proba(features))


def test_basic_2d_encoder_path_is_explicit_and_trainable():
    features, labels = data()
    features = features.mean(axis=(2, 3))
    encoder = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU())
    model = estimator(encoder).fit(features, labels)
    assert model.input_shape_ == (3,)
    assert model.predict(features).shape == (len(features),)


def test_basic_real_cnn13_encoder_trains_on_cpu():
    from pu_toolbox import build_encoder

    features = np.random.RandomState(49).normal(size=(4, 3, 32, 32)).astype(np.float32)
    labels = np.array([1, 1, 0, 0])
    template = build_encoder("cnn", backbone="cnn13", in_channels=3)
    before = next(template.parameters()).detach().clone()
    model = estimator(template, batch_size=2).fit(features, labels)
    assert not torch.equal(next(model.encoder_.parameters()).detach(), before)
    torch.testing.assert_close(next(template.parameters()).detach(), before, rtol=0, atol=0)
    assert np.isfinite(model.predict_proba(features)).all()


def test_edge_inplace_encoder_cannot_mutate_caller_arrays_during_probe_or_scoring():
    features, labels = data()
    before = features.copy()
    encoder = torch.nn.Sequential(
        torch.nn.ReLU(inplace=True), torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 4)
    )
    model = estimator(encoder).fit(features, labels)
    np.testing.assert_array_equal(features, before)
    model.predict_proba(features)
    np.testing.assert_array_equal(features, before)


@pytest.mark.parametrize(
    "overrides,exception,message",
    [
        ({"training_recipe": "clean_room"}, ValueError, "encoder requires"),
        ({"distribution_loss": "expectation"}, ValueError, "requires distribution_loss"),
        ({"encoder": "cnn"}, TypeError, "torch.nn.Module"),
    ],
)
def test_param_encoder_requires_explicit_staged_recipe_before_rng(overrides, exception, message):
    features, labels = data()
    model = estimator(**{"encoder": RecordingEncoder(), **overrides})
    before = torch.get_rng_state().clone()
    with pytest.raises(exception, match=message):
        model.fit(features, labels)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    assert model._is_fitted is False


@pytest.mark.parametrize("bad_output", ["rank", "empty", "nan"])
def test_param_invalid_encoder_features_are_rejected_without_mutating_template(bad_output):
    class BadEncoder(torch.nn.Module):
        def forward(self, inputs):
            if bad_output == "rank":
                return inputs
            if bad_output == "empty":
                return inputs.new_empty((len(inputs), 0))
            return inputs.new_full((len(inputs), 3), float("nan"))

    features, labels = data()
    template = BadEncoder()
    with pytest.raises(ValueError, match="encoder"):
        estimator(template).fit(features, labels)
    assert template.training is True


def test_edge_images_need_encoder_and_full_sample_shape_is_checked():
    features, labels = data()
    with pytest.raises(ValidationError, match="2-D"):
        estimator(None).fit(features, labels)
    with pytest.raises(ValueError, match="2-D features or 4-D"):
        estimator(RecordingEncoder()).fit(features[:, 0], labels)
    model = estimator(RecordingEncoder()).fit(features, labels)
    with pytest.raises(ValueError, match="sample shape"):
        model.predict_proba(np.zeros((2, 3, 5, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="sample shape"):
        model.predict_proba(np.zeros((0, 1, 4, 4), dtype=np.float32))


@pytest.mark.parametrize("failure_stage", ["eval", "forward"])
def test_edge_scoring_restores_modes_even_after_partial_failure(monkeypatch, failure_stage):
    features, labels = data()
    model = estimator(RecordingEncoder()).fit(features, labels)
    model.model_.train()
    model.encoder_.bn.eval()
    before = [module.training for module in model.model_.modules()]
    failure = RuntimeError("Dist-PU inference interrupted")

    def fail(*_args):
        model.encoder_.conv.training = False
        raise failure

    monkeypatch.setattr(model.model_, failure_stage, fail)
    with pytest.raises(RuntimeError) as caught:
        model.predict_proba(features)
    assert caught.value is failure
    assert [module.training for module in model.model_.modules()] == before


def test_basic_empty_scoring_pickle_and_checkpoint_replay(tmp_path, monkeypatch):
    from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer

    features, labels = data()
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(
        estimator(RecordingEncoder()), features, labels
    )
    model = trajectory.model
    assert [checkpoint.training_context["stage"] for checkpoint in trajectory.checkpoints] == [
        "warmup",
        "dist_pu",
    ]
    for checkpoint in trajectory.checkpoints:
        assert checkpoint.prediction_batch_size == 4
        assert np.isfinite(checkpoint.restore().decision_function(features)).all()
    np.testing.assert_array_equal(
        trajectory.checkpoints[-1].restore().decision_function(features),
        model.decision_function(features),
    )
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict_proba(features), model.predict_proba(features))

    def fail():
        raise AssertionError("empty prediction must not evaluate the model")

    monkeypatch.setattr(model.model_, "eval", fail)
    assert model.decision_function(features[:0]).shape == (0,)
    assert model.predict_proba(features[:0]).shape == (0, 2)


def test_edge_failed_refit_blocks_stale_probabilities_and_later_refit_recovers():
    features, labels = data()
    model = estimator(RecordingEncoder()).fit(features, labels)
    expected = model.predict_proba(features)
    with pytest.raises(ValueError, match="TS|os_or_ts"):
        model.fit(features, labels, os_or_ts="invalid")
    with pytest.raises(NotFittedError):
        model.predict_proba(features)
    model.fit(features, labels)
    np.testing.assert_array_equal(model.predict_proba(features), expected)
    with pytest.raises(NotFittedError):
        estimator(None).predict_proba(np.zeros((2, 3)))


def test_basic_legacy_mlp_pickle_without_encoder_attributes_still_predicts():
    features, labels = data()
    features = features.mean(axis=(2, 3))
    model = DistPUClassifier(0.4, hidden_dim=4, epochs=2, device="cpu").fit(features, labels)
    expected = model.predict_proba(features)
    del model.encoder
    del model.encoder_
    del model.input_shape_
    for name in (
        "distribution_loss",
        "training_recipe",
        "warmup_epochs",
        "pu_epochs",
        "warmup_learning_rate",
        "warmup_weight_decay",
        "pu_learning_rate",
        "pu_weight_decay",
        "warmup_entropy_weight",
        "pu_entropy_weight",
        "mixup_alpha",
        "mixup_entropy_weight",
        "mixup_bce_weight",
    ):
        delattr(model, name)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.encoder is None and restored.get_params()["encoder"] is None
    assert restored.get_params()["training_recipe"] == "clean_room"
    assert restored.epochs == 2
    np.testing.assert_array_equal(restored.predict_proba(features), expected)
