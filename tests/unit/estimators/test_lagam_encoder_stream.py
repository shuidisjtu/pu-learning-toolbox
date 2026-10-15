# ruff: noqa: N803, N806
"""LaGAM image streaming, virtual-step isolation and clean-support guards."""

import copy
import inspect
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox.core.exceptions import NotFittedError, ValidationError  # noqa: E402
from pu_toolbox.estimators.deep.lagam import (  # noqa: E402
    LaGAMClassifier,
    _virtual_features,
)
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer  # noqa: E402

pytestmark = pytest.mark.unit


class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 3, padding=1)
        self.shapes = []

    def forward(self, X):
        self.shapes.append(tuple(X.shape))
        return self.conv(X).tanh().mean((2, 3))


class SupportBound:
    """Test-only explicit support bridge; production trainer stays fail-closed."""

    def __init__(self, estimator, support):
        self.estimator, self.support = estimator, support

    def fit(self, X, y, *, epoch_callback=None):
        self.estimator.fit(X, y, support_data=self.support, epoch_callback=epoch_callback)
        return self


def data(images=True):
    shape = (17, 1, 4, 4) if images else (17, 3)
    support_shape = (8, *shape[1:])
    rng = np.random.RandomState(7)
    X = rng.normal(size=shape).astype("float32")
    support = rng.normal(size=support_shape).astype("float32"), np.array([1, 0] * 4)
    return X, np.r_[np.ones(6), np.zeros(11)], support


def model(**kwargs):
    params = dict(
        hidden_dim=4,
        warmup_epochs=1,
        max_epochs=3,
        batch_size=4,
        support_batch_size=3,
        num_clusters=2,
        random_state=3,
        device="cpu",
    )
    params.update(kwargs)
    return LaGAMClassifier(**params)


def test_basic_cnn_is_copied_trained_and_streams_data_pseudo_labels_and_groups():
    X, y, support = data()
    encoder = ImageEncoder()
    before = copy.deepcopy(encoder.state_dict())
    fitted = model(encoder=encoder).fit(X, y, support_data=support)
    assert fitted.encoder_ is not encoder and fitted.model_[0] is fitted.encoder_
    assert fitted.data_device_policy_ == "cpu_stream" and fitted.training_data_device_ == "cpu"
    assert fitted.pseudo_labels_.device.type == "cpu"
    assert any(not torch.equal(before[k], fitted.encoder_.state_dict()[k]) for k in before)
    assert encoder.shapes == []
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert fitted.predict(X).shape == (17,)
    assert all(0 < shape[0] <= 4 and shape[1:] == (1, 4, 4) for shape in fitted.encoder_.shapes)
    assert torch.all(fitted.pseudo_labels_[y == 1] == 1)
    assert fitted.optimizer_steps_ == 15 and fitted.virtual_steps_ == 10
    assert fitted.clustering_passes_ == 2 and fitted.checkpoint_epoch_count == 3


def test_basic_virtual_encoder_eval_preserves_bn_buffers_rng_and_mixed_modes():
    encoder = torch.nn.Sequential(
        torch.nn.Conv2d(1, 3, 1),
        torch.nn.BatchNorm2d(3),
        torch.nn.Dropout2d(0.5),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(1),
    )
    network = torch.nn.Sequential(encoder, torch.nn.Linear(3, 1)).train()
    encoder[2].eval()
    modes = [m.training for m in network.modules()]
    buffers = copy.deepcopy(network.state_dict())
    X = torch.ones(4, 1, 4, 4)
    rng = torch.get_rng_state().clone()
    first, second = _virtual_features(network, X, -X)
    assert first.shape == second.shape == (4, 3)
    assert not first.requires_grad and not second.requires_grad
    assert modes == [m.training for m in network.modules()]
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in network.state_dict().items():
        torch.testing.assert_close(value, buffers[key], atol=0, rtol=0)


def test_edge_virtual_feature_failure_restores_all_modes():
    network = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Linear(3, 1)).train()
    modes = [m.training for m in network.modules()]
    with pytest.raises(ValueError, match="2-D"):
        _virtual_features(network, torch.ones(2, 1, 4, 4), torch.ones(2, 1, 4, 4))
    assert modes == [m.training for m in network.modules()]


@pytest.mark.parametrize("warmup", [0, 1, 2])
def test_basic_manual_support_snapshots_record_actual_steps_and_bounded_replay(tmp_path, warmup):
    X, y, support = data()
    fitted = model(encoder=ImageEncoder(), warmup_epochs=warmup)
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(
        SupportBound(fitted, support), X, y
    )
    assert len(trajectory.checkpoints) == fitted.checkpoint_epoch_count == 3
    assert [c["stage"] for c in fitted.epoch_provenance_] == ["warmup"] * warmup + ["meta"] * (
        3 - warmup
    )
    assert fitted.optimizer_steps_ == 15 and fitted.virtual_steps_ == (3 - warmup) * 5
    assert fitted.clustering_passes_ == 3 - warmup
    for checkpoint, context in zip(trajectory.checkpoints, fitted.epoch_provenance_, strict=True):
        assert checkpoint.reference()["training_context"] == context
        assert checkpoint.reference()["prediction_batch_size"] == 4
        assert not checkpoint.reference()["training_resume_supported"]
    restored = trajectory.checkpoints[-1].restore(device="cpu")
    restored.model_[0].shapes.clear()
    np.testing.assert_array_equal(restored.decision_function(X), fitted.decision_function(X))
    assert [s[0] for s in restored.model_[0].shapes] == [4, 4, 4, 4, 1]


def test_determ_legacy_resident_and_feature_stream_training_remain_equal():
    X, y, support = data(images=False)
    resident = model().fit(X, y, support_data=support)
    streamed = model(data_device_policy="cpu_stream").fit(X, y, support_data=support)
    explicit = model(data_device_policy="resident").fit(X, y, support_data=support)
    assert resident.history_ == explicit.history_ == streamed.history_
    np.testing.assert_array_equal(resident.pseudo_labels_.numpy(), streamed.pseudo_labels_.numpy())
    np.testing.assert_allclose(
        resident.decision_function(X), streamed.decision_function(X), atol=1e-7
    )


def test_determ_cnn_clone_pickle_and_unused_prior_keep_scores():
    X, y, support = data()
    template = model(encoder=ImageEncoder(), meta_update_bias=False)
    first = clone(template).fit(X, y, support_data=support)
    second = clone(template).fit(X, y, support_data=support, class_prior=0.8)
    assert first.history_ == second.history_
    np.testing.assert_array_equal(first.decision_function(X), second.decision_function(X))
    restored = pickle.loads(pickle.dumps(first))
    np.testing.assert_array_equal(first.decision_function(X), restored.decision_function(X))


@pytest.mark.parametrize(
    "params",
    [
        {"data_device_policy": None},
        {"data_device_policy": []},
        {"data_device_policy": "all_gpu"},
        {"meta_update_bias": 1},
        {"meta_update_bias": None},
    ],
)
def test_param_bad_stream_or_virtual_scope_fails(params):
    X, y, support = data(images=False)
    with pytest.raises(ValueError):
        model(**params).fit(X, y, support_data=support)


def test_param_image_encoder_support_shapes_and_alias_rejection():
    X, y, support = data()
    with pytest.raises(ValidationError, match="2-D"):
        model().fit(X, y, support_data=support)
    with pytest.raises(TypeError, match="torch.nn.Module"):
        model(encoder="cnn").fit(X, y, support_data=support)
    with pytest.raises(ValueError, match="CPU-streamed"):
        model(encoder=ImageEncoder(), data_device_policy="resident").fit(X, y, support_data=support)
    with pytest.raises(ValueError, match="feature shape"):
        model(encoder=ImageEncoder()).fit(X, y, support_data=(support[0][:, :, :3], support[1]))
    with pytest.raises(ValueError, match="independent"):
        model(encoder=ImageEncoder()).fit(X, y, support_data=(X[:8], np.array([1, 0] * 4)))
    with pytest.raises(ValueError, match="2-D"):
        model(encoder=torch.nn.Identity()).fit(X, y, support_data=support)


def test_edge_minimal_support_empty_prediction_and_no_contrastive_clustering():
    X, y, support = data()
    fitted = model(encoder=ImageEncoder(), max_epochs=1, warmup_epochs=0, contrastive_weight=0).fit(
        X, y, support_data=(support[0][:2], support[1][:2])
    )
    assert fitted.n_support_ == 2 and fitted.clustering_passes_ == 0
    assert fitted.virtual_steps_ == fitted.optimizer_steps_ == 5
    assert fitted.predict(np.empty((0, 1, 4, 4))).shape == (0,)
    with pytest.raises(ValueError, match="shape"):
        fitted.predict(np.zeros((1, 1, 3, 4)))


def test_edge_late_callback_and_missing_support_refit_invalidate_success():
    X, y, support = data()
    fitted = model(encoder=ImageEncoder())
    completed = []

    def callback(epoch, current):
        assert current.predict(X).shape == (17,)
        completed.append(epoch)
        if epoch == 1:
            raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        fitted.fit(X, y, support_data=support, epoch_callback=callback)
    assert completed == [0, 1]
    with pytest.raises(NotFittedError):
        fitted.predict(X)
    fitted.fit(X, y, support_data=support)
    with pytest.raises(ValueError, match="clean support_data"):
        fitted.fit(X, y)
    with pytest.raises(NotFittedError):
        fitted.predict(X)


def test_basic_cnn_support_still_excluded_from_ui_pu_only_and_standard_trainer():
    from pu_toolbox.ui.parameters import classifier_catalog, cnn_candidates

    entry = next(c for c in classifier_catalog() if c["name"] == "lagam")
    assert "lagam" in cnn_candidates() and not entry["ui_ready"]
    assert entry["requires_clean_support"]
    X, y, _ = data()
    with pytest.raises(ValueError, match="PA-ineligible"):
        EpochCheckpointTrainer().fit(model(encoder=ImageEncoder()), X, y)
    assert "os_or_ts" not in inspect.signature(LaGAMClassifier.fit).parameters


def test_basic_source_receipt_keeps_sampling_conflict_and_unsigned_budget():
    root = Path(__file__).resolve().parents[3]
    receipt = json.loads(
        (root / "docs/research/pu_survey/data/lagam_source_review_20261015.json").read_text(
            encoding="utf-8"
        )
    )
    for name, value in receipt["toolbox_defaults"].items():
        assert inspect.signature(LaGAMClassifier).parameters[name].default == value
    assert receipt["source"]["commit"] == "362a7f41fcf3d4e4161fe99f4aab4e6128b6a5d8"
    assert receipt["sampling"]["published_split_view"] == "os"
    assert receipt["sampling"]["existing_registry_view"] == "ts"
    assert receipt["sampling"]["decision"] == "pending_method_owner_review"
    assert not receipt["formal_admission"] and not receipt["calibration_applied"]
    assert receipt["owner_signatures"] == [] and receipt["candidate_protocol_ref"] is None
    from pu_toolbox.experiment.method_ledger import load_ledger

    review = load_ledger()["methods"]["lagam"]["sampling_review"]
    assert review == dict(
        registry_view="ts",
        source_split_view="os",
        review_status="pending_method_owner_review",
        registry_relabelled=False,
    )
