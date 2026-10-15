# ruff: noqa: N803, N806
"""VPU injected CNN, bounded streaming, diagnostic and snapshot contracts."""

import copy
import math
import pickle

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox.core.exceptions import NotFittedError, ValidationError  # noqa: E402
from pu_toolbox.estimators.risk.vpu import (  # noqa: E402
    VPUClassifier,
    _stream_validation_risk,
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
        return self.conv(X).tanh().mean(dim=(2, 3))


def data(images=True):
    shape = (17, 1, 4, 4) if images else (17, 3)
    return np.random.RandomState(8).normal(size=shape).astype("float32"), np.r_[
        np.ones(6), np.zeros(11)
    ]


def model(**kwargs):
    params = dict(
        hidden_dim=4,
        depth=1,
        max_epochs=2,
        batch_size=4,
        iterations_per_epoch=2,
        random_state=2,
        device="cpu",
    )
    params.update(kwargs)
    return VPUClassifier(**params)


def test_basic_cnn_is_copied_trained_and_bounded_without_flattening_images():
    encoder = ImageEncoder()
    before = copy.deepcopy(encoder.state_dict())
    X, y = data()
    fitted = model(encoder=encoder).fit(X, y, pu_validation_data=(X, y))
    assert fitted.encoder_ is not encoder and fitted.model_[0] is fitted.encoder_
    assert fitted.data_device_policy_ == "cpu_stream" and fitted.training_data_device_ == "cpu"
    assert fitted.input_shape_ == (1, 4, 4)
    assert any(not torch.equal(before[k], fitted.encoder_.state_dict()[k]) for k in before)
    assert encoder.shapes == []
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    assert fitted.predict_proba(X).shape == (17, 2)
    assert all(1 <= shape[0] <= 4 and shape[1:] == (1, 4, 4) for shape in fitted.encoder_.shapes)
    assert all(parameter.requires_grad for parameter in fitted.encoder_.parameters())
    assert all(not parameter.requires_grad for parameter in fitted.model_[-1].parameters())


@pytest.mark.parametrize("view,pool_size", [("os", 11), ("ts", 17)])
def test_basic_streamed_view_preserves_pools_and_full_validation_diagnostic(view, pool_size):
    X, y = data()
    fitted = model(encoder=ImageEncoder()).fit(X, y, os_or_ts=view, pu_validation_data=(X, y))
    assert fitted.n_loss_unlabeled_ == pool_size and fitted.n_positive_ == 6
    with torch.no_grad():
        phi = fitted.model_[:-1]
        marginal = phi(torch.from_numpy(X)).flatten().double()
        positive = phi(torch.from_numpy(X[y == 1])).flatten().double()
        expected = float(torch.logsumexp(marginal, 0) - math.log(len(X)) - positive.mean())
        pool = X if view == "ts" else X[y == 0]
        expected_max = max(
            float(phi(torch.from_numpy(pool[i : i + 4])).max()) for i in range(0, len(pool), 4)
        )
    assert fitted.history_["val_risk"][-1] == pytest.approx(expected, abs=2e-7)
    assert fitted.max_log_phi_ == expected_max


def test_edge_stream_global_logsumexp_handles_unequal_chunks_and_extreme_scores():
    marginal = torch.tensor([[-10000.0], [-1.0], [-3.0], [-2000.0], [-2.0]])
    positive = marginal[[0, 1, 4]]
    expected = float(
        torch.logsumexp(marginal.double().flatten(), 0) - math.log(5) - positive.double().mean()
    )
    for batch in (1, 2, 4, 100):
        actual = _stream_validation_risk(torch.nn.Identity(), marginal, positive, batch, "cpu")
        assert actual == pytest.approx(expected, abs=1e-12)


def test_determ_streamed_features_keep_training_scores_and_resident_defaults_exact():
    X, y = data(images=False)
    resident = model().fit(X, y, pu_validation_data=(X, y))
    streamed = model(data_device_policy="cpu_stream").fit(X, y, pu_validation_data=(X, y))
    explicit = model(data_device_policy="resident").fit(X, y, pu_validation_data=(X, y))
    assert resident.data_device_policy_ == "resident"
    assert resident.history_ == explicit.history_
    for key in resident.history_:
        if key != "val_risk":
            assert resident.history_[key] == streamed.history_[key]
    np.testing.assert_array_equal(resident.decision_function(X), streamed.decision_function(X))
    np.testing.assert_allclose(
        resident.history_["val_risk"], streamed.history_["val_risk"], atol=5e-7
    )


def test_determ_cnn_clone_pickle_and_checkpoint_replay_keep_scores_and_chunks(tmp_path):
    X, y = data()
    template = model(encoder=ImageEncoder())
    first = clone(template).fit(X, y)
    second = clone(template).fit(X, y, class_prior=0.7)
    assert first.history_ == second.history_
    np.testing.assert_array_equal(first.decision_function(X), second.decision_function(X))
    restored = pickle.loads(pickle.dumps(first))
    np.testing.assert_array_equal(first.decision_function(X), restored.decision_function(X))
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(clone(template), X, y)
    checkpoint = trajectory.checkpoints[-1]
    assert checkpoint.reference()["prediction_batch_size"] == 4
    assert not checkpoint.reference()["training_resume_supported"]
    replay = checkpoint.restore(device="cpu")
    replay.model_[0].shapes.clear()
    np.testing.assert_array_equal(
        replay.decision_function(X), trajectory.model.decision_function(X)
    )
    assert [shape[0] for shape in replay.model_[0].shapes] == [4, 4, 4, 4, 1]


@pytest.mark.parametrize("policy", [None, [], "gpu_all"])
def test_param_invalid_data_policy_fails_loud(policy):
    with pytest.raises(ValueError, match="data_device_policy"):
        model(data_device_policy=policy).fit(*data(images=False))


def test_param_images_require_encoder_and_encoder_requires_streaming():
    with pytest.raises(ValidationError, match="2-D"):
        model().fit(*data())
    with pytest.raises(TypeError, match="torch.nn.Module"):
        model(encoder="cnn").fit(*data())
    with pytest.raises(ValueError, match="CPU-streamed"):
        model(encoder=ImageEncoder(), data_device_policy="resident").fit(*data())
    with pytest.raises(ValueError, match="2-D"):
        model(encoder=torch.nn.Identity()).fit(*data())


def test_param_validation_shape_and_clean_view_are_rejected():
    X, y = data()
    with pytest.raises(ValueError, match="input shape"):
        model(encoder=ImageEncoder()).fit(X, y, pu_validation_data=(X[:, :, :3], y))
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="PU view"):
        model(encoder=ImageEncoder()).fit(
            X, y, pu_validation_data=SimpleNamespace(view="clean", X=X, labels=y)
        )


def test_edge_empty_prediction_and_wrong_image_shape():
    fitted = model(encoder=ImageEncoder()).fit(*data())
    assert fitted.predict_proba(np.empty((0, 1, 4, 4))).shape == (0, 2)
    with pytest.raises(ValueError, match="input shape"):
        fitted.predict(np.zeros((1, 1, 3, 4)))


def test_edge_late_callback_failure_invalidates_fitted_state_after_predictable_epoch():
    X, y = data()
    fitted = model(encoder=ImageEncoder())
    completed = []

    def callback(epoch, current):
        assert current.predict(X).shape == (17,)
        completed.append(epoch)
        if epoch == 1:
            raise RuntimeError("callback failure")

    with pytest.raises(RuntimeError, match="callback failure"):
        fitted.fit(X, y, epoch_callback=callback)
    assert completed == [0, 1]
    with pytest.raises(NotFittedError):
        fitted.predict(X)


def test_basic_public_pipeline_accepts_injected_cnn(monkeypatch):
    from pu_toolbox.estimators.deep import vision
    from pu_toolbox.workflows import PUPipeline

    monkeypatch.setattr(vision, "build_encoder", lambda *args, **kwargs: ImageEncoder())

    X, y = data()
    report = PUPipeline(
        classifier="vpu",
        prior_estimator=None,
        cv=2,
        device="cpu",
        architecture="cnn",
        classifier_params=dict(max_epochs=1, hidden_dim=4, iterations_per_epoch=1, batch_size=4),
    ).fit_evaluate(X, y)
    assert report.final_model.predict(X).shape == (17,)
    assert report.final_model.data_device_policy_ == "cpu_stream"
