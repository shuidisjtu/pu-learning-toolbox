"""PAN end-to-end C/D encoders, fold-safe copies and inference snapshots."""

import copy
import os
import pickle

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.deep.pan import PANClassifier  # noqa: E402
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer  # noqa: E402

pytestmark = pytest.mark.unit


def images():
    values = np.random.RandomState(5).normal(size=(24, 3, 4, 4)).astype("float32")
    return values, np.r_[np.ones(8, int), np.zeros(16, int)]


def encoder():
    torch.manual_seed(11)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 3, 1),
        torch.nn.BatchNorm2d(3),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
    )


def model(**kwargs):
    params = dict(encoder=encoder(), max_epochs=2, batch_size=8, random_state=3, device="cpu")
    params.update(kwargs)
    return PANClassifier(**params)


def test_basic_native_cnn_trains_both_independent_encoders_not_source():
    features, labels = images()
    source = encoder().requires_grad_(False)
    before = copy.deepcopy(source.state_dict())
    fitted = model(encoder=source).fit(features, labels)
    assert fitted.encoder_ is not fitted.discriminator_encoder_
    assert fitted.encoder_ is not source
    for own in (fitted.encoder_, fitted.discriminator_encoder_):
        assert own[0].weight.requires_grad
        assert not torch.equal(own[0].weight, before["0.weight"])
    assert (
        fitted.encoder_[0].weight.data_ptr() != fitted.discriminator_encoder_[0].weight.data_ptr()
    )
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert not source[0].weight.requires_grad
    assert fitted.optimizer_steps_ == 8
    # Only active passes may change BN statistics; probe and inactive reward
    # forwards run in eval mode. C has one, D has two active passes per step.
    assert fitted.encoder_[1].num_batches_tracked.item() == 4
    assert fitted.discriminator_encoder_[1].num_batches_tracked.item() == 8


def test_seed_determinism_clone_full_pickle_and_weights_only_epoch_recovery(tmp_path):
    features, labels = images()
    fitted = clone(model())
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(fitted, features, labels)
    expected = fitted.decision_function(features)
    np.testing.assert_array_equal(
        clone(model()).fit(features, labels).decision_function(features), expected
    )
    np.testing.assert_array_equal(
        pickle.loads(pickle.dumps(fitted)).decision_function(features), expected
    )
    assert len(trajectory.checkpoints) == 2
    np.testing.assert_array_equal(
        trajectory.checkpoints[-1].restore(device="cpu").decision_function(features), expected
    )
    assert trajectory.checkpoints[-1].reference()["training_resume_supported"] is False
    assert fitted.predict_proba(features).shape == (24, 2)


def test_edge_empty_prediction_and_wrong_image_shape_fail_closed():
    features, labels = images()
    fitted = model().fit(features, labels)
    assert fitted.decision_function(features[:0]).shape == (0,)
    assert fitted.predict_proba(features[:0]).shape == (0, 2)
    with pytest.raises(ValueError, match="input shape"):
        fitted.predict(features[:, :, :2])
    with pytest.raises(ValueError, match="encoder"):
        model(encoder=None).fit(features, labels)
    with pytest.raises(ValueError, match="2-D features or 4-D"):
        model().fit(features[:, 0], labels)


@pytest.mark.parametrize("bad_encoder", ["not_module", torch.nn.Flatten(start_dim=0)])
def test_param_invalid_encoder_contract_is_refused(bad_encoder):
    features, labels = images()
    with pytest.raises((TypeError, ValueError), match="encoder"):
        model(encoder=bad_encoder).fit(features, labels)


def test_basic_cnn_custom_score_heads_and_prediction_preserve_training_modes():
    features, labels = images()
    classifier, discriminator = torch.nn.Linear(3, 1), torch.nn.Linear(3, 1)
    before = copy.deepcopy(classifier.state_dict()), copy.deepcopy(discriminator.state_dict())
    modes = []

    def callback(epoch, fitted):
        modes.append((fitted.model_.training, fitted.discriminator_.training))
        assert np.isfinite(fitted._decision_function(features)).all()
        assert (fitted.model_.training, fitted.discriminator_.training) == modes[-1]

    model(model=classifier, discriminator=discriminator).fit(
        features, labels, epoch_callback=callback
    )
    assert modes == [(True, True), (True, True)]
    for module, saved in zip((classifier, discriminator), before, strict=True):
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, saved[key], rtol=0, atol=0)


@pytest.mark.gpu
def test_cuda_native_cnn_checkpoint_roundtrip(tmp_path):
    if not torch.cuda.is_available():
        if os.environ.get("PU_REQUIRE_CUDA") == "1":
            pytest.fail("PU_REQUIRE_CUDA=1 requires CUDA")
        pytest.skip("CUDA unavailable")
    features, labels = images()
    fitted = model(max_epochs=1, device="cuda")
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(fitted, features, labels)
    assert next(fitted.encoder_.parameters()).is_cuda
    assert next(fitted.discriminator_encoder_.parameters()).is_cuda
    np.testing.assert_allclose(
        trajectory.checkpoints[-1].restore(device="cpu").decision_function(features),
        fitted.decision_function(features),
        atol=1e-5,
    )
