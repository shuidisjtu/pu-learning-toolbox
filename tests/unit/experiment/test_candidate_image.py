"""Issues 119/120: seeded shared vs private image paths without formal admission."""

import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.experiment.candidate_image import build_candidate_image_classifier  # noqa: E402
from pu_toolbox.experiment.feature_adapter import encoder_state_sha256  # noqa: E402
from pu_toolbox.experiment.image import (  # noqa: E402
    build_survey_image_encoder,
    fit_survey_image_preprocessing,
)

pytestmark = pytest.mark.unit


def preprocessing(augmentation="none"):
    images = np.random.RandomState(4).randint(0, 256, (8, 3, 8, 8)).astype(np.uint8)
    return fit_survey_image_preprocessing(
        images, dataset="cifar10", train_augmentation=augmentation
    )


def test_determ_shared_resnet_seed_is_rng_independent_and_normalizes_once():
    _, spec = preprocessing()
    torch.manual_seed(31)
    before = torch.get_rng_state().clone()
    first, record = build_candidate_image_classifier("pulda", spec, seed=7, class_prior=0.4)
    assert torch.equal(torch.get_rng_state(), before)
    torch.rand(31)
    second, second_record = build_candidate_image_classifier("pulda", spec, seed=7, class_prior=0.4)
    assert record == second_record
    assert first.encoder is not second.encoder
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        expected = build_survey_image_encoder(spec)
    assert encoder_state_sha256(expected) == record["encoder_initial_state_sha256"]
    assert encoder_state_sha256(first.encoder) == encoder_state_sha256(second.encoder)
    assert record["comparison_group"] == "end-to-end CNN"
    assert not record["formal_admission"] and record["approved_representation_ref"] is None
    assert not record["standard_resnet18_leaderboard_eligible"]
    norm = first.encoder[0]
    torch.testing.assert_close(norm.mean.flatten(), torch.tensor(spec.normalization_mean))
    assert record["image_preprocessing"]["backbone"]["first_layer"]["kernel_size"] == [3, 3]
    json.dumps(record, allow_nan=False)


def test_basic_pulda_real_resnet_updates_bn_and_encoder_not_shared_template():
    features, spec = preprocessing("simaugment")
    estimator, record = build_candidate_image_classifier("pulda", spec, seed=1, class_prior=0.5)
    before = copy.deepcopy(estimator.encoder.state_dict())
    estimator.set_params(
        warmup_epochs=1, pu_epochs=1, positive_batch_size=2, unlabeled_batch_size=2
    )
    estimator.fit(features, np.r_[np.ones(4, int), np.zeros(4, int)], os_or_ts="ts")
    assert record["batch_norm_policy"] == "train"
    assert estimator.train_transform_ is not None
    assert estimator.history_["phase"] == ["warmup", "pu_mixup"]
    assert encoder_state_sha256(estimator.encoder_) != record["encoder_initial_state_sha256"]
    for key, value in estimator.encoder.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    bn = [m for m in estimator.encoder_.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bn and all(m.num_batches_tracked.item() == 6 for m in bn)
    estimator.predict(features)
    assert all(m.num_batches_tracked.item() == 6 for m in bn)


def test_basic_gradpu_private_backbone_metadata_and_real_raw_image_fit():
    features, spec = preprocessing()
    estimator, record = build_candidate_image_classifier("gradpu", spec, seed=2)
    assert record["comparison_group"] == "method-private/no-BN CNN"
    assert record["image_preprocessing"]["backbone"]["name"] == "cnn13_no_bn"
    assert record["source_identity"] == "paper-only" and not record["source_reproduction"]
    assert record["resource_profile"]["status"] == "not_measured"
    assert not any(
        isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in estimator.encoder.modules()
    )
    before = encoder_state_sha256(estimator.encoder)
    estimator.set_params(max_epochs=1, batch_size=2, hidden_dim=4)
    estimator.fit(features, np.r_[np.ones(4, int), np.zeros(4, int)], os_or_ts="ts")
    assert estimator.calibration_applied_ and estimator.n_loss_unlabeled_ == 8
    assert estimator.encoder_ is not estimator.encoder
    assert encoder_state_sha256(estimator.encoder) == before
    assert encoder_state_sha256(estimator.encoder_) != before
    assert np.isfinite(estimator.decision_function(features)).all()


@pytest.mark.parametrize("seed", [True, -1, 2**32, 0.5, None])
def test_param_invalid_initialization_seed_is_rejected(seed):
    _, spec = preprocessing()
    with pytest.raises(ValueError, match="seed"):
        build_candidate_image_classifier("pulda", spec, seed=seed, class_prior=0.4)


def test_edge_no_silent_prior_augmentation_or_backbone_substitution():
    _, spec = preprocessing()
    with pytest.raises(ValueError, match="method"):
        build_candidate_image_classifier("nnpu", spec, seed=0)
    with pytest.raises(ValueError, match="population"):
        build_candidate_image_classifier("pulda", spec, seed=0)
    with pytest.raises(ValueError, match="does not use"):
        build_candidate_image_classifier("gradpu", spec, seed=0, class_prior=0.4)
    _, augmented = preprocessing("simaugment")
    with pytest.raises(ValueError, match="train_augmentation"):
        build_candidate_image_classifier("gradpu", augmented, seed=0)


@pytest.mark.parametrize(
    "prior", [True, False, 0, 1, -0.1, float("nan"), float("inf"), "0.4", np.array([0.4])]
)
def test_param_invalid_population_prior_is_refused_before_encoder_allocation(monkeypatch, prior):
    import pu_toolbox.experiment.candidate_image as module

    _, spec = preprocessing()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid prior must fail before constructing a backbone")

    monkeypatch.setattr(module, "build_survey_image_encoder", forbidden)
    with pytest.raises(ValueError, match="class_prior"):
        module.build_candidate_image_classifier("pulda", spec, seed=0, class_prior=prior)


def test_edge_numpy_prior_produces_strict_json_without_claiming_validation_of_source():
    _, spec = preprocessing()
    model, record = build_candidate_image_classifier(
        "pulda", spec, seed=0, class_prior=np.float32(0.4)
    )
    assert type(record["prior_contract"]["value"]) is float
    assert model.class_prior == float(np.float32(0.4))
    json.dumps(record, allow_nan=False)
    assert record["prior_contract"]["source"] == "caller_supplied_population_prior"
