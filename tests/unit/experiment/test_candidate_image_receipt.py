"""Bound technical fits record actual data/config without upgrading admission."""

import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.experiment.candidate_image import (  # noqa: E402
    build_candidate_image_classifier,
    fit_candidate_image_classifier,
)
from pu_toolbox.experiment.image import fit_survey_image_preprocessing  # noqa: E402
from pu_toolbox.utils.serialization import array_hash  # noqa: E402

pytestmark = pytest.mark.unit


def setup(method="pulda", augmentation="none"):
    raw = np.random.RandomState(2).randint(0, 256, (8, 3, 8, 8)).astype(np.uint8)
    features, preprocessing = fit_survey_image_preprocessing(
        raw, dataset="cifar10", train_augmentation=augmentation
    )
    model, record = build_candidate_image_classifier(
        method,
        preprocessing,
        seed=3,
        class_prior=0.4 if method == "pulda" else None,
    )
    if method == "pulda":
        model.set_params(
            hidden_dim=4,
            depth=1,
            warmup_epochs=1,
            pu_epochs=1,
            positive_batch_size=2,
            unlabeled_batch_size=2,
        )
    else:
        model.set_params(hidden_dim=4, max_epochs=1, batch_size=2)
    return model, record, features, np.r_[np.ones(4, int), np.zeros(4, int)]


@pytest.mark.parametrize(
    "method,augmentation", [("pulda", "none"), ("pulda", "simaugment"), ("gradpu", "none")]
)
def test_basic_actual_fit_records_data_parameters_calibration_and_completed_cost(
    method, augmentation
):
    model, record, features, labels = setup(method, augmentation)
    before = copy.deepcopy(record)
    epochs = []
    returned, receipt = fit_candidate_image_classifier(
        model,
        record,
        features,
        labels,
        epoch_callback=lambda epoch, fitted: epochs.append(epoch),
    )
    assert returned is model and record == before
    assert receipt["image_setup"] == record and receipt["image_setup"] is not record
    assert receipt["parameters_at_fit_entry"]["hidden_dim"] == 4
    assert receipt["train_input_sha256"] == array_hash(features)
    assert receipt["observed_PU_labels_sha256"] == array_hash(labels.astype(np.int64))
    assert receipt["training_view"] == "ts" and receipt["calibration_applied"]
    assert receipt["actual_device"] == "cpu" and receipt["encoder_trainable"]
    assert receipt["completed_epochs"] == len(epochs) == (2 if method == "pulda" else 1)
    assert receipt["optimizer_steps"] == (4 if method == "pulda" else 2)
    assert not receipt["selection_performed"] and not receipt["ready_for_formal_run"]
    assert (
        not receipt["four_role_isolation_verified"]
        and not receipt["label_budget_externally_verified"]
    )
    assert not receipt["population_mixture_restoration_verified"]
    if method == "gradpu":
        assert receipt["algorithm_training"] == model.training_provenance_
    assert len(receipt["implementation_sha256"]) == len(receipt["helper_sha256"]) == 64
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize(
    "fault",
    [
        "seed",
        "prior",
        "encoder",
        "transform",
        "group",
        "admission",
        "backbone",
        "pixels",
        "dtype",
        "shape",
        "old_record",
    ],
)
def test_param_stale_or_mislabelled_inputs_fail_before_any_fit(monkeypatch, fault):
    model, record, features, labels = setup()
    if fault == "seed":
        model.random_state = 4
    elif fault == "prior":
        model.class_prior = 0.3
    elif fault == "encoder":
        with torch.no_grad():
            next(model.encoder.parameters()).add_(0.1)
    elif fault == "transform":
        model.train_transform = torch.nn.Identity()
    elif fault == "group":
        record["comparison_group"] = "shared-main-leaderboard"
    elif fault == "admission":
        record["formal_admission"] = True
    elif fault == "backbone":
        record["image_preprocessing"]["backbone"]["name"] = "resnet50"
    elif fault == "pixels":
        features = features.copy()
        features[0, 0, 0, 0] += 0.01
    elif fault == "dtype":
        features = features.astype(np.float64)
    elif fault == "shape":
        features = features[:, :, :4]
    else:
        record.pop("train_transform_description")

    def forbidden(*args, **kwargs):
        pytest.fail("inconsistent setup must not start fit")

    monkeypatch.setattr(model, "fit", forbidden)
    with pytest.raises(ValueError):
        fit_candidate_image_classifier(model, record, features, labels)


def test_edge_callback_exception_is_not_retried_or_hidden():
    model, record, features, labels = setup()
    sentinel = RuntimeError("injected callback failure")
    calls = []

    def fail(epoch, fitted):
        calls.append(epoch)
        raise sentinel

    with pytest.raises(RuntimeError) as error:
        fit_candidate_image_classifier(model, record, features, labels, epoch_callback=fail)
    assert error.value is sentinel and calls == [0]
    assert model._is_fitted is False


def test_edge_mutated_training_pixels_do_not_get_a_valid_receipt():
    model, record, features, labels = setup()

    def mutate(epoch, fitted):
        features[0, 0, 0, 0] += 0.001

    with pytest.raises(RuntimeError, match="changed during fit"):
        fit_candidate_image_classifier(model, record, features, labels, epoch_callback=mutate)


def test_determ_repeated_bound_fits_have_same_predictions_and_receipt_identities():
    results = []
    for _ in range(2):
        model, record, features, labels = setup()
        _, receipt = fit_candidate_image_classifier(model, record, features, labels, os_or_ts="os")
        receipt.pop("elapsed_seconds_including_input_hashing")
        results.append((receipt, model.decision_function(features)))
    assert results[0][0] == results[1][0]
    assert not results[0][0]["calibration_applied"]
    np.testing.assert_array_equal(results[0][1], results[1][1])
