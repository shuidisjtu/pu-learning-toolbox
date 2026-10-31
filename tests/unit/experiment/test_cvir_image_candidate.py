"""CVIR shared image candidates bind observed-count priors and all paid stages."""

# ruff: noqa: E402

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.experiment.candidate_image import build_candidate_image_classifier  # noqa: E402
from pu_toolbox.experiment.candidate_image_runner import (
    CandidateImageCheckpointTrainer,
)  # noqa: E402
from pu_toolbox.experiment.cvir_adapter import cvir_risk_prior  # noqa: E402
from pu_toolbox.experiment.feature_adapter import encoder_state_sha256  # noqa: E402
from pu_toolbox.experiment.image import fit_survey_image_preprocessing  # noqa: E402

pytestmark = pytest.mark.unit


def setup(**parameters):
    raw = np.random.RandomState(17).randint(0, 256, (12, 1, 8, 8)).astype(np.uint8)
    pixels, preprocessing = fit_survey_image_preprocessing(
        raw, dataset="mnist", train_augmentation="none"
    )
    model, record = build_candidate_image_classifier("cvir", preprocessing, seed=3, device="cpu")
    model.set_params(**(dict(max_epochs=2, batch_size=4, hidden_dim=4) | parameters))
    return model, record, pixels, np.r_[np.ones(4, int), np.zeros(8, int)]


def trainer(record, path, bound=64 * 2**20):
    return CandidateImageCheckpointTrainer(
        record,
        checkpoint_dir=path,
        checkpoint_bytes_per_component=bound,
        population_prior_source="synthetic population",
    )


@pytest.mark.parametrize("view", ["os", "ts"])
@pytest.mark.parametrize("warm", [0, 1])
def test_basic_determ_actual_stage_cost_prior_and_plain_fit_parity(tmp_path, view, warm):
    model, record, pixels, labels = setup(warm_start_epochs=warm)
    alpha = cvir_risk_prior(0.5, n_samples=12, n_labeled=4, os_or_ts=view)
    direct = (
        copy.deepcopy(model)
        .set_params(unlabeled_positive_prior=alpha, prior_source="synthetic population")
        .fit(pixels, labels, os_or_ts=view)
    )
    trajectory = trainer(record, tmp_path).fit(
        model, pixels, labels, class_prior=0.5, os_or_ts=view
    )
    facts = model.training_provenance_
    receipt = facts["bound_fit_receipt"]
    assert model.unlabeled_positive_prior is None and model.prior_source is None
    assert model.unlabeled_positive_prior_ == alpha
    assert receipt["parameters_at_fit_entry"]["unlabeled_positive_prior"] == alpha
    assert receipt["parameters_at_fit_entry"]["prior_source"] == "synthetic population"
    assert receipt["calibration_applied"] is (view == "ts")
    assert not facts["population_prior_forwarded_to_fit"]
    assert not facts["population_prior_binding"]["uses_hidden_training_labels"]
    assert len(trajectory.checkpoints) == warm + 2
    warm_steps = 3 if view == "ts" else 2
    assert model.optimizer_steps_ == warm * warm_steps + 4
    assert [c.training_context for c in trajectory.checkpoints] == direct.epoch_provenance_
    assert [row["ranking_forward_batches"] for row in facts["epoch_cost_history"]] == (
        [0] * warm + [warm_steps] * 2
    )
    assert receipt["completed_epochs"] == warm + 2
    assert model.history_ == direct.history_
    assert encoder_state_sha256(model.encoder) == record["encoder_initial_state_sha256"]
    assert encoder_state_sha256(model.encoder_) != record["encoder_initial_state_sha256"]
    for key, value in model.model_.state_dict().items():
        torch.testing.assert_close(value, direct.model_.state_dict()[key], rtol=0, atol=0)
    np.testing.assert_array_equal(
        trajectory.checkpoints[-1].restore().decision_function(pixels),
        model.decision_function(pixels),
    )


@pytest.mark.parametrize(
    "fault",
    [
        "prebound",
        "native_ts",
        "resident",
        "warm",
        "lr",
        "missing_view",
        "impossible_prior",
        "singleton",
    ],
)
def test_param_invalid_binding_or_bn_budget_rejected_before_training(tmp_path, monkeypatch, fault):
    model, record, pixels, labels = setup()
    if fault == "prebound":
        model.unlabeled_positive_prior = 0.3
    elif fault == "native_ts":
        model.input_sampling = "ts"
    elif fault == "resident":
        model.data_device_policy = "resident"
    elif fault == "warm":
        model.warm_start_epochs = True
    elif fault == "lr":
        model.learning_rate = float("nan")
    elif fault == "singleton":
        model.batch_size = 3
    monkeypatch.setattr(type(model), "fit", lambda *a, **k: pytest.fail("invalid CVIR image fit"))
    with pytest.raises(ValueError):
        trainer(record, tmp_path).fit(
            model,
            pixels,
            labels,
            class_prior=0.2 if fault == "impossible_prior" else 0.5,
            os_or_ts=None if fault == "missing_view" else "os",
        )
    assert not list(tmp_path.rglob("*.pt"))


@pytest.mark.parametrize("fault", ["updates", "risk_prior", "negative_identity"])
def test_edge_corrupt_stage_refused_and_temporary_prior_restored(tmp_path, monkeypatch, fault):
    model, record, pixels, labels = setup(warm_start_epochs=0)
    original = model.fit

    def bad_fit(features, observed, *, epoch_callback, **kwargs):
        def callback(epoch, fitted):
            if fault == "updates":
                fitted.optimizer_steps_ -= 1
            elif fault == "risk_prior":
                fitted.unlabeled_positive_prior_ = 0.9
            else:
                fitted.selected_negative_indices_[0] = 0
            epoch_callback(epoch, fitted)

        return original(features, observed, epoch_callback=callback, **kwargs)

    monkeypatch.setattr(model, "fit", bad_fit)
    with pytest.raises(ValueError, match="CVIR"):
        trainer(record, tmp_path).fit(model, pixels, labels, class_prior=0.5, os_or_ts="os")
    assert model.unlabeled_positive_prior is None and model.prior_source is None
    assert not model._is_fitted and not hasattr(model, "training_provenance_")
    assert not list(tmp_path.rglob("*.pt"))


def test_param_population_source_required_and_edge_oversized_evidence_kept(tmp_path):
    model, record, pixels, labels = setup()
    with pytest.raises(ValueError, match="population_prior_source"):
        CandidateImageCheckpointTrainer(record, checkpoint_bytes_per_component=10)
    with pytest.raises(ValueError, match="byte bound"):
        trainer(record, tmp_path, bound=1).fit(
            model, pixels, labels, class_prior=0.5, os_or_ts="os"
        )
    assert not model._is_fitted and model.unlabeled_positive_prior is None
    assert len(list(tmp_path.rglob("*.pt"))) == 3
