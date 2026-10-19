"""Dist-PU's opt-in encoder path through CV; frozen Pilot stays feature-adapted."""

# ruff: noqa: E402, N803, N806

import pickle
from functools import wraps

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.deep import vision
from pu_toolbox.estimators.risk.dist_pu import DistPUClassifier
from pu_toolbox.experiment.pilot_plan import planned_runs
from pu_toolbox.experiment.survey_protocol import load_protocol
from pu_toolbox.workflows import PUPipeline

pytestmark = pytest.mark.integration


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1)
        self.first_training_weight = None

    def forward(self, inputs):
        if self.training and self.first_training_weight is None:
            self.first_training_weight = self.conv.weight.detach().clone()
        return self.conv(inputs).mean(dim=(2, 3))


def data():
    features = np.random.RandomState(43).normal(size=(24, 3, 4, 4)).astype(np.float32)
    return features, np.r_[np.ones(8, dtype=int), np.zeros(16, dtype=int)]


def staged_parameters():
    return dict(
        hidden_dim=4,
        batch_size=4,
        distribution_loss="author_histogram",
        training_recipe="author_staged",
        warmup_epochs=1,
        pu_epochs=1,
    )


def test_basic_determ_pipeline_cnn_trains_isolated_cv_copies_and_preserves_template(monkeypatch):
    templates, fits = [], []

    def build(*_args, **_kwargs):
        encoder = TinyEncoder()
        templates.append(encoder)
        return encoder

    original_fit = DistPUClassifier.fit

    @wraps(original_fit)
    def record(estimator, *args, **kwargs):
        result = original_fit(estimator, *args, **kwargs)
        fits.append(result)
        return result

    monkeypatch.setattr(vision, "build_encoder", build)
    monkeypatch.setattr(DistPUClassifier, "fit", record)
    features, labels = data()
    report = PUPipeline(
        classifier="dist_pu",
        classifier_params=staged_parameters(),
        architecture="cnn",
        cv=2,
        random_state=19,
        device="cpu",
        metrics=("pu_zero_one_risk",),
    ).fit_evaluate(features, labels, class_prior=0.4)

    assert len(templates) == 1 and len(fits) == 3
    assert all(fit.encoder is templates[0] for fit in fits)
    assert len({id(fit.encoder_) for fit in fits}) == 3
    for fit in fits:
        torch.testing.assert_close(
            fit.encoder_.first_training_weight, templates[0].conv.weight, rtol=0, atol=0
        )
        assert not torch.equal(fit.encoder_.conv.weight, templates[0].conv.weight)
        assert fit.training_provenance_["implementation_path"] == "author_staged_encoder_adapter"
    assert templates[0].first_training_weight is None
    assert report.final_model is fits[-1]
    assert report.provenance["architecture"] == "native_cnn"
    assert report.provenance["encoder"]["in_channels"] == 3
    restored = pickle.loads(pickle.dumps(report.final_model))
    np.testing.assert_array_equal(
        restored.predict_proba(features), report.final_model.predict_proba(features)
    )


def test_param_pipeline_does_not_silently_switch_default_recipe_for_cnn(monkeypatch):
    monkeypatch.setattr(vision, "build_encoder", lambda *_args, **_kwargs: TinyEncoder())
    features, labels = data()
    with pytest.raises(ValueError, match="encoder requires training_recipe='author_staged'"):
        PUPipeline(
            classifier="dist_pu", architecture="cnn", cv=2, device="cpu", random_state=19
        ).fit_evaluate(features, labels, class_prior=0.4)


def test_basic_mlp_pipeline_keeps_default_clean_room_path():
    features, labels = data()
    features = features.mean(axis=(2, 3))
    report = PUPipeline(
        classifier="dist_pu",
        classifier_params={"epochs": 2, "hidden_dim": 4},
        cv=2,
        random_state=19,
        device="cpu",
        metrics=("pu_zero_one_risk",),
    ).fit_evaluate(features, labels, class_prior=0.4)
    assert report.provenance["architecture"] == "native_mlp"
    assert report.final_model.encoder is None
    assert report.final_model.training_provenance_["implementation_path"] == (
        "clean_room_single_stage_full_batch"
    )


def test_edge_frozen_pilot_keeps_distpu_fullbatch_features_and_original_candidates():
    protocol = load_protocol()
    profile = protocol["method_profiles"]["dist_pu"]
    assert profile["budget"] == "fullbatch" and profile["model_family"] == "mlp128"
    assert "encoder" not in profile["params"] and "training_recipe" not in profile["params"]
    runs = [run for run in planned_runs(protocol) if run.method == "dist_pu"]
    assert runs and not any(run.training_path == "native_cnn" for run in runs)
    assert {run.training_path for run in runs if run.dataset == "cifar10"} == {
        "cnn_feature_adapter"
    }
    estimator = DistPUClassifier(0.4, **profile["params"])
    assert estimator.encoder is None and estimator.training_recipe == "clean_room"
