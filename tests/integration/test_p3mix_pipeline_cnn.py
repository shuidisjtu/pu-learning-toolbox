"""P3Mix variants through real convolution CV and four-role checkpoint selection."""

# ruff: noqa: E402, N803, N806

import json
import pickle
from functools import wraps
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.deep import vision
from pu_toolbox.estimators.deep.p3mix import P3MixClassifier
from pu_toolbox.experiment import DatasetPart
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.workflows import PUPipeline

pytestmark = pytest.mark.integration


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv = torch.nn.Conv2d(3, 4, 1)
        self.first_training_weight = None

    def forward(self, inputs):
        if self.training and self.first_training_weight is None:
            self.first_training_weight = self.conv.weight.detach().clone()
        return self.conv(self.relu(inputs)).mean(dim=(2, 3))


def data(seed=53):
    features = np.random.RandomState(seed).normal(size=(24, 3, 4, 4)).astype(np.float32)
    return features, np.r_[np.ones(8, dtype=int), np.zeros(16, dtype=int)]


@pytest.mark.parametrize("variant", ["base", "e", "c"])
@pytest.mark.parametrize("batching_policy", ["role_balanced", "merged_shuffle"])
def test_basic_determ_three_variants_train_independent_cv_encoders_and_roundtrip(
    monkeypatch, variant, batching_policy
):
    templates, fits = [], []

    def build(*_args, **_kwargs):
        template = TinyEncoder().requires_grad_(False)
        templates.append(template)
        return template

    original_fit = P3MixClassifier.fit

    @wraps(original_fit)
    def record(estimator, *args, **kwargs):
        fitted = original_fit(estimator, *args, **kwargs)
        fits.append(fitted)
        return fitted

    monkeypatch.setattr(vision, "build_encoder", build)
    monkeypatch.setattr(P3MixClassifier, "fit", record)
    features, labels = data()
    originals = features.copy(), labels.copy()
    report = PUPipeline(
        classifier="p3mix",
        classifier_params={
            "variant": variant,
            "hidden_dim": 4,
            "max_epochs": 2,
            "batch_size": 4,
            "pool_size": 3,
            "batching_policy": batching_policy,
        },
        architecture="cnn",
        cv=2,
        random_state=19,
        device="cpu",
        metrics=("pu_zero_one_risk",),
    ).fit_evaluate(features, labels, class_prior=0.4)

    assert len(templates) == 1 and len(fits) == 3
    assert len({id(fit.encoder_) for fit in fits}) == 3
    for fit in fits:
        assert fit.encoder is templates[0] and fit.encoder_ is not templates[0]
        torch.testing.assert_close(
            fit.encoder_.first_training_weight, templates[0].conv.weight, rtol=0, atol=0
        )
        assert all(parameter.requires_grad for parameter in fit.encoder_.parameters())
        assert not torch.equal(fit.encoder_.conv.weight, templates[0].conv.weight)
        assert fit.training_provenance_["run_view"] == "os"
        assert not fit.training_provenance_["calibration_applied"]
        assert not fit.training_provenance_["formal_admission"]
        assert (fit.teacher_ is not None) is (variant == "e")
        assert fit.candidate_training_positions_.size == 3
    assert templates[0].first_training_weight is None
    assert all(not parameter.requires_grad for parameter in templates[0].parameters())
    assert report.final_model is fits[-1]
    assert report.provenance["architecture"] == "native_cnn"
    restored = pickle.loads(pickle.dumps(report.final_model))
    np.testing.assert_array_equal(
        restored.predict_proba(features), report.final_model.predict_proba(features)
    )
    np.testing.assert_array_equal(features, originals[0])
    np.testing.assert_array_equal(labels, originals[1])


@pytest.mark.parametrize("variant", ["base", "e", "c"])
@pytest.mark.parametrize("reclaim", [False, True])
@pytest.mark.parametrize("batching_policy", ["role_balanced", "merged_shuffle"])
def test_basic_four_role_runner_uses_observed_train_labels_and_replayable_snapshots(
    monkeypatch, tmp_path, variant, reclaim, batching_policy
):
    parts = []
    for role in range(4):
        features, _ = data(57 + role)
        parts.append(
            DatasetPart(
                X=features,
                labels=np.tile([0, 1], 12),
                indices=np.arange(role * 24, (role + 1) * 24),
                view="clean",
                for_selection=role != 3,
            )
        )
    originals = [(part.X.copy(), part.labels.copy()) for part in parts]
    observed = []
    original_fit = P3MixClassifier.fit

    @wraps(original_fit)
    def record(estimator, features, labels, **kwargs):
        np.testing.assert_array_equal(features, parts[0].X)
        assert len(labels) == len(parts[0].labels)
        assert 0 < labels.sum() < parts[0].labels.sum()
        assert np.all(np.asarray(labels) <= parts[0].labels)
        observed.append(labels.copy())
        return original_fit(estimator, features, labels, **kwargs)

    monkeypatch.setattr(P3MixClassifier, "fit", record)
    template = TinyEncoder()
    before = {name: value.clone() for name, value in template.state_dict().items()}
    path = tmp_path / "manifest.json"
    result = ExperimentRunner(
        seed=0,
        class_prior=0.5,
        manifest_path=str(path),
        config={
            "method": "p3mix",
            "architecture": "cnn",
            "c": 0.5,
            "os_or_ts": "os",
            "checkpoint_bytes_per_component": 4096,
            "reclaim_unselected_checkpoints": reclaim,
        },
    ).fit(
        P3MixClassifier(
            encoder=template,
            variant=variant,
            hidden_dim=4,
            max_epochs=2,
            batch_size=4,
            pool_size=3,
            batching_policy=batching_policy,
            device="cpu",
        ),
        *parts,
    )
    assert len(observed) == 1
    assert set(result.test_metrics) == {"PA", "OA"}
    assert all(np.isfinite(metrics["accuracy"]) for metrics in result.test_metrics.values())
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["run_view"] == "os-compatible" and not manifest["calibration_applied"]
    assert manifest.get("execution_mode") != "versioned_pilot"
    assert "recipe_registry_sha256" not in manifest
    run = manifest["candidate_runs"][0]
    assert run["training_provenance"]["variant"] == variant
    assert run["training_provenance"]["batching_policy"] == batching_policy
    if batching_policy == "merged_shuffle":
        assert run["training_provenance"]["positive_rows_seen"] == 2 * int(observed[0].sum())
        assert run["training_provenance"]["unlabeled_rows_seen"] == 2 * int(
            (observed[0] == 0).sum()
        )
    assert not run["training_provenance"]["formal_admission"]
    references = run["epoch_checkpoints"]
    assert len(references) == 2
    assert [ref["training_context"]["stage_epoch"] for ref in references] == [1, 2]
    selected = {entry["checkpoint"]["path"] for entry in manifest["selection"].values()}
    for ref in references:
        assert ref["training_context"]["stage"] == "train"
        expected_reclaimed = reclaim and ref["path"] not in selected
        assert ref["reclaimed"] is expected_reclaimed
        assert Path(ref["path"]).is_file() is not expected_reclaimed
    for name, value in template.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    for part, (features, labels) in zip(parts, originals, strict=True):
        np.testing.assert_array_equal(part.X, features)
        np.testing.assert_array_equal(part.labels, labels)


def test_param_edge_ts_is_not_silently_calibrated_and_revokes_an_old_fit():
    from pu_toolbox.core.exceptions import NotFittedError

    features, labels = data()
    model = P3MixClassifier(encoder=TinyEncoder(), max_epochs=1, device="cpu")
    model.fit(features, labels)
    with pytest.raises(ValueError, match="TS is not hooked"):
        model.fit(features, labels, os_or_ts="ts")
    assert not model._is_fitted
    with pytest.raises(NotFittedError):
        model.predict_proba(features)
