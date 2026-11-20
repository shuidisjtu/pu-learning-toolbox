"""P3Mix four-role checkpoint selection and native TS refusal."""

# ruff: noqa: E402, N803, N806

import json
from functools import wraps
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.deep.p3mix import P3MixClassifier
from pu_toolbox.experiment import DatasetPart
from pu_toolbox.experiment.runner import ExperimentRunner

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
        training_population_prior=0.5,
        validation_population_prior=0.5,
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
