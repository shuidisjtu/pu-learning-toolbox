"""Full-data budget, diagnostic isolation and explicit collapse classification."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import validate_gradpu_encoder as script

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.unit


def test_determ_probe_is_balanced_repeatable_and_does_not_reorder_training():
    labels = np.r_[np.zeros(180), np.ones(140)]
    original = labels.copy()
    indices = script.probe_indices(labels, 0)
    np.testing.assert_array_equal(indices, script.probe_indices(labels, 0))
    np.testing.assert_array_equal(labels, original)
    assert len(set(indices)) == 256 and sum(labels[indices]) == 128


def test_basic_observation_preserves_modes_and_detects_constant_output():
    encoder = torch.nn.Linear(2, 2)
    torch.nn.init.zeros_(encoder.weight)
    torch.nn.init.zeros_(encoder.bias)
    model = torch.nn.Sequential(encoder, torch.nn.Tanh()).train()
    # A single-output head follows a two-feature encoder.
    model = torch.nn.Sequential(encoder, torch.nn.Linear(2, 1), torch.nn.Tanh()).train()
    encoder.eval()
    estimator = SimpleNamespace(model_=model, encoder_=encoder)
    rng = torch.get_rng_state().clone()
    metrics = script.observe(estimator, np.ones((20, 2), dtype=np.float32), np.tile([0, 1], 10))
    assert script.collapsed(metrics) and metrics["auc"] == 0.5
    assert model.training and not encoder.training
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("collapse", [False, True])
def test_param_full_data_200_epochs_or_explicit_numerical_failure(tmp_path, monkeypatch, collapse):
    from pu_toolbox.experiment import candidate_image

    x = np.random.RandomState(7).randint(0, 256, (24, 3, 8, 8)).astype("uint8")
    y = np.tile([0, 1], 12)
    data = tmp_path / "data"
    data.mkdir()
    for name in ["train", "clean_val"]:
        np.savez(data / f"{name}.npz", X=x, y=y)
    seen = []
    loader = np.load

    def load(path, **kwargs):
        seen.append(path.name)
        assert path.name in {"train.npz", "clean_val.npz"}
        return loader(path, **kwargs)

    class Estimator:
        model_ = torch.nn.Linear(1, 1)

        def set_params(self, **params):
            self.params = params

        def fit(self, inputs, observed, *, os_or_ts, epoch_callback):
            assert len(inputs) == 24 and os_or_ts == "ts"
            np.testing.assert_allclose(inputs, x / 255, atol=1e-7)
            assert self.params["batch_size"] == 256
            assert self.params["max_epochs"] == 200
            assert self.params["learning_rate"] == 1e-4
            self.history_ = dict(epoch=[], train_risk=[])
            for epoch in range(200):
                self.history_["epoch"].append(epoch)
                self.history_["train_risk"].append(0.1)
                epoch_callback(epoch, self)

    monkeypatch.setattr(script.np, "load", load)
    monkeypatch.setattr(
        candidate_image, "build_candidate_image_classifier", lambda *a, **k: (Estimator(), {})
    )
    monkeypatch.setattr(
        script,
        "observe",
        lambda *a: dict(feature_std=0.0 if collapse else 0.1, score_std=0.0 if collapse else 0.1),
    )
    out = tmp_path / "output"
    result = script.run(data, out, dataset="cifar10", arm="relu_gain_lr1e4", c=0.5, device="cpu")
    assert seen == ["train.npz", "clean_val.npz"]
    assert result["full_train"] and not result["test_set_read"] and not result["formal_admission"]
    assert result["epochs_completed"] == (10 if collapse else 200)
    assert result["status"] == (
        "failed_numerical_collapse" if collapse else "full_budget_completed_not_formal"
    )
    assert (out / "final_weights.pt").exists() is (not collapse)
    assert json.loads((out / "report.json").read_text())["status"] == result["status"]
    with pytest.raises(ValueError, match="never overwrite"):
        script.run(data, out, dataset="cifar10", arm="legacy", device="cpu")


@pytest.mark.parametrize(
    "options",
    [
        dict(arm="unknown"),
        dict(seed=True),
        dict(c=0),
        dict(backward_mode="joint"),
        dict(gp_chunk_size=0),
    ],
)
def test_edge_invalid_configuration_refused_before_reading_data(tmp_path, options):
    kwargs = dict(dataset="cifar10", arm="legacy", device="cpu")
    kwargs.update(options)
    with pytest.raises(ValueError):
        script.run(tmp_path / "missing", tmp_path / "out", **kwargs)
    assert not (tmp_path / "out").exists()
