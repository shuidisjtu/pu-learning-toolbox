"""Train-only ablation, stable subsets and refusal paths; no admission by probes."""

import json

import numpy as np
import pytest

from scripts import diagnose_gradpu_encoder as script

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.unit


def test_determ_stratified_indices_repeat_without_duplicates_or_input_mutation():
    labels = np.r_[np.ones(9, int), np.zeros(11, int)]
    original = labels.copy()
    indices = script.subset_indices(labels, 12, 7)
    np.testing.assert_array_equal(indices, script.subset_indices(labels, 12, 7))
    assert len(set(indices)) == 12 and set(labels[indices]) == {0, 1}
    np.testing.assert_array_equal(labels, original)


@pytest.mark.parametrize(
    "labels,samples",
    [(np.zeros(8), 4), (np.arange(8), 4), (np.tile([0, 1], 4), 9), (np.tile([0, 1], 4), 3)],
)
def test_param_bad_subset_refused(labels, samples):
    with pytest.raises(ValueError, match="binary train labels"):
        script.subset_indices(labels, samples, 0)


def test_edge_signal_metrics_detect_constant_output_and_restore_modes():
    encoder = torch.nn.Linear(2, 2)
    torch.nn.init.zeros_(encoder.weight)
    torch.nn.init.zeros_(encoder.bias)
    model = torch.nn.Sequential(encoder, torch.nn.Tanh()).train()
    encoder.eval()
    metrics = script.signal_metrics(encoder, torch.ones(4, 2), model)
    assert metrics["feature_std"] == metrics["score_std"] == 0
    assert model.training and not encoder.training


def test_basic_train_only_ablation_keeps_nominal_budget_and_writes_nonformal_receipts(
    tmp_path, monkeypatch
):
    from pu_toolbox.estimators.deep.grad_pu import GradPUClassifier
    from pu_toolbox.experiment import candidate_image

    train = tmp_path / "train.npz"
    np.savez(
        train,
        X=np.random.RandomState(7).randint(0, 256, (24, 3, 8, 8)).astype("uint8"),
        y=np.tile([0, 1], 12),
    )
    calls = []

    def build(method, preprocessing, *, seed, device, gradpu_initialization):
        calls.append(gradpu_initialization)
        encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        return GradPUClassifier(encoder=encoder, hidden_dim=3, random_state=seed, device=device), {}

    monkeypatch.setattr(candidate_image, "build_candidate_image_classifier", build)
    loaded = []
    load = np.load

    def train_only(path, **kwargs):
        loaded.append(path)
        assert path == train
        return load(path, **kwargs)

    monkeypatch.setattr(script.np, "load", train_only)
    result = script.run(
        train,
        tmp_path / "output",
        dataset="cifar10",
        seeds=[0],
        samples=20,
        batch_size=8,
        stop_epochs=2,
        c=0.5,
    )
    assert loaded == [train] and len(calls) == 3
    assert result["nominal_max_epochs"] == 200 and result["stop_epochs"] == 2
    assert (
        result["diagnostic_only"] and not result["formal_admission"] and not result["test_set_read"]
    )
    assert all(len(arm["epochs"]) == 2 for arm in result["arms"])
    assert all(arm["epochs"][-1]["beta"] < 0.02 for arm in result["arms"])
    assert json.loads((tmp_path / "output/report.json").read_text())["train_sha256"]
    with pytest.raises(ValueError, match="never overwrite"):
        script.run(train, tmp_path / "output", dataset="cifar10", seeds=[0])


@pytest.mark.parametrize(
    "name,options",
    [("test.npz", {}), ("train.npz", {"stop_epochs": 0}), ("train.npz", {"seeds": [0, 0]})],
)
def test_edge_refuses_test_input_and_invalid_schedule_before_loading(tmp_path, name, options):
    arguments = dict(dataset="cifar10", seeds=[0])
    arguments.update(options)
    with pytest.raises(ValueError):
        script.run(tmp_path / name, tmp_path / "out", **arguments)
    assert not (tmp_path / "out").exists()
