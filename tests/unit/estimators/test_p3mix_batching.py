"""Explicit paper-order batching, role accounting and legacy compatibility."""

import math
import pickle

import numpy as np
import pytest
from sklearn.base import clone

from pu_toolbox.estimators.deep import p3mix

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("variant", ["base", "e", "c"])
@pytest.mark.parametrize("batch_size", [1, 4, 32])
def test_param_merged_batches_cover_each_row_once_without_role_repetition(
    monkeypatch, variant, batch_size
):
    pytest.importorskip("torch")
    features = np.c_[np.arange(17), np.linspace(-1, 1, 17)].astype(np.float32)
    labels = np.zeros(17, dtype=int)
    labels[[2, 11]] = 1
    before = features.copy(), labels.copy()
    observed = []
    original = p3mix.p3mix_heuristic_batch

    def record(inputs, *args, **kwargs):
        observed.append(inputs[:, 0].astype(int).tolist())
        return original(inputs, *args, **kwargs)

    monkeypatch.setattr(p3mix, "p3mix_heuristic_batch", record)
    fitted = p3mix.P3MixClassifier(
        variant=variant,
        max_epochs=2,
        batch_size=batch_size,
        hidden_dim=4,
        pool_size=2,
        batching_policy="merged_shuffle",
        device="cpu",
        random_state=3,
    ).fit(features, labels)
    steps = math.ceil(len(features) / batch_size)
    assert fitted.optimizer_steps_ == 2 * steps
    for epoch in range(2):
        rows = sum(observed[epoch * steps : (epoch + 1) * steps], [])
        assert sorted(rows) == list(range(17)) and max(map(len, observed)) <= batch_size
        receipt = fitted.epoch_provenance_[epoch]
        assert receipt["positive_rows_seen"] == 2 and receipt["unlabeled_rows_seen"] == 15
        assert receipt["positive_rows_reused"] == receipt["unlabeled_rows_reused"] == 0
    assert np.isfinite(fitted.history_["loss"]).all()
    assert set(fitted.candidate_training_positions_) <= {2, 11}
    assert fitted.training_provenance_["missing_role_loss_is_engineering_choice"]
    assert not fitted.training_provenance_["calibration_applied"]
    np.testing.assert_array_equal(features, before[0])
    np.testing.assert_array_equal(labels, before[1])


def test_basic_determ_default_retains_balanced_role_behavior_and_explicit_policy():
    pytest.importorskip("torch")
    features = np.random.RandomState(4).normal(size=(17, 3)).astype(np.float32)
    labels = np.r_[np.ones(2, dtype=int), np.zeros(15, dtype=int)]
    default = p3mix.P3MixClassifier(max_epochs=2, batch_size=4, hidden_dim=4, device="cpu")
    explicit = clone(default).set_params(batching_policy="role_balanced")
    default.fit(features, labels)
    explicit.fit(features, labels)
    np.testing.assert_array_equal(
        default.decision_function(features), explicit.decision_function(features)
    )
    assert default.positive_rows_seen_ == 16 and default.unlabeled_rows_seen_ == 30
    assert default.history_ == explicit.history_


@pytest.mark.parametrize("policy", ["balanced", None, True, [], {}])
def test_param_edge_unknown_policy_rejected_before_fitting(policy):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="batching_policy"):
        p3mix.P3MixClassifier(batching_policy=policy).fit(np.ones((4, 2)), [1, 0, 0, 0])


@pytest.mark.parametrize("variant", ["base", "e", "c"])
def test_basic_determ_merged_policy_clone_and_pickle_preserve_predictions(variant):
    pytest.importorskip("torch")
    features = np.random.RandomState(9).normal(size=(13, 3)).astype(np.float32)
    labels = np.r_[np.ones(2, dtype=int), np.zeros(11, dtype=int)]
    estimator = p3mix.P3MixClassifier(
        variant=variant,
        batching_policy="merged_shuffle",
        max_epochs=2,
        batch_size=3,
        hidden_dim=4,
        device="cpu",
    )
    assert clone(estimator).batching_policy == "merged_shuffle"
    first = estimator.fit(features, labels)
    second = clone(estimator).fit(features, labels)
    restored = pickle.loads(pickle.dumps(first))
    np.testing.assert_array_equal(
        first.decision_function(features), second.decision_function(features)
    )
    np.testing.assert_array_equal(
        first.decision_function(features), restored.decision_function(features)
    )


def test_edge_singleton_batchnorm_uses_running_stats_and_restores_mode():
    torch = pytest.importorskip("torch")
    features = np.random.RandomState(11).normal(size=(13, 3)).astype(np.float32)
    labels = np.r_[np.ones(2, dtype=int), np.zeros(11, dtype=int)]
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4), torch.nn.ReLU(), torch.nn.Linear(4, 1)
    )
    modes = []
    estimator = p3mix.P3MixClassifier(
        model=model, max_epochs=2, batch_size=4, batching_policy="merged_shuffle", device="cpu"
    ).fit(
        features,
        labels,
        epoch_callback=lambda epoch, fitted: modes.append(fitted.model_[1].training),
    )
    assert modes == [True, True] and estimator.optimizer_steps_ == 8
    assert np.isfinite(estimator.decision_function(features)).all()
    assert model[1].num_batches_tracked.item() == 0
