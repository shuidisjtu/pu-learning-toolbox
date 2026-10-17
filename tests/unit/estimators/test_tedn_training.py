"""TEDn temporal order, CNN isolation, early convergence and snapshot replay."""

# Match public feature naming.
# ruff: noqa: N803, N806

import copy
import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import pu_toolbox.estimators.deep.tedn as module  # noqa: E402
from pu_toolbox.core.exceptions import NotFittedError  # noqa: E402
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer  # noqa: E402
from pu_toolbox.prior.bbe import best_bin_estimate  # noqa: E402
from tests.unit.estimators.test_tedn import data as large_data  # noqa: E402
from tests.unit.estimators.test_tedn import model as large_model  # noqa: E402

pytestmark = pytest.mark.unit


class RecordingEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.calls = []

    def forward(self, X):
        self.calls.append((self.training, X[:, 0, 0, 0].detach().cpu().numpy().copy(), len(X)))
        return self.bn(self.conv(X)).tanh().mean((2, 3))


def small_data(images=False):
    rng = np.random.RandomState(3)
    if images:
        X = rng.normal(size=(24, 1, 3, 3)).astype("float32")
        X[:, 0, 0, 0] = np.arange(24)
    else:
        X = rng.normal(size=(24, 2)).astype("float32")
    return X, np.r_[np.ones(8), np.zeros(16)]


def small_model(**kwargs):
    params = dict(
        input_sampling="ts",
        holdout_fraction=0.25,
        hidden_dim=4,
        warm_start_epochs=1,
        max_epochs=2,
        batch_size=3,
        random_state=11,
        device="cpu",
        learning_rate=0.01,
    )
    params.update(kwargs)
    return module.TEDnClassifier(**params)


def controlled_estimates(monkeypatch, alphas):
    """Stub only alpha for orchestration tests; real BBE covered separately.

    Candidate receipts are not used as numeric evidence for these stubs.
    """
    iterator = iter(alphas)

    def estimate(p, u, **kwargs):
        return replace(best_bin_estimate(p, u, **kwargs), alpha_u=next(iterator))

    monkeypatch.setattr(module, "best_bin_estimate", estimate)


def test_basic_real_estimation_and_ranking_happen_before_each_update(monkeypatch):
    X, roles = large_data()
    estimator = large_model()
    expected, calls = {}, []
    real = module.best_bin_estimate

    def capture(epoch, fitted):
        if fitted.checkpoint_stage_ == "tedn":
            np.testing.assert_array_equal(fitted.selected_negative_indices_, expected["negatives"])
        hp, hu = fitted.holdout_positive_indices_, fitted.holdout_unlabeled_indices_
        expected["p"] = torch.sigmoid(torch.from_numpy(fitted.decision_function(X[hp]))).numpy()
        expected["u"] = torch.sigmoid(torch.from_numpy(fitted.decision_function(X[hu]))).numpy()
        expected["steps"] = fitted.optimizer_steps_
        expected["rank"] = fitted.decision_function(X[fitted.train_unlabeled_indices_])

    def estimate(p, u, **kwargs):
        np.testing.assert_array_equal(p, expected["p"])
        np.testing.assert_array_equal(u, expected["u"])
        result = real(p, u, **kwargs)
        tu = estimator.train_unlabeled_indices_
        expected["negatives"] = tu[module.select_tedn_negatives(expected["rank"], result.alpha_u)]
        calls.append(expected["steps"])
        return result

    monkeypatch.setattr(module, "best_bin_estimate", estimate)
    estimator.fit(X, roles, epoch_callback=capture)
    assert calls == [r["optimizer_steps_before_update"] for r in estimator.bbe_history_]
    assert len(calls) == 4 and all(a < b for a, b in zip(calls, calls[1:], strict=False))


def test_basic_cnn_joint_training_excludes_holdout_and_bounds_every_forward(monkeypatch):
    X, roles = small_data(images=True)
    supplied = RecordingEncoder()
    before = copy.deepcopy(supplied.state_dict())
    controlled_estimates(monkeypatch, [0.25, 0.25])
    estimator = small_model(encoder=supplied).fit(X, roles)
    assert estimator.encoder_ is not supplied and estimator.model_[0] is estimator.encoder_
    assert supplied.calls == []
    for name, tensor in supplied.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    assert any(not torch.equal(before[n], t) for n, t in estimator.encoder_.state_dict().items())
    held = set(np.r_[estimator.holdout_positive_indices_, estimator.holdout_unlabeled_indices_])
    trained_ids = []
    training_calls = 0
    for training, ids, count in estimator.encoder_.calls:
        if training:
            assert 2 <= count <= 6
            assert set(ids).isdisjoint(held)
            trained_ids.extend(ids)
            training_calls += 1
        else:
            assert 1 <= count <= 3
    train = set(np.r_[estimator.train_positive_indices_, estimator.train_unlabeled_indices_])
    assert set(trained_ids) == train
    assert training_calls == estimator.optimizer_steps_ == 10
    assert estimator.encoder_.bn.num_batches_tracked.item() == 10
    assert estimator.history_["selected_negative_count"] == [9, 9]
    assert estimator.data_device_policy_ == "cpu_stream"
    assert estimator.training_data_device_ == "cpu"
    assert set(estimator.selected_negative_indices_) <= set(estimator.train_unlabeled_indices_)


@pytest.mark.parametrize("warmup", [0, 1, 2])
def test_basic_all_stage_snapshots_and_pickle_exact_replay(monkeypatch, tmp_path, warmup):
    X, roles = small_data(images=True)
    controlled_estimates(monkeypatch, [0.25, 0.25])
    estimator = small_model(encoder=RecordingEncoder(), warm_start_epochs=warmup)
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(estimator, X, roles)
    assert len(trajectory.checkpoints) == estimator.epochs_completed_ == warmup + 2
    assert estimator.optimizer_steps_ == warmup * 4 + 6
    assert [p["stage"] for p in estimator.epoch_provenance_] == ["warm_start"] * warmup + [
        "tedn"
    ] * 2
    for checkpoint, context in zip(
        trajectory.checkpoints, estimator.epoch_provenance_, strict=True
    ):
        reference = checkpoint.reference()
        assert reference["training_context"] == context
        assert reference["prediction_batch_size"] == 3
        assert reference["training_resume_supported"] is False
    restored = trajectory.checkpoints[-1].restore(device="cpu")
    np.testing.assert_array_equal(restored.decision_function(X), estimator.decision_function(X))
    cloned = pickle.loads(pickle.dumps(estimator))
    np.testing.assert_array_equal(cloned.decision_function(X), estimator.decision_function(X))
    assert cloned.bbe_history_ == estimator.bbe_history_


def test_basic_alpha_recomputed_not_frozen_and_zero_retains_all(monkeypatch):
    X, roles = small_data()
    controlled_estimates(monkeypatch, [0.25, 0.5, 0.0])
    estimator = small_model(max_epochs=3).fit(X, roles)
    assert estimator.history_["alpha_u"] == [0.25, 0.5, 0.0]
    assert estimator.history_["selected_negative_count"] == [9, 6, 12]
    assert estimator.optimizer_steps_ == 4 + 3 + 2 + 4
    assert estimator.estimate_unlabeled_prior() == 0


def test_edge_training_error_plateau_can_stop_below_checkpoint_upper_bound(monkeypatch, tmp_path):
    X, roles = small_data()
    controlled_estimates(monkeypatch, [0.25, 0.25])
    estimator = small_model(max_epochs=10, convergence_tol=2, convergence_patience=1)
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(estimator, X, roles)
    assert estimator.converged_ and estimator.stop_reason_ == "training_error_plateau"
    assert estimator.n_iter_ == 2 and estimator.epochs_completed_ == 3
    assert len(trajectory.checkpoints) == 3 < estimator.checkpoint_epoch_count == 11
    assert estimator.optimizer_steps_ == 10


def test_determ_holdout_has_no_direct_gradient_or_stopping_influence(monkeypatch):
    X, roles = small_data()
    controlled_estimates(monkeypatch, [0.25] * 4)
    first = small_model().fit(X, roles)
    changed = X.copy()
    held = np.r_[first.holdout_positive_indices_, first.holdout_unlabeled_indices_]
    changed[held] += 1000
    second = small_model().fit(changed, roles)
    # Holding BBE alpha fixed removes its legitimate indirect influence; all
    # training states and stopping statistics must then be exactly invariant.
    for name, tensor in first.model_.state_dict().items():
        torch.testing.assert_close(tensor, second.model_.state_dict()[name], rtol=0, atol=0)
    assert first.history_ == second.history_
    assert first.optimizer_steps_ == second.optimizer_steps_


def test_edge_real_constant_scores_alpha_one_invalidates_fit_without_fallback():
    X, roles = small_data()
    X[:] = 0
    estimator = small_model()
    stages = []
    with pytest.raises(ValueError, match="no retained U"):
        estimator.fit(X, roles, epoch_callback=lambda epoch, m: stages.append(m.checkpoint_stage_))
    assert estimator.bbe_history_[0]["alpha_u"] == 1
    assert estimator.optimizer_steps_ == 4
    assert stages == ["warm_start"] and estimator.n_iter_ == 0
    with pytest.raises(NotFittedError):
        estimator.predict(X)
    with pytest.raises(NotFittedError):
        estimator.estimate_unlabeled_prior()


def test_param_callback_failure_does_not_retry_bare_fit(monkeypatch):
    X, roles = small_data()
    controlled_estimates(monkeypatch, [0.25, 0.25])
    estimator = small_model()
    calls = []

    def fail(epoch, fitted):
        calls.append(epoch)
        raise TypeError("callback audit failed")

    with pytest.raises(TypeError, match="callback audit failed"):
        estimator.fit(X, roles, epoch_callback=fail)
    assert calls == [0] and estimator.optimizer_steps_ == 4
    assert not estimator._is_fitted


def test_edge_warm_start_snapshot_has_no_fake_alpha(monkeypatch):
    X, roles = small_data()
    controlled_estimates(monkeypatch, [0.25, 0.25])

    def check(epoch, fitted):
        if fitted.checkpoint_stage_ == "warm_start":
            with pytest.raises(NotFittedError, match="No TEDn round"):
                fitted.estimate_unlabeled_prior()

    small_model().fit(X, roles, epoch_callback=check)


def test_basic_ledger_and_source_contract_preserve_admission_boundaries():
    root = Path(__file__).resolve().parents[3]
    ledger = json.loads(
        (root / "pu_toolbox/experiment/method_ledger.json").read_text(encoding="utf-8")
    )
    cvir = ledger["methods"]["cvir"]
    extension = cvir["independent_extensions"]["tedn"]
    review = json.loads((root / extension["review_ref"]).read_text(encoding="utf-8"))
    implementation = review["implementation"]
    assert extension["implementation"].endswith("tedn.py:TEDnClassifier")
    assert implementation["status"] == "end_to_end_training_engineering_implemented"
    for key in ("calibration_applied", "calibration_hooked", "formal_admission"):
        assert extension[key] is implementation[key] is False
    assert implementation["registered_in_cli_ui_or_os_pipeline"] is False
    assert implementation["wired_to_existing_cvir"] is False
    assert implementation["fixed_population_prior_accepted"] is False
    assert review["owner_signatures"] == []
    assert cvir["calibration_applied"] is cvir["calibration_hooked"] is False


def test_basic_cnn_real_bbe_alternation_without_stub():
    rng = np.random.RandomState(42)
    X = np.r_[
        rng.normal(2, 0.4, (600, 1, 4, 4)),
        rng.normal(2, 0.4, (360, 1, 4, 4)),
        rng.normal(-2, 0.4, (840, 1, 4, 4)),
    ]
    roles = np.r_[np.ones(600), np.zeros(1200)]
    torch.manual_seed(12)
    encoder = torch.nn.Sequential(
        torch.nn.Conv2d(1, 3, 1),
        torch.nn.Tanh(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
    )
    original = copy.deepcopy(encoder.state_dict())
    estimator = large_model(encoder=encoder).fit(X, roles)
    assert estimator.n_iter_ == len(estimator.bbe_history_) == 4
    assert all(0 < alpha < 1 for alpha in estimator.history_["alpha_u"])
    assert estimator.optimizer_steps_ == 45 + sum(
        max(8, int(np.ceil(count / 64))) for count in estimator.history_["selected_negative_count"]
    )
    assert estimator.predict(X).shape == (1800,)
    assert any(
        not torch.equal(original[name], tensor)
        for name, tensor in estimator.encoder_.state_dict().items()
    )
    for name, tensor in encoder.state_dict().items():
        torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
