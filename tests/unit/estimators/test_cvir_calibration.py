"""CVIR calibration restores a train risk pool before nonlinear truncation."""

# ruff: noqa: E402, N803, N806

import copy
import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.core.exceptions import NotFittedError
from pu_toolbox.estimators.risk import cvir
from pu_toolbox.estimators.risk.cvir import CVIRClassifier
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer

pytestmark = pytest.mark.unit


def data(images=False):
    shape = (1, 4, 4) if images else (3,)
    X = np.random.RandomState(42).normal(size=(19, *shape)).astype("float32")
    return X, np.asarray([1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])


def model(*, images=False, **overrides):
    params = dict(
        unlabeled_positive_prior=0.4,
        prior_source="synthetic marginal parameter, not a measured fraction",
        hidden_dim=4,
        warm_start_epochs=1,
        max_epochs=2,
        batch_size=4,
        random_state=17,
        device="cpu",
    )
    if images:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3)
            params["encoder"] = torch.nn.Sequential(
                torch.nn.Conv2d(1, 3, 1),
                torch.nn.Tanh(),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
            )
    return CVIRClassifier(**(params | overrides))


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("warm", [0, 1])
@pytest.mark.parametrize(
    "policy,images", [("resident", False), ("cpu_stream", False), ("auto", True)]
)
def test_basic_determ_calibration_matches_explicit_native_role_expansion(
    seed, warm, policy, images
):
    X, y = data(images)
    P, U = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    # This duplicated-role fixture is an algebraic comparison, NOT a claim of
    # independent TS sampling or an upstream numerical reproduction.
    origins = np.r_[P, U, P]
    expanded_y = np.r_[np.ones(len(P), int), np.zeros(len(X), int)]
    kwargs = dict(
        images=images, random_state=seed, warm_start_epochs=warm, data_device_policy=policy
    )
    calibrated = model(**kwargs).fit(X, y, os_or_ts="ts")
    reference = model(**kwargs, input_sampling="ts").fit(X[origins], expanded_y)
    assert calibrated.history_ == reference.history_
    assert calibrated.epoch_provenance_ == reference.epoch_provenance_
    assert calibrated.optimizer_steps_ == reference.optimizer_steps_
    np.testing.assert_array_equal(
        calibrated.selected_negative_indices_, origins[reference.selected_negative_indices_]
    )
    np.testing.assert_array_equal(calibrated.decision_function(X), reference.decision_function(X))
    for key, value in calibrated.model_.state_dict().items():
        torch.testing.assert_close(value, reference.model_.state_dict()[key], rtol=0, atol=0)


def test_basic_global_pool_ranked_before_truncation_includes_p_but_preserves_positive_role(
    monkeypatch,
):
    X, y = data()
    calls = []

    def retain_last(scores, alpha):
        calls.append((len(scores), alpha))
        return np.asarray([len(scores) - 1])

    monkeypatch.setattr(cvir, "select_cvir_negatives", retain_last)
    fitted = model(warm_start_epochs=0).fit(X, y, os_or_ts="ts")
    assert calls == [(len(X), 0.4), (len(X), 0.4)]
    assert fitted.selected_negative_indices_.tolist() == [np.flatnonzero(y == 1)[-1]]
    assert fitted.n_positive_ == 6 and fitted.n_unlabeled_ == 13
    provenance = fitted.training_provenance_
    assert provenance["n_unlabeled_candidates"] == 19
    assert provenance["positive_rows_added_to_candidates"] == 6
    assert provenance["calibration_scope"] == "train_global_pool_before_truncation"
    assert provenance["empirical_unlabeled_positive_fraction"] is None


@pytest.mark.parametrize("policy", ["resident", "cpu_stream"])
def test_basic_negative_count_uses_restored_pool_and_warm_stage_costs(policy):
    X, y = data()
    fitted = model(data_device_policy=policy).fit(X, y, os_or_ts="ts")
    assert fitted.history_["selected_negative_count"] == [11, 11]
    # ceil(19/4) warm updates + 2 * max(ceil(6/4), ceil(11/4)).
    assert fitted.optimizer_steps_ == 5 + 2 * 3
    assert [r["optimizer_steps"] for r in fitted.epoch_provenance_] == [5, 8, 11]
    assert fitted.training_provenance_["risk_pool_order"] == "original_U_then_P"


@pytest.mark.parametrize("images", [False, True])
def test_basic_checkpoint_and_pickle_replay_preserve_calibrated_scores_and_inputs(tmp_path, images):
    X, y = data(images)
    before = X.copy(), y.copy()
    fitted = model(images=images)
    original_encoder = copy.deepcopy(fitted.encoder.state_dict()) if images else None
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(fitted, X, y, os_or_ts="ts")
    assert len(trajectory.checkpoints) == 3
    expected = fitted.decision_function(X)
    for restored in (trajectory.checkpoints[-1].restore(), pickle.loads(pickle.dumps(fitted))):
        np.testing.assert_allclose(restored.decision_function(X), expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(X, before[0])
    np.testing.assert_array_equal(y, before[1])
    if images:
        for key, value in fitted.encoder.state_dict().items():
            torch.testing.assert_close(value, original_encoder[key], rtol=0, atol=0)


@pytest.mark.parametrize("view", [None, "TS", "", 1])
def test_param_invalid_view_fails_before_training_or_rng(view, monkeypatch):
    X, y = data()

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid view reached RNG")

    monkeypatch.setattr(torch, "manual_seed", forbidden)
    with pytest.raises(ValueError, match="os_or_ts"):
        model().fit(X, y, os_or_ts=view)


def test_edge_native_input_refuses_double_calibration_and_population_prior_remains_rejected():
    X, y = data()
    with pytest.raises(ValueError, match="requires input_sampling=os"):
        model(input_sampling="ts").fit(X, y, os_or_ts="ts")
    with pytest.raises(ValueError, match="class_prior is the population prior"):
        model().fit(X, y, class_prior=0.4, os_or_ts="ts")


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_edge_failed_calibrated_refit_invalidates_public_prediction(error):
    X, y = data()
    fitted = model().fit(X, y)

    def fail(*args):
        raise error("callback failure")

    with pytest.raises(error, match="callback failure"):
        fitted.fit(X, y, os_or_ts="ts", epoch_callback=fail)
    with pytest.raises(NotFittedError):
        fitted.predict(X)
