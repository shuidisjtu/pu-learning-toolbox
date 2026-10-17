"""TEDn unweighted math, native-TS contracts and real synthetic alternation."""

# Public feature names follow the estimator API.
# ruff: noqa: N803, N806

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox.core.exceptions import NotFittedError, ValidationError  # noqa: E402
from pu_toolbox.estimators.deep import TEDnClassifier  # noqa: E402
from pu_toolbox.estimators.deep.tedn import (  # noqa: E402
    _tedn_objective,
    select_tedn_negatives,
)
from pu_toolbox.estimators.risk.cvir import CVIRClassifier  # noqa: E402

pytestmark = pytest.mark.unit


def data():
    rng = np.random.RandomState(42)
    X = np.r_[
        rng.normal(2, 0.4, (600, 2)),
        rng.normal(2, 0.4, (360, 2)),
        rng.normal(-2, 0.4, (840, 2)),
    ]
    return X, np.r_[np.ones(600), np.zeros(1200)]


def model(**kwargs):
    params = dict(
        input_sampling="ts",
        random_state=7,
        hidden_dim=8,
        warm_start_epochs=3,
        max_epochs=4,
        batch_size=64,
        learning_rate=0.03,
        device="cpu",
    )
    params.update(kwargs)
    return TEDnClassifier(**params)


@pytest.mark.math
def test_basic_unweighted_objective_and_gradients_not_cvir_prior_weighting():
    p = torch.tensor([-2.0, 1.0, 3.0], dtype=torch.float64, requires_grad=True)
    n = torch.tensor([-1.0, 2.0], dtype=torch.float64, requires_grad=True)
    actual = _tedn_objective(p, n)
    lp = np.logaddexp(0, -p.detach().numpy()).mean()
    ln = np.logaddexp(0, n.detach().numpy()).mean()
    assert actual.item() == pytest.approx(lp + ln, abs=1e-14)
    assert actual.item() != pytest.approx(0.3 * lp + 0.7 * ln)
    assert actual.item() != pytest.approx((lp + ln) / 2)
    actual.backward()
    np.testing.assert_allclose(p.grad.numpy(), (torch.sigmoid(p).detach().numpy() - 1) / 3)
    np.testing.assert_allclose(n.grad.numpy(), torch.sigmoid(n).detach().numpy() / 2)


@pytest.mark.parametrize("alpha,count", [(0, 7), (0.3, 5), (0.999, 1)])
def test_edge_discard_floor_count_stable_ties_and_zero_alpha(alpha, count):
    scores = np.array([2, 1, 1, 3, 4, 4, 5])
    selected = select_tedn_negatives(scores, alpha)
    np.testing.assert_array_equal(selected, [1, 2, 0, 3, 4, 5, 6][:count])
    assert len(selected) == count


@pytest.mark.parametrize("alpha", [True, np.nan, np.inf, -0.1, 1.1, "0.3"])
def test_param_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        select_tedn_negatives([0, 1], alpha)


@pytest.mark.parametrize("scores", [[], [[0.2]], [np.nan], [np.inf], [True], [1j], ["0.2"]])
def test_param_invalid_scores(scores):
    with pytest.raises(ValueError, match="scores"):
        select_tedn_negatives(scores, 0.3)


def test_edge_alpha_one_fails_without_clipping_or_forced_negative():
    with pytest.raises(ValueError, match="no retained U"):
        select_tedn_negatives(np.zeros(20), 1)


@pytest.mark.parametrize("seed", [0, 7])
def test_basic_real_bbe_end_to_end_role_isolation_and_all_optimizer_cost(seed):
    X, roles = data()
    estimator = model(random_state=seed).fit(
        X, roles, sample_ids=np.array([f"row-{i}" for i in range(len(X))])
    )
    tp, tu = estimator.train_positive_indices_, estimator.train_unlabeled_indices_
    hp, hu = estimator.holdout_positive_indices_, estimator.holdout_unlabeled_indices_
    assert set(np.r_[tp, tu]).isdisjoint(np.r_[hp, hu])
    assert sorted(np.r_[tp, tu, hp, hu]) == list(range(len(X)))
    assert np.all(roles[np.r_[tp, hp]] == 1)
    assert np.all(roles[np.r_[tu, hu]] == 0)
    assert (len(tp), len(tu), len(hp), len(hu)) == (480, 960, 120, 240)
    assert estimator.n_iter_ == 4
    assert len(estimator.bbe_history_) == len(estimator.history_["alpha_u"]) == 4
    assert set(estimator.selected_negative_indices_) <= set(tu)
    expected_steps = 3 * max(int(np.ceil(len(tp) / 64)), int(np.ceil(len(tu) / 64)))
    for record, alpha, count in zip(
        estimator.bbe_history_,
        estimator.history_["alpha_u"],
        estimator.history_["selected_negative_count"],
        strict=True,
    ):
        assert record["optimizer_steps_before_update"] == expected_steps
        assert record["alpha_u"] == alpha
        assert 0 <= alpha < 1
        assert count == len(tu) - int(np.floor(alpha * len(tu)))
        expected_steps += max(int(np.ceil(len(tp) / 64)), int(np.ceil(count / 64)))
    assert estimator.optimizer_steps_ == expected_steps
    assert estimator.epochs_completed_ == len(estimator.epoch_provenance_) == 7
    assert estimator.checkpoint_epoch_count == 7
    assert estimator.stop_reason_ == "max_epochs" and not estimator.converged_
    assert estimator.estimate_unlabeled_prior() == estimator.history_["alpha_u"][-1]
    assert estimator.predict(X).shape == estimator.decision_function(X).shape == (len(X),)
    with pytest.raises(NotImplementedError):
        estimator.predict_proba(X)
    assert estimator.training_provenance_["clean_labels_read"] is False
    assert estimator.training_provenance_["population_prior_conversion"] is None
    assert estimator.training_provenance_["formal_admission"] is False


def test_determ_clone_seed_and_minus_one_unlabeled_convention():
    X, roles = data()
    original = model()
    first = clone(original).fit(X, roles)
    second = clone(original).fit(X, np.where(roles == 1, 1, -1))
    np.testing.assert_array_equal(first.decision_function(X), second.decision_function(X))
    assert first.bbe_history_ == second.bbe_history_
    assert not original._is_fitted
    assert not isinstance(original, CVIRClassifier)
    assert original.get_params()["input_sampling"] == "ts"
    assert "unlabeled_positive_prior" not in original.get_params()


@pytest.mark.parametrize(
    "params",
    [
        {"input_sampling": None},
        {"input_sampling": "os"},
        {"input_sampling": "TS"},
        {"holdout_fraction": 0},
        {"holdout_fraction": 1},
        {"holdout_fraction": 0.999},
        {"delta": 0},
        {"gamma": 1},
        {"delta": np.nan},
        {"gamma": True},
        {"hidden_dim": 0},
        {"warm_start_epochs": -1},
        {"max_epochs": 0},
        {"max_epochs": True},
        {"batch_size": 0},
        {"convergence_patience": 0},
        {"convergence_tol": -1},
        {"convergence_tol": np.inf},
        {"convergence_tol": True},
        {"learning_rate": 0},
        {"learning_rate": np.nan},
        {"learning_rate": "0.1"},
    ],
)
def test_param_constructor_validation_before_any_training(params):
    X, roles = data()
    with pytest.raises(ValueError):
        model(**params).fit(X[:], roles)


def test_param_population_prior_weights_and_clean_truth_rejected():
    X, roles = data()
    with pytest.raises(ValueError, match="population class_prior"):
        model().fit(X, roles, class_prior=0.3)
    with pytest.raises(NotImplementedError, match="sample_weight"):
        model().fit(X, roles, sample_weight=np.ones(len(X)))
    with pytest.raises(TypeError, match="y_true"):
        model().fit(X, roles, y_true=roles)


@pytest.mark.parametrize("case", ["one_p", "one_u", "no_u", "duplicate_ids", "nan", "huge", "3d"])
def test_edge_invalid_role_identity_and_input_contracts(case):
    X, roles = data()
    ids = np.array([f"id-{i}" for i in range(len(X))])
    if case == "one_p":
        roles[:] = 0
        roles[0] = 1
    elif case == "one_u":
        roles[:] = 1
        roles[0] = 0
    elif case == "no_u":
        roles[:] = 1
    elif case == "duplicate_ids":
        ids[1] = ids[0]
    elif case == "nan":
        X[0, 0] = np.nan
    elif case == "huge":
        X[0, 0] = 1e300
    else:
        X = X[:, None, :]
    with pytest.raises((ValueError, ValidationError)):
        model().fit(X, roles, sample_ids=ids)


def test_edge_failed_refit_invalidates_previous_model_and_prior():
    X, roles = data()
    estimator = model().fit(X, roles)
    estimator.set_params(delta=0)
    with pytest.raises(ValueError, match="delta"):
        estimator.fit(X, roles)
    with pytest.raises(NotFittedError):
        estimator.predict(X)
    with pytest.raises(NotFittedError):
        estimator.estimate_unlabeled_prior()
    assert not hasattr(estimator, "alpha_u_")


def test_edge_prediction_shape_empty_and_nonfinite():
    X, roles = data()
    estimator = model().fit(X, roles)
    assert estimator.predict(np.empty((0, 2))).shape == (0,)
    with pytest.raises(ValueError, match="shape"):
        estimator.predict(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        estimator.predict(np.full((2, 2), np.nan))


def test_param_invalid_encoder_is_not_silently_ignored():
    X, roles = data()
    with pytest.raises(TypeError, match="encoder"):
        model(encoder="cnn").fit(X, roles)
