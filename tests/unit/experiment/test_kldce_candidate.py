"""KLDCE direct-fit parity, explicit prior semantics and honest incomplete cost scope."""

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.base import clone

from pu_toolbox.estimators.risk.kldce import KLDCEClassifier
from pu_toolbox.experiment.kernel_candidate import KLDCECandidateTrainer

pytestmark = pytest.mark.unit


def data(dtype=np.float64):
    features = np.random.RandomState(7).normal(size=(24, 2)).astype(dtype)
    labels = np.r_[np.ones(6, int), np.zeros(18, int)]
    return features, labels


def model(**parameters):
    return KLDCEClassifier(
        **(
            dict(
                flip_probability=0.2,
                mom_groups=2,
                max_acs_iter=10,
                max_inner_iter=1000,
                random_state=3,
            )
            | parameters
        )
    )


def trainer(tmp_path, policy="derived", **overrides):
    return KLDCECandidateTrainer(
        **(
            dict(prior_policy=policy, kernel_payload_cap_bytes=64 * 2**20, receipt_dir=tmp_path)
            | overrides
        )
    )


@pytest.mark.parametrize("policy", ["derived", "population_override"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_basic_determ_prior_choice_direct_fit_and_pickle_parity(tmp_path, policy, dtype):
    features, labels = data(dtype)
    original = features.copy(), labels.copy()
    candidate = model()
    direct = clone(candidate).fit(
        features, labels, **({"class_prior": 0.45} if policy == "population_override" else {})
    )
    adapter = trainer(tmp_path, policy)
    trajectory = adapter.fit(
        candidate, features, labels, class_prior=0.45, val_pu=object(), os_or_ts="os"
    )
    receipt = candidate.training_provenance_
    assert receipt["effective_training_prior"] == (
        0.45 if policy == "population_override" else 6 / (24 * 0.8)
    )
    assert receipt["preflight"]["external_class_prior_forwarded"] is (
        policy == "population_override"
    )
    assert not receipt["validation_used_for_training"] and not receipt["calibration_applied"]
    assert not trajectory.epochs and not trajectory.checkpoints
    assert receipt["neural_epoch_count"] == 0 and receipt["work_completed"] == 1
    assert receipt["acs_rounds_completed"] == direct.n_acs_iter_
    assert (
        receipt["qp_solves_observed"] is None and receipt["total_smo_pair_updates_observed"] is None
    )
    assert receipt["acs_history"] == direct.acs_history_
    assert json.loads(Path(adapter.receipt_paths[0]).read_text()) == receipt
    np.testing.assert_array_equal(
        candidate.decision_function(features), direct.decision_function(features)
    )
    restored = pickle.loads(pickle.dumps(candidate))
    np.testing.assert_array_equal(
        restored.decision_function(features), direct.decision_function(features)
    )
    np.testing.assert_array_equal(features, original[0])
    np.testing.assert_array_equal(labels, original[1])


@pytest.mark.parametrize(
    "parameters,policy,prior,view",
    [
        ({"mom_groups": 19}, "derived", 0.45, "os"),
        ({"max_acs_iter": True}, "derived", 0.45, "os"),
        ({"flip_probability": True}, "derived", 0.45, "os"),
        ({"flip_probability": 0.9}, "derived", 0.45, "os"),
        ({"sigma": 0}, "derived", 0.45, "os"),
        ({"covariance_ridge": -1}, "derived", 0.45, "os"),
        ({}, "population_override", None, "os"),
        ({"flip_probability": 0.8}, "population_override", 0.625, "os"),
        ({}, "derived", 0.45, "ts"),
    ],
)
def test_param_invalid_native_configuration_never_starts_fit(
    tmp_path, monkeypatch, parameters, policy, prior, view
):
    candidate = model(**parameters)
    monkeypatch.setattr(candidate, "fit", lambda *a, **k: pytest.fail("invalid KLDCE reached fit"))
    adapter = trainer(tmp_path, policy)
    with pytest.raises(ValueError):
        adapter.fit(candidate, *data(), class_prior=prior, os_or_ts=view)
    assert not candidate._is_fitted and adapter.last_receipt["work_started"] == 0
    assert adapter.last_receipt["acs_rounds_completed"] is None


def test_edge_named_payload_cap_refuses_before_kernel_or_fit(tmp_path, monkeypatch):
    candidate = model()
    monkeypatch.setattr(candidate, "fit", lambda *a, **k: pytest.fail("screened KLDCE fit"))
    adapter = trainer(tmp_path, kernel_payload_cap_bytes=1)
    with pytest.raises(ValueError, match="payload exceeds"):
        adapter.fit(candidate, *data(), os_or_ts="os")
    assert adapter.last_receipt["work_started"] == 0
    assert len(adapter.receipt_paths) == 1 and not candidate._is_fitted


def test_edge_internal_failure_never_retries_or_reuses_stale_acs_history(tmp_path, monkeypatch):
    candidate = model()
    adapter = trainer(tmp_path)
    adapter.fit(candidate, *data(), os_or_ts="os")
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise TypeError("internal QP failure")

    monkeypatch.setattr(candidate, "fit", fail)
    with pytest.raises(TypeError, match="internal QP failure"):
        adapter.fit(candidate, *data(), os_or_ts="os")
    assert len(calls) == 1 and not candidate._is_fitted
    assert not hasattr(candidate, "training_provenance_")
    assert adapter.last_receipt["work_started"] == 1 and adapter.last_receipt["work_completed"] == 0
    assert adapter.last_receipt["acs_rounds_completed"] is None
    assert len(adapter.receipt_paths) == 2
    assert json.loads(Path(adapter.receipt_paths[0]).read_text())["status"] == "completed"
    assert json.loads(Path(adapter.receipt_paths[1]).read_text())["status"] == "failed"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(prior_policy=None, kernel_payload_cap_bytes=100),
        dict(prior_policy="derived", kernel_payload_cap_bytes=True),
    ],
)
def test_param_explicit_prior_policy_and_positive_memory_cap_required(kwargs):
    with pytest.raises(ValueError):
        KLDCECandidateTrainer(**kwargs)


@pytest.mark.parametrize(
    "h,prior,feasible",
    [(0.2, 0.45, True), (0.5, 0.5, True), (0.8, 0.5, False), (0.9, 0.9, True), (0.9, 0.8, False)],
)
def test_param_edge_equality_box_screen_matches_original_phase_one_lp(tmp_path, h, prior, feasible):
    from pu_toolbox.estimators.risk.kldce import _find_feasible_init

    candidate = model(flip_probability=h)
    adapter = trainer(tmp_path, "population_override")
    features, labels = data()
    rhs = -18 / (2 * 24 * (1 - 2 * prior * h))
    coefficients = np.r_[np.ones(6), -np.ones(18), np.ones(18)].reshape(1, -1)
    lower = np.zeros(42)
    upper = np.r_[np.full(24, 1 / 24), np.full(18, 1 / 48)]
    if feasible:
        screen = adapter.preflight(candidate, features, labels, class_prior=prior, os_or_ts="os")
        point = _find_feasible_init(coefficients, rhs, lower, upper)
        assert screen["qp_equality_box_feasibility"]["rhs"] == rhs
        assert float((coefficients @ point).item()) == pytest.approx(rhs)
    else:
        with pytest.raises(ValueError, match="QP equality infeasible"):
            adapter.preflight(candidate, features, labels, class_prior=prior, os_or_ts="os")
        with pytest.raises(RuntimeError, match="infeasible"):
            _find_feasible_init(coefficients, rhs, lower, upper)


def test_edge_covariance_and_prediction_payloads_are_screened_without_allocating_them(
    tmp_path, monkeypatch
):
    candidate = model()
    monkeypatch.setattr(candidate, "fit", lambda *a, **k: pytest.fail("covariance screen fit"))
    adapter = trainer(tmp_path, kernel_payload_cap_bytes=20 * 2**20)
    features = np.zeros((24, 2048))
    with pytest.raises(ValueError, match="payload exceeds"):
        adapter.fit(candidate, features, data()[1], os_or_ts="os")
    small = trainer(tmp_path, kernel_payload_cap_bytes=1024)
    bundle = SimpleNamespace(
        train=SimpleNamespace(X=np.zeros((2, 1))), pu_val=SimpleNamespace(X=np.zeros((1024, 1)))
    )
    with pytest.raises(ValueError, match="prediction Gram payload"):
        small.prediction_screen(bundle)


def test_edge_realized_frequency_binding_refuses_requested_rate_substitution(tmp_path):
    adapter = trainer(tmp_path)
    meta = dict(mechanism="scar", c_realized=0.5, c_requested=0.3, n_labeled=6)
    adapter.validate_generation(meta, meta)
    assert (
        adapter.preflight(model(flip_probability=0.5), *data(), os_or_ts="os")[
            "generation_binding"
        ]["c_realized"]
        == 0.5
    )
    with pytest.raises(ValueError, match="differs from bound"):
        adapter.preflight(model(flip_probability=0.7), *data(), os_or_ts="os")
