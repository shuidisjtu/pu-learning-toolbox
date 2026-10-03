# ruff: noqa: N803, N806  # sklearn X/y naming convention
"""Iris smoke regression: fast trainability and interface checks.

Scope and limits (governance plan, phase 5).  This tier checks only that each
native classifier can *train* on a small offline tabular dataset and exposes a
sane prediction interface: output shapes, finiteness, label values, error
behaviour and run-to-run determinism.  It asserts **no** accuracy, AUC, prior
estimate quality or agreement with the true labels.  Iris proves nothing about
PU assumptions, SCAR/SAR validity, class-prior estimation or the paper
protocols; these results are a fast regression signal only and must never be
quoted as evidence.

Label conversion rule (recorded here because it is part of the test).
``sklearn.datasets.load_iris`` is a 3-class *supervised* dataset, so its labels
must be converted before a PU estimator can consume them:

1. binarise on ``y == 1`` (versicolor) against the other two classes,
   giving 50 positives of 150;
2. split train/test **stratified on the true binary labels** with a fixed
   seed -- the plan requires a stratified split, which is the one sanctioned
   use of the true labels;
3. convert **only the training fold**: ``make_pu_labels(..., "scar", c=0.5)``
   for PU semantics and ``make_pnu_labels(...)`` for the PNU-semantics method.

The test-truth labels never leave the fixture: it returns
``(X_train, y_train_pu, X_test)``.  The requirement not to select parameters
from test truth therefore holds by construction rather than by discipline.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from pu_toolbox.core.base import BasePUClassifier
from pu_toolbox.core.exceptions import NotFittedError
from pu_toolbox.core.tags import Backend
from pu_toolbox.preprocessing.pu_labeling import make_pnu_labels, make_pu_labels
from pu_toolbox.registry import get_algorithm, list_algorithms, register_all_builtin_methods
from tests.estimator_factories import FACTORY_MAP, fit_kwargs

pytestmark = pytest.mark.integration

_SEED = 42
_SCAR_C = 0.5
_TEST_SIZE = 0.3
_PNU_NEGATIVES = 30

# Methods that cannot run on this fixture.  Every entry must carry a
# reproducible reason -- an unexplained entry is exactly the "unjustified
# exemption" the governance plan forbids.  Deliberately empty unless a method
# proves inapplicable; see the design spec section 4.3.
_SKIP_REASONS: dict[str, str] = {
    "lagam": (
        "LaGAMClassifier.fit raises ValueError without an explicit clean support "
        "set (support_data=(X_support, y_clean)); the registry marks it "
        "requires_clean_support=True with scenario CASE_CONTROL, so a plain "
        "PU-labelled fixture cannot drive it"
    ),
}


def _candidate_metadata():
    """Registered native classifiers that have a factory, in registry order."""
    register_all_builtin_methods()
    return [
        meta
        for meta in list_algorithms(trainable_only=True)
        if meta.name in FACTORY_MAP and issubclass(get_algorithm(meta.name), BasePUClassifier)
    ]


_CANDIDATES = _candidate_metadata()
_PARAMS = [pytest.param(meta, id=meta.name) for meta in _CANDIDATES]


def _iris_split():
    """Return (X_train, y_true_train, X_test).  Test truth is not returned."""
    X, y = load_iris(return_X_y=True)
    y_bin = (y == 1).astype(int)
    X_train, X_test, y_train, _ = train_test_split(
        X, y_bin, test_size=_TEST_SIZE, stratify=y_bin, random_state=_SEED
    )
    return X_train, y_train, X_test


@pytest.fixture(scope="module")
def iris_pu():
    """(X_train, y_train in {+1, 0}, X_test) via SCAR with c=0.5."""
    X_train, y_train, X_test = _iris_split()
    return X_train, make_pu_labels(y_train, mechanism="scar", c=_SCAR_C, random_state=_SEED), X_test


@pytest.fixture(scope="module")
def iris_pnu():
    """(X_train, y_train in {+1, -1, 0}, X_test) for the PNU-semantics method."""
    X_train, y_train, X_test = _iris_split()
    return X_train, make_pnu_labels(y_train, n_negatives=_PNU_NEGATIVES, random_state=_SEED), X_test


def _skip_reason(meta) -> str | None:
    """None means the method must run; a string means explicit skip with reason."""
    if meta.backend is Backend.TORCH and importlib.util.find_spec("torch") is None:
        return "backend 'torch' requires the optional 'torch' extra"
    return _SKIP_REASONS.get(meta.name)


def _data_for(meta, iris_pu, iris_pnu):
    """Route the PNU-semantics method to its own fixture, everything else to PU."""
    return iris_pnu if meta.label_semantics == "pnu" else iris_pu


def _fit(meta, X, y):
    clf = FACTORY_MAP[meta.name]()
    clf.fit(X, y, **fit_kwargs(clf, y))
    return clf


@pytest.mark.parametrize("meta", _PARAMS)
def test_basic_fit_predict_decision_shape(meta, iris_pu, iris_pnu):
    reason = _skip_reason(meta)
    if reason:
        pytest.skip(reason)
    X_train, y_train, X_test = _data_for(meta, iris_pu, iris_pnu)
    clf = _fit(meta, X_train, y_train)
    decision = np.asarray(clf.decision_function(X_test))
    pred = np.asarray(clf.predict(X_test))
    assert decision.shape == (X_test.shape[0],)
    assert np.isfinite(decision).all()
    assert pred.shape == (X_test.shape[0],)
    assert set(np.unique(pred)).issubset({0, 1})


@pytest.mark.parametrize("meta", _PARAMS)
def test_basic_predict_proba_when_supported(meta, iris_pu, iris_pnu):
    reason = _skip_reason(meta)
    if reason:
        pytest.skip(reason)
    X_train, y_train, X_test = _data_for(meta, iris_pu, iris_pnu)
    clf = _fit(meta, X_train, y_train)
    # BasePUClassifier always exposes predict_proba, but raises
    # NotImplementedError when a subclass does not provide calibrated
    # probabilities, so absence must be detected by calling, not by hasattr.
    try:
        proba = np.asarray(clf.predict_proba(X_test))
    except NotImplementedError:
        pytest.skip("estimator exposes no predict_proba")
    assert proba.shape == (X_test.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("meta", _PARAMS)
def test_param_unfitted_raises(meta, iris_pu, iris_pnu):
    reason = _skip_reason(meta)
    if reason:
        pytest.skip(reason)
    _, _, X_test = _data_for(meta, iris_pu, iris_pnu)
    clf = FACTORY_MAP[meta.name]()
    with pytest.raises(NotFittedError):
        clf.predict(X_test)


@pytest.mark.parametrize("meta", _PARAMS)
def test_edge_single_sample_prediction(meta, iris_pu, iris_pnu):
    reason = _skip_reason(meta)
    if reason:
        pytest.skip(reason)
    X_train, y_train, X_test = _data_for(meta, iris_pu, iris_pnu)
    clf = _fit(meta, X_train, y_train)
    single = X_test[:1]
    assert np.asarray(clf.predict(single)).shape == (1,)
    assert np.asarray(clf.decision_function(single)).shape == (1,)


@pytest.mark.parametrize("meta", _PARAMS)
def test_determ_predictions_reproducible(meta, iris_pu, iris_pnu):
    reason = _skip_reason(meta)
    if reason:
        pytest.skip(reason)
    X_train, y_train, X_test = _data_for(meta, iris_pu, iris_pnu)
    first = _fit(meta, X_train, y_train)
    second = _fit(meta, X_train, y_train)
    assert np.array_equal(first.predict(X_test), second.predict(X_test))
    assert np.allclose(first.decision_function(X_test), second.decision_function(X_test))


def test_basic_skip_declarations_are_consistent():
    candidates = {meta.name for meta in _CANDIDATES}
    # The dynamic rule above fires only when torch is *absent*, so a
    # torch-backed method may legitimately be declared here -- on a machine
    # that has torch, the declared entry is the only thing that skips it.
    # What must never happen is an entry for a method this fixture could
    # actually drive: that would silently drop coverage.  So every declared
    # skip has to be backed by a capability the registry names and this
    # fixture does not supply.
    assert set(_SKIP_REASONS).issubset(candidates)
    assert all(reason.strip() for reason in _SKIP_REASONS.values())
    assert all(meta.requires_clean_support for meta in _CANDIDATES if meta.name in _SKIP_REASONS)
