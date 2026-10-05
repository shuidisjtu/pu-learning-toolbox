# ruff: noqa: B017, N802, N803, N806
"""Unified contract tests for ALL registered NATIVE algorithms.

Covers API contract compliance (architecture.md §5) plus baseline
basic/param/edge/determ categories.  New NATIVE algorithms get full
contract coverage by adding a factory entry to ``FACTORY_MAP``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.core.base import BasePriorEstimator, BasePUClassifier
from pu_toolbox.core.exceptions import NotFittedError
from pu_toolbox.core.tags import SampleWeightSupport
from pu_toolbox.estimators.risk.pnu import PNUClassifier
from pu_toolbox.registry import (
    clear_registry,
    get_algorithm_registry,
    list_algorithms,
    register_all_builtin_methods,
)
from tests.estimator_factories import FACTORY_MAP, fit_kwargs

torch = pytest.importorskip("torch", reason="PyTorch not installed")


_REPRESENTATIVE_ALGOS = [
    "elkan_noto",
    "upu",
    "pusb",
    "pusb_kernel",
    "recpe",
    "self_pu",
    "robust_pu",
    "split_pu",
    "kldce",
]

_REPRESENTATIVE_CLFS = [
    name
    for name in _REPRESENTATIVE_ALGOS
    if not isinstance(FACTORY_MAP[name](), BasePriorEstimator)
]

_ALL_PARAMS = [pytest.param(name, id=name) for name in _REPRESENTATIVE_ALGOS]
_CLF_PARAMS = [pytest.param(name, id=name) for name in _REPRESENTATIVE_CLFS]


# ── Data factories & helpers ──────────────────────────────────────


def _make_X_y(rng):
    X_pos = rng.randn(30, 5) + 2.0
    X_neg = rng.randn(60, 5) - 2.0
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(30, dtype=int), np.zeros(60, dtype=int)])
    return X, y


def _make_pnu_X_y(rng):
    X_pos = rng.randn(20, 5) + 2.0
    X_neg = rng.randn(30, 5) - 2.0
    X_unl = rng.randn(50, 5)
    X = np.vstack([X_pos, X_neg, X_unl])
    y = np.concatenate(
        [
            np.full(20, 1, dtype=int),
            np.full(30, -1, dtype=int),
            np.zeros(50, dtype=int),
        ]
    )
    return X, y


def _get_data_factory(clf):
    if isinstance(clf, PNUClassifier):
        return _make_pnu_X_y
    return _make_X_y


def _is_prior_estimator(clf):
    return isinstance(clf, BasePriorEstimator)


def _fit(clf, X, y):
    if _is_prior_estimator(clf):
        clf.fit(X, y)
    else:
        clf.fit(X, y, **fit_kwargs(clf, y))


# ══════════════════════════════════════════════════════════════════
# Baseline tests (all algorithms: basic / param / edge / determ)
# ══════════════════════════════════════════════════════════════════


@pytest.mark.contract
class TestBaseline:
    """Baseline coverage for every NATIVE algorithm."""

    def test_pusb_kernel_contract_fit_converges(self, rng):
        """The contract factory's BFGS solves must converge.

        Without this guard, a non-converging fit (max_iter too low for the
        data/grid) would pass the shape/determinism assertions below and fix
        a bad fit into the suite as baseline behavior.
        """
        clf = FACTORY_MAP["pusb_kernel"]()
        X, y = _make_X_y(rng)
        clf.fit(X, y, class_prior=0.5)
        assert np.all(clf.cv_convergence_), "pusb_kernel contract fit did not converge"

    def test_every_trainable_method_has_factory(self):
        """Every registered trainable method must have a contract factory.

        Guards against silent zero-coverage drift (e.g. llsvm was registered
        NATIVE but missing from FACTORY_MAP). New methods only need the
        factory entry; _REPRESENTATIVE_ALGOS is a performance pick.
        """
        register_all_builtin_methods()
        trainable = {m.name for m in list_algorithms(trainable_only=True)}
        assert set(FACTORY_MAP) == trainable, (
            f"factory map mismatch: missing={trainable - set(FACTORY_MAP)}, "
            f"extra={set(FACTORY_MAP) - trainable}"
        )

    @pytest.mark.parametrize("algo_name", _ALL_PARAMS)
    def test_basic_fit_and_output_shape(self, algo_name, rng):
        clf = FACTORY_MAP[algo_name]()
        X, y = _get_data_factory(clf)(rng)
        _fit(clf, X, y)
        if _is_prior_estimator(clf):
            pi = clf.estimate()
            assert 0.0 <= pi <= 1.0
        else:
            pred = clf.predict(X)
            assert pred.shape == (X.shape[0],)
            assert pred.dtype == int
            assert set(np.unique(pred)) <= {0, 1}
            scores = clf.decision_function(X)
            assert scores.shape == (X.shape[0],)
            assert np.isfinite(scores).all()

    @pytest.mark.parametrize("algo_name", _ALL_PARAMS)
    def test_param_invalid_labels_raises(self, algo_name, rng):
        clf = FACTORY_MAP[algo_name]()
        X, _ = _get_data_factory(clf)(rng)
        y_bad = np.zeros(X.shape[0], dtype=int)
        with pytest.raises(Exception):
            _fit(clf, X, y_bad)

    @pytest.mark.parametrize("algo_name", _ALL_PARAMS)
    def test_edge_single_sample_prediction(self, algo_name, rng):
        clf = FACTORY_MAP[algo_name]()
        X, y = _get_data_factory(clf)(rng)
        _fit(clf, X, y)
        if _is_prior_estimator(clf):
            assert isinstance(clf.estimate(), float)
        else:
            pred = clf.predict(X[:1])
            assert pred.shape == (1,)

    @pytest.mark.parametrize("algo_name", _ALL_PARAMS)
    def test_deterministic_predictions_across_runs(self, algo_name, rng):
        clf1 = FACTORY_MAP[algo_name]()
        clf2 = FACTORY_MAP[algo_name]()
        X, y = _get_data_factory(clf1)(rng)
        _fit(clf1, X, y)
        _fit(clf2, X, y)
        if _is_prior_estimator(clf1):
            assert clf1.estimate() == pytest.approx(clf2.estimate())
        else:
            np.testing.assert_array_equal(clf1.predict(X), clf2.predict(X))


# ══════════════════════════════════════════════════════════════════
# API contract tests (classifiers only, not prior estimators)
# ══════════════════════════════════════════════════════════════════


@pytest.mark.contract
class TestAPIContract:
    """API contract tests specific to BasePUClassifier subclasses."""

    @pytest.mark.parametrize("algo_name", _CLF_PARAMS)
    def test_not_fitted_raises(self, algo_name, rng):
        clf = FACTORY_MAP[algo_name]()
        X, _ = _get_data_factory(clf)(rng)
        with pytest.raises(NotFittedError):
            clf.predict(X)
        with pytest.raises(NotFittedError):
            clf.decision_function(X)

    @pytest.mark.parametrize("algo_name", _CLF_PARAMS)
    def test_classes_set_after_fit(self, algo_name, rng):
        clf = FACTORY_MAP[algo_name]()
        X, y = _get_data_factory(clf)(rng)
        _fit(clf, X, y)
        assert hasattr(clf, "classes_")
        np.testing.assert_array_equal(clf.classes_, np.array([0, 1]))

    @pytest.mark.parametrize("algo_name", _CLF_PARAMS)
    def test_get_params_set_params(self, algo_name):
        clf = FACTORY_MAP[algo_name]()
        params = clf.get_params()
        assert isinstance(params, dict)
        clf.set_params(**{k: v for k, v in params.items() if v is not None})
        updated = clf.get_params()
        for k, v in params.items():
            if v is None or isinstance(v, torch.nn.Module):
                continue
            assert updated[k] == v

    @pytest.mark.parametrize("algo_name", _CLF_PARAMS)
    def test_metadata_after_fit(self, algo_name, rng):
        clf = FACTORY_MAP[algo_name]()
        X, y = _get_data_factory(clf)(rng)
        _fit(clf, X, y)
        meta = clf.get_pu_metadata()
        assert meta["is_fitted"] is True
        assert "family" in meta
        assert "implementation_status" in meta
        assert "sample_weight_support" in meta

    @pytest.mark.parametrize("algo_name", _CLF_PARAMS)
    def test_score_samples_delegates_to_decision_function(self, algo_name, rng):
        clf = FACTORY_MAP[algo_name]()
        X, y = _get_data_factory(clf)(rng)
        _fit(clf, X, y)
        np.testing.assert_array_equal(
            clf.score_samples(X),
            clf.decision_function(X),
        )

    def test_sample_weight_semantics_are_explicit(self):
        """Every classifier declares one of the three documented behaviors."""
        for algo_name, factory in FACTORY_MAP.items():
            clf = factory()
            if _is_prior_estimator(clf):
                continue
            assert "sample_weight_support" in type(clf).__dict__, (
                f"{algo_name} must explicitly declare sample_weight_support"
            )
            support = clf.sample_weight_support
            assert isinstance(support, SampleWeightSupport)
            base_metadata = BasePUClassifier.get_pu_metadata(clf)
            assert base_metadata["sample_weight_support"] == support.value
            if support == SampleWeightSupport.IGNORED:
                fit_doc = clf.fit.__doc__ or ""
                assert "ignored" in fit_doc.lower(), (
                    f"{algo_name} ignores sample_weight but does not document it"
                )


# ══════════════════════════════════════════════════════════════════
# Registry ↔ native class consistency
# ══════════════════════════════════════════════════════════════════


@pytest.mark.contract
class TestRegistryClassBinding:
    """Every NATIVE registry entry has a valid bound class."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        clear_registry()
        register_all_builtin_methods()
        yield
        clear_registry()

    def test_all_native_entries_have_trainable_class(self):
        from pu_toolbox.registry import get_algorithm

        for meta in get_algorithm_registry().values():
            if not meta.trainable:
                continue
            estimator = get_algorithm(meta.name)
            assert estimator is not None
            assert hasattr(estimator, "fit")
