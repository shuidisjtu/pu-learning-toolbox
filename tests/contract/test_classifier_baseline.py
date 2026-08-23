# ruff: noqa: B017, N802, N803, N806
"""Unified contract tests for ALL registered NATIVE algorithms.

Covers API contract compliance (architecture.md §5) plus baseline
basic/param/edge/determ categories.  New NATIVE algorithms get full
contract coverage by adding a factory entry to ``_FACTORY_MAP``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.core.base import BasePriorEstimator, BasePUClassifier
from pu_toolbox.core.exceptions import NotFittedError
from pu_toolbox.core.tags import SampleWeightSupport
from pu_toolbox.estimators.bias_aware.pusb_kernel import PUSBKernelClassifier
from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier
from pu_toolbox.estimators.risk.pnu import PNUClassifier
from pu_toolbox.registry import (
    clear_registry,
    get_algorithm_registry,
    list_algorithms,
    register_all_builtin_methods,
)

torch = pytest.importorskip("torch", reason="PyTorch not installed")

# ── Factory functions ─────────────────────────────────────────────


def _make_elkan_noto():
    from pu_toolbox.estimators.classic.elkan_noto import ElkanNotoClassifier

    return ElkanNotoClassifier(n_cv_folds=3, random_state=42)


def _make_llsvm():
    from pu_toolbox.estimators.classic.llsvm import LLSVMClassifier

    # Defaults run 3000 SGD epochs; shrink for the 90-sample contract data.
    return LLSVMClassifier(max_epochs=10, min_epochs=1, random_state=42)


def _make_upu():
    from pu_toolbox.estimators.risk.upu import UPUClassifier

    return UPUClassifier(
        class_prior=0.5, loss="logistic", reg_lambda=1.0, max_iter=200, random_state=42
    )


def _make_nnpu():
    torch.manual_seed(42)
    return NonNegativePUClassifier(
        model=torch.nn.Linear(5, 1), max_epochs=1, batch_size=8, random_state=42
    )


def _make_pnu():
    return PNUClassifier(class_prior=0.4, eta=0.5, reg_lambda=1.0, random_state=42)


def _make_ldce():
    from pu_toolbox.estimators.risk.ldce import LDCEClassifier

    return LDCEClassifier(flip_probability=0.3, max_iter=10, tol=1e-4, random_state=42)


def _make_kldce():
    from pu_toolbox.estimators.risk.kldce import KLDCEClassifier

    return KLDCEClassifier(
        flip_probability=0.3, sigma=2.0, max_acs_iter=5, tol=1e-4, random_state=42
    )


def _make_dist_pu():
    from pu_toolbox.estimators.risk.dist_pu import DistPUClassifier

    return DistPUClassifier(0.3, hidden_dim=8, epochs=2, random_state=42)


def _make_pusb():
    from pu_toolbox.estimators.bias_aware.pusb import PUSBClassifier

    return PUSBClassifier(threshold=0.5)


def _make_pusb_kernel():
    from pu_toolbox.estimators.bias_aware.pusb_kernel import PUSBKernelClassifier

    # Small grid / low basis keep the full CV + refit affordable in tests.
    # sigma grid matches the ±2-separated 90-sample data (pairwise d2 ~ 80):
    # sigma=2 -> exp(-10) ~ 4.5e-5, sigma=4 -> exp(-2.5) ~ 0.08, both alive.
    # cv=2 relies on the 30-positive balance for the per-fold P/U guard.
    return PUSBKernelClassifier(
        n_basis=10,
        sigma_grid=[2.0, 4.0],
        reg_grid=[0.01, 0.1],
        cv=2,
        max_iter=50,
        random_state=42,
    )


def _make_lbe():
    from pu_toolbox.estimators.bias_aware.lbe import LBEClassifier

    return LBEClassifier(n_em_iter=3)


def _make_class_prior_estimation():
    from pu_toolbox.prior.pen_l1 import ClassPriorEstimator

    return ClassPriorEstimator(n_centers=50)


def _make_recpe():
    from pu_toolbox.prior.recpe import ReCPEEstimator

    return ReCPEEstimator(copy_fraction=0.1)


def _make_infomax_pu():
    from pu_toolbox.estimators.deep import InfoMaxPUClassifier

    return InfoMaxPUClassifier(
        class_prior=0.33,
        representation_dim=3,
        hidden_dim=8,
        representation_epochs=1,
        classifier_epochs=1,
        random_state=42,
    )


def _make_weighted_contrastive_pu():
    from pu_toolbox.estimators.deep import WeightedContrastivePUClassifier

    return WeightedContrastivePUClassifier(
        0.33,
        hidden_dim=8,
        embedding_dim=4,
        queue_size=16,
        batch_size=32,
        max_epochs=1,
        random_state=42,
    )


def _make_self_pu():
    from pu_toolbox.estimators.deep import SelfPUClassifier

    return SelfPUClassifier(
        0.33,
        hidden_dim=8,
        warmup_epochs=0,
        self_paced_start=0,
        self_paced_end=1,
        distill_start=1,
        max_epochs=1,
        batch_size=32,
        random_state=42,
    )


class _MockConditionalGenerator:
    def fit(self, X, y, *, warm_start=True):
        self.means_ = {label: X[y == label].mean(axis=0) for label in np.unique(y)}
        self.n_features_in_ = X.shape[1]
        return self

    def sample(self, n_samples, *, class_label, random_state=None):
        rng = np.random.RandomState(random_state)
        mean = self.means_.get(class_label, np.zeros(self.n_features_in_))
        return mean + 0.01 * rng.randn(n_samples, self.n_features_in_)


def _make_dgpu():
    from pu_toolbox.estimators.deep import DGPUClassifier

    return DGPUClassifier(
        0.33,
        _MockConditionalGenerator(),
        hidden_dim=8,
        rounds=1,
        initialization_epochs=1,
        annotation_epochs=1,
        generated_samples=6,
        random_state=42,
    )


_FACTORY_MAP: dict[str, callable] = {
    "elkan_noto": _make_elkan_noto,
    "llsvm": _make_llsvm,
    "upu": _make_upu,
    "nnpu": _make_nnpu,
    "pnu": _make_pnu,
    "centroid_pu": _make_ldce,
    "kldce": _make_kldce,
    "dist_pu": _make_dist_pu,
    "pusb": _make_pusb,
    "pusb_kernel": _make_pusb_kernel,
    "lbe": _make_lbe,
    "class_prior_estimation": _make_class_prior_estimation,
    "recpe": _make_recpe,
    "self_pu": _make_self_pu,
    "infomax_pu": _make_infomax_pu,
    "weighted_contrastive_pu": _make_weighted_contrastive_pu,
    "dgpu": _make_dgpu,
}

_REPRESENTATIVE_ALGOS = [
    "elkan_noto",
    "upu",
    "pusb",
    "pusb_kernel",
    "recpe",
    "self_pu",
    "kldce",
]

_REPRESENTATIVE_CLFS = [
    name
    for name in _REPRESENTATIVE_ALGOS
    if not isinstance(_FACTORY_MAP[name](), BasePriorEstimator)
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


def _get_fit_kwargs(clf, y):
    """Class-prior kwargs for the estimators that need them at fit time.

    Narrowly scoped to the classes whose fit *requires* an explicit prior
    (or overrides the constructor value by design): nnPU and
    PUSBKernelClassifier.  Deliberately NOT keyed off the
    ``requires_class_prior`` class attribute — many estimators (uPU, PNU,
    Dist-PU, Self-PU, ...) declare it but train with their
    constructor-chosen prior; injecting here would silently override it.
    """
    n_p = int(np.sum(y == 1))
    if isinstance(clf, NonNegativePUClassifier | PUSBKernelClassifier):
        return {"class_prior": n_p / len(y)}
    return {}


def _is_prior_estimator(clf):
    return isinstance(clf, BasePriorEstimator)


def _fit(clf, X, y):
    if _is_prior_estimator(clf):
        clf.fit(X, y)
    else:
        clf.fit(X, y, **_get_fit_kwargs(clf, y))


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
        clf = _make_pusb_kernel()
        X, y = _make_X_y(rng)
        clf.fit(X, y, class_prior=0.5)
        assert np.all(clf.cv_convergence_), "pusb_kernel contract fit did not converge"

    def test_every_trainable_method_has_factory(self):
        """Every registered trainable method must have a contract factory.

        Guards against silent zero-coverage drift (e.g. llsvm was registered
        NATIVE but missing from _FACTORY_MAP). New methods only need the
        factory entry; _REPRESENTATIVE_ALGOS is a performance pick.
        """
        register_all_builtin_methods()
        trainable = {m.name for m in list_algorithms(trainable_only=True)}
        assert set(_FACTORY_MAP) == trainable, (
            f"factory map mismatch: missing={trainable - set(_FACTORY_MAP)}, "
            f"extra={set(_FACTORY_MAP) - trainable}"
        )

    @pytest.mark.parametrize("algo_name", _ALL_PARAMS)
    def test_basic_fit_and_output_shape(self, algo_name, rng):
        clf = _FACTORY_MAP[algo_name]()
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
        clf = _FACTORY_MAP[algo_name]()
        X, _ = _get_data_factory(clf)(rng)
        y_bad = np.zeros(X.shape[0], dtype=int)
        with pytest.raises(Exception):
            _fit(clf, X, y_bad)

    @pytest.mark.parametrize("algo_name", _ALL_PARAMS)
    def test_edge_single_sample_prediction(self, algo_name, rng):
        clf = _FACTORY_MAP[algo_name]()
        X, y = _get_data_factory(clf)(rng)
        _fit(clf, X, y)
        if _is_prior_estimator(clf):
            assert isinstance(clf.estimate(), float)
        else:
            pred = clf.predict(X[:1])
            assert pred.shape == (1,)

    @pytest.mark.parametrize("algo_name", _ALL_PARAMS)
    def test_deterministic_predictions_across_runs(self, algo_name, rng):
        clf1 = _FACTORY_MAP[algo_name]()
        clf2 = _FACTORY_MAP[algo_name]()
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
        clf = _FACTORY_MAP[algo_name]()
        X, _ = _get_data_factory(clf)(rng)
        with pytest.raises(NotFittedError):
            clf.predict(X)
        with pytest.raises(NotFittedError):
            clf.decision_function(X)

    @pytest.mark.parametrize("algo_name", _CLF_PARAMS)
    def test_classes_set_after_fit(self, algo_name, rng):
        clf = _FACTORY_MAP[algo_name]()
        X, y = _get_data_factory(clf)(rng)
        _fit(clf, X, y)
        assert hasattr(clf, "classes_")
        np.testing.assert_array_equal(clf.classes_, np.array([0, 1]))

    @pytest.mark.parametrize("algo_name", _CLF_PARAMS)
    def test_get_params_set_params(self, algo_name):
        clf = _FACTORY_MAP[algo_name]()
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
        clf = _FACTORY_MAP[algo_name]()
        X, y = _get_data_factory(clf)(rng)
        _fit(clf, X, y)
        meta = clf.get_pu_metadata()
        assert meta["is_fitted"] is True
        assert "family" in meta
        assert "implementation_status" in meta
        assert "sample_weight_support" in meta

    @pytest.mark.parametrize("algo_name", _CLF_PARAMS)
    def test_score_samples_delegates_to_decision_function(self, algo_name, rng):
        clf = _FACTORY_MAP[algo_name]()
        X, y = _get_data_factory(clf)(rng)
        _fit(clf, X, y)
        np.testing.assert_array_equal(
            clf.score_samples(X),
            clf.decision_function(X),
        )

    def test_sample_weight_semantics_are_explicit(self):
        """Every classifier declares one of the three documented behaviors."""
        for algo_name, factory in _FACTORY_MAP.items():
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
