"""Cross-classifier API contract tests.

Verifies that every NATIVE classifier follows the BasePUClassifier
API contract consistently (architecture.md §5).
"""

# ruff: noqa: N802, N806

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.core.exceptions import NotFittedError
from pu_toolbox.estimators.classic.elkan_noto import ElkanNotoClassifier
from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier
from pu_toolbox.estimators.risk.upu import UPUClassifier
from pu_toolbox.registry import (
    clear_registry,
    get_algorithm_registry,
    register_all_builtin_methods,
)

torch = pytest.importorskip("torch", reason="PyTorch not installed")

# ── Factory functions (fresh instance per test) ──────────────────────


def _make_elkan_noto():
    return ElkanNotoClassifier(n_cv_folds=3, random_state=42)


def _make_upu():
    return UPUClassifier(
        class_prior=0.5, loss="logistic", reg_lambda=1.0,
        max_iter=200, random_state=42,
    )


def _make_nnpu():
    return NonNegativePUClassifier(
        model=torch.nn.Linear(5, 1), max_epochs=1, batch_size=8,
        random_state=42,
    )


_CLASSIFIER_FACTORIES = [
    pytest.param(_make_elkan_noto, id="ElkanNotoClassifier"),
    pytest.param(_make_upu, id="UPUClassifier"),
    pytest.param(_make_nnpu, id="NonNegativePUClassifier"),
]


def _make_X_y(rng):
    """Minimal (X, y_pu) for smoke-fitting all classifiers."""
    X_pos = rng.randn(30, 5) + 2.0
    X_neg = rng.randn(60, 5) - 2.0
    X = np.vstack([X_pos, X_neg])
    y_pu = np.concatenate([np.ones(30, dtype=int), np.zeros(60, dtype=int)])
    return X, y_pu


def _get_fit_kwargs(clf, y_pu) -> dict:
    """Return extra kwargs needed by fit()."""
    if isinstance(clf, NonNegativePUClassifier):
        n_p = int(np.sum(y_pu == 1))
        return {"class_prior": n_p / len(y_pu)}
    return {}


# ═════════════════════════════════════════════════════════════════════
# API contract
# ═════════════════════════════════════════════════════════════════════


class TestAPIContract:
    """Every NATIVE classifier obeys the same public API."""

    @pytest.mark.parametrize("factory", _CLASSIFIER_FACTORIES)
    def test_not_fitted_predict_raises(self, factory, rng):
        clf = factory()
        X, _ = _make_X_y(rng)
        with pytest.raises(NotFittedError):
            clf.predict(X)

    @pytest.mark.parametrize("factory", _CLASSIFIER_FACTORIES)
    def test_not_fitted_decision_function_raises(self, factory, rng):
        clf = factory()
        X, _ = _make_X_y(rng)
        with pytest.raises(NotFittedError):
            clf.decision_function(X)

    @pytest.mark.parametrize("factory", _CLASSIFIER_FACTORIES)
    def test_classes_set_after_fit(self, factory, rng):
        clf = factory()
        X, y_pu = _make_X_y(rng)
        clf.fit(X, y_pu, **_get_fit_kwargs(clf, y_pu))
        assert hasattr(clf, "classes_")
        np.testing.assert_array_equal(clf.classes_, np.array([0, 1]))

    @pytest.mark.parametrize("factory", _CLASSIFIER_FACTORIES)
    def test_predict_returns_binary(self, factory, rng):
        clf = factory()
        X, y_pu = _make_X_y(rng)
        clf.fit(X, y_pu, **_get_fit_kwargs(clf, y_pu))
        pred = clf.predict(X)
        assert pred.dtype == int
        assert set(np.unique(pred)) <= {0, 1}

    @pytest.mark.parametrize("factory", _CLASSIFIER_FACTORIES)
    def test_decision_function_shape(self, factory, rng):
        clf = factory()
        X, y_pu = _make_X_y(rng)
        clf.fit(X, y_pu, **_get_fit_kwargs(clf, y_pu))
        scores = clf.decision_function(X)
        assert scores.shape == (X.shape[0],)
        assert np.isfinite(scores).all()

    @pytest.mark.parametrize("factory", _CLASSIFIER_FACTORIES)
    def test_get_params_set_params(self, factory):
        clf = factory()
        params = clf.get_params()
        assert isinstance(params, dict)
        # set_params updates in-place and returns self
        clf.set_params(**{k: v for k, v in params.items() if v is not None})
        updated = clf.get_params()
        for k, v in params.items():
            if v is None:
                continue
            # nn.Module objects may compare by identity, skip them
            if isinstance(v, torch.nn.Module):
                continue
            assert updated[k] == v

    @pytest.mark.parametrize("factory", _CLASSIFIER_FACTORIES)
    def test_metadata_after_fit(self, factory, rng):
        clf = factory()
        X, y_pu = _make_X_y(rng)
        clf.fit(X, y_pu, **_get_fit_kwargs(clf, y_pu))
        meta = clf.get_pu_metadata()
        assert meta["is_fitted"] is True
        assert "family" in meta
        assert "implementation_status" in meta

    @pytest.mark.parametrize("factory", _CLASSIFIER_FACTORIES)
    def test_score_samples_delegates_to_decision_function(self, factory, rng):
        clf = factory()
        X, y_pu = _make_X_y(rng)
        clf.fit(X, y_pu, **_get_fit_kwargs(clf, y_pu))
        np.testing.assert_array_equal(
            clf.score_samples(X), clf.decision_function(X),
        )


# ═════════════════════════════════════════════════════════════════════
# Registry ↔ native class consistency
# ═════════════════════════════════════════════════════════════════════


class TestRegistryClassBinding:
    """Every NATIVE registry entry has a valid bound class."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        clear_registry()
        register_all_builtin_methods()
        yield
        clear_registry()

    def test_all_native_entries_have_trainable_class(self):
        """get_algorithm() succeeds for every NATIVE entry."""
        from pu_toolbox.registry import get_algorithm

        for meta in get_algorithm_registry().values():
            if not meta.trainable:
                continue
            estimator = get_algorithm(meta.name)
            assert estimator is not None, (
                f"NATIVE method {meta.name} has no bound class"
            )
            assert hasattr(estimator, "fit"), (
                f"NATIVE method {meta.name} class has no fit()"
            )

    def test_four_native_methods_bound(self):
        """Exactly 4 native methods are trainable."""
        from pu_toolbox.registry import list_algorithms

        trainable = list_algorithms(trainable_only=True)
        names = {m.name for m in trainable}
        assert names == {"elkan_noto", "upu", "nnpu", "recpe"}
