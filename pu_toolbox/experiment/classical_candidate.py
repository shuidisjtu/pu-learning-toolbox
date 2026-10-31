"""Opt-in actual-cost evidence for non-epoch PUET and PU-only Rank Pruning.

Core estimators and their frozen source reviews remain untouched. This adapter
records completed work, not paper reproduction, admission or neural epochs.
"""

from __future__ import annotations

import copy
import inspect
import time

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from .protocols import Trainer
from .tracking import RunTrajectory


class _Recorder:
    def __init__(self):
        self.events = []

    def __deepcopy__(self, memo):
        # sklearn clones each fold's model; the opt-in attempt owns one log.
        # No estimator/data/RNG state is shared between those cloned models.
        return self


class _RecordedProbabilityEstimator(BaseEstimator):
    def __init__(self, base_estimator, recorder):
        self.base_estimator = base_estimator
        self.recorder = recorder

    def fit(self, X, y, sample_weight=None):  # noqa: N803
        event = {
            "fit_index": len(self.recorder.events) + 1,
            "status": "started",
            "training_rows": len(y),
            "positive_rows": int(np.count_nonzero(y == 1)),
            "unlabeled_rows": int(np.count_nonzero(y == 0)),
            "weighted": sample_weight is not None,
            "heldout_prediction_rows": 0,
        }
        self.recorder.events.append(event)
        self._event = event
        started = time.perf_counter()
        try:
            # RP already cloned this wrapper and its nested base estimator.
            # Do not clone again: custom constructors may consume RNG state.
            self.model_ = self.base_estimator
            if sample_weight is None:
                self.model_.fit(X, y)
            else:
                self.model_.fit(X, y, sample_weight=sample_weight)
            self.classes_ = self.model_.classes_
            event["status"] = "completed"
            return self
        except BaseException as exc:
            event.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            event["elapsed_seconds"] = time.perf_counter() - started

    def predict_proba(self, X):  # noqa: N803
        result = self.model_.predict_proba(X)
        self._event["heldout_prediction_rows"] += len(X)
        return result


class ClassicalCandidateTrainer(Trainer):
    """Record each tree or nested fit; return a single model, no fake epochs.

    Inject into an unbound ExperimentRunner via ``config={"trainer": ...}``. Validation
    arrays are intentionally unused here; the runner selects on them later.
    ``last_receipt`` includes failed work for diagnostics but only a successful
    receipt is attached to the fitted model and therefore to its manifest.
    """

    def __init__(self):
        self.last_receipt = None

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None, os_or_ts=None):  # noqa: N803
        from ..estimators.classic.rank_pruning import RankPruningClassifier
        from ..estimators.risk.puet import PUExtraTreesClassifier

        self.last_receipt = None
        estimator._is_fitted = False
        if hasattr(estimator, "training_provenance_"):
            del estimator.training_provenance_
        if type(estimator) not in (PUExtraTreesClassifier, RankPruningClassifier):
            raise TypeError(
                "classical candidate cost adapter supports exact PUET or RP classes only"
            )
        if type(estimator) is PUExtraTreesClassifier and "_build_tree" in estimator.__dict__:
            raise ValueError("PUET instance tree-builder override cannot be instrumented safely")
        if os_or_ts not in (None, "os", "ts"):
            raise ValueError("os_or_ts must be os, ts or None")
        recorder = _Recorder()
        receipt = {
            "schema_version": "classical-candidate-cost-1",
            "method": "puet" if type(estimator) is PUExtraTreesClassifier else "rp",
            "status": "started",
            "execution_mode": "technical_preintegration",
            "formal_admission": False,
            "source_reproduction": False,
            "neural_epoch_count": 0,
            "fitted_inference_models": 0,
            "validation_used_for_training": False,
            "events": recorder.events,
        }
        self.last_receipt = receipt
        kwargs = {} if class_prior is None else {"class_prior": class_prior}
        if os_or_ts is not None:
            kwargs["os_or_ts"] = os_or_ts
        started = time.perf_counter()
        try:
            if type(estimator) is PUExtraTreesClassifier:
                self._fit_forest(estimator, X, y, recorder, kwargs)
                receipt.update(
                    budget_unit="tree_build",
                    trees_requested=estimator.n_estimators,
                    trees_completed=len(estimator.trees_),
                    bootstrap=estimator.bootstrap,
                    n_positive=estimator.n_positive_,
                    n_native_unlabeled=estimator.n_unlabeled_,
                    n_loss_unlabeled=estimator.n_loss_unlabeled_,
                    node_count=sum(2 * int(n) - 1 for n in estimator.n_leaves_),
                )
            else:
                self._fit_pruning(estimator, X, y, recorder, kwargs)
                receipt.update(
                    budget_unit="cross_fits_and_final_refit",
                    cross_fits_completed=estimator.n_cv_folds,
                    final_refits_completed=1,
                    retained_rows=len(estimator.retained_indices_),
                    pruned_rows=int(estimator.pruned_mask_.sum()),
                    rho1=float(estimator.rho1_),
                    external_class_prior_used=False,
                )
            receipt.update(
                status="completed",
                fitted_inference_models=1,
                run_view=estimator.training_view_,
                calibration_applied=bool(estimator.calibration_applied_),
            )
        except BaseException as exc:
            estimator._is_fitted = False
            receipt.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            receipt["elapsed_seconds"] = time.perf_counter() - started
            receipt["work_started"] = len(recorder.events)
            receipt["work_completed"] = sum(e["status"] == "completed" for e in recorder.events)
        estimator.training_provenance_ = copy.deepcopy(receipt)
        return RunTrajectory(epochs=[], model=estimator)

    @staticmethod
    def _fit_forest(estimator, X, y, recorder, kwargs):  # noqa: N803
        original = estimator._build_tree

        def build(features, labels, rows, **options):
            event = {
                "tree_index": len(recorder.events) + 1,
                "status": "started",
                "training_rows": len(rows),
                "positive_rows": int(np.count_nonzero(labels[rows] == 1)),
                "unlabeled_rows": int(np.count_nonzero(labels[rows] == 0)),
                "positive_weight": float(options["positive_weight"]),
                "unlabeled_weight": float(options["unlabeled_weight"]),
            }
            recorder.events.append(event)
            started = time.perf_counter()
            try:
                result = original(features, labels, rows, **options)
                event.update(status="completed", leaves=int(result[2]), depth=int(result[3]))
                return result
            except BaseException as exc:
                event.update(status="failed", error_type=type(exc).__name__)
                raise
            finally:
                event["elapsed_seconds"] = time.perf_counter() - started

        estimator._build_tree = build
        try:
            estimator.fit(X, y, **kwargs)
        finally:
            del estimator._build_tree

    @staticmethod
    def _fit_pruning(estimator, X, y, recorder, kwargs):  # noqa: N803
        original = estimator.base_estimator
        base = (
            original
            if original is not None
            else LogisticRegression(max_iter=1000, random_state=estimator.random_state)
        )
        if not callable(getattr(base, "predict_proba", None)):
            raise TypeError("RP base estimator requires predict_proba")
        if "sample_weight" not in inspect.signature(base.fit).parameters:
            raise TypeError("RP base estimator fit requires sample_weight")
        estimator.base_estimator = _RecordedProbabilityEstimator(base, recorder)
        try:
            estimator.fit(X, y, **kwargs)
        finally:
            estimator.base_estimator = original
            fitted = getattr(estimator, "estimator_", None)
            if isinstance(fitted, _RecordedProbabilityEstimator) and hasattr(fitted, "model_"):
                estimator.estimator_ = fitted.model_
