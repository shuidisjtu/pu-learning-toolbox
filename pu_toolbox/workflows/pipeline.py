# ruff: noqa: N803, N806

"""End-to-end PU workflow orchestration: ``PUPipeline``.

Combines data profiling, class-prior estimation, model training,
PU-stratified cross-validation, and diagnostic reporting into a single
callable workflow so that non-expert users only need ``(X, y_pu)`` and
a method name (or ``"auto"`` for recommender-driven selection).
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.base import clone

from ..core.base import BasePriorEstimator, BasePUClassifier
from ..core.config import POSITIVE_LABEL
from ..core.exceptions import PULearningError, RegistryError, ValidationError
from ..core.validation import validate_pu_X_y
from ..diagnostics.report import build_diagnostic_report
from ..metrics.classification import (
    pu_accuracy,
    pu_auc_roc,
    pu_estimated_precision,
    pu_f1,
    pu_negative_rate,
    pu_recall,
    pu_zero_one_risk,
)
from ..model_selection.split import PUStratifiedKFold
from ..preprocessing.data_profiler import ProfileIssue, profile_pu_data
from ..registry import RecommendationResult, recommend_from_profile
from ..registry.registry import get_algorithm, get_metadata
from .report import CVMetric, PipelineReport, PriorInfo

__all__ = ["DEFAULT_METRICS", "PipelineError", "PUPipeline"]

# Parameters the pipeline itself can always supply when instantiating a
# classifier from a registry name.  Anything else required by the
# constructor blocks auto-instantiation.
_ALWAYS_PROVIDED = {"class_prior", "random_state"}

# Sentinel distinguishing the documented default ``"recpe"`` from an
# explicit user choice (affects ``PriorInfo.auto_selected``).
_UNSET = object()

_BUILTINS_REGISTERED = False


def _ensure_registered() -> None:
    """Lazily register builtin methods (same pattern as the recommender)."""
    global _BUILTINS_REGISTERED  # noqa: PLW0603
    if not _BUILTINS_REGISTERED:
        from ..registry.builtin_methods import register_all_builtin_methods

        register_all_builtin_methods()
        _BUILTINS_REGISTERED = True


class PipelineError(PULearningError):
    """Workflow orchestration failed (unresolvable classifier/prior, etc.)."""


DEFAULT_METRICS: tuple[str, ...] = (
    "pu_zero_one_risk",
    "pu_recall",
    "pu_estimated_precision",
    "pu_auc_roc",
)

_METRIC_ALIASES = {
    "pu_risk": "pu_zero_one_risk",
    "risk": "pu_zero_one_risk",
    "auc": "pu_auc_roc",
    "roc_auc": "pu_auc_roc",
    "recall": "pu_recall",
    "precision": "pu_estimated_precision",
    "accuracy": "pu_accuracy",
    "f1": "pu_f1",
    "negative_rate": "pu_negative_rate",
}

_METRIC_SPECS = {
    "pu_zero_one_risk": {
        "needs_scores": True,
        "needs_prior": True,
        "needs_y_true": False,
        "basis": "class_prior_dependent",
    },
    "pu_recall": {
        "needs_scores": False,
        "needs_prior": False,
        "needs_y_true": False,
        "basis": "pu_observed",
    },
    "pu_estimated_precision": {
        "needs_scores": False,
        "needs_prior": True,
        "needs_y_true": False,
        "basis": "class_prior_dependent",
    },
    "pu_auc_roc": {
        "needs_scores": True,
        "needs_prior": False,
        "needs_y_true": True,
        "basis": "supervised_oracle",
    },
    "pu_accuracy": {
        "needs_scores": False,
        "needs_prior": False,
        "needs_y_true": True,
        "basis": "supervised_oracle",
    },
    "pu_f1": {
        "needs_scores": False,
        "needs_prior": False,
        "needs_y_true": True,
        "basis": "supervised_oracle",
    },
    "pu_negative_rate": {
        "needs_scores": False,
        "needs_prior": False,
        "needs_y_true": False,
        "basis": "pu_observed",
    },
}


def _resolve_metric_names(metrics: Sequence[str] | None) -> list[str]:
    """Normalize user-supplied metric names through the alias table."""
    if metrics is None:
        return list(DEFAULT_METRICS)
    resolved: list[str] = []
    for name in metrics:
        canonical = _METRIC_ALIASES.get(name, name)
        if canonical not in _METRIC_SPECS:
            raise ValueError(
                f"Unknown metric {name!r}. Available: "
                + ", ".join(sorted(_METRIC_SPECS))
                + f" (aliases: {sorted(_METRIC_ALIASES)})"
            )
        if canonical not in resolved:
            resolved.append(canonical)
    if not resolved:
        raise ValueError("metrics must contain at least one metric name.")
    return resolved


class PUPipeline:
    """One-call workflow: profile -> prior -> train -> CV -> evaluate.

    Parameters
    ----------
    classifier : str, BasePUClassifier, or "auto" (default "auto")
        Registry name of a PU classifier, a ready-to-use instance, or
        ``"auto"`` to select the best match via the recommender.  Registry
        names are resolved case-insensitively through aliases.  Classes
        whose constructor requires parameters other than ``class_prior`` /
        ``random_state`` (e.g. ``"ldce"`` needs ``flip_probability``)
        cannot be auto-instantiated from a name; pass an instance instead.
    prior_estimator : str, BasePriorEstimator, or None (default "recpe")
        Class-prior estimator used when the selected method needs a prior
        and none is supplied.  ``"km1"`` / ``"km2"`` map to
        :class:`~pu_toolbox.prior.kernel_mean.KernelMeanPriorEstimator`;
        other strings resolve through the registry.  ``None`` disables
        automatic estimation (missing priors then raise an error).
    cv : int or PU-stratified splitter or None (default None)
        Number of folds (``>= 2``) or a splitter instance with a ``split``
        method (e.g. :class:`~pu_toolbox.model_selection.PUStratifiedKFold`).
        ``None`` uses ``PUStratifiedKFold(5, shuffle=True, random_state=...)``.
    metrics : sequence of str or None (default None)
        Metric names to aggregate per fold; ``None`` uses
        :data:`DEFAULT_METRICS`.  Aliases: ``pu_risk``, ``auc``, ``recall``,
        ``precision``, ``accuracy``, ``f1``, ``negative_rate``.
    random_state : int or None (default 42)
        Seed propagated to profiling, the default CV splitter, and
        auto-instantiated estimators.  User-supplied instances keep their
        own randomness.

    Notes
    -----
    ``y_true`` is only ever used for oracle metrics and selection
    evidence; it is **never** used for class-prior estimation.
    """

    def __init__(
        self,
        *,
        classifier: str | BasePUClassifier = "auto",
        prior_estimator: str | BasePriorEstimator | None | object = _UNSET,
        cv: int | Any | None = None,
        metrics: Sequence[str] | None = None,
        random_state: int | None = 42,
    ) -> None:
        _ensure_registered()
        self.classifier = classifier
        self.random_state = random_state
        self.metrics = _resolve_metric_names(metrics)

        # -- classifier: fail fast on unresolvable names -----------------
        if isinstance(classifier, str) and classifier != "auto":
            self._classifier_cls = _resolve_classifier_name(classifier)
            self._classifier_name = classifier
        elif isinstance(classifier, BasePUClassifier):
            self._classifier_cls = None
            self._classifier_name = type(classifier).__name__
        elif classifier == "auto":
            self._classifier_cls = None
            self._classifier_name = "auto"
        else:
            raise TypeError(
                "classifier must be 'auto', a registry name, or a BasePUClassifier "
                f"instance; got {type(classifier).__name__}."
            )

        # -- prior estimator ---------------------------------------------
        if prior_estimator is _UNSET:
            self._prior_estimator = "recpe"
            self._prior_auto_selected = True
        else:
            self._prior_estimator = prior_estimator
            self._prior_auto_selected = False
        self._prior_cls = self._resolve_prior_estimator_class(self._prior_estimator)

        # -- CV ----------------------------------------------------------
        self.cv = cv
        if isinstance(cv, int):
            if cv < 2:
                raise ValueError(f"cv must be >= 2 folds, got {cv}.")
        elif cv is not None and not hasattr(cv, "split"):
            raise TypeError(
                f"cv must be an int or a splitter with a split() method; got {type(cv).__name__}."
            )

    # ── Constructor helpers ────────────────────────────────────────────

    def _resolve_prior_estimator_class(
        self, prior_estimator: str | BasePriorEstimator | None | object
    ) -> type[BasePriorEstimator] | None:
        """Resolve the prior-estimator *class*; ``None`` disables estimation."""
        if prior_estimator is _UNSET or prior_estimator is None:
            return None
        if isinstance(prior_estimator, BasePriorEstimator):
            return type(prior_estimator)
        if isinstance(prior_estimator, str):
            if prior_estimator in {"km1", "km2"}:
                from ..prior.kernel_mean import KernelMeanPriorEstimator

                return KernelMeanPriorEstimator
            _ensure_registered()
            try:
                cls = get_algorithm(prior_estimator)
            except RegistryError as exc:
                raise PipelineError(
                    f"Unknown prior estimator {prior_estimator!r}. "
                    "Use one of 'recpe', 'pen_l1'/'penl1', 'km1', 'km2', "
                    "or a BasePriorEstimator instance."
                ) from exc
            if not isinstance(cls, type) or not issubclass(cls, BasePriorEstimator):
                raise PipelineError(f"{prior_estimator!r} does not resolve to a prior estimator.")
            return cls
        raise TypeError(
            "prior_estimator must be a registry name, a BasePriorEstimator "
            f"instance, or None; got {type(prior_estimator).__name__}."
        )

    # ── fit_evaluate ───────────────────────────────────────────────────

    def fit_evaluate(
        self,
        X: Any,
        y_pu: np.ndarray,
        *,
        y_true: np.ndarray | None = None,
        class_prior: float | None = None,
    ) -> PipelineReport:
        """Run the full workflow and return a :class:`PipelineReport`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y_pu : array-like of shape (n_samples,)
            PU labels: ``+1`` labeled positive, ``0`` unlabeled.
        y_true : array-like of shape (n_samples,), optional
            True labels ``{0, 1}``.  Used only for oracle metrics and
            selection evidence -- never for class-prior estimation.
        class_prior : float, optional
            Explicit class prior ``(0, 1)``.  Takes precedence over the
            classifier's constructor value and over estimation.
        """
        X, y_pu = validate_pu_X_y(X, y_pu, estimator_name="PUPipeline")
        n_samples = X.shape[0]
        if y_true is not None:
            y_true = _validate_y_true(y_true, n_samples)

        splitter = self._resolved_splitter(X, y_pu)
        n_splits = _resolved_n_splits(splitter, X, y_pu)
        n_pos = int((y_pu == POSITIVE_LABEL).sum())
        if n_pos < n_splits:
            raise ValidationError(
                f"n_labeled_positives ({n_pos}) < n_splits ({n_splits}). "
                "Reduce the number of CV folds or provide more labeled positives."
            )

        # -- Resolve classifier ------------------------------------------
        auto_mode = self._classifier_name == "auto"
        if auto_mode:
            classifier_cls: type[BasePUClassifier] | None = None
            classifier_instance: BasePUClassifier | None = None
        elif self._classifier_cls is not None:
            classifier_cls = self._classifier_cls
            classifier_instance = None
        else:
            classifier_cls = None
            classifier_instance = self.classifier

        # -- Resolve class prior (explicit > constructor > estimate) -----
        if auto_mode:
            needs_prior = True
        elif classifier_cls is not None:
            needs_prior = bool(get_metadata(self._classifier_name).requires_class_prior)
        else:
            needs_prior = bool(classifier_instance.requires_class_prior)
        prior, prior_info = self._resolve_prior(
            X=X,
            y_pu=y_pu,
            class_prior=class_prior,
            classifier_instance=classifier_instance,
            needs_prior=needs_prior,
        )

        # -- Data profile (once, reused by recommender and report) -------
        profile = profile_pu_data(
            X,
            y_pu,
            y_true=y_true,
            class_prior=prior,
            random_state=self.random_state,
        )

        # -- Auto mode: recommend first, then instantiate ---------------
        skipped_candidates: list[dict[str, str]] = []
        recommendation: RecommendationResult | None = None
        if auto_mode:
            recommendation = recommend_from_profile(
                profile,
                class_prior=prior,
                top_k=10,
            )
            classifier_cls, skipped_candidates = _pick_first_instantiable(recommendation.candidates)
            if classifier_cls is None:
                raise PipelineError(
                    "auto mode found no candidate that can be auto-instantiated. "
                    "Pass an explicit classifier= instance (e.g. LDCE needs "
                    "flip_probability). Skipped: "
                    + "; ".join(f"{s['name']} ({s['reason']})" for s in skipped_candidates)
                )
            classifier_name = classifier_cls.__name__

        elif classifier_cls is not None:
            classifier_name = self._classifier_name
        else:
            classifier_name = type(classifier_instance).__name__

        # -- Cross-validation ---------------------------------------------
        per_fold: dict[str, list[float | None]] = {name: [] for name in self.metrics}
        fold_reasons: dict[str, list[str]] = {name: [] for name in self.metrics}
        for train_idx, test_idx in splitter.split(X, y_pu):
            clf = self._fresh_estimator(classifier_cls, classifier_instance, prior)
            clf.fit(X[train_idx], y_pu[train_idx], class_prior=prior)
            y_pu_fold = y_pu[test_idx]
            pred = clf.predict(X[test_idx])
            scores = _extract_scores(clf, X[test_idx])
            y_true_fold = y_true[test_idx] if y_true is not None else None
            for name in self.metrics:
                value, reason = _compute_metric(name, y_pu_fold, pred, scores, y_true_fold, prior)
                per_fold[name].append(value)
                if reason is not None:
                    fold_reasons[name].append(reason)

        cv_metrics = {
            name: CVMetric(
                name=name,
                per_fold=tuple(per_fold[name]),
                basis=_METRIC_SPECS[name]["basis"],
                reason=_aggregate_reason(
                    fold_reasons[name], all(value is None for value in per_fold[name])
                ),
            )
            for name in self.metrics
        }

        # -- Final model (full refit) + final diagnostic ------------------
        final_model = self._fresh_estimator(classifier_cls, classifier_instance, prior)
        final_model.fit(X, y_pu, class_prior=prior)
        if _extract_scores(final_model, X) is not None:
            diagnostic = build_diagnostic_report(
                X,
                y_pu,
                estimator=final_model,
                y_true=y_true,
                class_prior=prior,
                random_state=self.random_state,
            )
        else:
            # Models without a decision function cannot drive estimator-mode
            # diagnostics; fall back to explicit-output mode.
            diagnostic = build_diagnostic_report(
                X,
                y_pu,
                y_pred=final_model.predict(X),
                y_true=y_true,
                class_prior=prior,
                random_state=self.random_state,
            )

        # -- Assemble report ----------------------------------------------
        # NOTE: the profile itself flags a class prior below the labeled
        # positive fraction (inconsistent_class_prior) when supplied.
        issues: list[ProfileIssue] = list(profile.issues)
        if recommendation is not None:
            issues.extend(
                ProfileIssue(
                    "recommender_warning",
                    "warning",
                    message,
                    "Review the method choice or pass an explicit classifier=.",
                )
                for message in recommendation.global_warnings
            )
        issues.extend(
            ProfileIssue(
                f"metric_{name}_unavailable",
                "info",
                metric.reason,
                "Supply the missing input or remove the metric from metrics=.",
            )
            for name, metric in cv_metrics.items()
            if not metric.available and metric.reason
        )

        cv_provenance = _cv_provenance(splitter, n_splits)
        provenance = {
            "classifier": classifier_name,
            "classifier_mode": (
                "auto" if auto_mode else "name" if classifier_cls is not None else "instance"
            ),
            "prior_source": prior_info.source,
            "random_state": self.random_state,
            "y_true_supplied": y_true is not None,
            "skipped_candidates": skipped_candidates,
        }
        return PipelineReport(
            profile=profile,
            recommendation=recommendation,
            prior=prior_info,
            cv_metrics=cv_metrics,
            cv_provenance=cv_provenance,
            final_model=final_model,
            diagnostic=diagnostic,
            issues=tuple(issues),
            provenance=provenance,
        )

    # ── Internal helpers ────────────────────────────────────────────────

    def _resolve_prior(
        self,
        *,
        X: Any,
        y_pu: np.ndarray,
        class_prior: float | None,
        classifier_instance: BasePUClassifier | None,
        needs_prior: bool,
    ) -> tuple[float | None, PriorInfo]:
        """Resolve the class prior: explicit > constructor > estimation.

        ``y_true`` is deliberately not a parameter: class priors are never
        estimated from ground-truth labels.
        """
        if class_prior is not None:
            _validate_prior_value(class_prior, "class_prior")
            return class_prior, PriorInfo(
                value=class_prior, source="user", method_requires_prior=needs_prior
            )

        if classifier_instance is not None:
            carried = classifier_instance.get_params().get("class_prior")
            if carried is not None:
                _validate_prior_value(carried, "class_prior")
                return float(carried), PriorInfo(
                    value=float(carried),
                    source="constructor",
                    method_requires_prior=needs_prior,
                )

        prior_spec: str | BasePriorEstimator | None = self._prior_estimator
        if not needs_prior:
            return None, PriorInfo(value=None, source="none", method_requires_prior=False)
        if prior_spec is None:
            raise PipelineError(
                "The selected method requires a class prior, but none was "
                "provided or estimated (prior_estimator=None). Pass "
                "class_prior=..., configure prior_estimator=..., or choose "
                "a method that does not require a prior. Note: y_true is "
                "never used for class-prior estimation."
            )

        # Estimation (full data, once; never on y_true).
        estimator_name: str
        if isinstance(prior_spec, BasePriorEstimator):
            estimator = prior_spec
            estimator_name = type(estimator).__name__
        else:
            estimator = self._instantiate_prior(prior_spec)
            estimator_name = str(prior_spec)
        try:
            estimator.fit(X, y_pu)
            value = float(estimator.estimate())
        except Exception as exc:
            raise PipelineError(
                f"Class-prior estimation with {estimator_name!r} failed: {exc}"
            ) from exc
        _validate_prior_value(value, f"prior estimate from {estimator_name}")
        return value, PriorInfo(
            value=value,
            source="estimated",
            method_requires_prior=needs_prior,
            estimator=estimator_name,
            auto_selected=self._prior_auto_selected,
        )

    def _instantiate_prior(self, name: str) -> BasePriorEstimator:
        cls = self._prior_cls
        kwargs: dict[str, Any] = {}
        if cls is not None and issubclass(cls, BasePriorEstimator):
            # KernelMeanPriorEstimator requires variant selection.
            from ..prior.kernel_mean import KernelMeanPriorEstimator

            if cls is KernelMeanPriorEstimator:
                kwargs["variant"] = name
        return cls(**kwargs)

    def _fresh_estimator(
        self,
        cls: type[BasePUClassifier] | None,
        instance: BasePUClassifier | None,
        prior: float | None,
    ) -> BasePUClassifier:
        """Clone an instance, or instantiate a class with injected params."""
        if instance is not None:
            return clone(instance)
        assert cls is not None
        kwargs: dict[str, Any] = {}
        signature = inspect.signature(cls.__init__)
        if prior is not None and "class_prior" in signature.parameters:
            kwargs["class_prior"] = prior
        if "random_state" in signature.parameters and self.random_state is not None:
            kwargs["random_state"] = self.random_state
        return cls(**kwargs)

    def _resolved_splitter(self, X: Any, y_pu: np.ndarray) -> Any:
        if isinstance(self.cv, int):
            return PUStratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        if self.cv is None:
            return PUStratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        return self.cv


# ── Module-level helpers ────────────────────────────────────────────────


def _resolve_classifier_name(name: str) -> type[BasePUClassifier]:
    """Resolve a registry name, rejecting classes we cannot instantiate."""
    try:
        cls = get_algorithm(name)
    except RegistryError as exc:
        raise PipelineError(
            f"Unknown classifier {name!r}. Use 'auto' or a registered method "
            "name (see list_algorithms())."
        ) from exc
    if not isinstance(cls, type) or not issubclass(cls, BasePUClassifier):
        raise PipelineError(f"{name!r} does not resolve to a PU classifier.")
    missing = _missing_required_params(cls)
    if missing:
        raise PipelineError(
            f"Classifier {name!r} cannot be auto-instantiated: constructor "
            f"requires {sorted(missing)}. Pass an instance instead."
        )
    return cls


def _missing_required_params(cls: type) -> set[str]:
    """Required constructor params the pipeline cannot supply."""
    signature = inspect.signature(cls.__init__)
    return {
        param
        for param, p in signature.parameters.items()
        if param != "self"
        and p.default is inspect.Parameter.empty
        and param not in _ALWAYS_PROVIDED
    }


def _pick_first_instantiable(
    candidates: Sequence[Any],
) -> tuple[type[BasePUClassifier] | None, list[dict[str, str]]]:
    """Pick the top-ranked candidate we can instantiate; record skips."""
    skipped: list[dict[str, str]] = []
    for candidate in candidates:
        try:
            cls = get_algorithm(candidate.name)
        except RegistryError:
            skipped.append({"name": candidate.name, "reason": "not registered"})
            continue
        if not (isinstance(cls, type) and issubclass(cls, BasePUClassifier)):
            # Registry also holds prior estimators; the pipeline needs a classifier.
            skipped.append({"name": candidate.name, "reason": "not a PU classifier"})
            continue
        missing = _missing_required_params(cls)
        if missing:
            skipped.append(
                {
                    "name": candidate.name,
                    "reason": f"constructor requires {sorted(missing)}",
                }
            )
            continue
        return cls, skipped
    return None, skipped


def _extract_scores(clf: BasePUClassifier, X: Any) -> np.ndarray | None:
    """Best-effort score extraction: decision_function, then predict_proba."""
    if hasattr(clf, "decision_function"):
        try:
            return np.asarray(clf.decision_function(X), dtype=float)
        except Exception:  # noqa: BLE001 - scores are best-effort
            pass
    if hasattr(clf, "predict_proba"):
        try:
            proba = np.asarray(clf.predict_proba(X), dtype=float)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
        except Exception:  # noqa: BLE001 - scores are best-effort
            pass
    return None


def _compute_metric(
    name: str,
    y_pu_fold: np.ndarray,
    pred: np.ndarray,
    scores: np.ndarray | None,
    y_true_fold: np.ndarray | None,
    prior: float | None,
) -> tuple[float | None, str | None]:
    """Compute one metric for one fold; return (value, skip reason)."""
    spec = _METRIC_SPECS[name]
    if spec["needs_scores"] and scores is None:
        return None, "score-based metric requires a decision function"
    if spec["needs_prior"] and prior is None:
        return None, "class-prior-dependent metric requires a class prior"
    if spec["needs_y_true"] and y_true_fold is None:
        return None, "supervised-oracle metric requires y_true"
    try:
        if name == "pu_zero_one_risk":
            return pu_zero_one_risk(y_pu_fold, scores, prior), None
        if name == "pu_recall":
            return pu_recall(y_pu_fold, pred), None
        if name == "pu_estimated_precision":
            return pu_estimated_precision(y_pu_fold, pred, prior), None
        if name == "pu_auc_roc":
            return pu_auc_roc(y_true_fold, scores), None
        if name == "pu_accuracy":
            return pu_accuracy(y_true_fold, pred), None
        if name == "pu_f1":
            return pu_f1(y_true_fold, pred), None
        if name == "pu_negative_rate":
            return pu_negative_rate(y_pu_fold, pred), None
    except ValueError as exc:
        return None, f"fold metric failed: {exc}"
    raise PipelineError(f"Unreachable metric {name!r}.")


def _aggregate_reason(reasons: list[str], all_skipped: bool) -> str | None:
    """First reason when every fold was skipped, else None."""
    return reasons[0] if all_skipped and reasons else None


def _validate_y_true(y_true: np.ndarray, n_samples: int) -> np.ndarray:
    """Validate true labels: 1-D, matching length, values in {0, 1}."""
    y_true = np.asarray(y_true)
    if y_true.ndim != 1 or len(y_true) != n_samples:
        raise ValidationError(
            f"y_true must be 1-D with length {n_samples}; got shape {y_true.shape}."
        )
    unique = set(np.unique(y_true))
    if not unique <= {0, 1}:
        raise ValidationError(f"y_true must contain only {{0, 1}}; got {sorted(unique)}.")
    return y_true.astype(int, copy=False)


def _validate_prior_value(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise PipelineError(f"{name} must be in (0, 1); got {value!r}.")


def _resolved_n_splits(splitter: Any, X: Any, y_pu: np.ndarray) -> int:
    if hasattr(splitter, "get_n_splits"):
        return int(splitter.get_n_splits(X, y_pu))
    return 5


def _cv_provenance(splitter: Any, n_splits: int) -> dict[str, Any]:
    info: dict[str, Any] = {"n_splits": n_splits, "splitter": type(splitter).__name__}
    try:
        params = splitter.get_params()
        info["shuffle"] = params.get("shuffle")
        info["random_state"] = params.get("random_state")
    except Exception:  # noqa: BLE001 - provenance is best-effort
        pass
    return info
