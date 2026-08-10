# ruff: noqa: N803, N806

"""End-to-end PU workflow orchestration: ``PUPipeline``.

Combines data profiling, class-prior estimation, model training,
PU-stratified cross-validation, and diagnostic reporting into a single
callable workflow so that non-expert users only need ``(X, y_pu)`` and
a method name (or ``"auto"`` for recommender-driven selection).
"""

from __future__ import annotations

import contextlib
import inspect
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.base import clone

from ..core.base import BasePriorEstimator, BasePUClassifier
from ..core.config import POSITIVE_LABEL
from ..core.device import resolve_device_name
from ..core.exceptions import PULearningError, RegistryError, ValidationError
from ..core.tags import Backend, TrainingCost
from ..core.validation import (
    check_scalar_in_range,
    validate_pu_X_y,
    validate_true_binary_labels,
)
from ..diagnostics.report import PUDiagnosticReport, build_diagnostic_report
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
from ..preprocessing.data_profiler import ProfileIssue, PUDataProfile, profile_pu_data
from ..registry import RecommendationResult, recommend_from_profile
from ..registry.registry import get_algorithm, get_metadata
from .report import CVMetric, PipelineReport, PriorInfo

__all__ = ["DEFAULT_METRICS", "PipelineError", "PUPipeline"]

# Parameters the pipeline itself can always supply when instantiating a
# classifier from a registry name.  Anything else required by the
# constructor blocks auto-instantiation.
_ALWAYS_PROVIDED = {"class_prior", "random_state"}

# Sentinel distinguishing the documented default ``"pen_l1"`` from an
# explicit user choice (affects ``PriorInfo.auto_selected``).
_UNSET = object()


def _ensure_registered() -> None:
    """Register builtin methods.

    ``register_all_builtin_methods`` is idempotent (registry-level dedup),
    so calling it repeatedly is safe and never goes stale -- unlike the
    old module-level ``_BUILTINS_REGISTERED`` flag, which made the
    pipeline silently run an empty registry after ``clear_registry()``.
    """
    from ..registry.builtin_methods import register_all_builtin_methods

    register_all_builtin_methods()


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
    prior_estimator : str, BasePriorEstimator, or None (default "pen_l1")
        Class-prior estimator used when the selected method needs a prior
        and none is supplied.  ``"km1"`` / ``"km2"`` map to
        :class:`~pu_toolbox.prior.kernel_mean.KernelMeanPriorEstimator`;
        other strings resolve through the registry.  ``None`` disables
        automatic estimation (missing priors then raise an error).
        Since 1.2.1 the default is ``"pen_l1"`` (auto-sigma); ``"recpe"``
        remains available but is best kept for irreducibility-failure
        scenarios.
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
    architecture : str, default "mlp"
        Network architecture for deep classifiers: ``"mlp"`` (table data,
        default) or ``"cnn"`` (4-D NCHW images).  Requires an explicit
        ``classifier`` of ``"wconpu"`` or ``"infomax_pu"``.
    backbone : str, default "cnn13"
        CNN backbone when ``architecture="cnn"``: ``"cnn13"``,
        ``"resnet18"``, or ``"resnet50"``.
    device : str | None, default None
        Torch device passed to deep classifiers (e.g. ``"cuda"``).
        ``None`` / ``"auto"`` auto-detects: CUDA when torch and a GPU
        are available, otherwise CPU.
    max_epochs : int | None, default None
        Max training epochs injected into estimators whose constructor
        signature accepts ``max_epochs`` (wconpu, self_pu, nnpu);
        estimators with other epoch parameter names are left untouched.

    Notes
    -----
    ``y_true`` is only ever used for oracle metrics and selection
    evidence; it is **never** used for class-prior estimation.

    Deep classifiers (``"wconpu"``, ``"infomax_pu"``) with
    ``architecture="cnn"`` expect 4-D NCHW image tensors; class-prior
    estimation and data profiling then run on the flattened view.
    ``"auto"`` mode can select a deep method when the data is large and
    a GPU is available (``device=None``/``"auto"`` resolves to CUDA on
    GPU machines); the selected classifier then gets torch seeding and
    training-cost warnings like an explicit one.
    """

    def __init__(
        self,
        *,
        classifier: str | BasePUClassifier = "auto",
        prior_estimator: str | BasePriorEstimator | None | object = _UNSET,
        cv: int | Any | None = None,
        metrics: Sequence[str] | None = None,
        random_state: int | None = 42,
        architecture: str = "mlp",
        backbone: str = "cnn13",
        device: str | None = None,
        max_epochs: int | None = None,
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

        # -- Deep architecture selection --------------------------------
        if architecture not in {"mlp", "cnn"}:
            raise ValueError("architecture must be 'mlp' or 'cnn'")
        if backbone not in {"cnn13", "resnet18", "resnet50"}:
            raise ValueError("backbone must be 'cnn13', 'resnet18', or 'resnet50'")
        if self._classifier_cls is not None:
            self._is_deep = getattr(self._classifier_cls, "backend", None) == Backend.TORCH
        elif isinstance(classifier, BasePUClassifier):
            self._is_deep = getattr(classifier, "backend", None) == Backend.TORCH
        else:  # auto
            self._is_deep = False
        if architecture == "cnn":
            if not self._is_deep:
                raise PipelineError(
                    "architecture='cnn' requires an explicit deep classifier "
                    "with encoder support (e.g. wconpu or infomax_pu); got "
                    f"classifier={self._classifier_name!r}. For table data use "
                    "architecture='mlp' (default)."
                )
            encoder_cls = self._classifier_cls or type(classifier)
            if not _declares_encoder_parameter(encoder_cls):
                raise PipelineError(
                    f"architecture='cnn' requires classifier "
                    f"{self._classifier_name!r} to declare an 'encoder' "
                    "constructor parameter (encoder injection); this "
                    "algorithm is not yet adapted for CNN architectures. "
                    "Use architecture='mlp' (default) or choose wconpu / "
                    "infomax_pu."
                )
        self.architecture = architecture
        self.backbone = backbone
        self.device = device
        self.max_epochs = max_epochs
        self._encoder = None  # built lazily per fit_evaluate for CNN

        # -- prior estimator ---------------------------------------------
        if prior_estimator is _UNSET:
            self._prior_estimator = "pen_l1"
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
                    "Use one of 'recpe', 'pen_l1', 'km1', 'km2', "
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
        X, y_pu, y_true, analysis_X, splitter, n_splits = self._prepare_inputs(X, y_pu, y_true)

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
            X=analysis_X,
            y_pu=y_pu,
            class_prior=class_prior,
            classifier_instance=classifier_instance,
            needs_prior=needs_prior,
            allow_degradation=auto_mode,
        )

        # -- Data profile (once, reused by recommender and report) -------
        profile = profile_pu_data(
            analysis_X,
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
                class_prior_source=prior_info.source,
                # A resolved CUDA device declares GPU availability to the
                # recommender's GPU dimension; otherwise the GPU bonus
                # (gpu_max) is unreachable and deep methods are never
                # considered on large data.
                has_gpu=resolve_device_name(self.device) == "cuda",
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
            # Recompute the deep flag for the selected method: the
            # recommender can pick a TORCH method on large data, and the
            # training-cost warning / torch seeding below must see it.
            self._is_deep = getattr(classifier_cls, "backend", None) == Backend.TORCH

        elif classifier_cls is not None:
            classifier_name = self._classifier_name
        else:
            classifier_name = type(classifier_instance).__name__

        # -- Training-cost hint ------------------------------------------
        # Deep (TORCH) methods and registry HIGH-cost solvers (e.g. LLSVM
        # SGD, PUSB kernel grid CV) are refit n_splits+1 times; say so
        # instead of silently running for minutes to hours.
        heavy = self._is_deep
        with contextlib.suppress(Exception):  # instance mode: not a registry name
            heavy = heavy or get_metadata(classifier_name).training_cost == TrainingCost.HIGH
        if heavy and n_splits > 1:
            warnings.warn(
                f"{classifier_name} will be trained {n_splits + 1} times "
                "(CV folds + full refit); this method is HIGH-cost "
                "(deep training or heavy grid search). Reduce the number "
                "of folds for quicker runs.",
                stacklevel=2,
            )

        # -- Seed torch + build the shared CNN encoder (deep only) --------
        self._encoder = None
        if self._is_deep:
            if self.random_state is not None:
                import torch

                torch.manual_seed(self.random_state)
            if self.architecture == "cnn":
                from ..estimators.deep.vision import build_encoder

                self._encoder = build_encoder(
                    "cnn", backbone=self.backbone, in_channels=int(X.shape[1])
                )

        # -- Cross-validation ---------------------------------------------
        per_fold: dict[str, list[float | None]] = {name: [] for name in self.metrics}
        fold_reasons: dict[str, list[str]] = {name: [] for name in self.metrics}
        for train_idx, test_idx in splitter.split(X, y_pu):
            clf = self._fresh_estimator(classifier_cls, classifier_instance, prior)
            outcomes = self._fit_and_evaluate_one(
                clf,
                X[train_idx],
                y_pu[train_idx],
                X[test_idx],
                y_pu[test_idx],
                y_true[test_idx] if y_true is not None else None,
                prior,
            )
            for name in self.metrics:
                value, reason = outcomes[name]
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
                # Reuse the profile computed above: profiling twice with
                # identical inputs (including the CV selection diagnostic)
                # wasted one full pass per fit_evaluate.
                profile=profile,
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
                profile=profile,
            )

        # -- Assemble report ----------------------------------------------
        return self._build_report(
            profile=profile,
            prior_info=prior_info,
            recommendation=recommendation,
            cv_metrics=cv_metrics,
            classifier_name=classifier_name,
            auto_mode=auto_mode,
            classifier_cls=classifier_cls,
            skipped_candidates=skipped_candidates,
            y_true=y_true,
            splitter=splitter,
            n_splits=n_splits,
            final_model=final_model,
            diagnostic=diagnostic,
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
        allow_degradation: bool = False,
    ) -> tuple[float | None, PriorInfo]:
        """Resolve the class prior: explicit > constructor > estimation.

        ``y_true`` is deliberately not a parameter: class priors are never
        estimated from ground-truth labels.

        With ``allow_degradation`` (auto mode), a failed or invalid
        estimation degrades to a no-prior run instead of raising: the
        recommender then simply excludes prior-requiring methods.
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
            if allow_degradation:
                # Auto mode with estimation disabled: fall back to a
                # no-prior run; the recommender excludes prior-requiring
                # methods (recommend_from_profile accepts class_prior=None).
                return None, PriorInfo(
                    value=None,
                    source="none",
                    method_requires_prior=needs_prior,
                    degraded=(
                        "class-prior estimation disabled (prior_estimator=None); "
                        "auto mode will only consider methods that need no prior"
                    ),
                )
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
            _validate_prior_value(value, f"prior estimate from {estimator_name}")
        except Exception as exc:
            if allow_degradation:
                # Auto mode: fall back to a no-prior run; the recommender
                # will exclude prior-requiring methods and the report
                # records the degradation.
                return None, PriorInfo(
                    value=None,
                    source="none",
                    method_requires_prior=needs_prior,
                    estimator=estimator_name,
                    degraded=f"estimation with {estimator_name!r} failed: {exc}",
                )
            raise PipelineError(
                f"Class-prior estimation with {estimator_name!r} failed: {exc}"
            ) from exc
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
            clf = clone(instance)
            # The pipeline builds the CNN encoder even for instance paths
            # (architecture='cnn' + a declared encoder parameter); inject
            # it so the built encoder is consumed instead of discarded.
            if (
                self._encoder is not None
                and "encoder" in inspect.signature(type(instance).__init__).parameters
            ):
                clf.set_params(encoder=self._encoder)
            return clf
        assert cls is not None
        kwargs: dict[str, Any] = {}
        signature = inspect.signature(cls.__init__)
        if prior is not None and "class_prior" in signature.parameters:
            kwargs["class_prior"] = prior
        if "random_state" in signature.parameters and self.random_state is not None:
            kwargs["random_state"] = self.random_state
        if "encoder" in signature.parameters and self._encoder is not None:
            kwargs["encoder"] = self._encoder
        if "device" in signature.parameters and self.device is not None:
            kwargs["device"] = self.device
        if "max_epochs" in signature.parameters and self.max_epochs is not None:
            kwargs["max_epochs"] = self.max_epochs
        return cls(**kwargs)

    def _resolved_splitter(self, X: Any, y_pu: np.ndarray) -> Any:
        if isinstance(self.cv, int):
            return PUStratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        if self.cv is None:
            return PUStratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        return self.cv

    def _prepare_inputs(
        self,
        X: Any,
        y_pu: np.ndarray,
        y_true: np.ndarray | None,
    ) -> tuple[Any, np.ndarray, np.ndarray | None, Any, Any, int]:
        """Validate inputs and resolve the CV splitter for the workflow."""
        X, y_pu = validate_pu_X_y(
            X,
            y_pu,
            allow_nd=True,
            estimator_name="PUPipeline",
        )
        if X.ndim not in (2, 4):
            raise ValidationError(f"X must be 2-D (table) or 4-D (NCHW images); got ndim={X.ndim}.")
        if X.ndim == 4 and not (self._is_deep and self.architecture == "cnn"):
            raise PipelineError(
                "4-D image inputs require an explicit deep classifier "
                "(wconpu or infomax_pu) with architecture='cnn'."
            )
        if X.ndim == 2 and self.architecture == "cnn":
            raise PipelineError(
                "architecture='cnn' requires 4-D NCHW image inputs; "
                "got 2-D data. Use architecture='mlp' for tables."
            )
        # 4-D 图像：prior 估计与数据画像在展平视图上进行（标签层面的量）
        analysis_X = X.reshape(X.shape[0], -1) if X.ndim == 4 else X
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
        return X, y_pu, y_true, analysis_X, splitter, n_splits

    def _fit_and_evaluate_one(
        self,
        estimator: BasePUClassifier,
        X_train: Any,
        y_pu_train: np.ndarray,
        X_test: Any,
        y_pu_test: np.ndarray,
        y_true_test: np.ndarray | None,
        prior: float | None,
    ) -> dict[str, tuple[float | None, str | None]]:
        """Fit one fold's estimator and compute every metric on its test fold."""
        estimator.fit(X_train, y_pu_train, class_prior=prior)
        pred = estimator.predict(X_test)
        scores = _extract_scores(estimator, X_test)
        return {
            name: _compute_metric(name, y_pu_test, pred, scores, y_true_test, prior)
            for name in self.metrics
        }

    def _build_report(
        self,
        *,
        profile: PUDataProfile,
        prior_info: PriorInfo,
        recommendation: RecommendationResult | None,
        cv_metrics: dict[str, CVMetric],
        classifier_name: str,
        auto_mode: bool,
        classifier_cls: type[BasePUClassifier] | None,
        skipped_candidates: list[dict[str, str]],
        y_true: np.ndarray | None,
        splitter: Any,
        n_splits: int,
        final_model: BasePUClassifier,
        diagnostic: PUDiagnosticReport,
    ) -> PipelineReport:
        """Assemble the final :class:`PipelineReport` (issues + provenance)."""
        # NOTE: the profile itself flags a class prior below the labeled
        # positive fraction (inconsistent_class_prior) when supplied.
        issues: list[ProfileIssue] = list(profile.issues)
        if prior_info.degraded:
            issues.append(
                ProfileIssue(
                    "prior_estimation_failed",
                    "warning",
                    prior_info.degraded,
                    "Auto mode degraded to a no-prior run: methods that "
                    "require a class prior were excluded from the candidates.",
                )
            )
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
        prior_audit_flagged = prior_info.source == "estimated" and any(
            issue.code == "inconsistent_class_prior" for issue in profile.issues
        )
        provenance = {
            "classifier": classifier_name,
            "classifier_mode": (
                "auto" if auto_mode else "name" if classifier_cls is not None else "instance"
            ),
            "prior_source": prior_info.source,
            "prior_audit_flagged": prior_audit_flagged,
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


def _declares_encoder_parameter(cls: type) -> bool:
    """True if *cls* constructor declares an ``encoder`` parameter.

    The pipeline injects the CNN encoder only into classifiers whose
    signature accepts it (see ``_fresh_estimator``); ``architecture="cnn"``
    must fail fast for deep classifiers that would silently ignore the
    backbone choice (e.g. Self-PU declares ``backbone`` instead).
    """
    return "encoder" in inspect.signature(cls.__init__).parameters


def _missing_required_params(cls: type) -> set[str]:
    """Required constructor params the pipeline cannot supply."""
    signature = inspect.signature(cls.__init__)
    return {
        param
        for param, p in signature.parameters.items()
        if param != "self"
        # *args / **kwargs are never required: their default is also
        # ``empty``, which used to misread them as mandatory and block
        # auto-instantiation of any **kwargs constructor.
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
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
    # Defensive: _METRIC_SPECS and the if-chain above stay in lockstep;
    # an unhandled name is an internal invariant violation, not a
    # pipeline orchestration failure.
    raise AssertionError(f"Unreachable metric {name!r}.")


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
    validate_true_binary_labels(y_true, estimator_name="y_true")
    return y_true.astype(int, copy=False)


def _validate_prior_value(value: float, name: str) -> None:
    """Validate a user-supplied class prior (caller error -> ValueError).

    ``PipelineError`` is reserved for orchestration failures (unresolvable
    classifier/prior, failed estimation); a bad user input is a plain
    ``ValueError`` like every other constructor-argument check in the
    pipeline (cv, metrics, architecture).
    """
    check_scalar_in_range(value, 0.0, 1.0, name, inclusive=False)


def _resolved_n_splits(splitter: Any, X: Any, y_pu: np.ndarray) -> int:
    if hasattr(splitter, "get_n_splits"):
        return int(splitter.get_n_splits(X, y_pu))
    # Fail fast instead of silently assuming 5 folds: the fold-count
    # guard, provenance, and the deep training-cost warning all depend
    # on the real number of splits.
    raise ValueError(
        f"cv splitter {type(splitter).__name__} must implement "
        "get_n_splits(X, y) so the pipeline can validate fold counts."
    )


def _cv_provenance(splitter: Any, n_splits: int) -> dict[str, Any]:
    info: dict[str, Any] = {"n_splits": n_splits, "splitter": type(splitter).__name__}
    try:
        params = splitter.get_params()
        info["shuffle"] = params.get("shuffle")
        info["random_state"] = params.get("random_state")
    except Exception:  # noqa: BLE001 - provenance is best-effort
        pass
    return info
