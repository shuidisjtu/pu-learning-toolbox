# ruff: noqa: N803, N806

"""End-to-end PU workflow orchestration: ``PUPipeline``.

Combines data profiling, class-prior estimation, model training,
PU-stratified cross-validation, and diagnostic reporting into a single
callable workflow so that non-expert users only need ``(X, y_pu)`` and
a method name (or ``"auto"`` for recommender-driven selection).
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.base import BasePriorEstimator, BasePUClassifier
from ..core.device import resolve_device_name
from ..core.exceptions import RegistryError
from ..core.tags import Backend, SampleWeightSupport, TrainingCost
from ..core.validation import check_scalar_in_range
from ..diagnostics.report import PUDiagnosticReport, build_diagnostic_report
from ..model_selection.split import PUStratifiedKFold
from ..preprocessing.data_profiler import profile_pu_data
from ..progress import CancellationToken, ProgressCallback, emit_progress
from ..registry import RecommendationResult, recommend_from_profile
from ..registry.registry import get_algorithm, get_metadata
from ._errors import PipelineError
from ._evaluation import (
    _METRIC_SPECS,
    DEFAULT_METRICS,
    compute_metric,
    extract_proba,
    extract_scores,
    resolve_metric_names,
    run_cross_validation,
)
from ._inputs import prepare_pipeline_inputs
from ._models import (
    declares_encoder_parameter,
    fresh_estimator,
    missing_required_params,
    pick_first_instantiable,
    resolve_classifier_name,
    validate_classifier_params,
    validate_prior_param_types,
)
from ._reporting import build_pipeline_report
from .report import PipelineReport, PriorInfo

__all__ = ["DEFAULT_METRICS", "PipelineError", "PUPipeline"]

# Parameters owned by workflow orchestration.  Letting classifier_params
# override these would make CV folds disagree with report provenance (or, for
# encoders, reuse a fitted torch module across folds).
_MANAGED_CLASSIFIER_PARAMS = {"class_prior", "random_state", "encoder"}

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


class PUPipeline:
    """One-call workflow: profile -> prior -> train -> CV -> evaluate.

    Parameters
    ----------
    classifier : str, BasePUClassifier, or "auto" (default "auto")
        Registry name of a PU classifier, a ready-to-use instance, or
        ``"auto"`` to select the best match via the recommender.  Registry
        names are resolved case-insensitively through aliases. Classes with
        extra required constructor parameters can receive them through
        ``classifier_params``; auto mode still skips such classes because it
        has no method-specific parameter target.
    classifier_params : mapping or None (default None)
        Constructor parameters for a classifier selected by registry name.
        This also makes methods with additional required parameters usable by
        name (for example ``classifier="ldce"`` with
        ``classifier_params={"flip_probability": 0.2}``).  It cannot be
        combined with ``classifier="auto"`` or a classifier instance.
        ``class_prior``, ``random_state``, and ``encoder`` remain managed by
        the pipeline and must use their dedicated workflow arguments.
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
        classifier_params: dict[str, Any] | None = None,
        prior_estimator: str | BasePriorEstimator | None | object = _UNSET,
        prior_params: dict[str, Any] | None = None,
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
        self.classifier_params = dict(classifier_params or {})
        if any(not isinstance(key, str) or not key for key in self.classifier_params):
            raise TypeError("classifier_params keys must be non-empty strings.")
        managed = sorted(_MANAGED_CLASSIFIER_PARAMS.intersection(self.classifier_params))
        if managed:
            raise ValueError(
                "classifier_params cannot override pipeline-managed parameters "
                f"{managed}; use the dedicated PUPipeline arguments instead."
            )
        if self.classifier_params and classifier == "auto":
            raise ValueError(
                "classifier_params requires an explicit classifier name; "
                "parameters cannot be applied before auto selects a method."
            )
        if self.classifier_params and isinstance(classifier, BasePUClassifier):
            raise TypeError(
                "classifier_params cannot be combined with a classifier instance; "
                "configure the instance directly instead."
            )
        self.random_state = random_state
        self.metrics = resolve_metric_names(metrics)

        # -- classifier: fail fast on unresolvable names -----------------
        if isinstance(classifier, str) and classifier != "auto":
            self._classifier_cls = resolve_classifier_name(
                classifier, provided_params=set(self.classifier_params)
            )
            validate_classifier_params(self._classifier_cls, self.classifier_params, classifier)
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
            if not declares_encoder_parameter(encoder_cls):
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
        self.prior_params = dict(prior_params or {})
        if self.prior_params and isinstance(self._prior_estimator, BasePriorEstimator):
            raise TypeError(
                "prior_params cannot be combined with a prior-estimator instance; "
                "pass an already-configured instance instead"
            )
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
        sample_weight: np.ndarray | None = None,
        refit: bool = True,
        progress_callback: ProgressCallback | None = None,
        cancellation_token: CancellationToken | None = None,
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
        sample_weight : array-like of shape (n_samples,), optional
            Per-source-sample training weights. They are sliced with every
            CV training fold and passed to the final refit. The selected
            classifier must declare ``sample_weight_support='supported'``;
            ignored or unimplemented weight semantics fail explicitly.
        refit : bool, default True
            Fit a final model on all samples and build model diagnostics.
            Set to ``False`` for parameter-search trials that only need CV
            metrics; the returned report then has ``final_model=None`` and
            ``diagnostic=None``.
        progress_callback : callable, optional
            Receives immutable progress snapshots at workflow phase and CV-fold
            boundaries. The callback runs synchronously in the caller's thread.
        cancellation_token : CancellationToken, optional
            Cooperative cancellation checked between workflow phases, CV folds,
            and before/after final refit. An estimator already inside ``fit``
            finishes its current call before cancellation is observed.
        """
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        emit_progress(
            progress_callback,
            stage="prepare",
            completed=0,
            total=1,
            message="校验输入并准备交叉验证",
        )
        X, y_pu, y_true, analysis_X, splitter, n_splits = self._prepare_inputs(X, y_pu, y_true)
        sample_weight = _validate_sample_weight(sample_weight, X.shape[0])
        total_steps = n_splits + (4 if refit else 3)

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
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        prior, prior_info = self._resolve_prior(
            X=analysis_X,
            y_pu=y_pu,
            class_prior=class_prior,
            classifier_instance=classifier_instance,
            needs_prior=needs_prior,
            allow_degradation=auto_mode,
        )
        emit_progress(
            progress_callback,
            stage="prior",
            completed=1,
            total=total_steps,
            message="类先验处理完成",
        )

        # -- Data profile (once, reused by recommender and report) -------
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        profile = profile_pu_data(
            analysis_X,
            y_pu,
            y_true=y_true,
            class_prior=prior,
            random_state=self.random_state,
        )
        emit_progress(
            progress_callback,
            stage="profile",
            completed=2,
            total=total_steps,
            message="数据画像与模型准备完成",
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
            classifier_cls, skipped_candidates = pick_first_instantiable(recommendation.candidates)
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

        if sample_weight is not None:
            support_owner = classifier_cls or type(classifier_instance)
            support = getattr(
                support_owner,
                "sample_weight_support",
                SampleWeightSupport.NOT_IMPLEMENTED,
            )
            if support != SampleWeightSupport.SUPPORTED:
                value = support.value if isinstance(support, SampleWeightSupport) else str(support)
                raise PipelineError(
                    f"Classifier {classifier_name!r} cannot be used with sample_weight: "
                    f"declared support is {value!r}. Choose a classifier whose metadata "
                    "reports sample_weight_support='supported'."
                )

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
        def evaluate_fold(train_idx: np.ndarray, test_idx: np.ndarray):
            clf = self._fresh_estimator(classifier_cls, classifier_instance, prior)
            return self._fit_and_evaluate_one(
                clf,
                X[train_idx],
                y_pu[train_idx],
                X[test_idx],
                y_pu[test_idx],
                y_true[test_idx] if y_true is not None else None,
                prior,
                sample_weight[train_idx] if sample_weight is not None else None,
            )

        cv_metrics = run_cross_validation(
            X=X,
            y_pu=y_pu,
            splitter=splitter,
            n_splits=n_splits,
            metrics=self.metrics,
            evaluate_fold=evaluate_fold,
            total_steps=total_steps,
            progress_callback=progress_callback,
            cancellation_token=cancellation_token,
        )

        # -- Optional final model (full refit) + final diagnostic ----------
        final_model: BasePUClassifier | None = None
        diagnostic: PUDiagnosticReport | None = None
        if refit:
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()
            emit_progress(
                progress_callback,
                stage="refit",
                completed=total_steps - 1,
                total=total_steps,
                message="正在全量重训最佳模型",
            )
            final_model = self._fresh_estimator(classifier_cls, classifier_instance, prior)
            final_model.fit(X, y_pu, class_prior=prior, sample_weight=sample_weight)
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()
            if extract_scores(final_model, X) is not None:
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
        report = build_pipeline_report(
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
            random_state=self.random_state,
            classifier_params=self.classifier_params,
            sample_weight=sample_weight,
        )
        emit_progress(
            progress_callback,
            stage="complete",
            completed=total_steps,
            total=total_steps,
            message="分析完成",
        )
        return report

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
            # A prior_params variant override (e.g. --prior-estimator km2
            # --prior-param variant=km1) changes what actually ran; the
            # report must not claim the original name.
            variant = getattr(estimator, "variant", None)
            if variant and variant != estimator_name:
                estimator_name = f"{estimator_name} (variant={variant})"
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
        kwargs: dict[str, Any] = dict(self.prior_params)
        if cls is not None and issubclass(cls, BasePriorEstimator):
            # KernelMeanPriorEstimator requires variant selection; a user
            # prior_params entry for "variant" takes precedence.
            from ..prior.kernel_mean import KernelMeanPriorEstimator

            if cls is KernelMeanPriorEstimator:
                kwargs.setdefault("variant", name)
            validate_prior_param_types(cls, kwargs, name)
        try:
            return cls(**kwargs)
        except TypeError as exc:
            raise PipelineError(f"invalid prior parameters for '{name}': {exc}") from exc

    def _fresh_estimator(
        self,
        cls: type[BasePUClassifier] | None,
        instance: BasePUClassifier | None,
        prior: float | None,
    ) -> BasePUClassifier:
        """Clone an instance, or instantiate a class with injected params."""
        return fresh_estimator(
            cls=cls,
            instance=instance,
            prior=prior,
            classifier_params=self.classifier_params,
            random_state=self.random_state,
            encoder=self._encoder,
            device=self.device,
            max_epochs=self.max_epochs,
            classifier_name=self._classifier_name,
        )

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
        return prepare_pipeline_inputs(
            X,
            y_pu,
            y_true,
            is_deep=self._is_deep,
            architecture=self.architecture,
            resolve_splitter=self._resolved_splitter,
        )

    def _fit_and_evaluate_one(
        self,
        estimator: BasePUClassifier,
        X_train: Any,
        y_pu_train: np.ndarray,
        X_test: Any,
        y_pu_test: np.ndarray,
        y_true_test: np.ndarray | None,
        prior: float | None,
        sample_weight_train: np.ndarray | None,
    ) -> dict[str, tuple[float | None, str | None]]:
        """Fit one fold's estimator and compute every metric on its test fold."""
        estimator.fit(
            X_train,
            y_pu_train,
            class_prior=prior,
            sample_weight=sample_weight_train,
        )
        pred = estimator.predict(X_test)
        scores = extract_scores(estimator, X_test)
        proba = (
            extract_proba(estimator, X_test)
            if any(_METRIC_SPECS[name][3] for name in self.metrics)
            else None
        )
        return {
            name: compute_metric(name, y_pu_test, pred, scores, y_true_test, prior, proba)
            for name in self.metrics
        }


# ── Module-level helpers ────────────────────────────────────────────────


def _missing_required_params(cls: type, *, provided_params: set[str] | None = None) -> set[str]:
    """Compatibility wrapper for the CLI and existing private consumers."""
    return missing_required_params(cls, provided_params=provided_params)


def _validate_prior_value(value: float, name: str) -> None:
    """Validate a user-supplied class prior (caller error -> ValueError).

    ``PipelineError`` is reserved for orchestration failures (unresolvable
    classifier/prior, failed estimation); a bad user input is a plain
    ``ValueError`` like every other constructor-argument check in the
    pipeline (cv, metrics, architecture).
    """
    check_scalar_in_range(value, 0.0, 1.0, name, inclusive=False)


def _validate_sample_weight(
    sample_weight: np.ndarray | None,
    n_samples: int,
) -> np.ndarray | None:
    """Validate and copy workflow-level sample weights."""
    if sample_weight is None:
        return None
    try:
        weights = np.asarray(sample_weight, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_weight must contain numeric values.") from exc
    if weights.shape != (n_samples,):
        raise ValueError(f"sample_weight must have shape ({n_samples},); got {weights.shape}.")
    if not np.isfinite(weights).all():
        raise ValueError("sample_weight must contain only finite values.")
    if np.any(weights < 0.0):
        raise ValueError("sample_weight must be non-negative.")
    if not np.any(weights > 0.0):
        raise ValueError("sample_weight must contain at least one positive value.")
    return weights.copy()
