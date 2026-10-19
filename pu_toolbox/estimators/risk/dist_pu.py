# ruff: noqa: N803, N806, E501

"""Dist-PU label-distribution alignment classifier.

The default is the historical clean-room expectation-alignment objective. An
explicit ``distribution_loss="author_histogram"`` option adds the locked
author-code one-bin polar histogram loss, without claiming to reproduce its
two-stage training procedure. PyTorch remains an optional dependency.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from ...core.base import BasePUClassifier
from ...core.device import resolve_device
from ...core.fit_state import invalidate_failed_fit
from ...core.tags import (
    AlgorithmFamily,
    Assumption,
    Backend,
    ImplementationStatus,
    Maturity,
    SampleWeightSupport,
    Scenario,
    SourceStatus,
)
from ...core.training_views import RUN_VIEWS
from ...core.validation import check_scalar_in_range, validate_pu_X_y


def _distribution_regularizers(
    probs: Any,
    unlabeled_mask: Any,
    class_prior: float,
    *,
    include_positive_in_unlabeled: bool,
) -> tuple[Any, Any]:
    """Dist-PU's two label-distribution regularizers for one role set.

    Which rows carry the unlabeled role is the caller's decision, and it is a
    *boolean* rather than a view string on purpose: the OS path then holds no
    view value it could pass by mistake (survey protocol §2.3).  Under the
    calibrated (ts) view the role set is the whole training matrix -- the
    labeled positives join ``D_U``, exactly as case-control sampling puts them
    back -- so a physical row may serve both the positive term and this one.

    The unlabeled role set is the only thing the view changes: the alignment
    target stays the population prior (it is *not* re-derived from the observed
    P/U ratio), and both formulas are reproduced character for character from
    the pre-existing objective -- including the ``1e-6`` inside the logarithms,
    which is not the numerical guard (the caller clamps the logits).

    Tensor methods only -- ``torch`` is an optional dependency and must not be
    imported at module load.
    """
    role_probs = probs if include_positive_in_unlabeled else probs[unlabeled_mask]
    alignment = (role_probs.mean() - class_prior).pow(2)
    entropy = -(
        role_probs * (role_probs + 1e-6).log() + (1 - role_probs) * (1 - role_probs + 1e-6).log()
    ).mean()
    return alignment, entropy


def _polar_histogram(probs: Any) -> Any:
    """Reproduce the official Dist-PU one-bin polar score histogram.

    The locked source constructs two bins centered at 0 and 1 (`num_bins=1`),
    uses triangular membership with width 1, then averages membership over
    rows. For probabilities in [0, 1], each row contributes `(1-q, q)`.
    """
    import torch

    if probs.ndim != 1 or probs.numel() == 0:
        raise ValueError("Dist-PU histogram requires nonempty one-dimensional probabilities")
    if not torch.isfinite(probs).all() or torch.any((probs < 0) | (probs > 1)):
        raise ValueError("Dist-PU histogram probabilities must be finite and in [0, 1]")
    centers = probs.new_tensor([0.0, 1.0])
    distances = torch.abs(probs[:, None] - centers[None, :])
    memberships = torch.clamp(1.0 - distances, min=0.0)
    return memberships.mean(dim=0)


def _histogram_distribution_loss(
    positive_probs: Any,
    unlabeled_probs: Any,
    class_prior: float,
    *,
    include_positive_in_unlabeled: bool,
) -> tuple[Any, Any]:
    """Author-code histogram loss terms for the positive and marginal roles.

    Mirrors `LabelDistributionLoss(num_bins=1, proxy='polar', dist='L1')`:
    P targets `[0, 1]`, U targets `[1-pi, pi]`, and U loss weight `1/(2*pi)`.
    """
    import torch

    if positive_probs.ndim != 1 or positive_probs.numel() == 0:
        raise ValueError("Dist-PU histogram loss requires observed positive rows")
    marginal_probs = (
        torch.cat((unlabeled_probs, positive_probs))
        if include_positive_in_unlabeled
        else unlabeled_probs
    )
    if marginal_probs.ndim != 1 or marginal_probs.numel() == 0:
        raise ValueError("Dist-PU histogram loss requires marginal-role rows")
    p_histogram = _polar_histogram(positive_probs)
    u_histogram = _polar_histogram(marginal_probs)
    p_proxy = positive_probs.new_tensor([0.0, 1.0])
    u_proxy = positive_probs.new_tensor([1.0 - class_prior, class_prior])
    p_loss = (p_histogram - p_proxy).abs().mean()
    u_loss = (u_histogram - u_proxy).abs().mean() / (2.0 * class_prior)
    return p_loss, u_loss


def _author_histogram_batch_loss(
    logits: Any,
    labels: Any,
    class_prior: float,
    *,
    include_positive_in_unlabeled: bool = False,
    clamp_logits: bool = True,
) -> Any:
    """Apply the locked author's P/U histogram objective to one minibatch.

    The upstream loss returns a zero contribution for an absent P or U role;
    this matters for shuffled mini-batches even though the full training set
    must contain both roles.
    """
    import torch
    from torch.nn import functional

    logits = logits.reshape(-1)
    scores = (
        torch.sigmoid(torch.clamp(logits, min=-10.0, max=10.0))
        if clamp_logits
        else torch.sigmoid(logits)
    )
    labels = labels.reshape(-1)
    positive_scores = scores[labels == 1]
    unlabeled_scores = scores if include_positive_in_unlabeled else scores[labels == 0]
    loss = scores.new_zeros(())
    if positive_scores.numel():
        histogram = _polar_histogram(positive_scores)
        proxy = scores.new_tensor([0.0, 1.0])
        loss = loss + functional.l1_loss(histogram, proxy, reduction="mean")
    if unlabeled_scores.numel():
        histogram = _polar_histogram(unlabeled_scores)
        proxy = scores.new_tensor([1.0 - class_prior, class_prior])
        loss = loss + functional.l1_loss(histogram, proxy, reduction="mean") / (2.0 * class_prior)
    return loss


def _entropy_minimization(scores: Any) -> Any:
    """Match the author entropy objective on finite, nonempty probabilities."""
    if scores.numel() == 0:
        return scores.new_zeros(())
    return -(scores * scores.log() + (1.0 - scores) * (1.0 - scores).log()).mean()


class DistPUClassifier(BasePUClassifier):
    """MLP alignment or opt-in staged alignment with a trainable encoder.

    Injected encoders require ``author_staged`` + ``author_histogram`` so
    image training streams CPU-resident inputs in bounded mini-batches.
    The historical clean-room MLP objective remains the default.
    """

    family = AlgorithmFamily.RISK_ESTIMATION
    label_semantics = "pu"
    assumption = (Assumption.SCAR,)
    scenario = (Scenario.CASE_CONTROL,)
    requires_class_prior = True
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.OFFICIAL_EXACT
    backend = Backend.TORCH
    maturity = Maturity.RESEARCH
    sample_weight_support = SampleWeightSupport.IGNORED
    native_architectures = frozenset({"mlp", "cnn"})
    input_ndims = frozenset({2, 4})
    encoder_parameter = "encoder"
    trains_encoder = True

    def __init__(
        self,
        class_prior: float,
        *,
        hidden_dim: int = 64,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        alignment_weight: float = 1.0,
        entropy_weight: float = 0.05,
        mixup_weight: float = 0.1,
        distribution_loss: str = "expectation",
        training_recipe: str = "clean_room",
        warmup_epochs: int = 60,
        pu_epochs: int = 60,
        warmup_learning_rate: float = 5e-4,
        warmup_weight_decay: float = 5e-3,
        pu_learning_rate: float = 5e-5,
        pu_weight_decay: float = 1e-3,
        warmup_entropy_weight: float = 2e-3,
        pu_entropy_weight: float = 4e-3,
        mixup_alpha: float = 6.0,
        mixup_entropy_weight: float = 4e-2,
        mixup_bce_weight: float = 5.0,
        random_state: int | None = 0,
        device: str | None = None,
        encoder: Any | None = None,
    ) -> None:
        super().__init__()
        self.class_prior = class_prior
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.alignment_weight = alignment_weight
        self.entropy_weight = entropy_weight
        self.mixup_weight = mixup_weight
        self.distribution_loss = distribution_loss
        self.training_recipe = training_recipe
        self.warmup_epochs = warmup_epochs
        self.pu_epochs = pu_epochs
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_weight_decay = warmup_weight_decay
        self.pu_learning_rate = pu_learning_rate
        self.pu_weight_decay = pu_weight_decay
        self.warmup_entropy_weight = warmup_entropy_weight
        self.pu_entropy_weight = pu_entropy_weight
        self.mixup_alpha = mixup_alpha
        self.mixup_entropy_weight = mixup_entropy_weight
        self.mixup_bce_weight = mixup_bce_weight
        self.random_state = random_state
        self.device = device
        self.encoder = encoder

    def __setstate__(self, state):
        """Keep MLP estimator pickles made before encoder injection readable."""
        import inspect

        super().__setstate__(state)
        for name, parameter in inspect.signature(DistPUClassifier.__init__).parameters.items():
            if parameter.default is not inspect.Parameter.empty:
                self.__dict__.setdefault(name, parameter.default)
        self.__dict__.setdefault("encoder_", None)

    @invalidate_failed_fit
    def fit(
        self,
        X,
        y_pu,
        *,
        class_prior=None,
        sample_weight=None,
        os_or_ts="os",
        epoch_callback=None,
    ):
        """Fit the Dist-PU label-distribution classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, C, H, W)
            Dense features, or images with an explicit constructor encoder.
            Encoders require author_staged + author_histogram, are cloned
            and trained end to end, and must return finite 2-D features.
        y_pu : array-like of shape (n_samples,)
            PU labels.  +1 = labeled positive, 0 = unlabeled.
        class_prior : float, optional
            Override the constructor's ``class_prior``.  Must be in (0, 1).
        distribution_loss (constructor setting) : {"expectation", "author_histogram"}
            Select the default clean-room mean-squared alignment or the
            author's one-bin polar histogram L1 terms. The histogram option
            does not reproduce the author's complete two-stage recipe.
        sample_weight : ignored (present for sklearn compatibility; PU risk estimators do not use instance weights)
        os_or_ts : {"os", "ts"}, default "os"
            Training data view (survey protocol §2.3).  ``"ts"`` applies the
            TS-OS calibration to this estimator's two distribution regularizers:
            the empirical unlabeled role becomes ``D_U ∪ D_P``, so label
            alignment and entropy run over every training row and the alignment
            mean's denominator becomes ``n_P + n_U``.  The alignment *target*
            stays the population prior -- it is not re-derived from the observed
            P/U ratio -- and the positive term is untouched.

            Calibration moves a role assignment, never the physical training
            matrix: the network, optimizer, epoch count and Mixup pool are
            identical under both views, no row is duplicated, and the RNG call
            sequence is unaffected.  Both regularizers come from the same
            forward pass, so the view changes only which rows they average over.
        epoch_callback : callable, optional
            Called as ``epoch_callback(epoch, self)`` after each completed epoch.
            Labels are zero-based for clean_room and one-based cumulative
            for author_staged, whose checkpoint context also records stage.

        Returns
        -------
        self : DistPUClassifier
        """
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("DistPUClassifier requires the optional 'torch' dependency") from exc
        if self.encoder is not None and not isinstance(self.encoder, nn.Module):
            raise TypeError("encoder must be a torch.nn.Module")
        X, y_pu = validate_pu_X_y(
            X,
            y_pu,
            accept_sparse=False,
            allow_nd=self.encoder is not None,
            estimator_name="DistPUClassifier",
        )
        if X.ndim not in ({2, 4} if self.encoder is not None else {2}):
            raise ValueError("Dist-PU accepts 2-D features or 4-D images with an encoder")
        # Before any RNG use: an invalid view must not perturb the torch seed.
        if os_or_ts not in RUN_VIEWS:
            raise ValueError(f"os_or_ts must be 'os' or 'ts'; got {os_or_ts!r}.")
        pi = self.class_prior if class_prior is None else class_prior
        check_scalar_in_range(pi, 0.0, 1.0, "class_prior", inclusive=False)
        if self.epochs < 1 or self.hidden_dim < 1:
            raise ValueError("epochs must be >= 1 and hidden_dim must be >= 1")
        if self.distribution_loss not in {"expectation", "author_histogram"}:
            raise ValueError("distribution_loss must be 'expectation' or 'author_histogram'")
        if self.training_recipe not in {"clean_room", "author_staged"}:
            raise ValueError("training_recipe must be 'clean_room' or 'author_staged'")
        if self.encoder is not None and self.training_recipe != "author_staged":
            raise ValueError(
                "encoder requires training_recipe='author_staged' and "
                "distribution_loss='author_histogram' for mini-batch training"
            )
        if self.training_recipe == "author_staged":
            if self.distribution_loss != "author_histogram":
                raise ValueError(
                    "training_recipe='author_staged' requires distribution_loss='author_histogram'"
                )
            if type(self.batch_size) is not int or self.batch_size < 1:
                raise ValueError("batch_size must be a positive integer")
            for name in ("warmup_epochs", "pu_epochs"):
                value = getattr(self, name)
                if type(value) is not int or value < 1:
                    raise ValueError(f"{name} must be a positive integer")
            for name in (
                "warmup_learning_rate",
                "warmup_weight_decay",
                "pu_learning_rate",
                "pu_weight_decay",
                "warmup_entropy_weight",
                "pu_entropy_weight",
                "mixup_alpha",
                "mixup_entropy_weight",
                "mixup_bce_weight",
            ):
                value = getattr(self, name)
                if (
                    isinstance(value, bool | np.bool_)
                    or not isinstance(value, int | float | np.integer | np.floating)
                    or not np.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"{name} must be finite and nonnegative")
            if self.warmup_learning_rate <= 0 or self.pu_learning_rate <= 0:
                raise ValueError("staged training learning rates must be positive")
            if self.mixup_alpha <= 0:
                raise ValueError("mixup_alpha must be positive")
        X = np.asarray(X, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError("X must remain finite after float32 conversion")
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        device = resolve_device(self.device)
        self.device_ = device
        self.input_shape_ = tuple(X.shape[1:])
        self.n_features_in_ = X.shape[1]
        if self.encoder is None:
            self.encoder_ = None
            self.model_ = nn.Sequential(
                nn.Linear(X.shape[1], self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, 1)
            ).to(device)
        else:
            from ..deep._validation import validate_encoder_features

            self.encoder_ = copy.deepcopy(self.encoder).to(device, dtype=torch.float32)
            self.encoder_.requires_grad_(True).eval()
            with torch.no_grad():
                width = validate_encoder_features(
                    self.encoder_(torch.as_tensor(X[:1].copy(), device=device)),
                    encoder_param_name="encoder",
                )
            head = nn.Sequential(
                nn.Linear(width, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, 1)
            )
            self.model_ = nn.Sequential(self.encoder_, head).to(device, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        data_device = "cpu" if self.training_recipe == "author_staged" else device
        tx = torch.as_tensor(X, device=data_device)
        ty = torch.as_tensor(y_pu, device=data_device)
        p_mask, u_mask = ty == 1, ty == 0
        if self.training_recipe == "author_staged":
            return self._fit_author_staged(
                tx,
                ty,
                p_mask,
                u_mask,
                pi,
                os_or_ts,
                epoch_callback,
                torch,
            )
        # ── TS-OS calibration (protocol §2.3) ─────────────────────────────
        # The unlabeled role is carried by the unlabeled rows under the OS view
        # and by the whole training matrix under the calibrated (ts) view: the
        # labeled positives join D_U, exactly as case-control sampling puts them
        # back.  Only that role set changes; the positive term, the class prior,
        # the Mixup pool and the RNG stream are untouched.
        calibrated_view = os_or_ts == "ts"
        self.calibration_applied_ = calibrated_view
        self.optimizer_steps_ = 0
        self.epochs_completed_ = 0
        self.training_provenance_ = {
            "method": "dist_pu",
            "implementation_path": "clean_room_single_stage_full_batch",
            "distribution_loss": self.distribution_loss,
            "upstream_source_lock": "cb74be1a87176fd38270873c06374e53905b7354",
            "upstream_training_recipe_reproduced": False,
            "run_view": os_or_ts,
            "calibration_applied": calibrated_view,
            "optimizer": "Adam",
            "optimizer_steps_per_epoch": 1,
            "batch_size_parameter_effective": False,
            "configured_batch_size_not_used": repr(self.batch_size),
            "mixup_pool": "entire_training_matrix",
            "class_prior_used": float(pi),
            "formal_admission": False,
        }
        self.loss_history_ = []
        bce = nn.BCEWithLogitsLoss()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            logits = self.model_(tx).squeeze(1).clamp(-10, 10)
            probs = torch.sigmoid(logits)
            if self.distribution_loss == "expectation":
                positive_loss = bce(logits[p_mask], torch.ones_like(logits[p_mask]))
                alignment, entropy = _distribution_regularizers(
                    probs, u_mask, pi, include_positive_in_unlabeled=calibrated_view
                )
                loss = positive_loss + self.alignment_weight * alignment
            else:
                p_hist_loss, u_hist_loss = _histogram_distribution_loss(
                    probs[p_mask],
                    probs[u_mask],
                    pi,
                    include_positive_in_unlabeled=calibrated_view,
                )
                entropy_probs = probs if calibrated_view else probs[u_mask]
                loss = p_hist_loss + self.alignment_weight * u_hist_loss
                alignment = u_hist_loss
                entropy = -(
                    entropy_probs * (entropy_probs + 1e-6).log()
                    + (1 - entropy_probs) * (1 - entropy_probs + 1e-6).log()
                ).mean()
            loss = loss + self.entropy_weight * entropy
            if self.mixup_weight > 0 and len(X) > 1:
                perm = torch.randperm(len(X), device=device)
                lam = torch.rand((), device=device)
                mix_x = lam * tx + (1 - lam) * tx[perm]
                mix_y = lam * probs.detach() + (1 - lam) * probs.detach()[perm]
                loss = loss + self.mixup_weight * bce(self.model_(mix_x).squeeze(1), mix_y)
            loss.backward()
            optimizer.step()
            self.optimizer_steps_ += 1
            self.epochs_completed_ = epoch + 1
            self.loss_history_.append(float(loss.detach().cpu()))
            if epoch_callback is not None:
                epoch_callback(epoch, self)
        self.classes_ = np.array([0, 1])
        self._class_prior, self._X_shape_, self._is_fitted = pi, X.shape, True
        self.training_provenance_["optimizer_steps"] = self.optimizer_steps_
        self.training_provenance_["epochs_completed"] = self.epochs_completed_
        self.device_ = device
        return self

    @property
    def checkpoint_stages(self):
        if self.training_recipe == "author_staged":
            return ("warmup", "dist_pu")
        return None

    @property
    def checkpoint_epoch_count(self):
        if self.training_recipe == "author_staged":
            return self.warmup_epochs + self.pu_epochs
        return self.epochs

    @property
    def checkpoint_prediction_batch_size(self):
        return self.batch_size

    def _fit_author_staged(self, tx, ty, p_mask, u_mask, pi, os_or_ts, epoch_callback, torch):
        """Run audited staged losses on an MLP or explicit encoder adapter.

        This mirrors the author's optimizer/loss/pseudo-label/scheduler
        sequence, while remaining an engineering adapter: it does not port
        their image models, datasets, preprocessing, or test-set selection.
        """
        from torch.nn import functional

        n_samples = len(tx)
        device = self.device_
        if n_samples < 2 or not bool(p_mask.any()) or not bool(u_mask.any()):
            raise ValueError("author_staged requires at least one observed P and one U row")
        self.optimizer_steps_ = 0
        self.epochs_completed_ = 0
        self.loss_history_ = []
        self.stage_loss_history_ = {"warmup": [], "dist_pu": []}
        self.epoch_provenance_ = []
        self.checkpoint_stage_ = "warmup"
        self.checkpoint_stage_epoch_ = 0
        self.checkpoint_round_ = None
        self.calibration_applied_ = os_or_ts == "ts"
        self.training_provenance_ = {
            "method": "dist_pu",
            "implementation_path": "author_staged_encoder_adapter"
            if self.encoder is not None
            else "author_staged_mlp_adapter",
            "training_recipe": "author_staged",
            "distribution_loss": "author_histogram",
            "upstream_source_lock": "cb74be1a87176fd38270873c06374e53905b7354",
            "upstream_training_recipe_reproduced": False,
            "upstream_source_replay": False,
            "run_view": os_or_ts,
            "calibration_applied": self.calibration_applied_,
            "optimizer": "Adam",
            "batch_size": self.batch_size,
            "batch_size_parameter_effective": True,
            "warmup_epochs": self.warmup_epochs,
            "pu_epochs": self.pu_epochs,
            "warmup_learning_rate": float(self.warmup_learning_rate),
            "warmup_weight_decay": float(self.warmup_weight_decay),
            "pu_learning_rate": float(self.pu_learning_rate),
            "pu_weight_decay": float(self.pu_weight_decay),
            "warmup_scheduler": "CosineAnnealingLR(T_max=pu_epochs), step once after each epoch",
            "pu_scheduler": "CosineAnnealingLR(T_max=pu_epochs, eta_min=0.7*lr), step after each minibatch",
            "warmup_entropy_weight": float(self.warmup_entropy_weight),
            "pu_entropy_weight_max": float(self.pu_entropy_weight),
            "mixup_alpha": float(self.mixup_alpha),
            "mixup_entropy_weight": float(self.mixup_entropy_weight),
            "mixup_bce_weight": float(self.mixup_bce_weight),
            "pseudo_label_initialization": "model scores after warmup, indexed by training-row identity",
            "pseudo_label_inference_batch_size": min(128, self.batch_size)
            if self.encoder is not None
            else 128,
            "pseudo_label_update": "pre_optimizer_update_scores written only to current minibatch identities",
            "mixup_lambda": "Beta(alpha, alpha), no max(lambda, 1-lambda) bias",
            "mixup_partner": "uniform random permutation within current minibatch",
            "mixup_pool": "current minibatch; no full-dataset mixup in staged path",
            "class_prior_used": float(pi),
            "clean_labels_read": False,
            "test_labels_read": False,
            "formal_admission": False,
            "architecture_scope": "toolbox MLP adapter; upstream dataset-specific architectures not reproduced",
            "data_device_policy": "CPU-resident features/labels; only active mini-batches moved to the training device",
        }
        if self.encoder is not None:
            self.training_provenance_.update(
                architecture_scope="injected trainable encoder + toolbox MLP head; not upstream dataset-specific architectures",
                encoder_template_cloned=True,
                encoder_trained=True,
                input_shape=list(tx.shape[1:]),
                maximum_rows_per_device_forward=self.batch_size,
                singleton_batchnorm_policy="running_statistics_for_singleton_training_batches",
                encoder_input_isolation="batch_clone_before_model_forward",
            )

        warmup_optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.warmup_learning_rate,
            weight_decay=self.warmup_weight_decay,
        )
        warmup_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            warmup_optimizer, T_max=self.pu_epochs
        )
        np_rng = np.random.RandomState(self.random_state)
        self.model_.train()
        for epoch in range(self.warmup_epochs):
            permutation = torch.randperm(n_samples)
            batch_losses = []
            for start in range(0, n_samples, self.batch_size):
                indices = permutation[start : start + self.batch_size]
                batch_x = tx[indices].to(device)
                labels = ty[indices].to(device)
                logits = self._staged_logits(batch_x)
                scores = torch.sigmoid(torch.clamp(logits, min=-10.0, max=10.0))
                hist_loss = _author_histogram_batch_loss(
                    logits,
                    labels,
                    pi,
                    include_positive_in_unlabeled=self.calibration_applied_,
                    clamp_logits=False,
                )
                entropy_scores = scores if self.calibration_applied_ else scores[labels == 0]
                loss = hist_loss + self.warmup_entropy_weight * _entropy_minimization(
                    entropy_scores
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Dist-PU warm-up objective became nonfinite")
                warmup_optimizer.zero_grad()
                loss.backward()
                warmup_optimizer.step()
                self.optimizer_steps_ += 1
                batch_losses.append(float(loss.detach().cpu()))
            warmup_scheduler.step()
            self._record_author_epoch(
                "warmup", epoch + 1, float(np.mean(batch_losses)), epoch_callback
            )

        # Match MixupDataset.update_psudos(): inference in stable source row
        # order, with pseudo-labels subsequently indexed by each row identity.
        self.model_.eval()
        pseudo_labels = torch.empty(n_samples, dtype=torch.float32)
        with torch.no_grad():
            pseudo_batch_size = self.training_provenance_["pseudo_label_inference_batch_size"]
            for start in range(0, n_samples, pseudo_batch_size):
                stop = min(start + pseudo_batch_size, n_samples)
                logits = self._staged_logits(tx[start:stop].to(device))
                pseudo_labels[start:stop] = torch.sigmoid(logits).cpu()
        self.pseudo_labels_ = pseudo_labels

        pu_optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.pu_learning_rate,
            weight_decay=self.pu_weight_decay,
        )
        pu_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            pu_optimizer,
            T_max=self.pu_epochs,
            eta_min=0.7 * self.pu_learning_rate,
        )
        self.model_.train()
        for epoch in range(self.pu_epochs):
            co_entropy = (
                1.0 - np.cos((epoch / self.pu_epochs) * (np.pi / 2.0))
            ) * self.pu_entropy_weight
            permutation = torch.randperm(n_samples)
            batch_losses = []
            for start in range(0, n_samples, self.batch_size):
                indices = permutation[start : start + self.batch_size]
                batch_x = tx[indices].to(device)
                labels = ty[indices].to(device)
                targets = self.pseudo_labels_[indices].to(device).clone()
                targets[labels == 1] = 1.0

                lam = float(np_rng.beta(self.mixup_alpha, self.mixup_alpha))
                mix_permutation = torch.randperm(len(indices), device=device)
                mixed_x = lam * batch_x + (1.0 - lam) * batch_x[mix_permutation]
                target_a = targets
                target_b = targets[mix_permutation]
                mixed_logits = self._staged_logits(mixed_x)
                mixed_scores = torch.sigmoid(torch.clamp(mixed_logits, min=-10.0, max=10.0))
                raw_logits = torch.clamp(self._staged_logits(batch_x), min=-10.0, max=10.0)
                raw_scores = torch.sigmoid(raw_logits)

                base_loss = _author_histogram_batch_loss(
                    raw_logits,
                    labels,
                    pi,
                    include_positive_in_unlabeled=self.calibration_applied_,
                )
                entropy_scores = (
                    raw_scores if self.calibration_applied_ else raw_scores[labels == 0]
                )
                loss = (
                    base_loss
                    + co_entropy * _entropy_minimization(entropy_scores)
                    + self.mixup_entropy_weight * _entropy_minimization(mixed_scores)
                    + self.mixup_bce_weight
                    * (
                        lam * functional.binary_cross_entropy(mixed_scores, target_a)
                        + (1.0 - lam) * functional.binary_cross_entropy(mixed_scores, target_b)
                    )
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Dist-PU PU-stage objective became nonfinite")
                pu_optimizer.zero_grad()
                loss.backward()
                pu_optimizer.step()
                with torch.no_grad():
                    # The source scores were evaluated before optimizer.step().
                    self.pseudo_labels_[indices] = raw_scores.detach().cpu()
                pu_scheduler.step()
                self.optimizer_steps_ += 1
                batch_losses.append(float(loss.detach().cpu()))
            self._record_author_epoch(
                "dist_pu", epoch + 1, float(np.mean(batch_losses)), epoch_callback
            )

        self.model_.eval()
        self.classes_ = np.array([0, 1])
        self._class_prior, self._X_shape_, self._is_fitted = pi, tuple(tx.shape), True
        warmup_steps = self.warmup_epochs
        pu_steps = self.pu_epochs * int(np.ceil(n_samples / self.batch_size))
        self.training_provenance_["scheduler_step_counts"] = {
            "warmup": int(warmup_scheduler.last_epoch),
            "dist_pu": int(pu_scheduler.last_epoch),
        }
        self.training_provenance_["expected_scheduler_step_counts"] = {
            "warmup": warmup_steps,
            "dist_pu": pu_steps,
        }
        self.training_provenance_["optimizer_steps"] = self.optimizer_steps_
        self.training_provenance_["epochs_completed"] = self.epochs_completed_
        self.device_ = device
        return self

    def _staged_logits(self, inputs):
        """Keep legacy MLP math and retain singleton encoder batches with frozen BN."""
        if self.encoder is None:
            return self.model_(inputs).reshape(-1)
        import torch

        modes = [(module, module.training) for module in self.model_.modules()]
        try:
            if len(inputs) == 1:
                for module in self.model_.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.eval()
            logits = self.model_(inputs.clone())
            if logits.shape not in ((len(inputs),), (len(inputs), 1)):
                raise ValueError("Dist-PU model must return one logit per row")
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Dist-PU encoder model produced nonfinite logits")
            return logits.reshape(-1)
        finally:
            for module, training in modes:
                module.training = training

    def _record_author_epoch(self, stage, stage_epoch, loss, epoch_callback):
        self.checkpoint_stage_ = stage
        self.checkpoint_stage_epoch_ = stage_epoch
        self.epochs_completed_ += 1
        self.loss_history_.append(loss)
        self.stage_loss_history_[stage].append(loss)
        self.epoch_provenance_.append(
            {
                "epoch": self.epochs_completed_,
                "stage": stage,
                "stage_epoch": stage_epoch,
                "optimizer_steps": self.optimizer_steps_,
            }
        )
        if epoch_callback is not None:
            epoch_callback(self.epochs_completed_, self)

    def _decision_function(self, X):
        self._check_is_fitted()

        import torch
        from sklearn.utils.validation import check_array

        X = check_array(X, dtype=np.float32, allow_nd=True, ensure_min_samples=0)
        expected_shape = tuple(self._X_shape_[1:])
        if tuple(X.shape[1:]) != expected_shape:
            raise ValueError(f"X must have sample shape {expected_shape}")
        if not len(X):
            return np.empty(0, dtype=np.float32)
        modes = [(module, module.training) for module in self.model_.modules()]
        try:
            self.model_.eval()
            with torch.no_grad():
                if self.encoder is None:
                    return (
                        self.model_(torch.as_tensor(X, device=self.device_))
                        .squeeze(1)
                        .cpu()
                        .numpy()
                    )
                return np.concatenate(
                    [
                        self._staged_logits(
                            torch.as_tensor(X[start : start + self.batch_size], device=self.device_)
                        )
                        .cpu()
                        .numpy()
                        for start in range(0, len(X), self.batch_size)
                    ]
                )
        finally:
            for module, training in modes:
                module.training = training

    def _predict(self, X):
        return (self._decision_function(X) >= 0).astype(int)

    def predict_proba(self, X):
        score = 1.0 / (1.0 + np.exp(-np.clip(self._decision_function(X), -40, 40)))
        return np.column_stack([1.0 - score, score])
