# ruff: noqa: N803, N806
"""Auditable paper-objective Holistic-PU MLP/CNN adapter and trend components.

Paper equations (4),(5),(8) differ from the released simplified score and
jenkspy SSE objective. Variant names are explicit to prevent silent conflation.
"""

from __future__ import annotations

import copy

import numpy as np

from ...core.base import BasePUClassifier
from ...core.device import resolve_device
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
from ...core.training_views import build_training_view
from ...core.validation import validate_pu_X_y
from ._validation import validate_encoder_features


def holistic_trend_scores(probabilities, *, variant="paper_pairwise", scale=2.0):
    """Scores from sample-by-time positive probabilities (at least two times).

    paper_pairwise uses all ordered pairs and the signed robust log influence.
    author_adjacent matches the published adjacent unsigned-log expression,
    whose negative-delta behavior differs from the paper's signed influence.
    """
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("Holistic-PU needs nonempty sample-by-time probabilities, >=2 times")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Holistic-PU probabilities must be finite and in [0,1]")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if variant == "author_adjacent":
        if scale != 1.0:
            raise ValueError("author_adjacent source uses scale=1.0")
        changes = np.diff(values, axis=1)
        return np.log1p(changes + 0.5 * changes**2).mean(axis=1)
    if variant != "paper_pairwise":
        raise ValueError("variant must be paper_pairwise or author_adjacent")
    total = np.zeros(values.shape[0])
    times = values.shape[1]
    # O(n*t^2), but only O(n*t) temporary memory, no n*t*t tensor.
    for i in range(times - 1):
        changes = scale * (values[:, i + 1 :] - values[:, i : i + 1])
        magnitude = np.abs(changes)
        total += (np.sign(changes) * np.log1p(magnitude + 0.5 * magnitude**2)).sum(axis=1)
    return total * (2.0 / (times * (times - 1)))


def holistic_natural_break(scores, *, objective="paper_variance"):
    """Exact sorted binary split, ties intact and deterministic first minimum.

    paper_variance minimizes SSE_left/n_left + SSE_right/n_right (Eq.8).
    author_sse minimizes SSE_left+SSE_right (published jenkspy behavior).
    Constant scores fail rather than fabricating a class/prior estimate.
    Returns the largest low-group score, objective value, and high-group mask.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("natural break needs >=2 finite one-dimensional scores")
    if objective not in {"paper_variance", "author_sse"}:
        raise ValueError("objective must be paper_variance or author_sse")
    ordered = np.sort(values, kind="stable")
    cuts = np.flatnonzero(ordered[:-1] < ordered[1:]) + 1
    if not len(cuts):
        raise ValueError("constant trend scores cannot identify two natural-break classes")
    # Center/scale before prefix moments for finite, stable SSE on tiny scores.
    span = ordered[-1] - ordered[0]
    if not np.isfinite(span) or span <= 0:
        raise ValueError("natural-break score range must remain finite and positive")
    centered = (ordered - ordered[0]) / span
    sums, squares = np.cumsum(centered), np.cumsum(centered**2)
    left_sum, left_square = sums[cuts - 1], squares[cuts - 1]
    right_sum, right_square = sums[-1] - left_sum, squares[-1] - left_square
    left_count, right_count = cuts, len(values) - cuts
    left_sse = np.maximum(0, left_square - left_sum**2 / left_count)
    right_sse = np.maximum(0, right_square - right_sum**2 / right_count)
    costs = (
        left_sse / left_count + right_sse / right_count
        if objective == "paper_variance"
        else left_sse + right_sse
    )
    winner = int(np.argmin(costs))
    cutoff = float(ordered[cuts[winner] - 1])
    return cutoff, float(costs[winner] * span**2), values > cutoff


class HolisticPUClassifier(BasePUClassifier):
    """Balanced resampling, paper pairwise trend score, then pseudo-PN training.

    Fixed warmup is the default; lzo_positive_loss is an explicit engineering
    recipe for label-invariant positive mixup validation, not author-code replay.
    Source code's adjacent score/Jenks/fine-tuning differ from this paper path.
    pseudo_pn_initialization="reinitialize" builds a new network and optimizer;
    the legacy default "continue" retains both after trend partitioning.
    No test labels, clean support, or population prior are consumed in fit.
    """

    family = AlgorithmFamily.DEEP_PU
    label_semantics = "pu"
    assumption = (Assumption.SCAR,)
    scenario = (Scenario.SINGLE_TRAINING_SET,)
    requires_class_prior = False
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.OFFICIAL_RELATED
    backend = Backend.TORCH
    maturity = Maturity.EXPERIMENTAL
    sample_weight_support = SampleWeightSupport.NOT_IMPLEMENTED
    native_architectures = frozenset({"mlp", "cnn"})
    input_ndims = frozenset({2, 4})
    encoder_parameter = "encoder"
    trains_encoder = True
    checkpoint_stages = ("warmup", "pseudo_pn")

    @property
    def checkpoint_prediction_batch_size(self):
        """Preserve inference chunks; changing them can change float32 scores."""
        return self.batch_size

    @property
    def checkpoint_epoch_count(self):
        """Count both stages for peak checkpoint storage, not only pseudo-PN."""
        return self.warmup_epochs + self.max_epochs

    def __init__(
        self,
        *,
        hidden_dim=64,
        warmup_epochs=30,
        max_epochs=100,
        batch_size=64,
        learning_rate=1e-3,
        trend_scale=2.0,
        encoder=None,
        pseudo_pn_initialization="continue",
        warmup_selection="fixed",
        lzo_alpha=0.5,
        lzo_validation_size=None,
        random_state=0,
        device=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.trend_scale = trend_scale
        self.encoder = encoder
        self.pseudo_pn_initialization = pseudo_pn_initialization
        self.warmup_selection = warmup_selection
        self.lzo_alpha = lzo_alpha
        self.lzo_validation_size = lzo_validation_size
        self.random_state = random_state
        self.device = device

    def fit(
        self, X, y_pu, *, class_prior=None, sample_weight=None, os_or_ts="os", epoch_callback=None
    ):
        import torch
        from torch import nn
        from torch.nn import functional

        self._is_fitted = False
        if sample_weight is not None:
            raise NotImplementedError("Holistic-PU does not implement sample_weight")
        if os_or_ts != "os":
            raise ValueError("Holistic-PU OS resampling has no equivalent ts risk substitution")
        if not isinstance(
            self.pseudo_pn_initialization, str
        ) or self.pseudo_pn_initialization not in {"continue", "reinitialize"}:
            raise ValueError("pseudo_pn_initialization must be continue or reinitialize")
        if not isinstance(self.warmup_selection, str) or self.warmup_selection not in {
            "fixed",
            "lzo_positive_loss",
        }:
            raise ValueError("warmup_selection must be fixed or lzo_positive_loss")
        if (
            isinstance(self.lzo_alpha, (bool, np.bool_))
            or not isinstance(self.lzo_alpha, (int, float, np.integer, np.floating))
            or not np.isfinite(self.lzo_alpha)
            or self.lzo_alpha <= 0
        ):
            raise ValueError("lzo_alpha must be finite and positive")
        if self.lzo_validation_size is not None and (
            type(self.lzo_validation_size) is not int or self.lzo_validation_size < 1
        ):
            raise ValueError("lzo_validation_size must be None or a positive integer")
        for name in ("hidden_dim", "warmup_epochs", "max_epochs", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.warmup_epochs < 2:
            raise ValueError("warmup_epochs must be >=2 to measure predictive trends")
        for name in ("learning_rate", "trend_scale"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if class_prior is not None and (not np.isfinite(class_prior) or not 0 < class_prior < 1):
            raise ValueError("class_prior, if supplied, must be in (0,1); it is not used")
        X, y_pu = validate_pu_X_y(
            X, y_pu, accept_sparse=False, allow_nd=True, estimator_name="HolisticPUClassifier"
        )
        if X.ndim not in (2, 4):
            raise ValueError("Holistic-PU supports 2-D features or 4-D NCHW images")
        if X.ndim == 4 and self.encoder is None:
            raise ValueError("Holistic-PU 4-D images require an explicit encoder; no flattening")
        if self.encoder is not None and not isinstance(self.encoder, nn.Module):
            raise TypeError("encoder must be a torch.nn.Module")
        if self.encoder is not None and self.pseudo_pn_initialization == "reinitialize":
            self._check_encoder_reinitialization(self.encoder)
        X = np.asarray(X, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError("X must remain finite after float32 conversion")
        view = build_training_view(X, y_pu, requested_view="os")
        positive, unlabeled = view.positive_positions.copy(), view.native_unlabeled_positions.copy()
        if len(unlabeled) < 2:
            raise ValueError("Holistic-PU needs >=2 unlabeled samples for natural-break partition")
        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(int(rng.randint(0, 2**31)))
        self.device_ = resolve_device(self.device)
        if self.warmup_selection != "fixed" and self.device_.type not in {"cpu", "cuda"}:
            raise ValueError("LZO RNG restoration supports CPU or CUDA")
        self.model_, self.encoder_ = self._build_model(X)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        data = torch.as_tensor(X)
        self.classes_ = np.array([0, 1])
        self.n_features_in_, self._X_shape_ = X.shape[1], X.shape
        self.input_shape_ = tuple(X.shape[1:])
        self._class_prior = None
        self.training_view_, self.calibration_applied_ = "os", False
        self.n_positive_, self.n_unlabeled_ = len(positive), len(unlabeled)
        self.optimizer_steps_ = 0
        self.stage_optimizer_steps_ = {"warmup": 0, "pseudo_pn": 0}
        self.pseudo_pn_initialization_ = self.pseudo_pn_initialization
        self.pseudo_pn_optimizer_reset_ = False
        self.history_ = {
            "warmup_loss": [],
            "pseudo_pn_loss": [],
            "epoch": [],
            "phase": [],
            "train_loss": [],
            "optimizer_steps": [],
            "warmup_validation_loss": [],
        }
        self.stopping_rule_ = (
            "fixed_warmup_budget_not_LZO"
            if self.warmup_selection == "fixed"
            else "lzo_positive_loss_full_horizon_argmin"
        )
        self.trend_variant_, self.partition_objective_ = "paper_pairwise", "paper_variance"
        self.pseudo_label_indices_ = unlabeled.copy()
        self.selected_warmup_epoch_ = self.warmup_epochs
        self.executed_warmup_epochs_ = 0
        self.discarded_warmup_optimizer_steps_ = 0
        self._prepare_lzo(positive, rng)

        def update(loss, stage):
            if not torch.isfinite(loss):
                raise ValueError("Holistic-PU loss became non-finite")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.optimizer_steps_ += 1
            self.stage_optimizer_steps_[stage] += 1

        trajectories = []
        best_state, best_loss = None, float("inf")
        for epoch in range(self.warmup_epochs):
            self.model_.train()
            losses = []
            order = rng.permutation(unlabeled)
            for start in range(0, len(order), self.batch_size):
                u = order[start : start + self.batch_size]
                p = rng.choice(positive, len(u), replace=True)
                loss = (
                    functional.softplus(-self._logits(data[p].to(self.device_))).mean()
                    + functional.softplus(self._logits(data[u].to(self.device_))).mean()
                )
                update(loss, "warmup")
                losses.append(float(loss.detach().cpu()))
            self.model_.eval()
            trajectories.append(self._scores(X[unlabeled]))
            self.history_["warmup_loss"].append(float(np.mean(losses)))
            self.executed_warmup_epochs_ = epoch + 1
            if self.warmup_selection == "lzo_positive_loss":
                validation_loss = self._lzo_loss(X)
                if not np.isfinite(validation_loss):
                    raise ValueError("LZO validation loss must remain finite")
                self.history_["warmup_validation_loss"].append(float(validation_loss))
                # Trend equations require >=2 observations; epoch 1 is diagnostic
                # only. Strict comparison chooses the earliest exact tied risk.
                if epoch >= 1 and validation_loss < best_loss:
                    best_loss = validation_loss
                    self.selected_warmup_epoch_ = epoch + 1
                    best_state = self._capture_lzo_state(optimizer, rng)
            self._record_epoch(epoch, "warmup", losses, epoch_callback)
        from scipy.special import expit

        self.observed_prediction_trajectory_ = expit(np.stack(trajectories, axis=1))
        self.prediction_trajectory_ = (
            self.observed_prediction_trajectory_
            if self.selected_warmup_epoch_ == self.warmup_epochs
            else self.observed_prediction_trajectory_[:, : self.selected_warmup_epoch_]
        )
        self.lzo_losses_ = np.asarray(self.history_["warmup_validation_loss"], dtype=float)
        if self.warmup_selection == "lzo_positive_loss":
            if best_state is None:
                raise ValueError("LZO did not produce an eligible warmup terminal state")
            self.discarded_warmup_optimizer_steps_ = (
                self.stage_optimizer_steps_["warmup"] - best_state["optimizer_steps"]
            )
            if self.pseudo_pn_initialization == "continue":
                self.model_.load_state_dict(best_state["model"])
                optimizer.load_state_dict(best_state["optimizer"])
            rng.set_state(best_state["numpy_rng"])
            torch.set_rng_state(best_state["torch_rng"])
            if best_state["cuda_rng"] is not None:
                torch.cuda.set_rng_state(best_state["cuda_rng"], self.device_)
            # Internal terminal selection is not a checkpoint resume API. Keep
            # only one best CPU state and release it before pseudo-PN training.
            best_state = None
        self.trend_scores_ = holistic_trend_scores(
            self.prediction_trajectory_, scale=self.trend_scale
        )
        self.breakpoint_, self.partition_cost_, high = holistic_natural_break(self.trend_scores_)
        self.pseudo_labels_ = high.astype(int)  # toolkit positive=1 (source positive=0)
        self.estimated_unlabeled_prior_ = float(high.mean())  # diagnostic, not a supplied prior
        labels = y_pu.astype(np.float32).copy()
        labels[unlabeled] = self.pseudo_labels_
        targets = torch.as_tensor(labels)
        if self.pseudo_pn_initialization == "reinitialize":
            # Supplement Algorithm 2 step 14 and author main.py:329-335 discard
            # the warmup model. Reset the CNN too, not just its score head, and
            # never carry warmup Adam moments into a newly initialized network.
            restarted, restarted_encoder = self._build_model(X, reinitialize_encoder=True)
            old_shapes = {name: value.shape for name, value in self.model_.state_dict().items()}
            new_shapes = {name: value.shape for name, value in restarted.state_dict().items()}
            if old_shapes != new_shapes:
                raise ValueError("reinitialized model architecture must match warmup checkpoints")
            self.model_, self.encoder_ = restarted, restarted_encoder
            optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
            self.pseudo_pn_optimizer_reset_ = True
        # Supervise P + pseudo-labeled U; neither variant reads clean selection.
        for epoch in range(self.max_epochs):
            self.model_.train()
            losses = []
            order = rng.permutation(len(X))
            for start in range(0, len(order), self.batch_size):
                indices = order[start : start + self.batch_size]
                logits = self._logits(data[indices].to(self.device_))
                loss = functional.binary_cross_entropy_with_logits(
                    logits, targets[indices].to(self.device_)
                )
                update(loss, "pseudo_pn")
                losses.append(float(loss.detach().cpu()))
            self.history_["pseudo_pn_loss"].append(float(np.mean(losses)))
            self.model_.eval()
            self._record_epoch(self.warmup_epochs + epoch, "pseudo_pn", losses, epoch_callback)
        self.model_.eval()
        self._is_fitted = True
        return self

    def _prepare_lzo(self, positive, training_rng):
        """Fixed train-P-only mixup plan; never advance the training RNG stream."""
        self.lzo_validation_size_ = 0
        self.lzo_mixup_indices_ = np.empty((0, 2), dtype=int)
        self.lzo_mixup_weights_ = np.empty(0)
        self.lzo_evaluated_rows_ = self.lzo_forward_batches_ = 0
        self.lzo_candidate_epochs_ = ()
        self.lzo_selection_spec_ = None
        if self.warmup_selection == "fixed":
            return
        if len(positive) < 2:
            raise ValueError("LZO mixup requires >=2 labeled training positives")
        size = len(positive) if self.lzo_validation_size is None else self.lzo_validation_size
        validation_rng = np.random.RandomState(0)
        validation_rng.set_state(training_rng.get_state())
        self.lzo_validation_size_ = size
        self.lzo_mixup_indices_ = validation_rng.choice(positive, size=(size, 2), replace=True)
        self.lzo_mixup_weights_ = validation_rng.beta(self.lzo_alpha, self.lzo_alpha, size=size)
        if not np.isfinite(self.lzo_mixup_weights_).all():
            raise ValueError("LZO beta samples must remain finite")
        self.lzo_candidate_epochs_ = tuple(range(2, self.warmup_epochs + 1))
        self.lzo_selection_spec_ = {
            "variant": "holistic-lzo-positive-loss-engineering-v1",
            "source": "labeled_training_positive_only",
            "metric": "mean_positive_binary_cross_entropy",
            "mixup_alpha": float(self.lzo_alpha),
            "validation_size": size,
            "sampling": "with_replacement_same_row_pairs_allowed_fixed_across_epochs",
            "rng": "independent_copy_of_local_training_numpy_state_after_model_seed",
            "candidate_epochs": list(self.lzo_candidate_epochs_),
            "tie_rule": "earliest_exact_minimum",
            "evaluation": "full_warmup_horizon_not_patience_early_exit",
        }

    def _lzo_loss(self, X):
        """Stable positive CE on lazily mixed CPU rows; eval preserves torch RNG."""
        import torch

        total = 0.0
        devices = (
            [self.device_.index if self.device_.index is not None else torch.cuda.current_device()]
            if self.device_.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices):
            for start in range(0, self.lzo_validation_size_, self.batch_size):
                pairs = self.lzo_mixup_indices_[start : start + self.batch_size]
                weights = self.lzo_mixup_weights_[start : start + self.batch_size].astype(
                    np.float32
                )
                weights = weights.reshape((-1,) + (1,) * (X.ndim - 1))
                mixed = weights * X[pairs[:, 0]] + (1 - weights) * X[pairs[:, 1]]
                logits = np.asarray(self._scores(mixed), dtype=float)
                total += float(np.logaddexp(0.0, -logits).sum())
                self.lzo_evaluated_rows_ += len(pairs)
                self.lzo_forward_batches_ += 1
        return total / self.lzo_validation_size_

    @staticmethod
    def _cpu_state_copy(value):
        """Copy one best state onto CPU, preserving state_dict version metadata."""
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            copied = type(value)(
                (key, HolisticPUClassifier._cpu_state_copy(item)) for key, item in value.items()
            )
            if hasattr(value, "_metadata"):
                copied._metadata = copy.deepcopy(value._metadata)
            return copied
        if isinstance(value, (list, tuple)):
            return type(value)(HolisticPUClassifier._cpu_state_copy(item) for item in value)
        return copy.deepcopy(value)

    def _capture_lzo_state(self, optimizer, rng):
        """Retain only best RNG and, for continue, best weights/Adam moments."""
        import torch

        continues = self.pseudo_pn_initialization == "continue"
        return {
            "model": self._cpu_state_copy(self.model_.state_dict()) if continues else None,
            "optimizer": self._cpu_state_copy(optimizer.state_dict()) if continues else None,
            "optimizer_steps": self.optimizer_steps_,
            "numpy_rng": copy.deepcopy(rng.get_state()),
            "torch_rng": torch.get_rng_state().clone(),
            "cuda_rng": torch.cuda.get_rng_state(self.device_).clone()
            if self.device_.type == "cuda"
            else None,
        }

    @staticmethod
    def _check_encoder_reinitialization(encoder):
        """Refuse parameter owners without an explicit reset contract.

        Built-in Conv/Linear/BN layers implement reset_parameters. A custom
        parameter owner must do so too: copying initial/pretrained weights is
        not a substitute for randomly reinitializing the source algorithm.
        """
        for name, layer in encoder.named_modules():
            own_parameters = dict(layer.named_parameters(recurse=False))
            if {"weight_g", "weight_v"} <= own_parameters.keys():
                raise ValueError(
                    "reinitialize encoder does not support legacy weight_norm parameter owners; "
                    f"reset_parameters does not reset weight_g/weight_v at {name or '<root>'}"
                )
            if own_parameters and not callable(getattr(layer, "reset_parameters", None)):
                raise ValueError(
                    "reinitialize encoder requires reset_parameters on every parameter owner; "
                    f"missing at {name or '<root>'} ({type(layer).__name__})"
                )

    def _build_model(self, X, *, reinitialize_encoder=False):
        """Independent encoder/head; probing is eval-only and device-batched."""
        import torch
        from torch import nn

        own_encoder = None
        width = X.shape[1]
        if self.encoder is not None:
            own_encoder = copy.deepcopy(self.encoder).to(device=self.device_, dtype=torch.float32)
            if reinitialize_encoder:
                for layer in own_encoder.modules():
                    reset = getattr(layer, "reset_parameters", None)
                    has_own_state = list(layer.parameters(recurse=False)) or list(
                        layer.buffers(recurse=False)
                    )
                    if has_own_state and callable(reset):
                        reset()
            own_encoder.requires_grad_(True).eval()
            with torch.no_grad():
                features = own_encoder(torch.as_tensor(X[:1], device=self.device_))
            width = validate_encoder_features(features, encoder_param_name="encoder")
            if features.shape[0] != 1:
                raise ValueError("encoder must preserve the input batch dimension")
        head = nn.Sequential(
            nn.Linear(width, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, 1)
        ).to(self.device_)
        return (nn.Sequential(own_encoder, head) if own_encoder is not None else head), own_encoder

    def _record_epoch(self, epoch, phase, losses, callback):
        self.checkpoint_stage_ = phase
        self.checkpoint_stage_epoch_ = (
            epoch + 1 if phase == "warmup" else epoch - self.warmup_epochs + 1
        )
        self.checkpoint_round_ = None
        self.history_["epoch"].append(epoch)
        self.history_["phase"].append(phase)
        self.history_["train_loss"].append(float(np.mean(losses)))
        self.history_["optimizer_steps"].append(self.optimizer_steps_)
        if callback is not None:
            callback(epoch, self)

    def _logits(self, batch):
        """One finite logit per row; singleton tails use running BN statistics.

        Keep every training row instead of dropping one-row remainder batches.
        BN affine parameters still receive gradients; restore modes on failure.
        """
        import torch

        single_row_bn = (
            [
                layer
                for layer in self.model_.modules()
                if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm) and layer.training
            ]
            if len(batch) == 1
            else []
        )
        try:
            for layer in single_row_bn:
                layer.eval()
            logits = self.model_(batch)
        finally:
            for layer in single_row_bn:
                layer.train()
        if not torch.is_tensor(logits) or logits.shape != (len(batch), 1):
            raise ValueError("Holistic-PU model must return one logit per input row")
        if not torch.isfinite(logits).all():
            raise ValueError("Holistic-PU logits became non-finite")
        return logits[:, 0]

    def _scores(self, X):
        import torch

        if not len(X):
            return np.empty(0, dtype=np.float32)
        modes = [(layer, layer.training) for layer in self.model_.modules()]
        self.model_.eval()
        try:
            with torch.no_grad():
                return np.concatenate(
                    [
                        self._logits(
                            torch.as_tensor(X[i : i + self.batch_size], device=self.device_)
                        )
                        .cpu()
                        .numpy()
                        for i in range(0, len(X), self.batch_size)
                    ]
                )
        finally:
            for layer, training in modes:
                layer.training = training

    def _decision_function(self, X):
        from sklearn.utils.validation import check_array

        X = check_array(X, dtype=np.float32, allow_nd=True, ensure_min_samples=0)
        if tuple(X.shape[1:]) != self.input_shape_:
            raise ValueError("Holistic-PU input shape differs from training")
        return self._scores(X)

    def _predict(self, X):
        return (self._decision_function(X) >= 0).astype(int)

    def predict_proba(self, X):
        from scipy.special import expit

        self._check_is_fitted()
        positive = expit(self._decision_function(X))
        return np.column_stack((1 - positive, positive))
