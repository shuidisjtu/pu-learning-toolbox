# ruff: noqa: N803, N806
"""Auditable paper-objective Holistic-PU dense adapter and trend components.

Paper equations (4),(5),(8) differ from the released simplified score and
jenkspy SSE objective. Variant names are explicit to prevent silent conflation.
"""

from __future__ import annotations

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

    Fixed warmup budget is an explicit adaptation, not the paper's LZO stopping.
    Source code's adjacent score/Jenks/fine-tuning differ from this paper path.
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
    native_architectures = frozenset({"mlp"})
    input_ndims = frozenset({2})
    encoder_parameter = None
    trains_encoder = False
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
            X, y_pu, accept_sparse=False, estimator_name="HolisticPUClassifier"
        )
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
        self.model_ = nn.Sequential(
            nn.Linear(X.shape[1], self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, 1)
        ).to(self.device_)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        data = torch.as_tensor(X)
        self.classes_ = np.array([0, 1])
        self.n_features_in_, self._X_shape_ = X.shape[1], X.shape
        self._class_prior = None
        self.training_view_, self.calibration_applied_ = "os", False
        self.n_positive_, self.n_unlabeled_ = len(positive), len(unlabeled)
        self.optimizer_steps_ = 0
        self.history_ = {
            "warmup_loss": [],
            "pseudo_pn_loss": [],
            "epoch": [],
            "phase": [],
            "train_loss": [],
        }
        self.stopping_rule_ = "fixed_warmup_budget_not_LZO"
        self.trend_variant_, self.partition_objective_ = "paper_pairwise", "paper_variance"
        self.pseudo_label_indices_ = unlabeled.copy()

        def update(loss):
            if not torch.isfinite(loss):
                raise ValueError("Holistic-PU loss became non-finite")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.optimizer_steps_ += 1

        trajectories = []
        for epoch in range(self.warmup_epochs):
            self.model_.train()
            losses = []
            order = rng.permutation(unlabeled)
            for start in range(0, len(order), self.batch_size):
                u = order[start : start + self.batch_size]
                p = rng.choice(positive, len(u), replace=True)
                loss = (
                    functional.softplus(-self.model_(data[p].to(self.device_))).mean()
                    + functional.softplus(self.model_(data[u].to(self.device_))).mean()
                )
                update(loss)
                losses.append(float(loss.detach().cpu()))
            self.model_.eval()
            trajectories.append(self._scores(X[unlabeled]))
            self.history_["warmup_loss"].append(float(np.mean(losses)))
            self._record_epoch(epoch, "warmup", losses, epoch_callback)
        from scipy.special import expit

        self.prediction_trajectory_ = expit(np.stack(trajectories, axis=1))
        self.trend_scores_ = holistic_trend_scores(
            self.prediction_trajectory_, scale=self.trend_scale
        )
        self.breakpoint_, self.partition_cost_, high = holistic_natural_break(self.trend_scores_)
        self.pseudo_labels_ = high.astype(int)  # toolkit positive=1 (source positive=0)
        self.estimated_unlabeled_prior_ = float(high.mean())  # diagnostic, not a supplied prior
        labels = y_pu.astype(np.float32).copy()
        labels[unlabeled] = self.pseudo_labels_
        targets = torch.as_tensor(labels)
        # Continue the warmup classifier on P + pseudo-labeled U, no clean selection.
        for epoch in range(self.max_epochs):
            self.model_.train()
            losses = []
            order = rng.permutation(len(X))
            for start in range(0, len(order), self.batch_size):
                indices = order[start : start + self.batch_size]
                logits = self.model_(data[indices].to(self.device_)).reshape(-1)
                loss = functional.binary_cross_entropy_with_logits(
                    logits, targets[indices].to(self.device_)
                )
                update(loss)
                losses.append(float(loss.detach().cpu()))
            self.history_["pseudo_pn_loss"].append(float(np.mean(losses)))
            self.model_.eval()
            self._record_epoch(self.warmup_epochs + epoch, "pseudo_pn", losses, epoch_callback)
        self.model_.eval()
        self._is_fitted = True
        return self

    def _record_epoch(self, epoch, phase, losses, callback):
        self.checkpoint_stage_ = phase
        self.checkpoint_stage_epoch_ = (
            epoch + 1 if phase == "warmup" else epoch - self.warmup_epochs + 1
        )
        self.checkpoint_round_ = None
        self.history_["epoch"].append(epoch)
        self.history_["phase"].append(phase)
        self.history_["train_loss"].append(float(np.mean(losses)))
        if callback is not None:
            callback(epoch, self)

    def _scores(self, X):
        import torch

        with torch.no_grad():
            return np.concatenate(
                [
                    self.model_(torch.as_tensor(X[i : i + self.batch_size], device=self.device_))
                    .reshape(-1)
                    .cpu()
                    .numpy()
                    for i in range(0, len(X), self.batch_size)
                ]
            )

    def _decision_function(self, X):
        from sklearn.utils.validation import check_array

        X = check_array(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Holistic-PU feature count differs from training")
        return self._scores(X)

    def _predict(self, X):
        return (self._decision_function(X) >= 0).astype(int)

    def predict_proba(self, X):
        from scipy.special import expit

        self._check_is_fitted()
        positive = expit(self._decision_function(X))
        return np.column_stack((1 - positive, positive))
