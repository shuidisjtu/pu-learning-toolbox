"""Transform-Estimate-Discard, Garg et al. (NeurIPS 2021), Algorithm 3.

Independent native-TS engineering path, not a rename of fixed-alpha CVIR.
TEDn deliberately uses unweighted positive/retained-U losses (paper p8).
No clean U labels, external selection data, or population prior are accepted.
"""

# Feature/role arguments follow the project's X/y convention.
# ruff: noqa: N803, N806

from __future__ import annotations

import copy
from numbers import Real

import numpy as np

from ...core.base import BasePUClassifier
from ...core.device import resolve_device
from ...core.exceptions import NotFittedError
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
from ...core.validation import validate_pu_X_y
from ...prior.bbe import _unit_parameter, best_bin_estimate


def select_tedn_negatives(scores: np.ndarray, alpha_u: float) -> np.ndarray:
    """Discard floor(alpha_U*n_U) highest scores, stably retaining ties.

    alpha_U=0 retains all U. alpha_U=1 raises: there is no negative-role
    mean to optimize. Never clip alpha or silently force one retained row.
    """
    scores = np.asarray(scores)
    if scores.ndim != 1 or not len(scores) or scores.dtype.kind not in "iuf":
        raise ValueError("scores must be a nonempty 1-D real array")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if (
        isinstance(alpha_u, (bool, np.bool_))
        or not isinstance(alpha_u, Real)
        or not 0 <= alpha_u <= 1
    ):
        raise ValueError("alpha_u must be finite and in [0, 1]")
    count = len(scores) - int(np.floor(float(alpha_u) * len(scores)))
    if count == 0:
        raise ValueError(
            "BBE alpha_U=1 leaves no retained U: increase the explicit warm-start budget "
            "or inspect score informativeness; no clipping or one-row fallback is applied"
        )
    return np.argsort(scores, kind="stable")[:count]


def _tedn_objective(positive_logits, negative_logits):
    """Algorithm 3/p8: unweighted role means, not alpha-weighted CVIR."""
    from torch.nn import functional

    return (
        functional.softplus(-positive_logits).mean() + functional.softplus(negative_logits).mean()
    )


def _features(X, *, has_encoder):
    X = np.asarray(X)
    if (
        X.ndim not in ({2, 4} if has_encoder else {2})
        or any(d == 0 for d in X.shape[1:])
        or X.dtype.kind not in "iuf"
        or not np.isfinite(X).all()
    ):
        raise ValueError("TEDn needs finite 2-D features or 4-D images with an encoder")
    with np.errstate(over="ignore"):
        X = X.astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError("TEDn features must remain finite in float32")
    return X


def _logits(model, batch):
    import torch

    logits = model(batch)
    if logits.shape != (len(batch), 1):
        raise ValueError("TEDn network must preserve batch dimension and output one logit per row")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("TEDn network produced nonfinite logits")
    return logits.reshape(-1)


class TEDnClassifier(BasePUClassifier):
    """Train a score transform and repeatedly estimate/discard/update.

    Explicit input_sampling='ts' is required; the caller owns verification of
    independent P/marginal-U sampling and external identity/group isolation.
    Internal holdout rows are used only for BBE evaluation, never optimizer
    steps, BatchNorm training, or the training-error stopping criterion.

    The default MLP/Adam and finite convergence tolerance/patience are engineering
    choices, not the author's image/text recipes or an approved survey budget.
    This class is not registered in the OS pipeline or frozen survey matrix.
    """

    family = AlgorithmFamily.RISK_ESTIMATION
    label_semantics = "pu"
    assumption = (Assumption.SCAR,)
    scenario = (Scenario.CASE_CONTROL,)
    requires_class_prior = False
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.OFFICIAL_EXACT
    backend = Backend.TORCH
    maturity = Maturity.EXPERIMENTAL
    sample_weight_support = SampleWeightSupport.NOT_IMPLEMENTED
    native_architectures = frozenset({"mlp", "cnn"})
    input_ndims = frozenset({2, 4})
    encoder_parameter = "encoder"
    trains_encoder = True
    checkpoint_stages = ("warm_start", "tedn")

    def __init__(
        self,
        *,
        input_sampling: str | None = None,
        holdout_fraction: float = 0.2,
        delta: float = 0.1,
        gamma: float = 0.01,
        hidden_dim: int = 64,
        warm_start_epochs: int = 5,
        max_epochs: int = 20,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        convergence_tol: float = 1e-4,
        convergence_patience: int = 5,
        random_state: int | None = None,
        device: str | None = None,
        encoder=None,
    ) -> None:
        super().__init__()
        self.input_sampling = input_sampling
        self.holdout_fraction = holdout_fraction
        self.delta = delta
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.warm_start_epochs = warm_start_epochs
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.convergence_tol = convergence_tol
        self.convergence_patience = convergence_patience
        self.random_state = random_state
        self.device = device
        self.encoder = encoder

    @property
    def checkpoint_epoch_count(self):
        """Upper bound including warm-start; convergence may end training earlier."""
        return self.warm_start_epochs + self.max_epochs

    @property
    def checkpoint_prediction_batch_size(self):
        return self.batch_size

    def fit(
        self,
        X: np.ndarray,
        y_pu: np.ndarray,
        *,
        class_prior: float | None = None,
        sample_weight: np.ndarray | None = None,
        sample_ids: np.ndarray | None = None,
        epoch_callback=None,
    ) -> TEDnClassifier:
        """Fit from P/U roles alone; reject population pi and extra clean labels."""
        self._is_fitted = False
        for name in ("alpha_u_", "bbe_history_", "training_provenance_", "stop_reason_"):
            self.__dict__.pop(name, None)
        try:
            return self._fit(
                X,
                y_pu,
                class_prior=class_prior,
                sample_weight=sample_weight,
                sample_ids=sample_ids,
                epoch_callback=epoch_callback,
            )
        except Exception:
            self._is_fitted = False
            self.stop_reason_ = "fit_failed"
            raise

    def _fit(self, X, y_pu, *, class_prior, sample_weight, sample_ids, epoch_callback):
        import torch

        if class_prior is not None:
            raise ValueError(
                "TEDn estimates alpha_U from held-out P/U; population class_prior rejected"
            )
        if sample_weight is not None:
            raise NotImplementedError("TEDn does not implement sample_weight")
        if epoch_callback is not None and not callable(epoch_callback):
            raise TypeError("epoch_callback must be callable or None")
        if self.input_sampling != "ts":
            raise ValueError("TEDn requires explicit input_sampling='ts'; no OS-to-TS conversion")
        fraction = _unit_parameter(self.holdout_fraction, "holdout_fraction")
        _unit_parameter(self.delta, "delta")
        _unit_parameter(self.gamma, "gamma")
        for name in (
            "hidden_dim",
            "warm_start_epochs",
            "max_epochs",
            "batch_size",
            "convergence_patience",
        ):
            minimum = 0 if name == "warm_start_epochs" else 1
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("learning_rate", "convergence_tol"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value < 0
                or (name == "learning_rate" and value == 0)
            ):
                bound = "positive" if name == "learning_rate" else "nonnegative"
                raise ValueError(f"{name} must be finite and {bound}")
        if self.encoder is not None and not isinstance(self.encoder, torch.nn.Module):
            raise TypeError("encoder must be a torch.nn.Module")
        X, roles = validate_pu_X_y(
            X,
            y_pu,
            accept_sparse=False,
            allow_nd=self.encoder is not None,
            estimator_name="TEDnClassifier",
        )
        X = _features(X, has_encoder=self.encoder is not None)
        if sample_ids is not None:
            ids = np.asarray(sample_ids)
            if (
                ids.ndim != 1
                or len(ids) != len(X)
                or ids.dtype.kind != "U"
                or any(not s.strip() for s in ids)
                or len(np.unique(ids)) != len(ids)
            ):
                raise ValueError("sample_ids must be unique nonempty strings, one per row")
        rng = np.random.RandomState(self.random_state)
        train_roles, held_roles = [], []
        for role in (1, 0):
            order = rng.permutation(np.flatnonzero(roles == role))
            n_held = int(np.ceil(fraction * len(order)))
            if not 1 <= n_held < len(order):
                raise ValueError("each P/U role needs nonempty training and holdout rows")
            held_roles.append(order[:n_held])
            train_roles.append(order[n_held:])
        p_idx, u_idx = train_roles
        hp_idx, hu_idx = held_roles
        torch.manual_seed(int(rng.randint(0, 2**31)))
        device = resolve_device(self.device)
        layers, width = [], X.shape[1]
        self.encoder_ = None
        if self.encoder is not None:
            from ._validation import validate_encoder_features

            self.encoder_ = copy.deepcopy(self.encoder).to(device=device, dtype=torch.float32)
            self.encoder_.requires_grad_(True).eval()
            with torch.no_grad():
                features = self.encoder_(torch.as_tensor(X[p_idx[:1]], device=device))
            width = validate_encoder_features(features, encoder_param_name="encoder")
            if features.shape[0] != 1:
                raise ValueError("encoder must preserve the batch dimension")
            layers.append(self.encoder_)
        layers.extend(
            (
                torch.nn.Linear(width, self.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_dim, 1),
            )
        )
        self.model_ = torch.nn.Sequential(*layers).to(device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        data = torch.as_tensor(X, device="cpu")
        self.data_device_policy_, self.training_data_device_ = "cpu_stream", "cpu"
        self.input_shape_ = tuple(X.shape[1:])
        self.n_features_in_ = X.shape[1]
        self._X_shape_ = X.shape
        self.classes_ = np.array([0, 1])
        self.train_positive_indices_, self.train_unlabeled_indices_ = p_idx.copy(), u_idx.copy()
        self.holdout_positive_indices_, self.holdout_unlabeled_indices_ = (
            hp_idx.copy(),
            hu_idx.copy(),
        )
        self.optimizer_steps_, self.n_iter_, self.epochs_completed_ = 0, 0, 0
        self.converged_, self.stop_reason_ = False, "max_epochs"
        self.history_ = {
            k: []
            for k in (
                "warm_start_loss",
                "tedn_loss",
                "alpha_u",
                "selected_negative_count",
                "training_error",
            )
        }
        self.bbe_history_, self.epoch_provenance_ = [], []
        self.selected_negative_indices_ = np.array([], dtype=int)
        self.training_provenance_ = {
            "method": "tedn",
            "input_sampling": "ts",
            "sampling_design_verified": False,
            "prior_target": "unlabeled_positive_fraction_alpha_U",
            "population_prior_conversion": None,
            "train_positive": len(p_idx),
            "train_unlabeled": len(u_idx),
            "holdout_positive": len(hp_idx),
            "holdout_unlabeled": len(hu_idx),
            "holdout_fraction": float(fraction),
            "delta": float(self.delta),
            "gamma": float(self.gamma),
            "sample_ids_checked": sample_ids is not None,
            "train_holdout_positionally_disjoint": True,
            "clean_labels_read": False,
            "external_selection_data_read": False,
            "bbe_timing": "before_each_tedn_round_update_after_warm_start",
            "objective": "unweighted_sum_of_positive_and_retained_negative_mean_bce",
            "stopping_data": "training_positive_and_current_retained_unlabeled_only",
            "warm_start_epochs": self.warm_start_epochs,
            "tedn_round_cap": self.max_epochs,
            "checkpoint_epoch_count_is_upper_bound": True,
            "convergence_tol": float(self.convergence_tol),
            "convergence_patience": self.convergence_patience,
            "holdout_reuse_does_not_certify_sequential_confidence": True,
            "negative_count_rule": "n_U-floor(alpha_U*n_U); alpha_U=1 fails loud",
            "data_device_policy": "cpu_stream",
            "max_train_forward_rows": 2 * self.batch_size,
            "max_eval_forward_rows": self.batch_size,
            "calibration_applied": False,
            "calibration_hooked": False,
            "formal_admission": False,
        }

        def scores(indices):
            return self._score_data(data, indices, device)

        def train_epoch(negatives):
            positive_batches = np.array_split(
                rng.permutation(p_idx), range(self.batch_size, len(p_idx), self.batch_size)
            )
            negative_batches = np.array_split(
                rng.permutation(negatives), range(self.batch_size, len(negatives), self.batch_size)
            )
            losses = []
            self.model_.train()
            for step in range(max(len(positive_batches), len(negative_batches))):
                p, n = (
                    positive_batches[step % len(positive_batches)],
                    negative_batches[step % len(negative_batches)],
                )
                logits = _logits(self.model_, data[np.r_[p, n]].to(device))
                loss = _tedn_objective(logits[: len(p)], logits[len(p) :])
                if not torch.isfinite(loss):
                    raise FloatingPointError("TEDn loss became nonfinite")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.optimizer_steps_ += 1
                losses.append(float(loss.detach()))
            return float(np.mean(losses))

        def completed(stage, epoch):
            self.checkpoint_stage_, self.checkpoint_stage_epoch_ = stage, epoch
            self.checkpoint_round_ = epoch if stage == "tedn" else None
            self.epochs_completed_ += 1
            self.epoch_provenance_.append(
                dict(
                    stage=stage,
                    stage_epoch=epoch,
                    round_index=self.checkpoint_round_,
                    optimizer_steps=self.optimizer_steps_,
                )
            )
            self._is_fitted = True
            if epoch_callback is not None:
                epoch_callback(self.epochs_completed_ - 1, self)

        for epoch in range(self.warm_start_epochs):
            self.history_["warm_start_loss"].append(train_epoch(u_idx))
            completed("warm_start", epoch + 1)
        stable = 0
        for epoch in range(self.max_epochs):
            result = best_bin_estimate(
                torch.sigmoid(torch.from_numpy(scores(hp_idx))).numpy(),
                torch.sigmoid(torch.from_numpy(scores(hu_idx))).numpy(),
                delta=self.delta,
                gamma=self.gamma,
            )
            self.alpha_u_ = result.alpha_u
            self.bbe_history_.append(
                {
                    "round_index": epoch + 1,
                    "optimizer_steps_before_update": self.optimizer_steps_,
                    "alpha_u": result.alpha_u,
                    "threshold": result.threshold,
                    "positive_tail_count": result.positive_counts[result.selected_index],
                    "unlabeled_tail_count": result.unlabeled_counts[result.selected_index],
                    "selection_upper_bound": result.selection_upper_bounds[result.selected_index],
                    "n_positive": result.n_positive,
                    "n_unlabeled": result.n_unlabeled,
                    "delta": result.delta,
                    "gamma": result.gamma,
                }
            )
            selected = u_idx[select_tedn_negatives(scores(u_idx), self.alpha_u_)]
            self.selected_negative_indices_ = selected.copy()
            self.history_["selected_negative_count"].append(len(selected))
            self.history_["alpha_u"].append(self.alpha_u_)
            self.history_["tedn_loss"].append(train_epoch(selected))
            error = float(np.mean(scores(p_idx) < 0) + np.mean(scores(selected) >= 0))
            previous = self.history_["training_error"][-1] if epoch else None
            self.history_["training_error"].append(error)
            self.n_iter_ += 1
            stable = (
                stable + 1
                if previous is not None and abs(error - previous) <= self.convergence_tol
                else 0
            )
            if stable >= self.convergence_patience:
                self.converged_, self.stop_reason_ = True, "training_error_plateau"
            completed("tedn", epoch + 1)
            if self.converged_:
                break
        return self

    def _score_data(self, data, indices, device):
        import torch

        self.model_.eval()
        with torch.no_grad():
            return (
                np.concatenate(
                    [
                        _logits(
                            self.model_, data[indices[start : start + self.batch_size]].to(device)
                        )
                        .cpu()
                        .numpy()
                        for start in range(0, len(indices), self.batch_size)
                    ]
                )
                if len(indices)
                else np.empty(0, dtype=np.float32)
            )

    def _decision_function(self, X):
        import torch

        X = _features(X, has_encoder=self.encoder is not None)
        if tuple(X.shape[1:]) != self.input_shape_:
            raise ValueError("TEDn prediction X must match fitted feature/image shape")
        data = torch.as_tensor(X, device="cpu")
        return self._score_data(data, np.arange(len(X)), next(self.model_.parameters()).device)

    def _predict(self, X):
        return (self._decision_function(X) >= 0).astype(int)

    def estimate_unlabeled_prior(self):
        """Last pre-update BBE alpha_U; not a fresh estimate of the final model."""
        self._check_is_fitted()
        if not self.bbe_history_:
            raise NotFittedError("No TEDn round has completed; warm-start has no alpha_U estimate")
        return self.alpha_u_
