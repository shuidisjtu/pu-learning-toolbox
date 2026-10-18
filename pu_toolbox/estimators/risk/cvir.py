"""Conditional Value Ignoring Risk (CVIR), a paper-objective adaptation.

Garg et al., NeurIPS 2021, Algorithm 2.  The supplied mixture proportion is
the positive share *inside the unlabeled pool*.  BBE/TEDn prior estimation,
the author image/text recipe and convergence stopping rule are not here.
Injected image encoders and CPU-streamed batches are engineering adapters.
"""

# ruff: noqa: N803, N806, N812

from __future__ import annotations

import copy
from numbers import Real

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
from ...core.validation import validate_pu_X_y


def select_cvir_negatives(scores: np.ndarray, unlabeled_positive_prior: float) -> np.ndarray:
    """Keep the lowest-scored ``1-alpha`` share of the original U rows.

    Stable ordering makes tied scores deterministic.  The count is rounded
    down (and clamped to one), an explicit finite-sample adaptation of the
    paper's fractional population definition.
    """
    scores = np.asarray(scores)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("scores must be a non-empty finite one-dimensional array")
    if not np.isfinite(unlabeled_positive_prior) or not 0 < unlabeled_positive_prior < 1:
        raise ValueError("unlabeled_positive_prior must be in (0, 1)")
    count = max(1, int(np.floor((1 - unlabeled_positive_prior) * len(scores))))
    return np.argsort(scores, kind="stable")[:count]


class CVIRClassifier(BasePUClassifier):
    """Fit CVIR given the positive fraction within the original U pool.

    ``class_prior`` is always rejected. A caller who establishes that U is a
    sample of the population marginal can pass that verified fraction through
    ``unlabeled_positive_prior`` explicitly. The frozen survey runner passes
    population pi, so this adapter is not in its matrix.
    """

    family = AlgorithmFamily.RISK_ESTIMATION
    label_semantics = "pu"
    assumption = (Assumption.SCAR,)
    scenario = (Scenario.CASE_CONTROL,)
    requires_class_prior = False  # population pi is not the required alpha_U
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.OFFICIAL_EXACT
    backend = Backend.TORCH
    maturity = Maturity.EXPERIMENTAL
    sample_weight_support = SampleWeightSupport.NOT_IMPLEMENTED
    native_architectures = frozenset({"mlp", "cnn"})
    input_ndims = frozenset({2, 4})
    encoder_parameter = "encoder"
    trains_encoder = True
    checkpoint_stages = ("warm_start", "cvir")

    def __init__(
        self,
        *,
        unlabeled_positive_prior: float | None = None,
        hidden_dim: int = 64,
        warm_start_epochs: int = 1,
        max_epochs: int = 20,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        random_state: int | None = None,
        device: str | None = None,
        encoder=None,
        data_device_policy: str = "auto",
        input_sampling: str = "os",
        prior_source: str | None = None,
    ) -> None:
        super().__init__()
        self.unlabeled_positive_prior = unlabeled_positive_prior
        self.hidden_dim = hidden_dim
        self.warm_start_epochs = warm_start_epochs
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.device = device
        self.encoder = encoder
        self.data_device_policy = data_device_policy
        self.input_sampling = input_sampling
        self.prior_source = prior_source

    @property
    def checkpoint_epoch_count(self):
        """Warm-start and CVIR both emit one inference snapshot per epoch."""
        return self.warm_start_epochs + self.max_epochs

    @property
    def checkpoint_prediction_batch_size(self):
        """New streaming snapshots retain chunk size; legacy replay stays 256."""
        return (
            self.batch_size
            if self.encoder is not None or self.data_device_policy == "cpu_stream"
            else 256
        )

    @invalidate_failed_fit
    def fit(
        self,
        X: np.ndarray,
        y_pu: np.ndarray,
        *,
        class_prior: float | None = None,
        sample_weight: np.ndarray | None = None,
        epoch_callback=None,
    ) -> CVIRClassifier:
        """Fit paper-weighted CVIR, not the balanced loss in released code.

        Four-dimensional images require a trainable encoder. Auto keeps legacy
        resident 2-D behavior or streams encoder input from CPU. alpha_U remains
        explicit; no population-prior conversion or new TS route is inferred.
        """
        self._is_fitted = False
        try:
            return self._fit_impl(
                X,
                y_pu,
                class_prior=class_prior,
                sample_weight=sample_weight,
                epoch_callback=epoch_callback,
            )
        except Exception:
            self._is_fitted = False
            raise

    def _fit_impl(self, X, y_pu, *, class_prior, sample_weight, epoch_callback):
        import torch

        if sample_weight is not None:
            raise NotImplementedError("CVIR does not implement sample_weight")
        if class_prior is not None:
            raise ValueError(
                "CVIR needs unlabeled_positive_prior=alpha_U; class_prior is the "
                "population prior and cannot be substituted without a verified marginal U pool"
            )
        alpha = self.unlabeled_positive_prior
        if (
            isinstance(alpha, (bool, np.bool_))
            or not isinstance(alpha, Real)
            or not np.isfinite(alpha)
            or not 0 < alpha < 1
        ):
            raise ValueError("unlabeled_positive_prior must be in (0, 1)")
        if self.input_sampling not in ("os", "ts"):
            raise ValueError("input_sampling must be os or ts; this is not TS-OS calibration")
        if self.prior_source is not None and (
            not isinstance(self.prior_source, str) or not self.prior_source.strip()
        ):
            raise ValueError("prior_source must be a non-empty description or None")
        if self.input_sampling == "ts" and self.prior_source is None:
            raise ValueError("native TS input requires an explicit alpha_U prior_source")
        for name in ("hidden_dim", "warm_start_epochs", "max_epochs", "batch_size"):
            value = getattr(self, name)
            lower = 0 if name == "warm_start_epochs" else 1
            if type(value) is not int or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not isinstance(self.data_device_policy, str) or self.data_device_policy not in {
            "auto",
            "resident",
            "cpu_stream",
        }:
            raise ValueError("data_device_policy must be auto, resident or cpu_stream")
        if self.encoder is not None and not isinstance(self.encoder, torch.nn.Module):
            raise TypeError("encoder must be a torch.nn.Module")
        if self.encoder is not None and self.data_device_policy == "resident":
            raise ValueError("injected encoders require CPU-streamed data, not resident")
        X, y_pu = validate_pu_X_y(
            X,
            y_pu,
            accept_sparse=False,
            allow_nd=self.encoder is not None,
            estimator_name="CVIRClassifier",
        )
        X = _finite_features(X, allow_nd=self.encoder is not None)
        p_idx = np.flatnonzero(y_pu == 1)
        u_idx = np.flatnonzero(y_pu == 0)
        if not len(u_idx):
            raise ValueError("CVIR needs original unlabeled rows")

        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(int(rng.randint(0, 2**31)))
        device = resolve_device(self.device)
        stream = self.encoder is not None or self.data_device_policy == "cpu_stream"
        self.data_device_policy_ = "cpu_stream" if stream else "resident"
        data_device = torch.device("cpu") if stream else device
        self.training_data_device_ = str(data_device)
        layers, width = [], X.shape[1]
        self.encoder_ = None
        if self.encoder is not None:
            from ..deep._validation import validate_encoder_features

            self.encoder_ = copy.deepcopy(self.encoder).to(device=device, dtype=torch.float32)
            self.encoder_.requires_grad_(True).eval()
            with torch.no_grad():
                features = self.encoder_(torch.as_tensor(X[:1], device=device))
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
        model = torch.nn.Sequential(*layers).to(device)
        data = torch.as_tensor(X, device=data_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.model_ = model
        self.n_features_in_ = X.shape[1]
        self.input_shape_ = tuple(X.shape[1:])
        self.n_positive_, self.n_unlabeled_ = len(p_idx), len(u_idx)
        self.unlabeled_positive_prior_ = float(alpha)
        # TS here means caller-supplied independent P/marginal U, not an OS
        # partition whose risk role has silently been replaced by P union U.
        self.training_provenance_ = {
            "method": "cvir",
            "input_sampling": self.input_sampling,
            "path": "native_ts_input" if self.input_sampling == "ts" else "explicit_os_adaptation",
            "calibration_applied": False,
            "known_prior": True,
            "unlabeled_positive_prior": float(alpha),
            "prior_semantics": (
                "distribution-level positive fraction in the supplied original U pool"
            ),
            "prior_source": self.prior_source or "caller_supplied_source_unverified",
            "source_verified": False,
            "empirical_unlabeled_positive_fraction": None,
            "n_positive": len(p_idx),
            "n_unlabeled_candidates": len(u_idx),
            "positive_rows_added_to_candidates": 0,
            "unlabeled_roles": ["warm_start", "ranking", "negative_retention", "weighted_loss"],
            "formal_admission": False,
        }
        self.history_ = {"warm_start_loss": [], "cvir_loss": [], "selected_negative_count": []}
        self.optimizer_steps_ = 0
        self.epoch_provenance_ = []
        self.selected_negative_indices_ = np.array([], dtype=int)
        self._is_fitted = False

        def train_epoch(negative_idx: np.ndarray, *, reweight: bool) -> float:
            positive_order = rng.permutation(p_idx)
            negative_order = rng.permutation(negative_idx)
            steps = max(
                int(np.ceil(len(positive_order) / self.batch_size)),
                int(np.ceil(len(negative_order) / self.batch_size)),
            )
            losses = []
            model.train()
            for step in range(steps):
                p = positive_order[(step * self.batch_size) % len(p_idx) :][: self.batch_size]
                n = negative_order[(step * self.batch_size) % len(negative_idx) :][
                    : self.batch_size
                ]
                loss = _cvir_objective(
                    _logits(model, data[p].to(device)),
                    _logits(model, data[n].to(device)),
                    alpha,
                    reweight=reweight,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("CVIR loss became non-finite")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.optimizer_steps_ += 1
                losses.append(float(loss.detach()))
            return float(np.mean(losses))

        def completed(stage, stage_epoch):
            self.checkpoint_stage_ = stage
            self.checkpoint_stage_epoch_ = stage_epoch
            self.checkpoint_round_ = stage_epoch if stage == "cvir" else None
            self.epoch_provenance_.append(
                dict(
                    stage=stage,
                    stage_epoch=stage_epoch,
                    round_index=self.checkpoint_round_,
                    optimizer_steps=self.optimizer_steps_,
                )
            )
            self._is_fitted = True
            if epoch_callback is not None:
                epoch_callback(len(self.epoch_provenance_) - 1, self)

        for epoch in range(self.warm_start_epochs):
            self.history_["warm_start_loss"].append(train_epoch(u_idx, reweight=False))
            completed("warm_start", epoch + 1)
        for epoch in range(self.max_epochs):
            model.eval()
            with torch.no_grad():
                if stream:
                    scores = np.concatenate(
                        [
                            _logits(model, data[u_idx[start : start + self.batch_size]].to(device))
                            .cpu()
                            .numpy()
                            for start in range(0, len(u_idx), self.batch_size)
                        ]
                    )
                else:
                    scores = _logits(model, data[u_idx]).cpu().numpy()
            local_negative = select_cvir_negatives(scores, float(alpha))
            negative_idx = u_idx[local_negative]
            self.selected_negative_indices_ = negative_idx.copy()
            self.history_["selected_negative_count"].append(len(negative_idx))
            self.history_["cvir_loss"].append(train_epoch(negative_idx, reweight=True))
            completed("cvir", epoch + 1)
        self.classes_ = np.array([0, 1])
        return self

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        import torch

        X = _finite_features(X, allow_nd=self.encoder is not None)
        if tuple(X.shape[1:]) != self.input_shape_:
            raise ValueError("CVIR prediction X must have fitted feature/image shape")
        self.model_.eval()
        with torch.no_grad():
            device = next(self.model_.parameters()).device
            if not len(X):
                return np.empty(0, dtype=np.float32)
            if self.data_device_policy_ == "cpu_stream":
                return np.concatenate(
                    [
                        _logits(
                            self.model_,
                            torch.as_tensor(X[start : start + self.batch_size], device=device),
                        )
                        .cpu()
                        .numpy()
                        for start in range(0, len(X), self.batch_size)
                    ]
                )
            return _logits(self.model_, torch.as_tensor(X, device=device)).cpu().numpy()

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return (self._decision_function(X) >= 0).astype(int)


def _cvir_objective(positive_logits, negative_logits, alpha, *, reweight):
    """Paper Algorithm 2 weighted BCE; legacy warm-start is the unscaled sum."""
    from torch.nn import functional as F

    positive_loss = F.softplus(-positive_logits).mean()
    negative_loss = F.softplus(negative_logits).mean()
    return (
        alpha * positive_loss + (1 - alpha) * negative_loss
        if reweight
        else positive_loss + negative_loss
    )


def _logits(model, batch):
    """Check per-row output without changing the differentiable scalar head."""
    import torch

    output = model(batch)
    if output.shape != (len(batch), 1):
        raise ValueError("CVIR network must preserve the batch dimension and output one logit")
    if not torch.isfinite(output).all():
        raise FloatingPointError("CVIR network produced non-finite logits")
    return output.reshape(-1)


def _finite_features(X, *, allow_nd):
    X = np.asarray(X)
    if X.ndim not in ({2, 4} if allow_nd else {2}) or any(size == 0 for size in X.shape[1:]):
        raise ValueError("CVIR requires 2-D features or 4-D images with an encoder")
    if not np.issubdtype(X.dtype, np.number) or not np.isfinite(X).all():
        raise ValueError("CVIR X must be finite numeric")
    X = np.asarray(X, dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("CVIR X must remain finite after float32 conversion")
    return X
