# ruff: noqa: N803, N806, E501

"""Dist-PU label-distribution alignment classifier.

The implementation follows the paper's three train-time signals: positive
supervision, unlabeled expectation alignment, and entropy minimisation.  A
small Mixup term is included when ``mixup_weight`` is non-zero.  PyTorch is an
optional dependency and is imported only when ``fit`` is called.
"""

from __future__ import annotations

from typing import Any

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


class DistPUClassifier(BasePUClassifier):
    """Train a small MLP using Dist-PU's label-distribution objective."""

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
    native_architectures = frozenset({"mlp"})
    input_ndims = frozenset({2})
    encoder_parameter = None
    trains_encoder = False

    def __init__(
        self,
        class_prior: float,
        *,
        hidden_dim: int = 64,
        epochs: int = 100,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        alignment_weight: float = 1.0,
        entropy_weight: float = 0.05,
        mixup_weight: float = 0.1,
        random_state: int | None = 0,
        device: str | None = None,
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
        self.random_state = random_state
        self.device = device

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
        X : array-like of shape (n_samples, n_features)
            Feature matrix.  Must be dense.
        y_pu : array-like of shape (n_samples,)
            PU labels.  +1 = labeled positive, 0 = unlabeled.
        class_prior : float, optional
            Override the constructor's ``class_prior``.  Must be in (0, 1).
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
            Called as ``epoch_callback(epoch, self)`` after each optimizer step.

        Returns
        -------
        self : DistPUClassifier
        """
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("DistPUClassifier requires the optional 'torch' dependency") from exc
        X, y_pu = validate_pu_X_y(X, y_pu, accept_sparse=False, estimator_name="DistPUClassifier")
        # Before any RNG use: an invalid view must not perturb the torch seed.
        if os_or_ts not in RUN_VIEWS:
            raise ValueError(f"os_or_ts must be 'os' or 'ts'; got {os_or_ts!r}.")
        pi = self.class_prior if class_prior is None else class_prior
        check_scalar_in_range(pi, 0.0, 1.0, "class_prior", inclusive=False)
        if self.epochs < 1 or self.hidden_dim < 1:
            raise ValueError("epochs must be >= 1 and hidden_dim must be >= 1")
        X = np.asarray(X, dtype=np.float32)
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        device = resolve_device(self.device)
        self.model_ = nn.Sequential(
            nn.Linear(X.shape[1], self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, 1)
        ).to(device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        tx = torch.as_tensor(X, device=device)
        ty = torch.as_tensor(y_pu, device=device)
        p_mask, u_mask = ty == 1, ty == 0
        # ── TS-OS calibration (protocol §2.3) ─────────────────────────────
        # The unlabeled role is carried by the unlabeled rows under the OS view
        # and by the whole training matrix under the calibrated (ts) view: the
        # labeled positives join D_U, exactly as case-control sampling puts them
        # back.  Only that role set changes; the positive term, the class prior,
        # the Mixup pool and the RNG stream are untouched.
        calibrated_view = os_or_ts == "ts"
        self.loss_history_ = []
        bce = nn.BCEWithLogitsLoss()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            logits = self.model_(tx).squeeze(1).clamp(-10, 10)
            probs = torch.sigmoid(logits)
            positive_loss = bce(logits[p_mask], torch.ones_like(logits[p_mask]))
            alignment, entropy = _distribution_regularizers(
                probs, u_mask, pi, include_positive_in_unlabeled=calibrated_view
            )
            loss = positive_loss + self.alignment_weight * alignment + self.entropy_weight * entropy
            if self.mixup_weight > 0 and len(X) > 1:
                perm = torch.randperm(len(X), device=device)
                lam = torch.rand((), device=device)
                mix_x = lam * tx + (1 - lam) * tx[perm]
                mix_y = lam * probs.detach() + (1 - lam) * probs.detach()[perm]
                loss = loss + self.mixup_weight * bce(self.model_(mix_x).squeeze(1), mix_y)
            loss.backward()
            optimizer.step()
            self.loss_history_.append(float(loss.detach().cpu()))
            if epoch_callback is not None:
                epoch_callback(epoch, self)
        self.classes_ = np.array([0, 1])
        self._class_prior, self._X_shape_, self._is_fitted = pi, X.shape, True
        self.device_ = device
        return self

    def _decision_function(self, X):
        import torch

        with torch.no_grad():
            return (
                self.model_(torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device_))
                .squeeze(1)
                .cpu()
                .numpy()
            )

    def _predict(self, X):
        return (self._decision_function(X) >= 0).astype(int)

    def predict_proba(self, X):
        score = 1.0 / (1.0 + np.exp(-np.clip(self._decision_function(X), -40, 40)))
        return np.column_stack([1.0 - score, score])
