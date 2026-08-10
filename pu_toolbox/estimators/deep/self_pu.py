# ruff: noqa: N803, N806, N812

"""Clean-room Self-PU training with pacing, calibration, and distillation."""

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass
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
    Scenario,
    SourceStatus,
)
from ...core.validation import (
    check_scalar_in_range,
    validate_pu_X_y,
    validate_true_binary_labels,
)

__all__ = [
    "SelfPUClassifier",
    "TrustedSetManager",
    "calibrate_meta_weights",
    "dynamic_trust_target",
    "ema_update",
    "hard_distillation_loss",
]


def dynamic_trust_target(
    epoch: int,
    *,
    start_epoch: int,
    end_epoch: int,
    final_size: int,
) -> int:
    """Return the linearly scheduled trusted-set size for one epoch."""
    if epoch < start_epoch:
        return 0
    if epoch >= end_epoch:
        return int(final_size)
    progress = (epoch - start_epoch) / (end_epoch - start_epoch)
    return int(math.floor(final_size * progress))


@dataclass(frozen=True)
class TrustedSetUpdate:
    """Summary of one dynamic trusted-set refresh."""

    target_size: int
    actual_size: int
    positive_count: int
    negative_count: int
    entered_count: int
    exited_count: int


class TrustedSetManager:
    """Maintain a balanced, fully refreshed trusted subset of U samples."""

    def __init__(self, n_unlabeled: int) -> None:
        if n_unlabeled < 2:
            raise ValueError("n_unlabeled must be at least 2.")
        self.n_unlabeled = int(n_unlabeled)
        self.indices = np.empty(0, dtype=int)
        self.soft_labels = np.empty(0, dtype=float)
        self.directions = np.empty(0, dtype=int)

    def update(self, probabilities: np.ndarray, target_size: int) -> TrustedSetUpdate:
        """Re-rank all U samples and replace the trusted set in-and-out."""
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != (self.n_unlabeled,):
            raise ValueError(
                f"probabilities must have shape ({self.n_unlabeled},); got {probabilities.shape}."
            )
        if not np.isfinite(probabilities).all() or np.any(
            (probabilities < 0.0) | (probabilities > 1.0)
        ):
            raise ValueError("probabilities must be finite and lie in [0, 1].")
        if target_size < 0:
            raise ValueError("target_size must be non-negative.")

        pair_count = min(int(target_size) // 2, self.n_unlabeled // 2)
        order = np.argsort(probabilities, kind="stable")
        negative = order[:pair_count]
        positive = order[-pair_count:] if pair_count else np.empty(0, dtype=int)
        new_indices = np.concatenate([negative, positive]).astype(int, copy=False)
        new_directions = np.concatenate(
            [np.zeros(pair_count, dtype=int), np.ones(pair_count, dtype=int)]
        )

        previous = set(self.indices.tolist())
        current = set(new_indices.tolist())
        self.indices = new_indices
        self.soft_labels = probabilities[new_indices].copy()
        self.directions = new_directions
        return TrustedSetUpdate(
            target_size=int(target_size),
            actual_size=len(new_indices),
            positive_count=pair_count,
            negative_count=pair_count,
            entered_count=len(current - previous),
            exited_count=len(previous - current),
        )

    def targets_for(self, indices: np.ndarray) -> np.ndarray:
        """Return stored soft targets for trusted U-local indices."""
        lookup = dict(zip(self.indices.tolist(), self.soft_labels.tolist(), strict=True))
        try:
            return np.asarray([lookup[int(index)] for index in indices], dtype=float)
        except KeyError as exc:  # pragma: no cover - internal invariant guard
            raise ValueError("Requested index is not in the trusted set.") from exc


def calibrate_meta_weights(
    influences: np.ndarray,
    *,
    gamma: float,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Convert two-column meta influences into stable non-negative weights.

    Column 0 is soft-label CE and column 1 is the U-negative PU term.
    Each column is normalized independently. ``gamma`` limits the fraction
    of samples with active CE weight and its total mass when ``gamma*n < 1``.
    """
    influences = np.asarray(influences, dtype=float)
    if influences.ndim != 2 or influences.shape[1] != 2 or len(influences) == 0:
        raise ValueError("influences must have shape (n_samples, 2) with n_samples > 0.")
    if not np.isfinite(influences).all():
        raise ValueError("influences must contain only finite values.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")

    positive = np.maximum(influences, 0.0)
    weights = np.zeros_like(positive)
    fallback = [False, False]
    for column in range(2):
        total = float(positive[:, column].sum())
        if total <= np.finfo(float).eps:
            weights[:, column] = 1.0 / len(weights)
            fallback[column] = True
        else:
            weights[:, column] = positive[:, column] / total

    if gamma == 0.0:
        weights[:, 0] = 0.0
    else:
        active_count = max(1, int(math.floor(gamma * len(weights))))
        keep = np.argsort(weights[:, 0], kind="stable")[-active_count:]
        ce_weights = np.zeros(len(weights), dtype=float)
        ce_weights[keep] = weights[keep, 0]
        ce_total = ce_weights.sum()
        if ce_total > 0:
            ce_weights /= ce_total
        ce_weights *= min(1.0, gamma * len(weights))
        weights[:, 0] = ce_weights

    statistics: dict[str, float | bool] = {
        "ce_fallback": fallback[0],
        "pu_fallback": fallback[1],
        "ce_active_fraction": float(np.mean(weights[:, 0] > 0.0)),
        "ce_weight_sum": float(weights[:, 0].sum()),
        "pu_weight_sum": float(weights[:, 1].sum()),
        "zero_weight_fraction": float(np.mean(weights.sum(axis=1) == 0.0)),
    }
    return weights, statistics


def ema_update(teacher: Any, student: Any, decay: float) -> None:
    """Update teacher parameters and floating buffers from a student."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in [0, 1).")
    import torch

    with torch.no_grad():
        for teacher_parameter, student_parameter in zip(
            teacher.parameters(), student.parameters(), strict=True
        ):
            teacher_parameter.mul_(decay).add_(student_parameter, alpha=1.0 - decay)
        for teacher_buffer, student_buffer in zip(
            teacher.buffers(), student.buffers(), strict=True
        ):
            if teacher_buffer.is_floating_point():
                teacher_buffer.mul_(decay).add_(student_buffer, alpha=1.0 - decay)
            else:
                teacher_buffer.copy_(student_buffer)


def hard_distillation_loss(
    student_probabilities: Any,
    peer_probabilities: Any,
    pu_losses: Any,
    *,
    alpha: float,
) -> tuple[Any, Any]:
    """Return hard-mined student consistency loss and active fraction."""
    import torch

    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    if not (student_probabilities.shape == peer_probabilities.shape == pu_losses.shape):
        raise ValueError("student_probabilities, peer_probabilities, and pu_losses must align.")
    mse = (student_probabilities - peer_probabilities).square()
    active = pu_losses > alpha * mse
    if active.any():
        loss = mse[active].mean()
    else:
        loss = torch.zeros((), device=mse.device, dtype=mse.dtype)
    return loss, active.float().mean()


def _as_logits(output: Any) -> Any:
    if output.ndim == 1:
        return output
    if output.ndim == 2 and output.shape[1] == 1:
        return output[:, 0]
    raise ValueError("backbone must return shape (n_samples,) or (n_samples, 1).")


class SelfPUClassifier(BasePUClassifier):
    """Self-PU classifier with two paced students and EMA teachers.

    Clean validation labels enable the paper's self-calibrated meta weights
    and final teacher selection. Without them, fitting emits a warning and
    runs the explicit self-paced + distillation ablation.
    """

    family = AlgorithmFamily.DEEP_PU
    assumption = (Assumption.SCAR,)
    scenario = (Scenario.CASE_CONTROL,)
    requires_class_prior = True
    implementation_status = ImplementationStatus.NATIVE
    source_status = SourceStatus.OFFICIAL_EXACT
    backend = Backend.TORCH
    maturity = Maturity.RESEARCH

    def __init__(
        self,
        class_prior: float,
        *,
        backbone: Any | None = None,
        hidden_dim: int = 128,
        warmup_epochs: int = 10,
        self_paced_start: int = 10,
        self_paced_end: int = 50,
        distill_start: int = 50,
        max_epochs: int = 200,
        max_trust_ratio: float = 0.25,
        pace_1: float = 0.20,
        pace_2: float = 0.30,
        meta_step_size: float = 1e-3,
        reweight_gamma: float = 1 / 16,
        distillation_alpha: float = 10.0,
        ema_decay: float = 0.99,
        student_loss_weight: float = 1.0,
        teacher_loss_weight: float = 1.0,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        threshold: float = 0.5,
        require_validation: bool = False,
        random_state: int | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.class_prior = class_prior
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.warmup_epochs = warmup_epochs
        self.self_paced_start = self_paced_start
        self.self_paced_end = self_paced_end
        self.distill_start = distill_start
        self.max_epochs = max_epochs
        self.max_trust_ratio = max_trust_ratio
        self.pace_1 = pace_1
        self.pace_2 = pace_2
        self.meta_step_size = meta_step_size
        self.reweight_gamma = reweight_gamma
        self.distillation_alpha = distillation_alpha
        self.ema_decay = ema_decay
        self.student_loss_weight = student_loss_weight
        self.teacher_loss_weight = teacher_loss_weight
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.threshold = threshold
        self.require_validation = require_validation
        self.random_state = random_state
        self.device = device

    def _validate_parameters(self, class_prior: float) -> None:
        check_scalar_in_range(class_prior, 0.0, 1.0, "class_prior", inclusive=False)
        for name, value in (
            ("hidden_dim", self.hidden_dim),
            ("max_epochs", self.max_epochs),
            ("batch_size", self.batch_size),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1.")
        if not (
            0
            <= self.warmup_epochs
            <= self.self_paced_start
            <= self.self_paced_end
            <= self.max_epochs
        ):
            raise ValueError(
                "Require 0 <= warmup_epochs <= self_paced_start <= self_paced_end <= max_epochs."
            )
        if not self.self_paced_start <= self.distill_start <= self.max_epochs:
            raise ValueError("distill_start must lie between self_paced_start and max_epochs.")
        for name, value in (
            ("max_trust_ratio", self.max_trust_ratio),
            ("pace_1", self.pace_1),
            ("pace_2", self.pace_2),
        ):
            if not 0.0 <= value <= 0.5:
                raise ValueError(f"{name} must be in [0, 0.5].")
        for name, value in (
            ("reweight_gamma", self.reweight_gamma),
            ("ema_decay", self.ema_decay),
        ):
            upper_ok = value <= 1.0 if name == "reweight_gamma" else value < 1.0
            if value < 0.0 or not upper_ok:
                interval = "[0, 1]" if name == "reweight_gamma" else "[0, 1)"
                raise ValueError(f"{name} must be in {interval}.")
        if (
            self.meta_step_size <= 0
            or self.learning_rate <= 0
            or self.weight_decay < 0
            or self.distillation_alpha < 0
            or self.student_loss_weight < 0
            or self.teacher_loss_weight < 0
        ):
            raise ValueError("Learning, loss, and regularization parameters are out of range.")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be in (0, 1).")

    @staticmethod
    def _validate_clean_validation(validation_data: Any, input_shape: tuple[int, ...]):
        if not isinstance(validation_data, tuple) or len(validation_data) != 2:
            raise ValueError("validation_data must be a tuple (X_val, y_val).")
        X_val = np.asarray(validation_data[0], dtype=np.float32)
        y_val = np.asarray(validation_data[1])
        if X_val.ndim < 2 or X_val.shape[1:] != input_shape:
            raise ValueError(f"X_val must have sample shape {input_shape}.")
        if y_val.ndim != 1 or len(y_val) != len(X_val):
            raise ValueError("y_val must be 1-D and align with X_val.")
        if not np.isfinite(X_val).all():
            raise ValueError("X_val contains NaN or Inf values.")
        unique = set(np.unique(y_val))
        if unique == {-1, 1}:
            y_val = (y_val == 1).astype(np.float32)
        else:
            validate_true_binary_labels(y_val, estimator_name="y_val")
            y_val = y_val.astype(np.float32)
        if len(np.unique(y_val)) < 2:
            raise ValueError("y_val must contain both classes encoded as {-1, 1} or {0, 1}.")
        return X_val, y_val

    def _make_model(self, input_shape: tuple[int, ...], device: Any):
        from torch import nn

        if self.backbone is not None:
            model = copy.deepcopy(self.backbone)
        else:
            model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(int(np.prod(input_shape)), self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
            )
        return model.to(device)

    def _meta_weights(
        self,
        model: Any,
        positive_loss: Any,
        ce_losses: Any,
        pu_losses: Any,
        X_val: Any,
        y_val: Any,
    ) -> tuple[Any, dict[str, float | bool]]:
        import torch
        from torch.func import functional_call
        from torch.nn import functional as F

        meta = torch.zeros(
            (len(ce_losses), 2),
            dtype=ce_losses.dtype,
            device=ce_losses.device,
            requires_grad=True,
        )
        objective = positive_loss + (meta[:, 0] * ce_losses + meta[:, 1] * pu_losses).sum()
        named_parameters = dict(model.named_parameters())
        gradients = torch.autograd.grad(
            objective,
            tuple(named_parameters.values()),
            create_graph=True,
        )
        updated = {
            name: parameter - self.meta_step_size * gradient
            for (name, parameter), gradient in zip(named_parameters.items(), gradients, strict=True)
        }
        validation_logits = _as_logits(functional_call(model, updated, (X_val,)))
        validation_loss = F.binary_cross_entropy_with_logits(validation_logits, y_val)
        meta_gradient = torch.autograd.grad(validation_loss, meta)[0]
        weights_np, statistics = calibrate_meta_weights(
            -meta_gradient.detach().cpu().numpy(),
            gamma=self.reweight_gamma,
        )
        return (
            torch.as_tensor(weights_np, dtype=ce_losses.dtype, device=ce_losses.device),
            statistics,
        )

    def fit(
        self,
        X: np.ndarray,
        y_pu: np.ndarray,
        *,
        class_prior: float | None = None,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> SelfPUClassifier:
        """Fit Self-PU; clean validation enables calibration and teacher choice.

        sample_weight : NotImplementedError (deep estimators do not accept instance weights)
        """
        try:
            import torch
            from torch.nn import functional as F
        except ImportError as exc:
            raise ImportError("SelfPUClassifier requires optional dependency 'torch'.") from exc

        X, y_pu = validate_pu_X_y(
            X,
            y_pu,
            accept_sparse=False,
            allow_nd=True,
            estimator_name="SelfPUClassifier",
        )
        X = np.asarray(X, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf values.")
        resolved_prior = self.class_prior if class_prior is None else class_prior
        self._validate_parameters(resolved_prior)
        if sample_weight is not None:
            raise NotImplementedError(
                "SelfPUClassifier does not yet support sample_weight across all loss terms."
            )

        input_shape = tuple(X.shape[1:])
        clean_validation = None
        if validation_data is not None:
            clean_validation = self._validate_clean_validation(validation_data, input_shape)
        elif self.require_validation:
            raise ValueError("validation_data is required when require_validation=True.")
        else:
            warnings.warn(
                "validation_data was not supplied; running the explicit Self-PU "
                "ablation without meta reweighting or validation-based teacher selection.",
                UserWarning,
                stacklevel=2,
            )

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        rng = np.random.RandomState(self.random_state)
        device = resolve_device(self.device)
        tx = torch.as_tensor(X, dtype=torch.float32, device=device)
        positive_global = np.flatnonzero(y_pu == 1)
        unlabeled_global = np.flatnonzero(y_pu == 0)
        if len(unlabeled_global) < 2:
            raise ValueError("SelfPUClassifier requires at least two unlabeled samples.")

        if clean_validation is not None:
            X_val_np, y_val_np = clean_validation
            tx_val = torch.as_tensor(X_val_np, dtype=torch.float32, device=device)
            ty_val = torch.as_tensor(y_val_np, dtype=torch.float32, device=device)
        else:
            tx_val = ty_val = None

        base_model = self._make_model(input_shape, device)
        with torch.no_grad():
            _as_logits(base_model(tx[:1]))
        self.student_1_ = copy.deepcopy(base_model)
        self.student_2_ = copy.deepcopy(base_model)
        self.teacher_1_ = copy.deepcopy(base_model)
        self.teacher_2_ = copy.deepcopy(base_model)
        for teacher in (self.teacher_1_, self.teacher_2_):
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)

        optimizers = [
            torch.optim.Adam(
                student.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            for student in (self.student_1_, self.student_2_)
        ]
        schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
            for optimizer in optimizers
        ]
        managers = [TrustedSetManager(len(unlabeled_global)) for _ in range(2)]
        pace_ratios = [
            min(self.pace_1, self.max_trust_ratio),
            min(self.pace_2, self.max_trust_ratio),
        ]

        self.trusted_history_: list[dict[str, Any]] = []
        self.reweight_history_: list[dict[str, Any]] = []
        self.distillation_history_: list[dict[str, Any]] = []
        self.training_history_: list[dict[str, Any]] = []

        students = [self.student_1_, self.student_2_]
        teachers = [self.teacher_1_, self.teacher_2_]
        for epoch_index in range(self.max_epochs):
            epoch = epoch_index + 1
            if epoch <= self.warmup_epochs:
                stage = "warmup"
            elif epoch <= self.self_paced_start:
                stage = "base_nnpu"
            elif epoch < self.distill_start:
                stage = "self_paced"
            else:
                stage = "distillation"
            for student_id, (student, manager, pace_ratio) in enumerate(
                zip(students, managers, pace_ratios, strict=True), start=1
            ):
                student.eval()
                with torch.no_grad():
                    probabilities = (
                        torch.sigmoid(_as_logits(student(tx[unlabeled_global]))).cpu().numpy()
                    )
                final_size = int(math.floor(pace_ratio * len(unlabeled_global)))
                target = dynamic_trust_target(
                    epoch,
                    start_epoch=self.self_paced_start,
                    end_epoch=self.self_paced_end,
                    final_size=final_size,
                )
                update = manager.update(probabilities, target)
                self.trusted_history_.append(
                    {"epoch": epoch, "student": student_id, **update.__dict__}
                )

            positive_batch = rng.choice(
                positive_global,
                size=min(self.batch_size, len(positive_global)),
                replace=False,
            )
            unlabeled_local_batch = rng.choice(
                len(unlabeled_global),
                size=min(self.batch_size, len(unlabeled_global)),
                replace=False,
            )
            unlabeled_batch = unlabeled_global[unlabeled_local_batch]
            x_positive = tx[positive_batch]
            x_unlabeled = tx[unlabeled_batch]

            with torch.no_grad():
                cached_student_probabilities = [
                    torch.sigmoid(_as_logits(student(x_unlabeled))) for student in students
                ]
                cached_teacher_probabilities = [
                    torch.sigmoid(_as_logits(teacher(x_unlabeled))) for teacher in teachers
                ]

            for index, (student, teacher, manager, optimizer) in enumerate(
                zip(
                    students,
                    teachers,
                    managers,
                    optimizers,
                    strict=True,
                )
            ):
                student.train()
                positive_logits = _as_logits(student(x_positive))
                unlabeled_logits = _as_logits(student(x_unlabeled))
                R_p_plus = torch.sigmoid(-positive_logits).mean()
                R_p_minus = torch.sigmoid(positive_logits).mean()

                trusted_mask_np = np.isin(unlabeled_local_batch, manager.indices)
                trusted_mask = torch.as_tensor(trusted_mask_np, device=device)
                untrusted_mask = ~trusted_mask
                if not untrusted_mask.any():  # pragma: no cover - ratio validation prevents this
                    raise RuntimeError("No untrusted U samples remain in the training batch.")

                trusted_loss = torch.zeros((), device=device)
                if trusted_mask.any():
                    trusted_local = unlabeled_local_batch[trusted_mask_np]
                    trusted_targets = torch.as_tensor(
                        manager.targets_for(trusted_local),
                        dtype=torch.float32,
                        device=device,
                    )
                    trusted_loss = F.binary_cross_entropy_with_logits(
                        unlabeled_logits[trusted_mask], trusted_targets
                    )

                untrusted_logits = unlabeled_logits[untrusted_mask]
                pu_losses = torch.sigmoid(untrusted_logits)
                soft_targets = cached_teacher_probabilities[index][untrusted_mask]
                ce_losses = F.binary_cross_entropy_with_logits(
                    untrusted_logits,
                    soft_targets,
                    reduction="none",
                )
                positive_risk = resolved_prior * R_p_plus

                calibration_active = tx_val is not None and epoch > self.self_paced_start
                if calibration_active:
                    weights, weight_stats = self._meta_weights(
                        student,
                        positive_risk,
                        ce_losses,
                        pu_losses,
                        tx_val,
                        ty_val,
                    )
                else:
                    weights = torch.zeros((len(pu_losses), 2), dtype=pu_losses.dtype, device=device)
                    weights[:, 1] = 1.0 / len(pu_losses)
                    weight_stats = {
                        "ce_fallback": False,
                        "pu_fallback": False,
                        "ce_active_fraction": 0.0,
                        "ce_weight_sum": 0.0,
                        "pu_weight_sum": 1.0,
                        "zero_weight_fraction": 0.0,
                    }

                unlabeled_negative = (weights[:, 1] * pu_losses).sum()
                negative_risk = unlabeled_negative - resolved_prior * R_p_minus
                nnpu_loss = positive_risk + torch.clamp(negative_risk, min=0.0)
                calibrated_ce = (weights[:, 0] * ce_losses).sum()
                total_loss = nnpu_loss + trusted_loss + calibrated_ce

                student_consistency = torch.zeros((), device=device)
                teacher_consistency = torch.zeros((), device=device)
                active_fraction = torch.zeros((), device=device)
                if epoch >= self.distill_start:
                    current_probability = torch.sigmoid(untrusted_logits)
                    peer_probability = cached_student_probabilities[1 - index][untrusted_mask]
                    student_consistency, active_fraction = hard_distillation_loss(
                        current_probability,
                        peer_probability,
                        pu_losses,
                        alpha=self.distillation_alpha,
                    )
                    teacher_consistency = F.mse_loss(
                        torch.sigmoid(unlabeled_logits),
                        cached_teacher_probabilities[index],
                    )
                    total_loss = (
                        total_loss
                        + self.student_loss_weight * student_consistency
                        + self.teacher_loss_weight * teacher_consistency
                    )

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                ema_update(teacher, student, self.ema_decay)

                record = {
                    "epoch": epoch,
                    "student": index + 1,
                    "stage": stage,
                    "nnpu_loss": float(nnpu_loss.detach().cpu()),
                    "trusted_ce": float(trusted_loss.detach().cpu()),
                    "calibrated_ce": float(calibrated_ce.detach().cpu()),
                    "student_consistency": float(student_consistency.detach().cpu()),
                    "teacher_consistency": float(teacher_consistency.detach().cpu()),
                    "total_loss": float(total_loss.detach().cpu()),
                    "negative_risk": float(negative_risk.detach().cpu()),
                }
                self.training_history_.append(record)
                self.reweight_history_.append(
                    {
                        "epoch": epoch,
                        "student": index + 1,
                        "calibration_active": calibration_active,
                        **weight_stats,
                    }
                )
                self.distillation_history_.append(
                    {
                        "epoch": epoch,
                        "student": index + 1,
                        "active": epoch >= self.distill_start,
                        "hard_sample_fraction": float(active_fraction.detach().cpu()),
                        "student_mse": float(student_consistency.detach().cpu()),
                        "teacher_mse": float(teacher_consistency.detach().cpu()),
                    }
                )
            for scheduler in schedulers:
                scheduler.step()

        self.optimizer_states_ = [copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers]
        self.scheduler_states_ = [copy.deepcopy(scheduler.state_dict()) for scheduler in schedulers]
        self.trusted_indices_ = {}
        for student_id, manager in enumerate(managers, start=1):
            self.trusted_indices_[student_id] = {
                "indices": unlabeled_global[manager.indices].copy(),
                "soft_labels": manager.soft_labels.copy(),
                "directions": manager.directions.copy(),
            }

        teacher_metrics: list[float] = []
        if tx_val is not None:
            with torch.no_grad():
                for teacher in teachers:
                    prediction = (
                        torch.sigmoid(_as_logits(teacher(tx_val))) >= self.threshold
                    ).float()
                    teacher_metrics.append(float((prediction == ty_val).float().mean().cpu()))
            self.teacher_selection_basis_ = "clean_validation_accuracy"
            best_index = int(np.argmax(teacher_metrics))
        else:
            with torch.no_grad():
                for teacher in teachers:
                    p_scores = _as_logits(teacher(tx[positive_global]))
                    u_scores = _as_logits(teacher(tx[unlabeled_global]))
                    p_plus = torch.sigmoid(-p_scores).mean()
                    p_minus = torch.sigmoid(p_scores).mean()
                    u_minus = torch.sigmoid(u_scores).mean()
                    risk = resolved_prior * p_plus + torch.clamp(
                        u_minus - resolved_prior * p_minus, min=0.0
                    )
                    teacher_metrics.append(float(risk.cpu()))
            self.teacher_selection_basis_ = "training_nnpu_risk_ablation"
            best_index = int(np.argmin(teacher_metrics))

        self.teacher_selection_metrics_ = teacher_metrics
        self.best_teacher_index_ = best_index + 1
        self.best_teacher_ = teachers[best_index]
        self.best_teacher_.eval()
        self.class_prior_ = float(resolved_prior)
        self.classes_ = np.array([0, 1])
        self.input_shape_ = input_shape
        self.n_features_in_ = X.shape[1] if X.ndim == 2 else int(np.prod(input_shape))
        self.device_ = device
        self.calibration_mode_ = "clean_validation_meta" if tx_val is not None else "ablation"
        self._class_prior = self.class_prior_
        self._X_shape_ = X.shape
        self._is_fitted = True
        return self

    def _logits(self, X: np.ndarray) -> np.ndarray:
        import torch

        X = np.asarray(X, dtype=np.float32)
        if X.ndim < 2 or X.shape[1:] != self.input_shape_:
            raise ValueError(f"X must have sample shape {self.input_shape_}.")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf values.")
        self.best_teacher_.eval()
        with torch.no_grad():
            logits = _as_logits(
                self.best_teacher_(torch.as_tensor(X, dtype=torch.float32, device=self.device_))
            )
        return logits.cpu().numpy()

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._logits(X)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self._logits(X)
        positive = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        return np.column_stack([1.0 - positive, positive])

    def get_pu_metadata(self) -> dict[str, Any]:
        """Return base metadata plus Self-PU stage and selection diagnostics."""
        metadata = super().get_pu_metadata()
        if self._is_fitted:
            metadata.update(
                {
                    "calibration_mode": self.calibration_mode_,
                    "best_teacher_index": self.best_teacher_index_,
                    "teacher_selection_basis": self.teacher_selection_basis_,
                    "teacher_selection_metrics": list(self.teacher_selection_metrics_),
                    "final_trusted_sizes": {
                        str(key): len(value["indices"])
                        for key, value in self.trusted_indices_.items()
                    },
                }
            )
        return metadata

    def get_training_checkpoint(self) -> dict[str, Any]:
        """Return all state required to audit or resume Self-PU training."""
        self._check_is_fitted()
        return {
            "schema_version": "1.0",
            "estimator_params": copy.deepcopy(self.get_params(deep=False)),
            "class_prior": self.class_prior_,
            "input_shape": self.input_shape_,
            "best_teacher_index": self.best_teacher_index_,
            "calibration_mode": self.calibration_mode_,
            "student_states": [
                copy.deepcopy(self.student_1_.state_dict()),
                copy.deepcopy(self.student_2_.state_dict()),
            ],
            "teacher_states": [
                copy.deepcopy(self.teacher_1_.state_dict()),
                copy.deepcopy(self.teacher_2_.state_dict()),
            ],
            "optimizer_states": copy.deepcopy(self.optimizer_states_),
            "scheduler_states": copy.deepcopy(self.scheduler_states_),
            "trusted_sets": copy.deepcopy(self.trusted_indices_),
            "histories": {
                "trusted": copy.deepcopy(self.trusted_history_),
                "reweight": copy.deepcopy(self.reweight_history_),
                "distillation": copy.deepcopy(self.distillation_history_),
                "training": copy.deepcopy(self.training_history_),
            },
        }
