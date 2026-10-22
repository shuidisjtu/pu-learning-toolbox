# ruff: noqa: N803
"""Explicit clean-support checkpoint selection, separate from frozen PU runs.

Reuse the audited support adapters and weights-only writer. A private fit
binding supplies only support; external selection/test data never enters fit.
Completed meta-active epochs are candidates, not warmup or terminal fallbacks.
"""

from pathlib import Path

from sklearn.base import clone

from .checkpoints import EpochCheckpointTrainer
from .clean_support_adapters import LaGAMCleanSupportTrainer, SelfPUCleanSupportTrainer
from .strategies import ProtocolOA, ProtocolPACleanSupport


class _SupportFitBinding:
    """Private writer interface; never registered, returned or persisted as a model."""

    def __init__(self, estimator, adapter):
        self.estimator = estimator
        self.adapter = adapter

    def __getattr__(self, name):
        return getattr(self.estimator, name)

    def fit(self, X, y, *, class_prior=None, epoch_callback=None, os_or_ts=None):
        self.adapter._require_prepared(self.estimator, X, os_or_ts)

        def completed_meta_epoch(epoch, fitted):
            active = (
                epoch > fitted.self_paced_start
                if self.adapter.method_name == "self_pu"
                else epoch >= fitted.warmup_epochs
            )
            if active:
                epoch_callback(epoch, fitted)

        self.adapter._fit_support_estimator(
            self.estimator, X, y, class_prior=class_prior, epoch_callback=completed_meta_epoch
        )
        return self


class _SupportCheckpointSelection:
    """Add explicit checkpoint recording without relaxing the terminal adapters."""

    records_epoch_checkpoints = True

    def _validate_selection_protocols(self, protocols):
        if (
            not protocols
            or any(type(p) not in (ProtocolOA, ProtocolPACleanSupport) for p in protocols)
            or len({p.name for p in protocols}) != len(protocols)
        ):
            raise ValueError("support checkpoints require unique OA / PA+clean-support protocols")

    def validate_experiment(self, model, bundle, config, protocols):
        # Revoke all earlier preparation even if a new checkpoint option fails.
        self._prepared = False
        self._checkpoint_configured = False
        if config.get("capture_epoch_checkpoints", True) is not True:
            raise ValueError("support checkpoint selection cannot disable checkpoint capture")
        size = config.get("checkpoint_bytes_per_component")
        if type(size) is not int or size < 1:
            raise ValueError("support checkpoints need positive checkpoint_bytes_per_component")
        if self.method_name == "self_pu":
            model.set_params(require_validation=True, distillation_scope="paper_population")
        candidates = config.get("candidates", [{}])
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("support checkpoints need a nonempty explicit candidate list")
        epoch_bounds = []
        for parameters in candidates:
            if not isinstance(parameters, dict):
                raise ValueError("support checkpoint candidate parameters must be objects")
            if self.method_name == "self_pu" and (
                parameters.get("require_validation", True) is not True
                or parameters.get("distillation_scope", "paper_population") != "paper_population"
            ):
                raise ValueError("full-meta candidate cannot change support or population scope")
            candidate = clone(model).set_params(**parameters)
            if self.method_name == "self_pu":
                start = candidate.self_paced_start
            else:
                start = candidate.warmup_epochs
            if (
                type(candidate.max_epochs) is not int
                or type(start) is not int
                or not 0 <= start < candidate.max_epochs
            ):
                raise ValueError("every support candidate must reach a meta-active epoch")
            epoch_bounds.append(candidate.max_epochs)
        super().validate_experiment(model, bundle, config, protocols)
        self._checkpoint_epoch_bound = max(epoch_bounds)
        self._checkpoint_bytes_per_component = size
        self._audit.update(
            {
                "pa_clean_support_eligible": True,
                "pa_eligible": False,
                "selection_protocols": [p.name for p in protocols],
                "selection_scope": "completed_meta_epochs",
                "warmup_snapshots_eligible": False,
                "formal_admission": False,
                "budget_approval": "not_granted_by_adapter",
                "protocol_policy_basis": "survey-v2.0",
                "protocol_pending_review_items": ["v2_p4_clean_support_policy"]
                + (["v2_p10_self_pu_scope"] if self.method_name == "self_pu" else []),
            }
        )

    def bind_checkpoint_directory(self, root):
        """Called by runner preflight; the measured root is the writer's root."""
        self._checkpoint_dir = None if root is None else str(Path(root).resolve())
        self._checkpoint_configured = True

    def checkpoint_epoch_bound(self):
        return self._checkpoint_epoch_bound

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None, os_or_ts=None):
        self._require_prepared(estimator, X, os_or_ts)
        if not getattr(self, "_checkpoint_configured", False):
            raise ValueError("runner checkpoint preflight must bind the checkpoint directory")
        trajectory = EpochCheckpointTrainer(checkpoint_dir=self._checkpoint_dir).fit(
            _SupportFitBinding(estimator, self), X, y, class_prior=class_prior, os_or_ts="os"
        )
        sizes = [Path(row.path).stat().st_size for row in trajectory.checkpoints]
        if max(sizes) > self._checkpoint_bytes_per_component:
            raise ValueError("actual checkpoint exceeds checkpoint_bytes_per_component estimate")
        self._finish_support_fit(estimator, y)
        estimator.training_provenance_.update(
            {
                "selection_scope": "completed_meta_epochs",
                "captured_meta_epochs": [row.epoch for row in trajectory.epochs],
                "captured_components": sorted({row.component for row in trajectory.checkpoints}),
                "checkpoint_training_resume_supported": False,
                "checkpoint_bytes_per_component_estimate": self._checkpoint_bytes_per_component,
                "largest_checkpoint_bytes": max(sizes),
                "checkpoint_bytes_written_this_attempt": sum(sizes),
                "external_teacher_selection": (
                    "independent_per_protocol_over_both_teacher_snapshots"
                    if self.method_name == "self_pu"
                    else "not_applicable"
                ),
            }
        )
        trajectory.model = estimator
        return trajectory


class SelfPUCleanSupportCheckpointTrainer(_SupportCheckpointSelection, SelfPUCleanSupportTrainer):
    """SCAR OS full-meta Self-PU; independently select both EMA teacher streams."""


class LaGAMCleanSupportCheckpointTrainer(_SupportCheckpointSelection, LaGAMCleanSupportTrainer):
    """SCAR OS LaGAM; select completed post-warmup classifier snapshots."""
