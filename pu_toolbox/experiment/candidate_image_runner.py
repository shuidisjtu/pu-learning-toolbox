# ruff: noqa: N803
"""Audited PULDA/GradPU image trajectories for an unbound four-role runner.

Reuse the existing image construction/input contract and weights-only writer.
This bridge does not admit either method into a versioned Survey protocol.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
from sklearn.base import clone

from ..utils.serialization import strict_canonical_hash
from .candidate_image import _candidate_image_training_receipt, _validate_candidate_image_fit
from .checkpoints import EpochCheckpointTrainer


class CandidateImageCheckpointTrainer(EpochCheckpointTrainer):
    """Bind seeded images, all completed stages and a real per-snapshot bound."""

    records_epoch_checkpoints = True

    def __init__(self, image_setup, *, checkpoint_bytes_per_component, checkpoint_dir=None):
        super().__init__(checkpoint_dir=checkpoint_dir)
        if type(checkpoint_bytes_per_component) is not int or checkpoint_bytes_per_component <= 0:
            raise ValueError("checkpoint_bytes_per_component must be a positive integer")
        if not isinstance(image_setup, dict) or image_setup.get("method") not in (
            "pulda",
            "gradpu",
        ):
            raise ValueError("a PULDA or GradPU image construction record is required")
        self.image_setup = copy.deepcopy(image_setup)
        self._setup_digest = strict_canonical_hash(self.image_setup)
        self.checkpoint_bytes_per_component = checkpoint_bytes_per_component

    def _validate(self, model, features, labels, view):
        if strict_canonical_hash(self.image_setup) != self._setup_digest:
            raise ValueError("candidate image setup changed after binding")
        if view != "ts":
            raise ValueError("candidate image runner requires explicit ts view")
        bound = _validate_candidate_image_fit(model, self.image_setup, features, labels)
        if model.device not in ("cpu", "cuda"):
            raise ValueError("candidate image runner requires explicit cpu or cuda")
        if model.device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise ValueError("requested CUDA unavailable; no CPU fallback")
        if self.image_setup["image_preprocessing"]["augmentation"]["train"]["name"] != "none":
            raise ValueError("this runner binds the approved shared augmentation=none path")
        if self.image_setup["method"] == "pulda":
            model._validate_parameters()
        else:
            for key in ("hidden_dim", "max_epochs", "batch_size"):
                if type(getattr(model, key)) is not int or getattr(model, key) < 1:
                    raise ValueError(f"{key} must be a positive integer")
        return bound

    def validate_experiment(self, model, bundle, config, protocols):
        """Reject formal units and inconsistent candidates before any fitting."""
        from .strategies import ProtocolOA, ProtocolPA

        if config.get("survey_protocol") is not None or config.get("recipe_registry") is not None:
            raise ValueError("candidate image runner is exploratory, not a versioned Survey unit")
        if config.get("capture_epoch_checkpoints", True) is not True:
            raise ValueError("candidate image runner requires completed epoch checkpoints")
        bound = config.get("checkpoint_bytes_per_component")
        if type(bound) is not int or bound != self.checkpoint_bytes_per_component:
            raise ValueError("runner and writer checkpoint byte bounds must agree")
        if [type(protocol) for protocol in protocols] != [ProtocolPA, ProtocolOA]:
            raise ValueError("candidate image runner requires independent PA then OA selection")
        candidates = config.get("candidates", [{}])
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("candidate list must be nonempty")
        budgets = []
        for parameters in candidates:
            if not isinstance(parameters, dict) or set(parameters).intersection(
                {"encoder", "model", "train_transform", "random_state", "class_prior", "device"}
            ):
                raise ValueError("candidates cannot override image/prior/seed/device bindings")
            candidate = clone(model).set_params(**parameters)
            self._validate(candidate, bundle.train.X, bundle.train.labels, config.get("os_or_ts"))
            budgets.append(
                candidate.checkpoint_epoch_count
                if hasattr(candidate, "checkpoint_epoch_count")
                else candidate.max_epochs
            )
        self._epoch_bound = max(budgets)

    def validate_generation(self, train_meta, val_meta):
        if any(meta.get("mechanism") != "scar" for meta in (train_meta, val_meta)):
            raise ValueError(
                "candidate image runner supports SCAR only, not an approved SAR recipe"
            )

    def bind_checkpoint_directory(self, root):
        self.checkpoint_dir = root

    def checkpoint_epoch_bound(self):
        return self._epoch_bound

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None, os_or_ts=None):
        """Forward train PU data only; preserve native losses and paid stage costs."""
        started = time.perf_counter()
        bound = self._validate(estimator, X, y, os_or_ts)
        method = self.image_setup["method"]
        if method == "pulda" and class_prior != estimator.class_prior:
            raise ValueError("PULDA constructor prior differs from the runner population prior")
        previous = getattr(estimator, "training_provenance_", {})
        if isinstance(previous, dict) and previous.get("candidate_image_runner_schema"):
            del estimator.training_provenance_
        try:
            trajectory = super().fit(
                estimator,
                X,
                y,
                os_or_ts=os_or_ts,
                class_prior=class_prior if method == "pulda" else None,
            )
            receipt = _candidate_image_training_receipt(estimator, bound, started)
            self._verify_trajectory(estimator, y, trajectory)
            sizes = [Path(checkpoint.path).stat().st_size for checkpoint in trajectory.checkpoints]
            if any(size > self.checkpoint_bytes_per_component for size in sizes):
                raise ValueError("candidate image checkpoint exceeded declared byte bound")
            estimator.training_provenance_ = {
                "candidate_image_runner_schema": "candidate-image-checkpoint-runner-v1",
                "bound_fit_receipt": receipt,
                "selection_prior_not_forwarded_to_GradPU": method == "gradpu",
                "external_validation_forwarded_to_fit": False,
                "eligible_stages": ["warmup", "pu_mixup"] if method == "pulda" else ["pu"],
                "stage_eligibility_scope": "explicit_engineering_choice_not_formal_budget_approval",
                "checkpoint_bytes": sizes,
                "max_candidate_epoch_bound": getattr(self, "_epoch_bound", len(sizes)),
                "formal_admission": False,
            }
        except BaseException:
            # A rejected resource/receipt is not a successfully accepted model.
            estimator._is_fitted = False
            if hasattr(estimator, "training_provenance_"):
                del estimator.training_provenance_
            raise
        return trajectory

    def _verify_trajectory(self, estimator, labels, trajectory):
        """Verify every real snapshot, phase and cumulative optimizer count."""
        p = int(np.count_nonzero(labels == 1))
        u = len(labels) - p
        method = self.image_setup["method"]
        if method == "pulda":
            steps = math.ceil(u / estimator.unlabeled_batch_size)
            phases = ["warmup"] * estimator.warmup_epochs + ["pu_mixup"] * estimator.pu_epochs
        else:
            steps = max(math.ceil(p / estimator.batch_size), math.ceil(u / estimator.batch_size))
            phases = ["pu"] * estimator.max_epochs
        if (
            len(trajectory.checkpoints) != len(phases)
            or estimator.optimizer_steps_ != len(phases) * steps
        ):
            raise ValueError("candidate image completed checkpoint/cost coverage mismatch")
        counts = {}
        for index, (checkpoint, phase) in enumerate(
            zip(trajectory.checkpoints, phases, strict=True), 1
        ):
            counts[phase] = counts.get(phase, 0) + 1
            if checkpoint.training_context != {
                "stage": phase,
                "stage_epoch": counts[phase],
                "round_index": None,
                "optimizer_steps": index * steps,
            }:
                raise ValueError("candidate image phase or cumulative step identity mismatch")
