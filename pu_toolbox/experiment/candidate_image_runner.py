# ruff: noqa: N803
"""Audited candidate image trajectories for an unbound four-role runner.

Reuse the existing image construction/input contract and weights-only writer.
This bridge does not admit methods into a versioned Survey protocol.
"""

from __future__ import annotations

import copy
import math
import time
from numbers import Real
from pathlib import Path

import numpy as np
from sklearn.base import clone

from ..utils.serialization import strict_canonical_hash
from .candidate_image import (
    _candidate_image_training_receipt,
    _cvir_epoch_facts,
    _cvir_image_schedule,
    _p3mix_epoch_facts,
    _validate_candidate_image_fit,
    _vpu_normalization_facts,
)
from .checkpoints import EpochCheckpointTrainer


class CandidateImageCheckpointTrainer(EpochCheckpointTrainer):
    """Bind seeded images, all completed stages and a real per-snapshot bound."""

    records_epoch_checkpoints = True

    def __init__(
        self,
        image_setup,
        *,
        checkpoint_bytes_per_component,
        checkpoint_dir=None,
        population_prior_source=None,
    ):
        super().__init__(checkpoint_dir=checkpoint_dir)
        if type(checkpoint_bytes_per_component) is not int or checkpoint_bytes_per_component <= 0:
            raise ValueError("checkpoint_bytes_per_component must be a positive integer")
        if not isinstance(image_setup, dict) or image_setup.get("method") not in (
            "pulda",
            "gradpu",
            "pan",
            "vpu",
            "p3mix",
            "cvir",
        ):
            raise ValueError("a supported candidate image construction record is required")
        self.image_setup = copy.deepcopy(image_setup)
        self._setup_digest = strict_canonical_hash(self.image_setup)
        self.checkpoint_bytes_per_component = checkpoint_bytes_per_component
        self.population_prior_source = population_prior_source
        if image_setup["method"] == "cvir" and (
            not isinstance(population_prior_source, str) or not population_prior_source.strip()
        ):
            raise ValueError("CVIR image runner requires population_prior_source")

    def _validate(self, model, features, labels, view):
        if strict_canonical_hash(self.image_setup) != self._setup_digest:
            raise ValueError("candidate image setup changed after binding")
        expected_view = "os" if self.image_setup["method"] in {"pan", "p3mix"} else "ts"
        if (self.image_setup["method"] == "cvir" and view not in {"os", "ts"}) or (
            self.image_setup["method"] != "cvir" and view != expected_view
        ):
            raise ValueError(f"candidate image runner requires explicit {expected_view} view")
        bound = _validate_candidate_image_fit(model, self.image_setup, features, labels)
        if model.device not in ("cpu", "cuda"):
            raise ValueError("candidate image runner requires explicit cpu or cuda")
        if model.device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise ValueError("requested CUDA unavailable; no CPU fallback")
        if self.image_setup["image_preprocessing"]["augmentation"]["train"]["name"] != "none":
            raise ValueError("this exploratory runner binds augmentation=none only")
        if self.image_setup["method"] in {"pulda", "vpu"}:
            model._validate_parameters()
        else:
            for key in ("hidden_dim", "max_epochs", "batch_size"):
                if type(getattr(model, key)) is not int or getattr(model, key) < 1:
                    raise ValueError(f"{key} must be a positive integer")
        if self.image_setup["method"] == "pan":
            for key in ("learning_rate", "adversarial_weight"):
                value = getattr(model, key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not np.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(f"{key} must be finite and positive")
        if self.image_setup["method"] == "vpu" and model.data_device_policy != "cpu_stream":
            raise ValueError("VPU candidate image runner requires explicit cpu_stream policy")
        if self.image_setup["method"] == "p3mix":
            self._validate_p3mix_parameters(model)
        if self.image_setup["method"] == "cvir":
            if type(model.warm_start_epochs) is not int or model.warm_start_epochs < 0:
                raise ValueError("CVIR warm_start_epochs must be a nonnegative integer")
            if (
                isinstance(model.learning_rate, bool)
                or not isinstance(model.learning_rate, Real)
                or not np.isfinite(model.learning_rate)
                or model.learning_rate <= 0
            ):
                raise ValueError("CVIR learning_rate must be finite and positive")
        return bound

    @staticmethod
    def _validate_p3mix_parameters(model):
        """Refuse ambiguous budgets before planning or creating any artifacts."""
        if model.variant not in {"base", "e", "c"} or model.batching_policy not in {
            "role_balanced",
            "merged_shuffle",
        }:
            raise ValueError("P3Mix requires explicit supported variant/batching policy")
        if type(model.pool_size) is not int or model.pool_size < 1:
            raise ValueError("P3Mix pool_size must be a positive integer")
        from ..estimators.deep.p3mix import _is_finite_real

        for name in (
            "learning_rate",
            "beta",
            "mixup_alpha",
            "gamma",
            "early_learning_weight",
            "teacher_decay",
        ):
            value = getattr(model, name)
            if not _is_finite_real(value):
                raise ValueError(f"P3Mix {name} must be finite real")
        if (
            min(model.learning_rate, model.beta, model.mixup_alpha) <= 0
            or not 0.5 <= model.gamma <= 1
            or model.early_learning_weight < 0
            or (model.variant == "e" and model.early_learning_weight <= 0)
            or not 0 <= model.teacher_decay < 1
        ):
            raise ValueError("P3Mix parameter outside its native range")

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
                {
                    "encoder",
                    "model",
                    "discriminator",
                    "train_transform",
                    "random_state",
                    "class_prior",
                    "device",
                }
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

    def _checkpoint_context(self, fitted, epoch):
        if self.image_setup["method"] == "cvir":
            facts = _cvir_epoch_facts(
                fitted, self._cvir_labels, self._cvir_alpha, self._cvir_view, epoch
            )
            context = super()._checkpoint_context(fitted, epoch)
            if context != facts["training_context"]:
                raise ValueError("CVIR checkpoint context differs from its completed stage")
            self._cvir_epoch_records.append(facts)
            return context
        if self.image_setup["method"] == "p3mix":
            self._p3mix_epoch_records.append(_p3mix_epoch_facts(fitted, self._p3mix_labels, epoch))
            return super()._checkpoint_context(fitted, epoch)
        if self.image_setup["method"] == "vpu":
            if fitted.n_loss_unlabeled_ != fitted.n_positive_ + fitted.n_unlabeled_:
                raise ValueError("VPU completed epoch did not use the complete TS train marginal")
            steps = fitted.iterations_per_epoch or max(
                math.ceil(fitted.n_loss_unlabeled_ / fitted.batch_size),
                math.ceil(fitted.n_positive_ / fitted.batch_size),
            )
            if fitted.optimizer_steps_ != steps * (epoch + 1):
                raise ValueError("VPU update cost differs at completed epoch")
            self._vpu_epoch_records.append(
                dict(epoch=int(epoch), **_vpu_normalization_facts(fitted))
            )
            return dict(
                stage="variational",
                stage_epoch=epoch + 1,
                round_index=None,
                optimizer_steps=fitted.optimizer_steps_,
            )
        if self.image_setup["method"] != "pan":
            return super()._checkpoint_context(fitted, epoch)
        expected = 2 * math.ceil(fitted.n_unlabeled_ / fitted.batch_size) * (epoch + 1)
        if fitted.optimizer_steps_ != expected:
            raise ValueError("PAN C/D update cost differs at completed epoch")
        return dict(
            stage="adversarial",
            stage_epoch=epoch + 1,
            round_index=None,
            optimizer_steps=fitted.optimizer_steps_,
        )

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
        if method == "vpu":
            self._vpu_epoch_records = []
        if method == "p3mix":
            self._p3mix_labels = bound[2]
            self._p3mix_epoch_records = []
        if method == "cvir":
            from .cvir_adapter import cvir_risk_prior

            self._cvir_labels, self._cvir_view = bound[2], os_or_ts
            self._cvir_alpha = cvir_risk_prior(
                class_prior,
                n_samples=len(y),
                n_labeled=int(np.count_nonzero(y == 1)),
                os_or_ts=os_or_ts,
            )
            _cvir_image_schedule(self._cvir_labels, self._cvir_alpha, os_or_ts, estimator)
            self._cvir_epoch_records = []
        try:
            if method == "cvir":
                estimator.set_params(
                    unlabeled_positive_prior=self._cvir_alpha,
                    prior_source=self.population_prior_source,
                )
                # The receipt describes the estimator's actual fit entry,
                # not the unbound constructor restored after this attempt.
                bound[5].update(
                    unlabeled_positive_prior=self._cvir_alpha,
                    prior_source=self.population_prior_source,
                )
            try:
                trajectory = super().fit(
                    estimator,
                    X,
                    y,
                    os_or_ts=os_or_ts,
                    class_prior=class_prior if method == "pulda" else None,
                )
            finally:
                if method == "cvir":
                    estimator.set_params(unlabeled_positive_prior=None, prior_source=None)
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
            if method == "pan":
                estimator.training_provenance_.update(
                    selection_prior_not_forwarded_to_PAN=True,
                    eligible_stages=["adversarial"],
                    checkpoint_scope="classifier_C_inference_only; no_D_optimizer_resume",
                    updates_per_minibatch={"discriminator": 1, "classifier": 1},
                )
            if method == "vpu":
                estimator.training_provenance_.update(
                    selection_prior_not_forwarded_to_VPU=True,
                    eligible_stages=["variational"],
                    checkpoint_scope="normalized_phi_inference_only; no_optimizer_resume",
                    source_validation_risk_used_for_selection=False,
                    source_validation_diagnostic_forwarded=False,
                    learning_rate_history=list(estimator.history_["learning_rate"]),
                    optimizer_generation_history=list(estimator.history_["optimizer_generation"]),
                    normalization_history=list(self._vpu_epoch_records),
                )
            if method == "p3mix":
                estimator.training_provenance_.update(
                    selection_prior_not_forwarded_to_P3Mix=True,
                    eligible_stages=["train"],
                    checkpoint_scope="student_inference_only; no_teacher_optimizer_resume",
                    teacher_selection_eligible=False,
                    epoch_cost_history=copy.deepcopy(self._p3mix_epoch_records),
                )
            if method == "cvir":
                estimator.training_provenance_.update(
                    population_prior_forwarded_to_fit=False,
                    population_prior_binding={
                        "population_prior": float(class_prior),
                        "risk_pool_positive_prior": self._cvir_alpha,
                        "prior_source": self.population_prior_source,
                        "source_verified": False,
                        "uses_hidden_training_labels": False,
                    },
                    eligible_stages=["warm_start", "cvir"],
                    checkpoint_scope="classifier_inference_only; no_optimizer_resume",
                    epoch_cost_history=copy.deepcopy(self._cvir_epoch_records),
                )
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
        if method == "cvir":
            expected = [row["training_context"] for row in self._cvir_epoch_records]
            if (
                len(expected) != estimator.warm_start_epochs + estimator.max_epochs
                or [c.training_context for c in trajectory.checkpoints] != expected
                or estimator.optimizer_steps_ != expected[-1]["optimizer_steps"]
                or estimator.unlabeled_positive_prior_ != self._cvir_alpha
            ):
                raise ValueError("CVIR image checkpoint/stage cost coverage mismatch")
            return
        if method == "pulda":
            steps = math.ceil(u / estimator.unlabeled_batch_size)
            phases = ["warmup"] * estimator.warmup_epochs + ["pu_mixup"] * estimator.pu_epochs
        elif method == "pan":
            steps = 2 * math.ceil(u / estimator.batch_size)
            phases = ["adversarial"] * estimator.max_epochs
        elif method == "vpu":
            steps = estimator.iterations_per_epoch or max(
                math.ceil(len(labels) / estimator.batch_size), math.ceil(p / estimator.batch_size)
            )
            phases = ["variational"] * estimator.max_epochs
        elif method == "p3mix":
            steps = (
                math.ceil(len(labels) / estimator.batch_size)
                if estimator.batching_policy == "merged_shuffle"
                else max(math.ceil(p / estimator.batch_size), math.ceil(u / estimator.batch_size))
            )
            phases = ["train"] * estimator.max_epochs
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
