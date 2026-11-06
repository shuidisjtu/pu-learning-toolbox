"""Versioned execution of the owner-confirmed November backlog recipes.

Recipe permission is distinct from numerical reproduction and formal admission.
The exploratory candidate trainers keep their original fail-closed boundaries.
"""

# ruff: noqa: N803

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from sklearn.base import clone

from .checkpoints import EpochCheckpointTrainer
from .cvir_adapter import CVIRPopulationPriorTrainer
from .kernel_candidate import KLDCECandidateTrainer

ADAPTER = "survey-backlog-1"
PROTOCOLS = ("survey-v3.2",)
DEEP_METHODS = ("gradpu", "pan", "p3mix", "vpu", "cvir")
VIEWS = dict(gradpu="ts", pan="os", p3mix="os", vpu="ts", cvir="ts")


def backlog_trainer(method, protocol):
    """Choose only the adapter explicitly bound by the new protocol revision."""
    if protocol.get("protocol_version") not in PROTOCOLS:
        raise ValueError("backlog adapters require survey-v3.2")
    if method in DEEP_METHODS:
        return SurveyBacklogTrainer(method)
    if method in {"puet", "rp"}:
        from .classical_candidate import ClassicalCandidateTrainer

        return ClassicalCandidateTrainer()
    if method == "kldce":
        profile = protocol["method_profiles"][method]
        budget = protocol["budgets"][profile["budget"]]
        return SurveyKLDCETrainer(
            prior_policy="population_override",
            kernel_payload_cap_bytes=budget["kernel_payload_cap_bytes"],
        )
    raise ValueError("unknown backlog adapter method")


class SurveyKLDCETrainer(KLDCECandidateTrainer):
    """Keep KLDCE's SCAR/count/resource gates, but allow the bound project recipe."""

    def validate_experiment(self, model, bundle, config, protocols):
        from .kernel_candidate import validate_kldce_candidate_parameters
        from .strategies import ProtocolOA, ProtocolPA
        from .survey_protocol import load_protocol

        request = config.get("survey_protocol") or {}
        protocol = load_protocol(request["path"])
        if (
            protocol["protocol_version"] not in PROTOCOLS
            or request.get("method") != "kldce"
            or request.get("mechanism") != "scar"
            or config.get("os_or_ts") != "os"
            or [type(p) for p in protocols] != [ProtocolPA, ProtocolOA]
        ):
            raise ValueError("KLDCE requires its versioned SCAR/OS PA-OA binding")
        if config.get("capture_epoch_checkpoints", False):
            raise ValueError("KLDCE has no neural epoch checkpoints")
        validate_kldce_candidate_parameters(model)
        self.prediction_screen(bundle)


class SurveyBacklogTrainer(EpochCheckpointTrainer):
    """Full trajectories; SCAR PA/OA, SAR OA; selection data never enters fit."""

    records_epoch_checkpoints = True

    def __init__(self, method, *, checkpoint_dir=None):
        if method not in DEEP_METHODS:
            raise ValueError("unsupported backlog deep method")
        super().__init__(checkpoint_dir=checkpoint_dir)
        self.method = method

    def validate_experiment(self, model, bundle, config, protocols):
        from ..registry import get_algorithm
        from .strategies import ProtocolOA, ProtocolPA
        from .survey_protocol import load_protocol, resolve_unit, validate_parameters

        request = config.get("survey_protocol") or {}
        protocol = load_protocol(request["path"])
        row = resolve_unit(
            protocol, request["dataset"], request["method"], request["training_path"]
        )
        profile = protocol["method_profiles"][row["method"]]
        if (
            protocol["protocol_version"] not in PROTOCOLS
            or profile.get("execution_adapter") != ADAPTER
            or profile.get("algorithm") != self.method
            or type(model) is not get_algorithm(self.method)
        ):
            raise ValueError("backlog trainer requires its versioned algorithm/profile binding")
        if config.get("os_or_ts") != VIEWS[self.method]:
            raise ValueError("backlog recipe training view mismatch")
        memory = config.get("gradpu_activation_memory", "device")
        if memory not in ("device", "cpu_offload") or (
            memory != "device" and self.method != "gradpu"
        ):
            raise ValueError("invalid GradPU activation memory execution policy")
        if self.method == "gradpu" and model.activation_memory != memory:
            raise ValueError("GradPU activation memory policy must be explicitly recorded")
        backward_mode = config.get("gradpu_backward_mode", "joint")
        chunk_size = config.get("gradpu_gp_chunk_size")
        if (
            backward_mode not in ("joint", "split")
            or (chunk_size is not None and (type(chunk_size) is not int or chunk_size < 1))
            or (chunk_size is not None and backward_mode != "split")
        ):
            raise ValueError("invalid GradPU backward execution policy")
        if self.method != "gradpu" and (backward_mode != "joint" or chunk_size is not None):
            raise ValueError("GradPU backward execution policy requires gradpu")
        if self.method == "gradpu" and (
            getattr(model, "backward_mode", None) != backward_mode
            or getattr(model, "gp_chunk_size", None) != chunk_size
        ):
            raise ValueError("GradPU backward execution policy must be explicitly recorded")
        self.mechanism = request["mechanism"]
        self.recipe_protocol = protocol["protocol_version"]
        if self.mechanism not in protocol["c_tokens"]:
            raise ValueError("unknown backlog labeling mechanism")
        expected = [ProtocolPA, ProtocolOA] if self.mechanism == "scar" else [ProtocolOA]
        if [type(p) for p in protocols] != expected:
            raise ValueError("backlog selection must be PA/OA for SCAR and OA-only for SAR")
        if config.get("capture_epoch_checkpoints", True) is not True:
            raise ValueError("backlog requires all completed epoch checkpoints")
        size = config.get("checkpoint_bytes_per_component")
        if type(size) is not int or size < 1:
            raise ValueError("backlog requires positive checkpoint byte bound")
        self._checkpoint_bytes_per_component = size
        candidates = config.get("candidates")
        if candidates != protocol["candidate_pool"]:
            raise ValueError("backlog requires the frozen candidate pool")
        counts = []
        for parameters in candidates:
            validate_parameters(parameters, profile)
            candidate = clone(model).set_params(**parameters)
            if self.method == "cvir":
                CVIRPopulationPriorTrainer._validate_model(candidate, VIEWS[self.method])
            count = getattr(candidate, "checkpoint_epoch_count", candidate.max_epochs)
            if type(count) is not int or count < 1:
                raise ValueError("invalid backlog checkpoint epoch count")
            counts.append(count)
        self._epoch_bound = max(counts)

    def validate_generation(self, train_meta, val_meta):
        if any(meta.get("mechanism") != self.mechanism for meta in (train_meta, val_meta)):
            raise ValueError("generated labeling mechanism differs from backlog recipe")

    def bind_checkpoint_directory(self, root):
        self.checkpoint_dir = root

    def checkpoint_epoch_bound(self):
        return self._epoch_bound

    def _checkpoint_context(self, fitted, epoch):
        if self.method in {"pan", "vpu"}:
            return dict(
                stage="adversarial" if self.method == "pan" else "variational",
                stage_epoch=epoch + 1,
                round_index=None,
                optimizer_steps=fitted.optimizer_steps_,
            )
        return super()._checkpoint_context(fitted, epoch)

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None, os_or_ts=None):
        if os_or_ts != VIEWS[self.method] or not hasattr(self, "_epoch_bound"):
            raise ValueError("backlog trainer must be validated before fitting")
        try:
            if self.method == "cvir":
                adapter = CVIRPopulationPriorTrainer(
                    population_prior_source=(
                        f"{self.recipe_protocol} canonical population prior; TS risk pool"
                    ),
                    checkpoint_dir=self.checkpoint_dir,
                )
                trajectory = adapter.fit(
                    estimator, X, y, class_prior=class_prior, os_or_ts=os_or_ts
                )
            else:
                # These objectives do not consume population pi. PA still does.
                trajectory = super().fit(estimator, X, y, os_or_ts=os_or_ts)
            self._verify(estimator, y, trajectory)
            receipt = dict(getattr(estimator, "training_provenance_", {}))
            receipt.update(
                execution_adapter=ADAPTER,
                recipe_protocol=self.recipe_protocol,
                external_validation_forwarded_to_fit=False,
                mechanism=self.mechanism,
                sar_scope="stress_test_without_new_selection_bias_correction",
                completed_checkpoint_epochs=len(trajectory.checkpoints),
                optimizer_steps=estimator.optimizer_steps_,
                checkpoint_scope="inference_only_not_optimizer_resume",
                formal_admission=False,
                source_reproduction=False,
            )
            estimator.training_provenance_ = receipt
            return trajectory
        except BaseException:
            estimator._is_fitted = False
            raise

    def _verify(self, model, labels, trajectory):
        checkpoints = trajectory.checkpoints
        count = getattr(model, "checkpoint_epoch_count", model.max_epochs)
        if len(checkpoints) != count or count != self._epoch_bound:
            raise ValueError("backlog completed trajectory differs from frozen epoch budget")
        if any(
            Path(c.path).stat().st_size > self._checkpoint_bytes_per_component for c in checkpoints
        ):
            raise ValueError("backlog checkpoint exceeds declared byte bound")
        p = int(np.count_nonzero(np.asarray(labels) == 1))
        u = len(labels) - p
        if self.method == "cvir":
            phases = ["warm_start"] * model.warm_start_epochs + ["cvir"] * model.max_epochs
            if [c.training_context["stage"] for c in checkpoints] != phases:
                raise ValueError("CVIR checkpoint stages differ from both-stage budget")
            if checkpoints[-1].training_context["optimizer_steps"] != model.optimizer_steps_:
                raise ValueError("CVIR checkpoint cost differs from completed updates")
            return
        steps = max(math.ceil(p / model.batch_size), math.ceil(u / model.batch_size))
        phase = "pu"
        if self.method == "pan":
            steps, phase = 2 * math.ceil(u / model.batch_size), "adversarial"
        elif self.method == "p3mix":
            phase = "train"
            if model.batching_policy == "merged_shuffle":
                steps = math.ceil(len(labels) / model.batch_size)
        elif self.method == "vpu":
            phase = "variational"
            steps = model.iterations_per_epoch or max(
                math.ceil(len(labels) / model.batch_size), math.ceil(p / model.batch_size)
            )
            if model.n_loss_unlabeled_ != len(labels):
                raise ValueError("VPU TS risk pool must include the full train marginal")
        for i, checkpoint in enumerate(checkpoints, 1):
            if checkpoint.training_context != dict(
                stage=phase, stage_epoch=i, round_index=None, optimizer_steps=i * steps
            ):
                raise ValueError("backlog checkpoint stage/update identity mismatch")
        if model.optimizer_steps_ != count * steps:
            raise ValueError("backlog final optimizer update count mismatch")
