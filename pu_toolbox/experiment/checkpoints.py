# ruff: noqa: N803, N806
"""Bounded-memory, weights-only epoch checkpoints for offline PA/OA selection.

No clean validation or test data reaches the recording callback. Architecture
templates are kept once on CPU; each checkpoint holds only a file reference.
Files contain tensor state dictionaries, never pickled model objects or data.
These are inference snapshots, not optimizer/RNG training-resume checkpoints.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .protocols import Trainer
from .tracking import EpochRecord, RunTrajectory


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SnapshotPredictor:
    """Inference-only snapshot with the original raw-score prediction cutoff."""

    def __init__(self, network, *, device, cutoff=0.0, batch_size=256):
        self.model_ = network.to(device).eval()
        self.device = device
        self.cutoff = cutoff
        self.batch_size = batch_size
        self.selection_threshold = None
        self.selection_metrics = {}

    def decision_function(self, X):
        import torch

        X = np.asarray(X, dtype=np.float32)
        if not len(X):
            return np.empty(0, dtype=np.float32)
        with torch.no_grad():
            return np.concatenate(
                [
                    self.model_(
                        torch.as_tensor(X[start : start + self.batch_size], device=self.device)
                    )
                    .reshape(-1)
                    .cpu()
                    .numpy()
                    for start in range(0, len(X), self.batch_size)
                ]
            )

    def predict(self, X):
        scores = self.decision_function(X)
        threshold = self.selection_threshold
        if threshold is None:
            threshold = self.cutoff
        else:
            scale = self.selection_metrics.get("val_score_scale")
            if scale:
                scores = (scores - self.selection_metrics["val_score_min"]) / scale
        return (scores >= threshold).astype(int)

    def set_selection(self, threshold, metrics):
        # Preserve the exact affine operation, rather than invert the cutoff:
        # float32 boundary rounding can change predictions after inversion.
        self.selection_threshold = threshold
        self.selection_metrics = dict(metrics)


@dataclass
class EpochCheckpoint:
    epoch_position: int
    epoch_label: int
    component: str
    path: str
    sha256: str
    device: str
    cutoff: float
    template: object = field(repr=False, compare=False)
    persistent: bool = False
    validation_metrics: dict = field(default_factory=dict)
    validation_elapsed_seconds: float = 0.0

    def reference(self):
        return {
            "schema_version": "1.0",
            "format": "torch_weights_only",
            "epoch_position": self.epoch_position,
            "epoch_label": self.epoch_label,
            "component": self.component,
            "path": self.path if self.persistent else None,
            "sha256": self.sha256,
            "device": self.device,
            "score_cutoff": self.cutoff,
            "persistent": self.persistent,
            "training_resume_supported": False,
            "validation_metrics": copy.deepcopy(self.validation_metrics),
        }

    def restore(self, *, device=None):
        """Verify bytes before loading tensors; reject missing/corrupt weights."""
        import torch

        if _file_digest(self.path) != self.sha256:
            raise ValueError("checkpoint weight digest mismatch")
        network = copy.deepcopy(self.template)
        network.load_state_dict(torch.load(self.path, map_location="cpu", weights_only=True))
        return SnapshotPredictor(network, device=device or self.device, cutoff=self.cutoff)


def selection_models(trajectory):
    """Yield one restored snapshot at a time, or the legacy single fitted model."""
    if trajectory.checkpoints:
        for index, checkpoint in enumerate(trajectory.checkpoints):
            started_at = time.perf_counter()
            model = checkpoint.restore()
            checkpoint.validation_elapsed_seconds += time.perf_counter() - started_at
            yield index, checkpoint.epoch_position, model
    else:
        yield None, trajectory.best_epoch, trajectory.model


def load_epoch_checkpoint(reference, template, *, device=None):
    """Restore a persisted reference with an explicitly supplied architecture.

    The caller builds the architecture from the locked protocol, not arbitrary
    code deserialized from the checkpoint. Absolute artifact paths must remain
    available (or be explicitly relocated by the caller).
    """
    if reference.get("format") != "torch_weights_only" or not reference.get("path"):
        raise ValueError("checkpoint reference must identify persisted weights-only files")
    checkpoint = EpochCheckpoint(
        reference["epoch_position"],
        reference["epoch_label"],
        reference["component"],
        reference["path"],
        reference["sha256"],
        reference["device"],
        reference["score_cutoff"],
        template,
        persistent=True,
    )
    return checkpoint.restore(device=device)


def load_selected_checkpoint(selection, template, *, device=None):
    """Restore the selected weights and the selected VAL-side prediction rule."""
    model = load_epoch_checkpoint(selection["checkpoint"], template, device=device)
    threshold = selection.get("threshold")
    if threshold is not None:
        model.set_selection(threshold, selection["metrics"])
    return model


def record_validation(trajectory, index, protocol, metrics, started_at):
    if index is not None:
        checkpoint = trajectory.checkpoints[index]
        checkpoint.validation_metrics[protocol] = dict(metrics)
        checkpoint.validation_elapsed_seconds += time.perf_counter() - started_at


class EpochCheckpointTrainer(Trainer):
    """Capture completed epochs without repeated fit, optimizer resets or val leakage."""

    def __init__(self, *, supervised=False, checkpoint_dir=None):
        self.trains_on_real_labels = supervised
        self.checkpoint_dir = checkpoint_dir

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None):
        import torch

        params = inspect.signature(type(estimator).fit).parameters
        if "epoch_callback" not in params:
            raise ValueError(
                "epoch checkpoint trainer requires an explicit epoch_callback fit hook"
            )
        if self.trains_on_real_labels and class_prior is not None:
            raise ValueError("supervised checkpoint trainer must not receive a PU class prior")
        if self.checkpoint_dir is None:
            owner = tempfile.TemporaryDirectory(prefix="pu-epoch-checkpoints-")
            directory = Path(owner.name)
        else:
            owner = None
            root = Path(self.checkpoint_dir)
            root.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
        epochs, checkpoints, templates = [], [], {}

        def record(epoch, fitted):
            if epochs and epoch <= epochs[-1].epoch:
                raise ValueError("checkpoint epoch labels must be strictly increasing")
            history = getattr(fitted, "history_", {})
            metrics = {
                name: float(values[-1])
                for name, values in history.items()
                if name in ("val_risk", "train_risk", "nnpu_risk") and len(values)
            }
            epochs.append(EpochRecord(epoch=int(epoch), metrics=metrics))
            if hasattr(fitted, "teacher_1_"):
                networks = {"teacher_1": fitted.teacher_1_, "teacher_2": fitted.teacher_2_}
                threshold = fitted.threshold
                cutoff = float(np.log(threshold / (1 - threshold)))
            else:
                networks, cutoff = {"model": fitted.model_}, 0.0
            for component, network in networks.items():
                if component not in templates:
                    templates[component] = copy.deepcopy(network).cpu().eval()
                path = directory / f"epoch_{len(epochs):04d}_{component}.pt"
                state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in network.state_dict().items()
                }
                if any(not torch.isfinite(tensor).all() for tensor in state.values()):
                    raise FloatingPointError(
                        "completed epoch checkpoint contains non-finite weights"
                    )
                # Unique per-attempt directories prevent overwriting another run.
                torch.save(state, path)
                checkpoints.append(
                    EpochCheckpoint(
                        len(epochs),
                        int(epoch),
                        component,
                        str(path.resolve()),
                        _file_digest(path),
                        str(next(network.parameters()).device),
                        cutoff,
                        templates[component],
                        persistent=owner is None,
                    )
                )

        kwargs = {"epoch_callback": record}
        if class_prior is not None and "class_prior" in params:
            kwargs["class_prior"] = class_prior
        if not self.trains_on_real_labels and val_pu is not None:
            name = "pu_validation_data" if "pu_validation_data" in params else "validation_data"
            if name in params:
                kwargs[name] = val_pu
        # No TypeError fallback: a callback error must not become a second bare fit.
        estimator.fit(X, y, **kwargs)
        if not checkpoints:
            raise ValueError("epoch callback produced no completed checkpoints")
        return RunTrajectory(epochs, estimator, checkpoints=checkpoints, checkpoint_owner=owner)
