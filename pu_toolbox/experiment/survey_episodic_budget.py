"""Native episode, splitter-stop and full-horizon LZO budget verification.

This is the second explicit draft budget schema, not a formal protocol. No
training or budget defaults are supplied; every paid stage stays accounted.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..core.training_views import build_training_view
from ..utils.serialization import strict_canonical_hash

SCHEMA = "survey-staged-budget-2"
PARAMETERS = {
    "robust_pu": (
        "pretrain_epochs",
        "episodes",
        "inner_epochs",
        "batch_size",
        "hidden_dim",
        "pretrain_lr",
        "learning_rate",
    ),
    "split_pu": (
        "teacher_epochs",
        "split_epochs",
        "student_epochs",
        "rounds",
        "batch_size",
        "hidden_dim",
        "learning_rate",
        "agreement_threshold",
    ),
    "holistic_pu": (
        "warmup_epochs",
        "max_epochs",
        "batch_size",
        "hidden_dim",
        "learning_rate",
        "warmup_selection",
        "pseudo_pn_initialization",
        "lzo_alpha",
        "lzo_validation_size",
    ),
}
_INTS = {
    "pretrain_epochs",
    "episodes",
    "inner_epochs",
    "batch_size",
    "hidden_dim",
    "teacher_epochs",
    "split_epochs",
    "student_epochs",
    "rounds",
    "warmup_epochs",
    "max_epochs",
}
_CHOICES = {
    "warmup_selection": ("fixed", "lzo_positive_loss"),
    "pseudo_pn_initialization": ("continue", "reinitialize"),
}


def validate_parameters(algorithm, params):
    """Explicit native budget knobs, including stopping and internal selection."""
    if (
        algorithm not in PARAMETERS
        or not isinstance(params, dict)
        or (set(params) != set(PARAMETERS[algorithm]))
    ):
        raise ValueError("episodic budget requires every method-specific parameter")
    for name, value in params.items():
        if name in _CHOICES:
            valid = isinstance(value, str) and value in _CHOICES[name]
        elif name == "lzo_validation_size":
            valid = value is None or (type(value) is int and value > 0)
        elif name in _INTS:
            valid = type(value) is int and value >= (0 if name == "pretrain_epochs" else 1)
        else:
            valid = type(value) in (int, float) and math.isfinite(value) and value > 0
        if not valid:
            raise ValueError(f"invalid episodic budget parameter: {name}")
    if algorithm == "split_pu" and not params["agreement_threshold"] < 1:
        raise ValueError("splitter agreement threshold must be in (0, 1)")
    if algorithm == "holistic_pu" and params["warmup_epochs"] < 2:
        raise ValueError("Holistic-PU needs >=2 warmup epochs for predictive trends")


def schedule(algorithm, params):
    """Maximum native snapshot horizon, not the number of optimizer epochs."""
    if algorithm == "robust_pu":
        return [("pretrain", params["pretrain_epochs"]), ("self_paced", params["episodes"])]
    if algorithm == "split_pu":
        return [
            ("teacher", params["teacher_epochs"]),
            ("splitter", params["split_epochs"]),
            ("student", params["rounds"] * params["student_epochs"]),
        ]
    return [("warmup", params["warmup_epochs"]), ("pseudo_pn", params["max_epochs"])]


def validate_view(algorithm, params, view):
    """Refuse unsupported calibration before spending a training attempt."""
    if algorithm == "robust_pu" and view == "ts" and params["pretrain_epochs"] == 0:
        raise ValueError("Robust-PU TS requires a nonzero pretrain budget")
    if algorithm == "holistic_pu" and view not in (None, "os"):
        raise ValueError("Holistic-PU has no equivalent TS risk substitution")


def _phases(algorithm, params, *, split_count=None):
    phases = []
    for stage, count in schedule(algorithm, params):
        if stage == "student":
            phases.extend(
                (stage, epoch, turn)
                for turn in range(1, params["rounds"] + 1)
                for epoch in range(1, params["student_epochs"] + 1)
            )
        else:
            if stage == "splitter" and split_count is not None:
                count = split_count
            phases.extend((stage, epoch, None) for epoch in range(1, count + 1))
    return phases


def checkpoint_coverage(trajectory, contract):
    """Check native stage/round coverage; Split-PU's cap is an upper bound.

    The runner calls this only after per-candidate verification, which also
    validates the native stopping evidence and actual optimizer costs.
    """
    algorithm, params = contract["algorithm_id"], contract["parameter_values"]
    split_count = None
    if algorithm == "split_pu":
        split_count = sum(
            (c.training_context or {}).get("stage") == "splitter" for c in trajectory.checkpoints
        )
        if not 1 <= split_count <= params["split_epochs"]:
            return False
    phases = _phases(algorithm, params, split_count=split_count)
    if len(trajectory.epochs) != len(phases) or len(trajectory.checkpoints) != len(phases):
        return False
    return all(
        checkpoint.component == "model"
        and checkpoint.epoch_position == index
        and (
            (checkpoint.training_context or {}).get("stage"),
            (checkpoint.training_context or {}).get("stage_epoch"),
            (checkpoint.training_context or {}).get("round_index"),
        )
        == phase
        for index, (checkpoint, phase) in enumerate(
            zip(trajectory.checkpoints, phases, strict=True), 1
        )
    )


def _history_length(model, field, expected):
    if len(model.history_.get(field, ())) != expected:
        raise ValueError(f"episodic budget history coverage mismatch: {field}")


def _split_costs(model, params, training):
    p, u = len(training.positive_positions), len(training.native_unlabeled_positions)
    batch = params["batch_size"]
    agreements = model.history_.get("split_agreement", ())
    if not 1 <= len(agreements) <= params["split_epochs"] or any(
        type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
        for value in agreements
    ):
        raise ValueError("invalid native splitter stopping evidence")
    threshold = params["agreement_threshold"]
    if any(value >= threshold for value in agreements[:-1]) or (
        len(agreements) < params["split_epochs"] and agreements[-1] < threshold
    ):
        raise ValueError("splitter stopping rule differs from its declared threshold")
    easy, hard = model.n_easy_, model.n_hard_
    if type(easy) is not int or type(hard) is not int or min(easy, hard) < 1 or easy + hard != u:
        raise ValueError("splitter easy/hard counts must partition native U")
    _history_length(model, "teacher_risk", params["teacher_epochs"])
    _history_length(model, "student_loss", params["rounds"] * params["student_epochs"])
    _history_length(model, "round_weights", params["rounds"])
    return {
        "teacher": max(math.ceil(p / batch), math.ceil(u / batch)),
        "splitter": math.ceil((p + u) / batch),
        "student": max(math.ceil(n / batch) for n in (p, easy, hard)),
    }, len(agreements)


def _lzo_receipt(model, params, training, warmup_steps):
    """State rollback never refunds full-horizon updates or forward evaluations."""
    horizon = params["warmup_epochs"]
    if model.executed_warmup_epochs_ != horizon:
        raise ValueError("LZO must pay the full declared warmup horizon")
    if params["warmup_selection"] == "fixed":
        if (
            model.selected_warmup_epoch_ != horizon
            or model.discarded_warmup_optimizer_steps_ != 0
            or model.lzo_evaluated_rows_ != 0
            or model.lzo_forward_batches_ != 0
        ):
            raise ValueError("fixed warmup cannot claim LZO selection or unpaid work")
        return None
    losses = np.asarray(model.lzo_losses_)
    if losses.shape != (horizon,) or not np.isfinite(losses).all():
        raise ValueError("LZO loss evidence must cover every warmup epoch")
    selected = int(np.argmin(losses[1:])) + 2
    size = params["lzo_validation_size"] or len(training.positive_positions)
    rows, batches = horizon * size, horizon * math.ceil(size / params["batch_size"])
    if (
        model.selected_warmup_epoch_ != selected
        or model.discarded_warmup_optimizer_steps_ != (horizon - selected) * warmup_steps
        or model.lzo_validation_size_ != size
        or model.lzo_evaluated_rows_ != rows
        or model.lzo_forward_batches_ != batches
        or model.lzo_candidate_epochs_ != tuple(range(2, horizon + 1))
    ):
        raise ValueError("LZO selection/forward cost differs from its declared full horizon")
    pairs = np.asarray(model.lzo_mixup_indices_)
    if pairs.shape != (size, 2) or not np.isin(pairs, training.positive_positions).all():
        raise ValueError("LZO pseudo-validation pairs must use labeled training positives only")
    return {
        "selected_warmup_epoch": selected,
        "executed_warmup_epochs": horizon,
        "discarded_state_optimizer_steps_still_paid": model.discarded_warmup_optimizer_steps_,
        "forward_rows": rows,
        "forward_batches": batches,
        "source": "labeled_training_positive_only",
    }


def verify_trajectory(model, trajectory, contract, features, labels, view):
    """Verify actual stage costs after the shared estimator/contract preflight."""
    algorithm, params = contract["algorithm_id"], contract["parameter_values"]
    validate_view(algorithm, params, view)
    training = build_training_view(features, labels, requested_view=view or "os")
    p, u = len(training.positive_positions), len(training.native_unlabeled_positions)
    batch, extras = params["batch_size"], {}
    split_count = None
    if algorithm == "robust_pu":
        rates = {
            "pretrain": max(math.ceil(p / batch), math.ceil(u / batch)),
            "self_paced": params["inner_epochs"] * math.ceil((p + u) / batch),
        }
        _history_length(model, "pretrain_risk", params["pretrain_epochs"])
        _history_length(model, "episode_loss", params["episodes"])
    elif algorithm == "split_pu":
        rates, split_count = _split_costs(model, params, training)
        extras["splitter_epochs_completed"] = split_count
        extras["splitter_stopping_rule"] = "first_agreement_ge_threshold_or_declared_cap"
    else:
        rates = {"warmup": math.ceil(u / batch), "pseudo_pn": math.ceil((p + u) / batch)}
        _history_length(model, "warmup_loss", params["warmup_epochs"])
        _history_length(model, "pseudo_pn_loss", params["max_epochs"])
        extras["lzo"] = _lzo_receipt(model, params, training, rates["warmup"])
        if model.pseudo_pn_optimizer_reset_ != (
            params["pseudo_pn_initialization"] == "reinitialize"
        ):
            raise ValueError("Holistic-PU initialization/reset differs from budget")
    if not checkpoint_coverage(trajectory, contract):
        raise ValueError("episodic budget requires complete native stage/round coverage")
    phases = _phases(algorithm, params, split_count=split_count)
    cumulative, costs, sizes = 0, dict.fromkeys(rates, 0), []
    for checkpoint, (stage, epoch, turn) in zip(trajectory.checkpoints, phases, strict=True):
        cumulative += rates[stage]
        costs[stage] += rates[stage]
        if checkpoint.training_context != dict(
            stage=stage, stage_epoch=epoch, round_index=turn, optimizer_steps=cumulative
        ):
            raise ValueError("episodic stage/round or paid optimizer steps mismatch")
        size = Path(checkpoint.path).stat().st_size
        if size > contract["checkpoint_bytes_per_component"]:
            raise ValueError("episodic checkpoint exceeded its declared byte bound")
        sizes.append(size)
    if model.optimizer_steps_ != cumulative:
        raise ValueError("episodic total optimizer cost mismatch")
    if algorithm == "holistic_pu" and model.stage_optimizer_steps_ != costs:
        raise ValueError("Holistic-PU paid stage counters mismatch")
    return {
        "schema_version": "survey-staged-budget-receipt-2",
        "budget_sha256": strict_canonical_hash(contract),
        "checkpoint_epoch_count": len(phases),
        "checkpoint_epoch_bound": contract["checkpoint_epoch_count"],
        "checkpoint_count_is_upper_bound": algorithm == "split_pu",
        "eligible_stages": list(contract["eligible_stages"]),
        "stage_optimizer_steps": costs,
        "optimizer_steps": cumulative,
        "checkpoint_bytes": sizes,
        "formal_budget_approved": False,
        **extras,
    }
