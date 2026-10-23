"""Explicit GEN-PU/PULDA draft budgets and verification of paid trajectories.

No budget values are supplied here: callers must declare them in a versioned
profile. The schema does not grant formal admission or change native training.
"""

from __future__ import annotations

import math
from pathlib import Path

from ..core.training_views import build_training_view
from ..utils.serialization import strict_canonical_hash

FIELD = "staged_budget"
SCHEMA = "survey-staged-budget-1"
PARAMETERS = {
    "genpu": (
        "max_epochs",
        "classifier_epochs",
        "batch_size",
        "hidden_dim",
        "latent_dim",
        "learning_rate",
    ),
    "pulda": (
        "warmup_epochs",
        "pu_epochs",
        "positive_batch_size",
        "unlabeled_batch_size",
        "hidden_dim",
        "depth",
        "warmup_learning_rate",
        "learning_rate",
        "warmup_weight_decay",
        "weight_decay",
    ),
}
_INTS = {
    "max_epochs",
    "classifier_epochs",
    "batch_size",
    "hidden_dim",
    "latent_dim",
    "warmup_epochs",
    "pu_epochs",
    "positive_batch_size",
    "unlabeled_batch_size",
    "depth",
}


def _schedule(algorithm, params):
    if algorithm == "genpu":
        return [("synthetic_pn", params["classifier_epochs"])]
    return [("warmup", params["warmup_epochs"]), ("pu_mixup", params["pu_epochs"])]


def validate_contract(contract):
    """Closed shape, explicit numeric parameters, and exact selectable stages."""
    if (
        not isinstance(contract, dict)
        or set(contract)
        != {
            "schema_version",
            "algorithm_id",
            "parameter_values",
            "checkpoint_epoch_count",
            "checkpoint_bytes_per_component",
            "eligible_stages",
        }
        or contract.get("schema_version") != SCHEMA
    ):
        raise ValueError("invalid staged budget schema or fields")
    algorithm = contract["algorithm_id"]
    if not isinstance(algorithm, str) or algorithm not in PARAMETERS:
        raise ValueError("staged budget supports GEN-PU/PULDA only")
    params = contract["parameter_values"]
    if not isinstance(params, dict) or set(params) != set(PARAMETERS[algorithm]):
        raise ValueError("staged budget must declare every method-specific parameter")
    for name, value in params.items():
        minimum = 0 if name in ("warmup_epochs", "warmup_weight_decay", "weight_decay") else 1
        if name in _INTS:
            valid = type(value) is int and value >= minimum
        else:
            valid = (
                type(value) in (int, float)
                and math.isfinite(value)
                and (value >= 0 if minimum == 0 else value > 0)
            )
        if not valid:
            raise ValueError(f"invalid staged budget parameter: {name}")
    schedule = _schedule(algorithm, params)
    expected = sum(count for _, count in schedule)
    if type(contract["checkpoint_epoch_count"]) is not int or (
        contract["checkpoint_epoch_count"] != expected
    ):
        raise ValueError("staged budget checkpoint count differs from native stage schedule")
    if type(contract["checkpoint_bytes_per_component"]) is not int or (
        contract["checkpoint_bytes_per_component"] < 1
    ):
        raise ValueError("staged checkpoint byte bound must be a positive integer")
    if contract["eligible_stages"] != [stage for stage, count in schedule if count]:
        raise ValueError("staged budget must retain all native selectable stages in order")
    strict_canonical_hash(contract)


def validate_protocol_budgets(protocol):
    """Do not attach this extension to frozen legacy identities or wrong units."""
    from .survey_profile_identity import has_profile_identities

    explicit = has_profile_identities(protocol)
    for budget in protocol["budgets"].values():
        if isinstance(budget, dict) and FIELD in budget:
            if not explicit or set(budget) != {FIELD}:
                raise ValueError(
                    "staged budget requires explicit draft identity and no legacy fields"
                )
            validate_contract(budget[FIELD])
    if not explicit:
        return
    for profile in protocol["method_profiles"].values():
        algorithm = profile.get("algorithm")
        budget = protocol["budgets"].get(profile.get("budget"), {})
        if explicit and algorithm in PARAMETERS and FIELD not in budget:
            raise ValueError("GEN-PU/PULDA draft profile requires a staged budget")
        if FIELD not in budget:
            continue
        contract = budget[FIELD]
        if algorithm != contract["algorithm_id"]:
            raise ValueError("profile algorithm differs from staged budget")
        for name, value in contract["parameter_values"].items():
            if name not in profile["params"] or strict_canonical_hash(value) != (
                strict_canonical_hash(profile["params"][name])
            ):
                raise ValueError(f"profile must lock staged budget parameter: {name}")
    for row in protocol["execution_units"]:
        profile = protocol["method_profiles"][row["method"]]
        budget = protocol["budgets"][row["budget"]]
        if FIELD not in budget and FIELD not in protocol["budgets"].get(profile.get("budget"), {}):
            continue
        if row["budget"] != profile.get("budget"):
            raise ValueError("unit and profile staged budget must match")
        algorithm = profile["algorithm"]
        if row["training_path"] != "native_2d" or (
            row["model_family"] != f"{algorithm}_native"
            or row["backbone"] != f"{algorithm}_native_mlp"
        ):
            raise ValueError("staged draft requires its own native_2d architecture identity")
        expected = {
            "algorithm_id": algorithm,
            "parameter_values": {
                key: profile["params"][key]
                for key in ("hidden_dim", "latent_dim")
                if key in profile["params"]
            },
        }
        if algorithm == "pulda":
            expected["parameter_values"]["depth"] = profile["params"]["depth"]
        if protocol["backbone_specs"].get(row["backbone"]) != expected:
            raise ValueError("staged native architecture spec differs from profile")


def checkpoint_epochs(budget):
    """Snapshot epochs are not GEN-PU GAN epochs or PULDA PU epochs alone."""
    if FIELD in budget:
        validate_contract(budget[FIELD])
        return budget[FIELD]["checkpoint_epoch_count"]
    return budget.get("epochs")


def validate_estimator(model, contract):
    """Validate the actual class and parameters without constructing or training."""
    from ..registry import get_algorithm

    validate_contract(contract)
    if type(model) is not get_algorithm(contract["algorithm_id"]):
        raise ValueError("staged budget estimator class mismatch")
    actual = model.get_params(deep=False)
    for name, value in contract["parameter_values"].items():
        if strict_canonical_hash(actual.get(name)) != strict_canonical_hash(value):
            raise ValueError(f"estimator differs from staged budget parameter: {name}")
    if model.checkpoint_epoch_count != contract["checkpoint_epoch_count"]:
        raise ValueError("estimator staged checkpoint count mismatch")


def verify_trajectory(model, trajectory, contract, features, labels, view):
    """Reject incomplete/wrong-stage/over-bound snapshots before PA/OA selection."""
    validate_estimator(model, contract)
    algorithm = contract["algorithm_id"]
    params = contract["parameter_values"]
    training = build_training_view(features, labels, requested_view=view or "os")
    batch = params["batch_size"] if algorithm == "genpu" else params["unlabeled_batch_size"]
    # GEN-PU samples its view's risk pool; PULDA keeps native-U batches and
    # calibrates the expectations inside the loss instead of enlarging U.
    pool = (
        training.loss_unlabeled_positions
        if algorithm == "genpu"
        else training.native_unlabeled_positions
    )
    steps = math.ceil(len(pool) / batch)
    schedule = _schedule(algorithm, params)
    phases = [(stage, epoch) for stage, count in schedule for epoch in range(1, count + 1)]
    if len(trajectory.checkpoints) != len(phases) or len(trajectory.epochs) != len(phases):
        raise ValueError("staged budget requires every completed native checkpoint")
    prefix = 5 * steps * params["max_epochs"] if algorithm == "genpu" else 0
    sizes = []
    for index, (checkpoint, (stage, epoch)) in enumerate(
        zip(trajectory.checkpoints, phases, strict=True), 1
    ):
        expected = dict(
            stage=stage, stage_epoch=epoch, round_index=None, optimizer_steps=prefix + index * steps
        )
        if (
            checkpoint.component != "model"
            or checkpoint.epoch_position != index
            or (checkpoint.training_context != expected)
        ):
            raise ValueError("staged checkpoint stage or paid optimizer steps mismatch")
        size = Path(checkpoint.path).stat().st_size
        if size > contract["checkpoint_bytes_per_component"]:
            raise ValueError("staged checkpoint exceeded its declared byte bound")
        sizes.append(size)
    total = prefix + len(phases) * steps
    if getattr(model, "optimizer_steps_", None) != total:
        raise ValueError("staged total optimizer steps mismatch")
    costs = {stage: count * steps for stage, count in schedule}
    if algorithm == "genpu":
        costs.update(
            dict.fromkeys(
                (
                    "positive_discriminator",
                    "negative_discriminator",
                    "unlabeled_discriminator",
                    "positive_generator",
                    "negative_generator",
                ),
                params["max_epochs"] * steps,
            )
        )
        if model.stage_optimizer_steps_ != costs:
            raise ValueError("GEN-PU six-network paid optimizer budget mismatch")
    return {
        "schema_version": "survey-staged-budget-receipt-1",
        "budget_sha256": strict_canonical_hash(contract),
        "checkpoint_epoch_count": len(phases),
        "eligible_stages": list(contract["eligible_stages"]),
        "stage_optimizer_steps": costs,
        "optimizer_steps": total,
        "checkpoint_bytes": sizes,
        "formal_budget_approved": False,
    }
