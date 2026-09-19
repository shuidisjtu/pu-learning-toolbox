"""Versioned pilot bindings; generic DIY runs remain explicitly technical.

Design notes: resolve the matrix again in the runner rather than trusting a
caller-supplied manifest. Budgets describe actual estimator training units;
unmatched backbones never acquire a shared oracle by annotation alone.
See docs/research/pu_survey/survey_execution_plan.md, P2.0a.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL_PATH = Path(__file__).with_name("survey_protocol_v1.json")
ROLES = ("train", "pu_val", "clean_val", "test")

#: One frozen ResNet-18 state_dict, rounded up from the 44 MB measured in
#: epoch_checkpoint_delivery.md -- which is where the 200-epoch budget's
#: 8-9 GB per candidate per seed comes from.
RESNET18_COMPONENT_BYTES = 45 * 1024**2
_LOCKED_CONFIG = ("backbone", "budget", "representation", "training_path", "comparability_group")
_REVIEW_STATUSES = frozenset({"pending_collaborator_review", "changes_requested", "accepted"})
_REQUIRED_UNIT_FIELDS = frozenset(
    {
        "dataset",
        "method",
        "training_path",
        "budget",
        "model_family",
        "backbone",
        "representation",
        "comparability_group",
        "oracle_alignment",
        "runnable",
    }
)


def unit_checkpoint_bytes(protocol: dict, row: dict, *, input_dim: int) -> int | None:
    """Bytes one per-epoch checkpoint component costs for this execution unit.

    ``None`` when the unit's budget caps no epochs, i.e. a closed-form or
    kernel method that never checkpoints.  The mlp figure is the declared
    architecture's parameter count at four bytes each; the image figure is a
    constant because the frozen ResNet-18 is fixed.  Both are lower bounds:
    they ignore filesystem overhead and any candidate that changes the network
    size.
    """
    if not protocol["budgets"][row["budget"]].get("epochs"):
        return None
    backbone = row["backbone"]
    # Rows name the image backbone by variant (end-to-end vs random-frozen)
    # while backbone_specs holds one shared "image" entry, so the family is
    # matched by prefix rather than by a spec key that does not exist.
    if backbone.startswith("resnet18"):
        return RESNET18_COMPONENT_BYTES
    hidden_dims = protocol["backbone_specs"].get(backbone, {}).get("hidden_dims")
    if not hidden_dims:
        return None
    dims = [int(input_dim), *(int(dim) for dim in hidden_dims), 1]
    # Each Linear contributes weight a*b plus its bias b.
    return 4 * sum(
        weights * units + units for weights, units in zip(dims[:-1], dims[1:], strict=True)
    )


def digest(value: Any) -> str:
    """Hash canonical JSON, independent of whitespace and key order."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def array_digest(value: np.ndarray) -> str:
    """Hash shape, dtype and bytes so shape-preserving corruption is detected."""
    value = np.ascontiguousarray(value)
    result = hashlib.sha256()
    result.update(str(value.dtype).encode())
    result.update(json.dumps(value.shape).encode())
    result.update(value.tobytes())
    return result.hexdigest()


def load_protocol(path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    """Load and validate the matrix before data preparation or training."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("unsupported survey protocol schema_version")
    for key in (
        "protocol_version",
        "budgets",
        "method_profiles",
        "backbone_specs",
        "c_tokens",
        "review_status",
        "seeds",
        "candidate_pool",
        "formal_blockers",
        "selection_spec",
        "selection_spec_kind",
    ):
        if key not in payload:
            raise ValueError(f"survey protocol missing {key}")
    if not isinstance(payload["protocol_version"], str) or not payload["protocol_version"]:
        raise ValueError("survey protocol protocol_version must be a non-empty string")
    for key in ("budgets", "method_profiles", "backbone_specs", "c_tokens", "selection_spec"):
        if not isinstance(payload[key], dict) or not payload[key]:
            raise ValueError(f"survey protocol {key} must be a non-empty object")
    if (
        not isinstance(payload["review_status"], str)
        or payload["review_status"] not in _REVIEW_STATUSES
    ):
        raise ValueError("survey protocol review_status is not recognized")
    if payload["selection_spec_kind"] != "descriptive_documentation":
        raise ValueError("selection_spec_kind must be descriptive_documentation")
    seeds = payload["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("survey protocol seeds must be unique integers")
    candidates = payload["candidate_pool"]
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(not isinstance(candidate, dict) for candidate in candidates)
    ):
        raise ValueError("survey protocol candidate_pool must be a non-empty object list")
    blockers = payload["formal_blockers"]
    if (
        not isinstance(blockers, list)
        or any(not isinstance(item, str) or not item for item in blockers)
        or len(blockers) != len(set(blockers))
    ):
        raise ValueError("survey protocol formal_blockers must be unique non-empty strings")
    units = payload.get("execution_units")
    if not isinstance(units, list) or not units:
        raise ValueError("survey protocol execution_units must not be empty")
    identities = set()
    for row in units:
        if not isinstance(row, dict):
            raise ValueError("survey execution unit must be an object")
        missing = _REQUIRED_UNIT_FIELDS - row.keys()
        if missing:
            raise ValueError(f"survey execution unit missing {', '.join(sorted(missing))}")
        for key in _REQUIRED_UNIT_FIELDS - {"runnable"}:
            if not isinstance(row[key], str) or not row[key]:
                raise ValueError(f"survey execution unit {key} must be a non-empty string")
        if type(row["runnable"]) is not bool:
            raise ValueError("survey execution unit runnable must be a JSON boolean")
        if row["runnable"] is False and (
            not isinstance(row.get("non_runnable_reason"), str)
            or not row["non_runnable_reason"].strip()
        ):
            raise ValueError("non-runnable survey unit requires non_runnable_reason")
        if row["runnable"] is True and "non_runnable_reason" in row:
            raise ValueError("runnable survey unit must not declare non_runnable_reason")
        identity = (row["dataset"], row["method"], row["training_path"])
        if identity in identities:
            raise ValueError(f"duplicate survey execution unit: {identity}")
        identities.add(identity)
        if row["budget"] not in payload["budgets"]:
            raise ValueError(f"unknown survey budget: {row['budget']}")
        if row["method"] not in payload["method_profiles"]:
            raise ValueError(f"unknown method profile: {row['method']}")
    digest(payload)  # also reject non-finite values
    return payload


def resolve_unit(
    protocol: dict, dataset: str, method: str, training_path: str | None = None
) -> dict:
    """Resolve exactly one runnable row, never silently fall back to MLP."""
    matches = [
        row
        for row in protocol["execution_units"]
        if row["dataset"] == dataset
        and row["method"] == method
        and (training_path is None or row["training_path"] == training_path)
    ]
    if len(matches) != 1:
        raise ValueError("survey unit missing or ambiguous; specify --training-path")
    if matches[0].get("runnable") is not True:
        raise ValueError(matches[0].get("non_runnable_reason", "survey unit is not runnable"))
    return copy.deepcopy(matches[0])


def validate_parameters(params: dict, profile: dict) -> None:
    """Reject candidate/constructor overrides of architecture and budget fields."""
    if not isinstance(params, dict):
        raise ValueError("survey model parameters and candidates must be objects")
    forbidden = {"model", "backbone", "encoder", "random_state", "hidden_layer_sizes", "device"}
    for name, value in params.items():
        root = name.split("__", 1)[0]
        if root in forbidden:
            raise ValueError(f"protocol-locked parameter cannot be overridden: {name}")
        if root in profile["params"] and (
            "__" in name or digest(value) != digest(profile["params"][root])
        ):
            raise ValueError(f"protocol-locked budget parameter cannot be overridden: {name}")


def _model_identity(model, method: str) -> None:
    if method == "pn_oracle":
        from .survey_execution import PilotOracleMLP

        expected = PilotOracleMLP
    else:
        from pu_toolbox.registry import get_algorithm, register_all_builtin_methods

        register_all_builtin_methods()
        expected = get_algorithm(method)
    if type(model) is not expected:
        raise ValueError(f"survey unit method={method!r} does not match estimator class")


def _validate_model(model, profile: dict, row: dict, seed: int) -> None:
    _model_identity(model, row["method"])
    params = model.get_params(deep=False)
    for name, value in profile["params"].items():
        if name not in params or digest(params[name]) != digest(value):
            raise ValueError(f"estimator disagrees with protocol-locked parameter: {name}")
    if "random_state" in params and params["random_state"] != seed:
        raise ValueError("estimator random_state must equal the survey run seed")
    if row["method"] in {"nnpu", "self_pu"}:
        from .survey_execution import validate_score_model

        network = params["model"] if row["method"] == "nnpu" else params["backbone"]
        validate_score_model(network, row["model_family"])
    if row["training_path"] == "native_cnn" and params.get("encoder") is None:
        raise ValueError("native_cnn survey unit requires the bound ResNet-18 encoder")


def _validate_budget(model, budget: dict) -> None:
    """Declared artifact budgets must describe the estimator that actually trains."""
    params = model.get_params(deep=False)
    if "optimizer_steps_per_epoch" in budget and (
        type(model).__name__ != "SelfPUClassifier" or budget["optimizer_steps_per_epoch"] != 2
    ):
        raise ValueError("two-student budget must describe two Self-PU updates per epoch")
    if "epochs" in budget:
        actual = params.get("max_epochs", params.get("epochs"))
        if actual != budget["epochs"]:
            raise ValueError("declared epoch budget does not match actual estimator epochs")
    for name in (
        "batch_size",
        "learning_rate",
        "max_acs_iter",
        "max_inner_iter",
        "n_em_iter",
        "max_iter",
    ):
        actual = params.get(name)
        if name == "learning_rate" and "optimizer" in params and params["optimizer"] is None:
            # nnPU's None branch constructs Adam(lr=1e-3), not a scalar parameter.
            actual = 1e-3
        if name in budget and not isinstance(budget[name], str) and actual != budget[name]:
            raise ValueError(f"declared budget does not match estimator {name}")
    if "internal_hyperparameter_pairs" in budget:
        actual = len(params["sigma_grid"]) * len(params["reg_grid"])
        if (
            actual != budget["internal_hyperparameter_pairs"]
            or params["cv"] != budget["internal_cv_folds"]
        ):
            raise ValueError("declared internal CV budget disagrees with estimator search")


def runner_protocol_context(model, bundle, config: dict, seed: int, generator, protocols) -> dict:
    """Revalidate the bound unit and derive honest manifest eligibility."""
    request = config.get("survey_protocol")
    if request is None:
        return {
            "execution_mode": "technical_smoke",
            "formal_eligible": False,
            "formal_blockers": ["no_versioned_survey_protocol"],
        }
    protocol = load_protocol(request["path"])
    row = resolve_unit(
        protocol, request["dataset"], request["method"], request.get("training_path")
    )
    profile = protocol["method_profiles"][row["method"]]
    for name in _LOCKED_CONFIG:
        if name in config:
            raise ValueError(f"use the execution matrix, not a config override for {name}")
    _validate_model(model, profile, row, seed)
    _validate_budget(model, protocol["budgets"][row["budget"]])
    expected_architecture = "cnn" if row["training_path"] == "native_cnn" else "mlp"
    if config.get("architecture") != expected_architecture:
        raise ValueError("architecture disagrees with the bound survey training_path")
    mechanism = request["mechanism"]
    if mechanism not in protocol["c_tokens"]:
        raise ValueError("unknown protocol labeling mechanism")
    expected_generator = {
        "scar": "SCARGenerator",
        "sar_lbe_a": "SARLBEAGenerator",
        "sar_lbe_b": "SARLBEBGenerator",
    }[mechanism]
    expected_names = {"PA", "OA"} if mechanism == "scar" else {"OA"}
    if row["method"] == "pn_oracle":
        expected_generator, expected_names = "CleanLabelGenerator", {"OA"}
    from .survey_execution import SourceSpaceGenerator

    actual_generator = (
        generator.base_generator if type(generator) is SourceSpaceGenerator else generator
    )
    if (
        type(actual_generator).__name__ != expected_generator
        or {p.name for p in protocols} != expected_names
    ):
        raise ValueError("generator/selection protocols disagree with survey execution unit")
    candidates = config.get("candidates", [{}])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("survey candidate pool must be a non-empty list")
    for candidate in candidates:
        validate_parameters(candidate, profile)
    deviations = []
    if digest(protocol) != digest(load_protocol()):
        deviations.append("noncanonical_protocol_matrix")
    if candidates != protocol["candidate_pool"]:
        deviations.append("candidate_pool")
    if seed not in protocol["seeds"]:
        deviations.append("seed")
    if request.get("seeds") != protocol["seeds"]:
        deviations.append("seed_subset")
    if request.get("c_tokens") != protocol["c_tokens"][mechanism]:
        deviations.append("c_grid")
    if row["method"] != "pn_oracle":
        token = config.get("c_requested_token")
        if token not in protocol["c_tokens"][mechanism] or float(token) != config.get("c"):
            deviations.append("c")
    split_ref = config.get("split_ref", {})
    if split_ref.get("dataset") != row["dataset"] or split_ref.get("seed") != seed:
        deviations.append("split_dataset_or_seed")
    if request.get("split_ref_overridden"):
        deviations.append("split_ref_override")
    if any(key != "class_prior" for key in request.get("constructor_overrides", {})):
        deviations.append("constructor_override")
    representation = {
        "name": row["representation"],
        "feature_sha256": {role: array_digest(getattr(bundle, role).X) for role in ROLES},
        "split_sha256": digest(
            {role: np.asarray(getattr(bundle, role).indices).tolist() for role in ROLES}
        ),
    }
    adapter = config.get("adapter_manifest")
    image = config.get("image_manifest")
    if row["dataset"] == "cifar10":
        if not image or image.get("initialization_seed") != seed:
            raise ValueError("CIFAR unit requires seeded image preprocessing provenance")
        spec = protocol["backbone_specs"]["image"]
        if (
            image.get("input_size") != spec["input_size"]
            or image["backbone"]["weights"] is not None
        ):
            raise ValueError("image manifest disagrees with locked size/initialization")
        representation["image_preprocessing"] = copy.deepcopy(image)
        if row["training_path"] == "native_cnn":
            from .feature_adapter import _encoder_state_sha256

            if _encoder_state_sha256(model.encoder) != image["encoder_state_sha256"]:
                raise ValueError("native CNN encoder state does not match its image manifest")
    if row["training_path"] == "cnn_feature_adapter":
        if not adapter or adapter.get("training_path") != "cnn_feature_adapter":
            raise ValueError("cnn_feature_adapter unit requires its actual adapter manifest")
        if adapter.get("backbone_manifest") != image:
            raise ValueError("adapter and image preprocessing provenance disagree")
        if adapter["feature_sha256"] != representation["feature_sha256"]:
            # Adapter uses its own array hash encoding; recompute through that same helper.
            from .feature_adapter import _array_sha256

            if adapter["feature_sha256"] != {
                role: _array_sha256(getattr(bundle, role).X) for role in ROLES
            }:
                raise ValueError("adapter manifest does not match the actual input features")
        representation["adapter"] = copy.deepcopy(adapter)
    blockers = list(protocol["formal_blockers"])
    if protocol["review_status"] != "accepted" and "collaborator_review" not in blockers:
        blockers.append("collaborator_review")
    if "PA" in expected_names:
        blockers.append("PA_separation_proxy_not_preregistered_accuracy_threshold")
    if row["method"] == "self_pu":
        blockers.append("SelfPU_clean_validation_meta_reweighting_OA_integration")
    if deviations:
        blockers.append("protocol_deviation")
    return {
        "execution_mode": "versioned_pilot",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": digest(protocol),
        "execution_unit": row,
        "adaptation_level": "benchmark-adapted",
        "backbone": {"name": row["backbone"], "specs": protocol["backbone_specs"]},
        "budget": {
            **protocol["budgets"][row["budget"]],
            "actual_outer_candidates": len(candidates),
        },
        "representation": representation,
        "training_path": row["training_path"],
        "comparability_group": row["comparability_group"],
        "protocol_deviation": deviations,
        "formal_eligible": not blockers,
        "formal_blockers": blockers,
        "selection_checkpoint_scope": "pending_actual_trajectory_verification",
        "oracle_alignment": row["oracle_alignment"],
        "method_variant": "without_clean_validation_meta_reweighting"
        if row["method"] == "self_pu"
        else "shared_spec_engineering",
    }


def budget_fairness_fields(budget: dict) -> dict[str, Any]:
    """Reduce a budget to the values the fairness gates compare.

    The group key separates datasets, not budget families: the classical group
    holds a closed-form solve, an EM fit, an internal-CV fit and an alternating
    solver, which share no budget key at all.  Comparing the raw dictionaries
    would refuse the rows the protocol lists side by side, so the comparison
    rests on what the gates consume -- the epoch cap and the batch-size
    candidate set.

    A field the family does not number reports the placeholder ``1``.  That is
    safe because these values are only ever compared within one comparability
    group, where every member is subject to the same placeholder.
    """
    return {
        "max_epochs": _positive_int(budget.get("epochs")),
        "batch_size_candidates": (_positive_int(budget.get("batch_size")),),
    }


def _positive_int(value: Any) -> int:
    """``value`` when it is a positive integer, and the placeholder otherwise.

    A family may cap no epochs at all (a closed-form solve has none) or describe
    a field instead of numbering it (the fullbatch family records that the
    estimator's batch size is unused).  Both have to reach the gate as a
    positive int, so both take the placeholder rather than the gate learning a
    second accepted shape.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return 1
    return value


def validate_comparable_manifests(manifests: list[dict], *, require_formal: bool = True) -> None:
    """Fail closed before mixing paths, protocols, representations or budgets.

    This validates one dataset/seed/c selection group, not a cross-dataset
    leaderboard. Technical diagnostics must opt out of the formal gate.
    """
    if not manifests:
        raise ValueError("comparison requires at least one manifest")
    if require_formal and any(not item.get("formal_eligible", False) for item in manifests):
        raise ValueError("formal aggregation blocked; inspect formal_blockers")
    expected = manifests[0]
    expected_budget = budget_fairness_fields(expected.get("budget", {}))
    for item in manifests[1:]:
        for key in (
            "protocol_version",
            "protocol_sha256",
            "seed",
            "training_path",
            "comparability_group",
            "backbone",
        ):
            if item.get(key) != expected.get(key):
                raise ValueError(f"comparison mismatch: {key}")
        # Compared through the fields the fairness gates consume rather than as
        # a dictionary: one comparability group can hold several budget families.
        item_budget = budget_fairness_fields(item.get("budget", {}))
        for field, value in item_budget.items():
            if value != expected_budget[field]:
                raise ValueError(f"comparison mismatch: budget.{field}")
        for key in ("split_sha256", "feature_sha256"):
            if item["representation"][key] != expected["representation"][key]:
                raise ValueError(f"comparison representation mismatch: {key}")
        left, right = item.get("generation", {}), expected.get("generation", {})
        for role in ("train", "pu_val"):
            if (
                left.get(role, {}).get("mechanism") != "pn_oracle"
                and right.get(role, {}).get("mechanism") != "pn_oracle"
            ):
                for key in ("mechanism", "c_requested", "label_view_sha256"):
                    if left.get(role, {}).get(key) != right.get(role, {}).get(key):
                        raise ValueError(f"comparison label-view mismatch: {key}")
