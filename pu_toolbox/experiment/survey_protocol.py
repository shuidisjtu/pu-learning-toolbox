"""Versioned pilot bindings; generic DIY runs remain explicitly technical.

Design notes: resolve the matrix again in the runner rather than trusting a
caller-supplied manifest. Budgets describe actual estimator training units;
unmatched backbones never acquire a shared oracle by annotation alone.
See docs/research/pu_survey/survey_execution_plan.md, P2.0a.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from pu_toolbox.core.training_views import ROLES
from pu_toolbox.utils.serialization import array_hash, json_scalars, strict_canonical_hash

from .datasets import survey_dataset_catalog
from .feature_adapter import encoder_state_sha256
from .survey_profile_identity import (
    has_profile_identities,
    profile_algorithm,
    profile_manifest_identity,
    require_four_role_profile,
    validate_profile_identities,
)

PROTOCOL_PATH = Path(__file__).with_name("survey_protocol_v2.json")
#: The canonical matrix files.  A version bump adds a file and keeps the old one:
#: dated P3 evidence and the archived comparison matrices pin the bytes they were
#: prepared against, so re-reading the old version must stay possible.  Names
#: resolve to the file carrying that version -- a command written against
#: survey-v1.2 keeps re-running the v1.2 matrix rather than silently moving to
#: the current one.  A caller that resolves a name and a caller that resolves a
#: path have to agree about which matrix a version is, which is why this mapping
#: has one home.
PROTOCOL_PATHS: dict[str, Path] = {
    "survey-v1": Path(__file__).with_name("survey_protocol_v1.json"),
    "survey-v1.1": Path(__file__).with_name("survey_protocol_v1.json"),
    "survey-v1.2": Path(__file__).with_name("survey_protocol_v1.json"),
    "survey-v2.0": PROTOCOL_PATH,
    "survey-v3.0": Path(__file__).with_name("survey_protocol_v3.json"),
}
PROTOCOL_ALIASES = tuple(PROTOCOL_PATHS)


def resolve_protocol_path(value: str | Path) -> Path:
    """The matrix file *value* denotes: a shipped name, or the path it gives."""
    name = str(value)
    if name in PROTOCOL_PATHS:
        return PROTOCOL_PATHS[name]
    return Path(value).resolve()


#: One end-to-end ResNet-18 checkpoint component, rounded up from the 44 MB
#: measured in epoch_checkpoint_delivery.md -- which is where the 200-epoch
#: budget's 8-9 GB per candidate per seed comes from.  It sizes the rows that
#: train the ResNet itself; the adapter rows train a head on frozen features and
#: are sized by ADAPTER_HEAD_COMPONENT_BYTES instead.
RESNET18_COMPONENT_BYTES = 45 * 1024**2

#: One trainable MLP head under the frozen-encoder adapter path.  Measured
#: 2026-09-26 by serialising every runnable adapter row's real per-epoch
#: checkpoint through the production path (assemble_model with the shipped
#: mlp128 score model, written by EpochCheckpointTrainer): the largest component
#: file was 265,695 bytes at cifar10/self_pu, so 512 KiB leaves a 1.97x margin.
#: The frozen encoder is not in it -- the writer saves ``fitted.model_``, which
#: on this path is the head, not the features it reads.
#:
#: tests/unit/experiment/test_checkpoint_storage_profiles.py re-runs that
#: serialisation, so a head that outgrows this bound fails rather than quietly
#: keeping a stale constant plausible.
ADAPTER_HEAD_COMPONENT_BYTES = 512 * 1024

#: Storage profiles whose per-component size is a *constant*.  A constant is
#: evidence for the architecture it was measured on and for nothing else, and the
#: key is the whole identity because the method is not part of it: two methods
#: sharing a training path, backbone and model family share a head.  So a new
#: triple has to add a row here, and any change to the architecture behind an
#: existing row has to re-measure it -- either way with its own serialisation
#: evidence, rather than silently inheriting a bound that no longer covers it.
#: (A new method on an existing triple needs no new row, but the serialisation
#: test is parametrised over the matrix, so its head is measured all the same.)
_CONSTANT_COMPONENT_PROFILES: dict[tuple[str, str, str], tuple[int, str]] = {
    ("cnn_feature_adapter", "resnet18_random_frozen", "mlp128"): (
        ADAPTER_HEAD_COMPONENT_BYTES,
        "adapter_trainable_head",
    ),
    ("native_cnn", "resnet18_end_to_end", "resnet18_linear"): (
        RESNET18_COMPONENT_BYTES,
        "native_cnn_full_resnet18",
    ),
}
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


def _profile_key(row: dict) -> tuple[str, object, object]:
    """The identity a constant profile is registered under.

    ``backbone`` and ``model_family`` are read leniently so a row that omits one
    fails as an unsized row with its context spelled out, rather than as a
    ``KeyError`` from the lookup.  The loader requires both on every shipped
    unit, and a missing field can never match a registered profile -- which is
    the fail-closed direction either way.
    """
    return (row["training_path"], row.get("backbone"), row.get("model_family"))


def _unsized_message(row: dict) -> str:
    """Name every field a reader would need to add the missing profile."""
    return (
        f"no checkpoint storage profile for {row['dataset']}/{row['method']}/"
        f"{row['training_path']} (backbone={row.get('backbone')!r}, "
        f"model_family={row.get('model_family')!r}, budget={row['budget']!r})"
    )


def _checkpoint_storage(protocol: dict, row: dict, *, input_dim: int) -> tuple[int | None, str]:
    """Bytes one per-epoch checkpoint component costs, and the profile behind it.

    The single place the training path is decided.  ``None`` means this budget
    caps no epochs -- a closed-form or kernel method that never checkpoints --
    and is deliberately *not* how "unknown" is spelled: an unrecognised path or
    architecture raises, because a silently guessed size is a figure the pre-run
    guard would then hold a host to.
    """
    from .survey_staged_budget import FIELD, checkpoint_epochs, validate_contract

    budget = protocol["budgets"][row["budget"]]
    declared = budget.get("checkpoint_bytes_per_component")
    if type(declared) is int and declared > 0:
        # Methods that checkpoint more than the bare MLP (two teacher networks, a
        # fitted meta head) declare the measured per-component bound themselves;
        # the parametric MLP estimate below would under-count them.
        return declared, "explicit_budget_bound"
    if FIELD in budget:
        validate_contract(budget[FIELD])
        return budget[FIELD]["checkpoint_bytes_per_component"], "explicit_staged_budget_bound"
    if not checkpoint_epochs(budget):
        return None, "no_epoch_budget"
    profile = _CONSTANT_COMPONENT_PROFILES.get(_profile_key(row))
    if profile is not None:
        return profile
    if row["training_path"] == "native_2d":
        # Parametric rather than constant: the figure follows the row's own
        # declared architecture, so a wider network is covered by construction.
        hidden_dims = protocol["backbone_specs"].get(row.get("backbone"), {}).get("hidden_dims")
        if not hidden_dims:
            raise ValueError(_unsized_message(row))
        dims = [int(input_dim), *(int(dim) for dim in hidden_dims), 1]
        # Each Linear contributes weight a*b plus its bias b.
        return (
            4 * sum(w * u + u for w, u in zip(dims[:-1], dims[1:], strict=True)),
            "native_2d_mlp",
        )
    raise ValueError(
        f"{_unsized_message(row)}. Add a profile for this training path, backbone "
        "and model family, with its own serialisation evidence."
    )


def unit_checkpoint_bytes(protocol: dict, row: dict, *, input_dim: int) -> int | None:
    """Bytes one per-epoch checkpoint component costs for this execution unit.

    ``None`` when the unit's budget caps no epochs, i.e. a closed-form or
    kernel method that never checkpoints.  Which model applies is decided by the
    unit's training path, backbone and model family together -- the row's
    ``backbone`` names where the features come from, not what gets saved, and on
    the adapter path the two differ by three orders of magnitude.

    The two kinds of figure bound in **opposite** directions, and which one a
    row gets decides how its total should be read.  The MLP figure is the
    declared architecture's parameter count at four bytes each, so it is a
    *lower* bound: it ignores serialisation overhead and any candidate that
    changes the network size.  The constants are rounded up from real
    serialisations of the components they cover, so they are conservative
    *upper* bounds -- a host sized against one is not going to run out of room
    because a header was longer than expected.
    """
    return _checkpoint_storage(protocol, row, input_dim=input_dim)[0]


def unit_checkpoint_profile(protocol: dict, row: dict, *, input_dim: int) -> str:
    """Which storage profile decided that figure.

    Same call and same signature as ``unit_checkpoint_bytes``, so a report that
    explains the number cannot end up describing a different path than the
    number came from.
    """
    return _checkpoint_storage(protocol, row, input_dim=input_dim)[1]


def digest(value: Any) -> str:
    """Hash canonical JSON, independent of whitespace and key order.

    The artifact-identity digest: it delegates to
    :func:`~pu_toolbox.utils.serialization.strict_canonical_hash` so that the
    survey layer cannot drift from the one strict recipe.
    """
    return strict_canonical_hash(value)


def array_digest(value: np.ndarray) -> str:
    """Hash shape, dtype and bytes so shape-preserving corruption is detected.

    Kept as a named entry point because ``survey_execution`` imports it; the
    recipe itself is :func:`~pu_toolbox.utils.serialization.array_hash`.
    """
    return array_hash(value)


#: Fields a dataset may override on the shared image spec.  Everything else (the network,
#: its initialisation, normalisation, augmentation) is one choice for the whole protocol.
_PER_DATASET_IMAGE_FIELDS = frozenset({"in_channels", "input_size", "small_input_stem"})


def _validate_image_spec(backbone_specs: dict[str, Any]) -> None:
    """Reject a malformed ``per_dataset`` block; a flat spec (v1) has none and passes."""
    per_dataset = backbone_specs.get("image", {}).get("per_dataset")
    if per_dataset is None:
        return
    if not isinstance(per_dataset, dict) or not per_dataset:
        raise ValueError("survey protocol image per_dataset must be a non-empty object")
    for dataset, override in per_dataset.items():
        if not isinstance(override, dict) or not override:
            raise ValueError(f"survey protocol image per_dataset[{dataset!r}] must be an object")
        unknown = sorted(set(override) - _PER_DATASET_IMAGE_FIELDS)
        if unknown:
            raise ValueError(
                f"survey protocol image per_dataset[{dataset!r}] may only override "
                f"{sorted(_PER_DATASET_IMAGE_FIELDS)}, not {unknown}"
            )


def image_spec(protocol: dict[str, Any], dataset: str) -> dict[str, Any]:
    """The image backbone spec that applies to *dataset*.

    A flat spec (the shipped v1 matrix, written for CIFAR-10) applies to every dataset as
    it is.  A spec with ``per_dataset`` merges that dataset's overrides over the shared
    fields and has no spec at all for a dataset it does not list, so a dataset cannot
    silently inherit another one's input size.
    """
    spec = protocol["backbone_specs"]["image"]
    per_dataset = spec.get("per_dataset")
    if per_dataset is None:
        return copy.deepcopy(spec)
    if dataset not in per_dataset:
        raise ValueError(f"survey protocol has no image specification for dataset {dataset!r}")
    shared = {key: value for key, value in spec.items() if key != "per_dataset"}
    return copy.deepcopy({**shared, **per_dataset[dataset]})


def _validate_review_items(items: Any) -> None:
    """Reject a malformed review ledger; an absent ledger is the v1 shape.

    ``review_items`` records, per changed part of a protocol revision, whether it
    inherits acceptance or still waits for collaborator review.  The top-level
    ``review_status`` cannot express that mixture, so a revision that changes
    only some sections keeps ``accepted`` and lists the exceptions here.
    """
    if items is None:
        return
    if not isinstance(items, list) or not items:
        raise ValueError("survey protocol review_items must be a non-empty object list")
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("survey protocol review_items must be an object list")
        for key in ("item_id", "title", "origin"):
            if not isinstance(item.get(key), str) or not item[key]:
                raise ValueError(f"survey protocol review item missing {key}")
        if item["item_id"] in seen:
            raise ValueError(f"duplicate survey protocol review item: {item['item_id']}")
        seen.add(item["item_id"])
        sections = item.get("sections")
        if (
            not isinstance(sections, list)
            or not sections
            or any(not isinstance(section, str) or not section for section in sections)
        ):
            raise ValueError("survey protocol review item sections must be non-empty strings")
        if item.get("status") not in _REVIEW_STATUSES:
            raise ValueError("survey protocol review item status is not recognized")


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
    validate_profile_identities(payload)
    _validate_image_spec(payload["backbone_specs"])
    if (
        not isinstance(payload["review_status"], str)
        or payload["review_status"] not in _REVIEW_STATUSES
    ):
        raise ValueError("survey protocol review_status is not recognized")
    _validate_review_items(payload.get("review_items"))
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
    from .survey_staged_budget import validate_protocol_budgets

    validate_protocol_budgets(payload)
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


def bind_runtime_parameters(
    method: str, params: dict, profile: dict, *, mechanism: str, c_realized: float
) -> dict:
    """Bind explicitly declared mechanism-dependent constructor parameters."""
    allowed_mechanisms = profile.get("labeling_mechanisms")
    if allowed_mechanisms is not None and mechanism not in allowed_mechanisms:
        raise ValueError(
            f"method {method!r} supports labeling mechanisms {allowed_mechanisms}, "
            f"not {mechanism!r}"
        )

    bound = dict(params)
    bindings = profile.get("runtime_parameters", {})
    if not isinstance(bindings, dict):
        raise ValueError(f"method profile {method!r} runtime_parameters must be an object")
    for name, binding in bindings.items():
        if name in bound:
            raise ValueError(f"runtime-bound parameter cannot be overridden: {name}")
        if binding != {
            "source": "c_realized",
            "transform": "one_minus",
            "mechanism": "scar",
        }:
            raise ValueError(f"unsupported runtime binding for {method}.{name}")
        if mechanism != "scar":
            raise ValueError(f"runtime binding {method}.{name} is defined for SCAR only")
        if not 0.0 < float(c_realized) < 1.0:
            raise ValueError("SCAR c must be in (0, 1) for the runtime parameter binding")
        bound[name] = 1.0 - float(c_realized)
    return bound


def _model_identity(model, method: str, training_path: str) -> None:
    if method == "pn_oracle":
        from .survey_execution import PilotOracleCNN, PilotOracleMLP

        # PilotOracleCNN subclasses PilotOracleMLP, so the exact-type check below needs the
        # class the training path asks for; one expectation for both rejects every CNN oracle.
        expected = PilotOracleCNN if training_path == "native_cnn" else PilotOracleMLP
    else:
        from pu_toolbox.registry import get_algorithm, register_all_builtin_methods

        register_all_builtin_methods()
        expected = get_algorithm(method)
    if type(model) is not expected:
        raise ValueError(f"survey unit method={method!r} does not match estimator class")


def _validate_model(model, profile: dict, row: dict, seed: int) -> None:
    algorithm = profile.get("algorithm", row["method"])
    _model_identity(model, algorithm, row["training_path"])
    params = model.get_params(deep=False)
    for name, value in profile["params"].items():
        if name not in params or digest(params[name]) != digest(value):
            raise ValueError(f"estimator disagrees with protocol-locked parameter: {name}")
    if "random_state" in params and params["random_state"] != seed:
        raise ValueError("estimator random_state must equal the survey run seed")
    if algorithm in {"nnpu", "self_pu"}:
        from .survey_execution import validate_score_model

        network = params["model"] if algorithm == "nnpu" else params["backbone"]
        validate_score_model(network, row["model_family"])
    if row["training_path"] == "native_cnn" and params.get("encoder") is None:
        raise ValueError("native_cnn survey unit requires the bound ResNet-18 encoder")


def _validate_budget(model, budget: dict) -> None:
    """Declared artifact budgets must describe the estimator that actually trains."""
    from .survey_staged_budget import FIELD, validate_estimator

    if FIELD in budget:
        validate_estimator(model, budget[FIELD])
        return
    params = model.get_params(deep=False)
    if "optimizer_steps_per_epoch" in budget and (
        type(model).__name__ != "SelfPUClassifier" or budget["optimizer_steps_per_epoch"] != 2
    ):
        raise ValueError("two-student budget must describe two Self-PU updates per epoch")
    if "epochs" in budget:
        actual = params.get("max_epochs", params.get("epochs", params.get("episodes")))
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
    require_four_role_profile(
        protocol, row["method"], clean_support=bool(config.get("clean_support_provided"))
    )
    algorithm = profile_algorithm(protocol, row["method"])
    for name in _LOCKED_CONFIG:
        if name in config:
            raise ValueError(f"use the execution matrix, not a config override for {name}")
    expected_architecture = "cnn" if row["training_path"] == "native_cnn" else "mlp"
    if config.get("architecture") != expected_architecture:
        raise ValueError("architecture disagrees with the bound survey training_path")
    mechanism = request["mechanism"]
    if mechanism not in protocol["c_tokens"]:
        raise ValueError("unknown protocol labeling mechanism")
    if (
        profile.get("labeling_mechanisms") is not None
        and mechanism not in profile["labeling_mechanisms"]
    ):
        raise ValueError(f"method {row['method']!r} does not support mechanism {mechanism!r}")
    _validate_model(model, profile, row, seed)
    _validate_budget(model, protocol["budgets"][row["budget"]])
    token = config.get("c_requested_token")
    c_off_grid = row["method"] != "pn_oracle" and (
        token not in protocol["c_tokens"][mechanism] or float(token) != config.get("c")
    )
    binding_inputs = config.get("runtime_binding_inputs", {})
    if profile.get("runtime_parameters"):
        from .strategies import scar_realized_rate

        expected_c_realized = scar_realized_rate(bundle.train.labels, float(config["c"]))
        if binding_inputs != {"c_realized": expected_c_realized}:
            raise ValueError("runtime binding inputs disagree with fixed-count SCAR generation")
    elif binding_inputs:
        raise ValueError("unexpected runtime binding inputs for this method")
    runtime_expected = (
        bind_runtime_parameters(
            row["method"],
            request.get("constructor_overrides", {}),
            profile,
            mechanism=mechanism,
            c_realized=float(binding_inputs.get("c_realized", config.get("c"))),
        )
        if row["method"] != "pn_oracle"
        else {}
    )
    runtime_bindings_expected = {
        name: runtime_expected[name] for name in profile.get("runtime_parameters", {})
    }
    for name in profile.get("runtime_parameters", {}):
        if name not in model.get_params(deep=False) or digest(
            model.get_params(deep=False)[name]
        ) != digest(runtime_expected[name]):
            raise ValueError(f"estimator disagrees with runtime-bound parameter: {name}")
    if (
        row["method"] != "pn_oracle"
        and config.get("runtime_parameters", {}) != runtime_bindings_expected
    ):
        raise ValueError("manifest runtime parameters disagree with the mechanism binding")
    expected_generator = {
        "scar": "SCARGenerator",
        "sar_lbe_a": "SARLBEAGenerator",
        "sar_lbe_b": "SARLBEBGenerator",
    }[mechanism]
    if mechanism == "scar":
        # The audited extra-support path names its PA protocol separately, so the
        # unit records PA+clean-support instead of plain PA.
        expected_names = (
            {"PA+clean-support", "OA"} if config.get("clean_support_provided") else {"PA", "OA"}
        )
    else:
        expected_names = {"OA"}
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
    if c_off_grid:
        deviations.append("c")
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
            {
                role: json_scalars(
                    np.asarray(getattr(bundle, role).indices), name=f"{role} indices"
                )
                for role in ROLES
            }
        ),
    }
    adapter = config.get("adapter_manifest")
    image = config.get("image_manifest")
    if survey_dataset_catalog()[row["dataset"]].modality == "image":
        if not image or image.get("initialization_seed") != seed:
            raise ValueError("image unit requires seeded image preprocessing provenance")
        spec = image_spec(protocol, row["dataset"])
        if (
            image.get("input_size") != spec["input_size"]
            or image["backbone"]["weights"] is not None
        ):
            raise ValueError("image manifest disagrees with locked size/initialization")
        representation["image_preprocessing"] = copy.deepcopy(image)
        if row["training_path"] == "native_cnn":
            actual_encoder = encoder_state_sha256(model.encoder)
            declared_architecture = (
                protocol["backbone_specs"].get(row.get("backbone"), {}).get("architecture")
            )
            if actual_encoder != image["encoder_state_sha256"]:
                # GradPU trains a method-private CNN13 rather than the shared
                # ResNet-18; record that encoder instead of claiming the shared one.
                if declared_architecture != "cnn13_no_bn":
                    raise ValueError("native CNN encoder state does not match its image manifest")
                representation["method_private_encoder_state_sha256"] = actual_encoder
    if row["training_path"] == "cnn_feature_adapter":
        if not adapter or adapter.get("training_path") != "cnn_feature_adapter":
            raise ValueError("cnn_feature_adapter unit requires its actual adapter manifest")
        if adapter.get("backbone_manifest") != image:
            raise ValueError("adapter and image preprocessing provenance disagree")
        if adapter["feature_sha256"] != representation["feature_sha256"]:
            raise ValueError("adapter manifest does not match the actual input features")
        representation["adapter"] = copy.deepcopy(adapter)
    blockers = list(protocol["formal_blockers"])
    if has_profile_identities(protocol):
        blockers.append("profile_identity_extension_engineering_only")
    if protocol["review_status"] != "accepted" and "collaborator_review" not in blockers:
        blockers.append("collaborator_review")
    # Three blockers used to be appended here for protocol reasons.  D24
    # released the last two on the same single-party technical acceptance that
    # D12/D20 used: the PA criterion itself is implemented (see
    # p2_0a_review.md R9), and the pilot's ``self_pu`` scope is the no-meta
    # ablation -- which ``method_variant`` below still records.  The calibrated
    # view's ``ts_view_collaborator_review`` went the same way under D20.
    #
    # Removing a blocker from formal eligibility is NOT a collaborator review:
    # reports must not read these units as signed off.  The standing caveat
    # lives in the decision records, the method ledger's uncertainty notes and
    # the method cards.  The PA mappings are no longer blocked in the matrix:
    # D41 adjudicated all 54 as no_direct_anchor in survey-comparison-v5, which
    # records that no report anchors them rather than that a criterion is owed.
    if deviations:
        blockers.append("protocol_deviation")
    context = {
        "execution_mode": "versioned_pilot",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": digest(protocol),
        "execution_unit": row,
        "runtime_binding_inputs": binding_inputs,
        "runtime_parameters": runtime_bindings_expected,
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
        if algorithm == "self_pu"
        else "shared_spec_engineering",
    }
    context.update(profile_manifest_identity(protocol, row["method"]))
    if has_profile_identities(protocol):
        context.update(_profile_registration_context(protocol, row["method"]))
    if "review_items" in protocol:
        context["protocol_pending_review_items"] = [
            item["item_id"] for item in protocol["review_items"] if item["status"] != "accepted"
        ]
    return context


def _profile_registration_context(protocol: dict[str, Any], profile_id: str) -> dict[str, Any]:
    """The profile's comparison registration, or the honest "no mapping" marker.

    A missing map and a matrix the map was not frozen against (a narrowed draft,
    or the canonical file after an unreviewed edit) both fall back to
    ``missing_versioned_profile_mapping``: in neither case does a mapping describe
    this matrix.  A map that *does* bind the matrix but omits this profile is a
    defect and is raised, so "not mapped yet" cannot survive the file landing.
    """
    from .survey_profile_map import (
        load_profile_map,
        profile_map_available,
        profile_map_binds,
        profile_registration_status,
    )

    if not profile_map_available():
        return {"comparison_registration_status": "missing_versioned_profile_mapping"}
    profile_map = load_profile_map()
    if not profile_map_binds(profile_map, protocol):
        return {"comparison_registration_status": "missing_versioned_profile_mapping"}
    return profile_registration_status(profile_id, protocol=protocol, profile_map=profile_map)


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
    from .survey_staged_budget import FIELD, validate_contract

    if FIELD in budget:
        contract = budget[FIELD]
        validate_contract(contract)
        params = contract["parameter_values"]
        # The legacy gate projects one batch knob. PULDA iterates over U;
        # its P/U pair is one setting, not two search candidates. The detailed
        # staged comparison below checks *both* knobs and all paid stages.
        batches = (
            (params["unlabeled_batch_size"],)
            if contract["algorithm_id"] == "pulda"
            else (params["batch_size"],)
        )
        return {"max_epochs": contract["checkpoint_epoch_count"], "batch_size_candidates": batches}
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
    from .survey_staged_budget import FIELD, validate_contract

    for item in manifests[1:]:
        left = expected.get("budget", {}).get(FIELD)
        right = item.get("budget", {}).get(FIELD)
        if (left is None) != (right is None):
            raise ValueError("comparison mismatch: staged and legacy budgets cannot be mixed")
        if left is not None:
            validate_contract(right)
            # Equal snapshot caps do not imply equal paid GAN/episode work or
            # stopping rules. Byte bounds are resource guards, not training knobs.
            for field in ("schema_version", "algorithm_id", "parameter_values"):
                if digest(left[field]) != digest(right[field]):
                    raise ValueError(f"comparison mismatch: staged budget.{field}")
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
