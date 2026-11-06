"""Describe one experiment's sampling, parameter inputs and supervision roles."""

from __future__ import annotations

import copy
import inspect
from collections.abc import Mapping
from typing import Any


def finalize_research_settings(settings, candidate_runs) -> None:
    """Attach recorded candidate training facts to an experiment's settings."""
    settings["prior"]["candidate_training"] = [
        {
            "candidate_index": run["candidate_index"],
            "training_provenance": copy.deepcopy(run["training_provenance"]),
        }
        for run in candidate_runs
        if "training_provenance" in run
    ]
    settings["summary"] = _summarize(settings)


def _recorded_prior_fields(provenance, prefix=""):
    fields = {
        "class_prior",
        "unlabeled_positive_prior",
        "prior_source",
        "risk_pool",
        "population_prior",
        "risk_pool_positive_prior",
        "prior_semantics",
        "population_prior_forwarded_to_fit",
        "risk_class_prior",
    }
    if isinstance(provenance, Mapping):
        for name, value in provenance.items():
            path = f"{prefix}{name}"
            if name in fields:
                yield f"{path}={value}"
            elif isinstance(value, Mapping):
                yield from _recorded_prior_fields(value, f"{path}.")


def _summarize(settings):
    train = settings["labeling"]["train"]
    prior = settings["prior"]["training"]
    representation = settings["representation"]
    roles = ", ".join(
        f"{name}={role}" for name, role in settings["supervision"]["selection_roles"].items()
    )
    summary = (
        f"{str(settings['sampling']['input_sampling']).upper()} + "
        f"{train.get('mechanism', 'unspecified')} "
        f"({train.get('sampling_policy', 'unspecified')}; "
        f"c={train.get('c_requested', 'unspecified')}) + risk={settings['risk']['pool']} + "
        f"base prior input={prior['value']} ({prior['target']}; {prior['source']}) + "
        f"{roles} + {representation['training_path']} (ndim={representation['input_ndim']})"
    )
    for candidate in settings["prior"]["candidates"]:
        summary += (
            f" + candidate {candidate['candidate_index']} prior input={candidate['value']} "
            f"({candidate['source']})"
        )
        if "unlabeled_positive_prior" in candidate:
            summary += (
                f"; constructor unlabeled_positive_prior={candidate['unlabeled_positive_prior']}"
            )
    for candidate in settings["prior"].get("candidate_training", []):
        fields = ", ".join(_recorded_prior_fields(candidate["training_provenance"]))
        if fields:
            summary += f" + candidate {candidate['candidate_index']} recorded training: {fields}"
    return summary


def build_research_settings(
    *,
    model,
    trainer,
    config,
    context,
    generation,
    input_ndim,
    training_prior,
    selection_prior,
    selection_prior_source,
    selection_roles,
    label_view,
) -> dict[str, Any]:
    """Collect recorded facts and the parameter contract before candidate fitting."""
    parameters = model.get_params(deep=False)
    view_parameter = inspect.signature(model.fit).parameters.get("os_or_ts")
    default_view = view_parameter.default if view_parameter is not None else None
    if default_view is inspect.Parameter.empty:
        default_view = None
    requested_view = config.get("os_or_ts")
    effective_view = requested_view if requested_view is not None else default_view
    declared_target = getattr(model, "class_prior_target", None)
    pool = "estimator_defined"
    if label_view == "clean":
        pool = "supervised_labels"
    elif (
        declared_target == "native_unlabeled"
        and view_parameter is not None
        and effective_view in ("os", "ts")
    ):
        pool = "positive_union_unlabeled" if effective_view == "ts" else "native_unlabeled"
    target = declared_target
    if declared_target == "native_unlabeled" and effective_view == "ts":
        target = "risk_pool"
    prior_value = training_prior if training_prior is not None else parameters.get("class_prior")
    prior_source = (
        "runner.class_prior"
        if training_prior is not None
        else "estimator.class_prior"
        if prior_value is not None
        else "not_supplied"
    )
    training = {
        "value": prior_value,
        "target": target,
        "source": prior_source,
        "declared_target": declared_target,
    }
    candidate_inputs = []
    for index, candidate_parameters in enumerate(config.get("candidates", [{}])):
        candidate_prior = candidate_parameters.get("class_prior", parameters.get("class_prior"))
        candidate_source = (
            "candidate.class_prior"
            if "class_prior" in candidate_parameters
            else "estimator.class_prior"
        )
        candidate = {
            "candidate_index": index,
            "value": training_prior if training_prior is not None else candidate_prior,
            "source": (
                "runner.class_prior"
                if training_prior is not None
                else candidate_source
                if candidate_prior is not None
                else "not_supplied"
            ),
            "target": target,
        }
        for name in ("unlabeled_positive_prior", "prior_source"):
            if name in parameters or name in candidate_parameters:
                candidate[name] = copy.deepcopy(
                    candidate_parameters.get(name, parameters.get(name))
                )
        candidate_inputs.append(candidate)
    selection = {
        name: {"population": selection_prior, "source": selection_prior_source}
        for name in selection_roles
    }
    representation = copy.deepcopy(context.get("representation", {}))
    representation.update(
        {
            "input_ndim": input_ndim,
            "training_path": context.get(
                "training_path", config.get("training_path", "unspecified")
            ),
        }
    )
    for name in ("adapter_manifest", "image_manifest"):
        if name in config:
            representation[name] = copy.deepcopy(config[name])
    train_generation = generation["train"]
    sampling = {
        "input_sampling": train_generation.get("input_sampling", "unspecified"),
        "split_ref": copy.deepcopy(config.get("split_ref", {})),
    }
    settings = {
        "schema_version": "1.0",
        "sampling": sampling,
        "labeling": copy.deepcopy(generation),
        "risk": {
            "requested_view": requested_view,
            "estimator_default_view": default_view,
            "pool": pool,
        },
        "prior": {
            "training": training,
            "candidates": candidate_inputs,
            "selection": selection,
            "candidate_parameters": copy.deepcopy(config.get("candidates", [{}])),
        },
        "supervision": {
            "training_label_view": label_view,
            "trainer": type(trainer).__name__,
            "selection_roles": dict(selection_roles),
            "extra_supervision": copy.deepcopy(context.get("extra_supervision", {})),
        },
        "representation": representation,
    }
    settings["summary"] = _summarize(settings)
    return settings
