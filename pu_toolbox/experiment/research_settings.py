"""Describe one experiment's sampling, parameter inputs and supervision roles."""

from __future__ import annotations

import copy
import inspect
from typing import Any

PRIOR_CONSUMPTION_SCHEMA = "2.0"


def finalize_research_settings(settings) -> None:
    """Refresh the summary once actual prior consumption is known."""
    settings["summary"] = _summarize(settings)


def _summarize(settings):
    train = settings["labeling"]["train"]
    representation = settings["representation"]
    roles = ", ".join(
        f"{name}={role}" for name, role in settings["supervision"]["selection_roles"].items()
    )
    summary = (
        f"{str(settings['sampling']['input_sampling']).upper()} + "
        f"{train.get('mechanism', 'unspecified')} "
        f"({train.get('sampling_policy', 'unspecified')}; "
        f"c={train.get('c_requested', 'unspecified')}) + risk={settings['risk']['pool']} + "
    )
    summary += f"{roles} + {representation['training_path']} (ndim={representation['input_ndim']})"
    for row in settings["prior"]:
        owner = (
            f"candidate {row['candidate_index']}" if "candidate_index" in row else row["protocol"]
        )
        summary += (
            f" + {owner} {row['consumer']}={row['value']} "
            f"({row['partition']}/{row['target']}; {row['source']})"
        )
    return summary


def build_research_settings(
    *,
    model,
    trainer,
    config,
    context,
    generation,
    input_ndim,
    selection_roles,
    label_view,
) -> dict[str, Any]:
    """Collect experiment settings before candidate fitting."""
    view_parameter = inspect.signature(model.fit).parameters.get("os_or_ts")
    default_view = view_parameter.default if view_parameter is not None else None
    if default_view is inspect.Parameter.empty:
        default_view = None
    requested_view = config.get("os_or_ts")
    effective_view = requested_view if requested_view is not None else default_view
    risk_target = next(
        (use for use in getattr(model, "prior_consumers", ()) if use.ts_compatible_target), None
    )
    pool = "estimator_defined"
    if label_view == "clean":
        pool = "supervised_labels"
    elif risk_target is not None and view_parameter is not None and effective_view in ("os", "ts"):
        pool = risk_target.ts_compatible_target if effective_view == "ts" else risk_target.target
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
        "schema_version": PRIOR_CONSUMPTION_SCHEMA,
        "sampling": sampling,
        "labeling": copy.deepcopy(generation),
        "risk": {
            "requested_view": requested_view,
            "estimator_default_view": default_view,
            "pool": pool,
        },
        "prior": [],
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
