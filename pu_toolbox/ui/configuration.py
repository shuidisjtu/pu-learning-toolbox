"""Bridge portable run configurations to Streamlit session state."""

from __future__ import annotations

import json
from collections.abc import MutableMapping
from typing import Any

from pu_toolbox.run_config import RunConfiguration


def parse_json_mapping(text: str, *, field_name: str) -> dict[str, Any]:
    """Parse a JSON object used by the advanced parameter editors."""
    try:
        value = json.loads(text or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{field_name} is not valid JSON: line {exc.lineno}, column {exc.colno}."
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError(f"{field_name} keys must be non-empty strings.")
    return value


def apply_run_configuration(
    state: MutableMapping[str, Any],
    config: RunConfiguration,
    catalog_by_name: dict[str, dict[str, Any]],
) -> None:
    """Populate widget state from a validated portable configuration."""
    state["selection_mode"] = (
        "比较模型"
        if config.comparison_classifiers
        else "自动推荐"
        if config.classifier == "auto"
        else "手动选择"
    )
    state["comparison_classifiers"] = list(config.comparison_classifiers)
    if config.classifier != "auto":
        state["classifier"] = config.classifier
    state["cv"] = config.cv
    state["seed"] = config.random_state
    state["metrics"] = list(config.metrics)
    state["backbone"] = config.backbone
    if config.class_prior is not None:
        state["prior_method"] = "手动输入"
        state["class_prior"] = config.class_prior
    elif config.prior_estimator is None:
        state["prior_method"] = "不使用"
    else:
        state["prior_method"] = "自动估计"
        state["prior_estimator"] = config.prior_estimator
    state["tuning_enabled"] = bool(config.tuning_grid)
    state["tuning_text"] = json.dumps(
        config.tuning_grid, ensure_ascii=False, indent=2, sort_keys=True
    )
    state["scoring"] = config.scoring
    state["comparison_scoring"] = config.scoring

    typed_names: set[str] = set()
    if config.classifier in catalog_by_name:
        for item in catalog_by_name[config.classifier]["parameters"]:
            name = item["name"]
            if name not in config.classifier_params or item["kind"] == "json":
                continue
            typed_names.add(name)
            state[f"param_{config.classifier}_{name}"] = config.classifier_params[name]
            if item["nullable"]:
                state[f"param_{config.classifier}_{name}_enabled"] = True
        state[f"params_selected_{config.classifier}"] = sorted(
            name
            for name in typed_names
            if not next(
                item
                for item in catalog_by_name[config.classifier]["parameters"]
                if item["name"] == name
            )["required"]
        )
    advanced = {
        key: value for key, value in config.classifier_params.items() if key not in typed_names
    }
    state["parameter_text"] = json.dumps(advanced, ensure_ascii=False, indent=2, sort_keys=True)
