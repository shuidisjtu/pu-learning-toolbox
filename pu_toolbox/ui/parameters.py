"""Typed parameter metadata and Streamlit form rendering helpers."""

from __future__ import annotations

import ast
import inspect
from typing import Any

from pu_toolbox.core.base import BasePUClassifier
from pu_toolbox.registry import get_algorithm, list_algorithms, register_all_builtin_methods

_MANAGED_PARAMS = {"class_prior", "random_state", "encoder", "device"}


def cnn_candidates() -> set[str]:
    """Return trainable classifier names whose capability declaration includes cnn."""
    register_all_builtin_methods()
    return {
        metadata.name
        for metadata in list_algorithms(trainable_only=True)
        if "cnn" in metadata.native_architectures
    }


def classifier_catalog() -> list[dict[str, Any]]:
    """Return trainable classifier metadata and constructor fields for the UI."""
    register_all_builtin_methods()
    catalog: list[dict[str, Any]] = []
    for metadata in sorted(list_algorithms(trainable_only=True), key=lambda item: item.name):
        try:
            cls = get_algorithm(metadata.name)
        except Exception:  # noqa: BLE001 - unavailable optional implementations
            continue
        if not isinstance(cls, type) or not issubclass(cls, BasePUClassifier):
            continue
        parameters = []
        for name, parameter in inspect.signature(cls.__init__).parameters.items():
            if name == "self" or name in _MANAGED_PARAMS:
                continue
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            parameters.append(parameter_schema(name, parameter))
        ui_ready = all(
            not parameter["required"] or parameter["type"] in {"bool", "float", "int", "str"}
            for parameter in parameters
        )
        catalog.append(
            {
                "name": metadata.name,
                "family": metadata.family.value,
                "requires_class_prior": metadata.requires_class_prior,
                "supports_gpu": metadata.supports_gpu,
                "ui_ready": ui_ready,
                "parameters": parameters,
            }
        )
    return catalog


def _literal_choices(annotation: str) -> list[Any]:
    """Return literal values from a postponed ``Literal[...]`` annotation."""
    if not annotation.startswith("Literal[") or not annotation.endswith("]"):
        return []
    try:
        values = ast.literal_eval(f"({annotation[8:-1]},)")
    except (SyntaxError, ValueError):
        return []
    return list(values)


def parameter_schema(name: str, parameter: inspect.Parameter) -> dict[str, Any]:
    """Describe one constructor parameter for a typed UI editor."""
    annotation = (
        "unspecified"
        if parameter.annotation is inspect.Parameter.empty
        else str(parameter.annotation).replace("typing.", "")
    )
    required = parameter.default is inspect.Parameter.empty
    default = None if required else parameter.default
    choices = _literal_choices(annotation)
    if choices:
        kind = "choice"
    elif annotation == "bool" or isinstance(default, bool):
        kind = "bool"
    elif annotation in {"int", "int | None"} or (
        isinstance(default, int) and not isinstance(default, bool)
    ):
        kind = "int"
    elif annotation in {"float", "float | None"} or isinstance(default, float):
        kind = "float"
    elif annotation in {"str", "str | None"} or isinstance(default, str):
        kind = "str"
    else:
        kind = "json"
    return {
        "name": name,
        "type": annotation,
        "default": None if required else repr(default),
        "default_value": default,
        "required": required,
        "kind": kind,
        "choices": choices,
        "nullable": default is None and not required,
    }


def render_parameter_form(
    st: Any, classifier: str, parameters: list[dict[str, Any]]
) -> dict[str, Any]:
    """Render opt-in typed controls and return explicitly configured values."""
    editable = [item for item in parameters if item["kind"] != "json"]
    required = [item for item in editable if item["required"]]
    optional = [item for item in editable if not item["required"]]
    selected_names = st.multiselect(
        "需要调整的参数",
        [item["name"] for item in optional],
        help="未选择的参数沿用模型默认值；必填参数会始终显示。",
        key=f"params_selected_{classifier}",
    )
    selected = required + [item for item in optional if item["name"] in selected_names]
    values: dict[str, Any] = {}
    for item in selected:
        value = _render_parameter_field(st, classifier, item)
        if value is not None or not item["nullable"]:
            values[item["name"]] = value
    return values


def _render_parameter_field(st: Any, classifier: str, item: dict[str, Any]) -> Any:
    name = item["name"]
    key = f"param_{classifier}_{name}"
    default = item["default_value"]
    help_text = f"类型：{item['type']}；模型默认值：{item['default']}"
    if item["kind"] == "choice":
        choices = item["choices"]
        index = choices.index(default) if default in choices else 0
        return st.selectbox(name, choices, index=index, help=help_text, key=key)
    if item["kind"] == "bool":
        return st.checkbox(name, value=bool(default), help=help_text, key=key)
    if item["nullable"]:
        enabled = st.checkbox(
            f"设置 {name}",
            value=False,
            help=f"关闭时使用 None。{help_text}",
            key=f"{key}_enabled",
        )
        if not enabled:
            return None
    if item["kind"] == "int":
        return int(
            st.number_input(
                name,
                value=int(default) if default is not None else 1,
                step=1,
                help=help_text,
                key=key,
            )
        )
    if item["kind"] == "float":
        return float(
            st.number_input(
                name,
                value=float(default) if default is not None else 0.1,
                format="%.8g",
                help=help_text,
                key=key,
            )
        )
    return st.text_input(name, value=default or "", help=help_text, key=key)
