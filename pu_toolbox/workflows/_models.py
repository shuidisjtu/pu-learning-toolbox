"""Classifier resolution, construction, and parameter validation."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Sequence
from typing import Any, Literal, get_args, get_origin, get_type_hints

import numpy as np
from sklearn.base import clone

from ..core.base import BasePriorEstimator, BasePUClassifier
from ..core.exceptions import PipelineError, RegistryError
from ..registry.registry import get_algorithm

_ALWAYS_PROVIDED = {"class_prior", "random_state"}


def resolve_classifier_name(
    name: str, *, provided_params: set[str] | None = None
) -> type[BasePUClassifier]:
    try:
        cls = get_algorithm(name)
    except RegistryError as exc:
        raise PipelineError(
            f"Unknown classifier {name!r}. Use 'auto' or a registered method "
            "name (see list_algorithms())."
        ) from exc
    if not isinstance(cls, type) or not issubclass(cls, BasePUClassifier):
        raise PipelineError(f"{name!r} does not resolve to a PU classifier.")
    missing = missing_required_params(cls, provided_params=provided_params)
    if missing:
        raise PipelineError(
            f"Classifier {name!r} cannot be auto-instantiated: constructor "
            f"requires {sorted(missing)}. Supply them with classifier_params "
            "or pass a configured instance instead."
        )
    return cls


def declares_encoder_parameter(cls: type) -> bool:
    return "encoder" in inspect.signature(cls.__init__).parameters


def check_architecture_capability(cls: type, architecture: str, classifier_name: str) -> None:
    """Validate ``architecture`` against the class capability declaration.

    Runs alongside the constructor-signature check; both conclusions must
    agree.  A mismatch means the capability declaration has drifted from
    the implementation and must be fixed (fail-loud, never silently pick
    one side).
    """
    if architecture != "cnn":
        return
    signature_ok = declares_encoder_parameter(cls)
    capability_ok = (
        "cnn" in cls.native_architectures
        and 4 in cls.input_ndims
        and cls.encoder_parameter is not None
    )
    if signature_ok != capability_ok:
        raise PipelineError(
            f"architecture='cnn' capability mismatch for classifier "
            f"{classifier_name!r}: constructor signature says "
            f"{'encoder supported' if signature_ok else 'no encoder'}, but the "
            f"capability declaration says "
            f"{'cnn supported' if capability_ok else 'no cnn'} "
            f"(native_architectures={set(cls.native_architectures)!r}, "
            f"input_ndims={set(cls.input_ndims)!r}, "
            f"encoder_parameter={cls.encoder_parameter!r}). "
            "Fix the class capability declaration."
        )


def missing_required_params(cls: type, *, provided_params: set[str] | None = None) -> set[str]:
    signature = inspect.signature(cls.__init__)
    provided = provided_params or set()
    return {
        param
        for param, item in signature.parameters.items()
        if param != "self"
        and item.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and item.default is inspect.Parameter.empty
        and param not in _ALWAYS_PROVIDED
        and param not in provided
    }


def pick_first_instantiable(
    candidates: Sequence[Any],
) -> tuple[type[BasePUClassifier] | None, list[dict[str, str]]]:
    skipped: list[dict[str, str]] = []
    for candidate in candidates:
        try:
            cls = get_algorithm(candidate.name)
        except RegistryError:
            skipped.append({"name": candidate.name, "reason": "not registered"})
            continue
        if not (isinstance(cls, type) and issubclass(cls, BasePUClassifier)):
            skipped.append({"name": candidate.name, "reason": "not a PU classifier"})
            continue
        missing = missing_required_params(cls)
        if missing:
            skipped.append(
                {
                    "name": candidate.name,
                    "reason": f"constructor requires {sorted(missing)}",
                }
            )
            continue
        return cls, skipped
    return None, skipped


def fresh_estimator(
    *,
    cls: type[BasePUClassifier] | None,
    instance: BasePUClassifier | None,
    prior: float | None,
    classifier_params: dict[str, Any],
    random_state: int | None,
    encoder: Any,
    device: str | None,
    max_epochs: int | None,
    classifier_name: str,
) -> BasePUClassifier:
    if instance is not None:
        estimator = clone(instance)
        if encoder is not None and declares_encoder_parameter(type(instance)):
            estimator.set_params(encoder=encoder)
        return estimator
    assert cls is not None
    kwargs: dict[str, Any] = dict(classifier_params)
    signature = inspect.signature(cls.__init__)
    if prior is not None and "class_prior" in signature.parameters:
        kwargs["class_prior"] = prior
    if "random_state" in signature.parameters and random_state is not None:
        kwargs["random_state"] = random_state
    if "encoder" in signature.parameters and encoder is not None:
        kwargs["encoder"] = encoder
    if "device" in signature.parameters and device is not None:
        kwargs["device"] = device
    if "max_epochs" in signature.parameters and max_epochs is not None:
        kwargs["max_epochs"] = max_epochs
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            f"invalid classifier parameters for '{classifier_name}': {exc}"
        ) from exc


def validate_prior_param_types(
    cls: type[BasePriorEstimator], kwargs: dict[str, Any], name: str
) -> None:
    _validate_parameter_types(cls, kwargs, name, kind="prior")


def validate_classifier_params(
    cls: type[BasePUClassifier], kwargs: dict[str, Any], name: str
) -> None:
    if not kwargs:
        return
    signature = inspect.signature(cls.__init__)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    unknown = sorted(
        key for key in kwargs if key not in signature.parameters and not accepts_kwargs
    )
    if unknown:
        available = sorted(
            key
            for key, parameter in signature.parameters.items()
            if key != "self"
            and parameter.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        )
        raise PipelineError(
            f"invalid classifier parameters for '{name}': unknown {unknown}. "
            f"Available constructor parameters: {available}"
        )
    _validate_parameter_types(cls, kwargs, name, kind="classifier")


def _validate_parameter_types(
    cls: type,
    kwargs: dict[str, Any],
    name: str,
    *,
    kind: str,
) -> None:
    try:
        hints = get_type_hints(cls.__init__)
    except (TypeError, NameError):
        warnings.warn(
            f"cannot resolve {kind}-parameter types for {cls.__name__}; "
            f"skipping {kind}-param validation",
            UserWarning,
            stacklevel=3,
        )
        return
    for key, value in list(kwargs.items()):
        annotation = hints.get(key)
        if annotation is None:
            continue
        types = get_args(annotation) if get_origin(annotation) is not None else (annotation,)
        if bool in types and isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "1", "yes", "on"):
                kwargs[key] = True
            elif low in ("false", "0", "no", "off"):
                kwargs[key] = False
            else:
                raise PipelineError(
                    f"invalid {kind} parameter '{key}' for '{name}': "
                    f"value {value!r} is not a boolean (expected true/false)"
                )
            continue
        if any(item in (int, float) for item in types):
            converted = value
            if isinstance(value, str):
                try:
                    converted = int(value) if int in types else float(value)
                except ValueError as exc:
                    raise PipelineError(
                        f"invalid {kind} parameter '{key}' for '{name}': "
                        f"value {value!r} is not a number (expected {annotation})"
                    ) from exc
            if isinstance(converted, int | float) and not np.isfinite(converted):
                raise PipelineError(
                    f"invalid {kind} parameter '{key}' for '{name}': value {value!r} must be finite"
                )
            kwargs[key] = converted
            continue
        if get_origin(annotation) is Literal:
            allowed = get_args(annotation)
            if value not in allowed:
                raise PipelineError(
                    f"invalid {kind} parameter '{key}' for '{name}': "
                    f"value {value!r} is not one of {allowed}"
                )
