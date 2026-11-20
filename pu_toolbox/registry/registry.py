"""Discover and register concrete PU estimators by name or alias."""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass, field
from inspect import isabstract
from typing import NamedTuple

from ..core.base import BasePriorEstimator, BasePUClassifier
from ..core.exceptions import RegistryError
from .metadata import AlgorithmMetadata

_EstimatorClass = type[BasePUClassifier] | type[BasePriorEstimator]


class _RegisteredMethod(NamedTuple):
    metadata: AlgorithmMetadata
    estimator_cls: _EstimatorClass


class _Alias(NamedTuple):
    canonical_name: str
    deprecated: bool


@dataclass
class _Registry:
    methods: dict[str, _RegisteredMethod] = field(default_factory=dict)
    aliases: dict[str, _Alias] = field(default_factory=dict)

    def add(self, metadata: AlgorithmMetadata, estimator_cls: _EstimatorClass) -> None:
        name = _name_key(metadata.name)
        aliases = [_name_key(alias) for alias in metadata.aliases]
        deprecated_aliases = [_name_key(alias) for alias in metadata.deprecated_aliases]
        self._check_names(name, aliases)
        _sync_estimator_metadata(metadata, estimator_cls)

        metadata.name = name
        metadata.aliases = aliases
        metadata.deprecated_aliases = deprecated_aliases
        deprecated = set(deprecated_aliases)
        self.methods[name] = _RegisteredMethod(metadata, estimator_cls)
        self.aliases.update({alias: _Alias(name, alias in deprecated) for alias in aliases})

    def _check_names(self, name: str, aliases: list[str]) -> None:
        seen: set[str] = set()
        for key in (name, *aliases):
            if key in seen:
                raise RegistryError(f"Duplicate name or alias {key!r} for algorithm {name!r}.")
            if key in self.methods:
                raise RegistryError(f"Name or alias {key!r} is already registered as an algorithm.")
            existing = self.aliases.get(key)
            if existing is not None:
                raise RegistryError(
                    f"Name or alias {key!r} already refers to {existing.canonical_name!r}."
                )
            seen.add(key)

    def lookup(self, name: str) -> _RegisteredMethod:
        key = _name_key(name)
        alias = self.aliases.get(key)
        if alias is not None:
            key = alias.canonical_name
            if alias.deprecated:
                warnings.warn(
                    f"Algorithm alias {name!r} is deprecated and will be removed in a "
                    f"future release; use {key!r} instead.",
                    FutureWarning,
                    stacklevel=3,
                )
        method = self.methods.get(key)
        if method is None:
            raise RegistryError(f"Unknown algorithm {name!r}.")
        return method

    def remove(self, name: str) -> None:
        if self.methods.pop(name, None) is None:
            return
        aliases = [key for key, alias in self.aliases.items() if alias.canonical_name == name]
        for key in aliases:
            del self.aliases[key]


_REGISTRY: _Registry | None = None
_LOCK = threading.RLock()

_SYNC_FIELDS = (
    "family",
    "assumption",
    "scenario",
    "source_status",
    "backend",
    "maturity",
    "native_architectures",
    "input_ndims",
    "encoder_parameter",
    "trains_encoder",
    "label_semantics",
    "prior_consumers",
)


def _name_key(name: str) -> str:
    if not isinstance(name, str):
        raise RegistryError(f"Algorithm name must be a string, got {type(name).__name__}")
    return name.lower()


def _validate_estimator_class(estimator_cls: _EstimatorClass) -> None:
    if (
        not isinstance(estimator_cls, type)
        or not issubclass(estimator_cls, BasePUClassifier | BasePriorEstimator)
        or isabstract(estimator_cls)
    ):
        raise RegistryError(
            "estimator_cls must be a concrete BasePUClassifier or BasePriorEstimator subclass, "
            f"got {getattr(estimator_cls, '__name__', estimator_cls)}."
        )


def _sync_estimator_metadata(metadata: AlgorithmMetadata, estimator_cls: _EstimatorClass) -> None:
    """Copy declarations from the estimator, excluding abstract-base defaults."""
    abstract_bases = (BasePUClassifier, BasePriorEstimator)
    declaring_classes = [
        cls
        for cls in estimator_cls.__mro__
        if cls not in abstract_bases and not issubclass(cls, type)
    ]
    for field_name in _SYNC_FIELDS:
        if any(field_name in cls.__dict__ for cls in declaring_classes):
            value = getattr(estimator_cls, field_name)
            setattr(
                metadata,
                field_name,
                list(value) if field_name in {"assumption", "scenario"} else value,
            )


def _get_registry() -> _Registry:
    """Initialize once; the caller holds _LOCK."""
    global _REGISTRY
    if _REGISTRY is None:
        from .builtin_methods import _load_builtin_methods

        registry = _Registry()
        for metadata, estimator_cls in _load_builtin_methods():
            _validate_estimator_class(estimator_cls)
            registry.add(metadata, estimator_cls)
        _REGISTRY = registry
    return _REGISTRY


def register_method(metadata: AlgorithmMetadata, estimator_cls: _EstimatorClass) -> None:
    """Register metadata and a concrete estimator class together."""
    _validate_estimator_class(estimator_cls)
    with _LOCK:
        _get_registry().add(metadata, estimator_cls)


def unregister_method(name: str) -> None:
    """Remove a canonical name and its aliases; an unknown name is a no-op."""
    with _LOCK:
        _get_registry().remove(_name_key(name))


def get_algorithm(name: str) -> _EstimatorClass:
    """Get a concrete estimator class by name or alias, ignoring case."""
    with _LOCK:
        return _get_registry().lookup(name).estimator_cls


def get_metadata(name: str) -> AlgorithmMetadata:
    """Get registered metadata by name or alias, ignoring case."""
    with _LOCK:
        return _get_registry().lookup(name).metadata


def get_algorithm_registry() -> dict[str, AlgorithmMetadata]:
    """Return a shallow snapshot keyed by canonical name."""
    with _LOCK:
        return {name: entry.metadata for name, entry in _get_registry().methods.items()}


def list_algorithms(
    *, family: str | None = None, assumption: str | None = None
) -> list[AlgorithmMetadata]:
    """List registered metadata matching optional family and assumption filters."""
    results = []
    for metadata in get_algorithm_registry().values():
        if family is not None and metadata.family.value != family:
            continue
        if assumption is not None and not any(
            tag.value == assumption for tag in metadata.assumption
        ):
            continue
        results.append(metadata)
    return results


def clear_registry() -> None:
    """Discard the current registry; the next use reloads the built-ins."""
    global _REGISTRY
    with _LOCK:
        _REGISTRY = None
