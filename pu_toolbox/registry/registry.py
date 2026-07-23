"""Central algorithm registry.

All PU algorithms — whether native, adapter-wrapped, or ``api_only``
placeholders — are registered here with full metadata.  The registry
powers discovery, recommendation, and source-aware algorithm selection.
"""

from __future__ import annotations

import threading

from ..core.base import BasePriorEstimator, BasePUClassifier
from ..core.exceptions import RegistryError
from .aliases import (
    clear_aliases,
    register_alias,
    resolve_alias,
    unregister_alias,
)
from .metadata import AlgorithmMetadata

# ── Canonical registry ─────────────────────────────────────────────
_REGISTRY: dict[str, AlgorithmMetadata] = {}
_CLASSES: dict[str, type[BasePUClassifier] | type[BasePriorEstimator]] = {}
_lock: threading.RLock = threading.RLock()


def register_method(
    metadata: AlgorithmMetadata,
    estimator_cls: type[BasePUClassifier] | type[BasePriorEstimator] | None = None,
) -> None:
    """Register an algorithm in the toolbox.

    Parameters
    ----------
    metadata : AlgorithmMetadata
        Full metadata for the algorithm.
    estimator_cls : type or None
        Class reference.  ``None`` is allowed for ``api_only`` entries.

    Raises
    ------
    RegistryError
        If a method with the same canonical name is already registered,
        or if ``estimator_cls`` is not a subclass of
        :class:`~pu_toolbox.core.base.BasePUClassifier`.
    """
    if estimator_cls is not None and not issubclass(
        estimator_cls, BasePUClassifier | BasePriorEstimator
    ):
        raise RegistryError(
            f"estimator_cls must be a subclass of BasePUClassifier, "
            f"got {getattr(estimator_cls, '__name__', estimator_cls)}"
        )

    name = metadata.name
    with _lock:
        if name in _REGISTRY:
            raise RegistryError(
                f"Algorithm '{name}' is already registered. "
                "Use unregister_method() first if you need to replace it."
            )

        _REGISTRY[name] = metadata
        if estimator_cls is not None:
            _CLASSES[name] = estimator_cls
            _sync_class_metadata_to_registry(name, estimator_cls)

        # Register the canonical name itself as an alias (for case-insensitive lookup)
        register_alias(name, name)

        for alias in metadata.aliases:
            try:
                register_alias(alias, name)
            except ValueError as exc:
                raise RegistryError(str(exc)) from exc


_SYNC_FIELDS = (
    "family",
    "assumption",
    "scenario",
    "requires_class_prior",
    "implementation_status",
    "source_status",
    "backend",
    "maturity",
)


def _sync_class_metadata_to_registry(canonical: str, cls: type) -> None:
    """Copy overlapping metadata fields from the estimator class to the registry entry.

    Only syncs fields explicitly declared on the class itself (present in
    its MRO ``__dict__`` chain excluding the abstract bases), so inherited
    defaults from ``BasePUClassifier`` / ``BasePriorEstimator`` are not
    treated as authoritative overrides.
    """
    meta = _REGISTRY[canonical]
    _BASES = (BasePUClassifier, BasePriorEstimator)
    for field_name in _SYNC_FIELDS:
        declared = any(
            field_name in klass.__dict__
            for klass in cls.__mro__
            if klass not in _BASES and not issubclass(klass, type)
        )
        if not declared:
            continue
        value = getattr(cls, field_name)
        if isinstance(value, tuple):
            value = list(value)
        setattr(meta, field_name, value)


def bind_estimator_class(
    name: str, estimator_cls: type[BasePUClassifier] | type[BasePriorEstimator]
) -> None:
    """Bind a native estimator class to an already-registered method.

    Use this when upgrading an ``api_only`` entry to a runnable
    implementation without re-registering the entire metadata.

    Parameters
    ----------
    name : str
        Canonical method name (must already be registered).
    estimator_cls : type[BasePUClassifier]
        Estimator class to associate.

    Raises
    ------
    RegistryError
        If ``name`` is not registered or ``estimator_cls`` is invalid.
    """
    if estimator_cls is not None and not issubclass(
        estimator_cls, BasePUClassifier | BasePriorEstimator
    ):
        raise RegistryError(
            f"estimator_cls must be a subclass of BasePUClassifier, "
            f"got {getattr(estimator_cls, '__name__', estimator_cls)}"
        )
    canonical = _resolve_name(name)
    if canonical not in _REGISTRY:
        raise RegistryError(
            f"Unknown algorithm '{name}'. Register it first with register_method()."
        )
    with _lock:
        _CLASSES[canonical] = estimator_cls
        _sync_class_metadata_to_registry(canonical, estimator_cls)


def unregister_method(name: str) -> None:
    """Remove an algorithm from the registry (mainly for testing).

    Also removes all aliases that point to this canonical name.
    """
    with _lock:
        # Collect and remove aliases pointing to this name
        for alias, canonical in list(alias_map_items()):
            if canonical == name:
                unregister_alias(alias)
        _REGISTRY.pop(name, None)
        _CLASSES.pop(name, None)


def _resolve_name(name: str | None) -> str:
    """Resolve alias to canonical name, with None guard."""
    if not isinstance(name, str):
        raise RegistryError(f"Algorithm name must be a string, got {type(name).__name__}")
    canonical = resolve_alias(name)
    return name if canonical is None else canonical


def get_algorithm(name: str) -> type[BasePUClassifier] | type[BasePriorEstimator]:
    """Look up an algorithm class by name or alias.

    Parameters
    ----------
    name : str
        Canonical name or registered alias (case-insensitive).

    Returns
    -------
    type[BasePUClassifier]

    Raises
    ------
    RegistryError
        If the name is not found or the algorithm is ``api_only`` with
        no runnable class.
    """
    canonical = _resolve_name(name)

    cls = _CLASSES.get(canonical)
    if cls is not None:
        return cls

    # Check if it's registered but api_only (no class)
    if canonical in _REGISTRY:
        meta = _REGISTRY[canonical]
        raise RegistryError(
            f"Algorithm '{canonical}' is registered but not yet implemented "
            f"(status={meta.implementation_status.value}). "
            "Check back in a future release or consider an alternative method."
        )

    raise RegistryError(
        f"Unknown algorithm '{name}'. Use get_algorithm_registry() to list available algorithms."
    )


def get_metadata(name: str) -> AlgorithmMetadata:
    """Return the metadata for a registered algorithm.

    Raises
    ------
    RegistryError
        If not found.
    """
    canonical = _resolve_name(name)
    if canonical not in _REGISTRY:
        raise RegistryError(f"Unknown algorithm '{name}'.")
    return _REGISTRY[canonical]


def get_algorithm_registry() -> dict[str, AlgorithmMetadata]:
    """Return the full registry dict (name → metadata)."""
    return dict(_REGISTRY)


def list_algorithms(
    *,
    trainable_only: bool = False,
    family: str | None = None,
    assumption: str | None = None,
) -> list[AlgorithmMetadata]:
    """List registered algorithms, optionally filtered.

    Parameters
    ----------
    trainable_only : bool
        If True, exclude ``api_only`` and ``experimental`` entries.
    family : str or None
        Filter by :attr:`AlgorithmMetadata.family` value.
    assumption : str or None
        Filter by :attr:`AlgorithmMetadata.assumption` (checks membership).

    Returns
    -------
    list[AlgorithmMetadata]
    """
    results = []
    for meta in _REGISTRY.values():
        if trainable_only and not meta.trainable:
            continue
        if family is not None and meta.family.value != family:
            continue
        if assumption is not None and not any(a.value == assumption for a in meta.assumption):
            continue
        results.append(meta)
    return results


def alias_map_items():
    """Return a copy of all (alias, canonical) pairs (for safe iteration)."""
    from .aliases import _ALIAS_MAP

    yield from list(_ALIAS_MAP.items())


def clear_registry() -> None:
    """Clear the registry (mainly for testing)."""
    with _lock:
        _REGISTRY.clear()
        _CLASSES.clear()
        clear_aliases()
