"""Algorithm alias resolution.

Maps user-facing alias names to canonical registry keys so that users
can type ``"nnPU"``, ``"nnpu"``, or ``"non_negative_pu"`` and get the
same algorithm.
"""

from __future__ import annotations

# ── Alias → canonical name ─────────────────────────────────────────
# Populated by register_method() as algorithms are registered.
_ALIAS_MAP: dict[str, str] = {}


def register_alias(alias: str, canonical_name: str) -> None:
    """Register a case-insensitive alias for a canonical algorithm name.

    Parameters
    ----------
    alias : str
        Alternative name (e.g. ``"nnpu"``, ``"nn-PU"``).
    canonical_name : str
        The canonical registry key.
    """
    key = alias.lower()
    if key in _ALIAS_MAP and _ALIAS_MAP[key] != canonical_name:
        raise ValueError(
            f"Alias '{alias}' already maps to '{_ALIAS_MAP[key]}'; "
            f"cannot remap to '{canonical_name}'."
        )
    _ALIAS_MAP[key] = canonical_name


def unregister_alias(alias: str) -> None:
    """Remove an alias entry (used by ``unregister_method`` for cleanup).

    Safe to call even if the alias is not registered.
    """
    _ALIAS_MAP.pop(alias.lower(), None)


def clear_aliases() -> None:
    """Remove all alias mappings (used by ``clear_registry``)."""
    _ALIAS_MAP.clear()


def resolve_alias(name: str) -> str | None:
    """Resolve a possibly-aliased name to its canonical form.

    Returns
    -------
    str or None
        Canonical name if the alias is registered, otherwise ``None``
        (meaning the caller should also check the canonical registry).
    """
    if not isinstance(name, str):
        return None
    return _ALIAS_MAP.get(name.lower())
