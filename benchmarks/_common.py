"""Shared helpers for benchmark runners."""

from __future__ import annotations

import hashlib
import json


def canonical_hash(document: dict) -> str:
    """Stable sha256 hash of a JSON-serializable document.

    Uses sorted keys and compact separators so identical documents hash
    identically regardless of key insertion order.
    """
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
