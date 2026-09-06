"""Artifact manifest writer/loader (protocol §5.6 minimal subset).

Design notes: single JSON file, missing required key = fail loudly so
non-reproducible runs can never be silently accepted.
"""

from __future__ import annotations

import json

_REQUIRED_KEYS = (
    "seed",
    "split_ref",
    "generation",
    "selection",
    "test_results",
    "elapsed",
    "failures",
    "resources",
)


def write_manifest(path, payload: dict) -> None:
    missing = [k for k in _REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(f"manifest missing required keys: {missing}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_manifest(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
