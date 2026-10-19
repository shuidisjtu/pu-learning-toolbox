"""Shared fixtures for ledger/registry contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pu_toolbox.registry import register_all_builtin_methods

_LEDGER_PATH = (
    Path(__file__).resolve().parents[2] / "pu_toolbox" / "experiment" / "method_ledger.json"
)


@pytest.fixture(scope="module")
def ledger_methods() -> dict[str, dict]:
    """The ledger ``methods`` mapping, with the builtin registry populated."""
    register_all_builtin_methods()  # idempotent; safe alongside other modules
    payload = json.loads(_LEDGER_PATH.read_text(encoding="utf-8"))
    return payload["methods"]
