"""Shared fixtures for ledger/registry contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_LEDGER_PATH = (
    Path(__file__).resolve().parents[2] / "pu_toolbox" / "experiment" / "method_ledger.json"
)


@pytest.fixture(scope="module")
def ledger_methods() -> dict[str, dict]:
    """The ledger methods keyed by their canonical IDs."""
    payload = json.loads(_LEDGER_PATH.read_text(encoding="utf-8"))
    return payload["methods"]
