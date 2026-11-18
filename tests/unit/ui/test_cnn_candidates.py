# ruff: noqa: N802, N803, N806
"""UI candidate-set derivation from registry capability metadata."""

from __future__ import annotations

import pytest

from pu_toolbox.registry import list_algorithms
from pu_toolbox.ui.parameters import cnn_candidates

pytestmark = [pytest.mark.unit]


@pytest.mark.unit
def test_cnn_candidates_matches_registry_declarations():
    candidates = cnn_candidates()
    for name in candidates:
        meta = next(m for m in list_algorithms() if m.name == name)
        assert "cnn" in meta.native_architectures, name
    for meta in list_algorithms():
        if "cnn" in meta.native_architectures:
            assert meta.name in candidates, meta.name
