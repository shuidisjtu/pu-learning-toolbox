# ruff: noqa: N802, N803, N806
"""UI candidate-set derivation from registry capability metadata."""

from __future__ import annotations

import pytest

from pu_toolbox.registry import list_algorithms, register_all_builtin_methods
from pu_toolbox.ui.parameters import cnn_candidates

pytestmark = [pytest.mark.unit]


@pytest.mark.unit
def test_cnn_candidates_matches_registry_declarations():
    register_all_builtin_methods()
    candidates = cnn_candidates()
    for name in candidates:
        meta = next(m for m in list_algorithms(trainable_only=True) if m.name == name)
        assert "cnn" in meta.native_architectures, name
    for meta in list_algorithms(trainable_only=True):
        if "cnn" in meta.native_architectures:
            assert meta.name in candidates, meta.name


@pytest.mark.unit
def test_cnn_candidates_matches_current_declarations():
    """Phase-0 declarations: only infomax/wconpu support cnn today."""
    assert cnn_candidates() == {"infomax_pu", "weighted_contrastive_pu"}
