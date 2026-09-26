# tests/unit/experiment/test_checkpoint_storage_profiles.py

# ruff: noqa: N803, N806, S101

"""What one per-epoch checkpoint component costs, per training path.

That figure feeds two decisions which are not symmetric: a host is sized against
the accumulated total, and a run is refused by the pre-run guard.  Both used to
read one constant for every ``resnet18*`` row, which is only right for the rows
that train the ResNet itself -- the adapter rows train a head on frozen features
and save that head.

The adapter bound is pinned against the real construction path rather than
against a hand-computed parameter count, so an architecture change has to fail
these tests instead of quietly keeping a stale constant plausible.
"""

from __future__ import annotations

import pytest

from pu_toolbox.experiment.survey_protocol import (
    RESNET18_COMPONENT_BYTES,
    load_protocol,
    resolve_unit,
    unit_checkpoint_bytes,
)

pytestmark = pytest.mark.unit

#: The adapter head is built on the frozen encoder's output, so every adapter
#: row's estimator is sized from this width rather than from the raw pixels.
ADAPTER_FEATURE_DIM = 512


def _adapter_row(protocol: dict, method: str = "dist_pu") -> dict:
    return resolve_unit(protocol, "cifar10", method, "cnn_feature_adapter")


def _native_cnn_row(protocol: dict) -> dict:
    return resolve_unit(protocol, "cifar10", "nnpu", "native_cnn")


# --- the defect this file exists for -----------------------------------------


def test_basic_adapter_rows_do_not_cost_a_full_resnet():
    """A frozen encoder is never saved, so a row naming one may not cost one.

    The two rows differ in what ``fitted.model_`` holds: the native CNN trains
    and saves the ResNet, the adapter saves the head it trains on top of frozen
    features.  One constant for both overstates the adapter by ~178x, which is
    what a host used to be sized against.
    """
    protocol = load_protocol()

    adapter = unit_checkpoint_bytes(protocol, _adapter_row(protocol), input_dim=ADAPTER_FEATURE_DIM)
    end_to_end = unit_checkpoint_bytes(
        protocol, _native_cnn_row(protocol), input_dim=ADAPTER_FEATURE_DIM
    )

    assert end_to_end == RESNET18_COMPONENT_BYTES
    assert adapter < end_to_end
