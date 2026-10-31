"""A physical GPU allowlist cannot spill to idle disallowed cards."""

import pytest

from pu_toolbox.experiment.queue_scheduling import (
    allowed_devices,
    dispatch_slots,
    parse_gpu_indices,
)

pytestmark = pytest.mark.unit


def test_basic_determ_explicit_four_card_list():
    assert parse_gpu_indices("7, 4,6,5") == (4, 5, 6, 7)
    snapshot = {i: dict(pids=set(), util=0, free_mib=48000) for i in range(8)}
    allowed = allowed_devices(snapshot, parse_gpu_indices("4,5,6,7"))
    slots = dispatch_slots(allowed, [], idle_counts={i: 2 for i in snapshot}, low_counts={})
    assert [i for i, _ in slots] == [4, 5, 6, 7]
    assert len(snapshot) == 8


@pytest.mark.parametrize("value", ["", " ", "-1", "0,,1", "0,0", "0.5", "all", None])
def test_param_edge_invalid_allowlist_refused(value):
    with pytest.raises(ValueError):
        parse_gpu_indices(value)


def test_edge_busy_or_missing_allowed_cards_do_not_use_other_free_cards():
    free = dict(pids=set(), util=0, free_mib=48000)
    snapshot = {0: free, 4: dict(pids={99}, util=0, free_mib=48000)}
    assert (
        dispatch_slots(
            allowed_devices(snapshot, (4, 5, 6, 7)), [], idle_counts={0: 9, 4: 9}, low_counts={}
        )
        == []
    )
    assert allowed_devices(snapshot, None) == snapshot
