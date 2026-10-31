"""Opt-in two-training-worker packing with ownership and headroom guards."""

import pytest

from pu_toolbox.experiment.queue_scheduling import choose_pending, dispatch_slots

pytestmark = pytest.mark.unit


def compute(pid=10):
    return dict(
        lane="gpu",
        pid=pid,
        gpu={"index": 6},
        method="pulda",
        path="native_cnn",
        dataset="adni",
        expected_recipe={"bound": True},
        priority=3,
        seeds=[0],
        c_tokens=["0.1"],
    )


@pytest.mark.parametrize(
    "pids,free,util,streak,enabled,allowed",
    [
        ({10}, 45000, 85, 3, True, True),
        ({10}, 45000, 85, 3, False, False),
        ({10, 99}, 45000, 85, 3, True, False),
        ({10}, 30000, 85, 3, True, False),
        ({10}, 45000, 99, 3, True, False),
        ({10}, 45000, 85, 2, True, False),
    ],
)
def test_param_basic_edge_pack_requires_own_gpu_and_stable_headroom(
    pids, free, util, streak, enabled, allowed
):
    slots = dispatch_slots(
        {6: dict(pids=pids, free_mib=free, util=util)},
        [compute()],
        idle_counts={},
        low_counts={},
        packed_indices={6} if enabled else set(),
        packed_counts={6: streak},
    )
    assert ((6, "packed_compute_trial") in slots) is allowed


def test_edge_max_two_gpu_trainers_even_with_cpu_tail():
    first = compute()
    cpu = dict(compute(11), method="pusb_kernel", path="cnn_feature_adapter")
    dev = {6: dict(pids={10}, free_mib=45000, util=85)}
    args = dict(idle_counts={}, low_counts={}, packed_indices={6}, packed_counts={6: 4})
    assert dispatch_slots(dev, [first, cpu], **args) == [(6, "packed_compute_trial")]
    assert dispatch_slots(dev, [first, compute(12), cpu], **args) == []
    assert dispatch_slots(dev, [first, compute(12)], **args) == []


def test_basic_determ_only_measured_bound_native_family_selected():
    good = compute()
    unknown = dict(good, method="pan")
    unbound = dict(good, expected_recipe={})
    assert choose_pending([unknown, unbound, good], [], "packed_compute_trial") is good
    assert choose_pending([unknown, unbound], [], "packed_compute_trial") is None
    assert choose_pending([unknown, unbound, good], [], "packed_compute_trial") is good
