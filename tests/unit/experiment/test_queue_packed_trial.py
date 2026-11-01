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


@pytest.mark.parametrize(
    "pids,free,util,streak,enabled,cap,allowed",
    [
        ({10, 11}, 45000, 90, 3, True, 10, True),
        ({10, 11, 99}, 45000, 90, 3, True, 10, False),
        ({10, 11}, 32000, 90, 3, True, 10, False),
        ({10, 11}, 45000, 96, 3, True, 10, False),
        ({10, 11}, 45000, 90, 2, True, 10, False),
        ({10, 11}, 45000, 90, 3, False, 10, False),
        ({10, 11}, 45000, 90, 3, True, 2, False),
        (set(), 45000, 0, 3, True, 10, False),
    ],
)
def test_param_mixed_tail_safety_limits(pids, free, util, streak, enabled, cap, allowed):
    active = [
        dict(compute(pid), method="genpu", path="native_2d", dataset="twenty_newsgroups")
        for pid in (10, 11)
    ]
    slots = dispatch_slots(
        {6: dict(pids=pids, free_mib=free, util=util)},
        active,
        idle_counts={},
        low_counts={},
        mixed_tail_indices={6} if enabled else set(),
        mixed_tail_counts={6: streak},
        max_running_jobs=cap,
    )
    assert ((6, "mixed_cpu_tail_trial") in slots) is allowed


def test_edge_mixed_tail_never_stacks_or_selects_another_trainer():
    active = [
        dict(compute(pid), method="genpu", path="native_2d", dataset="twenty_newsgroups")
        for pid in (10, 11)
    ]
    tail = dict(compute(12), method="upu", path="cnn_feature_adapter")
    args = dict(idle_counts={}, low_counts={}, mixed_tail_indices={6}, mixed_tail_counts={6: 3})
    device = {6: dict(pids={10, 11}, free_mib=45000, util=90)}
    assert dispatch_slots(device, active + [tail], **args) == []
    assert dispatch_slots(device, [active[0], compute(11)], **args) == []
    assert dispatch_slots(device, [active[0], dict(active[1], expected_recipe={})], **args) == []
    assert choose_pending([*active, tail], active, "mixed_cpu_tail_trial") is tail
    assert choose_pending(active, active, "mixed_cpu_tail_trial") is None
    assert choose_pending([dict(tail, campaign_id="repair")], [], "mixed_cpu_tail_trial") is None
    assert choose_pending([dict(tail, expected_recipe={})], [], "mixed_cpu_tail_trial") is None


@pytest.mark.parametrize(
    "method", ["nnpu", "pulda", "holistic_pu", "robust_pu", "self_pu", "split_pu"]
)
def test_param_basic_mixed_tail_can_fill_second_measured_text_trainer(method):
    first = dict(compute(), method="genpu", path="native_2d", dataset="twenty_newsgroups")
    tail = dict(compute(11), method="pusb_kernel", path="cnn_feature_adapter")
    next_job = dict(first, method=method, pid=12)
    args = dict(idle_counts={}, low_counts={}, mixed_tail_indices={6}, mixed_tail_counts={6: 3})
    device = {6: dict(pids={10}, free_mib=45000, util=40)}
    assert dispatch_slots(device, [first, tail], **args) == [(6, "mixed_gpu_compute_trial")]
    assert choose_pending([tail, next_job], [first, tail], "mixed_gpu_compute_trial") is next_job
    assert dispatch_slots(device, [first, tail, next_job], **args) == []
    assert (
        choose_pending([tail, dict(next_job, expected_recipe={})], [], "mixed_gpu_compute_trial")
        is None
    )
    device[6]["pids"].add(99)
    assert dispatch_slots(device, [first, tail], **args) == []
