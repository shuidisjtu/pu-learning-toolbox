"""Four small GPU trainers per selected card, never outside process/memory caps."""

import pytest

from pu_toolbox.experiment.queue_scheduling import choose_pending, dispatch_slots

pytestmark = pytest.mark.unit


def job(pid=10, **changes):
    result = dict(
        lane="gpu",
        pid=pid,
        gpu={"index": 0},
        method="split_pu",
        path="cnn_feature_adapter",
        dataset="fashion_mnist",
        expected_recipe={"bound": True},
        priority=3,
        seeds=[0],
        c_tokens=["0.1"],
    )
    result.update(changes)
    return result


def slots(active, **changes):
    options = dict(pids={j["pid"] for j in active}, free_mib=45000, util=99)
    options.update(changes)
    return dispatch_slots(
        {0: options}, active, idle_counts={}, low_counts={}, dense_indices={0}, dense_counts={0: 3}
    )


@pytest.mark.parametrize("n,expected", [(1, True), (2, True), (3, True), (4, False)])
def test_param_basic_dense_card_allows_four_small_compute_workers(n, expected):
    assert ((0, "dense_adapter_trial") in slots([job(10 + i) for i in range(n)])) is expected


@pytest.mark.parametrize("change", [{"pids": {10, 99}}, {"free_mib": 30000}])
def test_param_edge_foreign_owner_or_low_headroom_refuses_more(change):
    assert not slots([job()], **change)


def test_edge_mixed_large_native_model_is_not_packed_as_small_adapter():
    assert not slots([job(method="pulda", path="native_cnn")])
    assert not slots([job(method="pusb_kernel")])


def test_basic_determ_global_cap_includes_cpu_jobs_and_selection_filters():
    active = [job()] + [dict(lane="cpu") for _ in range(9)]
    assert not dispatch_slots(
        {0: dict(pids={10}, util=99, free_mib=45000)},
        active,
        idle_counts={},
        low_counts={},
        dense_indices={0},
        dense_counts={0: 3},
        max_running_jobs=10,
    )
    good = job()
    bad = job(method="pulda", path="native_cnn")
    assert choose_pending([bad, good], [], "dense_adapter_trial") is good
    assert choose_pending([bad, good], [], "dense_adapter_trial") is good
    assert choose_pending([bad], [], "dense_adapter_trial") is None
