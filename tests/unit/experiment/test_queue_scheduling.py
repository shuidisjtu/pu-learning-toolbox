"""Fit ownership, immutable live batches and bounded GPU dispatch decisions."""

import copy

import pytest

from pu_toolbox.experiment.queue_scheduling import (
    dispatch_priority,
    dispatch_slots,
    split_pending_jobs,
    validate_fit_ownership,
)

pytestmark = pytest.mark.unit


def job(tmp_path, **changes):
    keys = [
        ["mnist", "genpu", "native_cnn", "scar", c, seed] for seed in (0, 1) for c in ("0.1", "0.3")
    ]
    return dict(
        label="example",
        root=str(tmp_path / "parent"),
        status="pending",
        dataset="mnist",
        method="genpu",
        path="native_cnn",
        mechanism="scar",
        oracle=False,
        seeds=[0, 1],
        c_tokens=["0.1", "0.3"],
        expected_keys=keys,
        expected_runs=4,
        lane="gpu",
        priority=3,
        **changes,
    )


def device(**changes):
    return dict(pids=set(), util=0, free_mib=48000, **changes)


def test_basic_determ_split_preserves_exact_keys_inputs_and_parent(tmp_path):
    parent = job(tmp_path)
    before = copy.deepcopy(parent)
    a, archived = split_pending_jobs([parent], tmp_path / "output")
    b, _ = split_pending_jobs([parent], tmp_path / "output")
    assert a == b and archived == [before] and parent == before
    assert set(validate_fit_ownership(a)) == set(map(tuple, parent["expected_keys"]))
    assert all(j["expected_runs"] == 1 and len(j["seeds"]) == len(j["c_tokens"]) == 1 for j in a)
    assert all(j["parent_label"] == parent["label"] for j in a)
    recovered, _ = split_pending_jobs(a, tmp_path / "output")
    assert recovered == a


@pytest.mark.parametrize("state", ["running", "completed", "failed", "blocked"])
def test_basic_live_or_historical_batch_retains_all_future_ownership(tmp_path, state):
    parent = job(tmp_path)
    parent["status"] = state
    children, archived = split_pending_jobs([parent], tmp_path / "out")
    assert children == [parent] and not archived


@pytest.mark.parametrize("filename", ["manifest.json", "epoch_0001_model.pt"])
def test_edge_partial_pending_group_cannot_be_silently_restarted(tmp_path, filename):
    parent = job(tmp_path)
    root = tmp_path / "parent"
    root.mkdir()
    (root / filename).write_text("evidence")
    with pytest.raises(ValueError, match="contains artifacts"):
        split_pending_jobs([parent], tmp_path / "out")


@pytest.mark.parametrize("field,value", [("expected_runs", 1), ("expected_keys", [])])
def test_param_invalid_counts_fail_closed(tmp_path, field, value):
    parent = job(tmp_path)
    parent[field] = value
    with pytest.raises(ValueError, match="count"):
        validate_fit_ownership([parent])


def test_param_duplicate_roots_or_fit_identity_are_refused(tmp_path):
    parent = job(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        validate_fit_ownership([parent, parent])
    other = copy.deepcopy(parent)
    other.update(label="other", root=str(tmp_path / "other"))
    with pytest.raises(ValueError, match="fit identity"):
        validate_fit_ownership([parent, other])


def test_basic_oracle_shards_have_no_c_axis(tmp_path):
    parent = job(tmp_path)
    parent.update(
        method="pn_oracle",
        oracle=True,
        c_tokens=[],
        expected_runs=2,
        expected_keys=[["mnist", "pn_oracle", "native_cnn", "scar", None, s] for s in (0, 1)],
    )
    children, _ = split_pending_jobs([parent], tmp_path / "out")
    assert len(children) == 2 and all(j["c_tokens"] == [] for j in children)


def test_basic_free_gpu_requires_two_ownerless_samples():
    assert dispatch_slots({3: device()}, [], idle_counts={3: 1}, low_counts={}) == []
    assert dispatch_slots({3: device()}, [], idle_counts={3: 2}, low_counts={}) == [
        (3, "exclusive")
    ]


@pytest.mark.parametrize(
    "util,free,pids,streak,expected",
    [
        (11, 47000, {10}, 3, [(6, "bounded_concurrency_trial")]),
        (11, 47000, {10}, 2, []),
        (90, 47000, {10}, 3, []),
        (11, 10000, {10}, 3, []),
        (11, 47000, {10, 99}, 3, []),
    ],
)
def test_param_concurrency_needs_own_low_load_memory_and_stability(
    util, free, pids, streak, expected
):
    active = [dict(lane="gpu", pid=10, gpu={"index": 6})]
    d = dict(util=util, free_mib=free, pids=pids)
    assert dispatch_slots({6: d}, active, idle_counts={}, low_counts={6: streak}) == expected


def test_edge_never_add_third_worker_or_race_cuda_startup():
    active = [dict(lane="gpu", pid=pid, gpu={"index": 6}) for pid in (10, 11)]
    assert (
        dispatch_slots(
            {6: dict(util=10, free_mib=47000, pids={10, 11})},
            active,
            idle_counts={},
            low_counts={6: 5},
        )
        == []
    )
    assert dispatch_slots({6: device()}, active[:1], idle_counts={6: 5}, low_counts={}) == []


def test_edge_foreign_process_prevents_immediate_slot_reuse():
    assert (
        dispatch_slots(
            {3: dict(util=25, free_mib=40000, pids={99})},
            [],
            idle_counts={3: 10},
            low_counts={3: 10},
        )
        == []
    )


def test_basic_priority_spreads_independent_long_tail_first(tmp_path):
    heavy = job(tmp_path)
    light = copy.deepcopy(heavy)
    light.update(path="cnn_feature_adapter", priority=1)
    assert dispatch_priority(heavy) < dispatch_priority(light)
