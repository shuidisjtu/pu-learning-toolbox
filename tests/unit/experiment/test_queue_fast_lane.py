"""Fast-lane fairness and CPU-preparation overlap safety boundaries."""

import pytest

from pu_toolbox.experiment.queue_scheduling import choose_pending, dispatch_slots, short_candidate

pytestmark = pytest.mark.unit


def item(**changes):
    result = dict(
        dataset="mnist",
        method="nnpu",
        path="native_cnn",
        seeds=[0],
        c_tokens=["0.1"],
        priority=3,
        lane="gpu",
        pid=10,
        gpu={"index": 6},
    )
    result.update(changes)
    return result


@pytest.mark.parametrize(
    "free,pids,streak,expected",
    [
        (48000, set(), 3, True),
        (32000, set(), 3, False),
        (48000, {99}, 3, False),
        (48000, set(), 2, False),
    ],
)
def test_param_preparation_overlap_requires_capacity_ownership_and_stability(
    free, pids, streak, expected
):
    slots = dispatch_slots(
        {6: dict(free_mib=free, pids=pids, util=0)},
        [item()],
        idle_counts={6: 9},
        low_counts={6: streak},
    )
    assert ((6, "prep_short_overlap") in slots) is expected


def test_edge_preparation_never_launches_third_or_unknown_model():
    devices = {6: dict(free_mib=48000, pids=set(), util=0)}
    for active in ([item(), item(pid=11)], [item(method="pulda")]):
        assert not dispatch_slots(devices, active, idle_counts={6: 9}, low_counts={6: 9})


def test_basic_short_lane_runs_beside_two_long_jobs():
    heavy = item()
    short = item(path="cnn_feature_adapter", method="dist_pu")
    assert choose_pending([heavy, short], [item(), item(pid=11)], "exclusive") is short
    assert choose_pending([heavy, short], [item()], "exclusive") is heavy
    assert choose_pending([heavy, short], [short, item()], "exclusive") is heavy


def test_edge_overlap_only_accepts_one_short_worker():
    heavy = item()
    short = item(path="cnn_feature_adapter", method="dist_pu")
    assert choose_pending([heavy], [heavy], "prep_short_overlap") is None
    assert choose_pending([short], [heavy], "prep_short_overlap") is short
    assert choose_pending([short], [short], "prep_short_overlap") is None


def test_basic_recovery_precedes_fast_lane_on_free_cards():
    retry = item(retry_of="original")
    short = item(path="cnn_feature_adapter", method="dist_pu")
    assert choose_pending([short, retry], [item(), item()], "exclusive") is retry


@pytest.mark.parametrize("dataset", ["spambase", "imdb", "twenty_newsgroups"])
def test_param_collaborator_candidates_not_promoted(dataset):
    assert not short_candidate(item(dataset=dataset, method="dist_pu", path="native_2d"))
