"""CPU estimator tails cannot monopolize otherwise idle GPU slots."""

import pytest

from pu_toolbox.experiment.queue_scheduling import choose_pending, dispatch_slots

pytestmark = pytest.mark.unit


def job(method="pusb_kernel", path="cnn_feature_adapter", **changes):
    value = dict(
        method=method,
        path=path,
        dataset="fashion_mnist",
        seeds=[0],
        c_tokens=["0.1"],
        priority=3,
        lane="gpu",
        pid=10,
        gpu={"index": 5},
    )
    value.update(changes)
    return value


@pytest.mark.parametrize("method", ["lbe", "pusb_kernel", "upu"])
def test_param_basic_cpu_tail_gets_one_compute_overlap(method):
    active = [job(method)]
    snapshot = {5: dict(pids=set(), util=0, free_mib=48000)}
    assert dispatch_slots(snapshot, active, idle_counts={5: 4}, low_counts={5: 3}) == [
        (5, "cpu_tail_compute_overlap")
    ]
    heavy = job("pulda", "native_cnn")
    assert choose_pending([job(), heavy], active, "cpu_tail_compute_overlap") is heavy
    assert choose_pending([job()], active, "cpu_tail_compute_overlap") is None


@pytest.mark.parametrize(
    "pids,free,streak", [({99}, 48000, 3), (set(), 30000, 3), (set(), 48000, 2)]
)
def test_param_edge_foreign_memory_or_unstable_sample_blocks_overlap(pids, free, streak):
    assert not dispatch_slots(
        {5: dict(pids=pids, util=0, free_mib=free)},
        [job()],
        idle_counts={5: 5},
        low_counts={5: streak},
    )


def test_edge_never_third_worker_even_if_both_cpu_jobs_have_no_context():
    assert not dispatch_slots(
        {5: dict(pids=set(), util=0, free_mib=48000)},
        [job(), job(pid=11)],
        idle_counts={5: 9},
        low_counts={5: 9},
    )


def test_basic_determ_free_slot_prefers_gpu_work_over_cpu_solver():
    cpu = job()
    gpu = job("pulda", "native_cnn", dataset="mnist")
    assert choose_pending([cpu, gpu], [], "exclusive") is gpu
    assert choose_pending([cpu], [], "exclusive") is cpu
