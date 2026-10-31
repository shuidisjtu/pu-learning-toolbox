"""Host-neutral, fail-closed policy for independent Survey fit scheduling.

This module changes dispatch granularity, not training or frozen recipes.
Already running batches retain ownership of every fit they may still execute.
"""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path


def parse_gpu_indices(value):
    """Explicit nonempty physical GPU allowlist, not a count or reservation."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("GPU indices must be a nonempty comma-separated list")
    tokens = value.split(",")
    if any(not t.strip().isdigit() for t in tokens):
        raise ValueError("GPU indices must be nonnegative integers")
    indices = tuple(sorted(int(t.strip()) for t in tokens))
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate GPU index")
    return indices


def allowed_devices(snapshot, indices):
    """Never substitute a disallowed GPU when an allowed one is busy/missing."""
    return snapshot if indices is None else {i: d for i, d in snapshot.items() if i in indices}


def validate_fit_ownership(jobs):
    """Reject duplicate identities/outputs before a controller starts work."""
    owners, roots, labels = {}, set(), set()
    for job in jobs:
        root = str(Path(job["root"]).resolve())
        if root in roots or job["label"] in labels:
            raise ValueError("duplicate job label or output root")
        roots.add(root)
        labels.add(job["label"])
        keys = job["expected_keys"]
        if not keys or len(keys) != job["expected_runs"]:
            raise ValueError("expected fit count does not match identities")
        for raw in keys:
            key = tuple(raw)
            scoped = (job["campaign_id"], *key) if job.get("campaign_id") else key
            if len(key) != 6 or scoped in owners:
                raise ValueError("duplicate or malformed fit identity")
            owners[scoped] = job["label"]
    return owners


def split_pending_jobs(jobs, output):
    """Split only untouched pending groups; never steal a live batch's tail."""
    validate_fit_ownership(jobs)
    result, parents = [], []
    for job in jobs:
        if job["status"] != "pending" or job.get("parent_label"):
            result.append(copy.deepcopy(job))
            continue
        if any(Path(job["root"]).rglob("manifest.json")) or any(Path(job["root"]).rglob("*.pt")):
            raise ValueError("pending group contains artifacts; refusing implicit resume")
        parents.append(copy.deepcopy(job))
        for key in job["expected_keys"]:
            dataset, method, path, mechanism, token, seed = key
            if (
                (dataset, method, path, mechanism)
                != (job["dataset"], job["method"], job["path"], job["mechanism"])
                or seed not in job["seeds"]
                or (not job["oracle"] and token not in job["c_tokens"])
            ):
                raise ValueError("fit identity disagrees with parent request")
            label = f"{job['label']}__fit_s{seed}_c{token if token is not None else 'PN'}"
            child = copy.deepcopy(job)
            child.update(
                label=label,
                parent_label=job["label"],
                seeds=[seed],
                c_tokens=[] if job["oracle"] else [token],
                expected_runs=1,
                expected_keys=[list(key)],
                root=str(Path(output) / "parallel" / label),
            )
            result.append(child)
    if set(validate_fit_ownership(result)) != set(validate_fit_ownership(jobs)):
        raise ValueError("split changed the planned fit identities")
    return result, parents


def dispatch_slots(
    devices,
    active_jobs,
    *,
    idle_counts,
    low_counts,
    low_limit=35,
    packed_indices=(),
    packed_counts=None,
    dense_indices=(),
    dense_counts=None,
    max_dense_workers=4,
    max_running_jobs=None,
):
    """Bounded dispatch; optional packed cards allow two known GPU trainers.

    A CUDA allocation is not a reservation. Never append to a card containing
    an unowned process, even if another user's memory allocation looks small.
    Startup allocations count before they appear in NVIDIA's process table.
    """
    active = [j for j in active_jobs if j.get("lane") == "gpu"]
    if max_running_jobs is not None and len(active_jobs) >= max_running_jobs:
        return []
    counts = Counter(j["gpu"]["index"] for j in active)
    own_pids = {j["pid"] for j in active}
    slots = []
    packed_counts = packed_counts or {}
    dense_counts = dense_counts or {}
    for index, device in sorted(devices.items()):
        pids = set(device["pids"])
        if pids - own_pids:
            continue
        if counts[index] == 0 and not pids:
            # Returning an owned slot promptly is safe once its process has
            # exited; stale utilization is not evidence of an active owner.
            if idle_counts.get(index, 0) >= 2 and device["free_mib"] >= 12288:
                slots.append((index, "exclusive"))
        elif (
            index in dense_indices
            and 1 <= counts[index] < max_dense_workers
            and pids
            and dense_counts.get(index, 0) >= 3
            and device["free_mib"] >= 32768
            and all(dense_adapter_candidate(j) for j in active if j["gpu"]["index"] == index)
        ):
            slots.append((index, "dense_adapter_trial"))
        elif (
            index in packed_indices
            and 1 <= counts[index] < 3
            and pids
            and packed_counts.get(index, 0) >= 3
            and device["util"] <= 95
            and device["free_mib"] >= 32768
            and len([j for j in active if j["gpu"]["index"] == index and not cpu_estimator_tail(j)])
            == 1
            and all(
                small_native_compute(j) or cpu_estimator_tail(j)
                for j in active
                if j["gpu"]["index"] == index
            )
        ):
            slots.append((index, "packed_compute_trial"))
        elif (
            counts[index] == 1
            and not pids
            and device["util"] <= low_limit
            and low_counts.get(index, 0) >= 3
            and device["free_mib"] >= 32768
            and any(j["gpu"]["index"] == index and cpu_estimator_tail(j) for j in active)
        ):
            # CPU fitting can follow CUDA extraction. Leave room for a late
            # return to CUDA; do not migrate or stop that worker.
            slots.append((index, "cpu_tail_compute_overlap"))
        elif (
            counts[index] == 1
            and not pids
            and device["util"] <= low_limit
            and low_counts.get(index, 0) >= 3
            and device["free_mib"] >= 32768
            and any(
                j["gpu"]["index"] == index
                and j.get("path") == "native_cnn"
                and j.get("method") in ("nnpu", "genpu")
                and j.get("dataset") in ("adni", "mnist", "fashion_mnist", "cifar10")
                for j in active
            )
        ):
            # One short adapter may overlap CPU preparation. Keep >=32 GiB
            # free before launch for both workers' later CUDA allocations;
            # this is a conservative admission heuristic, not a reservation.
            slots.append((index, "prep_short_overlap"))
        elif (
            counts[index] == 1
            and pids
            and device["util"] <= low_limit
            and low_counts.get(index, 0) >= 3
            and device["free_mib"] >= 16384
        ):
            slots.append((index, "bounded_concurrency_trial"))
    return slots


def short_candidate(job):
    """Bounded local fast lane; excludes collaborator's proposed tabular shards."""
    if job.get("campaign_id") or job.get("path") != "cnn_feature_adapter":
        return False
    dataset, method = job.get("dataset"), job.get("method")
    return (
        (
            dataset == "adni"
            and method in ("dist_pu", "robust_pu", "self_pu", "split_pu", "lbe", "holistic_pu")
        )
        or (dataset in ("cifar10", "mnist", "fashion_mnist") and method == "dist_pu")
        or (dataset == "cifar10" and method == "self_pu")
    )


def cpu_estimator_tail(job):
    """Only these feature-adapted estimators have CPU-only fitting."""
    return job.get("path") == "cnn_feature_adapter" and job.get("method") in {
        "lbe",
        "pusb_kernel",
        "upu",
    }


def small_native_compute(job):
    """Narrow measured recipe families; not a universal memory guarantee."""
    return (
        job.get("path") == "native_cnn"
        and job.get("method") in {"nnpu", "genpu", "pulda"}
        and job.get("dataset") in {"adni", "mnist", "fashion_mnist", "cifar10"}
        and bool(job.get("expected_recipe"))
    )


def dense_adapter_candidate(job):
    """Only frozen small MLP heads on shared image features; never CPU solvers."""
    return (
        job.get("path") == "cnn_feature_adapter"
        and job.get("method") in {"dist_pu", "self_pu", "robust_pu", "split_pu", "holistic_pu"}
        and job.get("dataset") in {"adni", "mnist", "fashion_mnist", "cifar10"}
        and bool(job.get("expected_recipe"))
        and not job.get("campaign_id")
    )


def choose_pending(pending, active, mode):
    """At most one fast-lane worker; leave remaining cards to long/recovery jobs."""
    ordered = sorted(pending, key=dispatch_priority)
    if not ordered:
        return None
    if mode == "dense_adapter_trial":
        return next((j for j in ordered if dense_adapter_candidate(j)), None)
    if mode == "packed_compute_trial":
        return next((j for j in ordered if small_native_compute(j)), None)
    if mode == "cpu_tail_compute_overlap":
        compute = [j for j in ordered if not cpu_estimator_tail(j)]
        images = [j for j in compute if j.get("path") == "native_cnn"]
        return (images or compute or [None])[0]
    shorts = [j for j in ordered if short_candidate(j)]
    short_active = any(short_candidate(j) for j in active)
    if mode == "prep_short_overlap":
        return shorts[0] if shorts and not short_active else None
    protected = [j for j in ordered if j.get("campaign_id") or j.get("retry_of")]
    if protected:
        return protected[0]
    if mode == "exclusive" and len(active) >= 2 and shorts and not short_active:
        return shorts[0]
    compute = [j for j in ordered if not cpu_estimator_tail(j)]
    return (compute or ordered)[0]


def dispatch_priority(job):
    """Spread independent image long tails before many tiny adapter fits."""
    long_tail = job["path"] == "native_cnn" and job["method"] in ("genpu", "nnpu")
    return (
        0 if job.get("campaign_id") else 1,
        0 if job.get("campaign_canary") else 1,
        {"spambase": 0, "connect_4": 1}.get(job["dataset"], 2) if job.get("campaign_id") else 0,
        0 if long_tail else 1,
        job.get("priority", 99),
        job["dataset"],
        job["method"],
        job["seeds"][0],
        str(job["c_tokens"]),
    )
