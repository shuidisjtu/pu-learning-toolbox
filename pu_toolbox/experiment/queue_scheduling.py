"""Host-neutral, fail-closed policy for independent Survey fit scheduling.

This module changes dispatch granularity, not training or frozen recipes.
Already running batches retain ownership of every fit they may still execute.
"""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path


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


def dispatch_slots(devices, active_jobs, *, idle_counts, low_counts, low_limit=35):
    """One fit per free card; at most one extra on a proven low-load own card.

    A CUDA allocation is not a reservation. Never append to a card containing
    an unowned process, even if another user's memory allocation looks small.
    Startup allocations count before they appear in NVIDIA's process table.
    """
    active = [j for j in active_jobs if j.get("lane") == "gpu"]
    counts = Counter(j["gpu"]["index"] for j in active)
    own_pids = {j["pid"] for j in active}
    slots = []
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
            counts[index] == 1
            and pids
            and device["util"] <= low_limit
            and low_counts.get(index, 0) >= 3
            and device["free_mib"] >= 16384
        ):
            slots.append((index, "bounded_concurrency_trial"))
    return slots


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
