"""Screen every frozen KLDCE fit before recovering independently feasible c groups.

Only train labels and NPZ feature headers are read. No kernel, encoder or model
is constructed; PA/OA labels never enter screening. Parent artifacts stay intact.
The output is a replacement campaign, not additional scientific fits or approval.
"""

from __future__ import annotations

import argparse
import copy
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import run_survey_backlog_campaign as backlog
from scripts.prepare_kldce_resource_plan import build_plan


def train_metadata(directory):
    """Read labels and feature shape without materializing image/feature arrays."""
    path = directory / "train.npz"
    with zipfile.ZipFile(path) as archive, archive.open("X.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("unsupported feature header version")
    with np.load(path, allow_pickle=False) as data:
        labels = data["y"]
    if (
        len(shape) not in (2, 4)
        or dtype.kind not in "iuf"
        or labels.ndim != 1
        or shape[0] != len(labels)
        or not np.isin(labels, [0, 1]).all()
        or not 0 < np.count_nonzero(labels == 1) < len(labels)
    ):
        raise ValueError("invalid KLDCE training role")
    return labels, shape, dtype


def screen(job, metadata, prior, budget):
    """Bind to actual SCAR rounding and the unchanged population-override recipe."""
    from pu_toolbox.experiment.strategies import SCARGenerator

    if job["method"] != "kldce" or job["mechanism"] != "scar":
        raise ValueError("only KLDCE SCAR jobs may be screened")
    labels, shape, dtype = metadata
    if job["path"] == "cnn_feature_adapter" and len(shape) == 4:
        dimension, itemsize = 512, 4
    elif job["path"] == "native_2d" and len(shape) == 2:
        dimension, itemsize = shape[1], dtype.itemsize
    else:
        raise ValueError("feature path differs from stored training shape")
    # SCAR uses labels only; no feature payload is needed for fixed-count marking.
    _, generation = SCARGenerator().generate(None, labels, float(job["c"]), job["seed"])
    positive = generation["n_labeled"]
    plan = build_plan(
        n_positive=positive,
        n_unlabeled=len(labels) - positive,
        feature_count=dimension,
        gram_itemsize=itemsize,
        prior_policy="population_override",
        flip_probability=1 - generation["c_realized"],
        population_prior=prior,
        max_acs_iter=budget["max_acs_iter"],
        max_inner_iter=budget["max_inner_iter"],
        memory_cap_bytes=budget["kernel_payload_cap_bytes"],
    )
    # Match the trainer's named-QP + covariance lower-bound gate. Passing is
    # not peak-memory approval; the controller retains its 650 GiB RAM gate.
    payload = plan["qp_named_array_payload_bytes"] + plan["covariance_single_matrix_payload_bytes"]
    reasons = []
    if not plan["qp_feasibility_screen"]["feasible"]:
        reasons.append(plan["qp_feasibility_screen"]["reason"])
    if payload > budget["kernel_payload_cap_bytes"]:
        reasons.append("named_QP_and_covariance_exceed_frozen_cap")
    return dict(
        generation=generation,
        feasibility=plan["qp_feasibility_screen"],
        named_payload_bytes=payload,
        kernel_payload_cap_bytes=budget["kernel_payload_cap_bytes"],
        blockers=reasons,
        model_or_kernel_constructed=False,
        passing_is_not_fit_or_peak_memory_approval=True,
    )


def prepare(output, parent):
    """Keep every requested fit visible, isolating inapplicability from fit failures."""
    from pu_toolbox.experiment.survey_protocol import digest, load_protocol, resolve_protocol_path
    from pu_toolbox.experiment.survey_recipe_registry import canonical_digest, load_registry

    if output.exists():
        raise ValueError("output exists; never overwrite a recovery")
    old = backlog.read(parent / "campaign.json")
    state = backlog.read(parent / "status.json")
    # The original v3.1 campaign predates the explicit version field, as in
    # backlog.worker. Its protocol/registry digests still have to match exactly.
    old.setdefault("protocol_version", "survey-v3.1")
    if old.get("protocol_version") not in backlog.REVISIONS:
        raise ValueError("unsupported parent protocol")
    protocol = load_protocol(resolve_protocol_path(old["protocol_version"]))
    registry = load_registry(
        backlog.ROOT / "pu_toolbox/experiment" / backlog.REVISIONS[old["protocol_version"]][1]
    )
    if (
        digest(protocol) != old["protocol_sha256"]
        or canonical_digest(registry) != old["registry_sha256"]
    ):
        raise ValueError("parent protocol/registry changed")
    status = {j["label"]: j for j in state["jobs"]}
    jobs = [copy.deepcopy(j) for j in old["jobs"] if j["method"] == "kldce"]
    if not jobs or len({j["label"] for j in jobs}) != len(jobs):
        raise ValueError("missing or duplicate KLDCE jobs")
    budget = protocol["budgets"][protocol["method_profiles"]["kldce"]["budget"]]
    metadata, inputs, canaries = {}, {}, set()
    for job in jobs:
        if status[job["label"]]["status"] not in {"failed", "held_family_failure"}:
            raise ValueError("recovery cannot duplicate pending, running or completed work")
        name = job["data_dir"]
        if name not in metadata:
            evidence = old["inputs"][name]
            backlog.verify_inputs(evidence)
            # Train bytes, not only timestamps, determine this mathematical screen.
            train = str(Path(name) / "train.npz")
            if backlog.service.sha(Path(train)) != evidence["files"][train]["sha256"]:
                raise ValueError("training input hash changed")
            inputs[name] = copy.deepcopy(evidence)
            metadata[name] = train_metadata(Path(name))
        assessment = screen(job, metadata[name], inputs[name]["population_prior"], budget)
        # Feasibility depends on c; c=0.1 is not a valid canary for c=0.5.
        family = job["family"] + "-c" + job["c"]
        canary = not assessment["blockers"] and family not in canaries
        if canary:
            canaries.add(family)
        job.update(
            parent_family=job["family"],
            family=family,
            canary=canary,
            status="blocked_preflight"
            if assessment["blockers"]
            else "pending"
            if canary
            else "waiting_canary",
            preflight=assessment,
        )
        if assessment["blockers"]:
            job["reason"] = "; ".join(assessment["blockers"])
    campaign = copy.deepcopy(old)
    campaign.update(
        source=str(backlog.ROOT),
        commit=backlog.clean_commit(),
        dependency_lock_sha256=backlog.service.sha(backlog.ROOT / "uv.lock"),
        jobs=jobs,
        inputs=inputs,
        total_fits=len(jobs),
        gpu_allowlist=[],
        max_gpu_jobs=0,
        max_cpu_jobs=1,
        reconciliation=dict(
            parent=str(parent.resolve()),
            parent_campaign_sha256=backlog.service.sha(parent / "campaign.json"),
            parent_status_sha256=backlog.service.sha(parent / "status.json"),
            replaces_labels=[j["label"] for j in jobs],
            additional_scientific_fits=0,
            policy="per-c canary after per-fit mathematical/resource screening; parent unchanged",
        ),
    )
    # Recheck statuses immediately before publication to catch concurrent adoption.
    current = {j["label"]: j["status"] for j in backlog.read(parent / "status.json")["jobs"]}
    if any(current[j["label"]] != status[j["label"]]["status"] for j in jobs):
        raise ValueError("parent job status changed during preparation")
    output.mkdir(parents=True, exist_ok=False)
    backlog.service.write(output / "campaign.json", campaign, exclusive=True)
    return campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign = prepare(args.output, args.parent)
    print(json.dumps(dict(Counter(j["status"] for j in campaign["jobs"])), sort_keys=True))


if __name__ == "__main__":
    main()
