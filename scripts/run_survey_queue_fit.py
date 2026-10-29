#!/usr/bin/env python3
"""One pinned-source correction fit; record D52 evidence in its own campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args(argv)
    job = json.loads(args.worker.read_text())
    source = Path(job["source"])
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=source).strip():
        raise ValueError("correction source became dirty")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if commit != job["source_code_commit"]:
        raise ValueError("correction source commit changed")
    for path, expected in [
        (source / "pu_toolbox/experiment/support_checkpoints.py", job["recorder_sha256"]),
        (source / "uv.lock", job["dependency_lock_sha256"]),
        (Path(job["support_receipt"]), job["support_receipt_sha256"]),
    ]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("correction recorder or support receipt changed")
    if Path(job["root"]).exists() and any(Path(job["root"]).iterdir()):
        raise ValueError("refusing to overwrite correction output")
    sys.path.insert(0, str(source))
    from pu_toolbox.experiment import batch_evidence
    from pu_toolbox.experiment.survey_protocol import digest, load_protocol
    from pu_toolbox.experiment.survey_recipe_registry import canonical_digest, load_registry

    protocol = load_protocol(source / "pu_toolbox/experiment/survey_protocol_v3.json")
    registry_path = source / "pu_toolbox/experiment/survey_recipe_registry_v1.json"
    registry = load_registry(registry_path)
    if digest(protocol) != job["protocol_sha256"]:
        raise ValueError("correction protocol changed")
    packet = batch_evidence.build_evidence(
        unit=dict(
            dataset=job["dataset"],
            method=job["method"],
            training_path=job["path"],
            mechanism="scar",
        ),
        seeds=job["seeds"],
        c_tokens=job["c_tokens"],
        protocol_version=protocol["protocol_version"],
        protocol_sha256=digest(protocol),
        recipe_registry=dict(
            version=registry["recipe_registry_version"], sha256=canonical_digest(registry)
        ),
    )
    if packet["code_commit"] != commit or packet["code_commit_dirty"]:
        raise ValueError("batch identity does not match pinned correction source")
    evidence = batch_evidence.write(Path(job["campaign_root"]) / batch_evidence.DIRECTORY, packet)
    support = str(Path(job["support_data"]) / "split_{seed}")
    command = [
        sys.executable,
        "-u",
        str(source / "scripts/run_survey_experiment.py"),
        job["data_root"],
        "--protocol",
        "survey-v3.0",
        "--dataset",
        job["dataset"],
        "--method",
        job["method"],
        "--training-path",
        job["path"],
        "--seeds",
        ",".join(map(str, job["seeds"])),
        "--c",
        ",".join(job["c_tokens"]),
        "--class-prior",
        str(job["class_prior"]),
        "--device",
        "cuda",
        "--out-dir",
        job["root"],
        "--recipe-registry",
        str(registry_path),
        "--recipe-binding-schema-version",
        "survey-recipe-binding-2",
        "--support-dir",
        support,
        "--split-ref",
        support + "/support_manifest.json",
        "--os-or-ts",
        job["os_or_ts"],
        "--reclaim-unselected-checkpoints",
        "--adapter-cache",
        str(Path(job["campaign_root"]) / "adapter-cache" / job["dataset"]),
        "--batch-evidence",
        str(evidence),
    ]
    print("CORRECTION_FIT", job["label"], "CANARY", job["campaign_canary"], flush=True)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
