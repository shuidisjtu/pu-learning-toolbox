"""Recover one failed GradPU canary without editing its parent campaign.

Full frozen recipe, separate artifacts and source identity. Optional sharing is
limited to the parent's *currently running* worker on one of its allowed cards;
unknown GPU processes, stale parent status and insufficient memory fail closed.
This does not release/rewrite the parent's held family or count as a new fit.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts import run_survey_backlog_campaign as backlog
from scripts._survey_host import lock_controller


def prepare(output, parent, label, gpu):
    if output.exists():
        raise ValueError("recovery output exists; never overwrite history")
    old = backlog.read(parent / "campaign.json")
    status = backlog.read(parent / "status.json")
    job = next(j for j in status["jobs"] if j["label"] == label)
    if job["status"] != "failed" or not job["canary"] or job["method"] != "gradpu":
        raise ValueError("recovery requires a failed GradPU canary")
    if gpu not in old["gpu_allowlist"] or gpu not in backlog.ALLOWED_GPUS:
        raise ValueError("recovery cannot expand the parent's four-card scope")
    campaign = copy.deepcopy(old)
    original = next(j for j in old["jobs"] if j["label"] == label)
    campaign.update(
        source=str(backlog.ROOT),
        commit=backlog.clean_commit(),
        dependency_lock_sha256=backlog.service.sha(backlog.ROOT / "uv.lock"),
        jobs=[dict(original, status="pending")],
        total_fits=1,
        inputs={original["data_dir"]: old["inputs"][original["data_dir"]]},
        gpu_allowlist=[gpu],
        max_gpu_jobs=1,
        max_cpu_jobs=0,
        gradpu_activation_memory="cpu_offload",
        recovery=dict(
            parent=str(parent.resolve()),
            parent_campaign_sha256=backlog.service.sha(parent / "campaign.json"),
            parent_failed_job=job,
            replaces_label=label,
            additional_scientific_fits=0,
            family_release="not automatic; full-canary acceptance required first",
        ),
    )
    for evidence in campaign["inputs"].values():
        backlog.verify_inputs(evidence, full_hash=True)
    output.mkdir(parents=True)
    backlog.service.write(output / "campaign.json", campaign, exclusive=True)
    return campaign


def gpu_processes(gpu):
    def query(args):
        return subprocess.check_output(
            ["nvidia-smi", *args, "--format=csv,noheader,nounits"], text=True, timeout=5
        )

    rows = list(csv.reader(query(["--query-gpu=index,uuid,memory.free"]).splitlines()))
    _, uuid, free = next(row for row in rows if int(row[0]) == gpu)
    apps = csv.reader(query(["--query-compute-apps=gpu_uuid,pid"]).splitlines())
    return int(free), {int(pid) for card, pid in apps if card.strip() == uuid.strip()}


def admission(campaign, *, allow_parent_sharing):
    parent = Path(campaign["recovery"]["parent"])
    if (
        backlog.service.sha(parent / "campaign.json")
        != campaign["recovery"]["parent_campaign_sha256"]
    ):
        return False, "parent campaign changed"
    gpu = campaign["gpu_allowlist"][0]
    free, pids = gpu_processes(gpu)
    if free < 32 * 1024 or backlog.memory_available() < 192 * 1024**3:
        return False, "waiting for 32 GiB GPU / 192 GiB host RAM headroom"
    if not pids:
        return True, "idle card"
    if not allow_parent_sharing:
        return False, "waiting for idle card"
    status = backlog.read(parent / "status.json")
    age = (
        datetime.now(timezone.utc) - datetime.fromisoformat(status["updated_at"])
    ).total_seconds()
    if not 0 <= age <= 45:
        return False, "parent status stale"
    known = {j["pid"] for j in status["jobs"] if j["status"] == "running" and j.get("gpu") == gpu}
    if pids != known or not all(owned_process(pid) for pid in pids):
        return False, "unknown GPU processes; sharing refused"
    return True, "sharing only the live parent worker, not an additional GPU"


def owned_process(pid):
    if sys.platform != "linux":
        return False
    return Path(f"/proc/{pid}").stat().st_uid == os.getuid()


def run(output, *, allow_parent_sharing=False):
    if (output / "status.json").exists():
        raise ValueError("existing recovery requires reconciliation, never implicit restart")
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    _controller_lock = lock_controller(output / "controller.lock")
    if (output / "status.json").exists():
        raise ValueError("existing recovery requires reconciliation, never implicit restart")
    campaign = backlog.read(output / "campaign.json")
    if campaign["source"] != str(backlog.ROOT) or campaign["commit"] != backlog.clean_commit():
        raise ValueError("recovery source changed")
    if backlog.service.sha(backlog.ROOT / "uv.lock") != campaign["dependency_lock_sha256"]:
        raise ValueError("recovery dependency lock changed")
    if len(campaign["jobs"]) != 1 or len(campaign["gpu_allowlist"]) != 1:
        raise ValueError("this controller only recovers a single canary")
    job = campaign["jobs"][0]
    process = None
    ready = 0
    while True:
        reason = None
        if process is None:
            try:
                allowed, reason = admission(campaign, allow_parent_sharing=allow_parent_sharing)
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                allowed, reason = False, str(error)
            ready = ready + 1 if allowed else 0
            if ready >= 2:
                gpu = campaign["gpu_allowlist"][0]
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=str(gpu),
                    PYTHONPATH=str(backlog.ROOT),
                    OMP_NUM_THREADS="2",
                    OPENBLAS_NUM_THREADS="2",
                    MKL_NUM_THREADS="2",
                )
                with (output / f"{job['label']}.log").open("x") as log:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            str(backlog.ROOT / "scripts/run_survey_backlog_campaign.py"),
                            "--output",
                            str(output),
                            "--worker",
                            job["label"],
                        ],
                        cwd=backlog.ROOT,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                job.update(status="running", gpu=gpu, pid=process.pid, started_at=backlog.now())
                print("RECOVERY_START", job["label"], reason, flush=True)
        elif process.poll() is not None:
            code = process.returncode
            try:
                if code:
                    raise ValueError(f"worker exit {code}")
                job.update(backlog.accept(output, job, campaign), status="completed")
            except (OSError, ValueError, KeyError) as error:
                job.update(status="failed", reason=str(error))
            job.update(exit_code=code, finished_at=backlog.now())
        backlog.service.write(
            output / "status.json",
            dict(
                jobs=[job],
                controller_pid=os.getpid(),
                updated_at=backlog.now(),
                query_error=reason,
                gpu_allowlist=campaign["gpu_allowlist"],
                formal_admission=False,
                recovery=campaign["recovery"],
            ),
        )
        if job["status"] in {"completed", "failed"}:
            print("RECOVERY_END", job["status"], flush=True)
            return
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--gpu", type=int, choices=backlog.ALLOWED_GPUS)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--allow-parent-sharing", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.output, allow_parent_sharing=args.allow_parent_sharing)
    else:
        if args.parent is None or args.label is None or args.gpu is None:
            parser.error("preparation requires --parent, --label and --gpu")
        prepare(args.output, args.parent, args.label, args.gpu)


if __name__ == "__main__":
    main()
