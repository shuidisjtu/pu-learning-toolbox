"""Read-only campaign dashboard: device telemetry, saved epochs and bounded ETAs.

No training imports, checkpoint loading, queue writes or process control. Multiple
--output arguments show independent campaigns together without merging identities.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

EPOCH = re.compile(r"epoch_(\d+)_(.+)\.pt$")
NAMES = {"fashion_mnist": "F-MNIST", "twenty_newsgroups": "20News"}


def read(path):
    return json.loads(path.read_text())


def timestamp(value):
    try:
        dt = datetime.fromisoformat(value)
        return dt.replace(tzinfo=timezone.utc).timestamp() if dt.tzinfo is None else dt.timestamp()
    except (TypeError, ValueError):
        return None


def duration(seconds):
    if seconds is None:
        return "?"
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes = (seconds + 59) // 60
    return f"{minutes}m" if minutes < 60 else f"{minutes // 60}h{minutes % 60:02d}m"


def gpu_telemetry():
    """Device-wide measurements, not claims of exclusive process ownership."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
            check=True,
        )
        devices = {}
        for row in csv.reader(result.stdout.splitlines()):
            index = int(row[0])
            values = []
            for value in row[1:4]:
                try:
                    values.append(float(value.strip()))
                except ValueError:
                    values.append(None)
            if len(values) != 3:
                raise ValueError("incomplete GPU telemetry row")
            devices[index] = dict(util=values[0], used=values[1], total=values[2])
        return devices, None
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error:
        return {}, f"GPU telemetry unavailable: {type(error).__name__}"


def epoch_progress(root, job, now):
    """Use complete contiguous saved epochs; never count teachers as separate epochs."""
    expected = job.get("checkpoint_epochs")
    if not expected:
        return None
    run = root / "runs" / job["label"]
    # Reclamation can leave only selected snapshots; do not turn that into 2/E training.
    if any(run.rglob("manifest.json")):
        return "manifest detected", "ETA: awaiting controller acceptance"
    attempts = defaultdict(lambda: defaultdict(dict))
    for path in run.rglob("epoch_*.pt"):
        match = EPOCH.fullmatch(path.name)
        if match is None:
            continue
        try:
            modified = path.stat().st_mtime
        except OSError:
            continue  # Concurrent reclamation is normal; this monitor never owns these files.
        if now - modified < 2:
            continue
        attempts[path.parent][int(match[1])][match[2]] = modified
    if len(attempts) > 1:
        return "multiple attempts", "ETA: unavailable (attempts must not be pooled)"
    if not attempts:
        return f"saved 0/{expected}", "ETA: unavailable (preparation / first epoch)"
    epochs = next(iter(attempts.values()))
    components = 2 if job["method"] == "self_pu_full_meta" else 1
    points = []
    for number in range(1, expected + 1):
        if len(epochs.get(number, {})) != components:
            break
        points.append(max(epochs[number].values()))
    count = len(points)
    progress = f"saved {count}/{expected} epochs ({100 * count / expected:.0f}%)"
    if count == expected:
        return progress, "ETA: training finished; selection/test time not estimated"
    if count == 0:
        return "partial/retained snapshots", "ETA: unavailable (non-contiguous snapshots)"
    age = max(0, now - points[-1])
    if count < 4:
        return progress, f"ETA: unavailable (<3 epoch intervals); last save {duration(age)} ago"
    gaps = (
        [b - a for a, b in zip(points[-6:-1], points[-5:], strict=False)]
        if count >= 6
        else [b - a for a, b in zip(points[:-1], points[1:], strict=False)]
    )
    if any(gap <= 0 for gap in gaps):
        return progress, "ETA: unavailable (non-monotonic snapshot timestamps)"
    step = statistics.median(gaps)
    if age > max(120, 3 * step):
        return progress, f"ETA: uncertain (last save {duration(age)} ago; recent epoch {step:.1f}s)"
    remaining = max(0, (expected - count) * step - age)
    return progress, (
        f"train ETA~{duration(remaining)}; {step:.1f}s/epoch "
        f"(last {len(gaps)} intervals); last save {duration(age)} ago"
    )


def classical_progress(job, jobs, elapsed):
    budget = {"puet": "100 trees", "rp": "3 folds + refit"}.get(job["method"], "solver")
    if job["path"] == "cnn_feature_adapter":
        return budget + " (budget)", "ETA: unavailable (feature extraction/fit has no live counter)"
    samples = []
    keys = ("method", "dataset", "path", "mechanism", "c")
    for other in jobs:
        if other["status"] != "completed" or any(other.get(k) != job.get(k) for k in keys):
            continue
        start, end = timestamp(other.get("started_at")), timestamp(other.get("finished_at"))
        if start is not None and end is not None and end > start:
            samples.append(end - start)
    if len(samples) < 3 or elapsed is None:
        return budget + " (budget)", f"ETA: unavailable ({len(samples)}/3 matching completed fits)"
    median = statistics.median(samples)
    if elapsed >= median:
        return budget + " (budget)", f"ETA: uncertain (elapsed exceeds median {duration(median)})"
    return (
        budget + " (budget)",
        f"fit ETA~{duration(median - elapsed)} (matching median, n={len(samples)})",
    )


def render(outputs, *, now=None, devices=None, gpu_error=None, width=160):
    """Render one snapshot; missing/stale state is visible, never interpreted as idle."""
    now = time.time() if now is None else now
    local_time = datetime.fromtimestamp(now, tz=timezone.utc).astimezone()
    lines = [f"Survey live monitor | {local_time:%Y-%m-%d %H:%M:%S %Z}"]
    states = []
    for root in outputs:
        try:
            campaign = read(root / "campaign.json")
            revision = campaign.get("protocol_version", "survey-v3.1")
            try:
                state = read(root / "status.json")
            except FileNotFoundError:
                lines.append(f"{revision}: waiting for first controller status (input checks).")
                continue
            if not isinstance(state.get("jobs"), list):
                raise ValueError("invalid jobs list")
            for job in state["jobs"]:
                if not isinstance(job, dict) or "status" not in job:
                    raise ValueError("invalid job status")
                if job["status"] == "running" and not {
                    "label",
                    "method",
                    "dataset",
                    "path",
                    "mechanism",
                    "seed",
                    "c",
                }.issubset(job):
                    raise ValueError("incomplete running job")
            states.append((root, campaign, state, revision))
        except (OSError, ValueError, KeyError, AttributeError, TypeError) as error:
            lines.append(f"{root.name}: status unavailable ({type(error).__name__}); not idle")
    assignments = defaultdict(list)
    recoveries = defaultdict(list)
    for root, campaign, state, _ in states:
        recovery = campaign.get("recovery")
        if not isinstance(recovery, dict):
            continue
        parent, label = recovery.get("parent"), recovery.get("replaces_label")
        if not isinstance(parent, str) or not isinstance(label, str):
            continue
        update = timestamp(state.get("updated_at"))
        freshness = " STALE" if update is None or now - update > 60 else ""
        for job in state["jobs"]:
            if job.get("label") == label:
                device = "unassigned" if job.get("gpu") is None else f"GPU{job['gpu']}"
                recoveries[(Path(parent).resolve(), label)].append(
                    f"recovery={job['status']}{freshness} {device} ({root.name})"
                )
    for _, _, state, _ in states:
        for job in state["jobs"]:
            if job["status"] == "running" and job.get("gpu") is not None:
                assignments[job["gpu"]].append(f"{job['method']}/{job['dataset']}")
    lines.append("GPU   UTIL     VRAM used/total (GiB)   QUEUE ASSIGNMENT (device-wide telemetry)")
    if gpu_error:
        lines.append(gpu_error)
    for index, gpu in sorted((devices or {}).items()):
        util = "N/A" if gpu["util"] is None else f"{gpu['util']:.0f}%"
        vram = "/".join(
            "N/A" if gpu[k] is None else f"{gpu[k] / 1024:.1f}" for k in ("used", "total")
        )
        owner = ", ".join(assignments[index]) or "no running job assigned in this view"
        lines.append(f"{index:<5} {util:<8} {vram:<23} {owner}")
    for root, campaign, state, revision in states:
        jobs = state["jobs"]
        counts = Counter(j["status"] for j in jobs)
        update = timestamp(state.get("updated_at"))
        age = None if update is None else max(0, now - update)
        stale = " STALE STATUS!" if age is None or age > 60 else ""
        summary = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        lines.extend(
            [
                f"[{revision}] {root.name} | {summary}",
                f"  status age={duration(age)}{stale}; "
                f"GPU allowlist={state.get('gpu_allowlist', [])}",
            ]
        )
        recovery = campaign.get("recovery")
        if isinstance(recovery, dict):
            lines.append(
                "  RECOVERY ATTEMPT: separate from original failure; "
                "not an additional scientific fit or automatic family release."
            )
        if campaign.get("reconciliation"):
            lines.append(
                "  REPLACEMENT GRID: same parent fit identities, not additional experiments; "
                "blocked_preflight is inapplicable, not waiting for a GPU."
            )
        for job in jobs:
            if job["status"] != "running":
                continue
            start = timestamp(job.get("started_at"))
            elapsed = None if start is None else max(0, now - start)
            try:
                progress, eta = epoch_progress(root, job, now) or classical_progress(
                    job, jobs, elapsed
                )
            except OSError:
                progress, eta = "snapshot read unavailable", "ETA: unavailable (filesystem changed)"
            if stale:
                eta = "ETA: unavailable (stale controller status)"
            device = "CPU" if job.get("gpu") is None else f"GPU{job['gpu']}"
            dataset = NAMES.get(job["dataset"], job["dataset"])
            lines.append(
                f"  {device:<5} {job['method']:<12} {dataset:<13} {job['mechanism']:<10} "
                f"s{job['seed']}/c{job['c']} elapsed={duration(elapsed)} | {progress}"
            )
            lines.append(f"        {eta} | PID={job.get('pid', '?')}")
        if state.get("query_error"):
            lines.append(f"  Scheduler: {state['query_error']}")
        failures = [j for j in jobs if j["status"] == "failed"]
        for job in failures[-2:]:
            replacements = recoveries.get((root.resolve(), job["label"]), [])
            prefix = "ORIGINAL FAILED" if replacements else "FAILED"
            device = "unassigned" if job.get("gpu") is None else f"GPU{job['gpu']}"
            lines.append(f"  {prefix} {device} {job['label']}: {job.get('reason', 'see log')}")
            for replacement in replacements:
                lines.append(f"    {replacement}; original failure retained")
    lines.append("ETA~ is approximate, for CURRENT fit only; training ETA excludes selection/test.")
    lines.append(
        "Queue finish ETA: unavailable (mixed unmeasured jobs). CPU jobs do not reserve GPUs."
    )
    return "\n".join(
        line if len(line) <= width else line[: max(0, width - 3)] + "..." for line in lines
    )


def monitor(outputs, *, interval=5, once=False):
    while True:
        devices, error = gpu_telemetry()
        content = render(
            outputs,
            devices=devices,
            gpu_error=error,
            width=shutil.get_terminal_size((160, 40)).columns,
        )
        print(("" if once else "\033[2J\033[H") + content, flush=True)
        if once:
            return
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", action="append", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.interval <= 60:
        parser.error("interval must be between 1 and 60 seconds")
    monitor([p.resolve() for p in args.output], interval=args.interval, once=args.once)


if __name__ == "__main__":
    main()
