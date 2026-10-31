"""Compact read-only terminal dashboard. Never launches or stops training."""

import argparse
import csv
import json
import math
import re
import select
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pu_toolbox.experiment.queue_scheduling import (
    choose_pending,
    cpu_estimator_tail,
    dispatch_priority,
)

BASE = Path("/data2/user/zihenglin/pu-learning-toolbox/results/survey-tasks-20261027")
NAMES = {
    "fashion_mnist": "F-MNIST",
    "twenty_newsgroups": "20News",
    "connect_4": "Connect4",
    "cifar10": "CIFAR10",
    "mnist": "MNIST",
    "adni": "ADNI-sub",
    "spambase": "Spambase",
    "imdb": "IMDB",
}
METHODS = {
    "self_pu_full_meta": "SelfPU-meta",
    "self_pu": "SelfPU-abl",
    "pusb_kernel": "PUSB-kernel",
    "pn_oracle": "PN-oracle",
    "robust_pu": "Robust-PU",
    "holistic_pu": "Holistic-PU",
    "split_pu": "Split-PU",
    "dist_pu": "Dist-PU",
}
PATHS = {"native_cnn": "CNN", "cnn_feature_adapter": "FEAT", "native_2d": "2D"}
MECHS = {"scar": "SCAR", "sar_lbe_a": "SAR-A", "sar_lbe_b": "SAR-B", None: "PN"}
EPOCHS = {
    "pn_oracle": 200,
    "nnpu": 200,
    "dist_pu": 200,
    "self_pu": 200,
    "self_pu_full_meta": 200,
    "pulda": 120,
    "robust_pu": 30,
    "split_pu": 40,
    "holistic_pu": 130,
    "lagam": 20,
    "pulns": 20,
    "genpu": 100,
}
VIEWS = ("running", "pending", "completed", "failed", "held")
VIEW_KEYS = {str(index + 1): view for index, view in enumerate(VIEWS)}


def read_states():
    states = []
    for directory in sorted(BASE.glob("gpu-batch-*")):
        path = directory / "queue_status.json"
        if path.exists():
            state = json.loads(path.read_text())
            state["directory"] = directory
            states.append(state)
    return states


def visible_jobs(states):
    jobs = [j for s in states for j in s["jobs"]]
    retries = {j["retry_of"] for j in jobs if j.get("retry_of")}
    fixed = {
        (j["method"], j.get("dataset", "cifar10"), j["path"])
        for j in jobs
        if j.get("preflight_evidence")
    }
    retries.update(
        j["root"]
        for j in jobs
        if j["status"] == "failed"
        and (j["method"], j.get("dataset", "cifar10"), j["path"]) in fixed
    )
    historical = [j for j in jobs if j["status"] == "failed" and j["root"] in retries]
    return [j for j in jobs if j not in historical], len(historical)


def gpu_status():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=4,
        check=True,
    )
    devices = {
        int(r[0]): dict(uuid=r[1].strip(), memory=float(r[2]), util=int(r[3]))
        for r in csv.reader(result.stdout.splitlines())
    }
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=4,
        check=True,
    )
    owners = {}
    for row in csv.reader(processes.stdout.splitlines()):
        if len(row) == 2 and row[1].strip().isdigit():
            owners.setdefault(row[0].strip(), set()).add(int(row[1]))
    for device in devices.values():
        device["pids"] = owners.get(device["uuid"], set())
    return devices


def progress(job):
    """Display only the active fit, never a retained checkpoint of a completed fit."""
    root = Path(job["root"])
    finished = set()
    for path in root.rglob("manifest.json"):
        try:
            m = json.loads(path.read_text())
            if m.get("selection") and not m.get("failures"):
                finished.add((m["seed"], m.get("c_requested_token")))
        except (OSError, ValueError, KeyError):
            pass
    facts = dict(
        done=f"{len(finished)}/{job.get('expected_runs', 15)}", seed="-", c="-", epoch="-", age="-"
    )
    if job.get("expected_runs") == 1 and job.get("expected_keys"):
        facts["seed"], facts["c"] = map(
            str, job["expected_keys"][0][5:6] + job["expected_keys"][0][4:5]
        )
    if job["status"] != "running":
        return facts
    remaining = [k for k in job.get("expected_keys", []) if (k[5], k[4]) not in finished]
    if remaining:
        facts["seed"], facts["c"] = str(remaining[0][5]), str(remaining[0][4])
    latest = None
    for path in root.rglob("*.pt"):
        run_dir = next((p for p in path.parents if p.name.startswith("seed_")), None)
        if run_dir is None or (run_dir / "manifest.json").exists():
            continue
        try:
            modified = path.stat().st_mtime
            if latest is None or modified > latest[0]:
                latest = modified, path
        except OSError:
            continue
    if latest is None:
        facts["epoch"] = "GAN/prePN" if job["method"] == "genpu" else "load/prep"
        return facts
    modified, path = latest
    for part in path.parts:
        if part.startswith("seed_"):
            facts["seed"] = part[5:]
        if part.startswith("c_"):
            facts["c"] = part[2:]
    match = re.search(r"epoch_(\d+)_", path.name)
    if match:
        prefix = "PN" if job["method"] == "genpu" else ""
        facts["epoch"] = f"{prefix}{int(match[1])}/{EPOCHS.get(job['method'], '?')}"
    age = max(0, int(time.time() - modified))
    facts["age"] = f"{age}s" if age < 60 else f"{age // 60}m"
    return facts


def table(columns, rows):
    def line(values):
        return "  ".join(
            str(v)[:width].ljust(width) for (_, width), v in zip(columns, values, strict=False)
        ).rstrip()

    return [
        line([key for key, _ in columns]),
        line(["-" * n for _, n in columns]),
        *[line([row.get(key, "-") for key, _ in columns]) for row in rows],
    ]


def duration(seconds):
    """Compact wall time, distinct from checkpoint freshness."""
    minutes = max(0, int(seconds // 60))
    return f"{minutes // 60}h{minutes % 60:02d}m"


def timing_detail(job, jobs, now=None):
    """Estimate only single-fit wall time from compatible completed single fits."""
    if job["status"] != "running" or not job.get("started_at"):
        return ""
    now = now or datetime.now(timezone.utc)
    try:
        elapsed = (now - datetime.fromisoformat(job["started_at"])).total_seconds()
    except (ValueError, TypeError):
        return "elapsed=unknown | ETA unavailable"
    if job.get("expected_runs") != 1:
        return f"batch elapsed={duration(elapsed)} | multi-fit ETA unavailable"
    # Do not pool different datasets, protocols, training sources or correction campaigns.
    identity = (
        "dataset",
        "method",
        "path",
        "mechanism",
        "source",
        "protocol_sha256",
        "campaign_id",
    )
    samples = []
    for previous in jobs:
        if previous["status"] != "completed" or previous.get("expected_runs") != 1:
            continue
        if any(previous.get(key) != job.get(key) for key in identity):
            continue
        if previous.get("expected_recipe") != job.get("expected_recipe"):
            continue
        try:
            seconds = (
                datetime.fromisoformat(previous["ended_at"])
                - datetime.fromisoformat(previous["started_at"])
            ).total_seconds()
            if seconds > 0:
                samples.append(seconds)
        except (KeyError, TypeError, ValueError):
            continue
    prefix = f"elapsed={duration(elapsed)}"
    if len(samples) < 3:
        return prefix + " | ETA unavailable (<3 comparable fits)"
    median = statistics.median(samples)
    if elapsed >= median:
        return prefix + f" | ETA uncertain; past median {duration(median)} (n={len(samples)})"
    return (
        prefix
        + f" | ETA~{duration(median - elapsed)} remaining"
        + f" (fit median={duration(median)}, n={len(samples)})"
    )


def render(
    view="running", page=0, width=None, height=None, states=None, devices=None, details=True
):
    terminal = shutil.get_terminal_size((120, 31))
    width = max(30, (width or terminal.columns) - 1)
    height = max(12, (height or terminal.lines) - 1)
    lines = [f"PU SURVEY | {datetime.now().astimezone():%m-%d %H:%M:%S} | 3s refresh"]
    try:
        states = read_states() if states is None else states
        jobs, historic = visible_jobs(states)
        current = max(states, key=lambda s: s["updated_at"])
        counts = {key: sum(j["status"] == key for j in current["jobs"]) for key in VIEWS}
        delay = max(
            0,
            int(
                (
                    datetime.now(timezone.utc) - datetime.fromisoformat(current["updated_at"])
                ).total_seconds()
            ),
        )
        lines.append(
            f"Latest queue {current.get('directory', Path('queue')).name}: "
            f"{counts['running']} RUN | {counts['pending']} WAIT | "
            f"{counts['completed']} DONE | {counts['failed']} FAIL | "
            + str(sum(j["status"] == "blocked" for j in current["jobs"]))
            + " BLOCK"
            + f" | {counts['held']} HOLD"
        )
        lines.append(
            f"Heartbeat {delay}s ago"
            + (" [STALE: inspect scheduler]" if delay > 60 else "")
            + f" | {historic} superseded failures hidden"
            + (
                f" | GPU allowlist={current['scheduler']['gpu_indices']}"
                if current.get("scheduler", {}).get("gpu_indices") is not None
                else ""
            )
        )
    except (OSError, ValueError, KeyError) as error:
        return "\n".join(line[:width] for line in [*lines, str(error)])
    try:
        devices = gpu_status() if devices is None else devices
        gpu_error = None
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        devices = {}
        gpu_error = str(error)
    selected = [j for j in jobs if j["status"] == view]
    if view == "running":
        selected.sort(key=lambda j: (j.get("gpu", {}).get("index", 99), j["method"]))
    elif view == "completed":
        selected.sort(key=lambda j: j.get("ended_at", ""), reverse=True)
    expanded = details and view == "running"
    capacity = max(1, (height - 14) // (2 if expanded else 1))
    pages = max(1, math.ceil(len(selected) / capacity))
    page = max(0, page) % pages
    selected = selected[page * capacity : (page + 1) * capacity]
    lines.extend(
        [
            f"VIEW {view.upper()} | page {page + 1}/{pages} | "
            "1:run 2:wait 3:done 4:fail 5:held | n/p:page d:detail",
            "",
        ]
    )
    columns = [
        ("DEV", 4),
        ("DATA", 9),
        ("METHOD", 11),
        ("MECH", 5),
        ("DONE", 6),
        ("S", 1),
        ("c", 4),
        ("PHASE", 9),
    ]
    if width < 70:
        columns = [(key, size) for key, size in columns if key not in ("S", "c")]
    if width < 60:
        columns = [(key, size) for key, size in columns if key != "MECH"]
    if width < 50:
        columns = [(key, size) for key, size in columns if key != "PHASE"]
    if width >= 85:
        columns[3:3] = [("PATH", 4)]
        columns[1:1] = [("SRC", 2)]
    if width >= 95:
        columns.extend([("U%", 3), ("GiB", 5)])
    if width >= 110:
        columns.extend([("AGE", 5), ("PID", 7)])
    rows = []
    own_pids = {j.get("pid") for j in jobs if j["status"] == "running"}
    for job in selected:
        info = progress(job)
        index = job.get("gpu", {}).get("index")
        device = devices.get(index, {})
        shared = bool(device.get("pids", set()) - own_pids)
        if (
            device
            and job["status"] == "running"
            and job.get("pid") not in device.get("pids", set())
        ):
            info["epoch"] = "CPU-solve" if cpu_estimator_tail(job) else "no-CUDA"
        dev = "CPU" if index is None else str(index) + ("*" if shared else "")
        rows.append(
            {
                "DEV": dev if view == "running" else "-",
                "SRC": "R!"
                if job.get("campaign_canary")
                else "R"
                if job.get("campaign_id")
                else "B",
                "DATA": NAMES.get(job.get("dataset", "cifar10"), job.get("dataset", "?")),
                "METHOD": METHODS.get(job["method"], job["method"]),
                "PATH": PATHS.get(job["path"], job["path"]),
                "MECH": MECHS.get(job.get("mechanism"), job.get("mechanism"))
                if not job["oracle"]
                else "PN",
                "DONE": info["done"],
                "S": info["seed"],
                "c": info["c"],
                "PHASE": info["epoch"],
                "U%": device.get("util", "-"),
                "GiB": f"{device['memory'] / 1024:.1f}" if device else "-",
                "AGE": info["age"],
                "PID": job.get("pid", "-") if view == "running" else "-",
            }
        )
    rendered_rows = table(columns, rows)
    lines.extend(rendered_rows[:2])
    for job, row, info_row in zip(selected, rows, rendered_rows[2:], strict=True):
        lines.append(info_row)
        if expanded:
            timing = timing_detail(job, jobs)
            if row["PHASE"] == "GAN/prePN":
                stage = "GAN epoch unreported (no PN snapshot)"
            elif row["PHASE"].startswith("PN"):
                stage = f"PN snapshot {row['PHASE'][2:]}; age={row['AGE']}"
            elif row["PHASE"] == "no-CUDA":
                stage = "Process alive; no GPU context observed"
            elif row["PHASE"] == "CPU-solve":
                stage = "CPU estimator; no CUDA context (GPU may overlap)"
            else:
                stage = f"snapshot age={row['AGE']}"
            lines.append(f"  {stage} | {timing}")
    if not rows:
        lines.append("(none)")
    lines.append("")
    if devices:
        lines.append(
            "GPU load: "
            + " | ".join(f"{i}:{d['util']}%/{d['memory'] / 1024:.1f}G" for i, d in devices.items())
        )
    elif gpu_error:
        lines.append("GPU query unavailable; queue still shown.")
    if view == "running":
        pending = sorted([j for j in jobs if j["status"] == "pending"], key=dispatch_priority)
        active = [j for j in jobs if j["status"] == "running"]
        preferred = choose_pending(pending, active, "exclusive")
        if preferred is not None:
            pending = [preferred, *[j for j in pending if j is not preferred]]
        lines.append(
            "NEXT (free-card candidates): "
            + "; ".join(
                f"{NAMES.get(j['dataset'], j['dataset'])}/"
                f"{METHODS.get(j['method'], j['method'])}/"
                f"{MECHS.get(j.get('mechanism'), 'PN')}"
                for j in pending[:2]
            )
        )
    lines.append("* = shared GPU; U%/GiB are whole-device, not this task alone.")
    lines.append("SRC B=baseline R=repair R!=canary; HOLD waits for a passed full-budget canary.")
    if expanded:
        lines.append(
            "ETA~ is wall-time estimate, NOT measured epoch%; "
            "GAN epochs are not exported by live workers."
        )
    own_counts = {}
    for job in jobs:
        if job["status"] == "running" and job.get("gpu"):
            i = job["gpu"]["index"]
            own_counts[i] = own_counts.get(i, 0) + 1
    shared_own = ",".join(f"GPU{i}={n}" for i, n in sorted(own_counts.items()) if n > 1)
    lines.append(
        ("Own concurrent workers: " + shared_own + " | " if shared_own else "")
        + "Ctrl+B,D detach; q stops monitor only."
    )
    return "\n".join(line[:width] for line in lines[:height])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--view", choices=VIEWS, default="running")
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--compact", action="store_true", help="Hide per-task timing details")
    args = parser.parse_args()
    view = args.view
    page = max(0, args.page - 1)
    details = not args.compact
    terminal_state = None
    if not args.once and sys.stdin.isatty():
        import termios
        import tty

        terminal_state = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
    try:
        while True:
            print(
                ("" if args.once else "\033[H\033[2J")
                + render(view, page, args.width, args.height, details=details),
                flush=True,
            )
            if args.once:
                break
            if terminal_state is None:
                time.sleep(3)
                continue
            ready, _, _ = select.select([sys.stdin], [], [], 3)
            if not ready:
                continue
            key = sys.stdin.read(1)
            if key == "q":
                break
            if key in VIEW_KEYS:
                view = VIEW_KEYS[key]
                page = 0
            elif key == "d":
                details = not details
            elif key == "n":
                page += 1
            elif key == "p":
                page = max(0, page - 1)
    except KeyboardInterrupt:
        pass
    finally:
        if terminal_state is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, terminal_state)


if __name__ == "__main__":
    main()
