"""Resource measurement helpers for experiment artifacts."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

#: Reserve for the one same-seed retry a candidate gets.  A retry writes into
#: its own ``attempt-N`` directory, and the abandoned attempt is never cleaned
#: up, so both copies can be on disk at once.
DEFAULT_CHECKPOINT_ATTEMPTS = 2


def aggregate_resource_usage(manifests: Iterable[dict]) -> dict[str, Any]:
    """Aggregate per-seed manifests into the protocol's full tuning cost."""
    items = list(manifests)
    if not items:
        raise ValueError("resource aggregation requires at least one manifest.")
    configuration_costs = []
    tuning_elapsed = 0.0
    peaks = []
    seeds = []
    generation_elapsed = 0.0
    for manifest in items:
        resources = manifest.get("resources")
        if not isinstance(resources, dict):
            raise ValueError("each manifest must contain a resources mapping.")
        seed = manifest.get("seed")
        seeds.append(seed)
        configuration_costs.extend(
            {"seed": seed, **cost} for cost in resources["single_configuration_costs"]
        )
        tuning_elapsed += float(resources["tuning"]["elapsed_seconds"])
        generation_elapsed += float(resources["data_generation_elapsed_seconds"])
        peak = resources.get("peak_gpu_memory_bytes")
        if peak is not None:
            peaks.append(int(peak))
    return {
        "schema_version": "1.0",
        "single_configuration_costs": configuration_costs,
        "total_tuning_elapsed_seconds": tuning_elapsed,
        "peak_gpu_memory_bytes": max(peaks, default=None),
        "data_generation_elapsed_seconds": generation_elapsed,
        "seed_count": len(set(seeds)),
        "seeds": sorted(set(seeds)),
    }


def begin_peak_gpu_memory_measurement(model, params: dict) -> str | None:
    """Reset torch's peak allocator counter for the candidate device."""
    if "device" not in params and not hasattr(model, "device"):
        return None
    device_hint = params.get("device", getattr(model, "device", None))
    if (
        device_hint is not None
        and str(device_hint) not in ("auto",)
        and not str(device_hint).startswith("cuda")
    ):
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        device = str(device_hint) if device_hint not in (None, "auto") else "cuda"
        torch.cuda.reset_peak_memory_stats(device)
        return device
    except (ImportError, RuntimeError, AssertionError):
        return None


def peak_gpu_memory_bytes(device: str | None) -> int | None:
    """Read peak allocated CUDA memory, or None when GPU measurement is unavailable."""
    if device is None:
        return None
    try:
        import torch

        return int(torch.cuda.max_memory_allocated(device))
    except (ImportError, RuntimeError, AssertionError):
        return None


def runtime_environment() -> dict[str, Any]:
    """Return the software and hardware facts required by experiment artifacts."""
    environment: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "numpy_version": np.__version__,
        "scikit_learn_version": _package_version("scikit-learn"),
        "torch_version": None,
        "cuda_available": False,
        "cuda_runtime_version": None,
        "cudnn_version": None,
        "gpu_devices": [],
        "gpu_driver_version": None,
    }
    try:
        import torch

        environment["torch_version"] = torch.__version__
        environment["cuda_available"] = bool(torch.cuda.is_available())
        environment["cuda_runtime_version"] = torch.version.cuda
        environment["cudnn_version"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            environment["gpu_devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
                }
                for index in range(torch.cuda.device_count())
            ]
            environment["gpu_driver_version"] = _nvidia_driver_version()
    except (ImportError, RuntimeError, AssertionError) as exc:
        environment["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    environment["executable"] = sys.executable
    return environment


def disk_free_bytes(directory: str | Path) -> int:
    """Free bytes on the filesystem holding ``directory``.

    The directory the guard cares about does not exist yet -- checkpointing
    creates it -- so this walks up to the nearest existing ancestor rather than
    creating anything just to measure it.
    """
    path = Path(directory)
    while not path.exists():
        parent = path.parent
        if parent == path:
            break
        path = parent
    return shutil.disk_usage(path).free


def checkpoint_disk_requirement(
    *,
    bytes_per_component: int,
    epochs: int,
    components: int,
    candidates: int,
    attempts: int = DEFAULT_CHECKPOINT_ATTEMPTS,
) -> int:
    """Bytes needed to keep every candidate's per-epoch checkpoints on disk.

    All arguments are counts of things, so each must be a positive integer;
    ``bool`` is rejected explicitly because it is an ``int`` subclass and a
    ``True`` here would silently mean "one".
    """
    values = {
        "bytes_per_component": bytes_per_component,
        "epochs": epochs,
        "components": components,
        "candidates": candidates,
        "attempts": attempts,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"checkpoint disk requirement needs a positive integer {name}")
    return bytes_per_component * epochs * components * candidates * attempts


def disk_space_preflight(
    *,
    required_bytes: int,
    directory: str | Path,
    free_bytes: int | None = None,
) -> dict[str, Any]:
    """Report whether the checkpoint directory can hold a run's checkpoints.

    ``free_bytes`` is injectable so the decision can be tested without
    depending on the host's actual free space; callers in production leave it
    unset and get a real probe.
    """
    if free_bytes is None:
        free_bytes = disk_free_bytes(directory)
    return {
        "required_bytes": int(required_bytes),
        "free_bytes": int(free_bytes),
        "missing_bytes": max(0, int(required_bytes) - int(free_bytes)),
        "directory": str(directory),
        "ready": int(free_bytes) >= int(required_bytes),
    }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _nvidia_driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    versions = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    return ",".join(versions) or None
