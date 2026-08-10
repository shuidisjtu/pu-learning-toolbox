"""Torch device resolution helpers (CUDA auto-detection)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def resolve_device_name(device: str | None) -> str:
    """Resolve a user-supplied device to a concrete device name.

    ``None`` / ``"auto"`` maps to ``"cuda"`` when torch is installed and
    CUDA is available, otherwise ``"cpu"``; any explicit string (e.g.
    ``"cuda:0"``) is returned unchanged. torch is imported lazily so
    core installs without the torch extra still work.
    """
    if device is not None and device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def resolve_device(device: str | None) -> torch.device:
    """Resolve ``device`` to a ``torch.device`` instance.

    The caller is expected to be on a torch code path already; without
    torch installed this raises ``ImportError``.
    """
    import torch

    return torch.device(resolve_device_name(device))
