"""Shared device-resolution helpers (CUDA auto-detection)."""

from __future__ import annotations

import sys

import pytest
import torch

pytestmark = pytest.mark.unit


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return None


def test_basic_resolve_name_none_cpu(no_cuda):
    from pu_toolbox.core.device import resolve_device_name

    assert resolve_device_name(None) == "cpu"


def test_basic_resolve_device_none_cpu(no_cuda):
    from pu_toolbox.core.device import resolve_device

    assert resolve_device(None) == torch.device("cpu")


def test_param_auto_equals_none(no_cuda):
    from pu_toolbox.core.device import resolve_device_name

    assert resolve_device_name("auto") == resolve_device_name(None)


def test_param_explicit_strings_passthrough():
    from pu_toolbox.core.device import resolve_device_name

    assert resolve_device_name("cuda:0") == "cuda:0"
    assert resolve_device_name("cpu") == "cpu"


def test_param_cuda_selected_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    from pu_toolbox.core.device import resolve_device, resolve_device_name

    assert resolve_device_name(None) == "cuda"
    assert resolve_device_name("auto") == "cuda"
    assert resolve_device("auto").type == "cuda"


def test_edge_torch_missing_falls_back_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)

    from pu_toolbox.core.device import resolve_device_name

    assert resolve_device_name(None) == "cpu"


def test_edge_resolve_device_without_torch_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)

    from pu_toolbox.core.device import resolve_device

    with pytest.raises(ImportError):
        resolve_device(None)


def test_determ_same_input_stable(no_cuda):
    from pu_toolbox.core.device import resolve_device_name

    assert resolve_device_name(None) == "cpu"
    assert resolve_device_name(None) == "cpu"
