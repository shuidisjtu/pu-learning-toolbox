"""Guard lazy registration and the dynamic registry-to-estimator dependency."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from pu_toolbox import registry
from pu_toolbox.core.exceptions import RegistryError

pytestmark = pytest.mark.contract

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("get_algorithm('nnPU').__name__", "NonNegativePUClassifier"),
        ("get_metadata('nn-pu').name", "nnpu"),
        ("'nnpu' in r.get_algorithm_registry()", "True"),
    ],
)
def test_basic_builtin_lookup_in_a_fresh_process(query, expected):
    child = (
        "from pu_toolbox.registry import get_algorithm, get_metadata;"
        "from pu_toolbox.registry import registry as r;"
        "assert r._REGISTRY is None;"
        f"print({query})"
    )
    completed = subprocess.run(
        [sys.executable, "-c", child],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == expected


def test_edge_registered_classes_are_defined_in_the_higher_layers():
    modules = {
        registry.get_algorithm(entry.name).__module__ for entry in registry.list_algorithms()
    }
    assert modules
    outside = sorted(
        name
        for name in modules
        if not name.startswith(("pu_toolbox.estimators", "pu_toolbox.prior"))
    )
    assert not outside, f"registered classes outside the higher layers: {outside}"


def test_determ_lookup_does_not_reload_until_registry_is_cleared():
    registry.clear_registry()
    first = registry.get_algorithm_registry()
    prior = registry.get_algorithm("recpe")
    registry.register_method(registry.AlgorithmMetadata(name="custom", paper="Custom"), prior)
    registry.unregister_method("nnpu")

    with pytest.raises(RegistryError, match="Unknown algorithm"):
        registry.get_algorithm("nn-pu")
    assert registry.get_algorithm("custom") is prior

    registry.clear_registry()
    assert registry.get_algorithm_registry() == first


def test_param_unknown_method_name_is_rejected():
    with pytest.raises(RegistryError, match="unknown_method_name"):
        registry.get_algorithm("unknown_method_name")
