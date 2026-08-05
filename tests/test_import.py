"""Smoke test: every leaf subpackage is importable."""

import pytest

SUBPACKAGES = [
    "pu_toolbox.diagnostics",
    "pu_toolbox.registry",
    "pu_toolbox.estimators.classic",
    "pu_toolbox.estimators.risk",
    "pu_toolbox.estimators.bias_aware",
    "pu_toolbox.estimators.deep",
    "pu_toolbox.metrics",
    "pu_toolbox.model_selection",
    "pu_toolbox.workflows",
    "pu_toolbox.cli",
]


@pytest.mark.unit
@pytest.mark.parametrize("pkg", SUBPACKAGES)
def test_basic_import_all_subpackages(pkg: str) -> None:
    """Every subpackage should import without errors."""
    __import__(pkg)
