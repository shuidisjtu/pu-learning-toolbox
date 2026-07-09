"""Smoke test: every subpackage is importable."""

import pytest

SUBPACKAGES = [
    "pu_toolbox",
    "pu_toolbox.core",
    "pu_toolbox.registry",
    "pu_toolbox.source_adapters",
    "pu_toolbox.datasets",
    "pu_toolbox.preprocessing",
    "pu_toolbox.prior",
    "pu_toolbox.propensity",
    "pu_toolbox.losses",
    "pu_toolbox.estimators",
    "pu_toolbox.estimators.classic",
    "pu_toolbox.estimators.risk",
    "pu_toolbox.estimators.bias_aware",
    "pu_toolbox.estimators.deep",
    "pu_toolbox.metrics",
    "pu_toolbox.model_selection",
    "pu_toolbox.advisor",
]


@pytest.mark.parametrize("pkg", SUBPACKAGES)
def test_import_all_subpackages(pkg: str) -> None:
    """Every subpackage should import without errors."""
    __import__(pkg)
