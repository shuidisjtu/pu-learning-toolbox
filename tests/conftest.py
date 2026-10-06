"""Shared pytest fixtures and low-noise terminal reporting."""

# ruff: noqa: N806

from __future__ import annotations

import numpy as np
import pytest

from tests.helpers import make_scar_data, set_test_seed


@pytest.fixture(scope="session", autouse=True)
def _fixed_seed():
    """Ensure deterministic tests via a fixed global seed."""
    set_test_seed(42)


@pytest.fixture
def rng():
    """Return a fresh numpy RandomState for per-test use."""
    return np.random.RandomState(42)


@pytest.fixture
def simple_x_y_pu(rng):
    """Small SCAR dataset: 2×100 samples, separation=4.0, c=0.5."""
    return make_scar_data(rng, n=100, c=0.5, n_features=5, separation=4.0)


def pytest_report_teststatus(report, config):
    """Suppress passed-test progress; failures retain pytest's normal output."""
    if report.passed:
        return "", "", ""
    return None


def pytest_sessionfinish(session, exitstatus):
    """Print one success line only when the run has no warnings or failures."""
    if exitstatus != pytest.ExitCode.OK:
        return
    terminalreporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminalreporter is None or terminalreporter.stats.get("warnings"):
        return
    terminalreporter.write_line("All Pass")
