"""Peak RSS units and explicit unavailable platforms without a resource import."""

import sys
from types import SimpleNamespace

import pytest

from pu_toolbox.experiment.resources import process_lifetime_peak_rss_bytes

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("platform,expected", [("linux", 7 * 1024), ("darwin", 7)])
def test_basic_param_rss_unit_conversion(platform, expected, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "resource",
        SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda who: SimpleNamespace(ru_maxrss=7)),
    )
    monkeypatch.setattr(sys, "platform", platform)
    assert process_lifetime_peak_rss_bytes() == expected


def test_edge_unsupported_platform_reports_null_not_an_import_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "resource", None)
    assert process_lifetime_peak_rss_bytes() is None


def test_determ_same_counter_is_not_relabelled_as_a_fit_only_peak(monkeypatch):
    calls = []

    def read(who):
        calls.append(who)
        return SimpleNamespace(ru_maxrss=19)

    monkeypatch.setitem(sys.modules, "resource", SimpleNamespace(RUSAGE_SELF=42, getrusage=read))
    monkeypatch.setattr(sys, "platform", "linux")
    assert process_lifetime_peak_rss_bytes() == process_lifetime_peak_rss_bytes() == 19 * 1024
    assert calls == [42, 42]
