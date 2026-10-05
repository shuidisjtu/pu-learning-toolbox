"""Public collection is bounded, verified and never overwrites an input."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "collect_data", Path(__file__).parents[3] / "scripts/collect_survey_public_data.py"
)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)
pytestmark = pytest.mark.unit


def configure(monkeypatch, *, expected):
    monkeypatch.setattr(
        collector,
        "resources",
        lambda name: [("test.zip", "https://example.invalid/archive", "sha256", expected)],
    )
    monkeypatch.setattr(collector, "urlopen", lambda *a, **k: io.BytesIO(b"raw"))


def test_basic_determ_download_receipt_and_verified_reuse(tmp_path, monkeypatch):
    configure(monkeypatch, expected=hashlib.sha256(b"raw").hexdigest())
    receipt = collector.collect("mnist", tmp_path)
    monkeypatch.setattr(collector, "urlopen", lambda *a, **k: pytest.fail("unexpected download"))
    assert collector.collect("mnist", tmp_path) == receipt
    assert receipt["files"][0]["publisher_checksum_verified"]
    assert not receipt["extracted"]


def test_param_publisher_mismatch_not_published(tmp_path, monkeypatch):
    configure(monkeypatch, expected="0" * 64)
    with pytest.raises(ValueError, match="checksum mismatch"):
        collector.collect("mnist", tmp_path)
    assert not list(tmp_path.rglob("*.zip"))
    assert not list(tmp_path.rglob(".download-*"))


def test_edge_local_receipt_is_not_publisher_checksum(tmp_path, monkeypatch):
    configure(monkeypatch, expected=None)
    first = collector.collect("connect_4", tmp_path)
    second = collector.collect("connect_4", tmp_path)
    assert first == second
    assert not second["files"][0]["publisher_checksum_verified"]
    target = tmp_path / "connect_4/test.zip"
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="existing file preserved"):
        collector.collect("connect_4", tmp_path)
    assert target.read_bytes() == b"changed"
    assert json.loads((target.parent / "provenance.json").read_text()) == first


def test_adni_not_allowed():
    with pytest.raises(ValueError, match="ADNI"):
        collector.resources("adni")
