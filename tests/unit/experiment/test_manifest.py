# tests/unit/experiment/test_manifest.py
import pytest

from pu_toolbox.experiment.manifest import load_manifest, write_manifest

pytestmark = pytest.mark.unit

REQUIRED = [
    "seed",
    "split_ref",
    "generation",
    "selection",
    "test_results",
    "elapsed",
    "failures",
    "resources",
]


def test_basic_write_and_read_roundtrip(tmp_path):
    p = tmp_path / "m.json"
    payload = {k: (0 if k != "elapsed" else 1.2) for k in REQUIRED}
    write_manifest(str(p), payload)
    assert load_manifest(str(p)) == payload


def test_param_missing_required_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="manifest.*missing"):
        write_manifest(str(tmp_path / "bad.json"), {"seed": 0})
