"""Storage probes validate recovery, without claiming formal candidate admission."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "storage_probe", Path(__file__).parents[3] / "scripts/profile_survey_candidate_storage.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
pytestmark = pytest.mark.unit


def test_pulda_both_stages_are_counted_and_weights_recover():
    pytest.importorskip("torch")
    report = probe.profile_storage("pulda")
    epochs = report["epoch_weights_bytes"]
    assert epochs["epoch_count"] == 2
    assert epochs["total"] >= 2 * epochs["min"] > 0
    assert epochs["roundtrip_verified"] and report["pickle_roundtrip_verified"]
    assert report["gpu_resource_audit"] == "not_performed"
    assert report["status"] == "synthetic_storage_probe_not_formal_admission"


def test_puet_has_no_neural_epoch_budget():
    report = probe.profile_storage("puet")
    assert report["epoch_weights_bytes"] is None
    assert report["estimator_pickle_bytes"] > 0 and report["pickle_roundtrip_verified"]
    assert report["gpu_resource_audit"] == "not_applicable"


@pytest.mark.parametrize("input_dim", [0, True, 1.5])
def test_invalid_dimension_fails_before_training(input_dim):
    with pytest.raises(ValueError, match="positive integer"):
        probe.profile_storage("puet", input_dim=input_dim)


def test_unknown_method_fails_before_training():
    with pytest.raises(ValueError, match="storage probe supports"):
        probe.profile_storage("unknown")


@pytest.mark.parametrize("method", ["pan", "rp", "pulns", "genpu", "holistic_pu"])
def test_basic_new_candidate_trusted_full_state_roundtrip(method):
    if method != "rp":
        pytest.importorskip("torch")
    report = probe.profile_storage(method, seed=1)
    assert report["pickle_roundtrip_verified"]
    assert report["estimator_pickle_bytes"] > 0
    if method in {"pan", "holistic_pu", "genpu"}:
        epochs = report["epoch_weights_bytes"]
        assert epochs["epoch_count"] == {"pan": 2, "holistic_pu": 4, "genpu": 1}[method]
        assert epochs["total"] == epochs["max"] * epochs["epoch_count"]
        assert epochs["roundtrip_verified"]
    else:
        assert report["epoch_weights_bytes"] is None
    assert report["elapsed_seconds_including_imports"] >= 0
    assert report["training_view"] == ("ts" if method == "genpu" else "os")
    if method == "pulns":
        assert report["clean_support"] == {
            "rows": 8,
            "labels": "clean_pn",
            "disjoint_ids": True,
        }
    else:
        assert report["clean_support"] is None
    if method != "rp":
        assert report["optimizer_steps"] > 0


def test_cli_all_methods_multiple_seeds_emit_strict_json(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["probe", "--method", "all", "--seeds", "0", "2"])
    monkeypatch.setattr(
        probe,
        "profile_storage",
        lambda method, *, seed, input_dim: {"method": method, "seed": seed},
    )
    probe.main()
    profiles = json.loads(capsys.readouterr().out)["profiles"]
    assert [(p["method"], p["seed"]) for p in profiles] == [
        (method, seed) for method in probe.METHODS for seed in (0, 2)
    ]


@pytest.mark.parametrize("arguments", [["--seeds", "1", "1"], ["--seed", "1", "--seeds", "2"]])
def test_invalid_seed_lists_fail_before_training(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["probe", *arguments])
    with pytest.raises(SystemExit) as exc:
        probe.main()
    assert exc.value.code == 2
