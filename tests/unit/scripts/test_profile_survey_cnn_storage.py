"""Real CNN storage scope, replay and refusal boundaries."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "cnn_storage_probe", Path(__file__).parents[3] / "scripts/profile_survey_cnn_storage.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
pytestmark = pytest.mark.unit


@pytest.mark.parametrize("method", probe.METHODS)
def test_basic_actual_cnn_snapshots_and_trusted_estimator_roundtrip(method):
    pytest.importorskip("torch")
    report = probe.profile_cnn_storage(method, base_channels=2)
    assert report["status"] == "synthetic_cnn_storage_probe_not_formal_admission"
    assert report["encoder"]["matches_toolbox_default_width"] is False
    assert report["input_shape"] == [24, 3, 8, 8]
    assert report["training_view"] == ("os" if method in {"pan", "pulns", "holistic_pu"} else "ts")
    assert report["encoder"]["backbone"] == ("cnn13_no_bn" if method == "gradpu" else "cnn13")
    snapshots = report["epoch_weights_bytes"]
    if method == "pulns":
        assert (
            snapshots is None
            and report["epoch_checkpoint_status"] == "unavailable_no_epoch_callback"
        )
        assert report["checkpoint_training_contexts"] == []
        support = report["clean_support"]
        assert set(support["train_ids"]).isdisjoint(support["support_ids"])
        assert (
            support["labels"] == "clean_pn" and support["formal_support_budget_approved"] is False
        )
    else:
        assert (
            snapshots["count"]
            == {
                "pulda": 2,
                "gradpu": 1,
                "robust_pu": 2,
                "split_pu": 4,
                "pan": 2,
                "genpu": 2,
                "holistic_pu": 4,
            }[method]
        )
        assert snapshots["all_snapshots_replayed"] and snapshots["final_scores_match"]
        assert snapshots["total"] >= snapshots["max"] >= snapshots["min"] > 0
    assert report["estimator_pickle_bytes"] > 0 and report["trusted_pickle_roundtrip_verified"]
    assert report["inference_model_parameter_count"] > 0 and report["optimizer_steps"] > 0
    assert report["cuda_peak"] is None and report["formal_budget_approved"] is False
    assert all(len(value) == 64 for value in report["source_files_sha256"].values())
    assert report["retained_module_storage"]["unique_storage_bytes_by_device"]["cpu"] > 0
    if report["stage_optimizer_steps"] is not None:
        assert sum(report["stage_optimizer_steps"].values()) == report["optimizer_steps"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "arguments",
    [
        {"method": "puet"},
        {"base_channels": 0},
        {"base_channels": True},
        {"image_size": 7},
        {"seed": -1},
        {"seed": True},
        {"device": "auto"},
    ],
)
def test_param_invalid_inputs_refused_before_training(arguments):
    kwargs = {"method": "pulda", **arguments}
    with pytest.raises(ValueError):
        probe.profile_cnn_storage(**kwargs)


def test_edge_requested_cuda_never_silently_falls_back(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        probe.profile_cnn_storage("pulda", device="cuda")


def test_determ_seed_and_source_identity_repeat_without_claiming_timing_determinism():
    pytest.importorskip("torch")
    first = probe.profile_cnn_storage("pulda", base_channels=2, seed=2)
    second = probe.profile_cnn_storage("pulda", base_channels=2, seed=2)
    for key in (
        "parameters",
        "optimizer_steps",
        "inference_model_parameter_count",
        "source_files_sha256",
    ):
        assert first[key] == second[key]
    assert first["epoch_weights_bytes"] == second["epoch_weights_bytes"]


def test_cli_duplicate_seeds_refused_before_training(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["probe", "--seeds", "1", "1"])
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 2


def test_edge_source_change_during_training_refuses_ambiguous_receipt(monkeypatch):
    pytest.importorskip("torch")
    from pu_toolbox.utils import serialization

    original = serialization.file_hash
    calls = 0

    def changed_source(path):
        nonlocal calls
        if Path(path).resolve() == Path(probe.__file__).resolve():
            calls += 1
            if calls > 1:
                return "0" * 64
        return original(path)

    monkeypatch.setattr(serialization, "file_hash", changed_source)
    with pytest.raises(RuntimeError, match="source changed during training"):
        probe.profile_cnn_storage("pulda", base_channels=2)
