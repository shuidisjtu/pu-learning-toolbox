"""Fresh-process recipe matrix, truthful scope and refuse-before-training guards."""

import copy
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = load_script("profile_survey_extended_cnn_storage")
probe = load_script("profile_survey_cnn_storage")
pytestmark = pytest.mark.unit


def child_receipt(command):
    options = dict(zip(command[2::2], command[3::2], strict=True))
    return {
        "schema_version": 2,
        "profiles": [
            {
                "method": options["--method"],
                "seed": int(options["--seeds"]),
                "device": options["--device"],
                "input_shape": [24, 3, int(options["--image-size"]), int(options["--image-size"])],
                "encoder": {"base_channels": int(options["--base-channels"])},
                "parameters": {
                    key.lstrip("-").replace("-", "_"): value for key, value in options.items()
                },
                "trusted_pickle_roundtrip_verified": True,
                "formal_budget_approved": False,
                "source_files_sha256": {
                    "scripts/profile_survey_cnn_storage.py": driver.source_digest(driver.PROBE)
                },
            }
        ],
    }


def test_basic_determ_seven_recipes_three_seeds_have_explicit_variants_and_fresh_commands(
    monkeypatch,
):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps(child_receipt(command)), "")

    monkeypatch.setattr(driver.subprocess, "run", run)
    report = driver.profile_recipes(base_channels=2, image_size=8)
    assert len(calls) == 21 and report["owner_signatures"] == []
    assert report["formal_budget_approved"] is False and report["candidate_protocol_ref"] is None
    assert [(p["probe_recipe"], p["seed"]) for p in report["profiles"]] == [
        (recipe, seed) for recipe in driver.RECIPES for seed in (0, 1, 2)
    ]
    assert all(p["single_profile_isolated_process"] for p in report["profiles"])
    assert all(kwargs["cwd"] == ROOT and kwargs["timeout"] == 300 for _, kwargs in calls)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate_recipe",
        "unknown",
        "empty",
        "duplicate_seed",
        "late_bad_seed",
        "width",
        "dimension",
    ],
)
def test_param_edge_invalid_sweep_never_starts_even_one_child(monkeypatch, fault):
    kwargs = dict(recipes=["genpu_identity"], seeds=[0], base_channels=2, image_size=8)
    if fault == "duplicate_recipe":
        kwargs["recipes"] *= 2
    elif fault == "unknown":
        kwargs["recipes"] = ["unknown"]
    elif fault == "empty":
        kwargs["recipes"] = []
    elif fault == "duplicate_seed":
        kwargs["seeds"] = [0, 0]
    elif fault == "late_bad_seed":
        kwargs["seeds"] = [0, -1]
    elif fault == "width":
        kwargs["base_channels"] = True
    else:
        kwargs["image_size"] = 7

    def no_run(*args, **kwargs):
        pytest.fail("invalid sweep must not start subprocess")

    monkeypatch.setattr(driver.subprocess, "run", no_run)
    with pytest.raises(ValueError):
        driver.profile_recipes(**kwargs)


@pytest.mark.parametrize(
    "fault", ["nonzero", "timeout", "identity", "source", "admission", "width", "recipe", "nan"]
)
def test_edge_child_failure_or_mutated_receipt_is_not_retried_or_published(monkeypatch, fault):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if fault == "timeout":
            raise subprocess.TimeoutExpired(command, 300)
        if fault == "nonzero":
            return subprocess.CompletedProcess(command, 1, "", "injected failure")
        data = child_receipt(command)
        row = data["profiles"][0]
        if fault == "identity":
            row["seed"] = 99
        elif fault == "source":
            row["source_files_sha256"]["scripts/profile_survey_cnn_storage.py"] = "0" * 64
        elif fault == "admission":
            row["formal_budget_approved"] = True
        elif fault == "width":
            row["encoder"]["base_channels"] = 4
        elif fault == "recipe":
            row["parameters"]["generator_output"] = "tanh"
        else:
            row["bad_number"] = float("nan")
        return subprocess.CompletedProcess(command, 0, json.dumps(data), "")

    monkeypatch.setattr(driver.subprocess, "run", run)
    with pytest.raises((ValueError, RuntimeError, subprocess.TimeoutExpired)):
        driver.profile_recipes(recipes=["genpu_identity"], seeds=[0], base_channels=2, image_size=8)
    assert len(calls) == 1


@pytest.mark.parametrize("initialization", ["continue", "reinitialize"])
def test_basic_lzo_full_paid_cost_and_extra_evaluation_rows_are_measured(initialization):
    pytest.importorskip("torch")
    report = probe.profile_cnn_storage(
        "holistic_pu",
        base_channels=2,
        warmup_selection="lzo_positive_loss",
        pseudo_pn_initialization=initialization,
    )
    terminal = report["holistic_terminal_selection"]
    assert terminal["executed_warmup_epochs_"] == 3
    assert terminal["selected_warmup_epoch_"] in (2, 3)
    assert terminal["lzo_evaluated_rows_"] == 15 and terminal["lzo_forward_batches_"] == 3
    assert terminal["pseudo_pn_optimizer_reset_"] == (initialization == "reinitialize")
    assert report["optimizer_steps"] == 9 and report["stage_optimizer_steps"] == {
        "warmup": 6,
        "pseudo_pn": 3,
    }
    assert report["epoch_weights_bytes"]["count"] == 4
    assert [c["optimizer_steps"] for c in report["checkpoint_training_contexts"]] == [2, 4, 6, 9]


def test_basic_pixel_tanh_and_independent_clean_support_are_not_implicit_pu_labels():
    pytest.importorskip("torch")
    pixels = probe.profile_cnn_storage("genpu", base_channels=2, generator_output="tanh")
    assert pixels["input_coordinates"] == "tanh_of_synthetic_standard_normal"
    assert pixels["calibration_applied"] and pixels["training_view"] == "ts"
    assert pixels["optimizer_steps"] == 21 and pixels["clean_support"] is None
    reward = probe.profile_cnn_storage("pulns", base_channels=2)
    assert not reward["calibration_applied"] and reward["training_view"] == "os"
    assert reward["epoch_weights_bytes"] is None
    assert reward["clean_support"]["selection_test_labels_provided"] is False


def test_basic_determ_retained_tensors_are_alias_deduplicated_not_optimizer_or_peak_memory():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    encoder = torch.nn.Linear(2, 3)
    template = copy.deepcopy(encoder)
    network = torch.nn.Sequential(encoder, torch.nn.Linear(3, 1))
    model = SimpleNamespace(encoder=template, encoder_=encoder, model_=network)
    report = probe.retained_module_storage(model)
    assert report["unique_parameter_count"] == 22  # template9 + encoder9 + head4
    assert report["unique_storage_bytes_by_device"] == {"cpu": 88}
    assert report["module_attributes"] == {"encoder": 9, "encoder_": 9, "model_": 13}


@pytest.mark.parametrize(
    "text", ["VmRSS: 12 kB\nVmHWM: 32 kB\n", "VmRSS: 12 kB\n", "VmRSS: 12 MB\nVmHWM: 32 kB\n"]
)
def test_param_edge_proc_units_and_unavailable_memory_never_become_zero_budget(tmp_path, text):
    path = tmp_path / "status"
    path.write_text(text, encoding="utf-8")
    result = probe.host_resident_memory(path)
    if "32 kB" in text and "12 kB" in text:
        assert result == {"rss_bytes": 12 * 1024, "high_water_bytes": 32 * 1024}
    else:
        assert result == {"rss_bytes": None, "high_water_bytes": None}
    assert probe.host_resident_memory(tmp_path / "absent")["high_water_bytes"] is None


def test_basic_real_subprocess_returns_separate_pid_and_source_bound_complete_profile():
    pytest.importorskip("torch")
    report = driver.profile_recipes(
        recipes=["genpu_tanh"], seeds=[0], base_channels=2, image_size=8
    )
    row = report["profiles"][0]
    assert row["process_id"] != os.getpid() and row["single_profile_isolated_process"]
    assert row["epoch_weights_bytes"]["all_snapshots_replayed"]
    assert row["trusted_pickle_roundtrip_verified"] and row["cuda_peak"] is None


def test_edge_driver_source_changes_during_child_refuses_report(monkeypatch):
    digest = driver.source_digest
    calls = 0

    def changed(path):
        nonlocal calls
        if Path(path).resolve() == Path(driver.__file__).resolve():
            calls += 1
            if calls > 1:
                return "0" * 64
        return digest(path)

    monkeypatch.setattr(driver, "source_digest", changed)
    monkeypatch.setattr(
        driver.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, json.dumps(child_receipt(command)), ""
        ),
    )
    with pytest.raises(RuntimeError, match="source changed"):
        driver.profile_recipes(recipes=["genpu_identity"], seeds=[0], base_channels=2, image_size=8)
