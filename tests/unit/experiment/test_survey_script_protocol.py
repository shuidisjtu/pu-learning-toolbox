# ruff: noqa: N803, F811
"""Protocol CLI checks and small end-to-end P2.0a evidence."""

import builtins
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from _survey_script_helpers import make_splits, run_manifest, survey_script  # noqa: F401

from pu_toolbox.experiment.survey_protocol import load_protocol

pytestmark = pytest.mark.unit


def _splits(path, *, seed=0, dataset="spambase"):
    path.mkdir()
    make_splits(path)
    (path / "split_manifest.json").write_text(json.dumps({"dataset": dataset, "seed": seed}))


@pytest.mark.parametrize("version", ["survey-v3.2"])
def test_basic_param_versioned_training_requires_no_comparison_modules_or_files(
    survey_script, tmp_path, monkeypatch, version
):
    """Scientific execution stays available when literature metadata is unavailable."""
    data, out = tmp_path / "data", tmp_path / "out"
    _splits(data)
    real_import, real_read = builtins.__import__, Path.read_text

    def guarded_import(name, *args, **kwargs):
        if name.rsplit(".", 1)[-1] in {"survey_comparison", "survey_profile_map"}:
            raise AssertionError(f"training imported optional comparison module: {name}")
        return real_import(name, *args, **kwargs)

    def guarded_read(path, *args, **kwargs):
        if path.name.startswith("survey_comparison"):
            raise AssertionError(f"training read optional comparison metadata: {path.name}")
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(Path, "read_text", guarded_read)
    args = _args(data, out)
    args[args.index("survey-v3.2")] = version
    assert survey_script.main(args) == 0
    payload = json.loads(run_manifest(out, "c_0.5", "seed_0").read_text(encoding="utf-8"))
    assert payload["protocol_version"] == version
    assert payload["budget"]["unit"] == "one_closed_form_fit"
    assert set(payload["selection"]) == {"PA", "OA"}
    assert "comparison" not in payload
    assert "comparison_registration_status" not in payload


def _args(data, out):
    return [
        str(data),
        "--protocol",
        "survey-v3.2",
        "--dataset",
        "spambase",
        "--method",
        "upu",
        "--class-prior",
        "0.3",
        "--c",
        "0.5",
        "--out-dir",
        str(out),
    ]


def test_basic_versioned_cli_records_protocol_and_actual_budget(survey_script, tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    _splits(data)
    assert survey_script.main(_args(data, out)) == 0
    payload = json.loads(run_manifest(out, "c_0.5", "seed_0").read_text(encoding="utf-8"))
    assert payload["protocol_version"] == "survey-v3.2"
    assert payload["budget"]["unit"] == "one_closed_form_fit"
    assert payload["formal_eligible"] is False
    assert payload["training_path"] == "native_2d"
    assert payload["code_identity"]["code_commit_state"] in {"recorded", "unavailable"}
    assert payload["dependency_lock"]["state"] == "recorded"
    assert len(payload["dependency_lock"]["sha256"]) == 64


@pytest.mark.parametrize("batch", [None, {"batch_id": "b" * 64}])
def test_param_run_identity_records_unavailable_or_retains_the_batch_reference(
    survey_script, tmp_path, monkeypatch, batch
):
    from pu_toolbox.experiment import batch_evidence

    code = {"code_commit": None, "code_commit_dirty": None, "code_commit_state": "unavailable"}
    lock = {"path": "uv.lock", "sha256": None, "state": "unavailable"}
    monkeypatch.setattr(batch_evidence, "code_identity", lambda: code)
    monkeypatch.setattr(batch_evidence, "dependency_lock", lambda: lock)
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "seed": 0,
                "split_ref": {},
                "generation": {},
                "selection": {},
                "test_results": {},
                "elapsed": 0,
                "failures": [],
                "resources": {},
            }
        ),
        encoding="utf-8",
    )

    survey_script._record_run_context(path, c_token="0.5", batch=batch)

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["c_requested_token"] == "0.5"
    if batch is None:
        assert manifest["code_identity"] == code
        assert manifest["dependency_lock"] == lock
    else:
        assert manifest["batch"] == batch
        assert "code_identity" not in manifest
        assert "dependency_lock" not in manifest


def test_param_locked_constructor_rejected_before_reading_splits(survey_script, tmp_path, capsys):
    args = _args(tmp_path / "missing", tmp_path / "out") + ["--model-params", '{"loss":"logistic"}']
    assert survey_script.main(args) == 1
    assert "protocol-locked" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_param_locked_candidate_rejected_before_reading_splits(survey_script, tmp_path, capsys):
    pool = tmp_path / "pool.json"
    pool.write_text('[{"max_iter":1}]')
    assert (
        survey_script.main(
            _args(tmp_path / "missing", tmp_path / "out") + ["--candidates", str(pool)]
        )
        == 1
    )
    assert "protocol-locked" in capsys.readouterr().err


def test_edge_oracle_native_cnn_requires_image_splits(survey_script, tmp_path, capsys):
    args = [
        str(tmp_path / "missing"),
        "--protocol",
        "survey-v3.2",
        "--dataset",
        "cifar10",
        "--oracle",
        "--training-path",
        "native_cnn",
    ]
    assert survey_script.main(args) == 1
    assert "missing split files" in capsys.readouterr().err


def test_edge_kldce_requires_splits_before_runtime_prior_binding(survey_script, tmp_path, capsys):
    args = _args(tmp_path / "missing", tmp_path / "out")
    args[args.index("upu")] = "kldce"
    assert survey_script.main(args) == 1
    assert "missing split files" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_param_dataset_and_seed_mismatches_rejected(survey_script, tmp_path, capsys):
    data = tmp_path / "data"
    _splits(data, seed=2)
    assert survey_script.main(_args(data, tmp_path / "out")) == 1
    assert "seed does not match" in capsys.readouterr().err


def test_edge_empty_seeds_rejected_before_data_read(survey_script, tmp_path, capsys):
    assert survey_script.main(_args(tmp_path / "missing", tmp_path / "out") + ["--seeds", ""]) == 1
    assert "non-empty" in capsys.readouterr().err


def test_basic_per_seed_root_loads_distinct_split_manifests(survey_script, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _splits(data / "split_0", seed=0)
    _splits(data / "split_1", seed=1)
    out = tmp_path / "out"
    assert survey_script.main(_args(data, out) + ["--seeds", "0,1"]) == 0
    assert (
        json.loads(run_manifest(out, "c_0.5", "seed_1").read_text(encoding="utf-8"))["split_ref"][
            "seed"
        ]
        == 1
    )


def test_determ_repeated_cli_preserves_label_and_representation_hashes(survey_script, tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    _splits(data)
    args = _args(data, out)
    assert survey_script.main(args) == 0
    path = run_manifest(out, "c_0.5", "seed_0")
    first = json.loads(path.read_text(encoding="utf-8"))
    assert survey_script.main(args) == 0
    second = json.loads(path.read_text(encoding="utf-8"))
    assert first["representation"] == second["representation"]
    assert first["generation"] == second["generation"]


def test_param_linear_pusb_not_a_protocol_pilot_row(survey_script, tmp_path, capsys):
    args = _args(tmp_path / "missing", tmp_path / "out")
    args[args.index("upu")] = "pusb"
    assert survey_script.main(args) == 1
    assert "unit missing" in capsys.readouterr().err


def test_basic_torch_oracle_is_oa_only_and_c_independent(survey_script, tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    _splits(data)
    protocol = load_protocol()
    protocol["method_profiles"]["pn_oracle"]["params"]["max_epochs"] = 2
    protocol["budgets"]["minibatch"]["epochs"] = 2
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(protocol))
    args = [
        str(data),
        "--protocol",
        str(path),
        "--dataset",
        "spambase",
        "--oracle",
        "--c",
        "0.1,0.3,0.5",
        "--out-dir",
        str(out),
    ]
    assert survey_script.main(args) == 0
    payload = json.loads(run_manifest(out, "c_independent", "seed_0").read_text(encoding="utf-8"))
    assert set(payload["selection"]) == {"OA"}
    assert payload["c_independent"] is True
    assert "noncanonical_protocol_matrix" in payload["protocol_deviation"]


def test_basic_cifar_adapter_script_assembles_and_reuses_cache(survey_script, tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    rng = np.random.RandomState(4)
    for index, role in enumerate(("train", "pu_val", "clean_val", "test")):
        n = 8
        np.savez(
            data / f"{role}.npz",
            X=rng.randint(0, 256, (n, 3, 32, 32), dtype=np.uint8),
            y=np.array([1, 0] * 4),
            indices=np.arange(index * n, (index + 1) * n),
        )
    (data / "split_manifest.json").write_text(json.dumps({"dataset": "cifar10", "seed": 0}))
    args = _args(data, out)
    args[args.index("spambase")] = "cifar10"
    args += ["--adapter-cache", str(tmp_path / "cache")]
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert survey_script.main(args) == 0
        path = run_manifest(out, "c_0.5", "seed_0")
        first = json.loads(path.read_text(encoding="utf-8"))
        assert first["representation"]["adapter"]["feature_dimension"] == 512
        assert survey_script.main(args) == 0
        second = json.loads(path.read_text(encoding="utf-8"))
        assert second["representation"]["adapter"]["cache_hit"] is True
        assert (
            first["representation"]["adapter"]["representation_sha256"]
            == second["representation"]["adapter"]["representation_sha256"]
        )
    finally:
        torch.set_num_threads(before)
