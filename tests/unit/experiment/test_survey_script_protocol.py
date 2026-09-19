# ruff: noqa: N803, F811
"""Protocol CLI checks and small end-to-end P2.0a evidence."""

import copy
import json

import numpy as np
import pytest
import torch
from _survey_script_helpers import make_splits, survey_script  # noqa: F401

from pu_toolbox.experiment import survey_comparison
from pu_toolbox.experiment.survey_protocol import PROTOCOL_PATH, load_protocol

pytestmark = pytest.mark.unit


def _splits(path, *, seed=0, dataset="spambase"):
    path.mkdir()
    make_splits(path)
    (path / "split_manifest.json").write_text(json.dumps({"dataset": dataset, "seed": seed}))


def _ghosted_comparison(monkeypatch):
    """Stand in for the shipped matrix with one mapping that covers no unit."""
    payload = json.loads(survey_comparison.COMPARISON_PATH.read_text(encoding="utf-8"))
    ghost = copy.deepcopy(payload["mappings"][0])
    ghost["mapping_id"] = "ghost_unit_mapping"
    ghost["result_selector"]["c_token"] = "0.9"
    payload["mappings"].append(ghost)
    # The loader binds the file to the protocol; the defect under test is the
    # coverage of an already-valid matrix, so the loader is stood in for.
    monkeypatch.setattr(
        survey_comparison, "load_comparison_protocol", lambda *args, **kwargs: payload
    )


def test_param_comparison_coverage_defect_aborts_before_any_split_is_read(
    survey_script, tmp_path, capsys, monkeypatch
):
    """A mapping that covers nothing is refused at startup, not left to the runs.

    Checked once, before seeds, encoders or output directories exist.  Per run
    it would be invisible: the unit it fails to cover is simply resolved by
    nothing, and the manifest would carry no comparison entry at all.
    """
    _ghosted_comparison(monkeypatch)

    assert survey_script.main(_args(tmp_path / "missing", tmp_path / "out")) == 1
    assert "ghost_unit_mapping" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_param_coverage_gate_follows_the_matrix_contents_not_its_path(
    survey_script, tmp_path, capsys, monkeypatch
):
    """A byte copy of the shipped matrix is still the shipped matrix.

    Keying the gate on the path let a copy skip it, while the run's comparison
    entry was still resolved against the shipped file -- so the copy bought no
    safety and lost the only check that catches a ghost mapping.
    """
    _ghosted_comparison(monkeypatch)
    copy_path = tmp_path / "matrix.json"
    copy_path.write_text(PROTOCOL_PATH.read_text(encoding="utf-8"))
    args = _args(tmp_path / "missing", tmp_path / "out")
    args[args.index("survey-v1.2")] = str(copy_path)

    assert survey_script.main(args) == 1
    assert "ghost_unit_mapping" in capsys.readouterr().err


def _args(data, out):
    return [
        str(data),
        "--protocol",
        "survey-v1.2",
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
    payload = json.loads((out / "c_0.5/seed_0/manifest.json").read_text())
    assert payload["protocol_version"] == "survey-v1.2"
    assert payload["budget"]["unit"] == "one_closed_form_fit"
    assert payload["formal_eligible"] is False
    assert payload["training_path"] == "native_2d"


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


def test_edge_oracle_native_cnn_cannot_silently_run_mlp(survey_script, tmp_path, capsys):
    args = [
        str(tmp_path / "missing"),
        "--protocol",
        "survey-v1",
        "--dataset",
        "cifar10",
        "--oracle",
        "--training-path",
        "native_cnn",
    ]
    assert survey_script.main(args) == 1
    assert "Phase 2" in capsys.readouterr().err


def test_edge_kldce_rejected_before_reading_splits(survey_script, tmp_path, capsys):
    args = _args(tmp_path / "missing", tmp_path / "out")
    args[args.index("upu")] = "kldce"
    assert survey_script.main(args) == 1
    assert "flip_probability" in capsys.readouterr().err
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
    assert json.loads((out / "c_0.5/seed_1/manifest.json").read_text())["split_ref"]["seed"] == 1


def test_determ_repeated_cli_preserves_label_and_representation_hashes(survey_script, tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    _splits(data)
    args = _args(data, out)
    assert survey_script.main(args) == 0
    path = out / "c_0.5/seed_0/manifest.json"
    first = json.loads(path.read_text())
    assert survey_script.main(args) == 0
    second = json.loads(path.read_text())
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
    payload = json.loads((out / "c_independent/seed_0/manifest.json").read_text())
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
        path = out / "c_0.5/seed_0/manifest.json"
        first = json.loads(path.read_text())
        assert first["representation"]["adapter"]["feature_dimension"] == 512
        assert survey_script.main(args) == 0
        second = json.loads(path.read_text())
        assert second["representation"]["adapter"]["cache_hit"] is True
        assert (
            first["representation"]["adapter"]["representation_sha256"]
            == second["representation"]["adapter"]["representation_sha256"]
        )
    finally:
        torch.set_num_threads(before)
