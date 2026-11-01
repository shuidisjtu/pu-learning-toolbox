# ruff: noqa: F811

"""Opt-in CVIR CLI: real prior binding and refusal of unsafe batch requests."""

import json
from pathlib import Path

import pytest
from _survey_script_helpers import (  # noqa: F401 - pytest fixtures
    make_splits,
    run_manifest,
    survey_script,
)

pytestmark = pytest.mark.unit


def request(data_dir, out_dir):
    return [
        str(data_dir),
        "--method",
        "cvir",
        "--class-prior",
        "0.4",
        "--c",
        "0.5",
        "--cvir-population-prior-source",
        "synthetic generation metadata",
        "--checkpoint-bytes-per-component",
        "65536",
        "--model-params",
        json.dumps(
            {
                "hidden_dim": 4,
                "warm_start_epochs": 1,
                "max_epochs": 2,
                "batch_size": 8,
            }
        ),
        "--out-dir",
        str(out_dir),
    ]


@pytest.mark.parametrize("view", [None, "os", "ts"])
def test_basic_determ_cli_binds_prior_checkpoints_selection_and_reclaims(
    survey_script, tmp_path, capsys, view
):
    pytest.importorskip("torch")
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_splits(data_dir)
    manifests = []
    for attempt in range(2):
        output = tmp_path / str(attempt)
        argv = request(data_dir, output) + ["--reclaim-unselected-checkpoints"]
        if view is not None:
            argv += ["--os-or-ts", view]
        assert survey_script.main(argv) == 0
        path = run_manifest(output, "c_0.5", "seed_0")
        manifest = json.loads(path.read_text())
        candidate = manifest["candidate_runs"][0]
        provenance = candidate["training_provenance"]
        binding = provenance["population_prior_binding"]
        resolved = "ts" if view is None else view
        assert binding["population_prior"] == 0.4
        assert binding["n_train"] == 18 and binding["n_labeled_positive"] == 3
        assert binding["risk_pool_positive_prior"] == pytest.approx(
            0.4 if resolved == "ts" else (0.4 * 18 - 3) / 15
        )
        assert binding["prior_source"] == "synthetic generation metadata"
        assert not binding["source_verified"] and not binding["uses_hidden_training_labels"]
        assert provenance["population_prior_forwarded_to_fit"] is False
        assert manifest["calibration_applied"] is (resolved == "ts")
        assert manifest["formal_eligible"] is False
        assert not manifest["failures"]
        assert set(manifest["test_results"]) == {"PA", "OA"}
        assert manifest["selection"]["PA"]["class_prior"]["population"] == 0.4
        assert len(candidate["epoch_checkpoints"]) == 3
        remaining = [c for c in candidate["epoch_checkpoints"] if Path(c["path"]).exists()]
        assert 1 <= len(remaining) <= 2
        ledger = json.loads((path.parent / "method_ledger_entry.json").read_text())
        assert ledger["run_view"] == f"{resolved}-compatible"
        manifests.append(manifest)
    assert manifests[0]["test_results"] == manifests[1]["test_results"]
    assert "Exploratory CVIR evidence only" in capsys.readouterr().out


@pytest.mark.parametrize(
    "extra,missing,error",
    [
        ([], ["--cvir-population-prior-source", "--checkpoint-bytes-per-component"], "requires"),
        ([], ["--cvir-population-prior-source"], "nonempty"),
        ([], ["--checkpoint-bytes-per-component"], "positive"),
        (["--cvir-population-prior-source", " "], [], "nonempty"),
        (["--checkpoint-bytes-per-component", "0"], [], "positive"),
        (["--checkpoint-bytes-per-component", "-1"], [], "positive"),
        ([], ["--class-prior"], "population --class-prior"),
        (["--class-prior", "nan"], [], "population --class-prior"),
        (["--class-prior", "1"], [], "population --class-prior"),
        (["--labeling-mechanism", "sar_lbe_a"], [], "SCAR only"),
        (["--labeling-mechanism", "sar_lbe_b"], [], "SCAR only"),
        (["--protocol", "survey-v3.0"], [], "without --protocol/--oracle"),
        (["--oracle"], [], "without --protocol/--oracle"),
        (["--method", "nnpu"], [], "without --protocol/--oracle"),
    ],
)
def test_param_request_gates_precede_split_loading(
    survey_script, tmp_path, monkeypatch, capsys, extra, missing, error
):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid CVIR request reached the data loader")

    monkeypatch.setattr(survey_script, "load_split_parts", forbidden)
    output = tmp_path / "out"
    argv = request(tmp_path / "absent", output)
    for flag in missing:
        index = argv.index(flag)
        del argv[index : index + 2]
    assert survey_script.main(argv + extra) == 1
    assert error in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize(
    "parameters",
    [
        {"unlabeled_positive_prior": 0.3},
        {"prior_source": "alpha not pi"},
        {"input_sampling": "ts"},
        {"max_epochs": 0},
        {"warm_start_epochs": -1},
    ],
)
def test_param_candidate_contract_is_checked_before_output(
    survey_script, tmp_path, capsys, parameters
):
    pytest.importorskip("torch")
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_splits(data_dir)
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps([{}, parameters]))
    output = tmp_path / "out"
    assert survey_script.main(request(data_dir, output) + ["--candidates", str(candidates)]) == 1
    assert "invalid CVIR candidate contract" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("seeds", ["", "0,0"])
def test_edge_empty_or_duplicate_seeds_do_not_produce_artifacts(
    survey_script, tmp_path, capsys, seeds
):
    pytest.importorskip("torch")
    make_splits(tmp_path)
    output = tmp_path / "out"
    assert survey_script.main(request(tmp_path, output) + ["--seeds", seeds]) == 1
    assert "nonempty, distinct seeds" in capsys.readouterr().err
    assert not output.exists()


def test_edge_existing_later_run_stops_batch_before_first_fit(
    survey_script, tmp_path, monkeypatch, capsys
):
    pytest.importorskip("torch")
    make_splits(tmp_path)
    output = tmp_path / "out"
    later = survey_script._run_directory(
        output,
        mechanism="scar",
        c_value=survey_script.CValue(0.5, "0.5"),
        seed=1,
        is_oracle=False,
        run_view="ts",
    )
    later.mkdir(parents=True)
    sentinel = later / "receipt.json"
    sentinel.write_text("preserve")

    def forbidden(*args, **kwargs):
        raise AssertionError("started first run before checking the full batch")

    monkeypatch.setattr(survey_script, "run_one", forbidden)
    assert survey_script.main(request(tmp_path, output) + ["--seeds", "0,1"]) == 1
    assert "refusing to overwrite CVIR evidence" in capsys.readouterr().err
    assert sentinel.read_text() == "preserve"
    assert not (later.parent / "seed_0").exists()


def test_edge_underestimated_snapshot_bound_fails_and_leaves_evidence(
    survey_script, tmp_path, capsys
):
    pytest.importorskip("torch")
    make_splits(tmp_path)
    output = tmp_path / "out"
    assert (
        survey_script.main(request(tmp_path, output) + ["--checkpoint-bytes-per-component", "1"])
        == 1
    )
    assert "all candidate runs failed" in capsys.readouterr().err
    manifest = json.loads(run_manifest(output, "c_0.5", "seed_0").read_text())
    assert "exceeded declared" in json.dumps(manifest["failures"])
    assert list(output.rglob("*.pt"))


def test_basic_cli_disk_preflight_includes_largest_candidate_and_warm_start(
    survey_script, tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    from pu_toolbox.experiment import resources

    make_splits(tmp_path)
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps([{}, {"max_epochs": 4}]))
    recorded = []
    original = resources.disk_space_preflight

    def audit(**kwargs):
        recorded.append(kwargs["required_bytes"])
        return original(**kwargs)

    monkeypatch.setattr(resources, "disk_space_preflight", audit)
    output = tmp_path / "out"
    assert survey_script.main(request(tmp_path, output) + ["--candidates", str(candidates)]) == 0
    assert recorded == [
        resources.checkpoint_disk_requirement(
            bytes_per_component=65536,
            epochs=5,
            components=1,
            candidates=2,
        )
    ]
    manifest = json.loads(run_manifest(output, "c_0.5", "seed_0").read_text())
    assert [len(c["epoch_checkpoints"]) for c in manifest["candidate_runs"]] == [3, 5]


def test_param_constructor_alpha_cannot_be_overwritten_by_population_cli(
    survey_script, tmp_path, capsys
):
    pytest.importorskip("torch")
    make_splits(tmp_path)
    output = tmp_path / "out"
    assert (
        survey_script.main(
            request(tmp_path, output)
            + [
                "--model-params",
                json.dumps({"unlabeled_positive_prior": 0.3, "prior_source": "alpha"}),
            ]
        )
        == 1
    )
    assert "cannot overwrite an explicit alpha_U/source" in capsys.readouterr().err
    assert not output.exists()
