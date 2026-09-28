# tests/unit/experiment/test_survey_script.py

# F811 is needed because ``survey_script`` is the imported fixture: pytest looks
# fixtures up by that name, so it repeats as a test parameter and pyflakes reads
# the repetition as a redefinition of the import.
# ruff: noqa: N803, S101, F811

import json

import pytest
from _survey_script_helpers import (  # noqa: F401 - pytest fixture
    make_splits,
    run_manifest,
    run_manifests,
    survey_script,
)

from pu_toolbox.experiment.manifest import load_manifest

pytestmark = pytest.mark.unit


def test_basic_run_survey_script_end_to_end(survey_script, tmp_path):
    """Example script completes: manifests carry all 8 required keys and PA/OA metrics."""
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_splits(data_dir)

    out_dir = tmp_path / "out"
    rc = survey_script.main(
        [
            str(data_dir),
            "--method",
            "upu",
            "--model-params",
            json.dumps({"class_prior": 0.3, "loss": "squared"}),
            "--c",
            "0.3",
            "--seeds",
            "0",
            "--class-prior",
            "0.3",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    manifest_path = run_manifest(out_dir, "c_0.3", "seed_0")
    manifest = load_manifest(manifest_path)
    for key in (
        "seed",
        "split_ref",
        "generation",
        "selection",
        "test_results",
        "elapsed",
        "failures",
        "resources",
    ):
        assert key in manifest
    assert set(manifest["test_results"]) == {"PA", "OA"}
    assert manifest["test_results"]["OA"]["auc_unavailable_reason"] is None
    assert (run_manifest(out_dir, "c_0.3", "seed_0").parent / "method_ledger_entry.json").is_file()


def test_param_survey_script_requires_population_prior(survey_script, tmp_path, capsys):
    """Ledger prior_semantics='population π' -> --class-prior is mandatory."""
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_splits(data_dir)

    rc = survey_script.main([str(data_dir), "--method", "upu"])
    assert rc == 1
    assert "pass --class-prior" in capsys.readouterr().err


def test_basic_script_runs_pusb_without_class_prior(survey_script, tmp_path):
    """The linear PUSB baseline takes no pi: it must run without --class-prior.

    The gate is the registry's requires_class_prior (False for this row), not
    the ledger's annotation of the kernel sibling.
    """
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_splits(data_dir)

    out_dir = tmp_path / "out"
    rc = survey_script.main(
        [
            str(data_dir),
            "--method",
            "pusb",
            "--c",
            "0.3",
            "--seeds",
            "0",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert (run_manifest(out_dir, "c_0.3", "seed_0").parent / "method_ledger_entry.json").is_file()


def test_param_script_requires_class_prior_for_pusb_kernel(survey_script, tmp_path, capsys):
    """Registry requires_class_prior=True for pusb_kernel -> --class-prior is mandatory.

    The gate fires before any estimator is built or any run directory is
    created: the deliberately invalid model params would otherwise surface a
    constructor error instead of the prior error.
    """
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_splits(data_dir)

    out_dir = tmp_path / "out"
    rc = survey_script.main(
        [
            str(data_dir),
            "--method",
            "pusb_kernel",
            "--model-params",
            json.dumps({"bogus_param": 1}),
            "--seeds",
            "0",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 1
    assert "pass --class-prior" in capsys.readouterr().err
    assert not out_dir.exists()


def test_basic_script_runs_pusb_kernel_with_class_prior(survey_script, tmp_path):
    """The official-aligned kernel row runs with pi and keeps its own ledger entry."""
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_splits(data_dir)

    out_dir = tmp_path / "out"
    rc = survey_script.main(
        [
            str(data_dir),
            "--method",
            "pusb_kernel",
            "--class-prior",
            "0.3",
            "--c",
            "1.0",
            "--seeds",
            "0",
            "--model-params",
            json.dumps({"n_basis": 16, "cv": 2, "max_iter": 100, "random_state": 0}),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    entry = json.loads(
        (run_manifest(out_dir, "c_1.0", "seed_0").parent / "method_ledger_entry.json").read_text(
            encoding="utf-8"
        )
    )
    assert entry["class"] == "PUSBKernelClassifier"
    assert "population" in entry["prior_semantics"]


def test_param_script_rejects_methods_outside_survey_ledger(survey_script, tmp_path, capsys):
    """A registered method outside the survey scope fails loud, not silently."""
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_splits(data_dir)

    rc = survey_script.main([str(data_dir), "--method", "elkan_noto", "--class-prior", "0.3"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "elkan_noto" in err
    assert "survey ledger" in err


def test_edge_survey_script_reports_missing_split_files(survey_script, tmp_path, capsys):
    """Missing .npz partitions produce a helpful error before any training."""
    rc = survey_script.main([str(tmp_path / "nothing"), "--method", "upu", "--class-prior", "0.3"])
    assert rc == 1
    assert "missing split files" in capsys.readouterr().err


def test_basic_oracle_script_writes_oa_only_results(survey_script, tmp_path):
    """--oracle runs the PN path: OA only, real labels, calibration recorded."""
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_splits(data_dir)

    out_dir = tmp_path / "oracle_out"
    rc = survey_script.main(
        [
            str(data_dir),
            "--oracle",
            "--model-params",
            json.dumps({"hidden_layer_sizes": (8,), "max_iter": 50, "random_state": 0}),
            "--seeds",
            "0",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    manifest = load_manifest(run_manifest(out_dir, "c_independent", "seed_0"))
    assert set(manifest["test_results"]) == {"OA"}
    assert manifest["generation"]["train"]["mechanism"] == "pn_oracle"
    assert manifest["c_independent"] is True
    assert manifest["broadcast_c_values"] == [0.1]
    integration = json.loads((out_dir / "oracle_integration.json").read_text(encoding="utf-8"))
    assert integration["selection_metric"] == "clean_val_accuracy"
    assert integration["class_prior_applied"] is False
    assert integration["runs_completed"] == 1


def test_oracle_deduplicates_runs_across_c_values(survey_script, tmp_path):
    """PN oracle runs once per seed even when multiple c values are requested."""
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_splits(data_dir)
    out_dir = tmp_path / "oracle_out"

    rc = survey_script.main(
        [
            str(data_dir),
            "--oracle",
            "--c",
            "0.1,0.3,0.5",
            "--seeds",
            "0,1",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    # One run per seed, and none of them keyed by a c token: the oracle is
    # c-independent, so no c_* directory may appear anywhere in the tree.
    manifests = run_manifests(out_dir)
    assert len(manifests) == 2
    assert all(path.parent.name.startswith("seed_") for path in manifests)
    integration = json.loads((out_dir / "oracle_integration.json").read_text(encoding="utf-8"))
    assert integration["runs_completed"] == 2
    assert integration["broadcast_c_values"] == [0.1, 0.3, 0.5]


def test_param_oracle_script_rejects_class_prior(survey_script, tmp_path, capsys):
    """The oracle trains on real labels; a class prior there is a semantic error."""
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_splits(data_dir)

    rc = survey_script.main([str(data_dir), "--oracle", "--class-prior", "0.3"])
    assert rc == 1
    assert "drop --class-prior" in capsys.readouterr().err


def test_param_oracle_script_rejects_method(survey_script, tmp_path, capsys):
    """--oracle must not silently ignore --method and run a different path."""
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_splits(data_dir)

    rc = survey_script.main([str(data_dir), "--oracle", "--method", "nnpu"])
    assert rc == 1
    assert "drop --method" in capsys.readouterr().err


def test_edge_oracle_script_leaves_no_calibration_file_when_runs_fail(survey_script, tmp_path):
    """The calibration contract must not outlive the results it describes.

    A failed output directory that still carries ``oracle_integration.json``
    would read as a valid oracle row to anything scanning for that file.
    """
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_splits(data_dir)
    # Every candidate fails on an unknown constructor parameter, so all runs are
    # excluded and the script exits non-zero.
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps([{"bogus_param": 1}]), encoding="utf-8")

    out_dir = tmp_path / "oracle_failed"
    rc = survey_script.main(
        [
            str(data_dir),
            "--oracle",
            "--candidates",
            str(candidates),
            "--seeds",
            "0",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 1

    manifest = load_manifest(run_manifest(out_dir, "c_independent", "seed_0"))
    assert manifest["failures"]  # the failure itself is recorded
    assert not (out_dir / "oracle_integration.json").exists()


def test_basic_survey_script_records_the_split_reference(survey_script, tmp_path):
    """Protocol §2.4 item 11: a run must say which split it used.

    The sample ids stay in the split manifest; the run manifest records the
    path, the role sizes and the index digest instead of inlining them.
    """
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_splits(data_dir)
    (data_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "toy",
                "seed": 0,
                "role_sizes": {"train": 18, "pu_val": 4, "clean_val": 4, "test": 4},
                "indices_sha256": "deadbeef",
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    rc = survey_script.main(
        [
            str(data_dir),
            "--method",
            "upu",
            "--model-params",
            json.dumps({"class_prior": 0.3, "loss": "squared"}),
            "--c",
            "0.3",
            "--seeds",
            "0",
            "--class-prior",
            "0.3",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    manifest = load_manifest(run_manifest(out_dir, "c_0.3", "seed_0"))
    assert manifest["split_ref"]["indices_sha256"] == "deadbeef"
    assert manifest["split_ref"]["role_sizes"]["train"] == 18
    assert manifest["split_ref"]["manifest_path"].endswith("split_manifest.json")


def test_param_oracle_script_explicit_split_ref_wins(survey_script, tmp_path):
    """--split-ref overrides the reference auto-filled from the split directory."""
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_splits(data_dir)
    (data_dir / "split_manifest.json").write_text(
        json.dumps({"indices_sha256": "auto"}), encoding="utf-8"
    )
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps({"note": "custom reference"}), encoding="utf-8")

    out_dir = tmp_path / "explicit_out"
    rc = survey_script.main(
        [
            str(data_dir),
            "--oracle",
            "--model-params",
            json.dumps({"hidden_layer_sizes": (8,), "max_iter": 50, "random_state": 0}),
            "--seeds",
            "0",
            "--split-ref",
            str(explicit),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    manifest = load_manifest(run_manifest(out_dir, "c_independent", "seed_0"))
    assert manifest["split_ref"] == {"note": "custom reference"}
