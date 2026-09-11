# tests/unit/experiment/test_survey_script.py

# ruff: noqa: N803, S101

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pu_toolbox.experiment.manifest import load_manifest

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts/run_survey_experiment.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_survey_experiment", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def survey_script():
    return _load_script_module()


def make_splits(data_dir: Path) -> None:
    """Four-part bundle with every role containing real positives; global ids 0..29."""
    rng = np.random.RandomState(0)
    x = rng.randn(30, 3)
    # train(0-17): 6 pos + 12 neg; pu_val(18-21): 2 pos; clean_val(22-25): 2 pos;
    # test(26-29): 2 pos + 2 neg (so AUC is defined).
    y = np.array([1] * 6 + [0] * 12 + [1] * 2 + [0] * 2 + [1] * 2 + [0] * 2 + [1] * 2 + [0] * 2)
    for name in ("train", "pu_val", "clean_val", "test"):
        start, end = {
            "train": (0, 18),
            "pu_val": (18, 22),
            "clean_val": (22, 26),
            "test": (26, 30),
        }[name]
        np.savez(
            data_dir / f"{name}.npz",
            X=x[start:end],
            y=y[start:end],
            indices=np.arange(start, end),
        )


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

    manifest_path = out_dir / "c_0.3" / "seed_0" / "manifest.json"
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
    assert (out_dir / "c_0.3" / "seed_0" / "method_ledger_entry.json").is_file()


def test_param_survey_script_requires_population_prior(survey_script, tmp_path, capsys):
    """Ledger prior_semantics='population π' -> --class-prior is mandatory."""
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_splits(data_dir)

    rc = survey_script.main([str(data_dir), "--method", "upu"])
    assert rc == 1
    assert "pass --class-prior" in capsys.readouterr().err


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

    manifest = load_manifest(out_dir / "c_0.1" / "seed_0" / "manifest.json")
    assert set(manifest["test_results"]) == {"OA"}
    assert manifest["generation"]["train"]["mechanism"] == "pn_oracle"
    integration = json.loads((out_dir / "oracle_integration.json").read_text(encoding="utf-8"))
    assert integration["selection_metric"] == "clean_val_accuracy"
    assert integration["class_prior_applied"] is False


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
