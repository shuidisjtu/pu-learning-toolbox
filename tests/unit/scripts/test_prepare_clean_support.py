"""Raw-reader support preparation is explicit, offline and non-destructive."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pu_toolbox.experiment.support_data import load_support_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from prepare_clean_support import main  # noqa: E402

pytestmark = pytest.mark.unit


def raw(tmp_path):
    root = tmp_path / "raw"
    (root / "spambase").mkdir(parents=True)
    source = np.random.RandomState(1).normal(size=(500, 57))
    values = np.column_stack((source, np.tile([0, 1], 250)))
    np.savetxt(root / "spambase" / "spambase.data", values, delimiter=",")
    return root


def arguments(root):
    return [
        "--dataset",
        "spambase",
        "--raw-dir",
        str(root),
        "--seed",
        "0",
        "--support-size",
        "40",
        "--support-seed",
        "0",
        "--group-policy",
        "not_applicable",
    ]


def test_basic_dry_run_then_explicit_materialization(tmp_path, capsys):
    root = raw(tmp_path)
    assert main(arguments(root)) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "dry_run"
    output = tmp_path / "prepared"
    args = arguments(root) + ["--out-dir", str(output), "--write"]
    assert main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "written" and report["formal_admission"] is False
    _, support, receipt = load_support_dataset(output)
    assert len(support.X) == 40 and "spambase.data" in receipt["reader_files_sha256"]
    assert main(args) == 1
    assert "overwrite" in capsys.readouterr().err


@pytest.mark.parametrize("kind", ["no_output", "overlap", "bad_budget", "missing_raw"])
def test_edge_invalid_output_or_source_fails_cleanly(tmp_path, capsys, kind):
    root = raw(tmp_path)
    args = arguments(root)
    if kind == "no_output":
        args += ["--write"]
    elif kind == "overlap":
        args += ["--write", "--out-dir", str(root / "new")]
    elif kind == "bad_budget":
        args += ["--support-size", "1"]
    else:
        args += ["--raw-dir", str(tmp_path / "absent")]
    assert main(args) == 1
    assert "error:" in capsys.readouterr().err


def test_determ_same_input_plan_has_same_receipt(tmp_path, capsys):
    root = raw(tmp_path)
    outputs = [tmp_path / "a", tmp_path / "b"]
    for output in outputs:
        assert main(arguments(root) + ["--out-dir", str(output), "--write"]) == 0
        capsys.readouterr()
    assert load_support_dataset(outputs[0])[2] == load_support_dataset(outputs[1])[2]


@pytest.mark.parametrize("flag", ["--support-size", "--group-policy"])
def test_param_budget_and_policy_are_never_implicit_defaults(tmp_path, flag):
    args = arguments(raw(tmp_path))
    index = args.index(flag)
    del args[index : index + 2]
    with pytest.raises(SystemExit):
        main(args)


def split_dir(root, dataset="imdb"):
    from pu_toolbox.experiment.datasets import prepare_survey_dataset

    root.mkdir(parents=True)
    rng = np.random.RandomState(0)
    x_source = rng.normal(size=(160, 384)).astype("float32")
    x_test = rng.normal(size=(40, 384)).astype("float32")
    y_source = np.tile([0, 1], 80)
    y_test = np.tile([0, 1], 20)
    bundle, manifest = prepare_survey_dataset(
        x_source, y_source, dataset=dataset, seed=0, X_test=x_test, y_test=y_test
    )
    manifest["preprocessing"] = {
        "kind": "sbert",
        "embedding_dimension": 384,
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    }
    for role in ("train", "pu_val", "clean_val", "test"):
        part = getattr(bundle, role)
        np.savez_compressed(root / f"{role}.npz", X=part.X, y=part.labels, indices=part.indices)
    (root / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def arguments_split(root):
    return [
        "--dataset",
        "imdb",
        "--split-dir",
        str(root),
        "--seed",
        "0",
        "--support-size",
        "20",
        "--group-policy",
        "not_applicable",
    ]


def test_basic_split_mode_materializes_text_product(tmp_path, capsys):
    source = split_dir(tmp_path / "splits")
    assert main(arguments_split(source)) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["support_seed"] == 0 and preview["source_mode"] == "frozen_split"
    output = tmp_path / "product"
    assert main(arguments_split(source) + ["--out-dir", str(output), "--write"]) == 0
    _, support, receipt = load_support_dataset(output)
    assert len(support.X) == 20
    assert receipt["support_seed_policy"].startswith("support_seed equals split_seed")
    assert receipt["preprocessing"]["kind"] == "sbert"


def test_edge_split_mode_rejects_mismatched_support_seed(tmp_path, capsys):
    source = split_dir(tmp_path / "splits")
    args = arguments_split(source) + ["--support-seed", "1"]
    assert main(args) == 1
    assert "support_seed equals split_seed" in capsys.readouterr().err


def test_edge_adni_requires_owner_approval_before_reading_data(tmp_path, capsys):
    args = [
        "--dataset",
        "adni",
        "--split-dir",
        str(tmp_path / "absent"),
        "--seed",
        "0",
        "--support-size",
        "20",
        "--group-policy",
        "unverifiable",
    ]
    assert main(args) == 1
    assert "owner-approval" in capsys.readouterr().err
