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
        "7",
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


@pytest.mark.parametrize("flag", ["--support-size", "--support-seed", "--group-policy"])
def test_param_budget_and_policy_are_never_implicit_defaults(tmp_path, flag):
    args = arguments(raw(tmp_path))
    index = args.index(flag)
    del args[index : index + 2]
    with pytest.raises(SystemExit):
        main(args)
