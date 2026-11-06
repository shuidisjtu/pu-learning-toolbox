# ruff: noqa: F811
"""User-prepared split arrays carry a reproducible identity without generated receipts."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from _survey_pilot_helpers import raw_driver  # noqa: F401
from _survey_script_helpers import make_splits, survey_script  # noqa: F401

from pu_toolbox.experiment.pilot_plan import completed_runs, split_digests
from pu_toolbox.experiment.survey_protocol import role_indices_digest

pytestmark = pytest.mark.unit


def prepared(tmp_path):
    root = tmp_path / "splits"
    data = root / "spambase/split_0"
    data.mkdir(parents=True)
    make_splits(data)
    return root, data


def test_basic_determ_real_user_arrays_finish_and_resume_a_grid_without_index_metadata(
    raw_driver, survey_script, tmp_path, monkeypatch, capsys
):
    splits, data = prepared(tmp_path)
    protocol = raw_driver.load_protocol(raw_driver.resolve_protocol_path("survey-v3.2"))
    protocol["seeds"] = [0]
    protocol["c_tokens"] = {"scar": ["0.5"]}
    protocol["execution_units"] = [
        row
        for row in protocol["execution_units"]
        if row["dataset"] == "spambase" and row["method"] == "upu"
    ]
    config = tmp_path / "current.json"
    config.write_text(json.dumps(protocol), encoding="utf-8")
    calls = []
    real_run = raw_driver.subprocess.run

    def execute(argv, **kwargs):
        if len(argv) < 2 or Path(argv[1]).name != "run_survey_experiment.py":
            return real_run(argv, **kwargs)
        calls.append(argv)
        return SimpleNamespace(returncode=survey_script.main(argv[2:]))

    monkeypatch.setattr(raw_driver.subprocess, "run", execute)
    args = [
        "--protocol",
        str(config),
        "--splits",
        str(splits),
        "--results",
        str(tmp_path / "results"),
        "--datasets",
        "spambase",
        "--methods",
        "upu",
        "--device",
        "cpu",
    ]
    assert raw_driver.main(args) == 0
    assert len(calls) == 1
    identity = split_digests(splits)[("spambase", 0)]
    assert len(completed_runs(tmp_path / "results", splits={("spambase", 0): identity})) == 1
    assert raw_driver.main(args) == 0
    assert len(calls) == 1
    assert "completed" in capsys.readouterr().out.lower()
    assert "indices_sha256" not in json.loads((data / "split_manifest.json").read_text())


@pytest.mark.parametrize("fault", ["wrong_declared_digest", "overlap", "missing_role"])
def test_param_invalid_array_identity_is_rejected_before_the_grid_trains(
    raw_driver, tmp_path, monkeypatch, fault
):
    splits, data = prepared(tmp_path)
    manifest = data / "split_manifest.json"
    if fault == "wrong_declared_digest":
        payload = json.loads(manifest.read_text())
        payload["indices_sha256"] = "0" * 64
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    elif fault == "missing_role":
        (data / "test.npz").unlink()
    else:
        with np.load(data / "test.npz") as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays["indices"][0] = 0
        np.savez(data / "test.npz", **arrays)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid split identity reached training")

    monkeypatch.setattr(raw_driver.subprocess, "run", forbidden)
    assert (
        raw_driver.main(
            [
                "--splits",
                str(splits),
                "--results",
                str(tmp_path / "results"),
                "--datasets",
                "spambase",
                "--methods",
                "upu",
                "--device",
                "cpu",
            ]
        )
        == 1
    )


def test_edge_standalone_cli_records_actual_indices_and_refuses_a_stale_claim(
    survey_script, tmp_path
):
    _, data = prepared(tmp_path)
    reference = survey_script.resolve_split_ref(data, None)
    indices = {}
    for role in ("train", "pu_val", "clean_val", "test"):
        with np.load(data / f"{role}.npz") as archive:
            indices[role] = archive["indices"]
    assert reference["indices_sha256"] == role_indices_digest(indices)
    payload = json.loads((data / "split_manifest.json").read_text())
    payload["indices_sha256"] = "0" * 64
    (data / "split_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="indices_sha256"):
        survey_script.resolve_split_ref(data, None)
