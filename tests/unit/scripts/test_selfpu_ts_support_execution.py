"""Explicit CLI TS full-meta scope, plan identity and independent replay."""

import copy
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import run_clean_support_experiment as script  # noqa: E402
from test_clean_support_checkpoint_execution import parameters, product  # noqa: E402

pytestmark = pytest.mark.unit


def plan(directory, **kwargs):
    return script.build_plan(
        directory,
        **(
            dict(
                method="self_pu",
                c_token="0.5",
                parameters=parameters("self_pu"),
                selection_scope="meta_checkpoints",
                checkpoint_bytes_per_component=65536,
                training_view="ts",
            )
            | kwargs
        ),
    )


def test_basic_real_cli_full_ts_meta_selection_and_replay(tmp_path):
    planned = plan(product(tmp_path))
    assert planned["training_view"] == "ts" and planned["extra_supervision"]["calibration_applied"]
    result = script.execute(planned, tmp_path / "result")
    assert result["training_view"] == "ts" and result["run_view"] == "ts-compatible"
    assert result["calibration_applied"] and result["accepted_engineering_run"]
    assert (
        result["selected_model_replay_verified"]
        and result["retry_count"] == result["warning_count"] == 0
    )
    assert result["extra_supervision"]["comparison_group"] == "extra_clean_support"
    assert not result["formal_admission"] and not result["extra_supervision"]["pa_eligible"]
    assert set(result["selection"]) == {"PA+clean-support", "OA"}
    assert (
        result["checkpoint_storage"]["peak_payload_bytes"]
        > result["checkpoint_storage"]["retained_bytes"]
        > 0
    )
    assert result["resources"]["checkpoint_disk_preflight"]["ready"]


def test_determ_same_ts_plan_and_separate_default_os_plan(tmp_path):
    directory = product(tmp_path)
    ts = plan(directory)
    assert ts == plan(directory)
    os = plan(directory, training_view="os")
    assert os["plan_payload_sha256"] != ts["plan_payload_sha256"]
    assert os["label_view_sha256"] == ts["label_view_sha256"]
    assert not os["extra_supervision"]["calibration_applied"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "lagam"},
        {"method": "pulns"},
        {"selection_scope": "terminal", "checkpoint_bytes_per_component": None},
        {"training_view": "TS"},
    ],
)
def test_param_no_implicit_ts_in_other_methods_or_terminal_scope(kwargs, tmp_path):
    with pytest.raises(ValueError, match="TS requires"):
        plan(product(tmp_path), **kwargs)


def test_edge_view_tampering_is_not_silently_replayed_as_os(tmp_path):
    planned = plan(product(tmp_path))
    mutated = copy.deepcopy(planned)
    mutated["training_view"] = "os"
    with pytest.raises(ValueError, match="changed after planning"):
        script.execute(mutated, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_determ_cli_dry_run_view_is_explicit_and_does_not_fit(tmp_path, monkeypatch, capsys):
    directory = product(tmp_path)
    path = tmp_path / "params.json"
    path.write_text(json.dumps(parameters("self_pu")))

    def forbidden(*args, **kwargs):
        pytest.fail("dry-run started training")

    monkeypatch.setattr(script.ExperimentRunner, "fit", forbidden)
    assert (
        script.main(
            [
                "--artifact-dir",
                str(directory),
                "--method",
                "self_pu",
                "--c",
                "0.5",
                "--parameters",
                str(path),
                "--training-view",
                "ts",
                "--selection-scope",
                "meta_checkpoints",
                "--checkpoint-bytes-per-component",
                "65536",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["training_view"] == "ts"
