"""Plan pinning and actual CLI-to-manifest single-candidate execution."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import run_genpu_recipe as script  # noqa: E402

pytestmark = pytest.mark.unit


def product(tmp_path):
    directory = tmp_path / "input"
    directory.mkdir()
    for i, role in enumerate(script.ROLES):
        rng = np.random.RandomState(i)
        np.savez(
            directory / f"{role}.npz",
            X=rng.normal(size=(16, 3)).astype("float32"),
            y=np.arange(16) % 2,
            indices=np.arange(100 * i, 100 * i + 16),
        )
    return directory


def plan(directory, **kwargs):
    return script.build_plan(
        directory,
        **(
            dict(
                dataset="spambase",
                prior=0.4,
                prior_source="known synthetic population prior",
                seed=0,
                device="cpu",
                gan_epochs=1,
                pn_epochs=3,
                batch_size=8,
                checkpoint_bytes=2**20,
                c_token="0.5",
            )
            | kwargs
        ),
    )


def test_basic_actual_plan_manifest_retention_and_two_model_replays(tmp_path):
    planned = plan(product(tmp_path))
    output = tmp_path / "result"
    summary = script.execute(planned, output)
    assert summary["accepted_engineering_run"] and summary["selected_models_replayed"]
    assert summary["warning_count"] == summary["retry_count"] == 0
    assert not summary["formal_admission"] and not summary["formal_budget_approved"]
    assert summary["checkpoint_peak_payload_bytes"] > summary["retained_checkpoint_bytes"] > 0
    assert summary["reclaimed_checkpoint_count"] >= 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["run_view"] == "ts-compatible" and manifest["calibration_applied"]
    assert set(summary["selection"]) == {"PA", "OA"}
    assert len(manifest["candidate_runs"]) == 1
    assert all((output / f"selected_{name}.pkl").is_file() for name in ("PA", "OA"))
    if sys.platform == "win32":
        assert summary["process_lifetime_peak_rss_bytes"] is None
    else:
        assert summary["process_lifetime_peak_rss_bytes"] > 0
    with pytest.raises(FileExistsError):
        script.execute(planned, output)


def test_edge_tampered_plan_and_file_rejected_without_creating_output(tmp_path):
    directory = product(tmp_path)
    planned = plan(directory)
    changed = copy.deepcopy(planned)
    changed["recipe"]["parameters"]["max_epochs"] += 1
    with pytest.raises(ValueError, match="plan changed"):
        script.execute(changed, tmp_path / "bad")
    with (directory / "test.npz").open("ab") as stream:
        stream.write(b"changed file identity")
    with pytest.raises(ValueError, match="identity changed"):
        script.execute(planned, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_determ_replanning_keeps_recipe_label_draws_and_all_digests_identical(tmp_path):
    directory = product(tmp_path)
    first, second = plan(directory), plan(directory)
    assert first == second
    different_seed = plan(directory, seed=7)
    assert first["recipe"]["recipe_sha256"] != different_seed["recipe"]["recipe_sha256"]
    assert first["input_files_sha256"] == different_seed["input_files_sha256"]
    assert first["label_view_sha256"] != different_seed["label_view_sha256"]


@pytest.mark.parametrize(
    "kwargs",
    [{"prior_source": ""}, {"checkpoint_bytes": True}, {"c_token": "0.50"}, {"dataset": "adni"}],
)
def test_param_bad_or_ambiguous_plan_requests(tmp_path, kwargs):
    with pytest.raises(ValueError):
        plan(product(tmp_path), **kwargs)


def test_edge_output_cannot_overwrite_input_or_ancestor(tmp_path):
    directory = product(tmp_path)
    planned = plan(directory)
    for output in (directory, tmp_path, directory / "nested"):
        with pytest.raises(ValueError, match="overlap"):
            script.execute(planned, output)


def test_edge_cli_execute_requires_output_before_planning(capsys):
    assert (
        script.main(
            [
                "--data-dir",
                "missing",
                "--dataset",
                "spambase",
                "--class-prior",
                "0.4",
                "--prior-source",
                "fixture",
                "--seed",
                "0",
                "--device",
                "cpu",
                "--gan-epochs",
                "1",
                "--pn-epochs",
                "3",
                "--batch-size",
                "8",
                "--checkpoint-bytes-per-component",
                "1048576",
                "--c",
                "0.5",
                "--execute",
            ]
        )
        == 1
    )
    assert "requires a new --out-dir" in capsys.readouterr().err
