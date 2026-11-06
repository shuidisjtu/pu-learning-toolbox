"""Split backward and GP chunking are explicit GradPU execution policies."""

# ruff: noqa: F811

import json

import pytest
from _survey_script_helpers import make_sar_splits, survey_script  # noqa: F401
from test_survey_backlog import small_protocol

pytestmark = pytest.mark.unit


def _write_split_manifest(data):
    data.mkdir()
    make_sar_splits(data)
    (data / "split_manifest.json").write_text(
        json.dumps({"dataset": "spambase", "seed": 0, "class_prior": {"population": 0.4}})
    )


def test_determ_recipe_binding_records_split_policy_and_rejects_hidden_override(tmp_path):
    from pu_toolbox import GradPUClassifier
    from pu_toolbox.experiment.runner import _recipe_context
    from pu_toolbox.experiment.survey_protocol import digest, load_protocol

    source, registry = small_protocol(tmp_path, "gradpu")
    protocol = load_protocol(source)
    model = GradPUClassifier(**protocol["method_profiles"]["gradpu"]["params"])
    context = dict(
        execution_mode="versioned_pilot",
        protocol_sha256=digest(protocol),
        training_path="native_2d",
        runtime_parameters={},
    )
    config = dict(
        recipe_registry=json.loads(registry.read_text()),
        survey_protocol=dict(
            path=str(source), method="gradpu", dataset="spambase", constructor_overrides={}
        ),
    )
    baseline = _recipe_context(model, config, context)

    model.set_params(backward_mode="split", gp_chunk_size=3)
    with pytest.raises(ValueError, match="outside the resolved"):
        _recipe_context(model, config, context)
    config["gradpu_backward_mode"] = "split"
    with pytest.raises(ValueError, match="backward execution policy"):
        _recipe_context(model, config, context)
    config["gradpu_gp_chunk_size"] = 3
    assert _recipe_context(model, config, context) == baseline

    config["gradpu_backward_mode"] = "auto"
    with pytest.raises(ValueError, match="backward execution policy"):
        _recipe_context(model, config, context)


@pytest.mark.parametrize("chunk", [None, 3])
def test_basic_bound_cli_records_split_policy_and_keeps_full_budget(
    tmp_path, survey_script, chunk, capsys
):
    data = tmp_path / "data"
    _write_split_manifest(data)
    source, registry = small_protocol(tmp_path, "gradpu")
    output = tmp_path / "out"
    arguments = [
        str(data),
        "--method",
        "gradpu",
        "--protocol",
        str(source),
        "--dataset",
        "spambase",
        "--recipe-registry",
        str(registry),
        "--device",
        "cpu",
        "--c",
        "0.5",
        "--class-prior",
        "0.4",
        "--out-dir",
        str(output),
        "--gradpu-backward-mode",
        "split",
        "--reclaim-unselected-checkpoints",
    ]
    if chunk is not None:
        arguments += ["--gradpu-gp-chunk-size", str(chunk)]
    code = survey_script.main(arguments)
    captured = capsys.readouterr()
    assert code == 0, captured.err + captured.out
    paths = list(output.rglob("manifest.json"))
    assert len(paths) == 1
    expected = "execution_split" if chunk is None else f"execution_split_chunk{chunk}"
    assert expected in paths[0].parts
    manifest = json.loads(paths[0].read_text())
    assert not manifest["failures"] and not manifest["formal_eligible"]
    candidate = manifest["candidate_runs"][0]
    assert len(candidate["epoch_checkpoints"]) == 2
    facts = candidate["training_provenance"]["activation_memory"]
    assert facts["backward_mode"] == "split"
    assert facts["gp_chunk_size"] == chunk
    assert not facts["effective_batch_changed"]
    assert not facts["precision_changed"]


def test_edge_chunk_size_requires_split_mode(tmp_path, survey_script, capsys):
    data = tmp_path / "data"
    _write_split_manifest(data)
    source, registry = small_protocol(tmp_path, "gradpu")
    code = survey_script.main(
        [
            str(data),
            "--method",
            "gradpu",
            "--protocol",
            str(source),
            "--dataset",
            "spambase",
            "--recipe-registry",
            str(registry),
            "--device",
            "cpu",
            "--c",
            "0.5",
            "--class-prior",
            "0.4",
            "--out-dir",
            str(tmp_path / "out"),
            "--gradpu-gp-chunk-size",
            "3",
        ]
    )
    assert code == 1
    assert "requires --gradpu-backward-mode split" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--protocol", "survey-v3.2", "--method", "vpu", "--dataset", "adni"],
    ],
)
def test_param_split_wrong_scope_rejected_before_loading_data(
    tmp_path, survey_script, extra, capsys
):
    code = survey_script.main(
        [
            str(tmp_path / "missing"),
            "--gradpu-backward-mode",
            "split",
            "--out-dir",
            str(tmp_path / "out"),
            *extra,
        ]
    )
    assert code == 1
    assert "backward execution policy requires" in capsys.readouterr().err
