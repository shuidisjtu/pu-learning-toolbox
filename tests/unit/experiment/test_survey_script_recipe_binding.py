"""Opt-in recipe CLI binding, with no changes to legacy output paths."""

# ruff: noqa: F811

import hashlib
import json

import pytest
from _survey_recipe_registry_helpers import locked_registry
from _survey_script_helpers import make_splits, survey_script  # noqa: F401

from pu_toolbox.experiment.survey_protocol import load_protocol
from pu_toolbox.experiment.survey_recipe_registry import canonical_digest, validate_manifest_binding

pytestmark = pytest.mark.unit


def command(data, out):
    return [
        str(data),
        "--protocol",
        "survey-v2.0",
        "--dataset",
        "spambase",
        "--method",
        "upu",
        "--c",
        "0.5",
        "--seeds",
        "0",
        "--class-prior",
        "0.3",
        "--out-dir",
        str(out),
    ]


def write_registry(tmp_path, value=None):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(value or locked_registry()))
    return path


def test_basic_determ_cli_binding_uses_separate_namespace_and_leaves_legacy_bytes(
    survey_script, tmp_path
):
    source, out = tmp_path / "splits", tmp_path / "results"
    source.mkdir()
    make_splits(source)
    args = command(source, out)
    assert survey_script.main(args) == 0
    legacy = next(out.rglob("manifest.json"))
    before = hashlib.sha256(legacy.read_bytes()).hexdigest()
    registry = locked_registry()
    path = write_registry(tmp_path, registry)
    assert survey_script.main([*args, "--recipe-registry", str(path)]) == 0
    bound = out / f"recipe_{canonical_digest(registry)}"
    manifest = json.loads(next(bound.rglob("manifest.json")).read_text())
    assert validate_manifest_binding(manifest, registry, protocol=load_protocol()) == []
    assert hashlib.sha256(legacy.read_bytes()).hexdigest() == before
    assert len(list(out.rglob("manifest.json"))) == 2


@pytest.mark.parametrize("defect", ["draft", "digest", "constructor"])
def test_param_bad_recipe_rejected_before_reading_missing_splits(
    survey_script, tmp_path, capsys, defect
):
    value = locked_registry()
    extra = []
    if defect == "draft":
        value["recipe_lifecycle"] = "draft"
    elif defect == "digest":
        extra = ["--recipe-registry-sha256", "0" * 64]
    else:
        extra = ["--model-params", '{"reg_lambda":0.001}']
    path = write_registry(tmp_path, value)
    out = tmp_path / "results"
    assert (
        survey_script.main(
            [*command(tmp_path / "absent", out), "--recipe-registry", str(path), *extra]
        )
        == 1
    )
    error = capsys.readouterr().err
    assert "error:" in error
    assert "No such file" not in error
    assert not out.exists()


def test_edge_recipe_request_requires_a_versioned_protocol(survey_script, tmp_path, capsys):
    path = write_registry(tmp_path)
    assert survey_script.main([str(tmp_path / "absent"), "--recipe-registry", str(path)]) == 1
    assert "requires --protocol" in capsys.readouterr().err


def test_edge_digest_alone_is_not_silently_ignored(survey_script, tmp_path, capsys):
    assert (
        survey_script.main(
            [
                *command(tmp_path / "absent", tmp_path / "results"),
                "--recipe-registry-sha256",
                "0" * 64,
            ]
        )
        == 1
    )
    assert "requires --recipe-registry" in capsys.readouterr().err
