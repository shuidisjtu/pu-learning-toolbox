# tests/unit/experiment/test_survey_script_sar.py

# F811 is needed because ``survey_script`` is the imported fixture: pytest looks
# fixtures up by that name, so it repeats as a test parameter and pyflakes reads
# the repetition as a redefinition of the import.
# ruff: noqa: N803, S101, F811

import json
import re

import pytest
from _survey_script_helpers import make_sar_splits, make_splits, survey_script  # noqa: F401

from pu_toolbox.experiment.manifest import load_manifest
from pu_toolbox.experiment.strategies import (
    ProtocolOA,
    SARLBEAGenerator,
    SARLBEBGenerator,
    SCARGenerator,
)

pytestmark = pytest.mark.unit


def test_basic_sar_lbe_a_runs_oa_only(survey_script, tmp_path):
    """SAR path: OA is the only selection protocol, and the run lands under the mechanism."""
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_sar_splits(data_dir)

    out_dir = tmp_path / "out"
    rc = survey_script.main(
        [
            str(data_dir),
            "--method",
            "lbe",
            "--labeling-mechanism",
            "sar_lbe_a",
            "--c",
            "0.05",
            "--seeds",
            "0",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    run_dir = out_dir / "sar_lbe_a" / "c_0.05" / "seed_0"
    manifest = load_manifest(run_dir / "manifest.json")
    assert set(manifest["selection"]) == {"OA"}
    assert set(manifest["test_results"]) == {"OA"}
    assert manifest["generation"]["train"]["mechanism"] == "sar_lbe_a"
    assert (run_dir / "method_ledger_entry.json").is_file()


@pytest.mark.parametrize("mechanism", ["sar_lbe_a", "sar_lbe_b"])
@pytest.mark.parametrize("c", ["0.05", "0.5"])
def test_param_sar_mechanisms_and_c_values(survey_script, tmp_path, mechanism, c):
    """Both SAR LBE variants run for both protocol c values, OA-only, token-named dirs."""
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_sar_splits(data_dir)

    out_dir = tmp_path / "out"
    rc = survey_script.main(
        [
            str(data_dir),
            "--method",
            "lbe",
            "--labeling-mechanism",
            mechanism,
            "--c",
            c,
            "--seeds",
            "0",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    manifest = load_manifest(out_dir / mechanism / f"c_{c}" / "seed_0" / "manifest.json")
    assert set(manifest["selection"]) == {"OA"}
    assert set(manifest["test_results"]) == {"OA"}
    assert manifest["generation"]["train"]["mechanism"] == mechanism


def test_param_sar_rejects_non_protocol_c(survey_script, tmp_path, capsys):
    """SAR only runs the protocol's c values (PU-Bench vary-e: {0.05, 0.5})."""
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_sar_splits(data_dir)

    out_dir = tmp_path / "out"
    rc = survey_script.main(
        [
            str(data_dir),
            "--method",
            "lbe",
            "--labeling-mechanism",
            "sar_lbe_a",
            "--c",
            "0.1",
            "--seeds",
            "0",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "0.05" in err  # the message names the allowed values
    assert not out_dir.exists()


def test_edge_sar_rejects_unknown_mechanism(survey_script, tmp_path):
    """A mechanism outside the CLI choices is an argparse error, not a silent fallback."""
    with pytest.raises(SystemExit) as excinfo:
        survey_script.main(
            [str(tmp_path), "--labeling-mechanism", "sar_lbe_c", "--out-dir", str(tmp_path / "out")]
        )
    assert excinfo.value.code == 2


def test_edge_sar_rejects_oracle_combination(survey_script, tmp_path, capsys):
    """--oracle trains on real labels: a labeling mechanism there is a semantic error."""
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_sar_splits(data_dir)

    rc = survey_script.main(
        [
            str(data_dir),
            "--oracle",
            "--labeling-mechanism",
            "sar_lbe_a",
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "drop" in err
    assert "labeling-mechanism" in err


def test_determ_sar_same_seed_same_marking_and_metrics(survey_script, tmp_path):
    """Same (mechanism, c, seed) reproduces the same P/U marking and the same OA metrics."""
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_sar_splits(data_dir)

    argv = [
        str(data_dir),
        "--method",
        "lbe",
        "--labeling-mechanism",
        "sar_lbe_a",
        "--c",
        "0.05",
        "--seeds",
        "0",
    ]
    assert survey_script.main([*argv, "--out-dir", str(tmp_path / "out_a")]) == 0
    assert survey_script.main([*argv, "--out-dir", str(tmp_path / "out_b")]) == 0

    first = load_manifest(tmp_path / "out_a" / "sar_lbe_a" / "c_0.05" / "seed_0" / "manifest.json")
    second = load_manifest(tmp_path / "out_b" / "sar_lbe_a" / "c_0.05" / "seed_0" / "manifest.json")
    assert (
        first["generation"]["train"]["label_view_sha256"]
        == second["generation"]["train"]["label_view_sha256"]
    )
    assert first["test_results"]["OA"] == second["test_results"]["OA"]


def test_basic_sar_manifest_records_generator_audit(survey_script, tmp_path):
    """The manifest carries the generator audit: requested vs clamped label counts.

    c=0.05 sits on both sides of the clamp on this bundle: train asks for
    round(40*0.05)=2 and gets 2, while pu_val asks for round(10*0.05)=0 and gets
    the clamped 1 — an audit field pair that must survive into the manifest.
    """
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_sar_splits(data_dir)

    out_dir = tmp_path / "out"
    rc = survey_script.main(
        [
            str(data_dir),
            "--method",
            "lbe",
            "--labeling-mechanism",
            "sar_lbe_a",
            "--c",
            "0.05",
            "--seeds",
            "0",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    manifest = load_manifest(out_dir / "sar_lbe_a" / "c_0.05" / "seed_0" / "manifest.json")
    train = manifest["generation"]["train"]
    assert train["generator"] == "SARLBEAGenerator"
    assert train["c_requested"] == 0.05
    assert train["n_positive"] == 40
    assert train["n_labeled_requested"] == 2
    assert train["n_labeled"] == 2
    assert train["generation_seed"] == 0
    assert re.fullmatch(r"[0-9a-f]{64}", train["label_view_sha256"])
    assert manifest["c_requested_token"] == "0.05"

    pu_val = manifest["generation"]["pu_val"]
    assert pu_val["n_labeled_requested"] == 0
    assert pu_val["n_labeled"] == 1
    assert pu_val["c_realized"] == pytest.approx(0.1)


def test_basic_scar_default_keeps_pa_oa(survey_script, tmp_path):
    """The default mechanism stays SCAR: PA/OA selection under the c_<token> path."""
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

    manifest = load_manifest(out_dir / "c_0.3" / "seed_0" / "manifest.json")
    assert set(manifest["selection"]) == {"PA", "OA"}
    assert manifest["generation"]["train"]["mechanism"] == "scar"
    assert manifest["selection"]["PA"]["protocol"] == "PA"
    assert manifest["c_requested_token"] == "0.3"


def test_edge_sar_token_shapes_the_output_path(survey_script, tmp_path):
    """The c path keeps the requested token: 0.05 must not be renormalised to 0.1."""
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_sar_splits(data_dir)

    out_dir = tmp_path / "out"
    rc = survey_script.main(
        [
            str(data_dir),
            "--method",
            "lbe",
            "--labeling-mechanism",
            "sar_lbe_a",
            "--c",
            "0.05",
            "--seeds",
            "0",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    assert (out_dir / "sar_lbe_a" / "c_0.05" / "seed_0" / "manifest.json").is_file()
    assert not (out_dir / "c_0.1").exists()
    assert not (out_dir / "sar_lbe_a" / "c_0.1").exists()


def test_param_sar_rejects_duplicate_numeric_tokens(survey_script, tmp_path, capsys):
    """Two tokens meaning the same c would silently overwrite one run directory."""
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_sar_splits(data_dir)

    out_dir = tmp_path / "out"
    rc = survey_script.main(
        [
            str(data_dir),
            "--method",
            "lbe",
            "--labeling-mechanism",
            "sar_lbe_a",
            "--c",
            "0.05,5e-2",
            "--seeds",
            "0",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "duplicate" in err.lower()
    assert "5e-2" in err and "0.05" in err  # both tokens are named
    assert not out_dir.exists()


def test_basic_sar_runs_non_lbe_classifier(survey_script, tmp_path):
    """The mechanism is orthogonal to --method: a non-LBE classifier runs under SAR-LBE-A."""
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_sar_splits(data_dir)

    out_dir = tmp_path / "out"
    rc = survey_script.main(
        [
            str(data_dir),
            "--method",
            "upu",
            "--class-prior",
            "0.4",
            "--model-params",
            json.dumps({"class_prior": 0.4, "loss": "squared"}),
            "--labeling-mechanism",
            "sar_lbe_a",
            "--c",
            "0.05",
            "--seeds",
            "0",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    run_dir = out_dir / "sar_lbe_a" / "c_0.05" / "seed_0"
    manifest = load_manifest(run_dir / "manifest.json")
    assert set(manifest["test_results"]) == {"OA"}
    assert manifest["generation"]["train"]["mechanism"] == "sar_lbe_a"
    entry = json.loads((run_dir / "method_ledger_entry.json").read_text(encoding="utf-8"))
    assert entry["class"] == "UPUClassifier"


def test_param_resolver_and_parser_guards(survey_script):
    """Module-level guards: mechanism resolvers and the five-gate c parser."""
    assert type(survey_script._resolve_labeling_generator("scar")) is SCARGenerator
    assert type(survey_script._resolve_labeling_generator("sar_lbe_a")) is SARLBEAGenerator
    assert type(survey_script._resolve_labeling_generator("sar_lbe_b")) is SARLBEBGenerator

    assert survey_script._resolve_labeling_protocols("scar") is None
    for mechanism in ("sar_lbe_a", "sar_lbe_b"):
        protocols = survey_script._resolve_labeling_protocols(mechanism)
        # An empty list would silently fall back to PA+OA in the runner.
        assert protocols, f"{mechanism} must return a non-empty protocol list"
        assert len(protocols) == 1
        assert isinstance(protocols[0], ProtocolOA)

    assert [(item.token, item.value) for item in survey_script._parse_c_values("0.05,0.5")] == [
        ("0.05", 0.05),
        ("0.5", 0.5),
    ]
    assert [item.token for item in survey_script._parse_c_values(" 0.05 , 0.5 ")] == ["0.05", "0.5"]

    for invalid in (
        "0",  # not in (0, 1]
        "1.5",  # above 1
        "",  # empty token
        "  ",  # whitespace-only token
        "nan",  # non-finite
        "inf",  # non-finite
        "abc",  # not a number
        "0.05,5e-2",  # duplicate value behind two tokens
    ):
        with pytest.raises(ValueError):
            survey_script._parse_c_values(invalid)
