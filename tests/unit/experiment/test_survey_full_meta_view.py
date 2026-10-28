"""Full-meta uses its versioned OS adapter; the PU-only ablation stays TS."""

# ruff: noqa: F811 -- imported pytest fixtures are injected by parameter name

import copy

import pytest
from _survey_pilot_helpers import raw_driver  # noqa: F401
from _survey_script_helpers import survey_script  # noqa: F401

from pu_toolbox.experiment.method_ledger import load_ledger
from pu_toolbox.experiment.pilot_plan import planned_runs
from pu_toolbox.experiment.survey_profile_identity import profile_training_view_request
from pu_toolbox.experiment.survey_protocol import PROTOCOL_PATHS, load_protocol
from pu_toolbox.registry import get_algorithm, register_all_builtin_methods

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def protocol():
    return load_protocol(PROTOCOL_PATHS["survey-v3.0"])


@pytest.mark.parametrize("requested, expected", [(None, "os"), ("os", "os"), ("ts", "ts")])
def test_param_full_meta_resolves_profile_without_rewriting_explicit_view(
    protocol, requested, expected
):
    before = copy.deepcopy(protocol)
    assert profile_training_view_request(protocol, "self_pu_full_meta", requested) == expected
    assert protocol == before


@pytest.mark.parametrize("method", ["self_pu", "nnpu", "lagam", "pulns", "pn_oracle"])
def test_param_other_profile_defaults_stay_unchanged(protocol, method):
    assert profile_training_view_request(protocol, method, None) is None


def test_determ_legacy_protocol_default_stays_unchanged():
    assert profile_training_view_request(load_protocol(), "self_pu", None) is None


def test_edge_absent_profile_schema_does_not_lookup_a_new_profile():
    assert profile_training_view_request({}, "self_pu", None) is None
    assert profile_training_view_request({}, "self_pu", "ts") == "ts"


@pytest.mark.parametrize("requested, expected", [(None, "os"), ("os", "os"), ("ts", "ts")])
def test_basic_cli_resolves_profile_before_loading_any_data(
    survey_script, monkeypatch, tmp_path, requested, expected, capsys
):
    seen = []

    def boundary(ledger, algorithm, view, **kwargs):
        seen.append((algorithm, view))
        raise ValueError("routing boundary reached")

    def forbidden(*args, **kwargs):
        pytest.fail("view preflight must precede all data loading")

    monkeypatch.setattr(survey_script, "resolve_training_view", boundary)
    monkeypatch.setattr(survey_script, "load_split_parts", forbidden)
    argv = [
        str(tmp_path / "not-loaded"),
        "--protocol",
        "survey-v3.0",
        "--dataset",
        "spambase",
        "--method",
        "self_pu_full_meta",
        "--training-path",
        "native_2d",
        "--support-dir",
        str(tmp_path / "not-loaded-support"),
        "--class-prior",
        "0.4",
    ]
    if requested is not None:
        argv.extend(["--os-or-ts", requested])
    assert survey_script.main(argv) == 1
    assert seen == [("self_pu", expected)]
    assert "routing boundary reached" in capsys.readouterr().err


@pytest.mark.parametrize("requested", [None, "os", "ts"])
def test_param_unit_and_resume_resolve_the_same_full_meta_view(
    protocol, raw_driver, survey_script, requested
):
    register_all_builtin_methods()
    runs = tuple(run for run in planned_runs(protocol) if run.method == "self_pu_full_meta")
    expected = survey_script.resolve_training_view(
        load_ledger(),
        "self_pu",
        profile_training_view_request(protocol, "self_pu_full_meta", requested),
        is_oracle=False,
        estimator_class=get_algorithm("self_pu"),
    )
    views = raw_driver.expected_run_views(runs, requested, protocol=protocol)
    assert set(views.values()) == {f"{expected}-compatible"}
    assert expected == ("ts" if requested == "ts" else "os")


def test_basic_ablation_still_defaults_to_ts_in_resume(protocol, raw_driver):
    runs = tuple(run for run in planned_runs(protocol) if run.method == "self_pu")
    assert set(raw_driver.expected_run_views(runs, None, protocol=protocol).values()) == {
        "ts-compatible"
    }
