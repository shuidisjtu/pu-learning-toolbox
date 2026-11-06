"""Current method-routing metadata agrees with registered estimator contracts."""

from __future__ import annotations

import inspect

import pytest

from pu_toolbox.core.tags import Scenario
from pu_toolbox.experiment.method_ledger import native_sampling_assumption
from pu_toolbox.experiment.training_views import resolve_training_view
from pu_toolbox.registry import get_algorithm, get_metadata

pytestmark = pytest.mark.contract

_EXPECTED_METHOD_KEYS = {
    "upu",
    "kldce",
    "dist_pu",
    "pusb",
    "pusb_kernel",
    "lbe",
    "lbe_neural",
    "nnpu",
    "self_pu",
    "vpu",
    "robust_pu",
    "split_pu",
    "cvir",
    "pulda",
    "puet",
    "gradpu",
    "lagam",
    "pan",
    "genpu",
    "holistic_pu",
    "rp",
    "pulns",
    "p3mix",
    "tedn",
}


def test_method_keys_are_the_registered_algorithms(ledger_methods: dict[str, dict]) -> None:
    for name in ledger_methods:
        assert get_metadata(name).name == name
    assert set(ledger_methods) == _EXPECTED_METHOD_KEYS


def test_calibration_applied_is_a_json_boolean(ledger_methods: dict[str, dict]) -> None:
    for name, entry in ledger_methods.items():
        for field in ("calibration_applied", "calibration_hooked"):
            assert isinstance(entry[field], bool), f"{name}.{field}"


def test_ledger_calibration_claims_have_a_real_fit_hook(
    ledger_methods: dict[str, dict],
) -> None:
    for name, entry in ledger_methods.items():
        if entry["calibration_applied"]:
            assert entry["run_view"] == "ts-compatible", name
            assert entry["calibration_hooked"], name
        elif entry["run_view"] == "ts-compatible":
            assert native_sampling_assumption(entry) == "ts", name
            assert entry.get("calibration_applicability") == "not_applicable_native_ts", name
        else:
            assert entry["run_view"] == "os-compatible", name
        if entry["calibration_hooked"]:
            assert native_sampling_assumption(entry) in {"ts", "both"}, name
            assert "os_or_ts" in inspect.signature(get_algorithm(name).fit).parameters, name


def test_linear_pusb_and_kernel_have_distinct_calibration_routes(
    ledger_methods: dict[str, dict],
) -> None:
    ledger = {"methods": ledger_methods}
    for name, expected in (("pusb", "os"), ("pusb_kernel", "ts")):
        resolved = resolve_training_view(
            ledger, name, None, is_oracle=False, estimator_class=get_algorithm(name)
        )
        assert resolved == expected
    with pytest.raises(ValueError, match="calibration_hooked=false"):
        resolve_training_view(
            ledger, "pusb", "ts", is_oracle=False, estimator_class=get_algorithm("pusb")
        )


def test_prior_semantics_consistent_with_registry_class_prior(
    ledger_methods: dict[str, dict],
) -> None:
    for name, entry in ledger_methods.items():
        requires_prior = get_metadata(name).requires_class_prior
        assert ("population" in entry["prior_semantics"]) == requires_prior, name


_SAMPLING_SCENARIOS = {
    "os": frozenset({Scenario.SINGLE_TRAINING_SET}),
    "ts": frozenset({Scenario.CASE_CONTROL}),
    "both": frozenset({Scenario.SINGLE_TRAINING_SET, Scenario.CASE_CONTROL}),
}


def test_native_sampling_assumption_matches_registry_scenario(
    ledger_methods: dict[str, dict],
) -> None:
    for name, entry in ledger_methods.items():
        registered = frozenset(get_metadata(name).scenario)
        sampling = registered & {Scenario.SINGLE_TRAINING_SET, Scenario.CASE_CONTROL}
        assert sampling == _SAMPLING_SCENARIOS[native_sampling_assumption(entry)], name
