# ruff: noqa: N803, N806
"""Coverage of the pre-registered comparison matrix against survey-v1.2."""

import json
from collections import Counter

import pytest

from pu_toolbox.experiment.survey_comparison import (
    COMPARISON_PATH,
    expected_result_units,
    load_comparison_protocol,
    validate_comparison_coverage,
)
from pu_toolbox.experiment.survey_protocol import load_protocol

pytestmark = pytest.mark.unit

#: Pinned for survey-v1.2.  A protocol bump must change this deliberately, not
#: silently: 18 runnable PU rows x (3 SCAR c x PA/OA + 2 SAR mechanisms x 2 c x OA)
#: + 3 c-independent oracle units + 4 non-runnable rows.
EXPECTED_UNITS = 187
EXPECTED_COMPOSITION = {"scar": 108, "sar_lbe_a": 36, "sar_lbe_b": 36, "c_independent": 3}


@pytest.fixture(scope="module")
def survey():
    return load_protocol()


@pytest.fixture(scope="module")
def comparison(survey):
    return load_comparison_protocol(survey=survey)


def _units_by(units, field):
    return {unit[field] for unit in units}


# --- enumeration is derived, not hand-listed ---------------------------------


def test_basic_enumeration_expands_every_runnable_row(survey):
    units = expected_result_units(survey)
    assert len(units) == EXPECTED_UNITS
    scar = [u for u in units if u["labeling_mechanism"] == "scar"]
    # 18 runnable PU rows x 3 c tokens x {pa, oa}
    assert len(scar) == EXPECTED_COMPOSITION["scar"]
    assert _units_by(scar, "selection_protocol") == {"pa", "oa"}
    assert _units_by(scar, "c_token") == {"0.1", "0.3", "0.5"}
    # AUROC is a supplementary metric; it must not enter v1 coverage.
    assert _units_by(units, "metric") == {"accuracy"}


def test_basic_sar_units_are_oa_only_on_the_sar_c_grid(survey):
    units = expected_result_units(survey)
    sar = [u for u in units if u["labeling_mechanism"].startswith("sar_")]
    assert len(sar) == EXPECTED_COMPOSITION["sar_lbe_a"] + EXPECTED_COMPOSITION["sar_lbe_b"]
    assert _units_by(sar, "selection_protocol") == {"oa"}
    assert _units_by(sar, "c_token") == {"0.05", "0.5"}


def test_determ_enumeration_is_stable_and_free_of_duplicate_units(survey):
    """The same protocol must always expand to the same unit set, with no repeats."""
    first = expected_result_units(survey)
    second = expected_result_units(survey)
    assert first == second
    fields = ("method", "dataset", "labeling_mechanism", "c_token", "selection_protocol")
    keys = [tuple(unit[field] for field in fields) for unit in first]
    assert len(set(keys)) == len(keys) == EXPECTED_UNITS


@pytest.mark.parametrize("dataset", ["spambase", "imdb", "cifar10"])
def test_param_each_dataset_covers_every_runnable_row(survey, comparison, dataset):
    units = [u for u in expected_result_units(survey) if u["dataset"] == dataset]
    covered = {
        (m["result_selector"]["method"], m["result_selector"]["training_path"])
        for m in comparison["mappings"]
        if m["scope"] == "unit" and m["result_selector"]["dataset"] == dataset
    }
    assert covered == {(u["method"], u["training_path"]) for u in units}


def test_basic_pn_rows_yield_one_c_independent_unit_and_no_sar(survey):
    units = expected_result_units(survey)
    pn = [u for u in units if u["method"] == "pn_oracle"]
    runnable_pn = [u for u in pn if u["c_token"] == "c_independent"]
    assert len(runnable_pn) == 3
    assert _units_by(runnable_pn, "selection_protocol") == {"oa"}
    assert not [u for u in pn if u["labeling_mechanism"].startswith("sar_")]


def test_basic_non_runnable_rows_are_not_expanded(survey):
    """KLDCE and the native-CNN oracle get one marker unit each, nothing invented."""
    units = expected_result_units(survey)
    marked = [u for u in units if u["c_token"] == "non_runnable"]
    assert len(marked) == 4
    assert {u["method"] for u in marked} == {"kldce", "pn_oracle"}
    kldce = [u for u in marked if u["method"] == "kldce"]
    assert len(kldce) == 3
    assert [u for u in marked if u["method"] == "pn_oracle"][0]["training_path"] == "native_cnn"


# --- the shipped matrix covers the enumeration -------------------------------


def test_basic_shipped_matrix_covers_every_unit_exactly_once(survey, comparison):
    summary = validate_comparison_coverage(survey, comparison)
    assert summary["units"] == EXPECTED_UNITS
    assert summary["unit_mappings"] == EXPECTED_UNITS
    assert summary["row_mappings"] >= 1


def test_basic_shipped_review_handoff_keeps_disputed_rows_pending(comparison):
    """The technical audit cannot silently turn into collaborator acceptance."""
    assert comparison["review_status"] == "pending_collaborator_review"
    assert comparison["formal_blockers"] == ["collaborator_review"]
    anchor_states = Counter(a["review_state"] for a in comparison["anchors"])
    mapping_states = Counter(m["review_state"] for m in comparison["mappings"])
    assert anchor_states == {"accepted": 18, "pending_review": 36}
    assert mapping_states == {"accepted": 187, "pending_review": 7}
    assert all(c["review_state"] == "pending_review" for c in comparison["contradictions"])

    pending_anchors = {
        a["anchor_id"] for a in comparison["anchors"] if a["review_state"] == "pending_review"
    }
    pending_rows = [m for m in comparison["mappings"] if m["review_state"] == "pending_review"]
    assert all(m["scope"] == "row" for m in pending_rows)
    assert pending_anchors == {
        anchor_id for mapping in pending_rows for anchor_id in mapping["anchor_ids"]
    }
    assert all(mapping["protocol_differences"] for mapping in pending_rows)


def test_edge_duplicate_unit_mapping_is_rejected(survey, comparison):
    payload = json.loads(json.dumps(comparison))
    unit = next(m for m in payload["mappings"] if m["scope"] == "unit")
    twin = json.loads(json.dumps(unit))
    twin["mapping_id"] = twin["mapping_id"] + "_twin"
    payload["mappings"].append(twin)
    with pytest.raises(ValueError, match="same result unit"):
        validate_comparison_coverage(survey, payload)


def test_edge_missing_unit_mapping_is_rejected(survey, comparison):
    payload = json.loads(json.dumps(comparison))
    dropped = next(m["mapping_id"] for m in payload["mappings"] if m["scope"] == "unit")
    payload["mappings"] = [m for m in payload["mappings"] if m["mapping_id"] != dropped]
    with pytest.raises(ValueError, match="no comparison mapping"):
        validate_comparison_coverage(survey, payload)


def test_edge_typoed_token_is_caught_either_way(survey, comparison):
    """A typo'd token drops a real unit *and* leaves a phantom mapping behind.

    Both are fatal; which one is reported first is an implementation choice, so
    this asserts each path separately: a token edit on an existing mapping must
    surface the uncovered unit, and a purely extra mapping must surface itself.
    """
    edited = json.loads(json.dumps(comparison))
    unit = next(m for m in edited["mappings"] if m["scope"] == "unit")
    unit["result_selector"]["c_token"] = "0.9"
    with pytest.raises(ValueError, match="no comparison mapping"):
        validate_comparison_coverage(survey, edited)

    extended = json.loads(json.dumps(comparison))
    unit = next(m for m in extended["mappings"] if m["scope"] == "unit")
    phantom = json.loads(json.dumps(unit))
    phantom["mapping_id"] = phantom["mapping_id"] + "_phantom"
    phantom["result_selector"]["c_token"] = "0.9"
    extended["mappings"].append(phantom)
    with pytest.raises(ValueError, match="cover no expected result unit"):
        validate_comparison_coverage(survey, extended)


# --- the shipped file itself -------------------------------------------------


def test_basic_shipped_file_binds_the_pinned_protocol_and_excludes_preintegration(survey):
    assert COMPARISON_PATH.is_file()
    shipped = load_comparison_protocol(survey=survey)
    assert shipped["comparison_version"] == "survey-comparison-v3"
    # Grad-PU and PUET are not in the survey matrix, so they must not be pre-registered.
    blob = json.dumps(shipped).lower()
    for absent in ("grad_pu", "gradpu", "puet"):
        assert absent not in blob
