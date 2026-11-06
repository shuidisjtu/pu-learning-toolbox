# tests/unit/experiment/test_survey_conclusions.py

# ruff: noqa: N803, N806, S101

"""The conclusion layer: what it may write, and what it must refuse.

The cards pin which signal fires; these pin the artifact the delivery asks for --
every conclusion names its signals, three kinds stay apart, and the builder
refuses a conclusion it cannot ground instead of writing a vague one.
"""

import pytest

from pu_toolbox.experiment.survey_conclusions import (
    KINDS,
    SIGNALS,
    ConclusionError,
    build_conclusions,
    conclusion_rows,
    render_conclusions,
    validate_conclusion,
)
from pu_toolbox.experiment.survey_diagnostics import NOT_COMPUTED, build_cards

pytestmark = pytest.mark.unit

_SEEDS = [0, 1, 2, 3, 4]


def _row(*, method="upu", c="0.5", mean=0.9, std=0.01, observed=5):
    return {
        "result_identity": {
            "method": method,
            "dataset": "mnist",
            "labeling_mechanism": "scar",
            "c_token": c,
            "selection_protocol": "oa",
            "metric": "accuracy",
            "training_path": "native_cnn",
        },
        "row_key": {
            "comparability_group": "mnist/native_cnn/minibatch",
            "run_view": "os-compatible",
            "mechanism": "scar",
            "method": method,
            "selection_protocol": "OA",
            "c_token": c,
        },
        "metric": {
            "mean": mean,
            "std": std,
            "mean_percent": mean * 100,
            "std_percent": std * 100,
            "n_observed": observed,
            "n_expected": 5,
            "missing_seeds": [seed for seed in _SEEDS if seed >= observed],
        },
        "status": "formal",
        "reasons": [],
    }


def _report(*rows):
    return {"protocol_version": "survey-v3.0", "protocol_sha256": "ab" * 32, "rows": list(rows)}


def _cards(*rows):
    return build_cards(_report(*rows), native_view={})


def _dip():
    """A c curve that dips: non-monotone is the one anomaly the card flags."""
    return _cards(
        _row(c="0.1", mean=0.80),
        _row(c="0.3", mean=0.70),
        _row(c="0.5", mean=0.90),
    )


def test_basic_every_conclusion_names_its_signals_and_a_registered_kind():
    bundle = build_conclusions(_report(), _dip())

    assert bundle["conclusions"]
    for record in bundle["conclusions"]:
        assert record["kind"] in KINDS
        assert record["signals"]
        assert set(record["signals"]) <= set(SIGNALS)


def test_basic_a_flagged_signal_yields_an_observation_and_an_inference_citing_it():
    bundle = build_conclusions(_report(), _dip())
    kinds = {record["conclusion_id"]: record["kind"] for record in bundle["conclusions"]}
    observations = [cid for cid, kind in kinds.items() if kind == "本实验观测"]
    inferences = [cid for cid, kind in kinds.items() if kind == "推断"]

    assert observations and inferences
    by_id = {record["conclusion_id"]: record for record in bundle["conclusions"]}
    for cid in inferences:
        cited = by_id[cid]["cites"]
        assert cited and all(kinds[item] == "本实验观测" for item in cited)


def test_basic_rendering_keeps_the_three_kinds_apart_and_never_claims_paper_agreement():
    text = render_conclusions(build_conclusions(_report(), _dip()))

    for kind in KINDS:
        assert kind in text
    assert "c_sensitivity" in text
    assert "与论文一致" not in text
    assert "不判定实现有错" in text


def test_basic_conclusion_rows_follow_the_same_order_as_the_markdown():
    bundle = build_conclusions(_report(), _dip())

    rows = conclusion_rows(bundle)
    ids = [row[0] for row in rows]

    assert ids == [record["conclusion_id"] for record in bundle["conclusions"]]


def test_edge_without_flags_no_conclusion_is_written_but_unattributed_signals_stay_listed():
    quiet = _cards(_row(c="0.1", mean=0.70), _row(c="0.3", mean=0.80), _row(c="0.5", mean=0.90))

    bundle = build_conclusions(_report(), quiet)

    assert bundle["conclusions"] == []
    assert "upu" in bundle["quiet_methods"]
    assert {item["signal"] for item in bundle["not_attributable"]} == set(NOT_COMPUTED)


def test_edge_a_registered_literature_fact_appears_with_its_source():
    fact = {
        "fact_id": "wang-2026-table3",
        "method": "upu",
        "statement": "论文报告该方法随 c 单调改善。",
        "signals": ["c_sensitivity"],
        "source": "Wang et al. 2026, Table 3",
    }

    bundle = build_conclusions(_report(), _dip(), literature=[fact])
    record = next(item for item in bundle["conclusions"] if item["kind"] == "文献事实")

    assert record["source"] == "Wang et al. 2026, Table 3"
    assert record["signals"] == ["c_sensitivity"]
    assert "来源：Wang et al. 2026, Table 3" in render_conclusions(bundle)


def test_param_a_conclusion_without_a_signal_is_refused():
    with pytest.raises(ConclusionError, match="signal"):
        validate_conclusion(
            {"conclusion_id": "x", "kind": "本实验观测", "method": "m", "signals": []}
        )


def test_param_an_unknown_signal_id_is_refused():
    with pytest.raises(ConclusionError, match="unknown signals"):
        validate_conclusion(
            {
                "conclusion_id": "x",
                "kind": "本实验观测",
                "method": "m",
                "signals": ["made_up"],
                "evidence": [{}],
            }
        )


def test_param_a_literature_fact_without_a_source_is_refused():
    with pytest.raises(ConclusionError, match="source"):
        validate_conclusion(
            {
                "conclusion_id": "x",
                "kind": "文献事实",
                "method": "m",
                "signals": ["c_sensitivity"],
                "source": "",
            }
        )


def test_param_an_observation_without_evidence_is_refused():
    with pytest.raises(ConclusionError, match="evidence"):
        validate_conclusion(
            {
                "conclusion_id": "x",
                "kind": "本实验观测",
                "method": "m",
                "signals": ["c_sensitivity"],
                "evidence": [],
            }
        )


def test_param_an_inference_that_cites_nothing_is_refused():
    with pytest.raises(ConclusionError, match="cite"):
        validate_conclusion(
            {
                "conclusion_id": "x",
                "kind": "推断",
                "method": "m",
                "signals": ["c_sensitivity"],
                "follow_up": "回看实现",
                "cites": [],
            }
        )


def test_param_an_inference_citing_an_unknown_conclusion_is_refused():
    with pytest.raises(ConclusionError, match="unknown conclusion"):
        validate_conclusion(
            {
                "conclusion_id": "x",
                "kind": "推断",
                "method": "m",
                "signals": ["c_sensitivity"],
                "follow_up": "回看实现",
                "cites": ["obs:missing"],
            },
            by_id={},
        )


def test_determ_the_same_report_and_literature_build_identical_conclusions():
    fact = {
        "fact_id": "f1",
        "method": "upu",
        "statement": "s",
        "signals": ["c_sensitivity"],
        "source": "src",
    }

    first = build_conclusions(_report(), _dip(), literature=[fact])
    second = build_conclusions(_report(), _dip(), literature=[fact])

    assert first == second
