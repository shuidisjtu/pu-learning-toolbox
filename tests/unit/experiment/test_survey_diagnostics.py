# tests/unit/experiment/test_survey_diagnostics.py

# ruff: noqa: N803, N806, S101

"""The diagnostic cards: which signals fire, and what a card may not say.

Rows are built by hand and the native-view map is passed in, so each case states
the one contrast it is about.  The card is a prompt to look, never a verdict: no
test here lets a flag read as "the implementation is wrong".
"""

import pytest

from pu_toolbox.experiment.survey_diagnostics import (
    NOT_COMPUTED,
    SIGNALS,
    build_cards,
    render_cards,
)

pytestmark = pytest.mark.unit

_SEEDS = [0, 1, 2, 3, 4]


def _row(
    *,
    method="upu",
    dataset="mnist",
    path="native_cnn",
    view="os-compatible",
    mechanism="scar",
    protocol="OA",
    c="0.5",
    mean=0.9,
    std=0.01,
    observed=5,
):
    seeds = _SEEDS[:observed]
    return {
        "result_identity": {
            "method": method,
            "dataset": dataset,
            "labeling_mechanism": "c_independent" if mechanism == "pn_oracle" else mechanism,
            "c_token": c,
            "selection_protocol": protocol.lower(),
            "metric": "accuracy",
            "training_path": path,
        },
        "row_key": {
            "comparability_group": f"{dataset}/{path}/minibatch",
            "run_view": view,
            "mechanism": mechanism,
            "method": method,
            "selection_protocol": protocol,
            "c_token": c,
        },
        "metric": {
            "mean": mean,
            "std": std,
            "mean_percent": mean * 100,
            "std_percent": std * 100,
            "n_observed": observed,
            "n_expected": 5,
            "missing_seeds": [s for s in _SEEDS if s not in seeds],
        },
        "status": "formal",
        "reasons": [],
    }


def _report(*rows):
    return {"protocol_version": "survey-v3.0", "rows": list(rows)}


def _flags(cards, method, signal):
    return [o for o in cards[method]["signals"][signal] if o["flag"]]


def _seen(cards, method, signal):
    return cards[method]["signals"][signal]


def test_basic_a_card_exists_per_method_and_lists_every_signal():
    cards = build_cards(_report(_row(method="a"), _row(method="b")), native_view={})

    assert set(cards) == {"a", "b"}
    for card in cards.values():
        assert set(card["signals"]) == set(SIGNALS)


def test_basic_signals_that_cannot_be_read_from_the_report_are_listed_as_not_computed():
    card = build_cards(_report(_row()), native_view={})["upu"]

    assert {item["signal"] for item in card["not_computed"]} == set(NOT_COMPUTED)
    assert all(item["reason"] for item in card["not_computed"])


def test_edge_a_row_missing_seeds_never_enters_a_signal():
    short = _row(method="a", observed=1, std=0.0)

    cards = build_cards(_report(short), native_view={})

    assert all(not obs for obs in cards["a"]["signals"].values())
    assert cards["a"]["excluded_rows"] == 1


@pytest.mark.parametrize(
    ("scar", "sar", "flagged", "delta"),
    [(0.80, 0.85, True, 5.0), (0.90, 0.80, False, -10.0)],
    ids=["sar-better-is-flagged", "sar-worse-is-only-reported"],
)
def test_basic_sar_better_than_scar_is_flagged(scar, sar, flagged, delta):
    rows = [_row(mechanism="scar", mean=scar), _row(mechanism="sar_lbe_a", mean=sar)]

    cards = build_cards(_report(*rows), native_view={})

    assert bool(_flags(cards, "upu", "mechanism_scar_to_sar")) is flagged
    assert _seen(cards, "upu", "mechanism_scar_to_sar")[0]["delta_points"] == pytest.approx(delta)


@pytest.mark.parametrize(
    ("native", "os_mean", "ts_mean", "flagged"),
    [
        ("ts", 0.90, 0.85, True),  # native TS, but better under OS
        ("ts", 0.900, 0.901, True),  # native TS, calibration changed nothing
        ("ts", 0.80, 0.90, False),  # native TS, calibration helps
        ("os", 0.90, 0.85, False),  # not native TS: contrast shown, never flagged
    ],
)
def test_basic_a_view_contrast_is_flagged_only_for_a_native_ts_method(
    native, os_mean, ts_mean, flagged
):
    rows = [_row(view="os-compatible", mean=os_mean), _row(view="ts-compatible", mean=ts_mean)]

    cards = build_cards(_report(*rows), native_view={"upu": native})

    assert bool(_flags(cards, "upu", "view_os_ts")) is flagged
    assert _seen(cards, "upu", "view_os_ts")


@pytest.mark.parametrize(
    ("means", "std", "flagged"),
    [
        ((0.70, 0.80, 0.90), 0.01, False),  # clean, monotone
        ((0.80, 0.70, 0.90), 0.01, True),  # dip
        ((0.50, 0.80, 0.82), 0.01, True),  # cliff between adjacent c
        ((0.900, 0.900, 0.900), 0.02, True),  # does not move at all
    ],
    ids=["clean", "dip", "cliff", "insensitive"],
)
def test_basic_c_curve_shapes_are_flagged_only_when_anomalous(means, std, flagged):
    rows = [_row(c=c, mean=m, std=std) for c, m in zip(("0.1", "0.3", "0.5"), means, strict=True)]

    cards = build_cards(_report(*rows), native_view={})

    assert bool(_flags(cards, "upu", "c_sensitivity")) is flagged


@pytest.mark.parametrize(
    ("method_mean", "flagged"), [(0.95, True), (0.80, False)], ids=["above-oracle", "below-oracle"]
)
def test_basic_a_method_above_its_pn_oracle_is_flagged(method_mean, flagged):
    method = _row(method="a", mean=method_mean)
    oracle = _row(method="pn_oracle", mechanism="pn_oracle", c="c_independent", mean=0.90)

    cards = build_cards(_report(method, oracle), native_view={})

    assert bool(_flags(cards, "a", "pn_oracle_alignment")) is flagged
    assert _seen(cards, "a", "pn_oracle_alignment")[0]["gap_points"] == pytest.approx(
        90.0 - method_mean * 100
    )


def test_basic_a_large_oa_over_pa_gap_is_flagged():
    oa = _row(protocol="OA", mean=0.90)
    pa = _row(protocol="PA", mean=0.70)

    assert _flags(build_cards(_report(oa, pa), native_view={}), "upu", "pa_oa_gap")


def test_basic_seed_spread_larger_than_the_gap_to_the_next_method_is_flagged():
    rows = [
        _row(method="a", mean=0.900, std=0.030),
        _row(method="b", mean=0.895, std=0.001),
    ]

    cards = build_cards(_report(*rows), native_view={})

    assert _flags(cards, "a", "seed_variance")
    assert not _flags(cards, "b", "seed_variance")


def test_basic_the_same_method_on_two_paths_is_compared_and_a_large_gap_is_flagged():
    native = _row(path="native_cnn", mean=0.95)
    adapter = _row(path="cnn_feature_adapter", mean=0.70)

    cards = build_cards(_report(native, adapter), native_view={})

    assert _flags(cards, "upu", "backbone_path")


def test_edge_every_card_line_names_its_signal_and_labels_observation_versus_inference():
    rows = [_row(mechanism="scar", mean=0.80), _row(mechanism="sar_lbe_a", mean=0.85)]

    text = render_cards(build_cards(_report(*rows), native_view={}))

    assert "mechanism_scar_to_sar" in text
    assert "本实验观测" in text
    assert "推断" in text
    assert "实现有错" not in text.replace("不判定实现有错", "")


def test_param_rendering_never_claims_agreement_with_the_paper():
    text = render_cards(build_cards(_report(_row()), native_view={}))

    assert "与论文一致" not in text
    assert "外部锚点" in text


def test_determ_the_same_report_builds_the_same_cards():
    rows = [_row(method="b", c="0.1"), _row(method="a", c="0.3"), _row(method="a", c="0.1")]

    assert build_cards(_report(*rows), native_view={}) == build_cards(
        _report(*reversed(rows)), native_view={}
    )
