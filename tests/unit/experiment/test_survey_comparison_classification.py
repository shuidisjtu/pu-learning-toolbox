# ruff: noqa: N803, N806
"""The eligibility classes the shipped matrix must actually respect.

Coverage asks "is every result unit mapped?"; this asks "is each mapping
classified the way the rules require?".  They fail for different reasons --
a missing mapping versus a wrong verdict -- so they are kept apart.
"""

import pytest

from pu_toolbox.experiment.survey_comparison import load_comparison_protocol
from pu_toolbox.experiment.survey_protocol import load_protocol

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def comparison():
    return load_comparison_protocol(survey=load_protocol())


def test_basic_every_scar_pa_unit_is_blocked_pending_the_pa_criterion(comparison):
    """R9 is unresolved, so no PA unit may be numerically adjudicated."""
    pa = [m for m in comparison["mappings"] if m["result_selector"]["selection_protocol"] == "pa"]
    assert pa
    assert {m["eligibility"] for m in pa} == {"blocked_pending_pa_criterion"}
    assert all(not m["anchor_ids"] for m in pa)


def test_basic_non_runnable_units_are_marked_non_runnable(comparison):
    marked = [
        m for m in comparison["mappings"] if m["result_selector"]["c_token"] == "non_runnable"
    ]
    assert len(marked) == 4
    assert {m["eligibility"] for m in marked} == {"non_runnable"}


def test_basic_pn_units_stay_background_only(comparison):
    pn = [m for m in comparison["mappings"] if m["result_selector"]["c_token"] == "c_independent"]
    assert len(pn) == 3
    assert {m["eligibility"] for m in pn} == {"background_only"}


def test_basic_native_cnn_and_adapter_units_never_share_a_mapping(comparison):
    """The two CIFAR paths are separate leaderboards; one mapping cannot span them."""
    by_method = {}
    for mapping in comparison["mappings"]:
        if mapping["scope"] != "unit":
            continue
        selector = mapping["result_selector"]
        # The non-runnable native-CNN oracle row is a marker, not a measurable unit.
        if selector["dataset"] != "cifar10" or selector["c_token"] == "non_runnable":
            continue
        by_method.setdefault(selector["method"], set()).add(selector["training_path"])
    # nnPU is the only runnable CIFAR row trained natively; everything else is the adapter.
    assert by_method["nnpu"] == {"native_cnn"}
    for method, paths in by_method.items():
        if method == "nnpu":
            continue
        assert paths == {"cnn_feature_adapter"}, method
