"""A TS-rejection-only fit parameter is not a calibration capability."""

import pytest

from pu_toolbox.experiment.method_ledger import load_ledger
from pu_toolbox.experiment.training_views import resolve_training_view
from pu_toolbox.registry import get_algorithm, register_all_builtin_methods

pytestmark = pytest.mark.unit


def test_param_edge_explicit_negative_calibration_disables_default_and_requested_ts():
    register_all_builtin_methods()
    ledger = load_ledger()
    cls = get_algorithm("pulns")
    assert (
        resolve_training_view(ledger, "pulns", None, is_oracle=False, estimator_class=cls) == "os"
    )
    assert (
        resolve_training_view(ledger, "pulns", "os", is_oracle=False, estimator_class=cls) == "os"
    )
    with pytest.raises(ValueError, match="calibration_hooked=false"):
        resolve_training_view(ledger, "pulns", "ts", is_oracle=False, estimator_class=cls)


def test_basic_native_ts_method_is_not_misdescribed_as_os_when_calibration_is_missing():
    ledger = load_ledger()
    with pytest.raises(ValueError) as error:
        resolve_training_view(ledger, "tedn", "ts", is_oracle=False)
    assert "calibration_hooked=false" in str(error.value)
    assert "native to 'ts' sampling" in str(error.value)
    assert "OS-to-TS risk-role substitution hook" in str(error.value)


def test_basic_genpu_real_risk_hook_selects_ts_and_old_frozen_methods_are_unchanged():
    register_all_builtin_methods()
    ledger = load_ledger()
    for name in ("genpu", "nnpu", "upu", "pusb_kernel", "dist_pu", "self_pu"):
        assert (
            resolve_training_view(
                ledger, name, None, is_oracle=False, estimator_class=get_algorithm(name)
            )
            == "ts"
        )
    assert (
        resolve_training_view(
            ledger, "genpu", "os", is_oracle=False, estimator_class=get_algorithm("genpu")
        )
        == "os"
    )
