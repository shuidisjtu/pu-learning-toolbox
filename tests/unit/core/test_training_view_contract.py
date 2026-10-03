# tests/unit/core/test_training_views.py

# ruff: noqa: N803, N806, S101

"""The neutral role view: what an estimator may consume without the survey layer.

The core constructs three roles from one training partition -- the positives,
the native unlabeled rows, and the rows the unlabeled risk acts on -- and owns
nothing else.  It knows no ledger, no method name and no manifest, because
estimators import it and may not import the survey experiment layer at all
(see the import-boundary test at the end).

The contract this file pins is mostly about *ownership*: which arrays the view
copied and froze, which it borrowed, and what it may never write through.
`frozen=True` on the dataclass prevents rebinding an attribute, not mutating
the array behind it, so both halves have to be tested separately.
"""

import subprocess
import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pu_toolbox.core.training_views import TrainingView, build_training_view

pytestmark = pytest.mark.unit


def _data():
    X = np.arange(24).reshape(6, 4)
    y_pu = np.array([1, 0, 0, 1, 0, 0])
    return X, y_pu


# --- the frozen vocabulary ----------------------------------------------------


def test_basic_role_and_view_vocabulary_is_frozen():
    """The vocabulary is a serialisation contract: role names become dict keys
    in hashed survey artifacts, so their exact value is pinned here rather than
    derived from the implementation.
    """
    from typing import get_args

    from pu_toolbox.core.training_views import ROLES, RUN_VIEWS, RunView, ViewRole

    assert ROLES == ("train", "pu_val", "clean_val", "test")
    assert RUN_VIEWS == ("os", "ts")
    # The runtime tuples must stay derived from the aliases, not restated.
    assert get_args(ViewRole) == ROLES
    assert get_args(RunView) == RUN_VIEWS


# --- the three roles, under each view ----------------------------------------


def test_basic_os_view_keeps_the_loss_role_equal_to_native_unlabeled():
    X, y_pu = _data()
    view = build_training_view(X, y_pu, requested_view="os")

    np.testing.assert_array_equal(view.positive_features, X[[0, 3]])
    np.testing.assert_array_equal(view.native_unlabeled_features, X[[1, 2, 4, 5]])
    np.testing.assert_array_equal(view.loss_unlabeled_features, X[[1, 2, 4, 5]])
    assert view.calibration_applied is False


def test_basic_ts_view_appends_the_positives_after_the_original_unlabeled():
    """The order is part of the contract: original U first, then the added P."""
    X, y_pu = _data()
    view = build_training_view(X, y_pu, requested_view="ts")

    np.testing.assert_array_equal(view.loss_unlabeled_features, X[[1, 2, 4, 5, 0, 3]])
    assert view.calibration_applied is True


def test_basic_the_positive_role_is_the_positives_under_both_views():
    """The positive term never follows the view; only the unlabeled role does."""
    X, y_pu = _data()

    for requested in ("os", "ts"):
        view = build_training_view(X, y_pu, requested_view=requested)
        np.testing.assert_array_equal(view.positive_features, X[[0, 3]])


@pytest.mark.parametrize("requested", ["os", "ts"])
def test_param_custom_indices_are_carried_into_the_roles(requested):
    """Source indices travel with the rows, so a consumer can trace them back."""
    X, y_pu = _data()
    indices = np.array([10, 11, 12, 13, 14, 15])
    view = build_training_view(X, y_pu, requested_view=requested, indices=indices)

    assert view.positive_indices.tolist() == [10, 13]
    assert view.native_unlabeled_indices.tolist() == [11, 12, 14, 15]
    expected_loss = [11, 12, 14, 15, 10, 13] if requested == "ts" else [11, 12, 14, 15]
    assert view.loss_unlabeled_indices.tolist() == expected_loss


# --- refusals -----------------------------------------------------------------


def test_edge_a_calibrated_view_is_refused_for_a_non_train_role():
    X, y_pu = _data()

    for role in ("pu_val", "clean_val", "test"):
        with pytest.raises(ValueError, match="train role only"):
            build_training_view(X, y_pu, requested_view="ts", role=role)


def test_edge_a_view_needs_both_groups_and_well_formed_indices():
    X, y_pu = _data()

    with pytest.raises(ValueError, match="both labeled-positive and unlabeled"):
        build_training_view(X, np.ones(len(X), dtype=int))
    with pytest.raises(ValueError, match="both labeled-positive and unlabeled"):
        build_training_view(X, np.zeros(len(X), dtype=int))
    with pytest.raises(ValueError, match="unique"):
        build_training_view(X, y_pu, indices=np.zeros(len(X), dtype=int))
    with pytest.raises(ValueError, match="must use canonical PU labels"):
        build_training_view(X, np.array([1, 0, 2, 1, 0, 0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        build_training_view(X, y_pu.reshape(2, 3))
    with pytest.raises(ValueError, match="non-empty"):
        build_training_view(np.empty((0, 4)), np.empty(0, dtype=int))


@pytest.mark.parametrize("role", ["pu_val", "clean_val", "test"])
def test_edge_every_role_needs_both_groups_not_only_the_train_role(role):
    """The non-train roles exist so a calibrated request for one can be refused
    by name -- not to describe evaluation partitions.  An empty role is refused
    whatever the role is, so relaxing this for a single-class validation fold
    has to be a deliberate decision rather than a quiet "fix"."""
    X, _ = _data()

    with pytest.raises(ValueError, match="both labeled-positive and unlabeled"):
        build_training_view(X, np.ones(len(X), dtype=int), role=role)
    with pytest.raises(ValueError, match="both labeled-positive and unlabeled"):
        build_training_view(X, np.zeros(len(X), dtype=int), role=role)


# --- ownership: what is copied and frozen, what is borrowed -------------------


def test_edge_the_callers_arrays_are_not_modified():
    X, y_pu = _data()
    indices = np.array([10, 11, 12, 13, 14, 15])
    X_before, y_before, indices_before = X.copy(), y_pu.copy(), indices.copy()

    build_training_view(X, y_pu, requested_view="ts", indices=indices)

    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y_pu, y_before)
    np.testing.assert_array_equal(indices, indices_before)
    # Freezing an owned copy must not reach back into the caller's array.
    assert indices.flags.writeable


def test_edge_owned_arrays_cannot_be_written_through():
    """`frozen=True` binds attributes; it does not make the arrays behind them
    read-only, so the view has to do that itself where it owns them."""
    X, y_pu = _data()
    view = build_training_view(X, y_pu, requested_view="ts", indices=np.arange(6))

    with pytest.raises(ValueError):
        view.loss_unlabeled_positions[:] = 0
    with pytest.raises(ValueError):
        view.positive_positions[:] = 0
    with pytest.raises(ValueError):
        view.source_indices[:] = 0
    with pytest.raises(FrozenInstanceError):
        view.requested_view = "ts"


def test_edge_the_features_are_borrowed_not_copied():
    """Bulk arrays are borrowed: copying a whole training set to hand it back
    unchanged is the memory blow-up the index-first design exists to avoid."""
    X, y_pu = _data()
    view = build_training_view(X, y_pu)

    assert np.shares_memory(view.source_features, X)
    assert np.shares_memory(view.source_labels, y_pu)


def test_basic_a_reference_consumer_reads_roles_and_denominators():
    """阶段 D 的 reference consumer：只取角色、只对分母，不含任何算法公式。

    它存在的意义是证明核心 API 可被消费，而不是证明某个算法已接入——
    生产 estimator 的迁移按 §6 阶段 D 推迟到第一个真实待接入算法。
    """

    def empirical_terms(view: TrainingView) -> tuple[int, int]:
        return len(view.positive_features), len(view.loss_unlabeled_features)

    X, y_pu = _data()
    assert empirical_terms(build_training_view(X, y_pu, requested_view="os")) == (2, 4)
    assert empirical_terms(build_training_view(X, y_pu, requested_view="ts")) == (2, 6)


# --- determinism, and the boundary that made this module neutral --------------


def test_determ_the_same_input_builds_the_same_view():
    X, y_pu = _data()
    first = build_training_view(X, y_pu, requested_view="ts", indices=np.arange(6))
    second = build_training_view(X.copy(), y_pu.copy(), requested_view="ts", indices=np.arange(6))

    for name in ("positive_positions", "native_unlabeled_positions", "loss_unlabeled_positions"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def test_determ_estimators_import_without_the_survey_experiment_layer():
    """§4.0 的依赖方向：estimators → core 允许，estimators → experiment 禁止。

    必须在子进程里跑——同一解释器内只要先导入过 experiment，这条边界就
    再也测不出来了。基线（修订时实测）：本导入拉起的 experiment 模块数为 0。
    """
    code = (
        "import sys; import pu_toolbox.estimators.risk.upu; "
        "pulled = sorted(m for m in sys.modules if m.startswith('pu_toolbox.experiment')); "
        "assert not pulled, f'estimator import pulled the survey layer: {pulled}'"
    )
    completed = subprocess.run(  # noqa: S603 - fixed interpreter, code built here
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
