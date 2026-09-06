# ruff: noqa: N803, N806

import numpy as np
import pytest

from pu_toolbox.experiment import calibrate_ts_os_batch

pytestmark = pytest.mark.unit


def _batch():
    X = np.arange(24).reshape(6, 4)
    y_pu = np.array([1, 0, 0, 1, 0, 0])
    indices = np.array([10, 11, 12, 13, 14, 15])
    return X, y_pu, indices


def test_basic_ts_view_keeps_positive_loss_and_adds_positives_to_unlabeled_loss():
    X, y_pu, indices = _batch()
    view = calibrate_ts_os_batch(
        X,
        y_pu,
        os_or_ts="ts",
        native_sampling_assumption="ts",
        method_name="native_ts_method",
        indices=indices,
    )

    np.testing.assert_array_equal(view.positive_features, X[[0, 3]])
    np.testing.assert_array_equal(view.loss_unlabeled_features, X[[1, 2, 4, 5, 0, 3]])
    assert view.positive_indices.tolist() == [10, 13]
    assert view.loss_unlabeled_indices.tolist() == [11, 12, 14, 15, 10, 13]
    assert view.run_view == "TS-compatible"
    assert view.calibration_applied is True


def test_basic_os_view_preserves_original_unlabeled_subset():
    X, y_pu, _ = _batch()
    view = calibrate_ts_os_batch(
        X,
        y_pu,
        native_sampling_assumption="os",
        method_name="native_os_method",
    )

    np.testing.assert_array_equal(view.positive_features, X[[0, 3]])
    np.testing.assert_array_equal(view.loss_unlabeled_features, X[[1, 2, 4, 5]])
    assert view.run_view == "OS"
    assert view.calibration_applied is False


@pytest.mark.parametrize("native_assumption", ["ts", "both"])
def test_param_ts_view_accepts_only_explicit_ts_capability(native_assumption):
    X, y_pu, _ = _batch()
    view = calibrate_ts_os_batch(
        X,
        y_pu,
        os_or_ts="ts",
        native_sampling_assumption=native_assumption,
        method_name="method",
    )
    assert view.manifest["native_sampling_assumption"] == native_assumption
    assert view.manifest["positive_rows_added_to_unlabeled_loss"] == 2


def test_determ_manifest_indices_and_hash_are_reproducible():
    X, y_pu, indices = _batch()
    kwargs = {
        "os_or_ts": "ts",
        "native_sampling_assumption": "ts",
        "method_name": "method",
        "indices": indices,
    }
    first = calibrate_ts_os_batch(X, y_pu, **kwargs)
    second = calibrate_ts_os_batch(X.copy(), y_pu.copy(), **kwargs)

    assert first.manifest == second.manifest
    assert len(first.manifest["indices_sha256"]) == 64


def test_edge_rejects_ts_for_os_method_and_non_train_roles():
    X, y_pu, _ = _batch()
    with pytest.raises(ValueError, match="declared native to TS"):
        calibrate_ts_os_batch(
            X,
            y_pu,
            os_or_ts="ts",
            native_sampling_assumption="os",
            method_name="method",
        )
    for role in ("pu_val", "clean_val", "test"):
        with pytest.raises(ValueError, match="train-only"):
            calibrate_ts_os_batch(
                X,
                y_pu,
                os_or_ts="ts",
                native_sampling_assumption="ts",
                role=role,
                method_name="method",
            )


def test_edge_rejects_batches_without_both_pu_groups_and_bad_indices():
    X, _, _ = _batch()
    with pytest.raises(ValueError, match="both labeled-positive and unlabeled"):
        calibrate_ts_os_batch(
            X,
            np.ones(len(X), dtype=int),
            native_sampling_assumption="os",
            method_name="method",
        )
    with pytest.raises(ValueError, match="unique"):
        calibrate_ts_os_batch(
            X,
            np.array([1, 0, 0, 1, 0, 0]),
            native_sampling_assumption="os",
            method_name="method",
            indices=np.zeros(len(X), dtype=int),
        )
