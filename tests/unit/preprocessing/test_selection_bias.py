# ruff: noqa: N803, N806

"""Tests for public SCAR/SAR simulation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.preprocessing import (
    make_sar_dataset,
    make_sar_labels,
    make_sar_propensity,
)


@pytest.fixture
def labeled_data():
    rng = np.random.RandomState(4)
    X = rng.normal(size=(400, 3))
    y_true = np.r_[np.ones(160, dtype=int), np.zeros(240, dtype=int)]
    return X, y_true


@pytest.mark.unit
class TestSARPropensity:
    def test_basic_scar_propensity_and_negative_zero(self, labeled_data):
        X, y_true = labeled_data
        propensity = make_sar_propensity(
            X,
            y_true,
            mechanism="scar",
            label_frequency=0.35,
        )
        assert propensity.shape == (400,)
        assert np.all(propensity[y_true == 0] == 0)
        assert np.all(propensity[y_true == 1] == pytest.approx(0.35))

    @pytest.mark.parametrize("mechanism", ["linear", "nonlinear", "non-linear"])
    def test_mechanism_calibrates_positive_frequency(self, labeled_data, mechanism):
        X, y_true = labeled_data
        propensity = make_sar_propensity(
            X,
            y_true,
            mechanism=mechanism,
            label_frequency=0.27,
            strength=1.5,
        )
        positive_propensity = propensity[y_true == 1]
        assert positive_propensity.mean() == pytest.approx(0.27, abs=1e-12)
        assert positive_propensity.std() > 0.05
        assert np.all((propensity >= 0) & (propensity <= 1))

    def test_feature_weights_change_linear_ordering(self, labeled_data):
        X, y_true = labeled_data
        first = make_sar_propensity(
            X,
            y_true,
            feature_weights=np.array([1.0, 0.0, 0.0]),
        )
        second = make_sar_propensity(
            X,
            y_true,
            feature_weights=np.array([0.0, 1.0, 0.0]),
        )
        assert not np.allclose(first, second)

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"mechanism": "unknown"}, ValueError, "Unknown mechanism"),
            ({"label_frequency": 0.0}, ValueError, "label_frequency"),
            ({"label_frequency": 1.1}, ValueError, "label_frequency"),
            ({"strength": -1.0}, ValueError, "strength"),
            ({"feature_weights": [0.0, 0.0, 0.0]}, ValueError, "non-zero"),
            ({"feature_weights": [1.0, 0.0]}, ValueError, "n_features"),
        ],
    )
    def test_invalid_parameters_raise_user_facing_errors(
        self,
        labeled_data,
        kwargs,
        error,
        match,
    ):
        X, y_true = labeled_data
        with pytest.raises(error, match=match):
            make_sar_propensity(X, y_true, **kwargs)

    @pytest.mark.parametrize(
        ("X", "y_true", "match"),
        [
            (np.ones(5), np.array([1, 1, 0, 0, 0]), "X must be 2-D"),
            (np.ones((5, 2)), np.array([1, 0]), "inconsistent lengths"),
            (np.ones((3, 2)), np.array([1, 0, 2]), "only.*0, 1"),
            (np.array([[1.0, np.nan], [0.0, 1.0]]), np.array([1, 0]), "finite"),
        ],
    )
    def test_invalid_x_y_validation(self, X, y_true, match):
        with pytest.raises(ValueError, match=match):
            make_sar_propensity(X, y_true)

    def test_edge_no_positive_and_unit_frequency(self, labeled_data):
        X, y_true = labeled_data
        no_positive = make_sar_propensity(X, np.zeros(len(y_true), dtype=int))
        assert np.all(no_positive == 0)

        full = make_sar_propensity(X, y_true, label_frequency=1.0)
        assert np.all(full[y_true == 1] == 1)
        assert np.all(full[y_true == 0] == 0)


@pytest.mark.unit
class TestSARLabels:
    def test_basic_labels_never_mark_true_negatives(self, labeled_data):
        X, y_true = labeled_data
        y_pu, propensity = make_sar_labels(
            X,
            y_true,
            mechanism="nonlinear",
            label_frequency=0.4,
            random_state=8,
            return_propensity=True,
        )
        assert y_pu.dtype == int
        assert set(np.unique(y_pu)) <= {0, 1}
        assert np.all(y_pu[y_true == 0] == 0)
        assert propensity[y_true == 1].mean() == pytest.approx(0.4)

    def test_seed_reproducibility(self, labeled_data):
        X, y_true = labeled_data
        first = make_sar_labels(X, y_true, random_state=17)
        second = make_sar_labels(X, y_true, random_state=17)
        np.testing.assert_array_equal(first, second)

    def test_edge_ensure_labeled_for_tiny_frequency(self):
        X = np.array([[0.0], [1.0], [-1.0]])
        y_true = np.array([1, 0, 0])
        guarded = make_sar_labels(
            X,
            y_true,
            label_frequency=1e-12,
            ensure_labeled=True,
            random_state=0,
        )
        pure_draw = make_sar_labels(
            X,
            y_true,
            label_frequency=1e-12,
            ensure_labeled=False,
            random_state=0,
        )
        assert guarded.sum() == 1
        assert pure_draw.sum() == 0


@pytest.mark.unit
class TestSARDataset:
    def test_dataset_output_shapes_and_counts(self):
        X, y_pu, y_true, propensity = make_sar_dataset(
            n_samples=500,
            n_features=4,
            class_prior=0.3,
            label_frequency=0.4,
            random_state=3,
        )
        assert X.shape == (500, 4)
        assert y_pu.shape == y_true.shape == propensity.shape == (500,)
        assert y_true.sum() == 150
        assert np.all(y_true[y_pu == 1] == 1)
        assert propensity[y_true == 1].mean() == pytest.approx(0.4, abs=1e-12)

    def test_dataset_mechanism_boundary_behavior(self):
        outputs = {
            mechanism: make_sar_dataset(
                n_samples=800,
                mechanism=mechanism,
                strength=1.5,
                random_state=9,
            )
            for mechanism in ("scar", "linear", "nonlinear")
        }
        scar_propensity = outputs["scar"][3]
        linear_propensity = outputs["linear"][3]
        nonlinear_propensity = outputs["nonlinear"][3]
        positive = outputs["scar"][2] == 1
        assert scar_propensity[positive].std() == pytest.approx(0.0)
        assert linear_propensity[positive].std() > 0
        assert not np.allclose(linear_propensity, nonlinear_propensity)

    def test_dataset_determinism_and_invalid_size(self):
        first = make_sar_dataset(random_state=21)
        second = make_sar_dataset(random_state=21)
        for left, right in zip(first, second, strict=True):
            np.testing.assert_allclose(left, right)
        with pytest.raises(ValueError, match="n_samples must be >= 2"):
            make_sar_dataset(n_samples=1)
