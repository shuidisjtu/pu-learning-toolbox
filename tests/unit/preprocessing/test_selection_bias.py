# ruff: noqa: N803, N806

"""Tests for public SCAR/SAR simulation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.preprocessing import (
    PULabelingConfig,
    SourcePartition,
    construct_pu_dataset,
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
class TestSARConstruction:
    def test_determ_linear_bernoulli_draw_uses_source_propensities(self, labeled_data):
        X, y_true = labeled_data
        config = PULabelingConfig("sar", "bernoulli", 8, c=0.4, propensity_rule="linear")
        result = construct_pu_dataset(
            SourcePartition(X, y_true, "fixture", "train"),
            config,
            positive_classes=(1,),
            negative_classes=(0,),
        )
        propensity = make_sar_propensity(X, y_true, mechanism="linear", label_frequency=0.4)
        expected = (np.random.RandomState(8).uniform(size=len(y_true)) < propensity).astype(int)
        np.testing.assert_allclose(result.audit.propensity, propensity)
        np.testing.assert_array_equal(result.observations.labels, expected)
        assert np.all(result.observations.labels[y_true == 0] == 0)
