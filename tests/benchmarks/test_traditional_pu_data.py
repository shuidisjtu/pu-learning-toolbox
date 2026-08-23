"""Data protocol tests (contract §2): SCAR/SAR/PNU shapes, h, ill-conditioning."""

# Dataset matrices follow sklearn's conventional X/y names.
# ruff: noqa: N803, N806

import numpy as np
import pytest

from benchmarks.traditional_pu.data import (
    is_ill_conditioned,
    make_pnu_data,
    make_sar_linear_data,
    make_scar_data,
    pnu_counts,
)

pytestmark = pytest.mark.unit


class TestScarData:
    def test_shape_and_contract_fields(self):
        X, y_pu, y_true, meta = make_scar_data(
            n_samples=500,
            n_features=5,
            class_prior=0.3,
            separation=2.0,
            label_frequency=0.5,
            random_state=0,
        )
        assert X.shape == (500, 5)
        assert set(np.unique(y_pu)) <= {0, 1}
        assert y_true.shape == (500,) and set(np.unique(y_true)) <= {0, 1}
        assert meta["mechanism"] == "scar"
        assert meta["real_h"] == pytest.approx(0.5)
        assert meta["pi_h_well_conditioned"] is True

    def test_deterministic_per_seed(self):
        a = make_scar_data(400, 5, 0.5, 2.0, 0.5, random_state=3)[0]
        b = make_scar_data(400, 5, 0.5, 2.0, 0.5, random_state=3)[0]
        assert np.array_equal(a, b)


class TestIllConditioned:
    def test_edge_known_cases(self):
        assert is_ill_conditioned(0.5, 1.0, tol=1e-6) is True
        assert is_ill_conditioned(0.3, 0.5, tol=1e-6) is False


class TestPnuData:
    def test_param_ratio_and_labels(self):
        X, y_pnu, y_true = make_pnu_data(
            n_p=25, n_n=25, n_u=100, n_features=5, separation=2.0, random_state=1
        )
        assert X.shape == (150, 5)
        ratios = np.bincount(np.asarray(y_pnu) + 1, minlength=3)
        assert ratios[2] == 25 and ratios[0] == 25 and ratios[1] == 100
        assert set(np.unique(y_true)) == {-1, 1}

    def test_basic_labeled_labels_consistent_with_ground_truth(self):
        # Contract §2.2: trusted labels are ground truth — every labeled
        # positive (y_pnu=+1) is a true positive (y_true=+1) and every
        # trusted negative (y_pnu=-1) is a true negative (y_true=-1).
        X, y_pnu, y_true = make_pnu_data(
            n_p=25, n_n=25, n_u=100, n_features=5, separation=2.0, random_state=1
        )
        labeled_pos = y_pnu == 1
        labeled_neg = y_pnu == -1
        assert labeled_pos.any() and labeled_neg.any()
        assert (y_true[labeled_pos] == 1).all()
        assert (y_true[labeled_neg] == -1).all()


class TestPnuCounts:
    def test_param_counts_track_scenario_size(self):
        # Contract §2.1/§2.2: the small (400) and mid (2000) scales must
        # produce different data — counts derive from ``n_samples``, so the
        # two cells no longer share a single hard-coded (25, 25, 100) size.
        small = pnu_counts("1:1:4", 400)
        mid = pnu_counts("1:1:4", 2000)
        assert small != mid
        assert sum(small) == 400
        assert sum(mid) == 2000
        assert mid == (334, 333, 1333)

    def test_param_counts_keep_ratio_layout(self):
        # Golden counts (largest-fraction remainder attached to U on ties).
        assert pnu_counts("1:1:4", 400) == (67, 66, 267)
        assert pnu_counts("1:2:4", 400) == (57, 114, 229)
        assert pnu_counts("1:1:8", 400) == (40, 40, 320)
        assert pnu_counts("1:2:4", 2000) == (286, 571, 1143)
        assert pnu_counts("1:1:8", 2000) == (200, 200, 1600)

    def test_edge_tiny_size_keeps_groups_nonempty(self):
        assert pnu_counts("1:1:4", 6) == (1, 1, 4)
        # all groups at least 1 even when the ratio would otherwise starve one
        assert min(pnu_counts("1:2:4", 4)) == 1

    def test_edge_unknown_ratio_rejected(self):
        with pytest.raises(ValueError, match="unknown PNU ratio"):
            pnu_counts("9:9:9", 400)


class TestSarLinearData:
    def test_basic_mechanism_recorded(self):
        _, _, _, meta = make_sar_linear_data(400, 5, 0.5, 2.0, 0.5, strength=1.0, random_state=2)
        assert meta["mechanism"] == "linear"
