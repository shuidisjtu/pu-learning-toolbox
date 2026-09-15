# tests/unit/experiment/test_strategies_labeling.py

# ruff: noqa: N806

import numpy as np
import pytest

from pu_toolbox.experiment.strategies import (
    CleanLabelGenerator,
    SARLBEAGenerator,
    SARLBEBGenerator,
    SCARGenerator,
)

pytestmark = pytest.mark.unit


def _y(n):
    return np.array([1] * (n // 2) + [0] * (n % 2 + n // 2))


def test_scar_fixed_count_and_both_classes():
    y_true = _y(100)  # 50 pos
    y_pu, meta = SCARGenerator().generate(np.zeros((100, 4)), y_true, 0.1, seed=0)
    assert int(meta["c_realized"] * 50) == 5  # round(0.1*50)=5
    assert (y_pu == 1).sum() == 5
    assert set(y_pu).issubset({0, 1})


def test_scar_deterministic_seed():
    X = np.zeros((50, 2))
    y = _y(50)
    a, _ = SCARGenerator().generate(X, y, 0.5, seed=7)
    b, _ = SCARGenerator().generate(X, y, 0.5, seed=7)
    assert np.array_equal(a, b)


def test_lbe_a_uses_posterior_scores():
    # 9 negatives at x=0.0..0.8, positives at x=0.9 and x=10.0. The posterior
    # helper is fitted on REAL labels, so P(y=1|x) peaks at x=10.0, and the
    # k=10 + smoothing posterior weight concentrates on it (fixed count = 1).
    X = np.array([[v] for v in np.arange(0, 0.9, 0.1)] + [[0.9], [10.0]], dtype=float)
    y_true = np.array([0] * 9 + [1, 1])
    y_pu, meta = SARLBEAGenerator().generate(X, y_true, 0.2, seed=0)  # -> 1 labeled
    assert y_pu.sum() == 1  # fixed count: exactly one labeled
    assert np.all(y_true[y_pu == 1] == 1)  # pool = positives only (S=1 ⟹ Y=1)
    assert y_pu[10] == 1  # top-score positive (x=10.0) must be picked
    assert meta["mechanism"] == "sar_lbe_a"
    assert meta["posterior_fit_on"] == "real_labels"
    assert meta["k"] == 10
    assert meta["smoothing"] == (0.9, 0.1)


def test_lbe_b_shrink_coef():
    X = np.array([[i] for i in range(10, 20)], dtype=float)
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_pu, meta = SARLBEBGenerator().generate(X, y, 0.1, seed=1)
    assert (y_pu == 1).sum() == 1
    assert np.all(y[y_pu == 1] == 1)  # pool = positives only (S=1 ⟹ Y=1)
    assert meta["shrink_coef"] == 1.0
    assert meta["mechanism"] == "sar_lbe_b"


def test_lbe_b_never_labels_high_weight_negatives():
    # Negatives at low x (posterior ≈ 0) hold the largest LBE-B weights
    # ((2.5 - score)^10 ≈ 9.5e3 vs ≈ 57 for positives), but the sampling
    # pool is true positives only, so no seed may ever label a negative.
    X = np.arange(10, dtype=float).reshape(-1, 1) * 5.0  # x = 0..45
    y_true = np.array([0] * 8 + [1, 1])  # positives at x=40, 45
    for seed in range(20):
        y_pu, _ = SARLBEBGenerator().generate(X, y_true, 0.1, seed=seed)
        assert (y_pu == 1).sum() == 1
        assert np.all(y_true[y_pu == 1] == 1)


def test_lbe_no_positives_returns_all_zero():
    y_true = np.zeros(10, dtype=int)
    for gen in (SARLBEAGenerator(), SARLBEBGenerator()):
        y_pu, meta = gen.generate(np.zeros((10, 2)), y_true, 0.5, seed=0)
        assert np.all(y_pu == 0)
        assert meta["c_realized"] == 0.0


def test_basic_clean_label_generator_passes_true_labels():
    """PN oracle: real labels pass through unchanged and the view stays clean."""
    y_true = _y(10)  # 5 positives
    y_view, meta = CleanLabelGenerator().generate(np.zeros((10, 2)), y_true, 0.3, seed=0)
    # every true positive survives: not a c-fraction of them (that is the defect)
    assert np.array_equal(y_view, y_true)
    assert int(np.sum(y_view == 1)) == 5
    assert meta["mechanism"] == "pn_oracle"
    assert meta["c_realized"] == 1.0
    # view declaration: PU generators stay "pu", the oracle one is "clean"
    assert SCARGenerator.output_view == "pu"
    assert CleanLabelGenerator.output_view == "clean"


def test_determ_clean_label_generator_ignores_seed_and_c():
    """Oracle results are c-independent: c only drives PU label marking."""
    X = np.zeros((10, 2))
    y_true = _y(10)
    a, _ = CleanLabelGenerator().generate(X, y_true, 0.1, seed=0)
    b, _ = CleanLabelGenerator().generate(X, y_true, 0.9, seed=123)
    assert np.array_equal(a, b)
    assert np.array_equal(a, y_true)


def test_lbe_generators_accept_4d_nchw_input():
    """CNN survey splits hand the generators 4-D NCHW tensors.

    The posterior helper is a linear model, so fit AND predict must consume
    the same flattened ``(n, -1)`` view (PU-Bench data_utils.py:417-421);
    fitting on 4-D and predicting on 4-D must not diverge.
    """
    X = np.random.RandomState(0).normal(size=(12, 1, 4, 4))
    y_true = np.array([1] * 6 + [0] * 6)
    for gen in (SARLBEAGenerator(), SARLBEBGenerator()):
        y_pu, _ = gen.generate(X, y_true, 0.5, seed=0)
        # clamping is a no-op here: min(6, max(1, round(0.5*6))) = 3
        assert int(np.sum(y_pu == 1)) == 3
        assert np.all(y_true[y_pu == 1] == 1)  # pool = positives only (S=1 ⟹ Y=1)
        again, _ = gen.generate(X, y_true, 0.5, seed=0)
        assert np.array_equal(y_pu, again)


def test_lbe_metadata_records_requested_and_actual_label_counts():
    """The protocol formula and the clamped count disagree at c·n₊ < 0.5."""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_true = np.array([1, 1, 0, 0])  # n_pos = 2
    for gen in (SARLBEAGenerator(), SARLBEBGenerator()):
        # requested = round(2*0.05) = 0, but _n_labeled clamps to one label
        _, meta = gen.generate(X, y_true, 0.05, seed=0)
        assert meta["n_labeled_requested"] == 0
        assert meta["n_labeled"] == 1
        assert meta["c_realized"] == 0.5
        assert meta["n_positive"] == 2
    # Non-boundary case: requested == realized.
    X12 = np.zeros((12, 2))
    y12 = _y(12)  # 6 positives
    for gen in (SARLBEAGenerator(), SARLBEBGenerator()):
        _, meta = gen.generate(X12, y12, 0.5, seed=0)  # round(0.5*6) = 3
        assert meta["n_labeled_requested"] == 3
        assert meta["n_labeled"] == 3


def test_scar_metadata_records_common_audit_fields():
    """SCAR and the LBE variants share one audit-field vocabulary."""
    y_true = _y(10)  # 5 positives
    _, meta = SCARGenerator().generate(np.zeros((10, 2)), y_true, 0.4, seed=3)
    assert meta["generator"] == "SCARGenerator"
    assert meta["mechanism"] == "scar"
    assert meta["c_requested"] == 0.4
    assert meta["c_realized"] == 0.4
    assert meta["n_positive"] == 5
    assert meta["n_labeled_requested"] == 2
    assert meta["n_labeled"] == 2
    assert meta["generation_seed"] == 3


def test_lbe_metadata_records_generation_seed():
    """LBE provenance: the seed that drove sampling is recorded verbatim."""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_true = np.array([1, 1, 0, 0])
    for gen, expected in (
        (SARLBEAGenerator(), "SARLBEAGenerator"),
        (SARLBEBGenerator(), "SARLBEBGenerator"),
    ):
        _, meta = gen.generate(X, y_true, 0.5, seed=5)
        assert meta["generator"] == expected
        assert meta["generation_seed"] == 5


def test_label_view_hash_is_stable_and_reflects_the_marking():
    """Protocol §2.4 item 4 audit hook: one (seed, c, mechanism) ⇒ one marking.

    The manifest exposes counts only, so two runs can agree on every printed
    number while marking different samples; the digest over the label view is
    what lets a reader verify afterwards that the methods compared under one
    (dataset, seed, c) really saw the same P/U marking.
    """
    from pu_toolbox.utils.serialization import canonical_hash

    X = np.random.RandomState(1).normal(size=(40, 2))
    y_true = np.array([1] * 20 + [0] * 20)
    for gen in (SCARGenerator(), SARLBEAGenerator(), SARLBEBGenerator()):
        y_a, meta_a = gen.generate(X, y_true, 0.5, seed=7)  # 10 labeled
        _, meta_b = gen.generate(X, y_true, 0.5, seed=7)
        assert meta_a["label_view_sha256"] == meta_b["label_view_sha256"]
        assert meta_a["label_view_sha256"] == canonical_hash({"y_pu": y_a.tolist()})
        # a different label count is a different marking by construction
        _, meta_c = gen.generate(X, y_true, 0.2, seed=7)  # 4 labeled
        assert meta_c["label_view_sha256"] != meta_a["label_view_sha256"]
