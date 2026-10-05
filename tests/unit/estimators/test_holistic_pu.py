# ruff: noqa: N806
"""Paper/code variant differences are explicit and independently testable."""

import pickle

import numpy as np
import pytest

from pu_toolbox.estimators.deep.holistic_pu import (
    HolisticPUClassifier,
    holistic_natural_break,
    holistic_trend_scores,
)

pytestmark = pytest.mark.unit


@pytest.mark.math
def test_pairwise_matches_independent_scalar_equation_and_score_orientation():
    values = np.array([[0.1, 0.4, 0.8], [0.8, 0.4, 0.1], [0.3, 0.3, 0.3]])
    expected = []
    for sequence in values:
        pairs = [2 * (sequence[j] - sequence[i]) for i in range(2) for j in range(i + 1, 3)]
        expected.append(np.mean([np.sign(d) * np.log(1 + abs(d) + d * d / 2) for d in pairs]))
    actual = holistic_trend_scores(values)
    np.testing.assert_allclose(actual, expected)
    assert actual[0] > 0 and actual[1] < 0 and actual[2] == 0
    np.testing.assert_allclose(actual[0], -actual[1])


def test_author_adjacent_is_not_silently_called_paper_signed_score():
    sequence = np.array([[0.8, 0.3]])
    author = holistic_trend_scores(sequence, variant="author_adjacent", scale=1)
    paper = holistic_trend_scores(sequence, scale=1)
    np.testing.assert_allclose(author, np.log(0.625))
    np.testing.assert_allclose(paper, -np.log(1.625))
    assert not np.allclose(author, paper)
    with pytest.raises(ValueError, match="scale=1.0"):
        holistic_trend_scores(sequence, variant="author_adjacent")


@pytest.mark.parametrize("objective", ["paper_variance", "author_sse"])
def test_natural_break_matches_brute_force_unique_splits(objective):
    scores = np.array([0.2, -0.6, 0.3, -0.5, 0.2, -0.7, 0.9])
    candidates = []
    for cutoff in np.unique(scores)[:-1]:
        low, high = scores[scores <= cutoff], scores[scores > cutoff]
        sse_low, sse_high = sum((low - low.mean()) ** 2), sum((high - high.mean()) ** 2)
        cost = (
            sse_low / len(low) + sse_high / len(high)
            if objective == "paper_variance"
            else sse_low + sse_high
        )
        candidates.append((cost, cutoff))
    expected_cost, expected_cutoff = min(candidates)
    cutoff, cost, mask = holistic_natural_break(scores, objective=objective)
    np.testing.assert_allclose(cost, expected_cost, atol=1e-12)
    assert cutoff == expected_cutoff
    np.testing.assert_array_equal(mask, scores > cutoff)
    assert all(np.unique(mask[scores == tied]).size == 1 for tied in np.unique(scores))


def test_edge_constant_and_invalid_input_fail_instead_of_fabricating_labels():
    for values in ([0.1, 0.1], [np.nan, 0.1], [0.1], [[0.1, 0.2]]):
        with pytest.raises(ValueError):
            holistic_natural_break(values)
    for values in ([[0.1]], [[-1, 0]], [[0, np.inf]], []):
        with pytest.raises(ValueError):
            holistic_trend_scores(values)


def test_large_sample_partition_does_not_require_quadratic_matrix():
    scores = np.r_[np.linspace(-0.8, -0.7, 20_000), np.linspace(0.7, 0.8, 20_000)]
    cutoff, _, high = holistic_natural_break(scores)
    assert cutoff == scores[19_999]
    assert np.count_nonzero(high) == 20_000


def fixture_data():
    rng = np.random.RandomState(5)
    X = rng.normal(size=(30, 3)).astype("float32")
    X[:10] += 1
    return X, np.r_[np.ones(10, int), np.zeros(20, int)]


def small_model(**kwargs):
    params = dict(
        hidden_dim=4, warmup_epochs=3, max_epochs=1, batch_size=8, random_state=1, device="cpu"
    )
    params.update(kwargs)
    return HolisticPUClassifier(**params)


def test_fit_seed_clone_pickle_trajectory_and_prior_unused():
    pytest.importorskip("torch")
    from sklearn.base import clone

    X, y = fixture_data()
    original = X.copy(), y.copy()
    fitted = clone(small_model()).fit(X, y)
    repeated = small_model().fit(X, y, class_prior=0.4)
    assert fitted.prediction_trajectory_.shape == (20, 3)
    np.testing.assert_array_equal(fitted.pseudo_label_indices_, np.flatnonzero(y == 0))
    np.testing.assert_array_equal(
        fitted.pseudo_labels_, (fitted.trend_scores_ > fitted.breakpoint_).astype(int)
    )
    assert 0 < fitted.estimated_unlabeled_prior_ < 1
    assert fitted.optimizer_steps_ == 13  # 3 warmup * 3 batches + 4 final batches
    assert fitted.stopping_rule_ == "fixed_warmup_budget_not_LZO"
    assert not fitted.calibration_applied_ and not fitted.model_.training
    np.testing.assert_array_equal(fitted.predict_proba(X), repeated.predict_proba(X))
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(fitted.decision_function(X), restored.decision_function(X))
    np.testing.assert_array_equal(X, original[0])
    np.testing.assert_array_equal(y, original[1])


def test_balanced_warmup_uses_equal_p_u_batch_sizes(monkeypatch):
    torch = pytest.importorskip("torch")
    X, y = fixture_data()
    original = torch.nn.functional.softplus
    sizes = []

    def capture(value, *args, **kwargs):
        sizes.append(len(value))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "softplus", capture)
    small_model().fit(X, y)
    assert sizes == [8, 8, 8, 8, 4, 4] * 3


def test_basic_epoch_checkpoints_cover_both_stages_without_changing_seed(tmp_path):
    torch = pytest.importorskip("torch")
    from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer

    X, y = fixture_data()
    plain = small_model().fit(X, y)
    model = small_model()
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(model, X, y)
    assert model.checkpoint_epoch_count == 4
    assert [checkpoint.epoch_position for checkpoint in trajectory.checkpoints] == [1, 2, 3, 4]
    assert model.history_["epoch"] == [0, 1, 2, 3]
    assert model.history_["phase"] == ["warmup"] * 3 + ["pseudo_pn"]
    assert len(model.history_["train_loss"]) == 4
    assert model.optimizer_steps_ == plain.optimizer_steps_
    np.testing.assert_array_equal(model.decision_function(X), plain.decision_function(X))
    np.testing.assert_array_equal(
        trajectory.checkpoints[-1].restore().decision_function(X), model.decision_function(X)
    )
    before = trajectory.checkpoints[0].restore().decision_function(X)
    assert np.isfinite(before).all()
    with torch.no_grad():
        next(model.model_.parameters()).add_(100)
    np.testing.assert_array_equal(before, trajectory.checkpoints[0].restore().decision_function(X))


def test_edge_callback_failure_does_not_mark_fit_successful():
    pytest.importorskip("torch")
    X, y = fixture_data()
    model = small_model()
    seen = []

    def fail(epoch, fitted):
        seen.append((epoch, fitted.history_["phase"][-1]))
        raise TypeError("synthetic callback failure")

    with pytest.raises(TypeError, match="synthetic callback failure"):
        model.fit(X, y, epoch_callback=fail)
    assert seen == [(0, "warmup")]
    assert not model._is_fitted


def test_constant_trajectory_failure_leaves_estimator_unfitted(monkeypatch):
    pytest.importorskip("torch")
    X, y = fixture_data()
    monkeypatch.setattr(HolisticPUClassifier, "_scores", lambda self, rows: np.zeros(len(rows)))
    estimator = small_model()
    with pytest.raises(ValueError, match="constant trend"):
        estimator.fit(X, y)
    assert not estimator._is_fitted


@pytest.mark.parametrize(
    "kwargs",
    [{"warmup_epochs": 1}, {"max_epochs": 0}, {"trend_scale": 0}, {"learning_rate": np.nan}],
)
def test_invalid_training_parameters(kwargs):
    pytest.importorskip("torch")
    X, y = fixture_data()
    with pytest.raises(ValueError):
        small_model(**kwargs).fit(X, y)


def test_ts_and_external_weights_fail_loud():
    pytest.importorskip("torch")
    X, y = fixture_data()
    with pytest.raises(ValueError, match="ts risk substitution"):
        small_model().fit(X, y, os_or_ts="ts")
    with pytest.raises(NotImplementedError, match="sample_weight"):
        small_model().fit(X, y, sample_weight=np.ones(len(y)))


@pytest.mark.gpu
def test_short_cuda_pickle_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer

    X, y = fixture_data()
    fitted = small_model(device="cuda")
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(fitted, X, y)
    assert next(fitted.model_.parameters()).is_cuda
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(fitted.predict(X), restored.predict(X))
    np.testing.assert_allclose(
        trajectory.checkpoints[-1].restore(device="cpu").decision_function(X),
        fitted.decision_function(X),
        atol=1e-5,
        rtol=1e-5,
    )
