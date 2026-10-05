# ruff: noqa: N806
"""GenPU paper game directions and train-only marginal calibration."""

import pickle

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")
from pu_toolbox.estimators.deep.gen_pu import (  # noqa: E402
    GenPUClassifier,
    genpu_discriminator_losses,
    genpu_generator_losses,
)

pytestmark = pytest.mark.unit


def parameters():
    return dict(class_prior=0.4, positive_weight=2.0, negative_weight=3.0, unlabeled_weight=1.5)


def data():
    rng = np.random.RandomState(4)
    return rng.normal(size=(24, 3)).astype("float32"), np.r_[np.ones(8, int), np.zeros(16, int)]


def model(**kwargs):
    params = dict(
        class_prior=0.4,
        hidden_dim=4,
        latent_dim=2,
        max_epochs=1,
        classifier_epochs=1,
        batch_size=8,
        random_state=2,
        device="cpu",
    )
    params.update(kwargs)
    return GenPUClassifier(**params)


@pytest.mark.math
def test_discriminator_game_matches_probability_formula_and_prior_mixture():
    probabilities = [0.8, 0.3, 0.7, 0.2, 0.6, 0.4, 0.1]
    logits = [torch.logit(torch.tensor([p], dtype=torch.float64)) for p in probabilities]
    losses = genpu_discriminator_losses(*logits, **parameters())
    expected = [
        -0.4 * 2 * (np.log(0.8) + np.log(0.7)),
        -0.6 * 3 * (np.log(0.7) + np.log(0.8)),
        -1.5 * (np.log(0.6) + 0.4 * np.log(0.6) + 0.6 * np.log(0.9)),
    ]
    np.testing.assert_allclose([loss.item() for loss in losses], expected)


@pytest.mark.math
def test_negative_generator_is_anti_gan_and_gradient_directions_are_not_reversed():
    dp, du_p, dn, du_n = [torch.tensor([0.0], requires_grad=True) for _ in range(4)]
    positive, negative = genpu_generator_losses(dp, du_p, dn, du_n, **parameters())
    positive.backward()
    negative.backward()
    assert dp.grad.item() < 0 and du_p.grad.item() < 0
    assert dn.grad.item() > 0 and du_n.grad.item() < 0
    np.testing.assert_allclose(negative.item(), 0.6 * (1.5 - 3) * np.log(0.5), atol=1e-6)


def test_extreme_logits_stable():
    logits = torch.tensor([-1000.0, 1000.0], requires_grad=True)
    losses = genpu_discriminator_losses(*([logits] * 7), **parameters())
    losses += genpu_generator_losses(*([logits] * 4), **parameters())
    assert all(torch.isfinite(loss) for loss in losses)
    sum(losses).backward()
    assert torch.isfinite(logits.grad).all()


def test_fit_calibration_counts_both_stages_seed_clone_pickle_and_no_input_mutation():
    X, y = data()
    original = X.copy(), y.copy()
    fitted = clone(model()).fit(X, y, os_or_ts="ts")
    repeated = model().fit(X, y, os_or_ts="ts")
    np.testing.assert_array_equal(fitted.decision_function(X), repeated.decision_function(X))
    assert fitted.calibration_applied_ and fitted.n_loss_unlabeled_ == len(X)
    assert fitted.n_positive_ == 8 and fitted.n_unlabeled_ == 16
    assert fitted.optimizer_steps_ == 18  # 3 batches * (3 D + 2 G + PN)
    assert fitted.objective_variant_ == "paper_minimax_prior_weighted"
    assert all(len(v) == 1 for v in fitted.history_.values())
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(restored.predict_proba(X), fitted.predict_proba(X))
    np.testing.assert_array_equal(X, original[0])
    np.testing.assert_array_equal(y, original[1])
    assert not fitted.model_.training


def test_calibration_changes_only_du_real_pool(monkeypatch):
    X, y = data()
    seen = []
    original_choice = np.random.RandomState

    class RecordingRng:
        def __init__(self, seed):
            self.rng = original_choice(seed)

        def randint(self, *args):
            return self.rng.randint(*args)

        def normal(self, *args, **kwargs):
            return self.rng.normal(*args, **kwargs)

        def choice(self, values, size):
            seen.append(np.array(values))
            return self.rng.choice(values, size)

    monkeypatch.setattr(np.random, "RandomState", RecordingRng)
    model().fit(X, y, os_or_ts="ts")
    assert all(np.array_equal(values, np.flatnonzero(y == 1)) for values in seen[::2])
    assert all(np.array_equal(np.sort(values), np.arange(len(y))) for values in seen[1::2])


def test_basic_determ_classifier_checkpoints_preserve_ts_training_and_full_update_budget(tmp_path):
    from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer

    X, y = data()
    plain = model(max_epochs=2, classifier_epochs=3).fit(X, y, os_or_ts="ts")
    fitted = model(max_epochs=2, classifier_epochs=3)
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(fitted, X, y, os_or_ts="ts")
    assert fitted.checkpoint_stage_ == "synthetic_pn"
    assert fitted.checkpoint_epoch_count == 3
    assert [checkpoint.epoch_position for checkpoint in trajectory.checkpoints] == [1, 2, 3]
    assert len(fitted.history_["gan_generator_loss"]) == 2
    assert len(fitted.history_["pn_loss"]) == 3
    assert fitted.optimizer_steps_ == plain.optimizer_steps_ == 39
    assert fitted.calibration_applied_ and fitted.n_loss_unlabeled_ == len(X)
    np.testing.assert_array_equal(fitted.decision_function(X), plain.decision_function(X))
    np.testing.assert_array_equal(
        trajectory.checkpoints[-1].restore().decision_function(X), fitted.decision_function(X)
    )
    before = trajectory.checkpoints[0].restore().decision_function(X)
    with torch.no_grad():
        next(fitted.model_.parameters()).add_(100)
    np.testing.assert_array_equal(before, trajectory.checkpoints[0].restore().decision_function(X))


def test_edge_callback_failure_does_not_repeat_gan_training_or_mark_successful():
    X, y = data()
    fitted = model()
    seen = []

    def fail(epoch, estimator):
        seen.append((epoch, estimator.optimizer_steps_))
        raise TypeError("synthetic snapshot failure")

    with pytest.raises(TypeError, match="synthetic snapshot failure"):
        fitted.fit(X, y, os_or_ts="ts", epoch_callback=fail)
    assert seen == [(0, 18)]
    assert not fitted._is_fitted


def test_prior_weights_and_invalid_view_fail_loud():
    X, y = data()
    with pytest.raises(ValueError, match="requires population"):
        model(class_prior=None).fit(X, y)
    with pytest.raises(ValueError, match="differs"):
        model().fit(X, y, class_prior=0.5)
    with pytest.raises(NotImplementedError, match="sample_weight"):
        model().fit(X, y, sample_weight=np.ones(len(y)))
    with pytest.raises(ValueError, match="requested_view"):
        model().fit(X, y, os_or_ts="both")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"latent_dim": 0},
        {"classifier_epochs": 0},
        {"positive_weight": -1},
        {"learning_rate": np.nan},
    ],
)
def test_invalid_parameters(kwargs):
    X, y = data()
    with pytest.raises(ValueError):
        model(**kwargs).fit(X, y)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_short_cuda_pickle_recovery(tmp_path):
    from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer

    X, y = data()
    fitted = model(device="cuda")
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(fitted, X, y, os_or_ts="ts")
    assert next(fitted.positive_generator_.parameters()).is_cuda
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(restored.predict(X), fitted.predict(X))
    np.testing.assert_allclose(
        trajectory.checkpoints[-1].restore(device="cpu").decision_function(X),
        fitted.decision_function(X),
        atol=1e-5,
        rtol=1e-5,
    )
