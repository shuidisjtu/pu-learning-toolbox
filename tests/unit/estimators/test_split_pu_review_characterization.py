"""Issue 122 characterization, not acceptance of bias or a new training recipe.

These tests expose existing semantics for an owner decision. They must be
updated alongside an approved change, not used to block a corrected sampler.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox import SplitPUClassifier  # noqa: E402
from pu_toolbox.estimators.deep.split_pu import splitpu_js_loss  # noqa: E402

pytestmark = pytest.mark.unit


def bernoulli_kl(left, right):
    return left * math.log(left / right) + (1 - left) * math.log((1 - left) / (1 - right))


def weighted_js(student, teacher, teacher_weight, normalizer_weight):
    mixture = teacher_weight * teacher + (1 - teacher_weight) * student
    numerator = teacher_weight * bernoulli_kl(teacher, mixture) + (
        1 - teacher_weight
    ) * bernoulli_kl(student, mixture)
    return numerator / (-normalizer_weight * math.log(normalizer_weight))


@pytest.mark.parametrize("seed", [0, 9])
@pytest.mark.parametrize("view", ["os", "ts"])
def test_basic_determ_teacher_batch_union_does_not_restore_population_mixture(
    monkeypatch, seed, view
):
    import pu_toolbox.estimators.deep.split_pu as module

    # Synthetic IDs only; true classes below are assertion-side metadata and
    # are never provided to fit as labels. Population: 40 positives / 100;
    # observed labels: 8 positives, 92 unlabeled (32 hidden positives).
    features = np.random.RandomState(2).normal(size=(100, 3)).astype(np.float32)
    features[:, 0] = np.arange(100) / 100
    labels = np.r_[np.ones(8, int), np.zeros(92, int)]
    score_inputs, batches = [], []
    score, risk = module._score, module._nnpu_train_step

    def capture_score(network, inputs):
        score_inputs.append(np.rint(inputs[:, 0].detach().cpu().numpy() * 100).astype(int))
        return score(network, inputs)

    def capture_risk(*args, **kwargs):
        p_ids, u_ids = score_inputs[-2:]
        assert np.all(p_ids < 8) and np.all(u_ids >= 8)
        assert kwargs["class_prior"] == 0.4
        batches.append((p_ids.copy(), u_ids.copy()))
        return risk(*args, **kwargs)

    monkeypatch.setattr(module, "_score", capture_score)
    monkeypatch.setattr(module, "_nnpu_train_step", capture_risk)
    estimator = SplitPUClassifier(
        class_prior=0.4,
        hidden_dim=4,
        teacher_epochs=1,
        split_epochs=1,
        student_epochs=1,
        rounds=1,
        batch_size=4,
        random_state=seed,
        device="cpu",
    ).fit(features, labels, os_or_ts=view)
    assert len(batches) == 23
    assert all(len(p) == len(u) == 4 for p, u in batches)
    all_u = np.concatenate([u for _, u in batches])
    np.testing.assert_array_equal(np.sort(all_u), np.arange(8, 100))
    observed = np.concatenate([np.r_[u, p] if view == "ts" else u for p, u in batches])
    observed_positive_fraction = np.mean(observed < 40)
    assert np.mean(np.arange(100) < 40) == 0.4  # full-set union is the population
    if view == "ts":
        assert observed_positive_fraction == pytest.approx((92 + 32) / (92 * 2))
        assert observed_positive_fraction - 0.4 > 0.27
    else:
        assert observed_positive_fraction == pytest.approx(32 / 92)
    assert estimator.calibration_applied_ is (view == "ts")


@pytest.mark.parametrize("student,teacher", [(0.4, 0.9), (0.2, 0.7)])
def test_basic_source_direction_paper_direction_and_simple_weight_flip_are_three_objectives(
    student, teacher
):
    s = torch.tensor([math.log(student / (1 - student))], dtype=torch.float64)
    t = torch.tensor([math.log(teacher / (1 - teacher))], dtype=torch.float64)
    source = float(splitpu_js_loss(s, t, teacher_weight=0.7))
    flipped = float(splitpu_js_loss(s, t, teacher_weight=0.3))
    paper_as_printed = weighted_js(student, teacher, 0.3, 0.3)
    assert source == pytest.approx(weighted_js(student, teacher, 0.7, 0.3))
    assert flipped == pytest.approx(weighted_js(student, teacher, 0.3, 0.7))
    assert not math.isclose(source, paper_as_printed, rel_tol=1e-3)
    assert not math.isclose(flipped, paper_as_printed, rel_tol=1e-3)
    assert flipped / paper_as_printed == pytest.approx(
        (0.3 * math.log(0.3)) / (0.7 * math.log(0.7))
    )


def test_edge_equal_predictions_hide_direction_and_normalization_difference():
    raw = torch.tensor([0.4], dtype=torch.float64)
    for weight in (0.3, 0.7):
        assert float(splitpu_js_loss(raw, raw, teacher_weight=weight)) == pytest.approx(
            0, abs=1e-15
        )


@pytest.mark.parametrize("weight", [0, 1, -0.1, float("nan")])
def test_param_invalid_js_weight_is_rejected_without_resolving_owner_choice(weight):
    with pytest.raises(ValueError, match="teacher_weight"):
        splitpu_js_loss(torch.tensor([0.0]), torch.tensor([1.0]), teacher_weight=weight)
