"""Independent scalar source rules, not upstream execution or paper-number replay."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.deep.grad_pu import gradpu_objective  # noqa: E402
from pu_toolbox.estimators.deep.robust_pu import self_paced_weights  # noqa: E402
from pu_toolbox.estimators.deep.split_pu import splitpu_js_loss  # noqa: E402
from pu_toolbox.estimators.risk.puet import _nnpu_quadratic_node_risk  # noqa: E402
from pu_toolbox.estimators.risk.pulda import (  # noqa: E402
    _PULDAEMA,
    pulda_distribution_alignment,
    pulda_two_way_margin,
)

pytestmark = [pytest.mark.unit, pytest.mark.math]


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


def scalar_pulda(values, previous):
    positive, unlabeled = values[:2], values[2:]
    moments = [
        np.mean([sigmoid(v) for v in unlabeled]),
        np.mean([sigmoid(v + 0.6) * sigmoid(-v) for v in positive]),
        np.mean([sigmoid(v + 0.6) * sigmoid(-v) for v in unlabeled]),
    ]
    if previous is not None:
        moments = [
            a * old + (1 - a) * cur
            for a, old, cur in zip((0.85, 0.5, 0.5), previous, moments, strict=True)
        ]

    def distance(left, right, temperature):
        diff = temperature * (left - right)
        return (math.log1p(math.exp(diff)) + math.log1p(math.exp(-diff))) / temperature

    lda = 0.6 * np.mean([1 - sigmoid(v) for v in positive])
    lda += distance(moments[0], 0.3, 3.5) / (1 if previous is None else 0.15)
    margin = 0.3 * np.mean([sigmoid(v) * sigmoid(0.6 - v) for v in positive])
    margin += distance(moments[2], 0.3 * moments[1], 1) / (1 if previous is None else 0.5)
    return lda, margin, moments


def test_basic_pulda_ema_three_batches_matches_scalar_source_and_gradient():
    ema, previous, batches = _PULDAEMA(0.85, 0.5), None, []
    labels = torch.tensor([1, 1, 0, 0])
    for values in ([0.4, -0.2, 0.7, -0.8], [1, -1, 0.2, 0.4], [-0.1, 0.3, -0.7, 1.2]):
        logits = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        batches.append(logits)
        old = previous
        expected_lda, expected_margin, previous = scalar_pulda(values, old)
        unlabeled, positive_negative, unlabeled_negative, u_correction, m_correction = ema.moments(
            logits, labels, 0.6
        )
        lda = pulda_distribution_alignment(
            logits,
            labels,
            class_prior=0.3,
            temperature=3.5,
            unlabeled_expectation=unlabeled,
            ema_correction=u_correction,
        )[0]
        margin = pulda_two_way_margin(
            logits,
            labels,
            class_prior=0.3,
            margin=0.6,
            temperature=1,
            positive_negative_expectation=positive_negative,
            unlabeled_negative_expectation=unlabeled_negative,
            ema_correction=m_correction,
        )[0]
        assert lda.item() == pytest.approx(expected_lda, abs=1e-12)
        assert margin.item() == pytest.approx(expected_margin, abs=1e-12)
    (lda + margin).backward()
    assert batches[0].grad is None and batches[1].grad is None
    for index in range(4):
        plus, minus = np.array(values, float), np.array(values, float)
        plus[index] += 1e-5
        minus[index] -= 1e-5
        derivative = (sum(scalar_pulda(plus, old)[:2]) - sum(scalar_pulda(minus, old)[:2])) / 2e-5
        assert logits.grad[index].item() == pytest.approx(derivative, abs=1e-8)


@pytest.mark.parametrize(
    "counts", [(2, 4, 0.1, 0.125), (6, 1, 0.1, 0.125), (0, 4, 0.1, 0.125), (2, 0, 0.1, 0.125)]
)
def test_param_puet_node_risk_matches_scalar_closed_form(counts):
    positive, unlabeled, positive_weight, unlabeled_weight = counts
    mass_p, mass_u = positive * positive_weight, unlabeled * unlabeled_weight
    expected = 0 if mass_u <= mass_p else 4 * mass_u * (mass_p / mass_u) * (1 - mass_p / mass_u)
    assert _nnpu_quadratic_node_risk(*counts) == pytest.approx(expected)


def test_basic_gradpu_raw_gradient_penalty_matches_scalar_and_second_derivative():
    raw_p = torch.tensor([-0.6, 0.8], dtype=torch.float64, requires_grad=True)
    raw_u = torch.tensor([-0.2, 0.5], dtype=torch.float64, requires_grad=True)
    inputs = torch.tensor([[0.1, 0.2], [-0.4, 0.7]], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([2.0, -3.0], dtype=torch.float64, requires_grad=True)
    total, parts = gradpu_objective(
        raw_p,
        raw_u,
        beta=0.4,
        alpha=0.2,
        interpolated_inputs=inputs,
        raw_interpolated=inputs @ weight,
    )
    expected_p = np.mean(
        [(1 - 0.4 * math.log((1 + math.tanh(v)) / 2)) * (1 - math.tanh(v)) for v in (-0.6, 0.8)]
    )
    expected_u = np.mean([1 + math.tanh(v) for v in (-0.2, 0.5)])
    assert total.item() == pytest.approx(expected_p + expected_u + 0.2 * 13)
    assert parts["gradient_penalty"].item() == pytest.approx(13)
    total.backward()
    torch.testing.assert_close(weight.grad, 0.4 * weight.detach())


@pytest.mark.parametrize("kind", ["hard", "linear", "welsch"])
def test_param_robust_spl_rules_match_scalar_boundaries_and_detach(kind):
    values = [0, 0.5, 1, 2]
    losses = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    rules = {
        "hard": lambda v: float(v < 1),
        "linear": lambda v: max(0, 1 - v),
        "welsch": lambda v: math.exp(-v),
    }
    actual = self_paced_weights(losses, 1, kind)
    np.testing.assert_allclose(actual.numpy(), [rules[kind](v) for v in values], atol=1e-12)
    assert not actual.requires_grad


@pytest.mark.parametrize("values", [([0, 0], [0, 0]), ([-0.7, 1.2], [0.5, -0.3])])
def test_basic_splitpu_js_matches_scalar_bernoulli_source_and_detaches_teacher(values):
    student_values, teacher_values = values
    student = torch.tensor(student_values, dtype=torch.float64, requires_grad=True)
    teacher = torch.tensor(teacher_values, dtype=torch.float64, requires_grad=True)
    expected = []
    for student_value, teacher_value in zip(student_values, teacher_values, strict=True):
        p, q = sigmoid(student_value), sigmoid(teacher_value)
        mixture = 0.7 * q + 0.3 * p
        kl_p = p * math.log(p / mixture) + (1 - p) * math.log((1 - p) / (1 - mixture))
        kl_q = q * math.log(q / mixture) + (1 - q) * math.log((1 - q) / (1 - mixture))
        expected.append(-(0.7 * kl_q + 0.3 * kl_p) / (0.3 * math.log(0.3)))
    actual = splitpu_js_loss(student, teacher)
    assert actual.item() == pytest.approx(np.mean(expected), abs=1e-12)
    actual.backward()
    assert teacher.grad is None and bool(torch.isfinite(student.grad).all())


def test_edge_invalid_source_component_inputs_fail_without_silent_fallback():
    with pytest.raises(ValueError, match="threshold"):
        self_paced_weights(torch.tensor([0.0]), 0)
    with pytest.raises(ValueError, match="teacher_weight"):
        splitpu_js_loss(torch.tensor([0.0]), torch.tensor([0.0]), teacher_weight=1)
    with pytest.raises(ValueError, match="batches"):
        pulda_distribution_alignment(
            torch.zeros(2), torch.ones(2), class_prior=0.3, temperature=3.5
        )


def test_determ_zero_difference_js_repeats_without_consuming_randomness():
    before = torch.random.get_rng_state().clone()
    logits = torch.tensor([0.0, 0.8], dtype=torch.float64)
    first, second = splitpu_js_loss(logits, logits), splitpu_js_loss(logits, logits)
    assert abs(first.item()) < 1e-12 and first.item() == second.item()
    assert torch.equal(before, torch.random.get_rng_state())
