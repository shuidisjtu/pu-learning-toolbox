"""Independent joint probabilities and gradients for neural LBE's E/M steps."""

import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.deep.lbe_neural import (  # noqa: E402
    lbe_latent_positive_posterior,
    lbe_neural_objective,
    lbe_observed_log_likelihood,
)

pytestmark = [pytest.mark.unit, pytest.mark.math]


def values():
    return (
        torch.tensor([0.2, -0.8, 1.1], dtype=torch.float64),
        torch.tensor([-0.4, 0.9, 0.6], dtype=torch.float64),
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
    )


def test_basic_estep_matches_independent_joint_normalization():
    logits, propensity, observed = values()
    p, c = logits.sigmoid(), propensity.sigmoid()
    positive = p * torch.where(observed == 1, c, 1 - c)
    negative = (1 - p) * (1 - observed)
    expected = positive / (positive + negative)
    actual = lbe_latent_positive_posterior(logits, propensity, observed)
    torch.testing.assert_close(actual, expected)
    assert actual[0] == 1 and not actual.requires_grad


def test_basic_mstep_matches_complete_likelihood_and_analytic_gradients():
    logits, propensity, observed = values()
    logits.requires_grad_()
    propensity.requires_grad_()
    q = torch.tensor([1.0, 0.3, 0.7], dtype=torch.float64, requires_grad=True)
    p, c = logits.sigmoid(), propensity.sigmoid()
    expected_class = -(q.detach() * p.log() + (1 - q.detach()) * (1 - p).log()).mean()
    expected_propensity = -(
        q.detach() * (observed * c.log() + (1 - observed) * (1 - c).log())
    ).mean()
    loss, parts = lbe_neural_objective(logits, propensity, observed, q)
    torch.testing.assert_close(parts["classifier_loss"], expected_class)
    torch.testing.assert_close(parts["propensity_loss"], expected_propensity)
    torch.testing.assert_close(loss, expected_class + expected_propensity)
    loss.backward()
    torch.testing.assert_close(logits.grad, (p.detach() - q.detach()) / 3)
    torch.testing.assert_close(propensity.grad, q.detach() * (c.detach() - observed) / 3)
    assert q.grad is None


def test_basic_observed_likelihood_is_not_the_expected_complete_loss():
    logits, propensity, observed = values()
    label_probability = logits.sigmoid() * propensity.sigmoid()
    expected = (
        observed * label_probability.log() + (1 - observed) * (1 - label_probability).log()
    ).sum()
    torch.testing.assert_close(lbe_observed_log_likelihood(logits, propensity, observed), expected)


def test_edge_saturated_logits_keep_posterior_loss_and_gradients_finite():
    logits = torch.tensor([1e4, -1e4, 1e4, -1e4], requires_grad=True)
    propensity = torch.tensor([1e4, 1e4, -1e4, -1e4], requires_grad=True)
    observed = torch.tensor([0.0, 0.0, 0.0, 1.0])
    q = lbe_latent_positive_posterior(logits, propensity, observed)
    assert torch.isfinite(q).all() and torch.all((q >= 0) & (q <= 1))
    assert q[0] == pytest.approx(0.5, abs=1e-3)
    loss, _ = lbe_neural_objective(logits, propensity, observed, q)
    assert torch.isfinite(loss)
    assert torch.isfinite(lbe_observed_log_likelihood(logits, propensity, observed))
    loss.backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(propensity.grad).all()


@pytest.mark.parametrize(
    "defect", ["empty", "shape", "nan", "label", "posterior", "positive_posterior"]
)
def test_param_invalid_formula_inputs_fail_loud(defect):
    logits, propensity, observed = values()
    q = torch.tensor([1.0, 0.3, 0.7], dtype=torch.float64)
    if defect == "empty":
        logits = logits[:0]
    elif defect == "shape":
        propensity = propensity[:, None]
    elif defect == "nan":
        logits[1] = float("nan")
    elif defect == "label":
        observed[1] = -1
    elif defect == "posterior":
        q[1] = 2
    else:
        q[0] = 0.9
    with pytest.raises(ValueError):
        lbe_neural_objective(logits, propensity, observed, q)


def test_determ_equations_do_not_modify_inputs():
    tensors = values()
    before = tuple(value.clone() for value in tensors)
    first = lbe_latent_positive_posterior(*tensors)
    torch.testing.assert_close(first, lbe_latent_positive_posterior(*tensors), rtol=0, atol=0)
    for value, original in zip(tensors, before, strict=True):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
