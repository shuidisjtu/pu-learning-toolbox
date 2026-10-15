"""Independent finite differences and two-column virtual-label equivalence."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.deep.lagam import lagam_meta_labels  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.math]


def fixture():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1)).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[0.3, -0.6]], dtype=torch.float64))
        model[0].bias.fill_(0.7)
    x = torch.tensor([[1.0, 2.0], [-1.0, 1.0], [0.2, -1.0], [-2.0, -0.3]], dtype=torch.float64)
    y = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    support_x = torch.tensor(
        [[1.2, -0.1], [0.6, -0.3], [-0.5, 0.8], [-1.0, 1.0]], dtype=torch.float64
    )
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    return model, x, y, support_x, labels


def finite_support_loss(perturbed, x, support_x, labels, update_bias):
    w, b = np.array([0.3, -0.6]), 0.7
    sigmoid = 1 / (1 + np.exp(-(x @ w + b)))
    coefficient = (
        -(1 - sigmoid) * perturbed / perturbed.sum()
        + sigmoid * (1 - perturbed) / (1 - perturbed).sum()
    )
    w = w - 0.1 * (coefficient[:, None] * x).sum(0)
    if update_bias:
        b -= 0.1 * coefficient.sum()
    logits = support_x @ w + b
    return (labels * np.logaddexp(0, -logits)).sum() / labels.sum() + (
        (1 - labels) * np.logaddexp(0, logits)
    ).sum() / (1 - labels).sum()


@pytest.mark.parametrize("update_bias", [False, True])
def test_basic_virtual_label_directions_match_independent_finite_difference(update_bias):
    model, x, y, support_x, labels = fixture()
    gradient = []
    for i in range(len(y)):
        high, low = y.numpy().copy(), y.numpy().copy()
        high[i] += 1e-5
        low[i] -= 1e-5
        gradient.append(
            (
                finite_support_loss(high, x.numpy(), support_x.numpy(), labels.numpy(), update_bias)
                - finite_support_loss(
                    low, x.numpy(), support_x.numpy(), labels.numpy(), update_bias
                )
            )
            / 2e-5
        )
    assert np.min(np.abs(gradient)) > 1e-6
    expected = np.where(y.numpy() == 1, 1, np.array(gradient) < 0)
    actual = lagam_meta_labels(model, x, y, support_x, labels, meta_lr=0.1, update_bias=update_bias)
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("update_bias", [False, True])
def test_determ_scalar_perturbation_equals_two_independent_columns_in_stable_loss(update_bias):
    model, x, y, support_x, labels = fixture()
    epsilon = torch.zeros((4, 2), dtype=torch.float64, requires_grad=True)
    target = torch.stack((1 - y, y), dim=1) + epsilon
    w, b = model[0].weight, model[0].bias
    logits = (x @ w.T + b).flatten()
    entries = torch.stack(
        (torch.nn.functional.softplus(logits), torch.nn.functional.softplus(-logits)), dim=1
    )
    loss = ((target * entries).sum(0) / target.sum(0)).sum()
    grads = torch.autograd.grad(loss, (w, b) if update_bias else (w,), create_graph=True)
    updated_w = w - 0.1 * grads[0]
    updated_b = b - 0.1 * grads[1] if update_bias else b
    support_logits = (support_x @ updated_w.T + updated_b).flatten()
    support = (torch.nn.functional.softplus(-support_logits) * labels).sum() / labels.sum() + (
        torch.nn.functional.softplus(support_logits) * (1 - labels)
    ).sum() / (1 - labels).sum()
    g = torch.autograd.grad(support, epsilon)[0]
    expected = (-g).argmax(1)
    expected[y == 1] = 1
    actual = lagam_meta_labels(model, x, y, support_x, labels, meta_lr=0.1, update_bias=update_bias)
    torch.testing.assert_close(actual, expected.float())


@pytest.mark.parametrize("bad", [0, 1, None, "false"])
def test_param_non_boolean_virtual_scope_rejected(bad):
    model, x, y, support_x, labels = fixture()
    with pytest.raises(ValueError, match="boolean"):
        lagam_meta_labels(model, x, y, support_x, labels, meta_lr=0.1, update_bias=bad)


def test_edge_tied_virtual_gradient_keeps_known_positive_and_chooses_negative():
    model = torch.nn.Sequential(torch.nn.Linear(1, 1)).double()
    zeros = torch.zeros(4, 1, dtype=torch.float64)
    y = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    actual = lagam_meta_labels(model, zeros, y, zeros, y, meta_lr=0.1, update_bias=False)
    torch.testing.assert_close(actual, y.float())
