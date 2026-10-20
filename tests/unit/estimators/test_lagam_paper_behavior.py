"""Independent witnesses for the documented LaGAM paper/adaptation boundaries."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.deep import lagam  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.math]


@pytest.mark.parametrize("positive_count", [1, 3])
def test_basic_balanced_class_loss_is_not_paper_per_example_mean(positive_count):
    logits = np.array([-0.7, 0.3, 1.1, -1.4], dtype=np.float64)
    labels = np.zeros(4)
    labels[:positive_count] = 1
    entries = labels * np.logaddexp(0, -logits) + (1 - labels) * np.logaddexp(0, logits)
    expected = entries[labels == 1].mean() + entries[labels == 0].mean()
    actual = lagam._balanced_soft_bce(torch.tensor(logits), torch.tensor(labels))
    assert actual.item() == pytest.approx(expected)
    assert not np.isclose(actual.item(), entries.mean())


def test_small_soft_mass_uses_explicit_toolbox_guard_not_source_denominator():
    logits = np.array([0.2, -1.1, 0.8])
    targets = np.array([0.10, 0.15, 0.0])
    positive = (targets * np.logaddexp(0, -logits)).sum()
    negative = ((1 - targets) * np.logaddexp(0, logits)).sum()
    expected = positive + negative / (1 - targets).sum()
    source_without_guard = positive / targets.sum() + negative / (1 - targets).sum()
    actual = lagam._balanced_soft_bce(torch.tensor(logits), torch.tensor(targets))
    assert actual.item() == pytest.approx(expected)
    assert not np.isclose(actual.item(), source_without_guard)


@pytest.mark.parametrize("update_bias", [False, True])
def test_exact_zero_gradient_distinguishes_main_equation_and_supplement(update_bias):
    model = torch.nn.Sequential(torch.nn.Linear(1, 1)).double()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()
    features = torch.zeros(4, 1, dtype=torch.float64)
    observed = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    # At zero features the weight step cannot affect support logits; balanced
    # support at zero logits also has exactly zero outer bias gradient.
    support_features = torch.zeros(2, 1, dtype=torch.float64)
    support_labels = torch.tensor([1.0, 0.0], dtype=torch.float64)
    actual = lagam.lagam_meta_labels(
        model,
        features,
        observed,
        support_features,
        support_labels,
        meta_lr=0.1,
        update_bias=update_bias,
    )
    np.testing.assert_array_equal(actual.numpy(), observed.numpy())
    zero_direction = np.zeros(2)
    main_equation_u = (zero_direction >= 0).astype(float)
    supplement_u = (zero_direction > 0).astype(float)
    np.testing.assert_array_equal(actual.numpy()[2:], supplement_u)
    assert np.all(actual.numpy()[2:] != main_equation_u)


def _contrastive_reference(q, k, groups=None, predictions=None):
    """NumPy Eq(2) with instance pairs or an explicit positive-pair mask."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    k = k / np.linalg.norm(k, axis=1, keepdims=True)
    n = len(q)
    logits = q @ np.concatenate([q, k]).T / 0.3
    logits[np.arange(n), np.arange(n)] = -np.inf
    maximum = logits.max(axis=1, keepdims=True)
    log_prob = logits - maximum - np.log(np.exp(logits - maximum).sum(axis=1, keepdims=True))
    instance = -log_prob[np.arange(n), n + np.arange(n)].mean()
    if groups is None:
        return instance
    mask = groups[:, None] == np.concatenate([groups, groups])[None, :]
    if predictions is not None:
        mask &= predictions[:, None] == np.concatenate([predictions, predictions])[None, :]
    mask[np.arange(n), np.arange(n)] = False
    group = -(np.where(mask, log_prob, 0).sum(axis=1) / mask.sum(axis=1)).mean()
    return instance + group


@pytest.mark.parametrize("use_groups", [False, True])
def test_contrastive_matches_same_cluster_reference_not_binary_cutoff(use_groups):
    q = np.array([[1.0, 0.2], [0.1, 1.0], [-0.7, 0.8], [0.6, -0.8]])
    k = np.array([[0.8, 0.1], [0.2, 0.9], [-0.5, 0.9], [0.7, -0.5]])
    groups = np.array([0, 0, 1, 1]) if use_groups else None
    actual = lagam.lagam_contrastive_loss(
        torch.tensor(q),
        torch.tensor(k),
        None if groups is None else torch.tensor(groups),
        temperature=0.3,
    )
    assert actual.item() == pytest.approx(_contrastive_reference(q, k, groups))
    if use_groups:
        cutoff = _contrastive_reference(q, k, groups, predictions=np.array([0, 1, 0, 1]))
        assert not np.isclose(actual.item(), cutoff)


@pytest.mark.parametrize("temperature", [0, float("nan"), float("inf")])
def test_param_invalid_contrastive_temperature_rejected(temperature):
    with pytest.raises(ValueError, match="temperature must be finite and positive"):
        lagam.lagam_contrastive_loss(torch.ones(2, 2), torch.ones(2, 2), temperature=temperature)


def test_determ_seeded_meta_steps_use_hard_observed_not_previous_ema(monkeypatch):
    observed = np.array([1, 1, 0, 0])
    received = []

    def detected_positive(model, features, labels, support_features, support_labels, **kwargs):
        received.append(labels.detach().cpu().numpy().copy())
        return torch.ones_like(labels)

    monkeypatch.setattr(lagam, "lagam_meta_labels", detected_positive)
    fitted = lagam.LaGAMClassifier(
        hidden_dim=3,
        warmup_epochs=0,
        max_epochs=2,
        batch_size=4,
        mix_weight=0,
        contrastive_weight=0,
        num_clusters=2,
        rho_start=0.5,
        rho_end=0.5,
        random_state=7,
        device="cpu",
    ).fit(
        np.array([[1, 2], [2, 1], [-1, -2], [-2, -1]], dtype=np.float32),
        observed,
        support_data=(np.array([[3, 2], [-3, -2]], dtype=np.float32), np.array([1, 0])),
    )
    assert len(received) == 2
    for labels in received:
        np.testing.assert_array_equal(np.sort(labels), np.sort(observed))
    np.testing.assert_array_equal(fitted.pseudo_labels_.numpy(), [1.0, 1.0, 0.75, 0.75])
    assert fitted.virtual_steps_ == fitted.optimizer_steps_ == 2
    repeated = lagam.LaGAMClassifier(**fitted.get_params()).fit(
        np.array([[1, 2], [2, 1], [-1, -2], [-2, -1]], dtype=np.float32),
        observed,
        support_data=(np.array([[3, 2], [-3, -2]], dtype=np.float32), np.array([1, 0])),
    )
    assert len(received) == 4
    for first, second in zip(received[:2], received[2:], strict=True):
        np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(fitted.pseudo_labels_.numpy(), repeated.pseudo_labels_.numpy())
    for name, value in fitted.model_.state_dict().items():
        torch.testing.assert_close(value, repeated.model_.state_dict()[name], rtol=0, atol=0)
