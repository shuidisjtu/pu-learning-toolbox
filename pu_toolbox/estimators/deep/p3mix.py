"""P3Mix batch components verified against the author's ICLR 2022 slides.

This is not a registered classifier: the full paper/source training and pool
schedule still require verification. No P3Mix-E/C implementation is claimed.
"""

from __future__ import annotations

import numpy as np


def p3mix_candidate_positions(positive_probabilities, *, pool_size):
    """Top-k binary entropy among known positives; ties use stable row order."""
    from scipy.special import xlogy

    probabilities = np.asarray(positive_probabilities, dtype=float)
    if probabilities.ndim != 1 or not len(probabilities):
        raise ValueError("candidate probabilities must be nonempty and one-dimensional")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("candidate probabilities must be finite and in [0,1]")
    if type(pool_size) is not int or pool_size < 1:
        raise ValueError("pool_size must be a positive integer")
    entropy = -xlogy(probabilities, probabilities) - xlogy(1 - probabilities, 1 - probabilities)
    return np.argsort(-entropy, kind="stable")[: min(pool_size, len(probabilities))]


def p3mix_training_candidate_pool(
    features, pu_labels, predicted_probabilities, sample_indices, *, pool_size
):
    """Build a candidate pool exclusively from this training partition's P.

    Labels must be observed PU labels, not hidden PN truth. Identities are
    preserved for downstream provenance checks; no clean validation/test
    input is accepted. This does not implement the paper's pool update schedule.
    """
    features = np.asarray(features)
    labels = np.asarray(pu_labels)
    probabilities = np.asarray(predicted_probabilities, dtype=float)
    indices = np.asarray(sample_indices)
    if (
        features.ndim != 2
        or not len(features)
        or features.shape[1] < 1
        or not np.isfinite(features).all()
    ):
        raise ValueError("training features must be nonempty finite 2-D")
    if labels.shape != (len(features),) or not np.all((labels == 0) | (labels == 1)):
        raise ValueError("training pu_labels must be matched binary observed labels")
    if (
        probabilities.shape != labels.shape
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("training predictions must be matched finite [0,1]")
    if (
        indices.shape != labels.shape
        or indices.dtype.kind not in "iu"
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("training sample_indices must be matched unique integer identities")
    positives = np.flatnonzero(labels == 1)
    if not len(positives):
        raise ValueError("candidate pool requires observed labeled training positives")
    positions = positives[p3mix_candidate_positions(probabilities[positives], pool_size=pool_size)]
    return {
        "features": features[positions].copy(),
        "sample_indices": indices[positions].copy(),
        "training_positions": positions.copy(),
    }


def p3mix_heuristic_batch(
    features,
    pu_labels,
    predicted_probabilities,
    candidate_features,
    *,
    gamma=0.9,
    alpha=0.5,
    random_state=0,
):
    """Slide 11/12's heuristic partners and lambda'=max(lambda,1-lambda).

    candidate_features must come exclusively from labeled P in the training
    partition; this pure array helper cannot verify that provenance by itself.
    Marginal U has inclusive 1-gamma <= probability <= gamma; all other
    partners are drawn uniformly from this minibatch's P+U, including self.
    Returns diagnostics as well as mixed features and soft labels.
    """
    features = np.asarray(features, dtype=float)
    labels = np.asarray(pu_labels)
    probabilities = np.asarray(predicted_probabilities, dtype=float)
    candidates = np.asarray(candidate_features, dtype=float)
    if features.ndim != 2 or not len(features) or not np.isfinite(features).all():
        raise ValueError("features must be nonempty finite 2-D")
    if labels.shape != (len(features),) or not np.all((labels == 0) | (labels == 1)):
        raise ValueError("pu_labels must be matched binary observed labels")
    if (
        probabilities.shape != labels.shape
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("predicted probabilities must be matched finite [0,1]")
    if (
        candidates.ndim != 2
        or not len(candidates)
        or candidates.shape[1] != features.shape[1]
        or not np.isfinite(candidates).all()
    ):
        raise ValueError("candidate_features must be nonempty finite matching-width 2-D")
    if not np.isfinite(gamma) or not 0.5 <= gamma <= 1:
        raise ValueError("gamma must be in [0.5,1]")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    marginal = (labels == 0) & (probabilities >= 1 - gamma) & (probabilities <= gamma)
    rng = np.random.RandomState(random_state)
    partners = rng.randint(0, len(features), size=len(features))
    partner_features = features[partners].copy()
    partner_labels = labels[partners].astype(float)
    candidate_positions = np.full(len(features), -1, dtype=int)
    selected = rng.randint(0, len(candidates), size=np.count_nonzero(marginal))
    candidate_positions[marginal] = selected
    partner_features[marginal] = candidates[selected]
    partner_labels[marginal] = 1.0
    lambdas = rng.beta(alpha, alpha, size=len(features))
    lambdas = np.maximum(lambdas, 1 - lambdas)
    return {
        "features": lambdas[:, None] * features + (1 - lambdas[:, None]) * partner_features,
        "soft_labels": lambdas * labels + (1 - lambdas) * partner_labels,
        "lambdas": lambdas,
        "marginal_mask": marginal,
        "minibatch_partner_positions": np.where(marginal, -1, partners),
        "candidate_partner_positions": candidate_positions,
    }


def p3mix_batch_loss(logits, soft_labels, original_pu_labels, *, unlabeled_weight):
    """Slide 10's separate mean_P + beta*mean_U, not a pooled-size mean."""
    import torch
    from torch.nn import functional

    logits, soft_labels = logits.reshape(-1), soft_labels.reshape(-1)
    observed = original_pu_labels.reshape(-1)
    if not logits.numel() or logits.shape != soft_labels.shape or logits.shape != observed.shape:
        raise ValueError("P3Mix loss requires matched nonempty logits/labels")
    if (
        not torch.isfinite(logits).all()
        or not torch.isfinite(soft_labels).all()
        or not torch.all((soft_labels >= 0) & (soft_labels <= 1))
    ):
        raise ValueError("P3Mix logits/soft labels must be finite with labels in [0,1]")
    if (
        not torch.all((observed == 0) | (observed == 1))
        or not (observed == 1).any()
        or not (observed == 0).any()
    ):
        raise ValueError("P3Mix loss requires both observed P and U")
    if not np.isfinite(unlabeled_weight) or unlabeled_weight <= 0:
        raise ValueError("unlabeled_weight must be finite and positive")
    losses = functional.binary_cross_entropy_with_logits(
        logits, soft_labels.detach(), reduction="none"
    )
    return losses[observed == 1].mean() + unlabeled_weight * losses[observed == 0].mean()
