"""Synthetic data per contract §2: SCAR main grid, linear SAR line, PNU, h bookkeeping."""

# Dataset matrices follow sklearn's conventional X/y names.
# ruff: noqa: N803, N806

from __future__ import annotations

import numpy as np

from pu_toolbox.preprocessing.selection_bias import make_sar_dataset

# Protocol P:N:U ratios (contract §2.2); absolute counts per scenario come
# from ``pnu_counts`` so the small/mid scale actually changes the data.
PNU_RATIOS: dict[str, tuple[int, int, int]] = {
    "1:1:4": (1, 1, 4),
    "1:2:4": (1, 2, 4),
    "1:1:8": (1, 1, 8),
}


def pnu_counts(ratio_label: str, n_samples: int) -> tuple[int, int, int]:
    """Derive exact P:N:U counts from a protocol ratio and a scenario size.

    The old implementation used hard-coded 1:1:4/1:2:4/1:1:8 counts for
    every cell, so the small (400) and mid (2000) scales produced identical
    data.  Counts are proportional to ``n_samples`` with the largest-fraction
    remainder distributed to the closest group; every group gets at least
    one sample and the counts sum to at most ``n_samples`` (contract
    §2.1/§2.2).
    """
    if ratio_label not in PNU_RATIOS:
        raise ValueError(f"unknown PNU ratio: {ratio_label!r}")
    r_p, r_n, r_u = PNU_RATIOS[ratio_label]
    total = r_p + r_n + r_u
    share = n_samples / total
    n_p = int(share * r_p)
    n_n = int(share * r_n)
    n_u = int(share * r_u)
    remainder = n_samples - (n_p + n_n + n_u)
    fracs = {
        "p": (share * r_p) - n_p,
        "n": (share * r_n) - n_n,
        "u": (share * r_u) - n_u,
    }
    for group in sorted(fracs, key=fracs.get, reverse=True)[:remainder]:
        if group == "p":
            n_p += 1
        elif group == "n":
            n_n += 1
        else:
            n_u += 1
    # P and N need at least one labeled sample; U (mixed positives/negatives)
    # may shrink to one under extreme tiny sizes but never to zero.
    return max(n_p, 1), max(n_n, 1), max(n_u, 1)


def is_ill_conditioned(class_prior: float, flip_probability: float, tol: float = 1e-6) -> bool:
    """Contract §2.3: |1 - 2πh| near zero ⇒ LDCE problem is ill-conditioned."""
    return abs(1.0 - 2.0 * class_prior * flip_probability) < tol


def _make_binary_data(
    *,
    n_samples: int,
    n_features: int,
    class_prior: float,
    separation: float,
    label_frequency: float,
    mechanism: str,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | bool | str]]:
    X, y_pu, y_true, _ = make_sar_dataset(
        n_samples=n_samples,
        n_features=n_features,
        class_prior=class_prior,
        separation=separation,
        mechanism=mechanism,
        label_frequency=label_frequency,
        random_state=random_state,
    )
    real_h = 1.0 - label_frequency  # LDCE semantics: P(flip a true positive to unlabeled)
    return (
        X,
        y_pu,
        y_true,
        {
            "mechanism": mechanism,
            "class_prior": float(class_prior),
            "label_frequency": float(label_frequency),
            "real_h": float(real_h),
            "pi_h_well_conditioned": bool(not is_ill_conditioned(class_prior, real_h)),
        },
    )


def make_scar_data(
    n_samples: int,
    n_features: int,
    class_prior: float,
    separation: float,
    label_frequency: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | bool | str]]:
    """SCAR binary PU dataset; y_true is hidden except for final evaluation."""
    return _make_binary_data(
        n_samples=n_samples,
        n_features=n_features,
        class_prior=class_prior,
        separation=separation,
        label_frequency=label_frequency,
        mechanism="scar",
        random_state=random_state,
    )


def make_sar_linear_data(
    n_samples: int,
    n_features: int,
    class_prior: float,
    separation: float,
    label_frequency: float,
    strength: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | bool | str]]:
    """Linear SAR robustness line; diagnostic only, never main ranking."""
    X, y_pu, y_true, _ = make_sar_dataset(
        n_samples=n_samples,
        n_features=n_features,
        class_prior=class_prior,
        separation=separation,
        mechanism="linear",
        label_frequency=label_frequency,
        strength=strength,
        random_state=random_state,
    )
    meta = {
        "mechanism": "linear",
        "class_prior": float(class_prior),
        "label_frequency": float(label_frequency),
        "real_h": float(1.0 - label_frequency),
        "pi_h_well_conditioned": True,
    }
    return X, y_pu, y_true, meta


def make_pnu_data(
    n_p: int,
    n_n: int,
    n_u: int,
    n_features: int,
    separation: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PNU tri-label data: {+1, -1} ground truth, {+1, -1, 0} observable.

    P = {+1} labeled positives, N = {-1} trusted negatives, U = unlabeled
    (mixed real positives/negatives, labels hidden as 0).

    Ground truth comes from a two-Gaussian mixture with centers at
    ±separation/2; U contains roughly half real positives, half real
    negatives (independent of P/N counts for the requested ratio).
    """
    rng = np.random.RandomState(random_state)
    n_pos = n_p + n_u // 2
    n_neg = n_n + n_u // 2
    c_pos = np.full(n_features, separation / 2.0)
    c_neg = np.full(n_features, -separation / 2.0)
    lhs = rng.randn(n_pos, n_features) * 0.5 + c_pos
    rhs = rng.randn(n_neg, n_features) * 0.5 + c_neg
    X_ = np.vstack([lhs, rhs])
    y_true_mix = np.array([1] * n_pos + [-1] * n_neg)
    y_obs = np.zeros(n_pos + n_neg, dtype=int)
    # P: first n_p of positives observed; N: first n_n of negatives trusted;
    # U: the rest become unlabeled (mixed).
    pos_idx = np.arange(n_pos)
    neg_idx = np.arange(n_neg) + n_pos
    y_obs[pos_idx[:n_p]] = 1
    y_obs[neg_idx[:n_n]] = -1
    perm = rng.permutation(n_pos + n_neg)
    X = X_[perm]
    y_true = y_true_mix[perm]
    y_pnu = y_obs[perm]
    return X, y_pnu, y_true
