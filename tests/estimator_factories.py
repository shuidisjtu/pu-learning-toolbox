# ruff: noqa: N803
"""Zero-argument estimator factories shared by test tiers.

Each entry builds one registered method with parameters shrunk so the tier
stays fast.  ``FACTORY_MAP`` maps a registry name to a zero-argument
callable returning a fresh, unfitted estimator.

The ``class_prior`` values below are **fixed test constants**.  They are not
derived from, and must not be re-tuned to, any fixture's true prior --
selecting a prior from test truth is forbidden by the governance plan
(phase 5).  The values were chosen per method by whoever integrated it; there
is no uniform policy (``pnu`` 0.4 matches the contract PNU fixture's 20/50,
eight methods use 0.33 matching the fixture's 30/90 -- four written as the
``class_prior`` keyword and four positionally -- while ``upu`` 0.5 and
``dist_pu`` 0.3 match neither).  Recorded here so nobody later mistakes them
for a deliberate design.

This module must stay importable **without** PyTorch installed: the Iris
smoke tier imports it in torch-free environments.  Do not add a module-level
``import torch`` or ``pytest.importorskip`` -- import torch lazily inside
the factory that needs it.
"""

from __future__ import annotations

import numpy as np

# ── Factory functions ─────────────────────────────────────────────


def _make_elkan_noto():
    from pu_toolbox.estimators.classic.elkan_noto import ElkanNotoClassifier

    return ElkanNotoClassifier(n_cv_folds=3, random_state=42)


def _make_llsvm():
    from pu_toolbox.estimators.classic.llsvm import LLSVMClassifier

    # Defaults run 3000 SGD epochs; shrink for the 90-sample contract data.
    return LLSVMClassifier(max_epochs=10, min_epochs=1, random_state=42)


def _make_upu():
    from pu_toolbox.estimators.risk.upu import UPUClassifier

    return UPUClassifier(
        class_prior=0.5, loss="logistic", reg_lambda=1.0, max_iter=200, random_state=42
    )


def _make_nnpu():
    import torch

    from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier

    torch.manual_seed(42)
    return NonNegativePUClassifier(max_epochs=1, batch_size=8, random_state=42)


def _make_pnu():
    from pu_toolbox.estimators.risk.pnu import PNUClassifier

    return PNUClassifier(class_prior=0.4, eta=0.5, reg_lambda=1.0, random_state=42)


def _make_ldce():
    from pu_toolbox.estimators.risk.ldce import LDCEClassifier

    return LDCEClassifier(flip_probability=0.3, max_iter=10, tol=1e-4, random_state=42)


def _make_kldce():
    from pu_toolbox.estimators.risk.kldce import KLDCEClassifier

    return KLDCEClassifier(
        flip_probability=0.3, sigma=2.0, max_acs_iter=5, tol=1e-4, random_state=42
    )


def _make_dist_pu():
    from pu_toolbox.estimators.risk.dist_pu import DistPUClassifier

    return DistPUClassifier(0.3, hidden_dim=8, epochs=2, random_state=42)


def _make_pusb():
    from pu_toolbox.estimators.bias_aware.pusb import PUSBClassifier

    return PUSBClassifier(threshold=0.5)


def _make_pusb_kernel():
    from pu_toolbox.estimators.bias_aware.pusb_kernel import PUSBKernelClassifier

    # Small grid / low basis keep the full CV + refit affordable in tests.
    # sigma grid matches the ±2-separated 90-sample data (pairwise d2 ~ 80):
    # sigma=2 -> exp(-10) ~ 4.5e-5, sigma=4 -> exp(-2.5) ~ 0.08, both alive.
    # cv=2 relies on the 30-positive balance for the per-fold P/U guard.
    return PUSBKernelClassifier(
        n_basis=10,
        sigma_grid=[2.0, 4.0],
        reg_grid=[0.01, 0.1],
        cv=2,
        max_iter=50,
        random_state=42,
    )


def _make_lbe():
    from pu_toolbox.estimators.bias_aware.lbe import LBEClassifier

    return LBEClassifier(n_em_iter=3)


def _make_class_prior_estimation():
    from pu_toolbox.prior.pen_l1 import ClassPriorEstimator

    return ClassPriorEstimator(n_centers=50)


def _make_recpe():
    from pu_toolbox.prior.recpe import ReCPEEstimator

    return ReCPEEstimator(copy_fraction=0.1)


def _make_infomax_pu():
    from pu_toolbox.estimators.deep import InfoMaxPUClassifier

    return InfoMaxPUClassifier(
        class_prior=0.33,
        representation_dim=3,
        hidden_dim=8,
        representation_epochs=1,
        classifier_epochs=1,
        random_state=42,
    )


def _make_weighted_contrastive_pu():
    from pu_toolbox.estimators.deep import WeightedContrastivePUClassifier

    return WeightedContrastivePUClassifier(
        0.33,
        hidden_dim=8,
        embedding_dim=4,
        queue_size=16,
        batch_size=32,
        max_epochs=1,
        random_state=42,
    )


def _make_self_pu():
    from pu_toolbox.estimators.deep import SelfPUClassifier

    return SelfPUClassifier(
        0.33,
        hidden_dim=8,
        warmup_epochs=0,
        self_paced_start=0,
        self_paced_end=1,
        distill_start=1,
        max_epochs=1,
        batch_size=32,
        random_state=42,
    )


class _MockConditionalGenerator:
    def fit(self, X, y, *, warm_start=True):
        self.means_ = {label: X[y == label].mean(axis=0) for label in np.unique(y)}
        self.n_features_in_ = X.shape[1]
        return self

    def sample(self, n_samples, *, class_label, random_state=None):
        rng = np.random.RandomState(random_state)
        mean = self.means_.get(class_label, np.zeros(self.n_features_in_))
        return mean + 0.01 * rng.randn(n_samples, self.n_features_in_)


def _make_dgpu():
    from pu_toolbox.estimators.deep import DGPUClassifier

    return DGPUClassifier(
        0.33,
        _MockConditionalGenerator(),
        hidden_dim=8,
        rounds=1,
        initialization_epochs=1,
        annotation_epochs=1,
        generated_samples=6,
        random_state=42,
    )


def _make_gradpu():
    from pu_toolbox.estimators.deep import GradPUClassifier

    return GradPUClassifier(hidden_dim=8, batch_size=32, max_epochs=1, random_state=42)


def _make_robust_pu():
    from pu_toolbox.estimators.deep import RobustPUClassifier

    return RobustPUClassifier(
        class_prior=0.33,
        hidden_dim=8,
        pretrain_epochs=1,
        episodes=1,
        batch_size=32,
        random_state=42,
    )


def _make_split_pu():
    from pu_toolbox.estimators.deep import SplitPUClassifier

    return SplitPUClassifier(
        class_prior=0.33,
        hidden_dim=8,
        teacher_epochs=1,
        split_epochs=1,
        student_epochs=1,
        rounds=1,
        batch_size=32,
        random_state=42,
    )


def _make_lagam():
    from pu_toolbox.estimators.deep import LaGAMClassifier

    return LaGAMClassifier(
        hidden_dim=8,
        warmup_epochs=1,
        max_epochs=2,
        batch_size=32,
        support_batch_size=8,
        num_clusters=2,
        random_state=42,
    )


def _make_vpu():
    from pu_toolbox.estimators.risk import VPUClassifier

    return VPUClassifier(hidden_dim=8, batch_size=32, max_epochs=1, random_state=42)


def _make_pulda():
    from pu_toolbox.estimators.risk import PULDAClassifier

    return PULDAClassifier(
        0.33,
        hidden_dim=8,
        warmup_epochs=1,
        pu_epochs=1,
        positive_batch_size=8,
        unlabeled_batch_size=32,
        random_state=42,
    )


def _make_puet():
    from pu_toolbox.estimators.risk import PUExtraTreesClassifier

    return PUExtraTreesClassifier(class_prior=0.33, n_estimators=3, max_depth=4, random_state=42)


FACTORY_MAP: dict[str, callable] = {
    "elkan_noto": _make_elkan_noto,
    "llsvm": _make_llsvm,
    "upu": _make_upu,
    "nnpu": _make_nnpu,
    "pnu": _make_pnu,
    "centroid_pu": _make_ldce,
    "kldce": _make_kldce,
    "dist_pu": _make_dist_pu,
    "pusb": _make_pusb,
    "pusb_kernel": _make_pusb_kernel,
    "lbe": _make_lbe,
    "class_prior_estimation": _make_class_prior_estimation,
    "recpe": _make_recpe,
    "self_pu": _make_self_pu,
    "infomax_pu": _make_infomax_pu,
    "weighted_contrastive_pu": _make_weighted_contrastive_pu,
    "dgpu": _make_dgpu,
    "gradpu": _make_gradpu,
    "robust_pu": _make_robust_pu,
    "split_pu": _make_split_pu,
    "lagam": _make_lagam,
    "vpu": _make_vpu,
    "pulda": _make_pulda,
    "puet": _make_puet,
}


def fit_kwargs(clf, y) -> dict[str, float]:
    """Class-prior kwargs for the estimators that need them at fit time.

    Narrowly scoped to the classes whose fit *requires* an explicit prior
    (or overrides the constructor value by design): nnPU and
    PUSBKernelClassifier.  Deliberately NOT keyed off the
    ``requires_class_prior`` class attribute — many estimators (uPU, PNU,
    Dist-PU, Self-PU, ...) declare it but train with their
    constructor-chosen prior; injecting here would silently override it.
    """
    from pu_toolbox.estimators.bias_aware.pusb_kernel import PUSBKernelClassifier
    from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier

    n_p = int(np.sum(y == 1))
    if isinstance(clf, NonNegativePUClassifier | PUSBKernelClassifier):
        return {"class_prior": n_p / len(y)}
    return {}
