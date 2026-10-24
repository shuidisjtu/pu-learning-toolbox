# ruff: noqa: N803
"""Explicit CVIR prior bindings for exploratory, isolated four-role experiments."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path

import numpy as np
from sklearn.base import clone

from .checkpoints import EpochCheckpointTrainer
from .protocols import Trainer


class CVIRKnownPriorTrainer(Trainer):
    """Keep selection's population pi separate from CVIR's configured alpha_U.

    The four-role runner generates OS labels. Native TS input must be supplied
    directly to CVIR, not manufactured by reinterpreting this runner's U rows.
    A source description is mandatory here; it is provenance, not independent
    statistical verification that the supplied prior/distribution is correct.
    """

    def validate_experiment(self, model, bundle, config, protocols):
        """Refuse formal execution and unsupported paths before any fitting."""
        self._validate(model, config.get("os_or_ts"))
        if config.get("survey_protocol") is not None:
            raise ValueError("CVIR known-prior adapter is exploratory, not a frozen Survey method")

    def validate_generation(self, train_meta, val_meta):
        """This adapter has no approved selection-bias correction for SAR."""
        if any(meta.get("mechanism") != "scar" for meta in (train_meta, val_meta)):
            raise ValueError("CVIR exploratory adapter supports SCAR only, not SAR adaptation")

    @staticmethod
    def _validate(model, view):
        from ..estimators.risk.cvir import CVIRClassifier

        if not isinstance(model, CVIRClassifier):
            raise TypeError("CVIRKnownPriorTrainer requires CVIRClassifier")
        if view not in (None, "os") or model.input_sampling != "os":
            raise ValueError("CVIR OS runner cannot supply native TS input or TS-OS calibration")
        if not isinstance(model.prior_source, str) or not model.prior_source.strip():
            raise ValueError("CVIR runner requires explicit alpha_U prior_source")
        alpha = model.unlabeled_positive_prior
        if isinstance(alpha, bool) or not isinstance(alpha, Real) or not 0 < alpha < 1:
            raise ValueError("CVIR runner requires finite alpha_U in (0, 1)")

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None, os_or_ts=None):
        """Do not forward population pi as alpha; use real epoch snapshots."""
        self._validate(estimator, os_or_ts)
        if class_prior is not None and (not math.isfinite(class_prior) or not 0 < class_prior < 1):
            raise ValueError("selection population prior must be in (0, 1)")
        result = EpochCheckpointTrainer().fit(estimator, X, y, val_pu=val_pu)
        estimator.training_provenance_["runner_fit_population_prior_ignored"] = class_prior
        estimator.training_provenance_["population_prior_forwarded_to_fit"] = False
        return result


def cvir_risk_prior(population_prior, *, n_samples, n_labeled, os_or_ts):
    """Protocol §3 known-prior conversion, using observed counts, never hidden Y.

    This is not a prior estimator or proof of finite-sample class proportions.
    An impossible or boundary OS conversion is rejected, not clipped.
    """
    if (
        isinstance(population_prior, bool | np.bool_)
        or not isinstance(population_prior, Real)
        or not math.isfinite(population_prior)
        or not 0 < population_prior < 1
    ):
        raise ValueError("population prior must be finite and in (0, 1)")
    if type(n_samples) is not int or type(n_labeled) is not int or not 0 < n_labeled < n_samples:
        raise ValueError("CVIR prior conversion requires positive P and U counts")
    if os_or_ts not in ("os", "ts"):
        raise ValueError("os_or_ts must be os or ts")
    alpha = (
        float(population_prior)
        if os_or_ts == "ts"
        else (population_prior * n_samples - n_labeled) / (n_samples - n_labeled)
    )
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("converted CVIR risk prior must be in (0, 1); refusing to clip")
    return float(alpha)


class CVIRPopulationPriorTrainer(EpochCheckpointTrainer):
    """Bind protocol population pi to the actual CVIR risk view, opt-in only.

    Default CVIR and the older explicit-alpha OS adapter are unchanged. This
    adapter never forwards population ``class_prior`` to estimator.fit; it
    temporarily binds the resolved constructor alpha/source for that fit and
    restores the caller's configuration even after an exception. The fitted
    ``unlabeled_positive_prior_`` and receipt retain the actual training alpha.
    """

    records_epoch_checkpoints = True

    def __init__(self, *, population_prior_source, checkpoint_dir=None):
        super().__init__(checkpoint_dir=checkpoint_dir)
        if not isinstance(population_prior_source, str) or not population_prior_source.strip():
            raise ValueError("population_prior_source must be a nonempty description")
        self.population_prior_source = population_prior_source

    @staticmethod
    def _validate_model(estimator, view):
        from ..estimators.risk.cvir import CVIRClassifier

        if not isinstance(estimator, CVIRClassifier):
            raise TypeError("CVIRPopulationPriorTrainer requires CVIRClassifier")
        if view not in (None, "os", "ts") or estimator.input_sampling != "os":
            raise ValueError("population-prior adapter requires OS source input and os/ts view")
        if estimator.unlabeled_positive_prior is not None or estimator.prior_source is not None:
            raise ValueError("population-prior adapter cannot overwrite an explicit alpha_U/source")

    def validate_experiment(self, model, bundle, config, protocols):
        """Validate all candidate identities and storage bounds before generation."""
        self._validate_model(model, config.get("os_or_ts"))
        if config.get("survey_protocol") is not None:
            raise ValueError("CVIR population adapter is exploratory, not a frozen Survey method")
        if config.get("capture_epoch_checkpoints", True) is not True:
            raise ValueError("CVIR population adapter requires completed epoch checkpoints")
        size = config.get("checkpoint_bytes_per_component")
        if type(size) is not int or size <= 0:
            raise ValueError("CVIR population adapter requires checkpoint_bytes_per_component")
        candidates = config.get("candidates", [{}])
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("CVIR requires a nonempty candidate list")
        bounds = []
        for parameters in candidates:
            if not isinstance(parameters, dict):
                raise ValueError("CVIR candidate parameters must be objects")
            candidate = clone(model).set_params(**parameters)
            self._validate_model(candidate, config.get("os_or_ts"))
            warm, main = candidate.warm_start_epochs, candidate.max_epochs
            if type(warm) is not int or type(main) is not int or warm < 0 or main <= 0:
                raise ValueError("CVIR candidate epoch budgets must be valid integers")
            bounds.append(warm + main)
        self._epoch_bound = max(bounds)
        self._checkpoint_bytes_per_component = size

    validate_generation = CVIRKnownPriorTrainer.validate_generation

    def bind_checkpoint_directory(self, root):
        """Use the exact directory checked by the runner's disk preflight."""
        self.checkpoint_dir = root

    def checkpoint_epoch_bound(self):
        """Include both stages and the largest candidate, not only the base model."""
        return self._epoch_bound

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None, os_or_ts=None):
        """Forward only train PU data; external selection remains OS and separate."""
        self._validate_model(estimator, os_or_ts)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(X) or not np.isin(labels, [0, 1]).all():
            raise ValueError("CVIR prior conversion requires aligned binary observed PU labels")
        view = "os" if os_or_ts is None else os_or_ts
        count = int(np.count_nonzero(labels == 1))
        alpha = cvir_risk_prior(class_prior, n_samples=len(labels), n_labeled=count, os_or_ts=view)
        estimator.set_params(
            unlabeled_positive_prior=alpha, prior_source=self.population_prior_source
        )
        try:
            result = super().fit(estimator, X, y, os_or_ts=view)
        finally:
            estimator.set_params(unlabeled_positive_prior=None, prior_source=None)
        estimator.training_provenance_["population_prior_binding"] = {
            "population_prior": float(class_prior),
            "risk_pool_positive_prior": alpha,
            "formula": "pi" if view == "ts" else "(pi*n-n_L)/(n-n_L)",
            "n_train": len(labels),
            "n_labeled_positive": count,
            "prior_source": self.population_prior_source,
            "source_verified": False,
            "uses_hidden_training_labels": False,
            "formal_admission": False,
        }
        estimator.training_provenance_["population_prior_forwarded_to_fit"] = False
        if hasattr(self, "_checkpoint_bytes_per_component") and any(
            Path(checkpoint.path).stat().st_size > self._checkpoint_bytes_per_component
            for checkpoint in result.checkpoints
        ):
            raise ValueError("CVIR checkpoint exceeded declared per-component byte bound")
        return result
