# ruff: noqa: N803
"""Explicit known-alpha CVIR adapter for exploratory OS experiments only."""

from __future__ import annotations

import math
from numbers import Real

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
