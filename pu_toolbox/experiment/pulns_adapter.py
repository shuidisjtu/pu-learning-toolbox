# ruff: noqa: N803, N806
"""Opt-in PULNS extra-supervision experiments, never a frozen PU-only recipe.

No automatic splitting or clean-val reuse: callers supply a separate clean
support partition and explicitly account for its origin and label budget.
"""

from __future__ import annotations

import hashlib

import numpy as np

from ..core.training_views import ROLES
from ..utils.serialization import canonical_hash
from .datasets import survey_dataset_catalog
from .protocols import Trainer
from .strategies import ProtocolOA
from .tracking import EpochRecord, RunTrajectory


def _ids(values, name, *, unique=True):
    values = np.asarray(values, dtype=object)
    if values.ndim != 1:
        raise ValueError(f"{name} IDs must be one-dimensional")
    result = values.tolist()
    if any(isinstance(v, bool) or not isinstance(v, int | str) or v == "" for v in result):
        raise ValueError(f"{name} needs stable nonempty string or integer IDs")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} IDs must be unique")
    return result


def _disjoint(roles):
    seen = set()
    for name, values in roles.items():
        if seen.intersection(values):
            raise ValueError(f"{name} identities overlap another data role")
        seen.update(values)


def _support_hash(part):
    features = np.asarray(part.X)
    return canonical_hash(
        {
            "indices": _ids(part.indices, "support"),
            "labels": np.asarray(part.labels).tolist(),
            "shape": list(features.shape),
            "dtype": str(features.dtype),
            "feature_sha256": hashlib.sha256(features.tobytes(order="C")).hexdigest(),
        }
    )


class PULNSExtraSupportTrainer(Trainer):
    """Explore PULNS with audited extra labels and terminal OA selection only.

    ``support`` is a clean DatasetPart with stable global sample IDs. Groups
    must be supplied for every role when subject/duplicate groups exist. For
    ADNI, unknown subject identity blocks this adapter, not just the report.
    ``prior_estimation_indices=[]`` explicitly declares no data-based prior
    estimation; otherwise its independent sample IDs are audited as a role.
    """

    def __init__(
        self,
        support,
        *,
        dataset,
        source,
        prior_estimation_indices,
        original_train_count,
        support_removed_from_train,
        group_policy,
        preparation_seed,
        sampling_rule,
        role_groups=None,
    ):
        self.support = support
        self.dataset = dataset
        self.source = source
        self.prior_estimation_indices = prior_estimation_indices
        self.original_train_count = original_train_count
        self.support_removed_from_train = support_removed_from_train
        self.group_policy = group_policy
        self.role_groups = role_groups
        self.preparation_seed = preparation_seed
        self.sampling_rule = sampling_rule

    def validate_experiment(self, model, bundle, config, protocols):
        """Validate actual runner role identities before spending any budget."""
        from ..estimators.deep.pulns import PULNSClassifier

        self._prepared = False
        if not isinstance(model, PULNSClassifier):
            raise TypeError("PULNSExtraSupportTrainer requires PULNSClassifier")
        if config.get("survey_protocol") is not None or config.get("os_or_ts") not in (None, "os"):
            raise ValueError("PULNS support adapter is exploratory OS only, not frozen Survey")
        if not protocols or any(type(p) is not ProtocolOA for p in protocols):
            raise ValueError("extra-supervised PULNS requires OA only; not PU-only PA eligible")
        if self.dataset not in survey_dataset_catalog():
            raise ValueError("dataset must name an approved Survey label mapping")
        declared_dataset = (config.get("split_ref") or {}).get("dataset")
        if declared_dataset is not None and declared_dataset != self.dataset:
            raise ValueError("support dataset disagrees with the actual split reference")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("independent support source must be recorded")
        if self.preparation_seed is not None and type(self.preparation_seed) is not int:
            raise ValueError("support preparation_seed must be an integer or explicit None")
        if not isinstance(self.sampling_rule, str) or not self.sampling_rule.strip():
            raise ValueError("support sampling_rule must be recorded")
        support = self.support
        X, y = np.asarray(support.X), np.asarray(support.labels)
        if support.view != "clean" or support.for_selection is not False:
            raise ValueError("support must be clean and for_selection=False")
        if X.shape[1:] != bundle.train.X.shape[1:] or X.ndim != bundle.train.X.ndim:
            raise ValueError("support feature shape must match training")
        if y.shape != (len(X),) or set(np.unique(y)) != {0, 1}:
            raise ValueError("support requires nonempty positive and negative clean labels")
        if not np.isfinite(X).all():
            raise ValueError("support must have finite features")
        roles = {name: _ids(getattr(bundle, name).indices, name) for name in ROLES}
        for name in ROLES:
            if len(roles[name]) != len(getattr(bundle, name).X):
                raise ValueError(f"{name} IDs must match feature rows")
        roles["prior_estimation"] = _ids(self.prior_estimation_indices, "prior_estimation")
        roles["support"] = _ids(support.indices, "support")
        if len(roles["support"]) != len(X):
            raise ValueError("support IDs must match feature rows")
        _disjoint(roles)
        if self.group_policy not in ("not_applicable", "grouped"):
            raise ValueError("group_policy must explicitly declare not_applicable or grouped")
        if self.dataset == "adni" and self.group_policy != "grouped":
            raise ValueError(
                "ADNI requires subject/group isolation; unknown groups block execution"
            )
        group_digest = None
        if self.group_policy == "grouped":
            if not isinstance(self.role_groups, dict) or set(self.role_groups) != set(roles):
                raise ValueError("grouped support audit requires every role's group IDs")
            groups = {name: _ids(self.role_groups[name], name, unique=False) for name in roles}
            if any(len(groups[name]) != len(roles[name]) for name in roles):
                raise ValueError("group IDs must align with sample IDs for every role")
            _disjoint(groups)
            group_digest = canonical_hash(groups)
        elif self.role_groups is not None:
            raise ValueError("group IDs cannot be ignored under not_applicable")
        if type(self.support_removed_from_train) is not bool:
            raise ValueError("support_removed_from_train must be an explicit bool")
        expected = len(bundle.train.X) + (len(X) if self.support_removed_from_train else 0)
        if type(self.original_train_count) is not int or self.original_train_count != expected:
            raise ValueError("original_train_count does not match declared support pool effect")
        spec = survey_dataset_catalog()[self.dataset]
        self._roles = roles
        self._support_digest = _support_hash(support)
        self._train_features_sha = hashlib.sha256(bundle.train.X.tobytes(order="C")).hexdigest()
        self._audit = {
            "method": "pulns",
            "dataset": self.dataset,
            "extra_supervision": True,
            "comparison_group": "extra_clean_support",
            "display_label": "PULNS + extra clean support (not PU-only)",
            "support_source": self.source,
            "support_samples": len(y),
            "preparation_seed": self.preparation_seed,
            "sampling_rule": self.sampling_rule,
            "support_positive_labels": int(np.sum(y == 1)),
            "support_negative_labels": int(np.sum(y == 0)),
            "support_label_uses": ["training_reward", "internal_best_classifier_selection"],
            "support_in_classifier_fit_rows": False,
            "original_train_count": self.original_train_count,
            "remaining_train_count": len(bundle.train.X),
            "support_removed_from_train": self.support_removed_from_train,
            "role_counts": {name: len(ids) for name, ids in roles.items()},
            "role_ids_sha256": canonical_hash(roles),
            "per_role_ids_sha256": {name: canonical_hash(ids) for name, ids in roles.items()},
            "support_sha256": self._support_digest,
            "group_policy": self.group_policy,
            "group_ids_sha256": group_digest,
            "isolation_status": "all_role_ids_disjoint",
            "all_role_isolation_verified": True,
            "identity_scope": (
                "caller-provided stable IDs; alias IDs for duplicates require group audit"
            ),
            "prior_estimation_used": bool(roles["prior_estimation"]),
            "survey_positive_classes": list(spec.positive_classes),
            "survey_negative_classes": list(spec.negative_classes),
            "raw_label_mapping_verified": False,
            "source_recipe_replay": False,
            "calibration_applied": False,
            "pa_eligible": False,
            "formal_admission": False,
        }
        self._prepared = True

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None, os_or_ts=None):
        """Use only explicit support; never receive clean_val/test labels."""
        from ..estimators.deep.pulns import PULNSClassifier

        if not getattr(self, "_prepared", False):
            raise ValueError("PULNS support trainer must pass runner all-role preflight first")
        if not isinstance(estimator, PULNSClassifier) or os_or_ts not in (None, "os"):
            raise ValueError("PULNS support adapter accepts PULNS OS input only")
        if (
            _support_hash(self.support) != self._support_digest
            or hashlib.sha256(X.tobytes(order="C")).hexdigest() != self._train_features_sha
        ):
            raise ValueError("support or training features changed after isolation preflight")
        estimator.fit(
            X,
            y,
            support_data=(self.support.X, self.support.labels),
            train_indices=np.asarray(self._roles["train"]),
            support_indices=np.asarray(self._roles["support"]),
        )
        estimator.training_provenance_ = dict(self._audit)
        estimator.training_provenance_["observed_labeled_positive_count"] = int(np.sum(y == 1))
        estimator.training_provenance_["total_observed_training_label_budget"] = int(
            np.sum(y == 1)
        ) + len(self.support.labels)
        estimator.training_provenance_["stage_optimizer_steps"] = dict(
            estimator.stage_optimizer_steps_
        )
        return RunTrajectory(epochs=[EpochRecord(epoch=1, metrics={})], model=estimator)

    def supervision_context(self):
        """Prominent manifest disclosure, not buried in method footnotes."""
        return dict(self._audit)

    def validate_generation(self, train_meta, val_meta):
        """Only the disclosed SCAR OS engineering path is admitted here."""
        if any(meta.get("mechanism") != "scar" for meta in (train_meta, val_meta)):
            raise ValueError("PULNS exploratory support adapter supports SCAR only")
