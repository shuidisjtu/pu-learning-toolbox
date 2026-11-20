"""Construct OS observations from a selected raw partition or a synthetic source."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np

from ..core.random import check_random_state
from ..utils.serialization import json_scalars, strict_canonical_hash
from ._sampling import fixed_positive_count, sample_bernoulli_labels, sample_positive_labels
from .selection_bias import make_sar_propensity

__all__ = [
    "GaussianSourceConfig",
    "PUAudit",
    "PUConstruction",
    "PULabelingConfig",
    "PUObservations",
    "SourcePartition",
    "construct_pu_dataset",
    "simulate_pu_dataset",
]


@dataclass(frozen=True)
class SourcePartition:
    """Caller-selected source partition with original labels and optional row IDs.

    Use a stable, distinct ``source_id`` (including revision if applicable) and
    partition name. The constructor does not read files or choose a partition.
    Raw text, categorical values and encoded images must be represented as
    numeric features before PU construction.
    """

    features: np.ndarray
    labels: np.ndarray
    source_id: str
    partition: str
    sample_ids: np.ndarray | None = None


@dataclass(frozen=True)
class PULabelingConfig:
    """One OS labeling condition, independent of the source and training view.

    SCAR fixed-count accepts exactly one of ``c`` (fraction of source positives)
    or ``n_labeled`` (absolute quota). SCAR/SAR Bernoulli requires ``c``. SAR also
    requires a named feature-based propensity rule. ``ensure_labeled`` explicitly
    opts into topping up a zero-label Bernoulli draw; the propensity recorded for
    audit is the pre-top-up probability.
    """

    mechanism: Literal["scar", "sar"]
    sampling_policy: Literal["fixed_count", "bernoulli"]
    random_state: int | np.random.RandomState
    c: float | None = None
    n_labeled: int | None = None
    propensity_rule: Literal["linear", "nonlinear"] | None = None
    strength: float = 1.0
    feature_weights: np.ndarray | None = None
    ensure_labeled: bool = False
    input_sampling: Literal["os"] = "os"


@dataclass(frozen=True)
class GaussianSourceConfig:
    """Binary Gaussian source with a rounded class count and explicit random state.

    Centers lie at +/- half the separation on each coordinate. Row ordering is
    explicit so an ordered teaching fixture need not change its source identities.
    """

    n_samples: int
    n_features: int
    population_prior: float
    separation: float
    random_state: int | np.random.RandomState
    row_order: Literal["shuffled", "class_blocked"] = "shuffled"


@dataclass(frozen=True)
class PUObservations:
    """Training-readable rows only: features, PU labels, and source row IDs."""

    features: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray


@dataclass(frozen=True)
class PUAudit:
    """Aligned truth and optional pre-top-up propensity for evaluation and audit."""

    true_labels: np.ndarray
    source_labels: np.ndarray
    propensity: np.ndarray | None


@dataclass(frozen=True)
class PUConstruction:
    """Observations, explicitly separate audit truth, and JSON-ready provenance."""

    observations: PUObservations
    audit: PUAudit
    manifest: dict[str, Any]


def _source_arrays(source: SourcePartition):
    for name in ("source_id", "partition"):
        value = getattr(source, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty source identifier")
    features = np.asarray(source.features)
    if features.ndim < 2 or not all(features.shape) or features.dtype.kind not in "biuf":
        raise ValueError("features must be a nonempty real numeric array with a sample axis")
    if features.dtype.kind == "f" and not np.isfinite(features).all():
        raise ValueError("features must be finite")
    labels = np.asarray(source.labels, dtype=object)
    if labels.shape != (len(features),):
        raise ValueError("source labels must be a 1-D vector aligned with features")
    if source.sample_ids is None:
        namespace = strict_canonical_hash([source.source_id, source.partition])
        ids = np.array([f"{namespace}:{position}" for position in range(len(features))])
        identity_scope = "source_partition_positions"
    else:
        ids = np.asarray(source.sample_ids, dtype=object)
        if ids.shape != labels.shape:
            raise ValueError("sample_ids must be a 1-D vector aligned with features")
        values = json_scalars(ids, name="sample_ids")
        if any(
            type(value) is not int and (not isinstance(value, str) or not value) for value in values
        ):
            raise ValueError("sample_ids must contain nonempty strings or integers")
        if len(set(values)) != len(values):
            raise ValueError("sample_ids must be unique within the source partition")
        identity_scope = "caller_supplied_source_ids"
    return features, labels, ids, identity_scope


def _class_values(values: Iterable, name: str) -> list:
    result = json_scalars(np.asarray(list(values), dtype=object), name=name)
    if any(value is None or value == "" for value in result):
        raise ValueError(f"{name} must contain valid source classes")
    return result


def _random_state(value) -> tuple[np.random.RandomState, dict]:
    if value is None or isinstance(value, bool | np.bool_):
        raise ValueError("random_state must be an explicit integer seed or RandomState")
    rng = check_random_state(value)
    if isinstance(value, Integral):
        return rng, {"seed": int(value)}
    kind, keys, position, has_gauss, cached = rng.get_state()
    return rng, {
        "external_state_sha256": strict_canonical_hash(
            [kind, keys.tolist(), position, has_gauss, cached]
        ),
    }


def _rate(value, name: str, *, allow_one=True) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    if not (0.0 < value <= 1.0 and (allow_one or value < 1.0)):
        raise ValueError(f"{name} must be in {'(0, 1]' if allow_one else '(0, 1)'}")
    return float(value)


def _validate_labeling(config: PULabelingConfig) -> tuple[float | None, int | None]:
    if config.input_sampling != "os":
        raise ValueError("these constructors support OS input only, not independent TS")
    if config.mechanism not in ("scar", "sar"):
        raise ValueError("mechanism must be 'scar' or 'sar'")
    if config.sampling_policy not in ("fixed_count", "bernoulli"):
        raise ValueError("sampling_policy must be 'fixed_count' or 'bernoulli'")
    if config.mechanism == "sar":
        if config.sampling_policy != "bernoulli":
            raise ValueError("SAR construction requires the Bernoulli policy")
        if config.propensity_rule not in ("linear", "nonlinear"):
            raise ValueError("SAR requires propensity_rule='linear' or 'nonlinear'")
    elif (
        config.propensity_rule is not None
        or config.feature_weights is not None
        or config.strength != 1.0
    ):
        raise ValueError("propensity_rule, feature_weights and strength apply only to SAR")
    if type(config.ensure_labeled) is not bool:
        raise ValueError("ensure_labeled must be a bool")

    c = _rate(config.c, "c") if config.c is not None else None
    quota = config.n_labeled
    if config.sampling_policy == "bernoulli":
        if c is None or quota is not None:
            raise ValueError("bernoulli requires c and does not accept n_labeled")
        return c, None
    if (c is None) == (quota is None):
        raise ValueError("fixed_count requires exactly one of c or n_labeled")
    if config.ensure_labeled:
        raise ValueError("ensure_labeled is an option for Bernoulli draws only")
    if quota is not None:
        if isinstance(quota, bool | np.bool_) or not isinstance(quota, Integral) or quota < 1:
            raise ValueError("n_labeled must be a positive integer")
        quota = int(quota)
    return c, quota


def _label_os(features, truth, config: PULabelingConfig):
    c, quota = _validate_labeling(config)
    rng, random_state = _random_state(config.random_state)
    n_positive = int(np.count_nonzero(truth))
    requested = quota
    propensity = None
    top_up = False
    if config.sampling_policy == "fixed_count":
        if quota is None:
            requested = round(c * n_positive)
            count = fixed_positive_count(n_positive, c)
        else:
            if quota > n_positive:
                raise ValueError(f"n_labeled ({quota}) exceeds the source positives ({n_positive})")
            count = quota
        observed = sample_positive_labels(truth, count, rng)
    else:
        if config.mechanism == "scar":
            propensity = np.where(truth == 1, c, 0.0)
        else:
            propensity = make_sar_propensity(
                features,
                truth,
                mechanism=config.propensity_rule,
                label_frequency=c,
                strength=config.strength,
                feature_weights=config.feature_weights,
            )
        observed, top_up = sample_bernoulli_labels(
            truth, propensity, rng, ensure_labeled=config.ensure_labeled
        )
    actual = int(observed.sum())
    settings = {
        "mechanism": config.mechanism,
        "sampling_policy": config.sampling_policy,
        "random_state": random_state,
        "c_requested": c,
        "c_realized": actual / n_positive if n_positive else None,
        "n_labeled_requested": requested,
        "n_labeled_actual": actual,
        "q": actual / len(truth),
        "ensure_labeled": config.ensure_labeled,
        "top_up_applied": top_up,
    }
    if config.mechanism == "sar":
        settings.update(
            propensity_rule=config.propensity_rule,
            strength=float(config.strength),
            feature_weights=(
                None
                if config.feature_weights is None
                else np.asarray(config.feature_weights).tolist()
            ),
        )
    return observed, propensity, settings


def construct_pu_dataset(
    source: SourcePartition,
    labeling: PULabelingConfig,
    *,
    positive_classes: Iterable,
    negative_classes: Iterable,
    excluded_classes: Iterable = (),
) -> PUConstruction:
    """Map source classes and mark positives in one already-loaded partition.

    Class sets must be disjoint, and every observed source class must be mapped
    or explicitly excluded. No download, role split, row shuffle, or estimator
    risk-view conversion is performed here.
    """
    features, labels, ids, identity_scope = _source_arrays(source)
    pos = _class_values(positive_classes, "positive_classes")
    neg = _class_values(negative_classes, "negative_classes")
    excluded = _class_values(excluded_classes, "excluded_classes")
    if len(set(pos + neg + excluded)) != len(pos) + len(neg) + len(excluded):
        raise ValueError("positive, negative and excluded class mappings must not overlap")
    if not pos or not neg:
        raise ValueError("positive_classes and negative_classes must not be empty")
    unmapped = set(_class_values(labels, "source labels")) - set(pos + neg + excluded)
    if unmapped:
        raise ValueError(f"unmapped source classes: {sorted(unmapped, key=repr)!r}")
    n_source = len(labels)
    if excluded:
        keep = ~np.isin(labels, np.asarray(excluded, dtype=object))
        features, labels, ids = features[keep], labels[keep], ids[keep]
    if len(labels) == 0:
        raise ValueError("no source rows remain after exclusions")
    truth = np.isin(labels, np.asarray(pos, dtype=object)).astype(int)
    observed, propensity, settings = _label_os(features, truth, labeling)
    unlabeled = observed == 0
    return PUConstruction(
        PUObservations(features, observed, ids),
        PUAudit(truth, labels, propensity),
        {
            "input_sampling": "os",
            "source": {
                "source_id": source.source_id,
                "partition": source.partition,
                "identity_scope": identity_scope,
                "source_rows": n_source,
                "retained_ids_sha256": strict_canonical_hash(json_scalars(ids, name="sample_ids")),
            },
            "mapping": {"positive": pos, "negative": neg, "excluded": excluded},
            "labeling": settings,
            "pool_sizes": {
                "labeled_positive": settings["n_labeled_actual"],
                "unlabeled": int(unlabeled.sum()),
            },
            "audit": {
                "empirical_population_prior": float(truth.mean()),
                "empirical_native_unlabeled_prior": float(truth[unlabeled].mean())
                if unlabeled.any()
                else None,
            },
        },
    )


def simulate_pu_dataset(source: GaussianSourceConfig, labeling: PULabelingConfig) -> PUConstruction:
    """Generate a clean binary Gaussian partition, then use the same OS constructor.

    Supply separate integer seeds for independent source and labeling streams, or
    the same RandomState object for one sequential stream. External state digests
    identify that input state; replay requires the caller to recreate it.
    """
    _validate_labeling(labeling)
    counts = ((source.n_samples, "n_samples", 2), (source.n_features, "n_features", 1))
    for value, name, minimum in counts:
        if isinstance(value, bool | np.bool_) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    prior = _rate(source.population_prior, "population_prior", allow_one=False)
    separation = source.separation
    if (
        isinstance(separation, bool | np.bool_)
        or not isinstance(separation, Real)
        or not np.isfinite(separation)
        or separation < 0
    ):
        raise ValueError("separation must be a finite nonnegative number")
    if source.row_order not in ("shuffled", "class_blocked"):
        raise ValueError("row_order must be 'shuffled' or 'class_blocked'")
    rng, random_state = _random_state(source.random_state)
    shared_stream = (
        isinstance(source.random_state, np.random.RandomState)
        and source.random_state is labeling.random_state
    )
    n_positive = min(source.n_samples - 1, max(1, round(source.n_samples * prior)))
    truth = np.zeros(source.n_samples, dtype=int)
    truth[:n_positive] = 1
    if source.row_order == "shuffled":
        rng.shuffle(truth)
    center = np.where(truth[:, None] == 1, separation / 2, -separation / 2)
    features = rng.normal(loc=center, scale=1.0, size=(source.n_samples, source.n_features))
    parameters = {
        "distribution": "gaussian_binary",
        "n_samples": int(source.n_samples),
        "n_features": int(source.n_features),
        "target_population_prior": prior,
        "separation": float(separation),
        "row_order": source.row_order,
        "random_state": random_state,
    }
    namespace = f"gaussian:{strict_canonical_hash(parameters)}"
    result = construct_pu_dataset(
        SourcePartition(features, truth, namespace, "generated"),
        labeling,
        positive_classes=(1,),
        negative_classes=(0,),
    )
    parameters["shared_labeling_stream"] = shared_stream
    return replace(result, manifest={**result.manifest, "synthetic_source": parameters})
