"""Behavior of the two OS constructors and their shared observation contract."""

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pu_toolbox.estimators.risk.upu import UPUClassifier
from pu_toolbox.preprocessing import (
    GaussianSourceConfig,
    PULabelingConfig,
    SourcePartition,
    construct_pu_dataset,
    simulate_pu_dataset,
)

pytestmark = pytest.mark.unit


def _source():
    labels = np.repeat(np.arange(10), 3)
    return SourcePartition(
        np.arange(len(labels) * 4).reshape(-1, 1, 2, 2),
        labels,
        source_id="digits-v1",
        partition="official_train",
        sample_ids=np.arange(100, 100 + len(labels)),
    )


@pytest.mark.parametrize(
    ("positives", "negatives", "n_labeled"),
    [((0, 2, 4, 6, 8), (1, 3, 5, 7, 9), 6), ((3,), (0, 1, 2, 4, 5, 6, 7, 8, 9), 1)],
)
def test_param_os_mapping_is_caller_controlled_and_preserves_source(
    positives, negatives, n_labeled
):
    source = _source()
    original = source.labels.copy()
    result = construct_pu_dataset(
        source,
        PULabelingConfig("scar", "fixed_count", 7, c=0.4),
        positive_classes=positives,
        negative_classes=negatives,
    )
    obs = result.observations
    assert_array_equal(obs.features, source.features)
    assert_array_equal(obs.sample_ids, source.sample_ids)
    assert_array_equal(source.labels, original)
    assert_array_equal(result.audit.true_labels, np.isin(original, positives))
    assert (obs.labels <= result.audit.true_labels).all()
    assert not hasattr(obs, "true_labels")
    assert result.manifest["source"]["partition"] == "official_train"
    assert result.manifest["labeling"]["n_labeled_actual"] == n_labeled
    assert result.manifest["labeling"]["q"] == n_labeled / len(original)


def test_edge_requires_explicit_exclusions_and_keeps_aligned_original_ids():
    source = _source()
    config = PULabelingConfig("scar", "fixed_count", 3, n_labeled=3)
    with pytest.raises(ValueError, match="unmapped"):
        construct_pu_dataset(source, config, positive_classes=[3], negative_classes=[2])
    with pytest.raises(ValueError, match="overlap"):
        construct_pu_dataset(source, config, positive_classes=[3], negative_classes=range(10))
    result = construct_pu_dataset(
        source,
        config,
        positive_classes=[3],
        negative_classes=[2],
        excluded_classes=[0, 1, 4, 5, 6, 7, 8, 9],
    )
    keep = np.isin(source.labels, [2, 3])
    assert_array_equal(result.observations.sample_ids, source.sample_ids[keep])
    assert_array_equal(result.audit.source_labels, source.labels[keep])
    assert result.manifest["labeling"]["n_labeled_requested"] == 3
    assert result.manifest["labeling"]["n_labeled_actual"] == 3


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (PULabelingConfig("scar", "fixed_count", 0, c=0.4, n_labeled=2), "exactly one"),
        (PULabelingConfig("scar", "fixed_count", 0, c=0.4, input_sampling="ts"), "OS"),
        (PULabelingConfig("sar", "fixed_count", 0, c=0.4, propensity_rule="linear"), "Bernoulli"),
        (PULabelingConfig("sar", "bernoulli", 0, c=0.4), "propensity_rule"),
        (PULabelingConfig("scar", "fixed_count", 0, n_labeled=4), "exceeds"),
    ],
)
def test_param_invalid_sampling_design_or_budget_refused(config, message):
    with pytest.raises(ValueError, match=message):
        construct_pu_dataset(
            _source(), config, positive_classes=[3], negative_classes=[0, 1, 2, 4, 5, 6, 7, 8, 9]
        )


def test_edge_missing_or_duplicate_source_identity_refused():
    source = _source()
    config = PULabelingConfig("scar", "fixed_count", 5, c=0.4)
    args = dict(positive_classes=[3], negative_classes=[0, 1, 2, 4, 5, 6, 7, 8, 9])
    with pytest.raises(ValueError, match="source_id"):
        construct_pu_dataset(replace(source, source_id="", sample_ids=None), config, **args)
    with pytest.raises(ValueError, match="unique"):
        construct_pu_dataset(replace(source, sample_ids=np.zeros(30, dtype=int)), config, **args)
    result = construct_pu_dataset(replace(source, sample_ids=None), config, **args)
    assert result.manifest["source"]["identity_scope"] == "source_partition_positions"
    assert len(set(result.observations.sample_ids)) == len(source.labels)


@pytest.mark.parametrize("mechanism", ["scar", "sar"])
def test_determ_simulation_and_existing_data_share_bernoulli_contract(mechanism):
    config = PULabelingConfig(
        mechanism,
        "bernoulli",
        7,
        c=0.3,
        propensity_rule="linear" if mechanism == "sar" else None,
    )
    source = GaussianSourceConfig(24, 2, 0.46, 2.0, 17)
    first = simulate_pu_dataset(source, config)
    second = simulate_pu_dataset(source, config)
    assert first.manifest == second.manifest
    assert_array_equal(first.observations.features, second.observations.features)
    assert_array_equal(first.observations.labels, second.observations.labels)
    obs, audit = first.observations, first.audit
    assert first.manifest["synthetic_source"]["target_population_prior"] == 0.46
    assert first.manifest["audit"]["empirical_population_prior"] == 11 / 24
    assert first.manifest["labeling"]["n_labeled_requested"] is None
    assert np.isclose(audit.propensity[audit.true_labels == 1].mean(), 0.3)
    assert (np.ptp(audit.propensity[audit.true_labels == 1]) > 0) == (mechanism == "sar")
    as_existing = construct_pu_dataset(
        SourcePartition(obs.features, audit.source_labels, "external", "train", obs.sample_ids),
        config,
        positive_classes=[1],
        negative_classes=[0],
    )
    assert_array_equal(as_existing.observations.labels, obs.labels)
    assert_array_equal(as_existing.audit.propensity, audit.propensity)


def test_basic_trainable_observations_need_no_truth_input():
    result = simulate_pu_dataset(
        GaussianSourceConfig(40, 2, 0.5, 2.0, 17),
        PULabelingConfig("scar", "fixed_count", 7, n_labeled=8),
    )
    obs = result.observations
    model = UPUClassifier(0.375, fit_intercept=False).fit(obs.features, obs.labels)
    assert model.predict(obs.features).shape == (40,)
    assert result.manifest["audit"]["empirical_native_unlabeled_prior"] == 0.375


def test_edge_bernoulli_top_up_is_opt_in_and_recorded():
    source = GaussianSourceConfig(12, 2, 0.5, 2.0, 8)
    config = PULabelingConfig("scar", "bernoulli", 7, c=1e-10)
    unmarked = simulate_pu_dataset(source, config)
    topped_up = simulate_pu_dataset(source, replace(config, ensure_labeled=True))
    assert unmarked.observations.labels.sum() == 0
    assert topped_up.observations.labels.sum() == 1
    assert topped_up.manifest["labeling"]["top_up_applied"] is True
    assert_array_equal(unmarked.audit.propensity, topped_up.audit.propensity)


def test_edge_mixed_original_classes_are_not_coerced_to_strings():
    labels = np.array([1, "1", 2, "exclude"], dtype=object)
    source = SourcePartition(np.arange(4)[:, None], labels, "mixed", "official_train")
    result = construct_pu_dataset(
        source,
        PULabelingConfig("scar", "fixed_count", 1, c=1),
        positive_classes=(1, "1"),
        negative_classes=(2,),
        excluded_classes=("exclude",),
    )
    assert_array_equal(result.audit.true_labels, [1, 1, 0])
    assert_array_equal(result.observations.labels, [1, 1, 0])


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("pu_labeling", "make_pu_labels"),
        ("pu_labeling", "make_scar_labels"),
        ("pu_labeling", "make_fixed_count_labels"),
        ("pu_labeling", "make_scar_dataset"),
        ("selection_bias", "make_sar_labels"),
        ("selection_bias", "make_sar_dataset"),
    ],
)
def test_edge_retired_os_functions_have_no_compatibility_exports(module, name):
    namespaces = ("pu_toolbox", "pu_toolbox.preprocessing", f"pu_toolbox.preprocessing.{module}")
    for namespace in namespaces:
        package = import_module(namespace)
        assert not hasattr(package, name)
        assert name not in package.__all__
