# ruff: noqa: N802, N803, N806
"""Capability-declaration contract tests for registered classifiers.

Invariants from the phase-0 design spec §6: declaration legality, registry
sync, tabular_only derivation, and cross-mechanism consistency with the
constructor-signature check (dual_architecture_plan.md §4.2).  The metadata
field set is closed and ``encoder_parameter`` is declarative only (issue #45,
route B).
"""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from pu_toolbox.core.base import BasePUClassifier
from pu_toolbox.core.tags import AlgorithmFamily, Backend, ImplementationStatus
from pu_toolbox.registry import (
    AlgorithmMetadata,
    list_algorithms,
    register_all_builtin_methods,
    register_method,
    unregister_method,
)
from pu_toolbox.registry.registry import get_algorithm
from pu_toolbox.workflows import PipelineError, PUPipeline

_LEGAL_NDIMS = {2, 4}
_LEGAL_ARCHS = {"mlp", "cnn"}


def _classifier_entries():
    """Yield (metadata, class) for every registered PU classifier."""
    register_all_builtin_methods()
    for meta in list_algorithms():
        cls = get_algorithm(meta.name)
        if isinstance(cls, type) and issubclass(cls, BasePUClassifier):
            yield meta, cls


@pytest.mark.contract
def test_declarations_are_legal():
    import inspect

    for meta, cls in _classifier_entries():
        assert cls.input_ndims, f"{meta.name}: input_ndims must be non-empty"
        assert cls.input_ndims <= _LEGAL_NDIMS, f"{meta.name}: input_ndims {cls.input_ndims}"
        assert cls.native_architectures <= _LEGAL_ARCHS, (
            f"{meta.name}: native_architectures {cls.native_architectures}"
        )
        if cls.encoder_parameter is not None:
            assert cls.encoder_parameter in inspect.signature(cls.__init__).parameters, (
                f"{meta.name}: encoder_parameter {cls.encoder_parameter!r} "
                "not in __init__ signature"
            )
        if cls.trains_encoder:
            assert cls.encoder_parameter is not None, (
                f"{meta.name}: trains_encoder=True requires encoder_parameter"
            )


@pytest.mark.contract
def test_registry_sync_matches_class():
    for meta, cls in _classifier_entries():
        assert meta.native_architectures == cls.native_architectures, meta.name
        assert meta.input_ndims == cls.input_ndims, meta.name
        assert meta.encoder_parameter == cls.encoder_parameter, meta.name
        assert meta.trains_encoder == cls.trains_encoder, meta.name


@pytest.mark.contract
def test_tabular_only_derived_from_empty_native_architectures():
    for meta, cls in _classifier_entries():
        expected = cls.native_architectures == frozenset()
        assert meta.is_tabular_only == expected, meta.name


@pytest.mark.contract
def test_cnn_capability_consistent_with_signature():
    from pu_toolbox.workflows._models import declares_encoder_parameter

    for meta, cls in _classifier_entries():
        if "cnn" in cls.native_architectures:
            assert 4 in cls.input_ndims, meta.name
            assert cls.encoder_parameter is not None, meta.name
            assert declares_encoder_parameter(cls), meta.name


_EXPECTED_DECLARATIONS = {
    "infomax_pu": (frozenset({"mlp", "cnn"}), frozenset({2, 4}), "encoder", True),
    "weighted_contrastive_pu": (frozenset({"mlp", "cnn"}), frozenset({2, 4}), "encoder", True),
    "self_pu": (frozenset({"mlp"}), frozenset({2, 4}), None, False),
    "nnpu": (frozenset({"mlp", "cnn"}), frozenset({2, 4}), "encoder", True),
    "dist_pu": (frozenset({"mlp"}), frozenset({2}), None, False),
    "dgpu": (frozenset({"mlp"}), frozenset({2}), None, False),
}


@pytest.mark.contract
def test_deep_capability_declarations():
    register_all_builtin_methods()
    for name, (archs, ndims, enc_param, trains) in _EXPECTED_DECLARATIONS.items():
        cls = get_algorithm(name)
        assert cls.native_architectures == archs, name
        assert cls.input_ndims == ndims, name
        assert cls.encoder_parameter == enc_param, name
        assert cls.trains_encoder == trains, name


_CAPABILITY_FIELDS = {"native_architectures", "input_ndims", "encoder_parameter", "trains_encoder"}


@pytest.mark.contract
def test_capability_field_set_is_exactly_the_four_declared_fields():
    """Issue #45 (route B): the metadata capability surface stays closed.

    ``adapter_architectures`` (route A) must not be reintroduced silently --
    it would surface here as a fifth architecture-named field.  Adding any
    capability field is a deliberate schema decision that updates this pin.
    """
    names = {f.name for f in fields(AlgorithmMetadata)}
    capability_named = {
        name for name in names if name in _CAPABILITY_FIELDS or "architecture" in name
    }
    assert capability_named == _CAPABILITY_FIELDS


class _EncoderDeclaredByNameOnly(BasePUClassifier):
    """Fixture: constructor accepts the metadata-declared name, not ``encoder``."""

    backend = Backend.TORCH

    def __init__(self, *, foo=None):
        self.foo = foo

    def fit(self, X, y_pu, *, class_prior=None, sample_weight=None):
        return self

    def _predict(self, X):
        return np.zeros(len(X))

    def _decision_function(self, X):
        return np.zeros(len(X))


@pytest.mark.contract
def test_encoder_parameter_metadata_never_drives_encoder_injection():
    """Issue #45 (route B): ``encoder_parameter`` is declarative metadata only.

    The registry entry below declares ``encoder_parameter="foo"`` and the
    fixture constructor even accepts ``foo``, yet the CNN path still refuses
    the method: injection stays keyed to the fixed ``encoder`` constructor
    parameter (signature check), never to the declared metadata name.
    """
    name = "fake_encoder_parameter_metadata"
    register_method(
        AlgorithmMetadata(
            name=name,
            paper="Contract-test fixture for issue #45 (route B).",
            family=AlgorithmFamily.CLASSIC_CALIBRATION,
            implementation_status=ImplementationStatus.NATIVE,
            native_architectures=frozenset({"cnn"}),
            input_ndims=frozenset({2, 4}),
            encoder_parameter="foo",
        ),
        _EncoderDeclaredByNameOnly,
    )
    try:
        with pytest.raises(PipelineError, match="declare an 'encoder' constructor parameter"):
            PUPipeline(classifier=name, architecture="cnn")
    finally:
        unregister_method(name)
