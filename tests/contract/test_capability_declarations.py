# ruff: noqa: N802, N803, N806
"""Capability-declaration contract tests for registered classifiers.

Invariants from the phase-0 design spec §6: declaration legality, registry
sync, tabular_only derivation, and cross-mechanism consistency with the
constructor-signature check (dual_architecture_plan.md §4.2).
"""

from __future__ import annotations

import pytest

from pu_toolbox.core.base import BasePUClassifier
from pu_toolbox.registry import list_algorithms, register_all_builtin_methods
from pu_toolbox.registry.registry import get_algorithm

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
