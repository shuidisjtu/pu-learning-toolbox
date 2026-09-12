# ruff: noqa: N802, N803, N806
"""Unit tests for check_architecture_capability (signature vs capability)."""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.core.base import BasePUClassifier
from pu_toolbox.core.tags import AlgorithmFamily, ImplementationStatus
from pu_toolbox.registry import AlgorithmMetadata, clear_registry, register_method
from pu_toolbox.registry.builtin_methods import register_all_builtin_methods
from pu_toolbox.workflows import PipelineError
from pu_toolbox.workflows._models import check_architecture_capability, cnn_capable_classifier_names


class _Capable(BasePUClassifier):
    native_architectures = frozenset({"cnn"})
    input_ndims = frozenset({2, 4})
    encoder_parameter = "encoder"
    trains_encoder = True

    def __init__(self, *, encoder=None):
        self.encoder = encoder

    def fit(self, X, y_pu, *, class_prior=None, sample_weight=None):
        return self

    def _predict(self, X):
        return np.zeros(len(X))

    def _decision_function(self, X):
        return np.zeros(len(X))


class _SigYesCapNo(BasePUClassifier):
    """Signature declares encoder but capability metadata does not (drift)."""

    def __init__(self, *, encoder=None):
        self.encoder = encoder

    def fit(self, X, y_pu, *, class_prior=None, sample_weight=None):
        return self

    def _predict(self, X):
        return np.zeros(len(X))

    def _decision_function(self, X):
        return np.zeros(len(X))


class _SigNoCapYes(BasePUClassifier):
    """Capability claims cnn but signature has no encoder param (drift)."""

    native_architectures = frozenset({"cnn"})
    input_ndims = frozenset({2, 4})
    encoder_parameter = "encoder"
    trains_encoder = True

    def fit(self, X, y_pu, *, class_prior=None, sample_weight=None):
        return self

    def _predict(self, X):
        return np.zeros(len(X))

    def _decision_function(self, X):
        return np.zeros(len(X))


@pytest.mark.unit
def test_capable_class_passes_cnn_check():
    check_architecture_capability(_Capable, "cnn", "fake")


@pytest.mark.unit
def test_signature_yes_capability_no_raises():
    with pytest.raises(PipelineError, match="mismatch.*capability declaration says"):
        check_architecture_capability(_SigYesCapNo, "cnn", "fake")


@pytest.mark.unit
def test_signature_no_capability_yes_raises():
    with pytest.raises(PipelineError, match="mismatch"):
        check_architecture_capability(_SigNoCapYes, "cnn", "fake")


@pytest.mark.unit
def test_mlp_architecture_never_checked():
    check_architecture_capability(_SigYesCapNo, "mlp", "fake")
    check_architecture_capability(_SigNoCapYes, "mlp", "fake")


def _fake_cnn_meta(name: str, status: ImplementationStatus) -> AlgorithmMetadata:
    return AlgorithmMetadata(
        name=name,
        paper=f"Test paper for {name}",
        family=AlgorithmFamily.CLASSIC_CALIBRATION,
        implementation_status=status,
        native_architectures=frozenset({"cnn"}),
        input_ndims=frozenset({2, 4}),
        encoder_parameter="encoder",
    )


@pytest.mark.unit
class TestCnnCapableClassifierNames:
    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        """Isolate registry state (mirrors tests/test_registry.py)."""
        clear_registry()
        yield
        clear_registry()

    def test_includes_registered_native_cnn_methods(self):
        # Regression (issue #45): the cnn hint silently omitted nnpu once
        # nnPU gained native CNN support; builtin registration must surface it.
        register_all_builtin_methods()
        names = cnn_capable_classifier_names()
        assert "nnpu" in names
        assert "infomax_pu" in names

    def test_reflects_newly_registered_cnn_method(self):
        """The hint derives from the registry, so new CNN-capable methods
        appear automatically without editing error/help copy."""
        register_all_builtin_methods()
        register_method(_fake_cnn_meta("fake_cnn_test", ImplementationStatus.NATIVE))
        assert "fake_cnn_test" in cnn_capable_classifier_names()

    def test_excludes_api_only_cnn_methods(self):
        """trainable_only must keep suggestable methods runnable."""
        register_all_builtin_methods()
        register_method(_fake_cnn_meta("fake_cnn_stub", ImplementationStatus.API_ONLY))
        assert "fake_cnn_stub" not in cnn_capable_classifier_names()
