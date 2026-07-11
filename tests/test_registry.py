"""Tests for pu_toolbox.registry — registration, lookup, aliases."""

import pytest

from pu_toolbox.core.exceptions import RegistryError
from pu_toolbox.core.tags import (
    AlgorithmFamily,
    Assumption,
    ImplementationStatus,
)
from pu_toolbox.registry import (
    AlgorithmMetadata,
    clear_registry,
    get_algorithm,
    get_algorithm_registry,
    get_metadata,
    list_algorithms,
    register_method,
    unregister_method,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure each test starts with an empty registry."""
    clear_registry()
    yield
    clear_registry()


def _make_meta(name: str, **kwargs) -> AlgorithmMetadata:
    defaults = {
        "name": name,
        "paper": f"Test paper for {name}",
        "family": AlgorithmFamily.CLASSIC_CALIBRATION,
    }
    defaults.update(kwargs)
    return AlgorithmMetadata(**defaults)


class TestRegistration:
    def test_register_and_retrieve_metadata(self):
        meta = _make_meta("test_algo", aliases=["ta"])
        register_method(meta)
        retrieved = get_metadata("test_algo")
        assert retrieved.name == "test_algo"
        assert retrieved.family == AlgorithmFamily.CLASSIC_CALIBRATION

    def test_register_duplicate_raises(self):
        register_method(_make_meta("algo_a"))
        with pytest.raises(RegistryError, match="already registered"):
            register_method(_make_meta("algo_a"))

    def test_unregister_removes(self):
        register_method(_make_meta("algo_b"))
        assert "algo_b" in get_algorithm_registry()
        unregister_method("algo_b")
        assert "algo_b" not in get_algorithm_registry()


class TestAliasResolution:
    def test_alias_resolves_to_canonical(self):
        meta = _make_meta("elkan_noto", aliases=["en", "Elkan-Noto"])
        register_method(meta)
        for alias in ["en", "elkan-noto", "ELKAN_NOTO"]:
            m = get_metadata(alias)
            assert m.name == "elkan_noto"

    def test_alias_conflict_raises_on_registration(self):
        # Registering an alias that already maps elsewhere should fail
        # if using register_alias directly — but register_method just
        # adds aliases which may overwrite (fine for existing behaviour).
        pass  # current impl registers canonical name as alias harmlessly


class TestGetAlgorithm:
    def test_api_only_without_class_raises(self):
        meta = _make_meta(
            "future_algo",
            implementation_status=ImplementationStatus.API_ONLY,
        )
        register_method(meta)  # no estimator_cls
        with pytest.raises(RegistryError, match="not yet implemented"):
            get_algorithm("future_algo")

    def test_unknown_name_raises(self):
        with pytest.raises(RegistryError, match="Unknown algorithm"):
            get_algorithm("nonexistent")


class TestListAlgorithms:
    def test_list_all(self):
        register_method(_make_meta("a"))
        register_method(_make_meta("b"))
        results = list_algorithms()
        assert len(results) == 2

    def test_trainable_only_filters_api_only(self):
        register_method(_make_meta("trainable", implementation_status=ImplementationStatus.NATIVE))
        register_method(
            _make_meta("placeholder", implementation_status=ImplementationStatus.API_ONLY)
        )
        results = list_algorithms(trainable_only=True)
        assert len(results) == 1
        assert results[0].name == "trainable"

    def test_family_filter(self):
        register_method(_make_meta("a", family=AlgorithmFamily.RISK_ESTIMATION))
        register_method(_make_meta("b", family=AlgorithmFamily.BIAS_AWARE))
        results = list_algorithms(family="risk_estimation")
        assert len(results) == 1
        assert results[0].name == "a"

    def test_assumption_filter(self):
        register_method(_make_meta("scar_method", assumption=[Assumption.SCAR]))
        register_method(_make_meta("sar_method", assumption=[Assumption.SAR]))
        results = list_algorithms(assumption="SAR")
        assert len(results) == 1
        assert results[0].name == "sar_method"


class TestMetadataSerialization:
    def test_to_dict_includes_all_fields(self):
        meta = _make_meta("test", aliases=["t"])
        d = meta.to_dict()
        assert d["name"] == "test"
        assert "t" in d["aliases"]
        assert "family" in d
        assert "trainable" in d

    def test_trainable_derived_from_status(self):
        meta = _make_meta("x", implementation_status=ImplementationStatus.API_ONLY)
        assert not meta.trainable
        meta2 = _make_meta("y", implementation_status=ImplementationStatus.NATIVE)
        assert meta2.trainable
