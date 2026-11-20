"""Tests for pu_toolbox.registry — registration, lookup, aliases."""

import pytest

from pu_toolbox.core.base import BasePUClassifier
from pu_toolbox.core.exceptions import RegistryError, ValidationError
from pu_toolbox.core.tags import AlgorithmFamily, Assumption
from pu_toolbox.prior.pen_l1 import ClassPriorEstimator
from pu_toolbox.registry import (
    AlgorithmMetadata,
    builtin_methods,
    clear_registry,
    get_algorithm,
    get_algorithm_registry,
    get_metadata,
    list_algorithms,
    register_method,
    unregister_method,
)


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    monkeypatch.setattr(builtin_methods, "_BUILTIN", {})
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


@pytest.mark.unit
class TestRegistration:
    def test_basic_register_and_retrieve_metadata(self):
        meta = _make_meta("test_algo", aliases=["ta"])
        register_method(meta, ClassPriorEstimator)
        retrieved = get_metadata("test_algo")
        assert retrieved.name == "test_algo"
        assert retrieved.family == AlgorithmFamily.CLASSIC_CALIBRATION
        assert get_algorithm("TA") is ClassPriorEstimator

    def test_param_register_duplicate_raises(self):
        register_method(_make_meta("algo_a"), ClassPriorEstimator)
        with pytest.raises(RegistryError, match="already registered"):
            register_method(_make_meta("algo_a"), ClassPriorEstimator)

    @pytest.mark.parametrize("estimator_cls", [None, BasePUClassifier, object])
    def test_param_unusable_class_is_rejected_before_registration(self, estimator_cls):
        meta = _make_meta("future_algo", aliases=["future"])
        with pytest.raises(RegistryError, match="concrete.*subclass"):
            register_method(meta, estimator_cls)
        assert not get_algorithm_registry()
        with pytest.raises(RegistryError, match="Unknown algorithm"):
            get_metadata("future")

    def test_basic_unregister_removes(self):
        register_method(_make_meta("algo_b"), ClassPriorEstimator)
        assert "algo_b" in get_algorithm_registry()
        unregister_method("algo_b")
        assert "algo_b" not in get_algorithm_registry()


@pytest.mark.unit
class TestAliasResolution:
    def test_basic_alias_resolves_to_canonical(self):
        meta = _make_meta("elkan_noto", aliases=["en", "Elkan-Noto"])
        register_method(meta, ClassPriorEstimator)
        for alias in ["en", "elkan-noto", "ELKAN_NOTO"]:
            m = get_metadata(alias)
            assert m.name == "elkan_noto"

    def test_deterministic_alias_lookup_is_case_insensitive(self):
        """Alias resolution normalises case for lookup."""
        meta = _make_meta("Elkan_Noto", aliases=["en", "Elkan-Noto"])
        register_method(meta, ClassPriorEstimator)
        assert list(get_algorithm_registry()) == ["elkan_noto"]
        assert get_metadata("EN").aliases == ["en", "elkan-noto"]
        for alias in ["EN", "en", "Elkan-Noto", "elkan-noto", "ELKAN_NOTO"]:
            m = get_metadata(alias)
            assert m.name == "elkan_noto", f"alias {alias!r} failed"

    @pytest.mark.parametrize(
        ("name", "aliases", "conflict"),
        [
            ("method_b", ["new_alias", "SHARED_ALIAS"], "shared_alias"),
            ("SHARED_ALIAS", ["new_alias"], "shared_alias"),
            ("method_b", ["new_alias", "METHOD_A"], "method_a"),
            ("method_b", ["new_alias", "NEW_ALIAS"], "new_alias"),
            ("method_b", ["new_alias", "METHOD_B"], "method_b"),
        ],
    )
    def test_param_name_conflicts_leave_registration_unchanged(self, name, aliases, conflict):
        existing = _make_meta("method_a", aliases=["shared_alias"])
        register_method(existing, ClassPriorEstimator)

        with pytest.raises(RegistryError, match=f"(?i){conflict}"):
            register_method(_make_meta(name, aliases=aliases), ClassPriorEstimator)

        assert get_algorithm_registry() == {"method_a": existing}
        assert get_metadata("shared_alias") is existing
        with pytest.raises(RegistryError, match="Unknown algorithm"):
            get_metadata("new_alias")
        register_method(_make_meta("method_b", aliases=["new_alias"]), ClassPriorEstimator)
        assert get_metadata("new_alias").name == "method_b"

    def test_edge_deprecated_alias_warns_and_must_be_registered(self):
        """Deprecated aliases resolve compatibly while guiding migration."""
        with pytest.raises(ValidationError, match="subset of aliases"):
            _make_meta("invalid", deprecated_aliases=["old"])

        meta = _make_meta("current", aliases=["Old"], deprecated_aliases=["OLD"])
        register_method(meta, ClassPriorEstimator)
        with pytest.warns(FutureWarning, match="use 'current' instead"):
            assert get_metadata("OLD").name == "current"


@pytest.mark.unit
class TestGetAlgorithm:
    def test_param_unknown_name_raises(self):
        with pytest.raises(RegistryError, match="Unknown algorithm"):
            get_algorithm("nonexistent")


@pytest.mark.unit
class TestListAlgorithms:
    def test_basic_list_all_counts(self):
        register_method(_make_meta("a"), ClassPriorEstimator)
        register_method(_make_meta("b"), ClassPriorEstimator)
        results = list_algorithms()
        assert len(results) == 2

    def test_basic_family_filter(self):
        register_method(
            _make_meta("a", family=AlgorithmFamily.RISK_ESTIMATION), ClassPriorEstimator
        )
        register_method(_make_meta("b", family=AlgorithmFamily.BIAS_AWARE), ClassPriorEstimator)
        results = list_algorithms(family="risk_estimation")
        assert len(results) == 1
        assert results[0].name == "a"

    def test_basic_assumption_filter(self):
        register_method(
            _make_meta("scar_method", assumption=[Assumption.SCAR]), ClassPriorEstimator
        )
        register_method(_make_meta("sar_method", assumption=[Assumption.SAR]), ClassPriorEstimator)
        results = list_algorithms(assumption="SAR")
        assert len(results) == 1
        assert results[0].name == "sar_method"


@pytest.mark.unit
class TestMetadataSerialization:
    def test_basic_to_dict_includes_all_fields(self):
        meta = _make_meta("test", aliases=["t"])
        d = meta.to_dict()
        assert d["name"] == "test"
        assert "t" in d["aliases"]
        assert "family" in d
        assert "maturity" in d
        assert "training_cost" in d
        assert d["training_cost"] == "unknown"  # enum serialized to .value
