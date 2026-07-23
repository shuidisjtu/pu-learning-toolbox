"""Tests for the 15 built-in paper-method registrations."""

import pytest

from pu_toolbox.core.tags import (
    AlgorithmFamily,
    SourceStatus,
)
from pu_toolbox.registry import (
    clear_registry,
    get_algorithm_registry,
    list_algorithms,
    register_all_builtin_methods,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registry()
    yield
    clear_registry()


@pytest.mark.unit
class TestBuiltinRegistration:
    """Smoke tests for the 15 built-in api_only method entries."""

    def test_basic_registers_exactly_15_methods(self):
        n = register_all_builtin_methods()
        assert n == 15
        assert len(get_algorithm_registry()) == 15

    def test_basic_implementation_status_distribution(self):
        register_all_builtin_methods()
        by_status: dict[str, int] = {}
        for meta in get_algorithm_registry().values():
            key = meta.implementation_status.value
            by_status[key] = by_status.get(key, 0) + 1

        # 10 native plus 5 api_only methods.
        assert by_status.get("native", 0) == 10
        assert by_status.get("api_only", 0) == 5

    def test_basic_source_status_distribution(self):
        """Verify counts match docs/resources_optimized.md §2."""
        register_all_builtin_methods()
        by_source: dict[str, int] = {}
        for meta in get_algorithm_registry().values():
            key = meta.source_status.value
            by_source[key] = by_source.get(key, 0) + 1

        assert by_source.get("official_exact", 0) == 8
        assert by_source.get("official_bundle", 0) + by_source.get("official_related", 0) == 3
        assert by_source.get("third_party_only", 0) == 1
        assert by_source.get("not_found", 0) == 3

    def test_basic_family_distribution(self):
        register_all_builtin_methods()
        families: dict[str, int] = {}
        for meta in get_algorithm_registry().values():
            key = meta.family.value
            families[key] = families.get(key, 0) + 1

        assert families.get("class_prior_estimation", 0) == 2  # CPE + ReCPE
        assert families.get("classic_calibration", 0) == 1  # Elkan-Noto
        assert families.get("risk_estimation", 0) == 6  # uPU, nnPU, PNU, Centroid, LLSVM, Dist-PU
        assert families.get("bias_aware", 0) == 2  # PUSB, LBE
        assert families.get("deep_pu", 0) == 4  # Self-PU, InfoMax, WConPU, DGPU

    @pytest.mark.parametrize(
        "name, expected_scar, expected_sar",
        [
            ("elkan_noto", True, False),
            ("nnpu", True, False),
            ("pusb", False, True),
            ("lbe", False, True),
            ("centroid_pu", True, False),  # SCAR only
            ("llsvm", True, True),  # supports both
        ],
    )
    def test_param_assumption_flags(self, name, expected_scar, expected_sar):
        register_all_builtin_methods()
        from pu_toolbox.registry import get_metadata

        meta = get_metadata(name)
        scar = any(a.value == "SCAR" for a in meta.assumption)
        sar = any(a.value == "SAR" for a in meta.assumption)
        assert scar == expected_scar, f"{name}: SCAR expected {expected_scar}"
        assert sar == expected_sar, f"{name}: SAR expected {expected_sar}"

    def test_deterministic_alias_lookup(self):
        register_all_builtin_methods()
        from pu_toolbox.registry import get_metadata

        # Test common aliases
        assert get_metadata("nnPU").name == "nnpu"
        assert get_metadata("en").name == "elkan_noto"
        assert get_metadata("distpu").name == "dist_pu"
        assert get_metadata("wcon_pu").name == "weighted_contrastive_pu"

    def test_edge_list_trainable_only(self):
        """Native implementations are trainable."""
        register_all_builtin_methods()
        trainable = list_algorithms(trainable_only=True)
        assert len(trainable) == 10
        names = {m.name for m in trainable}
        assert names == {
            "elkan_noto", "upu", "nnpu", "pnu", "recpe", "centroid_pu",
            "class_prior_estimation", "dist_pu", "pusb", "lbe",
        }

    def test_basic_list_by_family(self):
        register_all_builtin_methods()
        deep = list_algorithms(family="deep_pu")
        assert len(deep) == 4
        assert all(m.family == AlgorithmFamily.DEEP_PU for m in deep)

    def test_param_list_by_assumption(self):
        register_all_builtin_methods()
        sar_methods = list_algorithms(assumption="SAR")
        # At least PUSB, LBE should match
        names = {m.name for m in sar_methods}
        assert "pusb" in names
        assert "lbe" in names

    def test_basic_every_method_has_paper_title(self):
        register_all_builtin_methods()
        for meta in get_algorithm_registry().values():
            assert meta.paper, f"{meta.name} is missing paper title"
            assert len(meta.paper) > 10, f"{meta.name} paper title too short"

    def test_edge_official_exact_have_upstream_url(self):
        """Every official_exact method must have an upstream URL."""
        register_all_builtin_methods()
        for meta in get_algorithm_registry().values():
            if meta.source_status == SourceStatus.OFFICIAL_EXACT:
                assert meta.upstream_url is not None, (
                    f"{meta.name} is official_exact but missing upstream_url"
                )

    def test_metadata_synced_from_class_attributes(self):
        """After binding, registry metadata matches class-level attributes.

        Only checks fields explicitly declared on the class itself (not
        inherited defaults from the abstract bases).
        """
        from pu_toolbox.core.base import BasePriorEstimator, BasePUClassifier
        from pu_toolbox.registry import get_algorithm, get_metadata

        _BASES = (BasePUClassifier, BasePriorEstimator)

        def _declared_on_class(cls, field_name):
            return any(
                field_name in klass.__dict__
                for klass in cls.__mro__
                if klass not in _BASES and not issubclass(klass, type)
            )

        register_all_builtin_methods()
        for meta in get_algorithm_registry().values():
            if not meta.trainable:
                continue
            cls = get_algorithm(meta.name)
            synced = get_metadata(meta.name)
            for field_name in (
                "family", "implementation_status", "source_status",
                "backend", "maturity", "requires_class_prior",
            ):
                if not _declared_on_class(cls, field_name):
                    continue
                assert getattr(synced, field_name) == getattr(cls, field_name), (
                    f"{meta.name}.{field_name}: registry={getattr(synced, field_name)} "
                    f"!= class={getattr(cls, field_name)}"
                )
            if _declared_on_class(cls, "assumption"):
                assert synced.assumption == list(cls.assumption), (
                    f"{meta.name}.assumption mismatch"
                )
            if _declared_on_class(cls, "scenario"):
                assert synced.scenario == list(cls.scenario), (
                    f"{meta.name}.scenario mismatch"
                )
