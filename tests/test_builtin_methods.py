"""Tests for the 15 built-in paper-method registrations."""

import pytest

from pu_toolbox.core.tags import (
    AlgorithmFamily,
    ImplementationStatus,
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


class TestBuiltinRegistration:
    """Smoke tests for the 15 built-in api_only method entries."""

    def test_registers_exactly_15_methods(self):
        n = register_all_builtin_methods()
        assert n == 15
        assert len(get_algorithm_registry()) == 15

    def test_all_are_api_only(self):
        register_all_builtin_methods()
        for meta in get_algorithm_registry().values():
            assert meta.implementation_status == ImplementationStatus.API_ONLY, (
                f"{meta.name} should be api_only, got {meta.implementation_status}"
            )
            assert not meta.trainable, f"{meta.name} should not be trainable"

    def test_source_status_distribution(self):
        """Verify counts match docs/resources_optimized.md §2."""
        register_all_builtin_methods()
        by_source: dict[str, int] = {}
        for meta in get_algorithm_registry().values():
            key = meta.source_status.value
            by_source[key] = by_source.get(key, 0) + 1

        assert by_source.get("official_exact", 0) == 8
        assert by_source.get("official_bundle", 0) + by_source.get(
            "official_related", 0
        ) == 3
        assert by_source.get("third_party_only", 0) == 1
        assert by_source.get("not_found", 0) == 3

    def test_family_distribution(self):
        register_all_builtin_methods()
        families: dict[str, int] = {}
        for meta in get_algorithm_registry().values():
            key = meta.family.value
            families[key] = families.get(key, 0) + 1

        assert families.get("class_prior_estimation", 0) == 2   # CPE + ReCPE
        assert families.get("classic_calibration", 0) == 1      # Elkan-Noto
        assert families.get("risk_estimation", 0) == 6          # uPU, nnPU, PNU, Centroid, LLSVM, Dist-PU
        assert families.get("bias_aware", 0) == 2               # PUSB, LBE
        assert families.get("deep_pu", 0) == 4                  # Self-PU, InfoMax, WConPU, DGPU

    @pytest.mark.parametrize(
        "name, expected_scar, expected_sar",
        [
            ("elkan_noto", True, False),
            ("nnpu", True, False),
            ("pusb", False, True),
            ("lbe", False, True),
            ("centroid_pu", True, True),  # supports both
            ("llsvm", True, True),         # supports both
        ],
    )
    def test_assumption_flags(self, name, expected_scar, expected_sar):
        register_all_builtin_methods()
        from pu_toolbox.registry import get_metadata
        meta = get_metadata(name)
        scar = any(a.value == "SCAR" for a in meta.assumption)
        sar = any(a.value in ("SAR", "instance_dependent") for a in meta.assumption)
        assert scar == expected_scar, f"{name}: SCAR expected {expected_scar}"
        assert sar == expected_sar, f"{name}: SAR expected {expected_sar}"

    def test_lookup_by_alias(self):
        register_all_builtin_methods()
        from pu_toolbox.registry import get_metadata
        # Test common aliases
        assert get_metadata("nnPU").name == "nnpu"
        assert get_metadata("en").name == "elkan_noto"
        assert get_metadata("distpu").name == "dist_pu"
        assert get_metadata("wcon_pu").name == "weighted_contrastive_pu"

    def test_list_trainable_only_returns_empty(self):
        """None of the built-ins are trainable yet."""
        register_all_builtin_methods()
        trainable = list_algorithms(trainable_only=True)
        assert len(trainable) == 0

    def test_list_by_family(self):
        register_all_builtin_methods()
        deep = list_algorithms(family="deep_pu")
        assert len(deep) == 4
        assert all(m.family == AlgorithmFamily.DEEP_PU for m in deep)

    def test_list_by_assumption(self):
        register_all_builtin_methods()
        sar_methods = list_algorithms(assumption="SAR")
        # At least PUSB, LBE should match
        names = {m.name for m in sar_methods}
        assert "pusb" in names
        assert "lbe" in names

    def test_every_method_has_paper_title(self):
        register_all_builtin_methods()
        for meta in get_algorithm_registry().values():
            assert meta.paper, f"{meta.name} is missing paper title"
            assert len(meta.paper) > 10, f"{meta.name} paper title too short"

    def test_official_exact_have_upstream_url(self):
        """Every official_exact method must have an upstream URL."""
        register_all_builtin_methods()
        for meta in get_algorithm_registry().values():
            if meta.source_status == SourceStatus.OFFICIAL_EXACT:
                assert meta.upstream_url is not None, (
                    f"{meta.name} is official_exact but missing upstream_url"
                )
