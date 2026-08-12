"""Tests for the built-in paper-method registrations.

Counts and distributions are derived from the registry itself (the single
source of truth) plus the stats table in ``docs/dev/resources.md``, so
registering a new method requires no bookkeeping updates here.
"""

import re
from pathlib import Path

import pytest

from pu_toolbox.core.tags import (
    ImplementationStatus,
    SourceStatus,
)
from pu_toolbox.registry import (
    clear_registry,
    get_algorithm_registry,
    get_metadata,
    list_algorithms,
    register_all_builtin_methods,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_KNOWN_FAMILIES = {
    "class_prior_estimation",
    "classic_calibration",
    "risk_estimation",
    "bias_aware",
    "deep_pu",
}


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registry()
    yield
    clear_registry()


@pytest.mark.unit
class TestBuiltinRegistration:
    """Invariant checks for the built-in method entries."""

    def test_basic_registration_is_consistent(self):
        """Registration count equals registry size (no magic number)."""
        n = register_all_builtin_methods()
        assert n == len(get_algorithm_registry())

    def test_basic_implementation_status_distribution(self):
        """Every built-in method must be natively implemented (no placeholders)."""
        register_all_builtin_methods()
        for meta in get_algorithm_registry().values():
            assert meta.implementation_status == ImplementationStatus.NATIVE, (
                f"{meta.name} must be NATIVE, got {meta.implementation_status}"
            )

    def test_basic_source_status_matches_docs(self):
        """Registry source-status counts must match docs/dev/resources.md.

        The doc stats table is the rendered view of the registry; this test
        makes the pair drift-free without a separate generation script.
        """
        register_all_builtin_methods()
        by_source: dict[str, int] = {}
        for meta in get_algorithm_registry().values():
            key = meta.source_status.value
            by_source[key] = by_source.get(key, 0) + 1

        doc = (PROJECT_ROOT / "docs" / "dev" / "resources.md").read_text(encoding="utf-8")
        expected_single: dict[str, int] = {}
        combined_total: int | None = None
        for line in doc.splitlines():
            m = re.match(r"\|\s*`([^`]+)`\s*(?:/\s*`([^`]+)`)?\s*\|\s*(\d+)\s*\|", line)
            if not m:
                continue
            count = int(m.group(3))
            if m.group(2):
                # combined row: `official_bundle` / `official_related` | N
                # means the two states sum to N
                combined_total = count
            else:
                expected_single[m.group(1)] = count

        for state, count in expected_single.items():
            assert by_source.get(state, 0) == count, f"source_status {state!r}"
        assert (
            by_source.get("official_bundle", 0) + by_source.get("official_related", 0)
            == combined_total
        ), "official_bundle + official_related total"

    def test_basic_family_distribution(self):
        """All five algorithm families must be present (counts are free)."""
        register_all_builtin_methods()
        families = {m.family.value for m in get_algorithm_registry().values()}
        assert families >= _KNOWN_FAMILIES, f"missing families: {_KNOWN_FAMILIES - families}"

    def test_semantic_assumption_anchors(self):
        """Anchors: selection-biased methods are SAR-only, classic ones SCAR-only."""
        register_all_builtin_methods()
        pusb = get_metadata("pusb")
        assert [a.value for a in pusb.assumption] == ["SAR"]
        elkan_noto = get_metadata("elkan_noto")
        assert [a.value for a in elkan_noto.assumption] == ["SCAR"]

    def test_all_aliases_resolve_to_canonical(self):
        """Every alias of every entry resolves back to its canonical name."""
        register_all_builtin_methods()
        from pu_toolbox.registry import get_algorithm

        for meta in get_algorithm_registry().values():
            for alias in meta.aliases:
                resolved = get_metadata(alias)
                assert resolved.name == meta.name, (
                    f"alias {alias!r} resolves to {resolved.name}, expected {meta.name}"
                )
                assert get_algorithm(alias) is get_algorithm(meta.name), (
                    f"alias {alias!r} resolves to a different class than {meta.name}"
                )

    def test_edge_list_trainable_only(self):
        """Native implementations are trainable (set derived, no literal list)."""
        register_all_builtin_methods()
        trainable = list_algorithms(trainable_only=True)
        expected = {m.name for m in get_algorithm_registry().values() if m.trainable}
        assert {m.name for m in trainable} == expected
        assert trainable  # registry must never be empty

    def test_basic_ldce_kldce_resolve_to_distinct_classes(self):
        """kldce must resolve to the kernelized class, not the linear LDCE.

        Regression guard: kldce used to be aliased to ``centroid_pu``,
        silently resolving ``get_algorithm("kldce")`` to LDCEClassifier.
        """
        from pu_toolbox.estimators.risk.kldce import KLDCEClassifier
        from pu_toolbox.estimators.risk.ldce import LDCEClassifier
        from pu_toolbox.registry import get_algorithm, get_metadata

        register_all_builtin_methods()
        assert get_algorithm("kldce") is KLDCEClassifier
        assert get_algorithm("ldce") is LDCEClassifier
        assert get_algorithm("centroid_pu") is LDCEClassifier
        assert get_algorithm("kernelized_ldce") is KLDCEClassifier
        assert get_metadata("kldce").name == "kldce"
        assert "kldce" not in get_metadata("centroid_pu").aliases

    def test_basic_list_by_family(self):
        """Family filter is consistent with the registry (counts free)."""
        register_all_builtin_methods()
        for family in _KNOWN_FAMILIES:
            listed = list_algorithms(family=family)
            expected = [m for m in get_algorithm_registry().values() if m.family.value == family]
            assert len(listed) == len(expected), f"family={family}"

    def test_param_list_by_assumption(self):
        """Assumption filter matches the registry's SAR-tagged methods."""
        register_all_builtin_methods()
        sar_methods = list_algorithms(assumption="SAR")
        expected = {
            m.name
            for m in get_algorithm_registry().values()
            if any(a.value == "SAR" for a in m.assumption)
        }
        assert {m.name for m in sar_methods} == expected

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

        _bases = (BasePUClassifier, BasePriorEstimator)

        def _declared_on_class(cls, field_name):
            return any(
                field_name in klass.__dict__
                for klass in cls.__mro__
                if klass not in _bases and not issubclass(klass, type)
            )

        register_all_builtin_methods()
        for meta in get_algorithm_registry().values():
            if not meta.trainable:
                continue
            cls = get_algorithm(meta.name)
            synced = get_metadata(meta.name)
            for field_name in (
                "family",
                "implementation_status",
                "source_status",
                "backend",
                "maturity",
                "requires_class_prior",
            ):
                if not _declared_on_class(cls, field_name):
                    continue
                assert getattr(synced, field_name) == getattr(cls, field_name), (
                    f"{meta.name}.{field_name}: registry={getattr(synced, field_name)} "
                    f"!= class={getattr(cls, field_name)}"
                )
            if _declared_on_class(cls, "assumption"):
                assert synced.assumption == list(cls.assumption), f"{meta.name}.assumption mismatch"
            if _declared_on_class(cls, "scenario"):
                assert synced.scenario == list(cls.scenario), f"{meta.name}.scenario mismatch"

    def test_static_entries_match_class_attributes(self):
        """_BUILTIN entry literals must match class metadata BEFORE sync.

        The test above reads the registry AFTER binding, where
        _sync_class_metadata_to_registry has already overwritten entry
        fields with class values — so static drift in the literals is
        invisible to it (upu was False in the entry while the class
        says True, silently papered over for months).  This test
        snapshots the entry literals first, then registers, then
        compares against the bound classes.
        """
        from pu_toolbox.core.base import BasePriorEstimator, BasePUClassifier
        from pu_toolbox.registry import get_algorithm
        from pu_toolbox.registry.builtin_methods import _BUILTIN

        _bases = (BasePUClassifier, BasePriorEstimator)
        sync_fields = (
            "family",
            "assumption",
            "scenario",
            "requires_class_prior",
            "implementation_status",
            "source_status",
            "backend",
            "maturity",
        )

        def _declared_on_class(cls, field_name):
            return any(
                field_name in klass.__dict__
                for klass in cls.__mro__
                if klass not in _bases and not issubclass(klass, type)
            )

        # Snapshot BEFORE registering: register_method + the sync step
        # overwrite the metadata objects in place, so reading them after
        # registration would see the synced values, not the literals.
        snapshots = {
            meta.name: {
                field: (
                    list(getattr(meta, field))
                    if isinstance(getattr(meta, field), tuple | list)
                    else getattr(meta, field)
                )
                for field in sync_fields
            }
            for meta in _BUILTIN
        }

        register_all_builtin_methods()
        for name, snapshot in snapshots.items():
            cls = get_algorithm(name)  # api_only entries would fail here loudly
            for field, entry_value in snapshot.items():
                if not _declared_on_class(cls, field):
                    continue  # not synced from class; the literal is authoritative
                cls_value = getattr(cls, field)
                if isinstance(cls_value, tuple | list):
                    assert list(cls_value) == entry_value, (
                        f"{name}.{field}: entry={entry_value} != class={list(cls_value)}"
                    )
                else:
                    assert cls_value == entry_value, (
                        f"{name}.{field}: entry={entry_value} != class={cls_value}"
                    )

    def test_basic_every_method_has_explicit_training_cost(self):
        """All entries carry an explicit training-cost level (no UNKNOWN)."""
        from pu_toolbox.core.tags import TrainingCost

        register_all_builtin_methods()
        for meta in get_algorithm_registry().values():
            assert meta.training_cost != TrainingCost.UNKNOWN, (
                f"{meta.name} is missing an explicit training_cost"
            )

    def test_basic_heavy_fixed_epoch_methods_are_high_cost(self):
        """HIGH cost: LLSVM SGD, WConPU, InfoMax PU (fixed long-epoch
        solvers) and PUSB kernel (full sigma x reg grid CV + refit);
        short-epoch deep methods stay MEDIUM."""
        from pu_toolbox.core.tags import TrainingCost

        register_all_builtin_methods()
        high = {
            m.name
            for m in get_algorithm_registry().values()
            if m.training_cost == TrainingCost.HIGH
        }
        assert high == {"llsvm", "infomax_pu", "weighted_contrastive_pu", "pusb_kernel"}
