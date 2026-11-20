"""Tests for the built-in paper-method registrations.

Counts and distributions are derived from the registry itself (the single
source of truth), so registering a new method requires no bookkeeping
updates here. Per-method source status is recorded in each method card's
"源码状态与复现风险" section (docs/research/method_cards/), which is
reviewed manually rather than parsed here.
"""

from contextlib import nullcontext
from inspect import isabstract
from pathlib import Path

import pytest

from pu_toolbox.core.tags import (
    AlgorithmFamily,
    Assumption,
    Backend,
    Maturity,
    Scenario,
    SourceStatus,
)
from pu_toolbox.registry import (
    clear_registry,
    get_algorithm,
    get_algorithm_registry,
    get_metadata,
    list_algorithms,
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
def test_basic_registry_declares_distinct_prior_consumers_without_claiming_actual_use():
    kernel = get_metadata("pusb_kernel")
    assert kernel.required_prior_parameters == ("training_table_prior",)
    assert {use.parameter for use in kernel.prior_consumers} == {
        "training_table_prior",
        "risk_prior",
    }
    assert get_metadata("llsvm").required_prior_parameters == ()
    assert get_metadata("pan").prior_consumers == ()
    assert (
        not {"requires_class_prior", "class_prior_target", "risk_class_prior_target"}
        & kernel.to_dict().keys()
    )


@pytest.mark.unit
class TestBuiltinRegistration:
    """Invariant checks for the built-in method entries."""

    def test_basic_registration_is_consistent(self):
        from pu_toolbox.registry.builtin_methods import _BUILTIN

        assert len(_BUILTIN) == len(get_algorithm_registry())
        assert get_metadata("nnPU") is get_metadata("nnpu")

    def test_basic_every_builtin_resolves_to_a_concrete_class(self):
        from pu_toolbox.registry.builtin_methods import _BUILTIN

        assert len(list_algorithms()) == len(_BUILTIN)
        for meta in list_algorithms():
            assert not isabstract(get_algorithm(meta.name))

    def test_basic_family_distribution(self):
        """All five algorithm families must be present (counts are free)."""
        families = {m.family.value for m in get_algorithm_registry().values()}
        assert families >= _KNOWN_FAMILIES, f"missing families: {_KNOWN_FAMILIES - families}"

    def test_semantic_assumption_anchors(self):
        """Anchors: selection-biased methods are SAR-only, classic ones SCAR-only."""
        pusb = get_metadata("pusb")
        assert [a.value for a in pusb.assumption] == ["SAR"]
        elkan_noto = get_metadata("elkan_noto")
        assert [a.value for a in elkan_noto.assumption] == ["SCAR"]

    def test_all_aliases_resolve_to_canonical(self):
        """Every alias of every entry resolves back to its canonical name."""
        for meta in get_algorithm_registry().values():
            for alias in meta.aliases:
                warning = FutureWarning if alias in meta.deprecated_aliases else None
                with pytest.warns(warning) if warning else nullcontext():
                    resolved = get_metadata(alias)
                assert resolved.name == meta.name, (
                    f"alias {alias!r} resolves to {resolved.name}, expected {meta.name}"
                )
                with pytest.warns(warning) if warning else nullcontext():
                    assert get_algorithm(alias) is get_algorithm(meta.name), (
                        f"alias {alias!r} resolves to a different class than {meta.name}"
                    )

    def test_basic_ldce_kldce_resolve_to_distinct_classes(self):
        """kldce must resolve to the kernelized class, not the linear LDCE.

        Regression guard: kldce used to be aliased to ``centroid_pu``,
        silently resolving ``get_algorithm("kldce")`` to LDCEClassifier.
        """
        from pu_toolbox.estimators.risk.kldce import KLDCEClassifier
        from pu_toolbox.estimators.risk.ldce import LDCEClassifier

        assert get_algorithm("kldce") is KLDCEClassifier
        assert get_algorithm("ldce") is LDCEClassifier
        assert get_algorithm("centroid_pu") is LDCEClassifier
        assert get_algorithm("kernelized_ldce") is KLDCEClassifier
        assert get_metadata("kldce").name == "kldce"
        assert "kldce" not in get_metadata("centroid_pu").aliases

    def test_basic_list_by_family(self):
        """Family filter is consistent with the registry (counts free)."""
        for family in _KNOWN_FAMILIES:
            listed = list_algorithms(family=family)
            expected = [m for m in get_algorithm_registry().values() if m.family.value == family]
            assert len(listed) == len(expected), f"family={family}"

    def test_param_list_by_assumption(self):
        """Assumption filter matches the registry's SAR-tagged methods."""
        sar_methods = list_algorithms(assumption="SAR")
        expected = {
            m.name
            for m in get_algorithm_registry().values()
            if any(a.value == "SAR" for a in m.assumption)
        }
        assert {m.name for m in sar_methods} == expected

    def test_basic_every_method_has_paper_title(self):
        for meta in get_algorithm_registry().values():
            assert meta.paper, f"{meta.name} is missing paper title"
            assert len(meta.paper) > 10, f"{meta.name} paper title too short"

    def test_edge_official_exact_have_upstream_url(self):
        """Every official_exact method must have an upstream URL."""
        for meta in get_algorithm_registry().values():
            if meta.source_status == SourceStatus.OFFICIAL_EXACT:
                assert meta.upstream_url is not None, (
                    f"{meta.name} is official_exact but missing upstream_url"
                )

    def test_metadata_synced_from_class_attributes(self):
        """After registration, registry metadata matches class-level attributes.

        Only checks fields explicitly declared on the class itself (not
        inherited defaults from the abstract bases).
        """
        from pu_toolbox.core.base import BasePriorEstimator, BasePUClassifier

        _bases = (BasePUClassifier, BasePriorEstimator)

        def _declared_on_class(cls, field_name):
            return any(
                field_name in klass.__dict__
                for klass in cls.__mro__
                if klass not in _bases and not issubclass(klass, type)
            )

        for meta in get_algorithm_registry().values():
            cls = get_algorithm(meta.name)
            synced = get_metadata(meta.name)
            for field_name in (
                "family",
                "source_status",
                "backend",
                "maturity",
                "prior_consumers",
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

    def test_static_entries_do_not_redeclare_class_fields(self):
        """What a `_BUILTIN` literal is for: exactly what the class does not say.

        This replaces the older consistency check, which asserted the literals
        *equal* the class values.  That check had to exist because the literals
        duplicated live data -- `upu` sat with `False` in the entry while the
        class said `True`, silently papered over for months.  Removing the
        duplicate removes that failure mode; the assertion is now inverted, so
        re-introducing one fails here instead of drifting quietly.

        The source is parsed rather than the metadata read: a deleted keyword
        still answers `getattr` with the dataclass default, so attribute access
        cannot tell "omitted" from "declared".

        Two clauses:
        1. a field the class declares must NOT appear in the literal;
        2. a field the class does not declare must appear -- and for the entry
           whose literals are authoritative (`class_prior_estimation`), carry
           the ruled value rather than a fresh copy of itself.
        """
        import ast

        from pu_toolbox.core.base import BasePriorEstimator, BasePUClassifier
        from pu_toolbox.registry.registry import _SYNC_FIELDS

        entry_fields = (
            "family",
            "assumption",
            "scenario",
            "source_status",
            "backend",
            "maturity",
        )
        class_only_fields = tuple(f for f in _SYNC_FIELDS if f not in entry_fields)

        source = (PROJECT_ROOT / "pu_toolbox/registry/builtin_methods.py").read_text(
            encoding="utf-8"
        )
        written: dict[str, set[str]] = {}
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "AlgorithmMetadata":
                name = next(k.value.value for k in node.keywords if k.arg == "name")
                written[name] = {k.arg for k in node.keywords}

        bases = (BasePUClassifier, BasePriorEstimator)

        def _declared_on_class(cls, field_name):
            return any(
                field_name in klass.__dict__
                for klass in cls.__mro__
                if klass not in bases and not issubclass(klass, type)
            )

        assert written, "the AST parse found no entries"
        for name, fields in written.items():
            for field_name in class_only_fields:
                assert field_name not in fields, (
                    f"{name}.{field_name}: belongs to the class/dataclass, not to an entry literal"
                )
            cls = get_algorithm(name)
            for field_name in entry_fields:
                # Exactly one of the two may hold: the class declares the field
                # (so a literal is dead weight) XOR the literal carries it (the
                # class does not say it, so the literal is the source).
                assert _declared_on_class(cls, field_name) != (field_name in fields), (
                    f"{name}.{field_name}: "
                    + (
                        "the class declares it, so a literal here is dead weight"
                        if _declared_on_class(cls, field_name)
                        else "the class does not declare it, so the literal is the source"
                    )
                )

        # The CPE entry supplies capabilities its class does not declare.
        cpe = get_metadata("class_prior_estimation")
        assert cpe.family == AlgorithmFamily.CLASS_PRIOR_ESTIMATION
        assert cpe.required_prior_parameters == ()
        assert cpe.source_status == SourceStatus.OFFICIAL_RELATED
        assert cpe.backend == Backend.NUMPY
        assert cpe.maturity == Maturity.STABLE
        assert cpe.scenario == [Scenario.SINGLE_TRAINING_SET, Scenario.CASE_CONTROL]
        assert cpe.assumption == [Assumption.SCAR]
        # pusb / lbe: same shape (class inherits the base default, which the sync
        # deliberately excludes), so their literals are live too.
        assert get_metadata("pusb").required_prior_parameters == ()
        assert get_metadata("lbe").required_prior_parameters == ()

    def test_basic_every_method_has_explicit_training_cost(self):
        """All entries carry an explicit training-cost level (no UNKNOWN)."""
        from pu_toolbox.core.tags import TrainingCost

        for meta in get_algorithm_registry().values():
            assert meta.training_cost != TrainingCost.UNKNOWN, (
                f"{meta.name} is missing an explicit training_cost"
            )

    def test_basic_heavy_fixed_epoch_methods_are_high_cost(self):
        """HIGH cost: long-epoch solvers, PUET's 100-tree CPU forest,
        PULDA's two fixed 60-epoch stages, PUSB kernel (full grid CV + refit),
        and the multi-stage deep solvers Robust-PU (nnPU warm-up plus 20
        self-paced episodes), Split-PU (teacher/temporary/student per round)
        and LaGAM (a second-order meta-gradient every epoch), PAN (two-network
        adversarial updates), PULNS (sequential policy/probe/classifier), GenPU
        (five-network GAN plus PN), Holistic-PU (warmup/trajectory/pseudo-label
        training), P3Mix (per-epoch pool refresh and teacher variant), and
        TEDn (warm-start plus iterative estimate-discard); short-epoch deep
        methods stay MEDIUM.

        Pinned so a re-classification is a deliberate edit rather than a side
        effect of registering a method.
        """
        from pu_toolbox.core.tags import TrainingCost

        high = {
            m.name
            for m in get_algorithm_registry().values()
            if m.training_cost == TrainingCost.HIGH
        }
        assert high == {
            "llsvm",
            "infomax_pu",
            "weighted_contrastive_pu",
            "pusb_kernel",
            "gradpu",
            "puet",
            "pulda",
            "robust_pu",
            "lagam",
            "split_pu",
            "pan",
            "pulns",
            "genpu",
            "holistic_pu",
            "p3mix",
            "tedn",
        }
