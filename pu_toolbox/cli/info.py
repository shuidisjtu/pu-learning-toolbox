"""The ``list-methods`` and ``list-priors`` subcommands."""

from __future__ import annotations

import argparse

from ..core.base import BasePriorEstimator, BasePUClassifier
from ..core.exceptions import RegistryError
from ..registry.builtin_methods import register_all_builtin_methods
from ..registry.registry import get_algorithm, list_algorithms
from ..workflows.pipeline import _missing_required_params

__all__ = ["build_info_parser", "run_list_methods", "run_list_priors"]

# km1/km2 map to KernelMeanPriorEstimator variants in PUPipeline (not
# registry aliases), so they are the only names listed explicitly here.
# Registry names and aliases (recpe, pen_l1, cpe, ...) are collected by
# run_list_priors, keeping the accepted set in sync with the registry.
_PRIOR_NAMES = ("km1", "km2")


def build_info_parser(sub: argparse._SubParsersAction) -> None:
    """Attach ``list-methods`` and ``list-priors`` to *sub*."""
    methods = sub.add_parser("list-methods", help="list registered PU algorithms")
    methods.set_defaults(func=run_list_methods)
    priors = sub.add_parser("list-priors", help="list available class-prior estimators")
    priors.set_defaults(func=run_list_priors)


def run_list_methods(args: argparse.Namespace) -> None:
    """Print a table of registered classifiers with auto-instantiability."""
    register_all_builtin_methods()
    rows: list[tuple[str, str, str, str, str]] = []
    for meta in list_algorithms():
        # Canonical name plus its aliases (e.g. "ldce" for "centroid_pu"):
        # all of them are accepted by --classifier, so list every one.
        for name in (meta.name, *meta.aliases):
            cls = _resolve_class(name)
            if cls is None or not issubclass(cls, BasePUClassifier):
                continue
            auto_inst = "yes" if not _missing_required_params(cls) else "no"
            rows.append(
                (
                    name,
                    meta.family.value,
                    "yes" if meta.requires_class_prior else "no",
                    meta.implementation_status.value,
                    auto_inst,
                )
            )
    rows.sort(key=lambda row: row[0])
    if not rows:
        # Defensive: no registered classifier (e.g. a future api_only-only
        # registry) should print an empty table, not crash on max().
        print("No registered classifiers.")
        return
    # Dynamic Name width so long canonical names and aliases stay aligned.
    name_width = max(len(row[0]) for row in rows) + 2
    print(f"{'Name':<{name_width}}{'Family':<22}{'Prior':<6}{'Status':<8}{'Auto-inst':<10}")
    print("-" * (name_width + 46))
    for name, family, prior, status, auto in rows:
        print(f"{name:<{name_width}}{family:<22}{prior:<6}{status:<8}{auto:<10}")


def run_list_priors(args: argparse.Namespace) -> None:
    """Print the class-prior estimators accepted by ``--prior-estimator``."""
    register_all_builtin_methods()
    names = list(_PRIOR_NAMES)
    for meta in list_algorithms():
        cls = _resolve_class(meta.name)
        if cls is None or not issubclass(cls, BasePriorEstimator):
            continue
        # Canonical name plus its aliases (e.g. "cpe" for
        # class_prior_estimation): every one is accepted by --prior-estimator.
        for name in (meta.name, *meta.aliases):
            if name not in names:
                names.append(name)
    print("Pass one of these to --prior-estimator ('none' disables estimation):")
    for name in names:
        print(f"  {name}")


def _resolve_class(name: str) -> type | None:
    """Resolve a registry name to a class; ``None`` when unregistered."""
    try:
        cls = get_algorithm(name)
    except RegistryError:
        return None
    return cls if isinstance(cls, type) else None
