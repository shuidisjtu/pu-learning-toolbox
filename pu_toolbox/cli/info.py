"""The ``list-methods`` and ``list-priors`` subcommands."""

from __future__ import annotations

import argparse
import inspect

from ..core.base import BasePriorEstimator, BasePUClassifier
from ..core.exceptions import RegistryError
from ..registry.registry import get_algorithm, list_algorithms

__all__ = ["build_info_parser", "run_list_methods", "run_list_priors"]

# km1/km2 describe KernelMeanPriorEstimator variants, not registry aliases.
_PRIOR_NAMES = ("km1", "km2")


def build_info_parser(sub: argparse._SubParsersAction) -> None:
    """Attach ``list-methods`` and ``list-priors`` to *sub*."""
    methods = sub.add_parser("list-methods", help="list registered PU algorithms")
    methods.set_defaults(func=run_list_methods)
    priors = sub.add_parser("list-priors", help="list available class-prior estimators")
    priors.set_defaults(func=run_list_priors)


def run_list_methods(args: argparse.Namespace) -> None:
    """Print a table of registered classifiers with auto-instantiability."""
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for meta in list_algorithms():
        cls = _resolve_class(meta.name)
        if cls is None or not issubclass(cls, BasePUClassifier):
            continue
        # List canonical names and aliases accepted by the registry.
        for name in (meta.name, *meta.aliases):
            auto_inst = (
                "yes" if not _requires_extra_params(cls, meta.required_prior_parameters) else "no"
            )
            input_dims = ",".join(str(d) for d in sorted(meta.input_ndims))
            # Display order per design doc §4.2 (mlp before cnn) — sorted()
            # would render "cnn,mlp", contradicting the documented order.
            arch = (
                "tabular"
                if meta.is_tabular_only
                else ",".join(arch for arch in ("mlp", "cnn") if arch in meta.native_architectures)
            )
            rows.append(
                (
                    name,
                    meta.family.value,
                    ", ".join(meta.required_prior_parameters) or "-",
                    input_dims,
                    arch,
                    meta.label_semantics,
                    auto_inst,
                )
            )
    rows.sort(key=lambda row: row[0])
    if not rows:
        print("No registered classifiers.")
        return
    # Dynamic Name width so long canonical names and aliases stay aligned.
    name_width = max(len(row[0]) for row in rows) + 2
    prior_title = "Required prior inputs"
    prior_width = max(len(prior_title), *(len(row[2]) for row in rows)) + 2
    print(
        f"{'Name':<{name_width}}{'Family':<22}{prior_title:<{prior_width}}"
        f"{'Input':<7}{'Arch':<11}{'Labels':<8}{'Auto-inst':<10}"
    )
    print("-" * (name_width + prior_width + 58))
    for name, family, prior, input_dims, arch, labels, auto in rows:
        print(
            f"{name:<{name_width}}{family:<22}{prior:<{prior_width}}"
            f"{input_dims:<7}{arch:<11}{labels:<8}{auto:<10}"
        )


def run_list_priors(args: argparse.Namespace) -> None:
    """Print registered prior estimators and kernel-mean variants.

    Each line shows one estimator: the canonical name followed by its
    aliases (e.g. ``class_prior_estimation (aliases: cpe, pe [deprecated], pen_l1)``)
    so users can tell estimators apart instead of seeing 8 flat names.
    """
    entries: dict[str, list[str]] = {name: [] for name in _PRIOR_NAMES}
    for meta in list_algorithms():
        cls = _resolve_class(meta.name)
        if cls is None or not issubclass(cls, BasePriorEstimator):
            continue
        entries[meta.name] = [
            f"{alias} [deprecated]" if alias in meta.deprecated_aliases else alias
            for alias in meta.aliases
        ]
    print("Registered prior estimators and KernelMeanPriorEstimator variants (km1, km2):")
    for name in sorted(entries):
        aliases = entries[name]
        if aliases:
            print(f"  {name} (aliases: {', '.join(aliases)})")
        else:
            print(f"  {name}")


def _requires_extra_params(cls: type, prior_parameters: tuple[str, ...] = ()) -> bool:
    """Whether a constructor requires settings beyond a prior and random seed."""
    return any(
        name not in {"self", "random_state", *prior_parameters}
        and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and param.default is inspect.Parameter.empty
        for name, param in inspect.signature(cls.__init__).parameters.items()
    )


def _resolve_class(name: str) -> type | None:
    """Resolve a registry name to a class; ``None`` when unregistered."""
    try:
        cls = get_algorithm(name)
    except RegistryError:
        return None
    return cls if isinstance(cls, type) else None
