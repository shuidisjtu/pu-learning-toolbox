"""Check traditional-PU baseline configs that pin constructor defaults against drift.

Configs carrying ``locks_source_defaults: true`` declare every constructor
default explicitly so the frozen baseline stays reproducible when source
defaults evolve (contract §5).  This gate compares each pinned method against
the live constructor signature; any drift fails the check instead of silently
re-running the baseline with different parameters.

Runner-injected keys are exempt: ``random_state`` (every factory injects the
trial seed), ``class_prior`` (upu/nnpu/pnu inject the scenario prior) and
``flip_probability`` (ldce/kldce inject the real h) — they are required or
overwritten by the runner, never read from the config.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

from pu_toolbox.estimators.classic.elkan_noto import ElkanNotoClassifier
from pu_toolbox.estimators.classic.llsvm import LLSVMClassifier
from pu_toolbox.estimators.risk.kldce import KLDCEClassifier
from pu_toolbox.estimators.risk.ldce import LDCEClassifier
from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier
from pu_toolbox.estimators.risk.pnu import PNUClassifier
from pu_toolbox.estimators.risk.upu import UPUClassifier

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "benchmarks" / "traditional_pu" / "configs"
ESTIMATOR_CLASSES = {
    "elkan_noto": ElkanNotoClassifier,
    "upu": UPUClassifier,
    "nnpu": NonNegativePUClassifier,
    "pnu": PNUClassifier,
    "ldce": LDCEClassifier,
    "kldce": KLDCEClassifier,
    "llsvm": LLSVMClassifier,
}
# Keys the runner always injects (or that are required constructor args with
# no default), so a locked config must not carry them.
RUNNER_INJECTED_KEYS = {"random_state", "class_prior"}


def _constructor_defaults(method: str) -> dict:
    """Constructor parameters with defaults, minus runner-injected keys."""
    signature = inspect.signature(ESTIMATOR_CLASSES[method].__init__)
    defaults = {}
    for name, parameter in signature.parameters.items():
        if name == "self" or parameter.default is inspect.Parameter.empty:
            continue
        if name in RUNNER_INJECTED_KEYS:
            continue
        defaults[name] = parameter.default
    return defaults


def _locked_configs() -> list[tuple[str, dict]]:
    locked = []
    for path in sorted(CONFIGS_DIR.glob("*.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        if config.get("locks_source_defaults"):
            locked.append((path.name, config))
    return locked


def check_config(config_name: str, config: dict) -> list[str]:
    """Return drift issues for one locked config; empty list means pinned."""
    issues: list[str] = []
    for method, pinned in config["methods"].items():
        if method not in ESTIMATOR_CLASSES:
            issues.append(f"{config_name}: unknown method {method!r} in config")
            continue
        defaults = _constructor_defaults(method)
        missing = sorted(set(defaults) - set(pinned))
        extra = sorted(set(pinned) - set(defaults))
        if missing:
            issues.append(
                f"{config_name} methods.{method}: unpinned constructor defaults "
                f"{missing}; pin them explicitly (or add the key to "
                "RUNNER_INJECTED_KEYS if the runner injects it)."
            )
        if extra:
            issues.append(
                f"{config_name} methods.{method}: pinned keys not present in "
                f"the constructor signature {extra}."
            )
        for key in sorted(set(defaults) & set(pinned)):
            if pinned[key] != defaults[key]:
                issues.append(
                    f"{config_name} methods.{method}.{key}: config "
                    f"{pinned[key]!r} != constructor default {defaults[key]!r}."
                )
    return issues


def main() -> int:
    issues: list[str] = []
    locked = _locked_configs()
    if not locked:
        issues.append(
            "no baseline config carries locks_source_defaults: true; the current "
            "toolbox baseline is not pinned against source-default drift."
        )
    for config_name, config in locked:
        issues.extend(check_config(config_name, config))
    if issues:
        print("Baseline config consistency check failed:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    configs = ", ".join(name for name, _ in locked)
    print(
        f"Baseline config consistency check passed: {configs} pin constructor "
        "defaults exactly (runner-injected keys exempt)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
