#!/usr/bin/env python3
"""P4.1 central recipe registry gate: schema, protocol basis and digest.

Validates the frozen registry against the loaded survey protocol and recomputes
its digest, so that "which recipe did this survey adopt" is answered by a file
the repository can check rather than by a reviewer's memory.

Staging
-------
The registry JSON is a *freeze-stage* deliverable (plan §11): until it lands,
this gate has nothing to read.  A missing file is therefore reported as such and
exits 0 -- but it says so on stdout rather than printing a pass, because a gate
that reads green while checking nothing is how a gap disappears from a report.
The moment the file appears the run becomes enforcing, with no flag to flip.

Usage::

    uv run python scripts/check_survey_recipe_registry.py
    uv run python scripts/check_survey_recipe_registry.py --registry path/to/registry.json

Exit 0 when the registry is absent, or present and free of error-severity
findings; 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Where the frozen registry will live (plan §4).  Named here once so the gate
#: and the freeze-stage delivery cannot disagree about the path.
DEFAULT_REGISTRY = PROJECT_ROOT / "pu_toolbox" / "experiment" / "survey_recipe_registry_v1.json"


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="registry JSON to validate (defaults to the freeze-stage path)",
    )
    args = parser.parse_args(argv)

    registry_path: Path = args.registry
    if not registry_path.exists():
        print(
            f"Recipe registry not materialized yet: {registry_path}\n"
            "Nothing was validated.  The registry is a freeze-stage deliverable "
            "(plan §11); this gate turns enforcing as soon as the file exists."
        )
        return 0

    from pu_toolbox.experiment.survey_protocol import load_protocol
    from pu_toolbox.experiment.survey_recipe_registry import (
        canonical_digest,
        load_registry,
        validate_registry,
    )

    try:
        registry = load_registry(registry_path)
    except (OSError, ValueError) as error:
        print(f"Registry check failed: {error}")
        return 1

    findings = validate_registry(registry, load_protocol())
    errors = [finding for finding in findings if finding.severity == "error"]
    for finding in findings:
        print(f"  [{finding.severity}] {finding.code} at {finding.path}: {finding.message}")

    if errors:
        print(f"Recipe registry check failed: {len(errors)} error(s) in {registry_path.name}")
        return 1

    try:
        digest = canonical_digest(registry)
    except (TypeError, ValueError) as error:
        # A payload canonical JSON cannot serialize (NaN, a non-JSON value) is a
        # defect in the registry: the digest is what the manifest will bind to,
        # so an unhashable registry is not a registry.
        print(f"Recipe registry check failed: digest is not computable -- {error}")
        return 1

    print(
        f"Recipe registry check passed: {registry.get('recipe_registry_version')} "
        f"({len(registry.get('profiles') or {})} profile(s), {len(findings)} warning(s))\n"
        f"  registry sha256: {digest}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
