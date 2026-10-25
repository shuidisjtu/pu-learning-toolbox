#!/usr/bin/env python3
"""Survey profile -> comparison mapping gate: binding, coverage and vocabulary.

The map exists so that a v3 manifest can say *which* pre-registered entry its
profile inherits instead of reporting that no lookup path exists.  That claim is
only worth recording if the file is checked: a map that binds another protocol
digest, omits a protocol profile or inherits a method the matrix never
registered would turn the lookup into a plausible-looking miss.

Staging
-------
The gate reads the production file by default.  A missing file is reported on
stdout and exits 0, because at freeze time there was nothing to validate -- but
it never prints a pass, since a gate that reads green while checking nothing is
how the gap disappears from a report.  ``--require-present`` makes absence a
failure; CI uses it because the file is committed.

Usage::

    uv run python scripts/check_survey_profile_map.py
    uv run python scripts/check_survey_profile_map.py --map path/to/map.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Where the frozen map lives.  Named once so the gate, the resolver and the
#: delivery record cannot disagree about the path.  v3 is the effective map;
#: v1 and v2 stay on disk as the artifacts of the matrices they were written
#: against, and their resolution paths are superseded rather than maintained.
DEFAULT_MAP = PROJECT_ROOT / "pu_toolbox" / "experiment" / "survey_comparison_profile_map_v3.json"


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--map",
        type=Path,
        default=DEFAULT_MAP,
        dest="map_path",
        help="profile map JSON to validate (defaults to the shipped path)",
    )
    parser.add_argument(
        "--require-present",
        action="store_true",
        help="fail if the named artifact is absent rather than skipping validation",
    )
    args = parser.parse_args(argv)

    map_path: Path = args.map_path
    if not map_path.exists():
        print(
            f"Survey profile map not materialized yet: {map_path}\n"
            "Nothing was validated.  The map is a freeze-stage deliverable; this "
            "gate turns enforcing as soon as the file exists."
        )
        return 1 if args.require_present else 0

    from pu_toolbox.experiment.survey_profile_map import (
        bound_comparison,
        bound_protocol,
        digest,
        load_profile_map,
        validate_profile_map,
    )

    try:
        profile_map = load_profile_map(map_path)
    except (OSError, ValueError) as error:
        print(f"Profile map check failed: {error}")
        return 1

    # The bound files are loaded from the versions the map names, so validating a
    # v3 map against the v2 default would report every reference as drift instead
    # of checking the binding.
    try:
        protocol = bound_protocol(profile_map)
        comparison = bound_comparison(profile_map)
    except (OSError, ValueError) as error:
        print(f"Profile map check failed: cannot load the bound files -- {error}")
        return 1

    try:
        counts = validate_profile_map(profile_map, protocol=protocol, comparison=comparison)
    except ValueError as error:
        print(f"Profile map check failed: {error}")
        return 1

    try:
        map_digest = digest(profile_map)
    except (TypeError, ValueError) as error:
        # A payload canonical JSON cannot serialize cannot be pinned by a
        # manifest, which is the only thing the digest is for.
        print(f"Profile map check failed: digest is not computable -- {error}")
        return 1

    print(
        f"Profile map check passed: {profile_map['map_version']} "
        f"({counts['profiles']} profile(s): {counts['mapped']} mapped / "
        f"{counts['not_in_matrix']} not_in_matrix, {counts['deviations']} deviation(s))\n"
        f"  protocol basis: {protocol['protocol_version']}\n"
        f"  comparison basis: {comparison['comparison_version']}\n"
        f"  profile map sha256: {map_digest}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
