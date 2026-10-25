"""Print a deterministic P4.1 registry built from existing frozen declarations.

This does not write a registry, change protocols or train models; it prints the
payload, and committing that payload is the reviewed act. ``--lifecycle draft``
prints the preparation artifact; ``--lifecycle locked`` marks every runnable
profile locked and expands each candidate into a ``resolved_snapshot``, which is
the freeze scope the owner set on 2026-10-25. Profiles without runnable units
stay excluded either way, with their reason.
"""

from __future__ import annotations

import argparse
import json

from pu_toolbox.experiment.survey_protocol import load_protocol, resolve_protocol_path
from pu_toolbox.experiment.survey_recipe_registry import (
    build_draft_registry,
    build_locked_registry,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="survey-v2.0")
    parser.add_argument(
        "--lifecycle",
        choices=("draft", "locked"),
        default="draft",
        help="draft prints the preparation artifact; locked expands and freezes runnable profiles",
    )
    args = parser.parse_args()
    build = build_locked_registry if args.lifecycle == "locked" else build_draft_registry
    try:
        registry = build(load_protocol(resolve_protocol_path(args.protocol)))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(registry, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
