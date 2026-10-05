"""Print a deterministic P4.1 draft from existing frozen pilot declarations.

This does not write a registry, lock recipes, change protocols or train models.
The generated draft is reviewed and tracked separately from the frozen registry.
"""

from __future__ import annotations

import argparse
import json

from pu_toolbox.experiment.survey_protocol import load_protocol, resolve_protocol_path
from pu_toolbox.experiment.survey_recipe_registry import build_draft_registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="survey-v1.2")
    args = parser.parse_args()
    try:
        registry = build_draft_registry(load_protocol(resolve_protocol_path(args.protocol)))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(registry, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
