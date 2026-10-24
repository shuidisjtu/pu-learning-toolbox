"""Print/check v3 profile/unit requests; never train, approve or edit protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pu_toolbox.experiment.survey_v3_proposal import (
    build_proposal,
    expand_unit_requests,
    validate_proposal,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--summary", action="store_true")
    output_mode.add_argument("--units", action="store_true")
    args = parser.parse_args(argv)
    try:
        proposal = (
            json.loads(args.check.read_text(encoding="utf-8")) if args.check else build_proposal()
        )
        rows = validate_proposal(proposal) if args.check else expand_unit_requests(proposal)
        if args.summary:
            output = {
                "proposal_sha256": proposal["proposal_sha256"],
                "target_algorithm_count": len(proposal["scope"]["target_algorithms"]),
                "P3_1_algorithm_count": len(proposal["scope"]["p3_1_algorithms"]),
                "profile_count_including_variants": len(proposal["method_profile_requests"]),
                "dataset_count": len(proposal["datasets"]),
                "blocked_dataset_profile_requests": len(rows),
                "runnable_request_count": sum(row["runnable"] for row in rows),
                "registered_outside_target_scope": proposal["scope"][
                    "registered_outside_target_scope"
                ],
                "formal_admission": False,
            }
        else:
            output = rows if args.units else proposal
        print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
